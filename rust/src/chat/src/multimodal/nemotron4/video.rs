// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Python-compatible NM4 video frame selection and restricted local media access.

use std::io::{Cursor, Write};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::{Arc, LazyLock};
use std::time::Duration;

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD;
use bytes::Bytes;
use futures::{StreamExt, TryStreamExt, stream};
use image::{DynamicImage, ImageDecoder, ImageFormat, ImageReader, RgbImage};
use llm_multimodal::{
    ImageFetchConfig, MediaConnector, MediaContentPart, MediaSource, TrackedMedia, VideoClip,
    VideoSource,
};
use tokio::io::AsyncReadExt;
use tokio::process::Command;

use super::super::FetchedMedia;
use super::Processor;
use crate::error::{Result, bail_multimodal, multimodal};

const MAX_INPUT_BYTES: usize = 256 * 1024 * 1024;
const MAX_DECODED_BYTES: usize = 1024 * 1024 * 1024;
const VIDEO_TIMEOUT: Duration = Duration::from_secs(30);

fn media_source(url: String) -> Result<MediaSource> {
    let parsed = reqwest_0_13::Url::parse(&url)
        .map_err(|error| multimodal!("invalid media URL: {error}"))?;
    match parsed.scheme() {
        "file" => Ok(MediaSource::File(
            parsed.to_file_path().map_err(|()| multimodal!("invalid local media URL"))?,
        )),
        "data" => Ok(MediaSource::DataUrl(url)),
        "http" | "https" => Ok(MediaSource::Url(url)),
        _ => bail_multimodal!("media URL must use http, https, data or file"),
    }
}

pub(super) async fn fetch_media(
    processor: &Processor,
    connector: &Arc<MediaConnector>,
    parts: Vec<MediaContentPart>,
) -> Result<FetchedMedia> {
    let video_count = parts
        .iter()
        .filter(|part| {
            matches!(
                part,
                MediaContentPart::VideoUrl { .. } | MediaContentPart::VideoData { .. }
            )
        })
        .count();
    if video_count > 1 {
        bail_multimodal!("Nemotron4 supports one video per request");
    }
    let fetched = stream::iter(
        parts.into_iter().filter(|part| !matches!(part, MediaContentPart::Text { .. })),
    )
    .map(|part| async move {
        match part {
            MediaContentPart::ImageUrl {
                url,
                detail,
                uuid,
                max_long_side_pixel,
            } => {
                let image = connector
                    .fetch_image(
                        media_source(url)?,
                        ImageFetchConfig {
                            detail: detail.unwrap_or_default(),
                            max_long_side_pixel,
                            transparency: llm_multimodal::TransparencyPolicy::CompositeOverWhite,
                        },
                    )
                    .await?;
                Ok::<_, crate::error::Error>((TrackedMedia::Image(image), uuid))
            }
            MediaContentPart::ImageData {
                data, detail, uuid, ..
            } => {
                let image = connector
                    .fetch_image(
                        MediaSource::InlineBytes(data),
                        ImageFetchConfig {
                            detail: detail.unwrap_or_default(),
                            max_long_side_pixel: None,
                            transparency: llm_multimodal::TransparencyPolicy::CompositeOverWhite,
                        },
                    )
                    .await?;
                Ok((TrackedMedia::Image(image), uuid))
            }
            MediaContentPart::VideoUrl {
                url,
                uuid,
                fps,
                max_long_side_pixel,
            } => Ok((
                TrackedMedia::Video(
                    fetch_video_with_options(
                        processor,
                        media_source(url)?,
                        fps,
                        max_long_side_pixel,
                    )
                    .await?,
                ),
                uuid,
            )),
            MediaContentPart::VideoData { data, uuid, .. } => Ok((
                TrackedMedia::Video(fetch_video(processor, MediaSource::InlineBytes(data)).await?),
                uuid,
            )),
            _ => bail_multimodal!("Nemotron4 supports image and video media inputs"),
        }
    })
    .buffered(8)
    .try_collect::<Vec<_>>()
    .await?;
    let mut result = FetchedMedia {
        images: Vec::new(),
        image_uuids: Vec::new(),
        videos: Vec::new(),
        video_uuids: Vec::new(),
        audios: Vec::new(),
        audio_uuids: Vec::new(),
    };
    for (media, uuid) in fetched {
        match media {
            TrackedMedia::Image(frame) => {
                result.images.push(frame);
                result.image_uuids.push(uuid);
            }
            TrackedMedia::Video(clip) => {
                result.videos.push(clip);
                result.video_uuids.push(uuid);
            }
            _ => unreachable!("only image and video fetches are scheduled"),
        }
    }
    Ok(result)
}

async fn local_video_path(path: &Path, allowed: Option<&Path>) -> Result<PathBuf> {
    let root = allowed
        .ok_or_else(|| multimodal!("local video inputs require --allowed-local-media-path"))?;
    let root = tokio::fs::canonicalize(root)
        .await
        .map_err(|error| multimodal!("invalid allowed local media directory: {error}"))?;
    let path = tokio::fs::canonicalize(path)
        .await
        .map_err(|error| multimodal!("cannot resolve local video: {error}"))?;
    if !path.starts_with(root) {
        bail_multimodal!("local video is outside --allowed-local-media-path");
    }
    Ok(path)
}

async fn read_source(
    processor: &Processor,
    source: MediaSource,
) -> Result<(Bytes, VideoSource, Option<String>)> {
    let (data, origin, jpeg_frames) = match source {
        MediaSource::InlineBytes(data) => (data, VideoSource::InlineBytes, None),
        MediaSource::File(path) => {
            let path =
                local_video_path(&path, processor.allowed_local_media_path.as_deref()).await?;
            let file = tokio::fs::File::open(&path)
                .await
                .map_err(|error| multimodal!("cannot open local video: {error}"))?;
            let mut data = Vec::new();
            file.take((MAX_INPUT_BYTES + 1) as u64)
                .read_to_end(&mut data)
                .await
                .map_err(|error| multimodal!("cannot read local video: {error}"))?;
            (data, VideoSource::File { path }, None)
        }
        MediaSource::DataUrl(url) => {
            if url.len() > MAX_INPUT_BYTES * 4 / 3 + 4096 {
                bail_multimodal!("video data URL exceeds input limit");
            }
            let (header, data) =
                url.split_once(',').ok_or_else(|| multimodal!("invalid video data URL"))?;
            if !header.ends_with(";base64") {
                bail_multimodal!("video data URL must be base64 encoded");
            }
            if header.eq_ignore_ascii_case("data:video/jpeg;base64") {
                (
                    data.as_bytes().to_vec(),
                    VideoSource::DataUrl,
                    Some(data.to_owned()),
                )
            } else {
                (
                    STANDARD
                        .decode(data)
                        .map_err(|error| multimodal!("invalid base64 video: {error}"))?,
                    VideoSource::DataUrl,
                    None,
                )
            }
        }
        MediaSource::Url(url) => {
            static CLIENT: LazyLock<reqwest_0_13::Client> =
                LazyLock::new(reqwest_0_13::Client::new);
            let mut response = CLIENT
                .get(&url)
                .timeout(Duration::from_secs(10))
                .send()
                .await
                .map_err(|error| multimodal!("video fetch failed: {error}"))?
                .error_for_status()
                .map_err(|error| multimodal!("video fetch failed: {error}"))?;
            if response.content_length().is_some_and(|length| length > MAX_INPUT_BYTES as u64) {
                bail_multimodal!("video exceeds input limit");
            }
            let mut data = Vec::new();
            while let Some(chunk) = response
                .chunk()
                .await
                .map_err(|error| multimodal!("video fetch failed: {error}"))?
            {
                if data.len() + chunk.len() > MAX_INPUT_BYTES {
                    bail_multimodal!("video exceeds input limit");
                }
                data.extend_from_slice(&chunk);
            }
            (data, VideoSource::Url { url }, None)
        }
    };
    if data.len() > MAX_INPUT_BYTES {
        bail_multimodal!("video exceeds input limit");
    }
    Ok((Bytes::from(data), origin, jpeg_frames))
}

async fn fetch_video_with_options(
    processor: &Processor,
    source: MediaSource,
    fps: Option<f64>,
    max_long_side_pixel: Option<u32>,
) -> Result<Arc<VideoClip>> {
    if max_long_side_pixel.is_some() {
        bail_multimodal!("Nemotron4 video does not support max_long_side_pixel");
    }
    let mut processor = processor.clone();
    if let Some(fps) = fps {
        if !fps.is_finite() || fps <= 0.0 {
            bail_multimodal!("video fps must be positive and finite");
        }
        processor.video_fps = Some(fps);
    }
    fetch_video(&processor, source).await
}

async fn fetch_video(processor: &Processor, source: MediaSource) -> Result<Arc<VideoClip>> {
    let (bytes, source, jpeg_frames) = read_source(processor, source).await?;
    let frames = if let Some(data) = jpeg_frames {
        let count = processor.video_decode_frames;
        tokio::task::spawn_blocking(move || decode_jpeg_frames(&data, count, MAX_DECODED_BYTES))
            .await
            .map_err(|error| multimodal!("video frame decoding task failed: {error}"))??
    } else {
        tokio::time::timeout(VIDEO_TIMEOUT, decode_video(processor, &bytes))
            .await
            .map_err(|_| multimodal!("video decoding exceeded 30 seconds"))??
    };
    if frames.is_empty() {
        bail_multimodal!("video contains no frames");
    }
    let hash = llm_multimodal::hasher::hash_video(&bytes);
    Ok(Arc::new(VideoClip::new(frames, bytes, source, hash)))
}

fn decode_jpeg_frames(data: &str, count: usize, byte_limit: usize) -> Result<Vec<DynamicImage>> {
    let mut remaining = byte_limit;
    data.split(',')
        .take(count)
        .map(|part| {
            let bytes = STANDARD
                .decode(part)
                .map_err(|error| multimodal!("invalid base64 video frame: {error}"))?;
            let mut reader = ImageReader::with_format(Cursor::new(bytes), ImageFormat::Jpeg);
            let mut limits = image::Limits::default();
            limits.max_alloc = Some(remaining as u64);
            reader.limits(limits);
            let decoder = reader
                .into_decoder()
                .map_err(|error| multimodal!("invalid JPEG video frame: {error}"))?;
            let (width, height) = decoder.dimensions();
            // Reserve RGB bytes before decoding, including grayscale JPEGs expanded later.
            let frame_bytes = (width as usize)
                .checked_mul(height as usize)
                .and_then(|n| n.checked_mul(3))
                .ok_or_else(|| multimodal!("video dimensions overflow"))?;
            remaining = remaining
                .checked_sub(frame_bytes)
                .ok_or_else(|| multimodal!("video exceeds decoded-byte limit"))?;
            DynamicImage::from_decoder(decoder)
                .map_err(|error| multimodal!("invalid JPEG video frame: {error}"))
        })
        .collect()
}

/// Match Python VideoBackend: uniformly select original frame indices, with fps as a ceiling.
fn frame_indices(total: usize, source_fps: f64, max_frames: usize, fps: Option<f64>) -> Vec<usize> {
    let mut count = total.min(max_frames);
    if let Some(fps) = fps {
        count = count.min((total as f64 / source_fps * fps).floor() as usize);
    }
    count = count.max(1);
    let step = (total - 1) as f64 / (count - 1).max(1) as f64;
    (0..count)
        .map(|index| {
            if count == 1 {
                0
            } else if index == count - 1 {
                total - 1
            } else {
                // np.linspace constructs float64 values before flooring to integers.
                (index as f64 * step).floor() as usize
            }
        })
        .collect()
}

async fn probe_video(path: &Path) -> Result<serde_json::Value> {
    let mut command = Command::new("ffprobe");
    command.kill_on_drop(true).args([
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,nb_read_frames,avg_frame_rate:stream_side_data=rotation:stream_tags=rotate",
        "-count_frames",
        "-of",
        "json",
    ]);
    let probe =
        command.arg(path).output().await.map_err(|error| {
            multimodal!("ffprobe is required for encoded video inputs: {error}")
        })?;
    if !probe.status.success() {
        bail_multimodal!(
            "ffprobe could not read video metadata: {}",
            String::from_utf8_lossy(&probe.stderr)
        );
    }
    let metadata: serde_json::Value = serde_json::from_slice(&probe.stdout)
        .map_err(|error| multimodal!("invalid video metadata: {error}"))?;
    Ok(metadata["streams"][0].clone())
}

fn positive_frame_count(value: &serde_json::Value) -> Option<usize> {
    value.as_str().and_then(|text| text.parse::<usize>().ok()).filter(|&n| n > 0)
}

fn display_dimensions(stream: &serde_json::Value) -> Result<(usize, usize)> {
    let dimension = |name: &str| {
        stream[name]
            .as_u64()
            .and_then(|n| usize::try_from(n).ok())
            .filter(|&n| n > 0)
            .ok_or_else(|| multimodal!("video has no valid {name}"))
    };
    let (width, height) = (dimension("width")?, dimension("height")?);
    let rotation = stream["side_data_list"]
        .as_array()
        .and_then(|items| items.iter().find_map(|item| item["rotation"].as_f64()))
        .or_else(|| stream["tags"]["rotate"].as_str()?.parse::<f64>().ok())
        .unwrap_or(0.0)
        .rem_euclid(360.0);
    if !rotation.is_finite() {
        bail_multimodal!("video has invalid rotation metadata");
    }
    // ffmpeg autorotation transposes near-right-angle display matrices.
    if (rotation - 90.0).abs() < 1.0 || (rotation - 270.0).abs() < 1.0 {
        Ok((height, width))
    } else {
        Ok((width, height))
    }
}

async fn decode_video(processor: &Processor, bytes: &[u8]) -> Result<Vec<DynamicImage>> {
    let mut file = tempfile::NamedTempFile::new()
        .map_err(|error| multimodal!("cannot create video decode file: {error}"))?;
    file.write_all(bytes)
        .map_err(|error| multimodal!("cannot write video decode file: {error}"))?;
    // Container sample counts include hidden MP4 edit-list lead-in frames.
    let stream = probe_video(file.path()).await?;
    let total = positive_frame_count(&stream["nb_read_frames"])
        .ok_or_else(|| multimodal!("video has no valid decoded frame count"))?;
    let (width, height) = display_dimensions(&stream)?;
    let rate = stream["avg_frame_rate"]
        .as_str()
        .and_then(|text| text.split_once('/'))
        .and_then(|(n, d)| Some(n.parse::<f64>().ok()? / d.parse::<f64>().ok()?))
        .filter(|n| n.is_finite() && *n > 0.0)
        .ok_or_else(|| multimodal!("video has no valid frame rate"))?;
    let indices = frame_indices(
        total,
        rate,
        processor.video_decode_frames,
        processor.video_fps,
    );
    let frame_bytes = height
        .checked_mul(width)
        .and_then(|n| n.checked_mul(3))
        .ok_or_else(|| multimodal!("video dimensions overflow"))?;
    let expected = frame_bytes
        .checked_mul(indices.len())
        .filter(|&n| n <= MAX_DECODED_BYTES)
        .ok_or_else(|| multimodal!("video exceeds decoded-byte limit"))?;
    let filter = format!(
        "select={}",
        indices
            .iter()
            .map(|index| format!("eq(n\\,{index})"))
            .collect::<Vec<_>>()
            .join("+")
    );
    let mut child = Command::new("ffmpeg")
        .kill_on_drop(true)
        .args(["-v", "error", "-threads", "1", "-i"])
        .arg(file.path())
        .args([
            "-vf", &filter, "-vsync", "0", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|error| multimodal!("ffmpeg is required for encoded video inputs: {error}"))?;
    let mut output = Vec::with_capacity(expected);
    child
        .stdout
        .take()
        .expect("stdout is piped")
        .take((expected + 1) as u64)
        .read_to_end(&mut output)
        .await
        .map_err(|error| multimodal!("video decoding failed: {error}"))?;
    if output.len() != expected {
        bail_multimodal!("decoded video dimensions or frame count differ from metadata");
    }
    if !child
        .wait()
        .await
        .map_err(|error| multimodal!("video decoder failed: {error}"))?
        .success()
    {
        bail_multimodal!("ffmpeg video decoding failed");
    }
    Ok(output
        .chunks_exact(frame_bytes)
        .map(|frame| {
            DynamicImage::ImageRgb8(
                RgbImage::from_raw(width as u32, height as u32, frame.to_vec())
                    .expect("decoded frame length checked"),
            )
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_selection_matches_python_uniform_sampling() {
        assert_eq!(frame_indices(10, 30.0, 4, None), vec![0, 3, 6, 9]);
        assert_eq!(
            frame_indices(100, 25.0, 128, Some(2.0)),
            vec![0, 14, 28, 42, 56, 70, 84, 99]
        );
        assert_eq!(frame_indices(1, 30.0, 32, Some(2.0)), vec![0]);
        assert_eq!(frame_indices(3, 30.0, 32, None), vec![0, 1, 2]);
    }

    #[test]
    fn frame_selection_preserves_numpy_float64_rounding() {
        let indices = frame_indices(31, 30.0, 23, None);
        assert_eq!(indices[11], 14);
        assert_eq!(indices.last(), Some(&30));
        let indices = frame_indices(154, 30.0, 64, None);
        assert_eq!((indices[21], indices[42], indices[49]), (50, 101, 118));
        assert_eq!(indices.last(), Some(&153));
    }

    #[test]
    fn rotated_video_uses_autorotated_dimensions() {
        for rotation in [90, -90, 270, 450] {
            let stream = serde_json::json!({
                "width": 384, "height": 256,
                "side_data_list": [{"rotation": rotation}],
            });
            assert_eq!(display_dimensions(&stream).unwrap(), (256, 384));
        }
        let stream = serde_json::json!({
            "width": 384, "height": 256, "tags": {"rotate": "180"},
        });
        assert_eq!(display_dimensions(&stream).unwrap(), (384, 256));
    }

    #[test]
    fn jpeg_frame_list_enforces_cumulative_decoded_byte_limit() {
        let frame = DynamicImage::ImageRgb8(RgbImage::new(16, 16));
        let mut bytes = Cursor::new(Vec::new());
        frame.write_to(&mut bytes, ImageFormat::Jpeg).unwrap();
        let encoded = STANDARD.encode(bytes.into_inner());
        let data = format!("{encoded},{encoded}");
        assert_eq!(
            decode_jpeg_frames(&data, 2, 16 * 16 * 3 * 2).unwrap().len(),
            2
        );
        assert!(decode_jpeg_frames(&data, 2, 16 * 16 * 3 * 2 - 1).is_err());
        assert_eq!(decode_jpeg_frames(&data, 1, 16 * 16 * 3).unwrap().len(), 1);
    }

    #[tokio::test]
    #[ignore = "requires ffmpeg, ffprobe, and OpenCV fixtures in NM4_VIDEO_FIXTURES"]
    // Generate fixtures with tests/fixtures/generate_nemotron4_video.py, then run:
    // NM4_VIDEO_FIXTURES=/path/to/fixtures cargo test -p vllm-chat \
    //     encoded_video_frames_match_opencv_oracle -- --ignored
    async fn encoded_video_frames_match_opencv_oracle() {
        let root =
            PathBuf::from(std::env::var_os("NM4_VIDEO_FIXTURES").expect("set NM4_VIDEO_FIXTURES"));
        let cases: serde_json::Value =
            serde_json::from_slice(&std::fs::read(root.join("decode-oracle.json")).unwrap())
                .unwrap();
        for case in cases.as_array().unwrap() {
            let name = case["file"].as_str().unwrap();
            let mut processor =
                Processor::new(&serde_json::json!({}), &Default::default()).unwrap();
            processor.video_decode_frames = case["num_frames"].as_u64().unwrap() as usize;
            processor.video_fps = Some(case["fps"].as_f64().unwrap());
            let bytes = std::fs::read(root.join(name)).unwrap();
            let frames = decode_video(&processor, &bytes).await.unwrap();
            let expected = case["frames"].as_array().unwrap();
            assert_eq!(frames.len(), expected.len(), "{name}");
            for (frame, reference) in frames.iter().zip(expected) {
                let actual = frame.to_rgb8();
                let expected =
                    image::open(root.join(reference["file"].as_str().unwrap())).unwrap().to_rgb8();
                assert_eq!(actual.dimensions(), expected.dimensions(), "{name}");
                let max_error = actual
                    .as_raw()
                    .iter()
                    .zip(expected.as_raw())
                    .map(|(a, b)| a.abs_diff(*b))
                    .max()
                    .unwrap();
                // OpenCV's bundled FFmpeg and system libswscale may round RGB differently.
                assert!(max_error <= 2, "{name}: RGB byte error {max_error}");
            }
        }
    }

    #[tokio::test]
    async fn local_video_access_stays_inside_canonical_allowed_root() {
        let root = tempfile::tempdir().unwrap();
        let outside = tempfile::NamedTempFile::new().unwrap();
        let inside = root.path().join("video.mp4");
        std::fs::write(&inside, b"fixture").unwrap();
        assert!(local_video_path(&inside, None).await.is_err());
        assert!(local_video_path(&inside, Some(root.path())).await.is_ok());
        assert!(local_video_path(outside.path(), Some(root.path())).await.is_err());
        let link = root.path().join("escape.mp4");
        std::os::unix::fs::symlink(outside.path(), &link).unwrap();
        assert!(local_video_path(&link, Some(root.path())).await.is_err());
    }
}
