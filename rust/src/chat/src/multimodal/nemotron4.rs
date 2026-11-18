// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Nemotron4's training transform and ragged image/frame payloads.

use std::collections::HashMap;
use std::sync::Arc;

use image::{DynamicImage, ImageDecoder};
use itertools::izip;
use llm_multimodal::registry::{ModelRegistryError, RegistryResult};
use llm_multimodal::{
    ImageFrame, Modality, ModelMetadata, ModelProcessorSpec, PreprocessedEncoderInputs,
    PromptReplacement, TokenId, VideoClip,
};
use serde_json::Value;
use vllm_engine_core_client::protocol::dtype::ModelDtype;
use vllm_engine_core_client::protocol::multimodal::{
    MmBatchedField, MmField, MmFieldElem, MmKwargValue, MmKwargsItem,
};

use super::{
    MediaOptions, MultimodalModelContext, PreparedItem, PreparedMedia, VisionModalitySupport,
    tensor,
};
use crate::error::{Result, bail_multimodal, multimodal};

mod resize;
#[cfg(test)]
mod tests;
mod video;

pub(super) static SPEC: Spec = Spec;
pub(super) struct Spec;

/// Where the resolved media-token spellings live in the model config the spec reads.
const MEDIA_TOKENS_KEY: &str = "__vllm_rs_nemotron4_media_tokens";

/// The media tokens' spellings, for checkpoints that name none (Nemotron 4 step-7000 layout).
const LEGACY_IMAGE_TOKENS: [(&str, &str); 3] = [
    ("image_token", "<img>"),
    ("image_break_token", "<image_break>"),
    ("image_end_token", "</img>"),
];
const VIDEO_TOKENS: [&str; 2] = ["<nemo:video>", "<video>"];

/// Resolve the image and video tokens' spellings (as the Python model class does: Daniel Afrimi's
/// d966b3af50 and its follow-up) and record them in `config` for the spec: each named by
/// `processor_config.json`, else `tokenizer_config.json`; an unnamed image token is the token with
/// config.json's `image_token_id`, unnamed break and end tokens keep their old names, and an
/// unnamed video token is `<nemo:video>` when the tokenizer has it, else `<video>`. Every name
/// must be a token of the tokenizer.
pub(super) fn resolve_media_tokens(
    config: &mut Value,
    processor_config: Option<&std::path::Path>,
    tokenizer_config: Option<&std::path::Path>,
    tokenizer: &dyn vllm_text::tokenizer::Tokenizer,
) -> Result<()> {
    let read = |path: Option<&std::path::Path>| -> Value {
        path.and_then(|p| std::fs::read_to_string(p).ok())
            .and_then(|t| serde_json::from_str(&t).ok())
            .unwrap_or(Value::Null)
    };
    let sources = [read(processor_config), read(tokenizer_config)];
    let named = |key: &str| -> Option<String> {
        sources.iter().find_map(|c| {
            let v = c.get(key)?;
            v.as_str()
                .or_else(|| v.get("content").and_then(Value::as_str))
                .filter(|s| !s.is_empty())
                .map(str::to_owned)
        })
    };
    let mut tokens = serde_json::Map::new();
    let mut missing = Vec::new();
    for (key, legacy) in LEGACY_IMAGE_TOKENS {
        let spelling = match named(key) {
            Some(name) => name,
            None if key == "image_token" => config
                .get("image_token_id")
                .and_then(Value::as_u64)
                .and_then(|id| tokenizer.id_to_token(id as u32))
                .unwrap_or_else(|| legacy.to_owned()),
            None => legacy.to_owned(),
        };
        if tokenizer.token_to_id(&spelling).is_none() {
            missing.push(format!("{key}={spelling:?}"));
        }
        tokens.insert(key.to_owned(), Value::String(spelling));
    }
    if !missing.is_empty() {
        bail_multimodal!(
            "Nemotron4: the tokenizer has no {}. Name the tokenizer's image tokens in \
             processor_config.json.",
            missing.join(", ")
        );
    }
    let video = match named("video_token") {
        Some(name) => {
            if tokenizer.token_to_id(&name).is_none() {
                bail_multimodal!(
                    "Nemotron4: the tokenizer has no video_token={name:?} (named in \
                     processor_config.json or tokenizer_config.json)"
                );
            }
            name
        }
        None => VIDEO_TOKENS
            .iter()
            .find(|t| tokenizer.token_to_id(t).is_some())
            .unwrap_or(&VIDEO_TOKENS[1])
            .to_string(),
    };
    tokens.insert("video_token".to_owned(), Value::String(video));
    if let Value::Object(c) = config {
        c.insert(MEDIA_TOKENS_KEY.to_owned(), Value::Object(tokens));
    }
    Ok(())
}

/// A resolved media token's spelling (the legacy one when the config was not resolved).
fn media_token(metadata: &ModelMetadata, key: &str) -> String {
    metadata
        .config
        .get(MEDIA_TOKENS_KEY)
        .and_then(|t| t.get(key))
        .and_then(Value::as_str)
        .map(str::to_owned)
        .unwrap_or_else(|| {
            match key {
                "image_token" => "<img>",
                "image_break_token" => "<image_break>",
                "image_end_token" => "</img>",
                _ => "<video>",
            }
            .to_owned()
        })
}

impl ModelProcessorSpec for Spec {
    fn name(&self) -> &'static str {
        "nemotron4"
    }

    fn matches(&self, metadata: &ModelMetadata) -> bool {
        metadata.config_model_type() == Some("nemotron4")
    }

    fn placeholder_token(&self, metadata: &ModelMetadata) -> RegistryResult<String> {
        Ok(media_token(metadata, "image_token"))
    }

    fn placeholder_token_id(&self, metadata: &ModelMetadata) -> RegistryResult<TokenId> {
        metadata.token_id(&media_token(metadata, "image_token"))
    }

    fn placeholder_token_for(
        &self,
        metadata: &ModelMetadata,
        modality: Modality,
    ) -> RegistryResult<String> {
        match modality {
            Modality::Image => self.placeholder_token(metadata),
            Modality::Video => Ok(media_token(metadata, "video_token")),
            _ => Err(ModelRegistryError::UnsupportedModality {
                spec: self.name(),
                modality,
            }),
        }
    }

    fn placeholder_token_id_for(
        &self,
        metadata: &ModelMetadata,
        modality: Modality,
    ) -> RegistryResult<TokenId> {
        self.placeholder_token_for(metadata, modality)?;
        self.placeholder_token_id(metadata)
    }

    fn modality_limits(&self, _: &ModelMetadata) -> RegistryResult<HashMap<Modality, usize>> {
        Ok(HashMap::from([
            (Modality::Image, usize::MAX),
            (Modality::Video, 1),
        ]))
    }

    fn prompt_replacements(
        &self,
        metadata: &ModelMetadata,
        preprocessed: &PreprocessedEncoderInputs,
    ) -> RegistryResult<Vec<PromptReplacement>> {
        let patch = metadata.config_u32(&["vision_config", "patch_size"]).unwrap_or(14);
        let merge = metadata.config_u32(&["spatial_merge_size"]).unwrap_or(2);
        let multiple = patch.checked_mul(merge).filter(|&value| value > 0).ok_or_else(|| {
            ModelRegistryError::MissingConfigField {
                field: "positive vision patch_size * spatial_merge_size".to_owned(),
            }
        })?;
        let tokens = ImageTokens::new(metadata)?;
        Ok(preprocessed
            .item_sizes
            .iter()
            .map(|&(height, width)| {
                replacement(
                    Modality::Image,
                    &tokens.marker,
                    tokens.for_image(height as usize, width as usize, multiple as usize),
                )
            })
            .collect())
    }
}

struct ImageTokens {
    /// The image placeholder's spelling.
    marker: String,
    image: TokenId,
    row_end: TokenId,
    end: TokenId,
}

impl ImageTokens {
    fn new(metadata: &ModelMetadata) -> RegistryResult<Self> {
        let marker = media_token(metadata, "image_token");
        Ok(Self {
            image: metadata.token_id(&marker)?,
            row_end: metadata.token_id(&media_token(metadata, "image_break_token"))?,
            end: metadata.token_id(&media_token(metadata, "image_end_token"))?,
            marker,
        })
    }

    fn for_image(&self, height: usize, width: usize, multiple: usize) -> Vec<TokenId> {
        let rows = height / multiple;
        let cols = width / multiple;
        let mut result = Vec::with_capacity(rows * (cols + 1));
        for row in 0..rows {
            result.extend(std::iter::repeat_n(self.image, cols));
            result.push(if row + 1 == rows {
                self.end
            } else {
                self.row_end
            });
        }
        result
    }
}

fn replacement(modality: Modality, marker: &str, tokens: Vec<TokenId>) -> PromptReplacement {
    PromptReplacement::sequence(modality, marker, tokens)
}

#[derive(Clone)]
pub(super) struct Processor {
    multiple: usize,
    longest_edge: usize,
    video_frames: usize,
    video_decode_frames: usize,
    video_fps: Option<f64>,
    allowed_local_media_path: Option<std::path::PathBuf>,
}

struct Pixels {
    values: Vec<f32>,
    height: usize,
    width: usize,
}

impl Pixels {
    fn into_wire(self, dtype: ModelDtype) -> Result<MmKwargValue> {
        let value = tensor::KwargValue::from_f32_parts(
            self.values,
            vec![3, self.height, self.width],
            dtype,
        )?;
        (&value).try_into()
    }
}

impl Processor {
    pub(super) async fn fetch_media(
        &self,
        connector: &Arc<llm_multimodal::MediaConnector>,
        parts: Vec<llm_multimodal::MediaContentPart>,
    ) -> Result<super::FetchedMedia> {
        video::fetch_media(self, connector, parts).await
    }
    pub(super) fn new(config: &Value, options: &MediaOptions) -> Result<Self> {
        let patch = config
            .pointer("/vision_config/patch_size")
            .and_then(Value::as_u64)
            .unwrap_or(14);
        let merge = config.get("spatial_merge_size").and_then(Value::as_u64).unwrap_or(2);
        let multiple = patch.checked_mul(merge).filter(|&value| value > 0 && value <= 1036).ok_or_else(|| multimodal!("Nemotron4 requires a positive vision patch_size * spatial_merge_size no greater than 1036"))? as usize;
        if options.mm_processor_kwargs.keys().any(|key| key != "size") {
            bail_multimodal!(
                "Nemotron4 Rust preprocessing supports only the size processor override"
            );
        }
        let longest_edge = match options.mm_processor_kwargs.get("size") {
            None => 1036,
            Some(value) => {
                let object = value.as_object().ok_or_else(|| {
                    multimodal!("Nemotron4 size must be an object with longest_edge")
                })?;
                if object.len() != 1 {
                    bail_multimodal!("Nemotron4 size supports only longest_edge");
                }
                object
                    .get("longest_edge")
                    .and_then(Value::as_u64)
                    .filter(|&n| n > 0)
                    .ok_or_else(|| {
                        multimodal!("Nemotron4 longest_edge must be a positive integer")
                    })? as usize
            }
        };
        if options.media_io_kwargs.keys().any(|key| key != "video") {
            bail_multimodal!("Nemotron4 Rust media_io_kwargs supports only video options");
        }
        let video = options
            .media_io_kwargs
            .get("video")
            .map(|value| {
                value
                    .as_object()
                    .ok_or_else(|| multimodal!("video media_io_kwargs must be an object"))
            })
            .transpose()?;
        if video.is_some_and(|object| object.keys().any(|key| key != "num_frames" && key != "fps"))
        {
            bail_multimodal!("Nemotron4 Rust video options support num_frames and fps");
        }
        let frames = video
            .and_then(|object| object.get("num_frames"))
            .map(|value| {
                value.as_u64().filter(|&n| n > 0).map(|n| n as usize).ok_or_else(|| {
                    multimodal!("Nemotron4 video num_frames must be a positive integer")
                })
            })
            .transpose()?;
        let video_fps = video
            .and_then(|object| object.get("fps"))
            .map(|value| {
                value
                    .as_f64()
                    .filter(|fps| fps.is_finite())
                    .ok_or_else(|| multimodal!("video fps must be finite"))
            })
            .transpose()?
            .filter(|&fps| fps > 0.0);
        Ok(Self {
            multiple,
            longest_edge,
            video_frames: frames.unwrap_or(16),
            video_decode_frames: frames.unwrap_or(32),
            video_fps,
            allowed_local_media_path: options.allowed_local_media_path.clone(),
        })
    }

    /// Match Nemotron4ImageProcessor: FP32 rescaling precedes OpenCV area resizing.
    fn preprocess(&self, image: &DynamicImage) -> Result<Pixels> {
        let rgba = image.to_rgba8();
        let (source_width, source_height) = rgba.dimensions();
        if source_width == 0 || source_height == 0 {
            bail_multimodal!("Nemotron4 cannot preprocess an empty image");
        }
        let (height, width) = resize::output_size(
            source_height as usize,
            source_width as usize,
            self.longest_edge,
        );
        if height == 0 || width == 0 {
            bail_multimodal!("Nemotron4 resized image has an empty dimension");
        }
        if height % self.multiple != 0 || width % self.multiple != 0 {
            bail_multimodal!(
                "Nemotron4 resized image dimensions must be divisible by vision patch_size * spatial_merge_size"
            );
        }
        let mut rgb = Vec::with_capacity(source_width as usize * source_height as usize * 3);
        for pixel in rgba.pixels() {
            let alpha = u32::from(pixel[3]);
            for &channel in &pixel.0[..3] {
                let value = (u32::from(channel) * alpha + 255 * (255 - alpha) + 127) / 255;
                rgb.push(value as f32 / 255.0);
            }
        }
        let resized = resize::area(
            &rgb,
            source_height as usize,
            source_width as usize,
            height,
            width,
        );
        let mean = [0.48145466_f32, 0.4578275, 0.40821073];
        // Keep the Python training transform's decimal constants visible.
        #[allow(clippy::excessive_precision)]
        let std = [0.26862954_f32, 0.26130258, 0.27577711];
        let size = height * width;
        let mut values = vec![0.0; 3 * size];
        for (index, pixel) in resized.chunks_exact(3).enumerate() {
            for channel in 0..3 {
                values[channel * size + index] = (pixel[channel] - mean[channel]) / std[channel];
            }
        }
        Ok(Pixels {
            values,
            height,
            width,
        })
    }

    fn preprocess_image_frame(&self, frame: &ImageFrame) -> Result<Pixels> {
        // The shared loader preserves the encoded orientation. Python's
        // ImageMediaIO applies ImageOps.exif_transpose before its transform.
        // As in Python, malformed or absent metadata leaves pixels unchanged.
        let orientation = image::ImageReader::new(std::io::Cursor::new(frame.raw_bytes()))
            .with_guessed_format()
            .ok()
            .and_then(|reader| reader.into_decoder().ok())
            .and_then(|mut decoder| decoder.orientation().ok())
            .unwrap_or(image::metadata::Orientation::NoTransforms);
        if orientation == image::metadata::Orientation::NoTransforms {
            return self.preprocess(frame.image()?);
        }
        let mut image = frame.image()?.clone();
        image.apply_orientation(orientation);
        self.preprocess(&image)
    }

    pub(super) async fn prepare_images(
        &self,
        frames: Vec<Arc<ImageFrame>>,
        uuids: Vec<Option<String>>,
        context: &MultimodalModelContext,
        support: &VisionModalitySupport,
        dtype: ModelDtype,
    ) -> Result<PreparedMedia> {
        if frames.len() != uuids.len() {
            bail_multimodal!("Nemotron4 image UUID count does not match the image count");
        }
        let processor = self.clone();
        let processed = tokio::task::spawn_blocking(move || {
            frames
                .into_iter()
                .map(|frame| {
                    Ok((
                        processor.preprocess_image_frame(&frame)?,
                        frame.hash.clone(),
                    ))
                })
                .collect::<Result<Vec<_>>>()
        })
        .await
        .map_err(|error| multimodal!("Nemotron4 image preprocessing task failed: {error}"))??;
        let tokens = ImageTokens::new(&context.metadata())?;
        let mut replacements = Vec::with_capacity(processed.len());
        let mut items = Vec::with_capacity(processed.len());
        for ((pixels, hash), uuid) in izip!(processed, uuids) {
            replacements.push(replacement(
                Modality::Image,
                &tokens.marker,
                tokens.for_image(pixels.height, pixels.width, self.multiple),
            ));
            items.push(item("pixel_values", pixels.into_wire(dtype)?, hash, uuid));
        }
        Ok(PreparedMedia {
            modality: Modality::Image,
            placeholder: support.placeholder.clone(),
            replacements,
            items,
        })
    }

    pub(super) async fn prepare_videos(
        &self,
        clips: Vec<Arc<VideoClip>>,
        uuids: Vec<Option<String>>,
        context: &MultimodalModelContext,
        support: &VisionModalitySupport,
        dtype: ModelDtype,
    ) -> Result<PreparedMedia> {
        if clips.len() != 1 || uuids.len() != 1 {
            bail_multimodal!("Nemotron4 supports one video per request");
        }
        let processor = self.clone();
        let clip = Arc::clone(&clips[0]);
        let processed = tokio::task::spawn_blocking(move || {
            let frames = clip.materialized_frames().map_err(|error| multimodal!("{error}"))?;
            if frames.is_empty() {
                bail_multimodal!("Nemotron4 requires at least one video frame");
            }
            let count = frames.len().min(processor.video_frames);
            (0..count)
                .map(|index| {
                    let frame = if count == 1 {
                        0
                    } else {
                        index * (frames.len() - 1) / (count - 1)
                    };
                    processor.preprocess(&frames[frame])
                })
                .collect::<Result<Vec<_>>>()
        })
        .await
        .map_err(|error| multimodal!("Nemotron4 video preprocessing task failed: {error}"))??;
        let tokens = ImageTokens::new(&context.metadata())?;
        let separator = context.tokenizer().encode("\n", false)?;
        let mut prompt = Vec::new();
        let mut pixels = Vec::with_capacity(processed.len());
        for frame in processed {
            if !prompt.is_empty() {
                prompt.extend(separator.iter().map(|&token| token as TokenId));
            }
            prompt.extend(tokens.for_image(frame.height, frame.width, self.multiple));
            pixels.push(frame.into_wire(dtype)?);
        }
        Ok(PreparedMedia {
            modality: Modality::Video,
            placeholder: support.placeholder.clone(),
            replacements: vec![replacement(
                Modality::Video,
                &media_token(&context.metadata(), "video_token"),
                prompt,
            )],
            items: vec![item(
                "pixel_values_videos",
                MmKwargValue::List(pixels),
                clips[0].hash.clone(),
                uuids.into_iter().next().unwrap(),
            )],
        })
    }
}

fn item(key: &str, value: MmKwargValue, hash: String, uuid: Option<String>) -> PreparedItem {
    let data = MmKwargsItem::from([(
        key.to_owned(),
        MmFieldElem {
            data: Some(value),
            field: MmField::Batched(MmBatchedField { keep_on_cpu: false }),
        },
    )]);
    PreparedItem { data, hash, uuid }
}
