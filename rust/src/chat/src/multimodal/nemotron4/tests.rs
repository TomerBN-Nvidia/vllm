// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::io::Cursor;

use base64::Engine;
use image::{Rgb, RgbImage, Rgba, RgbaImage};
use llm_multimodal::{MediaContentPart, PreProcessorConfig};
use vllm_engine_core_client::protocol::dtype::TensorDtype;
use vllm_tokenizer::test_utils::TestTokenizer;

use super::*;
use crate::multimodal::{MultimodalModelInfo, TokenizerResolver, expand_prompt_token_ids};

const IMAGE: u32 = 1000;
const ROW_END: u32 = 1001;
const IMAGE_END: u32 = 1002;

fn config() -> Value {
    serde_json::json!({"model_type":"nemotron4", "vision_config":{"patch_size":14}, "spatial_merge_size":2})
}

fn info(options: &MediaOptions) -> MultimodalModelInfo {
    let tokenizer = TestTokenizer::new()
        .with_regular_token("<img>", IMAGE)
        .with_regular_token("<image_break>", ROW_END)
        .with_regular_token("</img>", IMAGE_END);
    let context = MultimodalModelContext {
        model_id: "nm4-test".to_owned(),
        model_type: Some("nemotron4".to_owned()),
        config: config(),
        tokenizer: TokenizerResolver(Arc::new(tokenizer)),
    };
    MultimodalModelInfo::from_loaded(
        context,
        PreProcessorConfig::default(),
        PreProcessorConfig::default(),
        PreProcessorConfig::default(),
        HashMap::new(),
        options,
    )
    .unwrap()
    .unwrap()
}

fn png(image: &DynamicImage) -> Vec<u8> {
    let mut output = Cursor::new(Vec::new());
    image.write_to(&mut output, image::ImageFormat::Png).unwrap();
    output.into_inner()
}

#[test]
fn transparency_is_composited_on_white_before_rescaling() {
    let processor = Processor::new(&config(), &MediaOptions::default()).unwrap();
    let transparent = DynamicImage::ImageRgba8(RgbaImage::from_pixel(28, 28, Rgba([255, 0, 0, 0])));
    let white = DynamicImage::ImageRgb8(RgbImage::from_pixel(28, 28, Rgb([255, 255, 255])));
    assert_eq!(
        processor.preprocess(&transparent).unwrap().values,
        processor.preprocess(&white).unwrap().values
    );
    let half_alpha =
        DynamicImage::ImageRgba8(RgbaImage::from_pixel(28, 28, Rgba([255, 0, 0, 128])));
    let composed = DynamicImage::ImageRgb8(RgbImage::from_pixel(28, 28, Rgb([255, 127, 127])));
    assert_eq!(
        processor.preprocess(&half_alpha).unwrap().values,
        processor.preprocess(&composed).unwrap().values
    );
}

#[test]
fn encoded_image_orientation_matches_pillow_before_preprocessing() {
    let fixtures: Vec<Value> = serde_json::from_str(include_str!(
        "../../../tests/fixtures/nemotron4_orientation.json"
    ))
    .unwrap();
    let processor = Processor::new(&config(), &MediaOptions::default()).unwrap();
    for fixture in fixtures {
        let decode = |field: &str| {
            base64::engine::general_purpose::STANDARD
                .decode(fixture[field].as_str().unwrap())
                .unwrap()
        };
        let bytes = decode("input");
        let frame = ImageFrame::new(
            image::load_from_memory(&bytes).unwrap(),
            bytes.into(),
            llm_multimodal::ImageDetail::Auto,
            llm_multimodal::ImageSource::InlineBytes,
            "orientation-fixture".to_owned(),
        );
        let expected_image = image::load_from_memory(&decode("expected")).unwrap();
        let expected = processor.preprocess(&expected_image).unwrap();
        let actual = processor.preprocess_image_frame(&frame).unwrap();
        assert_eq!(
            (actual.height, actual.width),
            (expected.height, expected.width),
            "orientation {}",
            fixture["orientation"]
        );
        assert_eq!(
            actual.values, expected.values,
            "orientation {}",
            fixture["orientation"]
        );
    }
}

#[tokio::test]
async fn mixed_image_sizes_keep_unpadded_pixels_and_matching_embed_positions() {
    let info = info(&MediaOptions::default());
    let images = [(28, 28), (84, 56)]
        .into_iter()
        .enumerate()
        .map(|(index, (width, height))| MediaContentPart::ImageData {
            data: png(&DynamicImage::ImageRgb8(RgbImage::from_pixel(
                width,
                height,
                Rgb([255, 0, 0]),
            ))),
            mime_type: Some("image/png".to_owned()),
            uuid: Some(format!("image-{index}")),
            detail: None,
        })
        .collect();
    let mut prompt = vec![90, IMAGE, 91, IMAGE, 92];
    let features = info
        .prepare_multimodal(images, &mut prompt, ModelDtype::BFloat16, None)
        .await
        .unwrap();
    assert_eq!(
        prompt,
        [
            90, IMAGE, IMAGE_END, 91, IMAGE, IMAGE, IMAGE, ROW_END, IMAGE, IMAGE, IMAGE, IMAGE_END,
            92
        ]
    );
    let shapes = features
        .iter()
        .map(|feature| {
            let pixel = feature.data.as_ref().unwrap().get("pixel_values").unwrap();
            let MmKwargValue::Tensor(tensor) = pixel.data.as_ref().unwrap() else {
                panic!("image tensor")
            };
            assert_eq!(tensor.dtype, TensorDtype::Bf16);
            assert!(matches!(pixel.field, MmField::Batched(_)));
            tensor.shape.clone()
        })
        .collect::<Vec<_>>();
    assert_eq!(shapes, [vec![3, 28, 28], vec![3, 56, 84]]);
    assert_eq!(
        (
            features[0].mm_position.offset,
            features[0].mm_position.length
        ),
        (1, 2)
    );
    assert_eq!(
        (
            features[1].mm_position.offset,
            features[1].mm_position.length
        ),
        (4, 8)
    );
}

#[tokio::test]
async fn video_frames_expand_multitoken_marker_and_keep_nested_frame_payload() {
    let info = info(&MediaOptions::default());
    let images = (0..3)
        .map(|index| DynamicImage::ImageRgb8(RgbImage::from_pixel(28, 28, Rgb([index * 60, 0, 0]))))
        .collect();
    let clip = Arc::new(VideoClip::new(
        images,
        bytes::Bytes::from_static(b"fixture"),
        llm_multimodal::VideoSource::InlineBytes,
        "video-hash".to_owned(),
    ));
    let prepared = info
        .prepare_videos(
            vec![clip],
            vec![Some("video-uuid".to_owned())],
            ModelDtype::BFloat16,
        )
        .await
        .unwrap();
    let mut prompt = info.context.tokenizer().encode("before<video>after", false).unwrap();
    assert!(prepared.placeholder.marker_token_ids.len() > 1);
    let ranges = expand_prompt_token_ids(&mut prompt, std::slice::from_ref(&prepared)).unwrap();
    let decoded = info.context.tokenizer().decode(&prompt, false).unwrap();
    assert_eq!(decoded, "before<img></img>\n<img></img>\n<img></img>after");
    let video = prepared.items[0].data["pixel_values_videos"].data.as_ref().unwrap();
    let MmKwargValue::List(frames) = video else {
        panic!("frame list")
    };
    assert_eq!(frames.len(), 3);
    assert_eq!(ranges[&Modality::Video][0].length, 8);
}

#[test]
fn high_resolution_override_and_video_options_are_effective() {
    let context = info(&MediaOptions::default()).context;
    let preprocessor_config = PreProcessorConfig {
        size: Some(HashMap::from([("longest_edge".to_owned(), 2048)])),
        ..Default::default()
    };
    let model = MultimodalModelInfo::from_loaded(
        context,
        preprocessor_config,
        PreProcessorConfig::default(),
        PreProcessorConfig::default(),
        HashMap::new(),
        &MediaOptions::default(),
    )
    .unwrap()
    .unwrap();
    let crate::multimodal::VisionProcessor::Nemotron4(processor) =
        &model.image.as_ref().unwrap().processor
    else {
        panic!("Nemotron4 processor expected");
    };
    assert_eq!(
        resize::output_size(1540, 1700, processor.longest_edge),
        (1540, 1708)
    );
    let options = MediaOptions {
        mm_processor_kwargs: HashMap::from([(
            "size".to_owned(),
            serde_json::json!({"longest_edge":2048}),
        )]),
        media_io_kwargs: HashMap::from([(
            "video".to_owned(),
            serde_json::json!({"fps":2,"num_frames":128}),
        )]),
        ..Default::default()
    };
    let processor = Processor::new(&config(), &options).unwrap();
    assert_eq!(
        resize::output_size(1540, 1700, processor.longest_edge),
        (1540, 1708)
    );
    assert_eq!(processor.video_frames, 128);
    assert_eq!(processor.video_decode_frames, 128);
    assert_eq!(processor.video_fps, Some(2.0));
    for bad in [
        serde_json::json!({"longest_edge":0}),
        serde_json::json!({"height":1024}),
    ] {
        let options = MediaOptions {
            mm_processor_kwargs: HashMap::from([("size".to_owned(), bad)]),
            ..Default::default()
        };
        assert!(Processor::new(&config(), &options).is_err());
    }
}

fn media_tokens(
    config: Value,
    processor_config: Option<&str>,
    tokenizer_config: Option<&str>,
    tokenizer: &TestTokenizer,
) -> Result<Value> {
    let dir = tempfile::tempdir().unwrap();
    let write = |name: &str, body: Option<&str>| {
        body.map(|b| {
            let p = dir.path().join(name);
            std::fs::write(&p, b).unwrap();
            p
        })
    };
    let pc = write("processor_config.json", processor_config);
    let tc = write("tokenizer_config.json", tokenizer_config);
    let mut config = config;
    super::resolve_media_tokens(&mut config, pc.as_deref(), tc.as_deref(), tokenizer)?;
    Ok(config[super::MEDIA_TOKENS_KEY].clone())
}

#[test]
fn media_token_names_come_from_the_configs_or_the_image_token_id() {
    let nemo = TestTokenizer::new()
        .with_regular_token("<nemo:img>", 318)
        .with_regular_token("<nemo:img_break>", 320)
        .with_regular_token("<nemo:/img>", 319)
        .with_regular_token("<nemo:video>", 330);
    let named = r#"{"image_token": "<nemo:img>", "image_break_token": "<nemo:img_break>", "image_end_token": "<nemo:/img>"}"#;
    expect_test::expect![[r#"
        Object {
            "image_token": String("<nemo:img>"),
            "image_break_token": String("<nemo:img_break>"),
            "image_end_token": String("<nemo:/img>"),
            "video_token": String("<nemo:video>"),
        }
    "#]]
    .assert_debug_eq(&media_tokens(config(), Some(named), None, &nemo).unwrap());
    // named in tokenizer_config.json instead
    assert_eq!(
        media_tokens(config(), None, Some(named), &nemo).unwrap()["image_end_token"],
        "<nemo:/img>"
    );
    // nothing named: the image token is image_token_id's, break and end keep the old names
    let v5 = TestTokenizer::new()
        .with_regular_token("<image>", 318)
        .with_regular_token("<image_break>", 320)
        .with_regular_token("</img>", 319);
    let mut cfg = config();
    cfg["image_token_id"] = serde_json::json!(318);
    let tokens = media_tokens(cfg, None, None, &v5).unwrap();
    assert_eq!(tokens["image_token"], "<image>");
    assert_eq!(tokens["video_token"], "<video>");
}

#[test]
fn media_token_names_the_tokenizer_lacks_are_an_error() {
    let tokenizer = TestTokenizer::new().with_regular_token("<img>", IMAGE);
    let error = media_tokens(config(), None, None, &tokenizer).unwrap_err();
    assert!(
        error.to_string().contains("image_break_token=\"<image_break>\""),
        "{error}"
    );
    let full = TestTokenizer::new()
        .with_regular_token("<img>", IMAGE)
        .with_regular_token("<image_break>", ROW_END)
        .with_regular_token("</img>", IMAGE_END);
    let error =
        media_tokens(config(), Some(r#"{"video_token": "<vid>"}"#), None, &full).unwrap_err();
    assert!(
        error.to_string().contains("video_token=\"<vid>\""),
        "{error}"
    );
}
