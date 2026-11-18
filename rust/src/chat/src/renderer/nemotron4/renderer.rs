// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! [`Nemotron4ChatRenderer`]: the HF renderer's chat template, messages and kwargs, rendered by the
//! structural encoder into token ids. Selected with the nemotron4 parsers (or
//! `--renderer nemotron4`); `NEMOTRON4_STRUCTURAL_CHAT=0` keeps the stock HF renderer, as the
//! Python plugin's switch does.

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};

use serde_json::Value as Json;
use thiserror_ext::AsReport as _;
use tracing::{info, warn};
use vllm_text::Prompt;
use vllm_text::backend::hf::{
    HfChatTemplate, HfTokenizerConfig, ResolvedModelFiles, TokenizerSource, load_tokenizer_config,
};
use vllm_text::tokenizer::DynTokenizer;

use super::{RenderInput, StructuralRenderer};
use crate::backend::LoadModelBackendsOptions;
use crate::error::{Error, Result};
use crate::renderer::hf::{
    ChatTemplateContentFormat, ChatTemplateContentFormatOption, CompiledChatTemplate,
    MultimodalRenderInfo, ProbeContext, effective_template_kwargs, load_chat_template,
    resolve_chat_template, to_template_messages, to_template_tools,
};
use crate::renderer::{ChatRenderer, RenderedPrompt};
use crate::request::{ChatContent, ChatContentPart, ChatMessage, ChatRequest};

/// The chat template a model serves with: the configured override, else a dedicated chat template
/// file with content, else `tokenizer_config.json`'s.
fn load_default_template(
    files: &ResolvedModelFiles,
    options: &LoadModelBackendsOptions,
) -> Result<Option<String>> {
    let HfTokenizerConfig { chat_template, .. } =
        load_tokenizer_config(files.tokenizer_config_path.as_deref())?;
    let mut template = chat_template
        .as_ref()
        .and_then(HfChatTemplate::default_template)
        .map(str::to_owned);
    if let Some(configured_template) = options.chat_template.as_deref() {
        template = Some(
            resolve_chat_template(configured_template)
                .map_err(|error| Error::ChatTemplate(error.to_report_string()))?,
        );
        info!("using configured chat template override");
    } else if let Some(chat_template_path) = files.chat_template_path.as_deref() {
        // If independent chat template file(s) exist and contain non-empty content,
        // they take priority over template entries in the tokenizer config
        let file_template = load_chat_template(chat_template_path)
            .map_err(|error| Error::ChatTemplate(error.to_report_string()))?;

        if file_template.as_ref().is_some_and(|t| !t.trim().is_empty()) {
            info!(
                path = %chat_template_path.display(),
                "loaded dedicated chat template file, overriding tokenizer_config chat_template"
            );
            template = file_template;
        } else {
            warn!(
                path = %chat_template_path.display(),
                "ignoring empty dedicated chat template file and falling back to tokenizer_config chat_template"
            );
        }
    }

    Ok(template)
}

/// The key under which [`string_content`] hands the data wrapper the parts of a string-format
/// content (an internal shape: client content always passes through this renderer's conversion).
pub(super) const STRING_CONTENT_KEY: &str = "__nemotron4_string_content";

/// A string-format content with media: its parts, the media as the placeholder spellings this
/// renderer inserts (template text, as the template's own media tokens are), the text parts client
/// data. The HF renderer joins the same parts with "\n" into one string, where the placeholders
/// would be client text.
fn string_content(
    content: &ChatContent,
    multimodal: Option<&MultimodalRenderInfo>,
) -> Option<Json> {
    let ChatContent::Parts(parts) = content else {
        return None;
    };
    if !parts.iter().any(ChatContentPart::is_multimodal) {
        return None;
    }
    let token = |t: Option<&String>| t.cloned().unwrap_or_default();
    let segments: Vec<Json> = parts
        .iter()
        .filter(|part| !part.is_empty_text())
        .map(|part| match part {
            ChatContentPart::Text { text } => serde_json::json!({"text": text}),
            ChatContentPart::ToolReference { name } => serde_json::json!({"text": name}),
            ChatContentPart::ImageUrl { .. } => {
                serde_json::json!({"media": token(multimodal.and_then(|m| m.image_token.as_ref()))})
            }
            ChatContentPart::VideoUrl { .. } => {
                serde_json::json!({"media": token(multimodal.and_then(|m| m.video_token.as_ref()))})
            }
            ChatContentPart::InputAudio { .. } | ChatContentPart::AudioUrl { .. } => {
                serde_json::json!({"media": token(multimodal.and_then(|m| m.audio_token.as_ref()))})
            }
        })
        .collect();
    Some(serde_json::json!({ STRING_CONTENT_KEY: segments }))
}

/// Whether the structural renderer is on (`NEMOTRON4_STRUCTURAL_CHAT`, default on).
pub fn structural_chat_enabled() -> bool {
    std::env::var("NEMOTRON4_STRUCTURAL_CHAT").map_or(true, |v| v != "0")
}

/// The most per-request template overrides kept compiled.
const MAX_OVERRIDES: usize = 16;

/// One compiled template and the content format its messages get.
struct Compiled {
    encoder: StructuralRenderer,
    content_probe: CompiledChatTemplate,
}

/// The nemotron4 structural renderer.
pub struct Nemotron4ChatRenderer {
    default: Option<Arc<Compiled>>,
    default_template_kwargs: HashMap<String, Json>,
    content_format: ChatTemplateContentFormatOption,
    multimodal: Option<MultimodalRenderInfo>,
    added_vocab: Arc<[(String, u32)]>,
    special_tokens: Vec<(String, Json)>,
    tokenizer: DynTokenizer,
    overrides: Mutex<Vec<(String, Arc<Compiled>)>>,
}

fn template_error(message: impl Into<String>) -> Error {
    Error::ChatTemplate(message.into())
}

/// The added tokens of `tokenizer.json`, refusing flags the encoder can't model (as the Python
/// plugin: lstrip / rstrip / single_word, and normalized under a normalizer that changes text).
fn added_tokens(tokenizer_json: &Path) -> Result<Vec<(String, u32)>> {
    let text = std::fs::read_to_string(tokenizer_json).map_err(|e| {
        template_error(format!(
            "nemotron4 structural chat rendering: cannot read {}: {}",
            tokenizer_json.display(),
            e.as_report()
        ))
    })?;
    let spec: Json = serde_json::from_str(&text).map_err(|e| {
        template_error(format!(
            "nemotron4 structural chat rendering: cannot parse {}: {}",
            tokenizer_json.display(),
            e.as_report()
        ))
    })?;
    let normalizer = &spec["normalizer"];
    let identity = normalizer.is_null()
        || (normalizer["type"] == "Sequence"
            && normalizer["normalizers"].as_array().is_none_or(Vec::is_empty));
    let mut out = Vec::new();
    for t in spec["added_tokens"].as_array().into_iter().flatten() {
        let flag = |k: &str| t[k].as_bool().unwrap_or(false);
        if flag("lstrip")
            || flag("rstrip")
            || flag("single_word")
            || (flag("normalized") && !identity)
        {
            return Err(template_error(format!(
                "nemotron4 structural chat rendering can't model this tokenizer (added token flags \
                 not supported: {t}); NEMOTRON4_STRUCTURAL_CHAT=0 renders prompts stock"
            )));
        }
        if let (Some(content), Some(id)) = (t["content"].as_str(), t["id"].as_u64()) {
            out.push((content.to_owned(), id as u32));
        }
    }
    Ok(out)
}

/// The special tokens the template gets (`bos_token`, ... and `additional_special_tokens`), from
/// `special_tokens_map.json` and `tokenizer_config.json` (the latter's values winning).
fn special_tokens(tokenizer_config: Option<&Path>) -> Vec<(String, Json)> {
    const NAMED: [&str; 7] = [
        "bos_token",
        "eos_token",
        "unk_token",
        "sep_token",
        "pad_token",
        "cls_token",
        "mask_token",
    ];
    let mut out: Vec<(String, Json)> = Vec::new();
    let Some(config) = tokenizer_config else {
        return out;
    };
    let mut put = |k: &str, v: Json| {
        if let Some(slot) = out.iter_mut().find(|(n, _)| n == k) {
            slot.1 = v;
        } else {
            out.push((k.to_owned(), v));
        }
    };
    for path in [
        config.with_file_name("special_tokens_map.json"),
        config.to_path_buf(),
    ] {
        let Ok(text) = std::fs::read_to_string(&path) else {
            continue;
        };
        let Ok(c) = serde_json::from_str::<Json>(&text) else {
            continue;
        };
        let spelling = |v: &Json| {
            v.as_str()
                .map(str::to_owned)
                .or_else(|| v["content"].as_str().map(str::to_owned))
        };
        for k in NAMED {
            if let Some(s) = c.get(k).and_then(spelling) {
                put(k, Json::String(s));
            }
        }
        if let Some(list) = c.get("additional_special_tokens").and_then(Json::as_array) {
            put(
                "additional_special_tokens",
                Json::Array(list.iter().filter_map(spelling).map(Json::String).collect()),
            );
        }
    }
    out
}

impl Nemotron4ChatRenderer {
    /// Load from the model files, as the HF renderer loads its template.
    pub fn load(
        files: &ResolvedModelFiles,
        options: LoadModelBackendsOptions,
        multimodal: Option<MultimodalRenderInfo>,
        tokenizer: DynTokenizer,
    ) -> Result<Self> {
        let Some(TokenizerSource::HuggingFace(tokenizer_json)) = &files.tokenizer else {
            return Err(template_error(
                "nemotron4 structural chat rendering can't model this tokenizer (no tokenizer.json); \
                 NEMOTRON4_STRUCTURAL_CHAT=0 renders prompts stock",
            ));
        };
        let added_vocab: Arc<[(String, u32)]> = added_tokens(tokenizer_json)?.into();
        let special_tokens = special_tokens(files.tokenizer_config_path.as_deref());
        let template = load_default_template(files, &options)?;
        let mut renderer = Self {
            default: None,
            default_template_kwargs: options.default_chat_template_kwargs,
            content_format: options.chat_template_content_format,
            multimodal,
            added_vocab,
            special_tokens,
            tokenizer,
            overrides: Mutex::new(Vec::new()),
        };
        renderer.default = template.map(|t| renderer.compile(&t)).transpose()?.map(Arc::new);
        info!(
            version = vllm_parser::unified::NEMOTRON4_VERSION,
            "nemotron4 structural chat rendering: the chat template compiled to a symbolic \
             encoder (NEMOTRON4_STRUCTURAL_CHAT=0 disables)"
        );
        Ok(renderer)
    }

    fn compile(&self, template: &str) -> Result<Compiled> {
        let content_probe = CompiledChatTemplate::new(
            template.to_owned(),
            self.content_format,
            None,
            &ProbeContext {
                special_tokens: None,
                template_kwargs: Some(&self.default_template_kwargs),
            },
        )
        .map_err(|e| template_error(e.to_report_string()))?;
        let encoder =
            StructuralRenderer::new(template, &self.added_vocab, self.special_tokens.clone())
                .map_err(|e| template_error(e.to_string()))?;
        Ok(Compiled {
            encoder,
            content_probe,
        })
    }

    fn compiled_for(&self, request: &ChatRequest) -> Result<Arc<Compiled>> {
        let Some(template) = request.chat_options.chat_template.as_ref() else {
            return self.default.clone().ok_or(Error::MissingChatTemplate);
        };
        let mut cache = self.overrides.lock().expect("override cache");
        if let Some((_, c)) = cache.iter().find(|(t, _)| t == template) {
            return Ok(c.clone());
        }
        let compiled = Arc::new(self.compile(template)?);
        if cache.len() >= MAX_OVERRIDES {
            cache.remove(0);
        }
        cache.push((template.clone(), compiled.clone()));
        Ok(compiled)
    }
}

impl ChatRenderer for Nemotron4ChatRenderer {
    fn render(&self, request: &ChatRequest) -> Result<RenderedPrompt> {
        let compiled = self.compiled_for(request)?;
        let content_format = compiled.content_probe.content_format_for(&request.messages);
        let messages =
            to_template_messages(&request.messages, content_format, self.multimodal.as_ref())?;
        let mut messages: Vec<Json> = messages
            .iter()
            .map(serde_json::to_value)
            .collect::<std::result::Result<_, _>>()
            .map_err(|e| template_error(e.to_report_string()))?;
        if content_format == ChatTemplateContentFormat::String {
            // the placeholders this renderer writes into string content are structural
            for (json, message) in messages.iter_mut().zip(&request.messages) {
                let content = match message {
                    ChatMessage::System { content }
                    | ChatMessage::Developer { content, .. }
                    | ChatMessage::User { content }
                    | ChatMessage::Custom { content, .. }
                    | ChatMessage::ToolResponse { content, .. } => content,
                    ChatMessage::Assistant { .. } => continue,
                };
                if let Some(parts) = string_content(content, self.multimodal.as_ref()) {
                    json["content"] = parts;
                }
            }
        }
        let tools: Option<Vec<Json>> = (!request.tools().is_empty())
            .then(|| to_template_tools(request.tools()).iter().map(serde_json::to_value).collect())
            .transpose()
            .map_err(|e| template_error(e.to_report_string()))?;
        let effective = effective_template_kwargs(&self.default_template_kwargs, request);
        let kwargs: serde_json::Map<String, Json> = effective.clone().into_iter().collect();
        let documents = request.documents.as_deref();
        let rendered = compiled
            .encoder
            .render(
                &RenderInput {
                    messages: &messages,
                    tools: tools.as_deref(),
                    documents,
                    add_generation_prompt: request.chat_options.add_generation_prompt(),
                    continue_final_message: request.chat_options.continue_final_message(),
                    template_kwargs: &kwargs,
                },
                self.tokenizer.as_ref(),
            )
            .map_err(|e| template_error(e.to_string()))?;
        Ok(RenderedPrompt {
            prompt: Prompt::TokenIds(rendered.ids),
            media_order: None,
            effective_template_kwargs: effective,
        })
    }
}
