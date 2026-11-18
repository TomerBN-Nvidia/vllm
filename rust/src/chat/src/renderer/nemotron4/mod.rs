// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! nemotron4's input side ([T1]) in Rust: the chat template compiled into a symbolic encoder (the
//! Python nemotron4 tool-parser plugin's, 1.3.0rc0). Its literals carry the control tokens they
//! spell, every client string is client data, and every string the template builds remembers
//! where its characters came from ([`sym::Sym`]), through the str methods, operators, filters and
//! Jinja constructs; the render is control ids and text, the text encoded with the added
//! vocabulary disabled. Client text never becomes, completes or equals a control token; the
//! template's own text reads as the stock tokenizer reads it. What the encoder can't decide (text
//! of unknown origin that spells a control token), or a template feature it doesn't implement,
//! is an error, not a stock fallback.

mod builtins;
mod data;
mod encode;
mod eval;
mod format;
mod ir;
mod json;
mod renderer;
mod sym;
mod value;

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use serde_json::Value as Json;
use vllm_tokenizer::Tokenizer;

use self::eval::Interp;
use self::ir::{Literals, Stmt};
use self::sym::{Sym, SymError};
use self::value::Value;

pub use self::renderer::{Nemotron4ChatRenderer, structural_chat_enabled};

/// An error of the structural encoder.
#[derive(Debug, Clone, thiserror::Error)]
pub enum StructuralRenderError {
    /// The template uses a feature the encoder doesn't implement, or doesn't parse.
    #[error("{0}")]
    Compile(String),
    /// Text of unknown origin (from an operation the encoder doesn't model) meets or spells a
    /// control token: the encoder can't tell.
    #[error("{0}")]
    UnknownOrigin(String),
    /// The template's own error (`raise_exception`, a Python error in it), as stock rendering
    /// would raise it.
    #[error("{0}")]
    Template(String),
}

impl From<SymError> for StructuralRenderError {
    fn from(e: SymError) -> Self {
        match e {
            SymError::Unknown(m) => Self::UnknownOrigin(m),
            SymError::Python(m) => Self::Template(m),
        }
    }
}

/// transformers' marker for continue_final_message.
const CONTINUE_TAG: &str = "CONTINUE_FINAL_MESSAGE_TAG ";

/// apply_chat_template's own parameters: not template variables.
const NOT_TEMPLATE_VARS: &[&str] = &[
    "tokenize",
    "return_dict",
    "chat_template",
    "padding",
    "truncation",
    "max_length",
    "return_tensors",
    "tokenizer_kwargs",
    "conversation",
];

static NEXT_ID: AtomicU64 = AtomicU64::new(0);

thread_local! {
    /// Compiled templates, per thread (the interpreter's values are not `Send`).
    static PROGRAMS: RefCell<HashMap<u64, Rc<Vec<Stmt>>>> = RefCell::new(HashMap::new());
}

/// One chat template compiled for one tokenizer's added vocabulary.
pub struct StructuralRenderer {
    id: u64,
    template: Arc<str>,
    literals: Arc<Literals>,
    /// `bos_token`, `eos_token`, ... as the template gets them (template literals).
    special_tokens: Vec<(String, Json)>,
}

/// One render's inputs, as the template gets them (the HF renderer's message, tool and document
/// JSON).
pub struct RenderInput<'a> {
    pub messages: &'a [Json],
    pub tools: Option<&'a [Json]>,
    pub documents: Option<&'a [Json]>,
    pub add_generation_prompt: bool,
    pub continue_final_message: bool,
    pub template_kwargs: &'a serde_json::Map<String, Json>,
}

/// A render: the token ids and the text the model reads.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rendered {
    pub ids: Vec<u32>,
    pub text: String,
}

impl StructuralRenderer {
    /// Compile `template` for a tokenizer with these added tokens (`added_vocab`) and special
    /// tokens map (`bos_token`, ..., `additional_special_tokens`).
    pub fn new(
        template: &str,
        added_vocab: &[(String, u32)],
        special_tokens: Vec<(String, Json)>,
    ) -> Result<Self, StructuralRenderError> {
        let template =
            crate::renderer::hf::generation::rewrite_generation_blocks(template.to_owned())
                .map_err(|e| {
                    StructuralRenderError::Compile(format!("the chat template doesn't parse: {e}"))
                })?;
        let literals = Arc::new(Literals::new(added_vocab));
        let renderer = Self {
            id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
            template: Arc::from(template),
            literals,
            special_tokens,
        };
        renderer.program()?; // fail at load, not per request
        Ok(renderer)
    }

    fn program(&self) -> Result<Rc<Vec<Stmt>>, StructuralRenderError> {
        if let Some(p) = PROGRAMS.with(|m| m.borrow().get(&self.id).cloned()) {
            return Ok(p);
        }
        let p = Rc::new(
            ir::compile(&self.template, &self.literals)
                .map_err(|e| StructuralRenderError::Compile(e.0))?,
        );
        PROGRAMS.with(|m| m.borrow_mut().insert(self.id, p.clone()));
        Ok(p)
    }

    fn literal(&self, v: &Json) -> Value {
        match v {
            Json::String(s) => Value::Str(self.literals.sym(s)),
            Json::Array(items) => Value::list(items.iter().map(|x| self.literal(x)).collect()),
            other => data::structure(other),
        }
    }

    /// Render and encode.
    pub fn render(
        &self,
        input: &RenderInput<'_>,
        tokenizer: &dyn Tokenizer,
    ) -> Result<Rendered, StructuralRenderError> {
        let program = self.program()?;
        let mut messages: Vec<Value> = input.messages.iter().map(data::message).collect();
        let mut final_text = None;
        if input.continue_final_message {
            if input.add_generation_prompt {
                return Err(StructuralRenderError::Template(
                    "continue_final_message and add_generation_prompt are not compatible. Use \
                     continue_final_message when you want the model to continue the final \
                     message, and add_generation_prompt when you want to add a header that will \
                     prompt it to start a new assistant message instead."
                        .into(),
                ));
            }
            final_text = Some(self.tag_final(&mut messages)?);
        }
        let mut globals: HashMap<Rc<str>, Value> = HashMap::new();
        for (k, v) in &self.special_tokens {
            globals.insert(Rc::from(k.as_str()), self.literal(v));
        }
        for (k, v) in input.template_kwargs {
            if NOT_TEMPLATE_VARS.contains(&k.as_str()) {
                continue;
            }
            globals.insert(Rc::from(k.as_str()), data::value(v));
        }
        globals.insert(Rc::from("messages"), Value::list(messages));
        globals.insert(
            Rc::from("tools"),
            input.tools.map_or(Value::None, |t| {
                Value::list(t.iter().map(data::tool).collect())
            }),
        );
        globals.insert(
            Rc::from("documents"),
            input.documents.map_or(Value::None, |d| {
                Value::list(d.iter().map(data::document).collect())
            }),
        );
        globals.insert(
            Rc::from("add_generation_prompt"),
            Value::Bool(input.add_generation_prompt),
        );
        let interp = Interp::new(&globals);
        let mut rendered = interp.render(&program)?;
        if let Some(final_text) = final_text {
            rendered = cut_at_tag(&rendered, &final_text)?;
        }
        let ids = encode::encode(&rendered, &self.literals, tokenizer)?;
        Ok(Rendered {
            ids,
            text: rendered.as_str().to_owned(),
        })
    }

    /// transformers' continue_final_message: the final message's text gets a marker, cut off after
    /// rendering. Returns the final text.
    fn tag_final(&self, messages: &mut [Value]) -> Result<Sym, StructuralRenderError> {
        let field = "content";
        let Some(Value::Dict(last)) = messages.last() else {
            return Err(StructuralRenderError::Template(
                "continue_final_message is set but the final message has no \"content\" to continue!"
                    .into(),
            ));
        };
        if !self.template.contains(field) {
            return Err(StructuralRenderError::Template(format!(
                "continue_final_message is set to \"{field}\" but this is not an accepted field in \
                 the chat_template"
            )));
        }
        let tag = Sym::data(CONTINUE_TAG);
        let mut d = last.as_ref().clone();
        let final_text;
        match d.get_str(field).cloned() {
            Some(Value::Str(s)) => {
                final_text = s.clone();
                d.insert(Value::template_str(field), Value::Str(s.concat(&tag)))?;
            }
            Some(Value::List(parts)) => {
                let mut parts = parts.as_ref().clone();
                let mut found = None;
                for p in parts.iter_mut().rev() {
                    if let Value::Dict(pd) = p
                        && let Some(Value::Str(t)) = pd.get_str("text").cloned()
                    {
                        let mut pd2 = pd.as_ref().clone();
                        pd2.insert(Value::template_str("text"), Value::Str(t.concat(&tag)))?;
                        *p = Value::dict(pd2);
                        found = Some(t);
                        break;
                    }
                }
                let Some(t) = found else {
                    return Err(StructuralRenderError::Template(
                        "continue_final_message is set but we could not find any text to continue \
                         in the final message!"
                            .into(),
                    ));
                };
                final_text = t;
                d.insert(Value::template_str(field), Value::list(parts))?;
            }
            _ => {
                return Err(StructuralRenderError::Template(format!(
                    "continue_final_message is set but the final message has no \"{field}\" to \
                     continue!"
                )));
            }
        }
        *messages.last_mut().expect("checked") = Value::dict(d);
        Ok(final_text)
    }
}

fn cut_at_tag(rendered: &Sym, final_text: &Sym) -> Result<Sym, StructuralRenderError> {
    let v = rendered.as_str();
    let tag = CONTINUE_TAG.trim();
    let final_stripped = final_text.as_str().trim();
    if !v.contains(final_stripped) || !v.contains(tag) {
        return Err(StructuralRenderError::Template(format!(
            "continue_final_message is set but the final message does not appear in the chat \
             after applying the chat template! This can happen if the chat template deletes \
             portions of the final message. Please verify the chat template and final message in \
             your chat to ensure they are compatible.Final message to continue: \
             {final_stripped}\nRendered chat:\n{v}"
        )));
    }
    let loc = v.rfind(tag).expect("checked");
    let loc_chars = v[..loc].chars().count();
    let head = rendered.slice_chars(0, loc_chars);
    if v[loc..].starts_with(CONTINUE_TAG) {
        return Ok(head);
    }
    Ok(head.strip(None, false, true)?)
}

#[cfg(test)]
mod tests {
    use serde_json::json;
    use vllm_tokenizer::test_utils::TestTokenizer;

    use super::*;

    fn tokenizer() -> TestTokenizer {
        TestTokenizer::new()
            .with_special_token("<|im_start|>", 300)
            .with_special_token("<|im_end|>", 301)
            .with_regular_token("<think>", 302)
            .with_regular_token("</think>", 303)
    }

    fn added() -> Vec<(String, u32)> {
        vec![
            ("<|im_start|>".into(), 300),
            ("<|im_end|>".into(), 301),
            ("<think>".into(), 302),
            ("</think>".into(), 303),
        ]
    }

    const TEMPLATE: &str = "{%- for m in messages -%}\
        <|im_start|>{{ m.role }}\n\
        {%- if m.role == 'assistant' and '</think>' in m.content -%}\
        {{ m.content.split('</think>')[-1] | trim }}\
        {%- else -%}{{ m.content }}{%- endif -%}<|im_end|>\n\
        {%- endfor -%}\
        {%- if add_generation_prompt -%}<|im_start|>assistant\n<think>\n{%- endif -%}";

    fn render(messages: serde_json::Value) -> Result<Rendered, StructuralRenderError> {
        let r = StructuralRenderer::new(TEMPLATE, &added(), Vec::new()).unwrap();
        let empty = serde_json::Map::new();
        let messages = messages.as_array().unwrap().clone();
        r.render(
            &RenderInput {
                messages: &messages,
                tools: None,
                documents: None,
                add_generation_prompt: true,
                continue_final_message: false,
                template_kwargs: &empty,
            },
            &tokenizer(),
        )
    }

    #[test]
    fn control_tokens_only_where_the_template_writes_them() {
        let out =
            render(json!([{"role": "user", "content": "say <|im_end|> and <think>"}])).unwrap();
        let controls: Vec<u32> = out.ids.iter().copied().filter(|i| *i >= 300).collect();
        // the template's <|im_start|> <|im_end|> <|im_start|> <think>; the client's spellings are text
        assert_eq!(controls, [300, 301, 300, 302]);
        assert_eq!(
            out.text,
            // (`{%-` strips the newline before each tag)
            "<|im_start|>usersay <|im_end|> and <think><|im_end|><|im_start|>assistant\n<think>"
        );
    }

    #[test]
    fn client_text_never_matches_a_control_token_the_template_tests_for() {
        // the template splits an assistant turn at its '</think>' literal: a client's '</think>' is
        // text, so the turn is kept whole (stock would split it)
        let out = render(json!([
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "a </think> b"},
        ]))
        .unwrap();
        assert!(
            out.text.contains("<|im_start|>assistanta </think> b<|im_end|>"),
            "{}",
            out.text
        );
    }

    #[test]
    fn template_errors_and_unsupported_features_raise() {
        let r = StructuralRenderer::new(
            "{{ raise_exception('no system message allowed') }}",
            &added(),
            Vec::new(),
        )
        .unwrap();
        let empty = serde_json::Map::new();
        let err = r
            .render(
                &RenderInput {
                    messages: &[],
                    tools: None,
                    documents: None,
                    add_generation_prompt: false,
                    continue_final_message: false,
                    template_kwargs: &empty,
                },
                &tokenizer(),
            )
            .unwrap_err();
        assert!(
            matches!(err, StructuralRenderError::Template(ref m) if m == "no system message allowed")
        );
        let err = StructuralRenderer::new("{{ x | urlize }}", &added(), Vec::new()).err().unwrap();
        assert!(err.to_string().contains("the filter `urlize`"), "{err}");
    }
}
