// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! nemotron4: the Rust port of the nemotron4 parsers (Python plugins
//! `nemotron4_tool_parser_plugin.py` and `nemotron4_reasoning_parser.py`, version 1.3.0rc0),
//! reasoning and tool calls in one parser.
//!
//! Every rule is the Python plugins' (their docstrings and README.md have the reasons); the Rust
//! frontend gives each generated token's id with its text, so control tokens are told from their
//! spellings by id everywhere:
//! - the reasoning split ([R1]-[R6], [R8]): `split`;
//! - the content: tool calls in qwen3_coder's markup ([T2]-[T5], [F1]-[F13]): `calls` and `args`;
//! - the control tokens as the tokenizer spells them ([T5], [R8]): `tokens`.
//!
//! Streamed and not streamed are one code path here, with one difference that only a whole turn can
//! decide (as the Python plugins under vLLM's DelegatingParser): not streamed, a turn in which no
//! call was found has as its content the text as generated (an empty `<tool_call></tool_call>`
//! block or a special token in it included); streamed, what the parser dropped is gone by then. A
//! turn's result is what the Python parsers return when not streamed.

mod args;
mod calls;
mod split;
mod tokens;

use std::collections::{HashMap, HashSet};

use serde_json::Value;
use vllm_tokenizer::{DecodedText, DynTokenizer, TokenAnchor, TokenAttribution};
use xgrammar_structural_tag::ToolChoice;
use xgrammar_structural_tag::format::Format;
use xgrammar_structural_tag::tool::ToolChoiceValue;

use self::calls::{Calls, IdTerm, Item, Out};
use self::split::{Marker, Piece, Split};
use self::tokens::ControlTokens;
use super::{Result, UnifiedParser, UnifiedParserError, UnifiedParserOutput, UnifiedParserRequest};
use crate::output_grammar::{self, BuiltOutputGrammar, OutputGrammarContext};
use crate::tool::Tool;

/// The version of the Python parsers this port implements.
pub const NEMOTRON4_VERSION: &str = "1.3.0rc0";

/// A request's tool choice, as the parser needs it.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum Nemotron4ToolChoice {
    #[default]
    Auto,
    None,
    Required,
    Named(String),
}

/// What a request tells the parser beyond its tools.
#[derive(Debug, Clone)]
pub struct Nemotron4Options {
    pub tool_choice: Nemotron4ToolChoice,
    /// The chat-template kwargs the prompt was rendered with (the server's defaults and the
    /// request's): [R1]'s switches when the prompt can't tell, and [R5]'s `force_nonempty_content`.
    pub template_kwargs: HashMap<String, Value>,
    /// The chat template, to pick among several namespaced spellings of a control token ([T5]).
    pub chat_template: Option<String>,
    /// The tokenizer's special tokens (transformers' `all_special_ids`), which the content parser
    /// drops; `None`: the tokens `tokenizer.json` flags special.
    pub special_ids: Option<Vec<u32>>,
    /// Whether the response is streamed. Not streamed, the content is held until the turn ends (see
    /// the module docs).
    pub stream: bool,
    /// The end tokens the engine stops on (the tokenizer's eos, the generation and model configs'):
    /// stop tokens besides both Nemotron end tokens, as the Python plugins take the generation
    /// config's from vLLM's model config ([R6], [R7]).
    pub end_ids: Option<Vec<u32>>,
}

impl Default for Nemotron4Options {
    fn default() -> Self {
        Self {
            tool_choice: Nemotron4ToolChoice::default(),
            template_kwargs: HashMap::new(),
            chat_template: None,
            special_ids: None,
            stream: true,
            end_ids: None,
        }
    }
}

/// A switch that is on unless set to "0" (the Python plugins' `os.environ.get(name, "1") != "0"`).
fn env_not_off(name: &str) -> bool {
    std::env::var(name).map_or(true, |v| v != "0")
}

/// A switch that is on only when set to "1" (`os.environ.get(name, "0") == "1"`).
fn env_on(name: &str) -> bool {
    std::env::var(name).is_ok_and(|v| v == "1")
}

/// Reasoning and tool calls for Nemotron chat templates; see the module docs.
pub struct Nemotron4UnifiedParser {
    tokenizer: DynTokenizer,
    tools: Vec<Tool>,
    controls: ControlTokens,
    kind_of: HashMap<u32, Marker>,
    stop_ids: HashSet<u32>,
    options: Nemotron4Options,
    prompt_open: Option<bool>,
    split: Option<Split>,
    calls: Option<Calls>,
    /// Not streamed: the content's text as generated, and the content parser's output, held until
    /// the turn ends.
    raw: String,
    held: Vec<Out>,
}

impl Nemotron4UnifiedParser {
    pub fn new(tools: &[Tool], tokenizer: DynTokenizer) -> Result<Self> {
        let controls = tokens::resolve(tokenizer.as_ref(), None)
            .map_err(|message| UnifiedParserError::ParsingFailed { message })?;
        Self::with_controls(tools, tokenizer, controls)
    }

    fn with_controls(
        tools: &[Tool],
        tokenizer: DynTokenizer,
        controls: ControlTokens,
    ) -> Result<Self> {
        for name in [tokens::THINK, tokens::THINK_END] {
            if controls.get(name).is_none() {
                return Err(UnifiedParserError::MissingToken {
                    token: format!(
                        "{name} (neither that spelling nor one namespaced form of it, such as \
                         <nemo:think>: set NEMOTRON4_CONTROL_TOKEN_PREFIX or \
                         NEMOTRON4_CONTROL_TOKENS)"
                    ),
                });
            }
        }
        let mut kind_of = HashMap::new();
        for (name, marker) in [
            (tokens::THINK, Marker::Think),
            (tokens::THINK_END, Marker::End),
            (tokens::TOOL_CALL, Marker::Call),
            (tokens::TOOL_CALL_END, Marker::Close),
        ] {
            if let Some(id) = controls.id(name) {
                kind_of.insert(id, marker);
            }
        }
        let stop_ids = Self::nemotron_end_ids(&controls);
        Ok(Self {
            tokenizer,
            tools: tools.to_vec(),
            controls,
            kind_of,
            stop_ids,
            options: Nemotron4Options::default(),
            prompt_open: None,
            split: None,
            calls: None,
            raw: String::new(),
            held: Vec::new(),
        })
    }

    /// [R6] [R7] Both Nemotron end tokens, as the tokenizer spells them.
    fn nemotron_end_ids(controls: &ControlTokens) -> HashSet<u32> {
        [tokens::TURN_END, tokens::SEQUENCE_END]
            .iter()
            .filter_map(|n| controls.id(n))
            .collect()
    }

    /// The request's options; before the output arrives.
    pub fn set_options(&mut self, options: Nemotron4Options) -> Result<()> {
        if options.chat_template.is_some() {
            // the template may pick among several namespaced spellings
            let controls =
                tokens::resolve(self.tokenizer.as_ref(), options.chat_template.as_deref())
                    .map_err(|message| UnifiedParserError::ParsingFailed { message })?;
            let fresh = Self::with_controls(&self.tools, self.tokenizer.clone(), controls)?;
            self.controls = fresh.controls;
            self.kind_of = fresh.kind_of;
        }
        // [R6] [R7] the stop tokens: both Nemotron end tokens and the end tokens the engine stops on
        self.stop_ids = Self::nemotron_end_ids(&self.controls);
        self.stop_ids.extend(options.end_ids.iter().flatten().copied());
        self.options = options;
        self.split = None;
        self.calls = None;
        self.raw.clear();
        self.held.clear();
        Ok(())
    }

    fn thinking_off(&self) -> bool {
        let kw = &self.options.template_kwargs;
        let effort_none = kw
            .get("reasoning_effort")
            .and_then(Value::as_str)
            .is_some_and(|e| crate::python::py_strip(e).to_lowercase() == "none");
        kw.get("enable_thinking") == Some(&Value::Bool(false)) || effort_none
    }

    fn calls_enabled(&self) -> bool {
        self.kind_of.values().any(|m| *m == Marker::Call)
            && !self.tools.is_empty()
            && self.options.tool_choice != Nemotron4ToolChoice::None
    }

    fn ensure_started(&mut self) {
        if self.split.is_some() {
            return;
        }
        let reasoning = self.prompt_open.unwrap_or_else(|| !self.thinking_off());
        let force =
            self.options.template_kwargs.get("force_nonempty_content") == Some(&Value::Bool(true));
        self.split = Some(Split::new(reasoning, force, self.calls_enabled()));
        let tool_choice = &self.options.tool_choice;
        let bare_calls = env_on("NEMOTRON4_BARE_CALLS")
            || matches!(
                tool_choice,
                Nemotron4ToolChoice::Required | Nemotron4ToolChoice::Named(_)
            );
        let suppress = *tool_choice == Nemotron4ToolChoice::None && !self.tools.is_empty();
        let mut content_literals: Vec<String> = [
            tokens::THINK,
            tokens::THINK_END,
            tokens::TOOL_CALL,
            tokens::TOOL_CALL_END,
        ]
        .iter()
        .map(|n| self.controls.spelling(n).to_string())
        .collect();
        for id in self.drop_ids() {
            if let Some(t) = self.tokenizer.id_to_token(id) {
                content_literals.push(t);
            }
        }
        let schemas = args::ToolSchemas::new(&self.tools, env_not_off("NEMOTRON4_SCHEMA_TYPES"));
        self.calls = Some(Calls::new(
            schemas,
            self.controls.spelling(tokens::TOOL_CALL_END).to_string(),
            &content_literals,
            bare_calls,
            suppress,
            env_not_off("NEMOTRON4_FAST_ARGS"),
        ));
    }

    fn drop_ids(&self) -> Vec<u32> {
        let configured: HashSet<u32> = self.kind_of.keys().copied().collect();
        match &self.options.special_ids {
            Some(ids) => ids.iter().copied().filter(|i| !configured.contains(i)).collect(),
            None => self
                .tokenizer
                .added_vocab()
                .iter()
                .map(|(_, id)| *id)
                .filter(|id| self.tokenizer.is_special_id(*id) && !configured.contains(id))
                .collect(),
        }
    }

    /// The delta's tokens as spans of its text: (id, start, end), in order.
    fn spans(delta: &DecodedText) -> (usize, Vec<(u32, usize, usize)>) {
        let n = delta.attributions.len();
        let offsets: Vec<(u32, usize, bool)> = delta
            .attributions
            .iter()
            .map(|a| match a.anchor {
                TokenAnchor::Visible { byte_offset } => (a.token_id, byte_offset as usize, true),
                TokenAnchor::ZeroWidth { byte_offset } => (a.token_id, byte_offset as usize, false),
            })
            .collect();
        let lead = offsets.iter().find(|(_, _, v)| *v).map_or(delta.text.len(), |(_, o, _)| *o);
        let mut spans = Vec::with_capacity(n);
        for (k, &(id, off, visible)) in offsets.iter().enumerate() {
            if !visible {
                spans.push((id, off, off));
                continue;
            }
            // the bytes up to the next visible token at a later offset; tokens sharing an offset
            // give theirs to the last of them
            let next_same =
                offsets[k + 1..].iter().find(|(_, _, v)| *v).is_some_and(|(_, o, _)| *o == off);
            if next_same {
                spans.push((id, off, off));
                continue;
            }
            let end = offsets[k + 1..]
                .iter()
                .find(|(_, o, v)| *v && *o > off)
                .map_or(delta.text.len(), |(_, o, _)| *o);
            spans.push((id, off, end));
        }
        (lead, spans)
    }

    /// The delta as pieces: control tokens by id, the text between them ([R2]); the stop token's
    /// text is dropped ([R7]).
    fn pieces(&self, delta: &DecodedText) -> Vec<Piece> {
        let (lead, spans) = Self::spans(delta);
        let mut pieces = Vec::new();
        let mut cur = DecodedText::default();
        let text = &delta.text;
        if lead > 0 {
            cur.text.push_str(&text[..lead]);
        }
        for &(id, start, end) in &spans {
            if let Some(&m) = self.kind_of.get(&id) {
                if !cur.is_empty() {
                    pieces.push(Piece {
                        kind: None,
                        dt: std::mem::take(&mut cur),
                    });
                }
                let dt = DecodedText {
                    text: text[start..end].to_string(),
                    attributions: [TokenAttribution {
                        token_id: id,
                        anchor: anchor(0, end > start),
                    }]
                    .into_iter()
                    .collect(),
                };
                pieces.push(Piece { kind: Some(m), dt });
                continue;
            }
            let at = cur.text.len() as u32;
            if self.stop_ids.contains(&id) {
                // [R6] [R7] a stop token (decoded only with include_stop_str_in_output): neither text
                // nor a reasoning token
                continue;
            }
            cur.attributions.push(TokenAttribution {
                token_id: id,
                anchor: anchor(at, end > start),
            });
            cur.text.push_str(&text[start..end]);
        }
        if !cur.is_empty() {
            pieces.push(Piece {
                kind: None,
                dt: cur,
            });
        }
        pieces
    }

    fn content_items<'a>(&self, pieces: &'a [Piece], drops: &HashSet<u32>) -> Vec<Item<'a>> {
        let mut items = Vec::new();
        for p in pieces {
            if let Some(m) = p.kind {
                let t = match m {
                    Marker::Think => IdTerm::ThinkStart,
                    Marker::End => IdTerm::ThinkEnd,
                    Marker::Call => IdTerm::ToolStart,
                    Marker::Close => IdTerm::ToolEnd,
                };
                items.push(Item::Term(t, p.dt.text.as_str()));
                continue;
            }
            // the text, split at the special tokens the engine drops
            let (_, spans) = Self::spans(&p.dt);
            let text = p.dt.text.as_str();
            let mut seg_start = 0;
            for &(id, start, end) in &spans {
                if drops.contains(&id) {
                    if start > seg_start {
                        items.push(Item::Text(&text[seg_start..start]));
                    }
                    items.push(Item::Drop);
                    seg_start = end;
                }
            }
            if seg_start < text.len() {
                items.push(Item::Text(&text[seg_start..]));
            }
        }
        items
    }

    fn emit(&mut self, fed: split::Fed, last: bool, output: &mut UnifiedParserOutput) {
        for r in fed.reasoning {
            output.push_reasoning(r);
        }
        if self.options.tool_choice == Nemotron4ToolChoice::None {
            // tool_choice "none": no tool parsing; the content is the text as generated, call
            // markup included (vLLM's DelegatingParser returns the reasoning parser's content as
            // is, streamed or not)
            for p in fed.content {
                if !p.dt.text.is_empty() {
                    output.push_text(p.dt.text);
                }
            }
            return;
        }
        let drops: HashSet<u32> = self.drop_ids().into_iter().collect();
        let items = self.content_items(&fed.content, &drops);
        let calls = self.calls.as_mut().expect("started");
        let mut outs = calls.feed(&items);
        if last {
            outs.extend(calls.finish());
        }
        if !self.options.stream {
            // not streamed: as vLLM's DelegatingParser, the content parser's result only if it
            // found a call, else the text as generated (blank: none, under a forced tool choice)
            for p in &fed.content {
                self.raw.push_str(&p.dt.text);
            }
            self.held.extend(outs);
            if !last {
                return;
            }
            let held = std::mem::take(&mut self.held);
            let raw = std::mem::take(&mut self.raw);
            if held.iter().any(|o| matches!(o, Out::Call(_))) {
                outs = held;
            } else {
                let forced = matches!(
                    self.options.tool_choice,
                    Nemotron4ToolChoice::Required | Nemotron4ToolChoice::Named(_)
                );
                outs = if raw.is_empty() || (forced && crate::python::py_strip(&raw).is_empty()) {
                    Vec::new()
                } else {
                    vec![Out::Text(raw)]
                };
            }
        }
        for o in outs {
            match o {
                Out::Text(t) => output.push_text(t),
                Out::Call(c) => output.push_call(c),
            }
        }
    }
}

impl From<&ToolChoice> for Nemotron4ToolChoice {
    fn from(choice: &ToolChoice) -> Self {
        match choice {
            ToolChoice::Value(ToolChoiceValue::None) => Self::None,
            ToolChoice::Value(ToolChoiceValue::Required) => Self::Required,
            ToolChoice::NamedFunction(named) => Self::Named(named.function.name.clone()),
            // auto, and the allowed-tools shapes the chat API does not produce
            _ => Self::Auto,
        }
    }
}

/// [T5] The format with the tokenizer's spellings of the control tokens in its strings (triggers,
/// tags, excludes), the tools' JSON schemas left as they are.
fn respell(format: Format, renamed: &[(&str, &str)]) -> Option<Format> {
    fn walk(v: Value, renamed: &[(&str, &str)]) -> Value {
        match v {
            Value::String(s) => {
                let mut out = String::with_capacity(s.len());
                let mut rest = s.as_str();
                'outer: while !rest.is_empty() {
                    // longest spelling first, as the Python plugin's alternation sorted by length
                    for (from, to) in renamed {
                        if let Some(tail) = rest.strip_prefix(from) {
                            out.push_str(to);
                            rest = tail;
                            continue 'outer;
                        }
                    }
                    let c = rest.chars().next().expect("non-empty");
                    out.push(c);
                    rest = &rest[c.len_utf8()..];
                }
                Value::String(out)
            }
            Value::Array(a) => Value::Array(a.into_iter().map(|x| walk(x, renamed)).collect()),
            Value::Object(o) => Value::Object(
                o.into_iter()
                    .map(|(k, x)| {
                        let x = if k == "json_schema" {
                            x
                        } else {
                            walk(x, renamed)
                        };
                        (k, x)
                    })
                    .collect(),
            ),
            other => other,
        }
    }
    let value = serde_json::to_value(format).ok()?;
    serde_json::from_value(walk(value, renamed)).ok()
}

fn anchor(at: u32, visible: bool) -> TokenAnchor {
    if visible {
        TokenAnchor::Visible { byte_offset: at }
    } else {
        TokenAnchor::ZeroWidth { byte_offset: at }
    }
}

impl UnifiedParser for Nemotron4UnifiedParser {
    fn create(tools: &[Tool], tokenizer: DynTokenizer) -> Result<Box<dyn UnifiedParser>>
    where
        Self: Sized + 'static,
    {
        Ok(Box::new(Self::new(tools, tokenizer)?))
    }

    fn configure(&mut self, request: &UnifiedParserRequest<'_>) -> Result<()> {
        self.set_options(Nemotron4Options {
            tool_choice: request.tool_choice.into(),
            template_kwargs: request.template_kwargs.clone(),
            chat_template: request.chat_template.map(str::to_string),
            special_ids: request.special_token_ids.map(<[u32]>::to_vec),
            end_ids: request.end_token_ids.map(<[u32]>::to_vec),
            stream: request.stream,
        })
    }

    fn build_output_grammar(
        &self,
        ctx: &OutputGrammarContext<'_>,
    ) -> output_grammar::Result<Option<BuiltOutputGrammar>> {
        // as the Python tool parser: the qwen_3_coder structural tag (forced tool choice, strict
        // tools), applied after the reasoning, with this tokenizer's control-token spellings ([T5])
        let builder = xgrammar_structural_tag::Model::Qwen3Coder.builder();
        let Some(format) = output_grammar::visible_format_from_builder(Some(builder), ctx)? else {
            return Ok(None);
        };
        let mut renamed: Vec<(&str, &str)> = tokens::CONTROL_TOKENS
            .iter()
            .map(|name| (*name, self.controls.spelling(name)))
            .filter(|(name, spelling)| name != spelling)
            .collect();
        renamed.sort_by_key(|(name, _)| std::cmp::Reverse(name.len()));
        let format = if renamed.is_empty() {
            format
        } else {
            respell(format, &renamed).ok_or(
                output_grammar::OutputGrammarError::UnexpectedBuilderFormat {
                    builder: "qwen_3_coder",
                },
            )?
        };
        Ok(Some(BuiltOutputGrammar::final_output_only(format)))
    }

    fn initialize(&mut self, prompt_token_ids: &[u32]) -> Result<()> {
        // [R1] the last <think> or </think> of the prompt's final turn
        let think = self.controls.id(tokens::THINK);
        let end = self.controls.id(tokens::THINK_END);
        let turn = self.controls.id(tokens::TURN_START);
        self.prompt_open = None;
        for &t in prompt_token_ids.iter().rev() {
            if Some(t) == think {
                self.prompt_open = Some(true);
                break;
            }
            if Some(t) == end {
                self.prompt_open = Some(false);
                break;
            }
            if Some(t) == turn {
                break;
            }
        }
        self.split = None;
        self.calls = None;
        self.raw.clear();
        self.held.clear();
        Ok(())
    }

    fn preserve_special_tokens(&self) -> bool {
        true // as the Python tool parser's engine sets skip_special_tokens false
    }

    fn parse_into(&mut self, delta: DecodedText, output: &mut UnifiedParserOutput) -> Result<()> {
        self.ensure_started();
        let pieces = self.pieces(&delta);
        let fed = self.split.as_mut().expect("started").feed(pieces, false);
        self.emit(fed, false, output);
        Ok(())
    }

    fn finish(&mut self) -> Result<UnifiedParserOutput> {
        self.ensure_started();
        let mut output = UnifiedParserOutput::default();
        let fed = self.split.as_mut().expect("started").feed(Vec::new(), true);
        self.emit(fed, true, &mut output);
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vllm_tokenizer::DynTokenizer;
    use vllm_tokenizer::test_utils::TestTokenizer;

    use super::*;
    use crate::unified::UnifiedParserEvent;

    fn tokenizer() -> DynTokenizer {
        Arc::new(
            TestTokenizer::new()
                .with_regular_token("<think>", 300)
                .with_regular_token("</think>", 301)
                .with_regular_token("<tool_call>", 302)
                .with_regular_token("</tool_call>", 303)
                .with_special_token("<|im_start|>", 304)
                .with_special_token("<|im_end|>", 305)
                .with_special_token("<|endoftext|>", 306),
        )
    }

    /// (reasoning, reasoning tokens, content) of a turn cut while reasoning that ends on
    /// `<|endoftext|>`, decoded with the stop token in the text (as `include_stop_str_in_output`
    /// has it), with `end_ids` as the engine's end tokens.
    fn cut_turn(end_ids: Option<Vec<u32>>) -> (String, usize, String) {
        let tok = tokenizer();
        let mut parser = Nemotron4UnifiedParser::new(&[], tok.clone()).unwrap();
        parser
            .set_options(Nemotron4Options {
                end_ids,
                ..Nemotron4Options::default()
            })
            .unwrap();
        let prompt = tok.encode("<|im_start|>assistant\n<think>\n", false).unwrap();
        parser.initialize(&prompt).unwrap();
        let mut ids = tok.encode("Plan, still going", false).unwrap();
        ids.push(306);
        let mut decoder = tok.create_decode_stream(&prompt, false, 0);
        let mut out = UnifiedParserOutput::default();
        for &id in &ids {
            decoder.push_token(id).unwrap();
        }
        if let Some(dt) = decoder.next_chunk() {
            parser.parse_into(dt, &mut out).unwrap();
        }
        if let (Some(dt), _) = decoder.flush(None).unwrap() {
            parser.parse_into(dt, &mut out).unwrap();
        }
        out.append(parser.finish().unwrap());
        let (mut reasoning, mut count, mut content) = (String::new(), 0, String::new());
        for event in out.events {
            match event {
                UnifiedParserEvent::Reasoning(dt) => {
                    count += dt.attributions.len();
                    reasoning.push_str(&dt.text);
                }
                UnifiedParserEvent::Text(text) => content.push_str(&text),
                UnifiedParserEvent::ToolCall(_) => {}
            }
        }
        (reasoning, count, content)
    }

    #[test]
    fn the_engines_end_tokens_are_stop_tokens() {
        // [R6] [R7] an end token the generation config names (Qwen3.5-3.8's <|endoftext|>): its
        // text dropped and not counted as reasoning, as both Nemotron end tokens'
        let text_ids = tokenizer().encode("Plan, still going", false).unwrap().len();
        let (reasoning, count, content) = cut_turn(Some(vec![305, 306]));
        assert_eq!(reasoning, "Plan, still going");
        assert_eq!(count, text_ids);
        assert_eq!(content, "");
        // without them it is an ordinary special token of the reasoning
        let (reasoning, count, _) = cut_turn(None);
        assert_eq!(reasoning, "Plan, still going<|endoftext|>");
        assert_eq!(count, text_ids + 1);
    }
}
