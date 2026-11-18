// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! The content side of a turn: tool calls in qwen3_coder's markup, found as the Python
//! nemotron4_tool parser finds them (vLLM's parser engine with the qwen3 grammar, and the plugin's
//! Nemotron4StreamingEngine and Nemotron4Parser).
//!
//! - The `<tool_call>` / `</tool_call>` / `<think>` / `</think>` terminals are token ids only
//!   ([T2]): their spellings written with ordinary tokens are text. The function and parameter
//!   markup (`<function=`, `</function>`, `<parameter=`, `</parameter>`, `>`) is text, lexed with
//!   prefix hold-back as the engine's lexer does.
//! - A call opens only once its markup shows it is one ([F4], [F5]): after `<tool_call>`, in the
//!   function name, and after `<function=NAME>`, until a `<parameter=` or `</function>` commits it;
//!   anything else reverts the held text to content. Only the `<tool_call>` token opens one ([T4]),
//!   unless bare calls are allowed.
//! - A `</think>` token in the content is text ([F6]). After `</function>`, whitespace, another
//!   function or the closer belong to the block; other text ends it and is content ([F12]).
//! - Content keeps its whitespace ([F13]); text after the calls follows the text before them
//!   ([F7]), as one non-streamed parse returns it.

use std::collections::HashSet;

use super::args::{self, OpenString, ToolSchemas};
use crate::python::{py_isspace, py_lstrip, py_strip};
use crate::tool::ToolCallDelta;

/// A token-id terminal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum IdTerm {
    ThinkStart,
    ThinkEnd,
    ToolStart,
    ToolEnd,
}

/// One item of the content stream.
#[derive(Debug, Clone)]
pub(crate) enum Item<'a> {
    /// A token-id terminal, with its text.
    Term(IdTerm, &'a str),
    /// A special token the engine drops (its text with it).
    Drop,
    /// Text, lexed for the text terminals.
    Text(&'a str),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Term {
    Id(IdTerm),
    FuncPrefix,
    FuncEnd,
    ParamStart,
    ParamEnd,
    CloseAngle,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    Content,
    ToolPreamble,
    ToolName,
    ToolArgs,
    ToolBetween,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Hold {
    Pre,
    Name,
    After,
}

/// What the engine emits; values as the Python engine's SemanticEvents carry them.
#[derive(Debug, Clone)]
enum Event {
    Text(String),
    CallStart(usize),
    CallName(usize, String),
    ArgChunk(usize, String),
    CallEnd(usize),
}

const FUNC_PREFIX: &str = "<function=";
const FUNC_END: &str = "</function>";
const PARAM_START: &str = "<parameter=";
const PARAM_END: &str = "</parameter>";
const MAX_NAME: usize = 256;

/// The engine's incremental lexer: literals matched longest-first at the buffer's start, a buffer
/// that is a proper prefix of a literal held until more text arrives (or the stream ends).
#[derive(Debug, Clone)]
struct Lexer {
    /// (literal, its terminal, or `None` for a literal that is content: the token-id terminals' and
    /// the dropped special tokens' spellings, which a stream with token ids reads as text)
    literals: Vec<(String, Option<Term>)>,
    prefixes: HashSet<String>,
    first_chars: HashSet<char>,
    buffer: String,
}

#[derive(Debug, Clone)]
enum Lexed {
    Content(String),
    Term(Term, String),
}

impl Lexer {
    fn new(content_literals: &[String]) -> Self {
        let mut literals: Vec<(String, Option<Term>)> = vec![
            (FUNC_PREFIX.into(), Some(Term::FuncPrefix)),
            (FUNC_END.into(), Some(Term::FuncEnd)),
            (PARAM_START.into(), Some(Term::ParamStart)),
            (PARAM_END.into(), Some(Term::ParamEnd)),
            (">".into(), Some(Term::CloseAngle)),
        ];
        for lit in content_literals {
            if !lit.is_empty() && !literals.iter().any(|(l, _)| l == lit) {
                literals.push((lit.clone(), None));
            }
        }
        let mut prefixes = HashSet::new();
        let mut first_chars = HashSet::new();
        for (lit, _) in &literals {
            let chars: Vec<(usize, char)> = lit.char_indices().collect();
            first_chars.insert(chars[0].1);
            for &(k, _) in &chars[1..] {
                prefixes.insert(lit[..k].to_string());
            }
        }
        Self {
            literals,
            prefixes,
            first_chars,
            buffer: String::new(),
        }
    }

    fn feed(&mut self, text: &str) -> Vec<Lexed> {
        if self.buffer.is_empty() && !text.chars().any(|c| self.first_chars.contains(&c)) {
            return if text.is_empty() {
                Vec::new()
            } else {
                vec![Lexed::Content(text.to_string())]
            };
        }
        self.buffer.push_str(text);
        self.drain(false)
    }

    fn flush(&mut self) -> Vec<Lexed> {
        let mut out = Vec::new();
        if !self.buffer.is_empty() {
            out = self.drain(true);
        }
        if !self.buffer.is_empty() {
            out.push(Lexed::Content(std::mem::take(&mut self.buffer)));
        }
        out
    }

    fn drain(&mut self, last: bool) -> Vec<Lexed> {
        let mut out = Vec::new();
        while !self.buffer.is_empty() {
            if !self.buffer.chars().any(|c| self.first_chars.contains(&c)) {
                out.push(Lexed::Content(std::mem::take(&mut self.buffer)));
                break;
            }
            let best = self
                .literals
                .iter()
                .filter(|(lit, _)| self.buffer.starts_with(lit.as_str()))
                .max_by_key(|(lit, _)| lit.len())
                .cloned();
            if self.prefixes.contains(&self.buffer) && !last {
                match &best {
                    Some((lit, _)) => {
                        let longer = self.literals.iter().any(|(l, _)| {
                            l.len() > lit.len() && l.starts_with(self.buffer.as_str())
                        });
                        if !longer {
                            let (lit, term) = best.unwrap();
                            self.buffer.drain(..lit.len());
                            out.push(Self::token(lit, term));
                            continue;
                        }
                        break;
                    }
                    None => break,
                }
            }
            if let Some((lit, term)) = best {
                self.buffer.drain(..lit.len());
                out.push(Self::token(lit, term));
            } else {
                let end = self.content_boundary();
                let piece: String = self.buffer.drain(..end).collect();
                out.push(Lexed::Content(piece));
            }
        }
        out
    }

    fn token(lit: String, term: Option<Term>) -> Lexed {
        match term {
            Some(t) => Lexed::Term(t, lit),
            None => Lexed::Content(lit),
        }
    }

    /// `_find_content_boundary`: the first position (after the first character) where a literal, or
    /// the start of one, begins; else the whole buffer.
    fn content_boundary(&self) -> usize {
        let buf = &self.buffer;
        for (i, c) in buf.char_indices().skip(1) {
            if !self.first_chars.contains(&c) {
                continue;
            }
            let rest = &buf[i..];
            for (lit, _) in &self.literals {
                if rest.starts_with(lit.as_str()) || lit.starts_with(rest) {
                    return i;
                }
            }
        }
        buf.len()
    }
}

/// `_BODY_START = \s*<\s*parameter\s*=`: matched at the start of `s`, or `s` the start of a match.
fn body_start(s: &str) -> (bool, bool) {
    // (matches at the start, s is a prefix of a match)
    let t = s.trim_start_matches(char::is_whitespace);
    if t.is_empty() {
        return (false, true);
    }
    let Some(r) = t.strip_prefix('<') else {
        return (false, false);
    };
    let r = r.trim_start_matches(char::is_whitespace);
    if r.is_empty() {
        return (false, true);
    }
    if !r.starts_with("parameter") {
        return (false, "parameter".starts_with(r));
    }
    let r = r["parameter".len()..].trim_start_matches(char::is_whitespace);
    if r.is_empty() {
        return (false, true);
    }
    if r.starts_with('=') {
        (true, r == "=")
    } else {
        (false, false)
    }
}

#[derive(Debug, Clone, Default)]
struct Slot {
    name: String,
    args: String,
    name_sent: bool,
    string_keys: Option<HashSet<String>>,
    streamed_json: String,
    open: Option<OpenString>,
}

/// The tool-call state machine over the content stream, with the delta assembly of the Python
/// parser.
#[derive(Debug, Clone)]
pub(crate) struct Calls {
    schemas: ToolSchemas,
    lexer: Lexer,
    state: State,
    hold: Option<Hold>,
    wrapped: bool,
    held: String,
    held_name: String,
    held_after: String,
    between: String,
    implicit_close: bool,
    closer: String,
    bare_calls: bool,
    suppress: bool,
    fast_args: bool,
    tool_index: i64,
    slots: Vec<Slot>,
    // content assembly ([F7] [F13]): text before the first call, and the text after calls
    // (deferred)
    seen_call: bool,
    lead: String,
    lead_released: bool,
    rest: String,
    rest_released: bool,
}

/// What one feed produced, in order.
#[derive(Debug, Clone)]
pub(crate) enum Out {
    Text(String),
    Call(ToolCallDelta),
}

impl Calls {
    /// `closer`: the tokenizer's spelling of `</tool_call>` ([F12] [T5]); `content_literals`: the
    /// spellings the lexer reads as text (the token-id terminals' and the dropped special tokens');
    /// `bare_calls`: [T4]'s exception; `suppress`: tool_choice "none" (calls are parsed and
    /// dropped).
    pub(crate) fn new(
        schemas: ToolSchemas,
        closer: String,
        content_literals: &[String],
        bare_calls: bool,
        suppress: bool,
        fast_args: bool,
    ) -> Self {
        Self {
            schemas,
            lexer: Lexer::new(content_literals),
            state: State::Content,
            hold: None,
            wrapped: false,
            held: String::new(),
            held_name: String::new(),
            held_after: String::new(),
            between: String::new(),
            implicit_close: false,
            closer,
            bare_calls,
            suppress,
            fast_args,
            tool_index: -1,
            slots: Vec::new(),
            seen_call: false,
            lead: String::new(),
            lead_released: false,
            rest: String::new(),
            rest_released: false,
        }
    }

    /// Feed content items; the output is what a client may receive now.
    pub(crate) fn feed(&mut self, items: &[Item<'_>]) -> Vec<Out> {
        let mut events = Vec::new();
        for item in items {
            match *item {
                Item::Term(t, text) => {
                    let flushed = self.lexer.flush();
                    self.process_lexed(flushed, &mut events);
                    let ev = self.on_terminal(Term::Id(t), text);
                    events.extend(ev);
                }
                Item::Drop => {
                    let flushed = self.lexer.flush();
                    self.process_lexed(flushed, &mut events);
                    let ev = self.on_drop();
                    events.extend(ev);
                }
                Item::Text(text) => {
                    let lexed = self.lexer.feed(text);
                    self.process_lexed(lexed, &mut events);
                }
            }
        }
        self.assemble(events, false)
    }

    /// The end of the output.
    pub(crate) fn finish(&mut self) -> Vec<Out> {
        let mut events = Vec::new();
        if self.hold.is_none() {
            self.base_finish(&mut events);
            self.between.clear();
            self.implicit_close = false;
        } else {
            let flushed = self.lexer.flush();
            self.process_lexed(flushed, &mut events);
            if let Some(hold) = self.hold {
                if hold == Hold::After && self.wrapped {
                    events.extend(self.commit());
                } else {
                    events.extend(self.revert(None, ""));
                }
            }
            self.base_finish(&mut events);
        }
        self.assemble(events, true)
    }

    fn base_finish(&mut self, events: &mut Vec<Event>) {
        let flushed = self.lexer.flush();
        self.process_lexed(flushed, events);
        if matches!(
            self.state,
            State::ToolPreamble | State::ToolArgs | State::ToolName | State::ToolBetween
        ) {
            if self.tool_index >= 0 {
                events.push(Event::CallEnd(self.idx()));
            }
            self.state = State::Content;
        }
    }

    fn process_lexed(&mut self, lexed: Vec<Lexed>, events: &mut Vec<Event>) {
        for tok in lexed {
            match tok {
                Lexed::Content(text) => {
                    if !text.is_empty() {
                        let ev = self.emit_for_state(&text);
                        events.extend(ev);
                    }
                }
                Lexed::Term(t, text) => {
                    let ev = self.on_terminal(t, &text);
                    events.extend(ev);
                }
            }
        }
    }

    // ── the state machine (StreamingParserEngine + Nemotron4StreamingEngine) ─────────────────────

    fn has_transition(state: State, t: Term) -> bool {
        use IdTerm::*;
        use State::*;
        matches!(
            (state, t),
            (Content, Term::Id(ToolStart))
                | (Content, Term::FuncPrefix)
                | (ToolPreamble, Term::Id(ToolEnd))
                | (ToolPreamble, Term::FuncPrefix)
                | (ToolName, Term::CloseAngle)
                | (ToolName, Term::FuncEnd)
                | (ToolArgs, Term::FuncEnd)
                | (ToolArgs, Term::ParamStart)
                | (ToolArgs, Term::ParamEnd)
                | (ToolBetween, Term::Id(ToolEnd))
                | (ToolBetween, Term::Id(ToolStart))
                | (ToolBetween, Term::FuncPrefix)
        )
    }

    /// The base engine's `_on_terminal` with the qwen3 transitions (no `(CONTENT, THINK_END)`:
    /// [F6]).
    fn base_on_terminal(&mut self, t: Term, value: &str) -> Vec<Event> {
        use IdTerm::*;
        use State::*;
        // (next state, a call starts, a call ends, the value is an argument chunk); events carry
        // the call's index
        let (next, start, end, arg) = match (self.state, t) {
            (Content, Term::Id(ToolStart)) => (ToolPreamble, true, false, false),
            (Content, Term::FuncPrefix) => (ToolName, true, false, false),
            (ToolPreamble, Term::Id(ToolEnd)) => (Content, false, true, false),
            (ToolPreamble, Term::FuncPrefix) => (ToolName, false, false, false),
            (ToolName, Term::CloseAngle) => (ToolArgs, false, false, false),
            (ToolName, Term::FuncEnd) | (ToolArgs, Term::FuncEnd) => {
                (ToolBetween, false, true, false)
            }
            (ToolArgs, Term::ParamStart) | (ToolArgs, Term::ParamEnd) => {
                (ToolArgs, false, false, true)
            }
            (ToolBetween, Term::Id(ToolEnd)) => (Content, false, false, false),
            (ToolBetween, Term::Id(ToolStart)) => (ToolPreamble, true, false, false),
            (ToolBetween, Term::FuncPrefix) => (ToolName, true, false, false),
            _ => return self.emit_for_state(value),
        };
        self.state = next;
        let mut events = Vec::new();
        if start {
            self.tool_index += 1;
            events.push(Event::CallStart(self.idx()));
        }
        if end {
            events.push(Event::CallEnd(self.idx()));
        }
        if arg {
            events.push(Event::ArgChunk(self.idx(), value.into()));
        }
        events
    }

    /// The current call's index (the engine's `tool_index`).
    fn idx(&self) -> usize {
        self.tool_index.max(0) as usize
    }

    fn base_emit_for_state(&mut self, text: &str) -> Vec<Event> {
        match self.state {
            State::ToolArgs => vec![Event::ArgChunk(self.idx(), text.into())],
            State::Content => vec![Event::Text(text.into())],
            State::ToolName => vec![Event::CallName(self.idx(), text.into())],
            _ => vec![],
        }
    }

    fn start_hold(&mut self, hold: Hold, value: &str, wrapped: bool, state: State) {
        self.clear_hold();
        self.hold = Some(hold);
        self.wrapped = wrapped;
        self.held = value.into();
        self.state = state;
    }

    fn clear_hold(&mut self) {
        self.hold = None;
        self.wrapped = false;
        self.held.clear();
        self.held_name.clear();
        self.held_after.clear();
    }

    fn on_drop(&mut self) -> Vec<Event> {
        // dropped in every state (in a hold, by _on_held_terminal; else no transition, and the
        // engine drops it)
        Vec::new()
    }

    fn on_terminal(&mut self, t: Term, value: &str) -> Vec<Event> {
        if self.hold.is_some() {
            return self.on_held_terminal(t, value);
        }
        let state = self.state;
        if state == State::ToolBetween {
            if !self.between.is_empty() {
                let mut events = self.flush_between();
                events.extend(self.on_terminal(t, value));
                return events;
            }
            if !Self::has_transition(state, t) {
                return Vec::new(); // call markup after </function>: dropped
            }
        }
        if self.implicit_close && matches!(t, Term::Id(IdTerm::ToolStart | IdTerm::ToolEnd)) {
            self.implicit_close = false;
            if t == Term::Id(IdTerm::ToolEnd) && state == State::Content {
                // the late closer of a block that the text after its </function> ended
                return Vec::new();
            }
        }
        if t == Term::Id(IdTerm::ToolStart) && matches!(state, State::Content | State::ToolBetween)
        {
            self.start_hold(Hold::Pre, value, true, State::ToolPreamble); // [F5]
            return Vec::new();
        }
        if t == Term::FuncPrefix && state == State::Content && !self.bare_calls {
            return self.emit_for_state(value); // [T4] no <tool_call> before it: text
        }
        if t == Term::FuncPrefix && matches!(state, State::Content | State::ToolBetween) {
            self.start_hold(
                Hold::Name,
                value,
                state == State::ToolBetween,
                State::ToolName,
            ); // [F4]
            return Vec::new();
        }
        if t == Term::FuncPrefix && state == State::ToolArgs {
            return self.emit_for_state(value); // argument text
        }
        self.base_on_terminal(t, value)
    }

    fn on_held_terminal(&mut self, t: Term, value: &str) -> Vec<Event> {
        match self.hold.expect("in a hold") {
            Hold::Pre => {
                if t == Term::FuncPrefix {
                    self.hold = Some(Hold::Name);
                    self.held.push_str(value);
                    self.state = State::ToolName;
                    return Vec::new();
                }
                if t == Term::Id(IdTerm::ToolEnd) {
                    self.clear_hold(); // <tool_call></tool_call>: no call, nothing to show
                    self.state = State::Content;
                    return Vec::new();
                }
                self.revert(Some(t), value)
            }
            Hold::Name => {
                if t == Term::CloseAngle && self.name_ok() {
                    self.hold = Some(Hold::After);
                    self.held.push_str(value);
                    return Vec::new();
                }
                if t == Term::FuncEnd && self.name_ok() {
                    let mut events = self.commit();
                    events.extend(self.base_on_terminal(t, value));
                    return events;
                }
                self.revert(Some(t), value)
            }
            Hold::After => {
                if matches!(t, Term::ParamStart | Term::FuncEnd) {
                    let mut events = self.commit();
                    events.extend(self.base_on_terminal(t, value));
                    return events;
                }
                if t == Term::Id(IdTerm::ToolEnd) && self.wrapped {
                    let mut events = self.commit();
                    events.push(Event::CallEnd(self.idx()));
                    self.state = State::Content;
                    return events;
                }
                self.revert(Some(t), value)
            }
        }
    }

    fn emit_for_state(&mut self, text: &str) -> Vec<Event> {
        let Some(hold) = self.hold else {
            if self.state == State::ToolBetween {
                return self.after_function(text); // [F12]
            }
            return self.base_emit_for_state(text);
        };
        self.held.push_str(text);
        if text.is_empty() {
            return Vec::new();
        }
        match hold {
            Hold::Pre => {
                if py_strip(text).is_empty() {
                    Vec::new()
                } else {
                    self.revert(None, "")
                }
            }
            Hold::After => {
                self.held_after.push_str(text);
                let (matched, partial) = body_start(&self.held_after);
                if matched {
                    return self.commit();
                }
                if !py_strip(text).is_empty() && !partial {
                    return self.revert(None, "");
                }
                Vec::new()
            }
            Hold::Name => {
                self.held_name.push_str(text);
                let name = &self.held_name;
                let stripped = py_strip(name);
                if name.chars().count() > MAX_NAME
                    || (name.contains('\n') && !stripped.is_empty())
                    || (!self.wrapped && stripped.chars().any(py_isspace))
                {
                    return self.revert(None, "");
                }
                Vec::new()
            }
        }
    }

    /// [F12] Text after a call's `</function>`, inside its block.
    fn after_function(&mut self, text: &str) -> Vec<Event> {
        let buf = format!("{}{}", self.between, text);
        let rest = py_lstrip(&buf);
        let closer = self.closer.clone();
        if rest.is_empty()
            || (rest.chars().count() < closer.chars().count() && closer.starts_with(rest))
        {
            self.between = buf;
            return Vec::new();
        }
        self.between.clear();
        self.state = State::Content;
        if let Some(after) = rest.strip_prefix(closer.as_str()) {
            return if after.is_empty() {
                Vec::new()
            } else {
                self.base_emit_for_state(after)
            };
        }
        self.implicit_close = true;
        self.base_emit_for_state(&buf)
    }

    fn flush_between(&mut self) -> Vec<Event> {
        let text = std::mem::take(&mut self.between);
        if py_strip(&text).is_empty() {
            return Vec::new();
        }
        self.state = State::Content;
        self.implicit_close = true;
        self.base_emit_for_state(&text)
    }

    fn name_ok(&self) -> bool {
        let name = py_strip(&self.held_name);
        !name.is_empty() && (self.wrapped || self.schemas.is_empty() || self.schemas.declares(name))
    }

    fn commit(&mut self) -> Vec<Event> {
        let name = py_strip(&self.held_name).to_string();
        let after = std::mem::take(&mut self.held_after);
        self.clear_hold();
        self.tool_index += 1;
        self.state = State::ToolArgs;
        let mut events = vec![
            Event::CallStart(self.idx()),
            Event::CallName(self.idx(), name),
        ];
        if !after.is_empty() {
            events.extend(self.emit_for_state(&after));
        }
        events
    }

    fn revert(&mut self, t: Option<Term>, value: &str) -> Vec<Event> {
        let text = std::mem::take(&mut self.held);
        self.clear_hold();
        self.state = State::Content;
        let mut events = if text.is_empty() {
            Vec::new()
        } else {
            vec![Event::Text(text)]
        };
        if let Some(t) = t {
            events.extend(self.on_terminal(t, value));
        }
        events
    }

    // ── delta assembly (ParserEngine._events_to_delta and the tool-call slots) ───────────────────

    fn assemble(&mut self, events: Vec<Event>, finished: bool) -> Vec<Out> {
        let mut out = Vec::new();
        for e in events {
            match e {
                Event::Text(t) => self.text(t, &mut out),
                Event::CallStart(idx) => {
                    if self.suppress {
                        continue;
                    }
                    if !self.seen_call {
                        self.seen_call = true;
                        // text before the first call: whitespace only is dropped when calls follow
                        if !self.lead_released {
                            self.lead.clear();
                        }
                    }
                    while self.slots.len() <= idx {
                        self.slots.push(Slot::default());
                    }
                }
                Event::CallName(idx, name) => {
                    if self.suppress {
                        continue;
                    }
                    if let Some(slot) = self.slots.get_mut(idx) {
                        slot.name.push_str(&name);
                    }
                }
                Event::ArgChunk(idx, v) => {
                    if self.suppress {
                        continue;
                    }
                    self.arg_chunk(idx, &v, &mut out);
                }
                Event::CallEnd(idx) => {
                    if self.suppress {
                        continue;
                    }
                    self.tool_end(idx, &mut out);
                }
            }
        }
        if finished {
            if !self.seen_call && !self.lead_released && !self.lead.is_empty() {
                // no calls: the whitespace-only content is the content
                out.push(Out::Text(std::mem::take(&mut self.lead)));
            }
            self.lead.clear();
            self.rest.clear();
        }
        out
    }

    fn text(&mut self, t: String, out: &mut Vec<Out>) {
        if t.is_empty() {
            return;
        }
        if !self.seen_call {
            if self.lead_released {
                out.push(Out::Text(t));
                return;
            }
            self.lead.push_str(&t);
            if !py_strip(&self.lead).is_empty() {
                self.lead_released = true;
                out.push(Out::Text(std::mem::take(&mut self.lead)));
            }
            return;
        }
        // [F7] text after a call: shown only if some of it is more than whitespace
        if self.rest_released {
            out.push(Out::Text(t));
            return;
        }
        self.rest.push_str(&t);
        if !py_strip(&self.rest).is_empty() {
            self.rest_released = true;
            out.push(Out::Text(std::mem::take(&mut self.rest)));
        }
    }

    fn emit_name(&mut self, idx: usize, out: &mut Vec<Out>, arguments: String) {
        let slot = &mut self.slots[idx];
        let name = slot.name.clone();
        if name.is_empty() {
            return;
        }
        slot.name_sent = true;
        slot.string_keys = self.schemas.string_keys(&name);
        out.push(Out::Call(ToolCallDelta {
            tool_index: idx,
            name: Some(name),
            arguments,
        }));
    }

    fn arg_chunk(&mut self, idx: usize, value: &str, out: &mut Vec<Out>) {
        if idx >= self.slots.len() {
            return;
        }
        if !value.is_empty() {
            self.slots[idx].args.push_str(value);
        }
        if !self.slots[idx].name_sent {
            if !self.slots[idx].name.is_empty() {
                self.emit_name(idx, out, String::new());
            }
            return;
        }
        if value.is_empty() {
            return;
        }
        if let Some(diff) = self.compute_arg_delta(idx, value) {
            out.push(Out::Call(ToolCallDelta {
                tool_index: idx,
                name: None,
                arguments: diff,
            }));
        }
    }

    /// `_compute_arg_delta` with [F10]'s fast path.
    fn compute_arg_delta(&mut self, idx: usize, raw_delta: &str) -> Option<String> {
        if let Some(mut ext) = self.slots[idx].open.take()
            && let Some(diff) = ext.extend(raw_delta)
        {
            self.slots[idx].open = Some(ext);
            if diff.is_empty() {
                return None;
            }
            self.slots[idx].streamed_json.push_str(&diff);
            return Some(diff);
        }
        let diff = self.base_arg_delta(idx);
        if self.fast_args {
            self.track_open_string(idx);
        }
        diff
    }

    fn base_arg_delta(&mut self, idx: usize) -> Option<String> {
        let slot = &self.slots[idx];
        let current = if slot.name.is_empty() {
            crate::python::dumps(&crate::python::PyVal::Dict(args::convert(&slot.args, true)))
        } else {
            args::partial_arguments(&self.schemas, &slot.name, &slot.args)
        };
        let safe = args::safe_arg_prefix(&current, slot.string_keys.as_ref());
        let slot = &mut self.slots[idx];
        if safe.is_empty() || safe == slot.streamed_json {
            return None;
        }
        let diff = if slot.streamed_json.is_empty() {
            safe.clone()
        } else {
            safe.strip_prefix(slot.streamed_json.as_str())?.to_string()
        };
        if diff.is_empty() {
            return None;
        }
        slot.streamed_json = safe;
        Some(diff)
    }

    fn track_open_string(&mut self, idx: usize) {
        let slot = &self.slots[idx];
        if !slot.name_sent {
            return;
        }
        let params = args::parameters(&slot.args);
        let Some((name, (a, b), last)) = params.last() else {
            return;
        };
        if !last {
            return;
        }
        if let Some(keys) = &slot.string_keys
            && !keys.contains(name)
        {
            return;
        }
        let region = &slot.args[*a..*b];
        if let Some(ext) = OpenString::start(region) {
            let want = format!("\"{}", ext.streamed(region));
            if slot.streamed_json.ends_with(&want) {
                self.slots[idx].open = Some(ext);
            }
        }
    }

    /// `_handle_tool_end`: the rest of the final arguments.
    fn tool_end(&mut self, idx: usize, out: &mut Vec<Out>) {
        if idx >= self.slots.len() {
            return;
        }
        let slot = &self.slots[idx];
        let final_json = args::final_arguments(&self.schemas, &slot.name, &slot.args);
        let prev = slot.streamed_json.clone();
        let remaining = if final_json.len() > prev.len() {
            if !prev.is_empty() && !final_json.starts_with(&prev) {
                None
            } else {
                Some(final_json[prev.len()..].to_string())
            }
        } else {
            None
        };
        if remaining.is_some() {
            self.slots[idx].streamed_json = final_json;
        }
        self.slots[idx].open = None;
        if !self.slots[idx].name_sent {
            self.emit_name(idx, out, remaining.unwrap_or_default());
            return;
        }
        if let Some(r) = remaining
            && !r.is_empty()
        {
            out.push(Out::Call(ToolCallDelta {
                tool_index: idx,
                name: None,
                arguments: r,
            }));
        }
    }
}
