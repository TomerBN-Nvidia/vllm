// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! The reasoning split ([R1]-[R5]): the Python nemotron4 reasoning parser's `_Split`, one state
//! machine over pieces (a control token, or the text between them, with the generated tokens'
//! attributions).

use vllm_tokenizer::{DecodedText, TokenAnchor, TokenAttribution};

use crate::python::{py_lstrip, py_strip};

/// The control tokens the split tells apart (by id).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Marker {
    Think,
    End,
    Call,
    Close,
}

/// A control token, or text between them.
#[derive(Debug, Clone)]
pub(crate) struct Piece {
    pub(crate) kind: Option<Marker>,
    pub(crate) dt: DecodedText,
}

impl Piece {
    fn text(&self) -> &str {
        &self.dt.text
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Run {
    Open,
    Body,
    After,
}

const FUNCTION: &str = "<function=";

/// `_BLOCK_DONE = </function>\s*\Z`.
fn block_done(body: &str) -> bool {
    body.trim_end_matches(char::is_whitespace).ends_with("</function>")
}

/// What one feed returns: reasoning pieces, and content pieces (past the transition).
#[derive(Debug, Default)]
pub(crate) struct Fed {
    pub(crate) reasoning: Vec<DecodedText>,
    pub(crate) content: Vec<Piece>,
}

#[derive(Debug, Clone)]
pub(crate) struct Split {
    content: bool,
    force: bool,
    calls: bool,
    fresh: bool,
    strip_nl: bool,
    pending: Vec<DecodedText>,
    closed: bool,
    trail: Vec<Piece>,
    run: Option<Run>,
    run_held: Vec<Piece>,
    lead: String,
    body: String,
}

impl Split {
    pub(crate) fn new(reasoning: bool, force: bool, calls: bool) -> Self {
        Self {
            content: !reasoning,
            force,
            calls,
            fresh: true,
            strip_nl: false,
            pending: Vec::new(),
            closed: false,
            trail: Vec::new(),
            run: None,
            run_held: Vec::new(),
            lead: String::new(),
            body: String::new(),
        }
    }

    pub(crate) fn feed(&mut self, pieces: Vec<Piece>, last: bool) -> Fed {
        let mut out = Fed::default();
        for p in pieces {
            if self.content {
                out.content.push(p);
                continue;
            }
            self.piece(p, &mut out);
        }
        if last && !self.content {
            self.finish(&mut out);
        }
        out
    }

    fn reason(&mut self, dt: DecodedText, out: &mut Fed) {
        if self.force {
            self.pending.push(dt);
        } else {
            out.reasoning.push(dt);
        }
    }

    fn start_content(&mut self, out: &mut Fed, held: Vec<Piece>) {
        out.reasoning.append(&mut self.pending);
        self.content = true;
        out.content.append(&mut self.trail);
        out.content.extend(held);
        self.closed = false;
    }

    fn piece(&mut self, mut p: Piece, out: &mut Fed) {
        if p.text().is_empty() && p.kind.is_none() {
            // ids without text: they go where the text around them goes
            if self.run.is_some() {
                self.run_held.push(p);
            } else if self.closed {
                self.trail.push(p);
            } else {
                self.reason(p.dt, out);
            }
            return;
        }
        if self.fresh {
            self.fresh = false;
            if p.kind == Some(Marker::Think) {
                self.strip_nl = true; // [R4] the output opens with <think>: dropped
                return;
            }
        }
        if self.strip_nl {
            self.strip_nl = false;
            if p.kind.is_none() && p.text().starts_with('\n') {
                p.dt = drop_first_byte(p.dt);
                if p.text().is_empty() {
                    self.reason(p.dt, out); // the newline alone: its id is still reasoning's
                    return;
                }
            }
        }
        if self.closed {
            // [R5] after </think>: content only once something but whitespace comes
            if p.kind.is_none() && py_strip(p.text()).is_empty() {
                self.trail.push(p);
                return;
            }
            self.start_content(out, vec![p]);
            return;
        }
        if self.run.is_some() {
            // [R3]
            if self.run_holds(p.kind, p.text()) {
                self.run_held.push(p);
                return;
            }
            let held = std::mem::take(&mut self.run_held);
            for h in held {
                self.reason(h.dt, out); // a draft or a mention: reasoning after all
            }
            self.run = None;
        }
        if p.kind == Some(Marker::End) {
            if self.force {
                self.closed = true;
            } else {
                self.content = true; // the </think> itself is dropped
            }
            return;
        }
        if p.kind == Some(Marker::Call) && self.calls {
            self.run = Some(Run::Open);
            self.run_held = vec![p];
            self.lead.clear();
            self.body.clear();
            return;
        }
        self.reason(p.dt, out);
    }

    fn run_holds(&mut self, kind: Option<Marker>, text: &str) -> bool {
        match self.run.expect("in a run") {
            Run::Open => {
                if kind.is_some() {
                    return false; // a control token right after <tool_call>
                }
                self.lead.push_str(text);
                let lead = py_lstrip(&self.lead);
                if lead.starts_with(FUNCTION) {
                    self.body = lead.to_string();
                    self.run = Some(Run::Body);
                    return true;
                }
                FUNCTION.starts_with(lead)
            }
            Run::Body => {
                if kind == Some(Marker::Close) {
                    self.run = Some(Run::After);
                    return true;
                }
                if kind == Some(Marker::End) {
                    return false;
                }
                if kind == Some(Marker::Call) && block_done(&self.body) {
                    self.run = Some(Run::Open); // the block ended without its </tool_call>
                    self.lead.clear();
                    return true;
                }
                self.body.push_str(text);
                true
            }
            Run::After => {
                if kind.is_none() && py_strip(text).is_empty() {
                    return true;
                }
                if kind == Some(Marker::Call) {
                    self.run = Some(Run::Open);
                    self.lead.clear();
                    return true;
                }
                false
            }
        }
    }

    fn finish(&mut self, out: &mut Fed) {
        if let Some(run) = self.run {
            let complete = run == Run::After || (run == Run::Body && block_done(&self.body));
            let mut held = std::mem::take(&mut self.run_held);
            self.run = None;
            if complete {
                // [R3] complete calls finish the output: the reasoning ended at the run's
                // <tool_call>
                if self.force {
                    // [R5] and it never ended otherwise: all of it is content
                    let mut all: Vec<Piece> = std::mem::take(&mut self.pending)
                        .into_iter()
                        .map(|dt| Piece { kind: None, dt })
                        .collect();
                    all.append(&mut held);
                    held = all;
                }
                self.start_content(out, held);
                return;
            }
            for h in held {
                self.reason(h.dt, out);
            }
        }
        if self.force {
            // [R5] the reasoning never ended, or only whitespace followed: it is the content
            let held: Vec<Piece> = std::mem::take(&mut self.pending)
                .into_iter()
                .map(|dt| Piece { kind: None, dt })
                .collect();
            self.trail.clear();
            self.closed = false;
            self.start_content(out, held);
        }
    }
}

/// The piece without its first byte (a newline), its tokens kept (the newline's own as zero-width).
fn drop_first_byte(dt: DecodedText) -> DecodedText {
    let text = dt.text[1..].to_string();
    let attributions = dt
        .attributions
        .iter()
        .map(|a| {
            let anchor = match a.anchor {
                TokenAnchor::Visible { byte_offset: 0 } => {
                    TokenAnchor::ZeroWidth { byte_offset: 0 }
                }
                TokenAnchor::Visible { byte_offset } => TokenAnchor::Visible {
                    byte_offset: byte_offset - 1,
                },
                TokenAnchor::ZeroWidth { byte_offset } => TokenAnchor::ZeroWidth {
                    byte_offset: byte_offset.saturating_sub(1),
                },
            };
            TokenAttribution {
                token_id: a.token_id,
                anchor,
            }
        })
        .collect();
    DecodedText { text, attributions }
}
