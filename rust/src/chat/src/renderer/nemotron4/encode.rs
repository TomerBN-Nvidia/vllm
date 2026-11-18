// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! The render to token ids: the template's text (and its control tokens) is read as the stock
//! tokenizer reads it, added-token spellings being those tokens; client data is only ever text,
//! and a spelling that client data completes stays text; text of unknown origin that spells a
//! control token is an error. Text is encoded with the added vocabulary disabled
//! (`encode_ordinary`).

use vllm_tokenizer::Tokenizer;

use super::ir::Literals;
use super::sym::{Kind, Sym, SymError};

enum Item {
    Text(String),
    Control(u32),
}

/// The render's token ids.
pub(crate) fn encode(
    rendered: &Sym,
    literals: &Literals,
    tokenizer: &dyn Tokenizer,
) -> Result<Vec<u32>, SymError> {
    let mut plan: Vec<Item> = Vec::new();
    let mut cur = String::new();
    let runs: Vec<(Kind, &str)> = rendered.runs().collect();
    let mut i = 0;
    while i < runs.len() {
        let (kind, text) = runs[i];
        if kind == Kind::Data {
            cur.push_str(text);
            i += 1;
            continue;
        }
        // a span of the template's text: control tokens, template text and text of unknown origin
        let mut j = i;
        let mut span = String::new();
        let mut unknown: Vec<(usize, usize)> = Vec::new();
        while j < runs.len() && runs[j].0 != Kind::Data {
            let (k, t) = runs[j];
            if k == Kind::Unknown {
                unknown.push((span.len(), span.len() + t.len()));
            }
            span.push_str(t);
            j += 1;
        }
        let mut pos = 0;
        for (a, b, id) in literals.find_all(&span) {
            if unknown.iter().any(|&(ua, ub)| ua < b && a < ub) {
                let lo = span[..a].char_indices().rev().nth(29).map_or(0, |(x, _)| x);
                let hi = (b..=span.len())
                    .filter(|x| span.is_char_boundary(*x))
                    .nth(30)
                    .unwrap_or(span.len());
                return Err(SymError::Unknown(format!(
                    "nemotron4 structural rendering: text of unknown origin spells the control \
                     token {:?} (in {:?}); the chat template uses an operation the encoder \
                     doesn't model",
                    &span[a..b],
                    &span[lo..hi]
                )));
            }
            cur.push_str(&span[pos..a]);
            plan.push(Item::Text(std::mem::take(&mut cur)));
            plan.push(Item::Control(id));
            pos = b;
        }
        cur.push_str(&span[pos..]);
        i = j;
    }
    plan.push(Item::Text(cur));
    let texts: Vec<&str> = plan
        .iter()
        .filter_map(|item| match item {
            Item::Text(t) if !t.is_empty() => Some(t.as_str()),
            _ => None,
        })
        .collect();
    let encode = |t: &&str| {
        tokenizer
            .encode_ordinary(t)
            .map_err(|e| SymError::Python(format!("encoding failed: {e}")))
    };
    // long prompts: the segments encoded in parallel (each on its own either way, so the ids are
    // the same), as the Python plugin's batch encoding
    let total: usize = texts.iter().map(|t| t.len()).sum();
    let encoded: Vec<Vec<u32>> = if total >= BATCH_MIN_BYTES && texts.len() > 1 {
        use rayon::prelude::*;
        texts.par_iter().map(encode).collect::<Result<_, _>>()?
    } else {
        texts.iter().map(encode).collect::<Result<_, _>>()?
    };
    let mut encoded = encoded.into_iter();
    let mut ids = Vec::new();
    for item in plan {
        match item {
            Item::Control(id) => ids.push(id),
            Item::Text(t) if t.is_empty() => {}
            Item::Text(_) => ids.extend(encoded.next().expect("one per text")),
        }
    }
    Ok(ids)
}

/// Texts of at least this many bytes are encoded as a parallel batch of their segments.
const BATCH_MIN_BYTES: usize = 16384;
