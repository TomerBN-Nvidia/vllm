// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! [T5] / [R8] The control tokens as the tokenizer spells them: today's spelling if the vocabulary
//! has it, else its one namespaced form (`<nemo:think>`; a closer as `</nemo:think>` or
//! `<nemo:/think>`; a turn token as `<|nemo:im_start|>` or `<nemo:im_start>`).
//! `NEMOTRON4_CONTROL_TOKEN_PREFIX` picks a namespace; `NEMOTRON4_CONTROL_TOKENS` (a JSON object,
//! today's spelling to this tokenizer's) names any other spelling. Of several namespaced
//! candidates, the one the chat template writes.

use std::collections::HashMap;

use vllm_tokenizer::Tokenizer;

/// Today's spellings, the names the parser uses for the control tokens.
pub(crate) const THINK: &str = "<think>";
pub(crate) const THINK_END: &str = "</think>";
pub(crate) const TOOL_CALL: &str = "<tool_call>";
pub(crate) const TOOL_CALL_END: &str = "</tool_call>";
pub(crate) const TURN_START: &str = "<|im_start|>";
pub(crate) const TURN_END: &str = "<|im_end|>";
pub(crate) const SEQUENCE_END: &str = "</s>";
pub(crate) const CONTROL_TOKENS: [&str; 7] = [
    THINK,
    THINK_END,
    TOOL_CALL,
    TOOL_CALL_END,
    TURN_START,
    TURN_END,
    SEQUENCE_END,
];

/// One resolved control token.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Control {
    pub(crate) spelling: String,
    pub(crate) id: u32,
}

/// Each control token's spelling and id in one tokenizer, or `None` where it has none.
#[derive(Debug, Clone, Default)]
pub(crate) struct ControlTokens {
    pub(crate) by_name: HashMap<&'static str, Option<Control>>,
}

impl ControlTokens {
    pub(crate) fn get(&self, name: &str) -> Option<&Control> {
        self.by_name.get(name).and_then(Option::as_ref)
    }

    pub(crate) fn id(&self, name: &str) -> Option<u32> {
        self.get(name).map(|c| c.id)
    }

    pub(crate) fn spelling<'a>(&'a self, name: &'a str) -> &'a str {
        self.get(name).map_or(name, |c| c.spelling.as_str())
    }
}

fn is_ns_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || matches!(c, '_' | '.' | '-')
}

/// Whether `candidate` is a namespaced form of `name` (`_namespaced`):
/// `<{slash}{pipe}NS:{core}{pipe}>`, or for a closer `<NS:/{core}>`, or for a pipe token
/// `<NS:{core}>`; NS the prefix if given, else `[A-Za-z0-9_.\-]+:`.
fn is_namespaced_form(name: &str, candidate: &str, prefix: Option<&str>) -> bool {
    let Some(inner) = name.strip_prefix('<').and_then(|s| s.strip_suffix('>')) else {
        return false;
    };
    let (slash, inner) = match inner.strip_prefix('/') {
        Some(rest) => ("/", rest),
        None => ("", inner),
    };
    let (pipe, core, pipe_end) = match inner.strip_prefix('|') {
        Some(rest) => match rest.strip_suffix('|') {
            Some(core) => ("|", core, "|"),
            None => ("|", rest, ""),
        },
        None => match inner.strip_suffix('|') {
            Some(core) => ("", core, "|"),
            None => ("", inner, ""),
        },
    };
    if core.is_empty() || core.contains(['<', '>', '|', '/', ':']) {
        return false;
    }
    let ns_ok = |ns: &str| match prefix {
        Some(p) => ns == p,
        None => ns.len() > 1 && ns.ends_with(':') && ns[..ns.len() - 1].chars().all(is_ns_char),
    };
    let Some(c) = candidate.strip_prefix('<').and_then(|s| s.strip_suffix('>')) else {
        return false;
    };
    // form 1: <{slash}{pipe}NS{core}{pipe_end}>
    let head = format!("{slash}{pipe}");
    let tail = format!("{core}{pipe_end}");
    if let Some(mid) = c.strip_prefix(head.as_str()).and_then(|s| s.strip_suffix(tail.as_str()))
        && ns_ok(mid)
    {
        return true;
    }
    if !slash.is_empty() {
        // form 2: <NS/{core}>
        if let Some(mid) = c.strip_suffix(&format!("/{core}"))
            && ns_ok(mid)
        {
            return true;
        }
    }
    if !pipe.is_empty() {
        // form 3: <NS{core}>
        if let Some(mid) = c.strip_suffix(core)
            && ns_ok(mid)
        {
            return true;
        }
    }
    false
}

/// Resolve the control tokens of `tokenizer` (`_control_tokens`). `template`: the chat template, to
/// pick among several namespaced candidates.
pub(crate) fn resolve(
    tokenizer: &dyn Tokenizer,
    template: Option<&str>,
) -> Result<ControlTokens, String> {
    let named: HashMap<String, String> = match std::env::var("NEMOTRON4_CONTROL_TOKENS") {
        Ok(raw) if !raw.is_empty() => serde_json::from_str(&raw).map_err(|_| {
            format!(
                "NEMOTRON4_CONTROL_TOKENS must be a JSON object of strings, e.g. \
                 '{{\"<think>\": \"<x_think>\"}}', not {raw:?}"
            )
        })?,
        _ => HashMap::new(),
    };
    let prefix = std::env::var("NEMOTRON4_CONTROL_TOKEN_PREFIX").ok().filter(|p| !p.is_empty());
    let template = template.unwrap_or("");
    let added: Vec<&(String, u32)> = tokenizer.added_vocab().iter().collect();
    let marker = prefix.clone().unwrap_or_else(|| ":".into());
    let candidates: Vec<&(String, u32)> = added
        .iter()
        .copied()
        .filter(|(t, _)| t.starts_with('<') && t.ends_with('>') && t.contains(marker.as_str()))
        .collect();
    let mut by_name = HashMap::new();
    for name in CONTROL_TOKENS {
        if let Some(spelling) = named.get(name) {
            by_name.insert(
                name,
                tokenizer.token_to_id(spelling).map(|id| Control {
                    spelling: spelling.clone(),
                    id,
                }),
            );
            continue;
        }
        let mut found: Vec<&(String, u32)> = candidates
            .iter()
            .copied()
            .filter(|(t, _)| is_namespaced_form(name, t, prefix.as_deref()))
            .collect();
        found.sort();
        if found.len() > 1 {
            found.retain(|(t, _)| template.contains(t.as_str()));
        }
        let here = tokenizer.token_to_id(name);
        let pick = if found.len() == 1
            && (prefix.is_some()
                || here.is_none()
                || (template.contains(found[0].0.as_str()) && !template.contains(name)))
        {
            Some(Control {
                spelling: found[0].0.clone(),
                id: found[0].1,
            })
        } else {
            here.map(|id| Control {
                spelling: name.to_string(),
                id,
            })
        };
        by_name.insert(name, pick);
    }
    Ok(ControlTokens { by_name })
}

#[cfg(test)]
mod tests {
    use super::is_namespaced_form;

    #[test]
    fn namespaced_forms() {
        assert!(is_namespaced_form("<think>", "<nemo:think>", None));
        assert!(is_namespaced_form("</think>", "</nemo:think>", None));
        assert!(is_namespaced_form("</think>", "<nemo:/think>", None));
        assert!(is_namespaced_form(
            "<|im_start|>",
            "<|nemo:im_start|>",
            None
        ));
        assert!(is_namespaced_form("<|im_start|>", "<nemo:im_start>", None));
        assert!(is_namespaced_form("</s>", "</nemo:s>", None));
        assert!(!is_namespaced_form("<think>", "<nemo:thinkx>", None));
        assert!(!is_namespaced_form("<think>", "<x:think>", Some("nemo:")));
    }
}
