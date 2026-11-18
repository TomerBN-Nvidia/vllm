// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! transformers' `tojson` filter (`json.dumps(x, ensure_ascii=False, indent=..., separators=...,
//! sort_keys=...)`) with runs: a string's runs are kept between its quotes, a control token keeps
//! its id when its spelling needs no escaping, everything `json.dumps` itself writes is template
//! text.

use std::fmt::Write as _;

use vllm_parser::python::{escape_str_into, float_repr};

use super::sym::{Kind, Sym, SymBuilder};
use super::value::{VResult, Value, type_error};

fn escape(text: &str, ensure_ascii: bool) -> String {
    let mut out = String::with_capacity(text.len());
    if !ensure_ascii {
        escape_str_into(text, &mut out);
        return out;
    }
    for c in text.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if c.is_ascii() && (c as u32) >= 0x20 => out.push(c),
            c => {
                let mut buf = [0u16; 2];
                for unit in c.encode_utf16(&mut buf) {
                    let _ = write!(out, "\\u{unit:04x}");
                }
            }
        }
    }
    out
}

struct Encoder {
    ensure_ascii: bool,
    indent: Option<String>,
    item_sep: String,
    key_sep: String,
    sort_keys: bool,
    out: SymBuilder,
}

impl Encoder {
    fn t(&mut self, text: &str) {
        self.out.push(Kind::Template, text);
    }

    fn string(&mut self, s: &Sym) {
        self.t("\"");
        for (kind, text) in s.runs() {
            let esc = escape(text, self.ensure_ascii);
            let kind = if esc == text {
                kind
            } else if matches!(kind, Kind::Control(_)) {
                Kind::Template
            } else {
                kind
            };
            self.out.push(kind, &esc);
        }
        self.t("\"");
    }

    fn newline(&mut self, level: usize) {
        if let Some(ind) = &self.indent {
            let s = format!("\n{}", ind.repeat(level));
            self.t(&s);
        }
    }

    fn value(&mut self, v: &Value, level: usize) -> VResult<()> {
        match v {
            Value::Str(s) => self.string(s),
            Value::None => self.t("null"),
            Value::Bool(true) => self.t("true"),
            Value::Bool(false) => self.t("false"),
            Value::Int(i) => self.t(&i.to_string()),
            Value::Float(f) => self.t(&float_repr(*f)),
            Value::List(items) | Value::Tuple(items) => {
                if items.is_empty() {
                    self.t("[]");
                    return Ok(());
                }
                self.t("[");
                self.newline(level + 1);
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        let sep = self.item_sep.clone();
                        self.t(&sep);
                        self.newline(level + 1);
                    }
                    self.value(item, level + 1)?;
                }
                self.newline(level);
                self.t("]");
            }
            Value::Dict(d) => {
                if d.is_empty() {
                    self.t("{}");
                    return Ok(());
                }
                let mut items: Vec<(&Value, &Value)> = d.iter().collect();
                if self.sort_keys {
                    let mut err = None;
                    items.sort_by(|a, b| match a.0.py_lt(b.0) {
                        Ok(true) => std::cmp::Ordering::Less,
                        Ok(false) => {
                            if b.0.py_lt(a.0).unwrap_or(false) {
                                std::cmp::Ordering::Greater
                            } else {
                                std::cmp::Ordering::Equal
                            }
                        }
                        Err(e) => {
                            err.get_or_insert(e);
                            std::cmp::Ordering::Equal
                        }
                    });
                    if let Some(e) = err {
                        return Err(e);
                    }
                }
                self.t("{");
                self.newline(level + 1);
                for (i, (k, v)) in items.into_iter().enumerate() {
                    if i > 0 {
                        let sep = self.item_sep.clone();
                        self.t(&sep);
                        self.newline(level + 1);
                    }
                    match k {
                        Value::Str(s) => self.string(s),
                        Value::Int(i) => self.t(&format!("\"{i}\"")),
                        Value::Float(f) => self.t(&format!("\"{}\"", float_repr(*f))),
                        Value::Bool(b) => self.t(if *b { "\"true\"" } else { "\"false\"" }),
                        Value::None => self.t("\"null\""),
                        other => {
                            return Err(type_error(format!(
                                "keys must be str, int, float, bool or None, not {}",
                                other.type_name()
                            )));
                        }
                    }
                    let sep = self.key_sep.clone();
                    self.t(&sep);
                    self.value(v, level + 1)?;
                }
                self.newline(level);
                self.t("}");
            }
            other => {
                return Err(type_error(format!(
                    "Object of type {} is not JSON serializable",
                    other.type_name()
                )));
            }
        }
        Ok(())
    }
}

/// `x | tojson(indent=..., ensure_ascii=..., separators=..., sort_keys=...)`.
pub(crate) fn tojson(
    x: &Value,
    ensure_ascii: bool,
    indent: Option<&Value>,
    separators: Option<&Value>,
    sort_keys: bool,
) -> VResult<Sym> {
    let indent = match indent {
        None | Some(Value::None) => None,
        Some(Value::Str(s)) => Some(s.as_str().to_string()),
        Some(v) => Some(" ".repeat(v.as_int().unwrap_or(0).max(0) as usize)),
    };
    let (item_sep, key_sep) = match separators {
        None | Some(Value::None) => {
            if indent.is_none() {
                (", ".into(), ": ".into())
            } else {
                (",".into(), ": ".into())
            }
        }
        Some(v) => {
            let parts = v.iterate()?;
            if parts.len() != 2 {
                return Err(type_error("separators must be a pair"));
            }
            (
                parts[0].to_sym().as_str().to_string(),
                parts[1].to_sym().as_str().to_string(),
            )
        }
    };
    let mut e = Encoder {
        ensure_ascii,
        indent,
        item_sep,
        key_sep,
        sort_keys,
        out: SymBuilder::new(),
    };
    e.value(x, 0)?;
    Ok(e.out.finish())
}
