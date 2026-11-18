// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! The conversation as the template gets it, each string a `Sym`: client strings (system, user,
//! assistant content and reasoning, tool-call ids, names and arguments, tool results, tool names,
//! descriptions and parameter schemas' names and texts, chat_template_kwargs, documents) are
//! client data; the chat schema's fields (keys, roles, types, JSON-schema keywords and their
//! `type` values, flags templates test) are template text (the Python plugin's `_ClientData`).

use serde_json::Value as Json;

use super::sym::Sym;
use super::value::{Dict, Value};

const SCHEMA_NAME_MAPS: &[&str] = &[
    "properties",
    "patternProperties",
    "$defs",
    "definitions",
    "dependentSchemas",
];
const SCHEMA_SUBSCHEMA: &[&str] = &[
    "items",
    "additionalProperties",
    "not",
    "if",
    "then",
    "else",
    "contains",
    "propertyNames",
];
const SCHEMA_SUBSCHEMA_LISTS: &[&str] = &["anyOf", "oneOf", "allOf", "prefixItems"];
const SCHEMA_KEYWORDS: &[&str] = &[
    "type",
    "description",
    "title",
    "default",
    "examples",
    "enum",
    "const",
    "required",
    "format",
    "pattern",
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "multipleOf",
    "minLength",
    "maxLength",
    "minItems",
    "maxItems",
    "uniqueItems",
    "minProperties",
    "maxProperties",
    "nullable",
    "$ref",
    "$schema",
    "$id",
    "$comment",
    "deprecated",
    "readOnly",
    "writeOnly",
    "contentMediaType",
    "contentEncoding",
    "dependentRequired",
    "unevaluatedItems",
    "unevaluatedProperties",
    "strict",
];

fn is_schema_keyword(k: &str) -> bool {
    SCHEMA_NAME_MAPS.contains(&k)
        || SCHEMA_SUBSCHEMA.contains(&k)
        || SCHEMA_SUBSCHEMA_LISTS.contains(&k)
        || SCHEMA_KEYWORDS.contains(&k)
}

fn data(s: &str) -> Value {
    Value::Str(Sym::data(s))
}

fn text(s: &str) -> Value {
    Value::Str(Sym::template(s))
}

fn scalar(v: &Json) -> Value {
    match v {
        Json::Null => Value::None,
        Json::Bool(b) => Value::Bool(*b),
        Json::Number(n) => match n.as_i64() {
            Some(i) => Value::Int(i),
            None => Value::Float(n.as_f64().unwrap_or(0.0)),
        },
        Json::String(s) => text(s),
        Json::Array(_) | Json::Object(_) => unreachable!("not a scalar"),
    }
}

fn dict_of(pairs: impl IntoIterator<Item = (Value, Value)>) -> Value {
    let mut d = Dict::new();
    for (k, v) in pairs {
        d.insert(k, v).expect("string keys");
    }
    Value::dict(d)
}

/// A chat-schema field: its strings are template text.
pub(crate) fn structure(x: &Json) -> Value {
    match x {
        Json::Object(o) => dict_of(o.iter().map(|(k, v)| (text(k), structure(v)))),
        Json::Array(a) => Value::list(a.iter().map(structure).collect()),
        other => scalar(other),
    }
}

/// Arbitrary JSON data (tool-call arguments, chat_template_kwargs): keys and string leaves are
/// data.
pub(crate) fn value(v: &Json) -> Value {
    match v {
        Json::String(s) => data(s),
        Json::Object(o) => dict_of(o.iter().map(|(k, x)| (data(k), value(x)))),
        Json::Array(a) => Value::list(a.iter().map(value).collect()),
        other => scalar(other),
    }
}

/// A JSON schema: keywords are structure (templates test them), names and free text are data;
/// `type` values stay template text (a closed vocabulary templates compare with).
fn schema(s: &Json) -> Value {
    let Json::Object(o) = s else { return value(s) };
    dict_of(o.iter().map(|(k, v)| {
        let k_str = k.as_str();
        if SCHEMA_NAME_MAPS.contains(&k_str)
            && let Json::Object(names) = v
        {
            return (
                text(k),
                dict_of(names.iter().map(|(n, x)| (data(n), schema(x)))),
            );
        }
        if SCHEMA_SUBSCHEMA.contains(&k_str) && v.is_object() {
            return (text(k), schema(v));
        }
        if SCHEMA_SUBSCHEMA_LISTS.contains(&k_str)
            && let Json::Array(items) = v
        {
            return (text(k), Value::list(items.iter().map(schema).collect()));
        }
        if k_str == "type" {
            return (text(k), structure(v));
        }
        if is_schema_keyword(k_str) {
            return (text(k), value(v));
        }
        (data(k), value(v))
    }))
}

fn content(c: &Json) -> Value {
    if let Some(Json::Array(segments)) = c.get(super::renderer::STRING_CONTENT_KEY) {
        // string-format content with media (renderer.rs): text parts are data, the media
        // placeholders the renderer inserted template text, joined with "\n"
        let mut b = super::sym::SymBuilder::new();
        for (i, seg) in segments.iter().enumerate() {
            if i > 0 {
                b.push(super::sym::Kind::Template, "\n");
            }
            if let Some(t) = seg.get("text").and_then(Json::as_str) {
                b.push(super::sym::Kind::Data, t);
            } else if let Some(m) = seg.get("media").and_then(Json::as_str) {
                b.push(super::sym::Kind::Template, m);
            }
        }
        return Value::Str(b.finish());
    }
    match c {
        Json::String(s) => data(s),
        Json::Array(parts) => Value::list(
            parts
                .iter()
                .map(|p| match p {
                    Json::Object(part) => {
                        let is_ref =
                            part.get("type").and_then(Json::as_str) == Some("tool_reference");
                        let has_output = part.contains_key("output");
                        dict_of(part.iter().map(|(k, v)| {
                            let v = match (k.as_str(), v) {
                                ("text", Json::String(s)) => data(s),
                                // vLLM 0.32's tool_reference parts name a tool: data
                                ("name", Json::String(s)) if is_ref => data(s),
                                // list-of-outputs tool results (GLM-5.3's template)
                                ("output", v) => content(v),
                                ("tool_call_id" | "id", Json::String(s)) if has_output => data(s),
                                (_, v) => structure(v),
                            };
                            (text(k), v)
                        }))
                    }
                    Json::String(s) => data(s),
                    other => structure(other),
                })
                .collect(),
        ),
        other => structure(other),
    }
}

/// One message.
pub(crate) fn message(m: &Json) -> Value {
    let Json::Object(o) = m else {
        return structure(m);
    };
    dict_of(o.iter().map(|(k, v)| {
        let v = match (k.as_str(), v) {
            ("content", v) => content(v),
            ("reasoning" | "reasoning_content" | "name" | "tool_call_id", Json::String(s)) => {
                data(s)
            }
            ("tool_calls", Json::Array(calls)) => {
                Value::list(calls.iter().map(tool_call).collect())
            }
            (_, v) => structure(v),
        };
        (text(k), v)
    }))
}

fn tool_call(tc: &Json) -> Value {
    let Json::Object(o) = tc else {
        return structure(tc);
    };
    dict_of(o.iter().map(|(k, v)| {
        let v = match (k.as_str(), v) {
            ("id", Json::String(s)) => data(s),
            ("function", Json::Object(f)) => dict_of(f.iter().map(|(fk, fv)| {
                let fv = match (fk.as_str(), fv) {
                    ("name", Json::String(s)) => data(s),
                    ("arguments", fv) => value(fv),
                    (_, fv) => structure(fv),
                };
                (text(fk), fv)
            })),
            (_, v) => structure(v),
        };
        (text(k), v)
    }))
}

/// One tool.
pub(crate) fn tool(t: &Json) -> Value {
    let Json::Object(o) = t else {
        return structure(t);
    };
    let f = o.get("function").unwrap_or(t);
    let inner = match f {
        Json::Object(fo) => dict_of(fo.iter().map(|(k, v)| match (k.as_str(), v) {
            ("name" | "description", Json::String(s)) => (text(k), data(s)),
            ("parameters", v) => (text(k), schema(v)),
            // flags templates test (GLM-5.3: defer_loading)
            ("strict" | "type" | "defer_loading", v) => (text(k), structure(v)),
            // unknown function-level keys: some templates render them as tags
            (_, v) => (data(k), value(v)),
        })),
        _ => dict_of(std::iter::empty()),
    };
    if o.contains_key("function") {
        return dict_of(o.iter().map(|(k, v)| {
            (
                text(k),
                if k == "function" {
                    inner.clone()
                } else {
                    structure(v)
                },
            )
        }));
    }
    inner
}

/// One document.
pub(crate) fn document(d: &Json) -> Value {
    let Json::Object(o) = d else { return value(d) };
    dict_of(o.iter().map(|(k, v)| (text(k), value(v))))
}
