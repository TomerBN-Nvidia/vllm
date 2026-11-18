// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Tool-call arguments: the raw `<parameter=NAME>VALUE</parameter>` text of a call to JSON ([F1]
//! [F2] [F3]), its values coerced by the tool's schema ([F8] [F9] [F11]), and streamed as a growing
//! prefix of the final JSON ([F10]).
//!
//! The Python plugin's functions of the same names are the reference; the patterns below are
//! theirs, with `\s` as the `regex` module's (Unicode White_Space).

use std::collections::HashSet;

use serde_json::{Map, Value};

use crate::python::{self, PyVal, py_strip};
use crate::tool::Tool;

fn ws(c: char) -> bool {
    c.is_whitespace()
}

/// Skip `\s*` from byte `i`.
fn skip_ws(t: &str, mut i: usize) -> usize {
    while let Some(c) = t[i..].chars().next() {
        if !ws(c) {
            break;
        }
        i += c.len_utf8();
    }
    i
}

/// `<\s*parameter\s*=` at byte `i`: the end of the match.
fn match_open_start(t: &str, i: usize) -> Option<usize> {
    if !t[i..].starts_with('<') {
        return None;
    }
    let j = skip_ws(t, i + 1);
    let j = t[j..].starts_with("parameter").then_some(j + 9)?;
    let j = skip_ws(t, j);
    t[j..].starts_with('=').then_some(j + 1)
}

/// `<\s*parameter\s*=\s*([^>]*)>` at byte `i`: (the name's span, the end of the match).
fn match_open(t: &str, i: usize) -> Option<((usize, usize), usize)> {
    let j = match_open_start(t, i)?;
    let j = skip_ws(t, j);
    let close = t[j..].find('>')? + j;
    Some(((j, close), close + 1))
}

/// `<\s*/\s*parameter\s*>` at byte `i`: the end of the match.
fn match_close(t: &str, i: usize) -> Option<usize> {
    if !t[i..].starts_with('<') {
        return None;
    }
    let j = skip_ws(t, i + 1);
    let j = t[j..].starts_with('/').then_some(j + 1)?;
    let j = skip_ws(t, j);
    let j = t[j..].starts_with("parameter").then_some(j + 9)?;
    let j = skip_ws(t, j);
    t[j..].starts_with('>').then_some(j + 1)
}

/// The start of every non-overlapping match of `pat` (patterns that begin with `<`).
fn finditer(t: &str, pat: fn(&str, usize) -> Option<usize>) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut i = 0;
    while let Some(k) = t[i..].find('<') {
        let start = i + k;
        match pat(t, start) {
            Some(end) => {
                out.push((start, end));
                i = end;
            }
            None => i = start + 1,
        }
    }
    out
}

/// `_VALUE_END.fullmatch(tail, partial=True)`: `tail` is a closer or an opener, or the start of
/// one.
fn is_arriving_tag(tail: &str) -> bool {
    // <\s*/\s*parameter\s*> | <\s*parameter\s*=
    let mut it = tail.char_indices().peekable();
    let Some((_, '<')) = it.next() else {
        return false;
    };
    let rest = &tail[1..];
    let after_lt = skip_ws(rest, 0);
    if after_lt == rest.len() {
        return true;
    }
    let r = &rest[after_lt..];
    let partial_word = |r: &str, word: &str| -> Option<usize> {
        // Some(consumed) if r starts with word, None if r is a proper prefix of word (end of
        // input), else fail
        if r.starts_with(word) {
            Some(word.len())
        } else if word.starts_with(r) {
            None
        } else {
            Some(usize::MAX)
        }
    };
    // the closer branch
    if let Some(r2) = r.strip_prefix('/') {
        let j = skip_ws(r2, 0);
        let r3 = &r2[j..];
        return match partial_word(r3, "parameter") {
            None => true,
            Some(usize::MAX) => false,
            Some(n) => {
                let r4 = &r3[n..];
                let k = skip_ws(r4, 0);
                let r5 = &r4[k..];
                r5.is_empty() || r5 == ">"
            }
        };
    }
    // the opener branch
    match partial_word(r, "parameter") {
        None => true,
        Some(usize::MAX) => false,
        Some(n) => {
            let r4 = &r[n..];
            let k = skip_ws(r4, 0);
            let r5 = &r4[k..];
            r5.is_empty() || r5 == "="
        }
    }
}

/// `_arriving_tag(value)`: the suffix from the last `<` when it may still become a closer or an
/// opener.
pub(crate) fn arriving_tag(value: &str) -> &str {
    match value.rfind('<') {
        Some(start) if is_arriving_tag(&value[start..]) => &value[start..],
        _ => "",
    }
}

/// `_trim_wrapping_newlines`: one leading and one trailing newline are the template's markup.
fn trim_wrapping_newlines(v: &str) -> &str {
    let v = v.strip_prefix('\n').unwrap_or(v);
    v.strip_suffix('\n').unwrap_or(v)
}

/// `_opens_parameter(text, pos)` [F3]: an opener counts at the start of the call, at the start of a
/// line, or right after a `</parameter>` (whitespace allowed).
fn opens_parameter(text: &str, pos: usize) -> bool {
    let mut j = pos;
    while j > 0 && matches!(text.as_bytes()[j - 1], b' ' | b'\t' | b'\r' | b'\n') {
        j -= 1;
    }
    if j == 0 || text[j..pos].contains('\n') {
        return true;
    }
    // _CLOSE_AT_END.search(text, max(0, j - 64), j): a closer that ends at j and starts at most 64
    // chars before it
    let window_start = text[..j].char_indices().rev().nth(63).map_or(0, |(k, _)| k);
    let window = &text[window_start..j];
    finditer(window, match_close)
        .last()
        .is_some_and(|&(_, end)| end == window.len())
}

/// `_parameters(raw)`: (name, its raw region, whether its opener is the last) per parameter, first
/// occurrence only.
pub(crate) fn parameters(raw: &str) -> Vec<(String, (usize, usize), bool)> {
    let starts: Vec<usize> = finditer(raw, match_open_start)
        .into_iter()
        .map(|(s, _)| s)
        .filter(|&s| opens_parameter(raw, s))
        .collect();
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for (k, &start) in starts.iter().enumerate() {
        let end = starts.get(k + 1).copied().unwrap_or(raw.len());
        let Some(((ns, ne), oend)) = match_open(raw, start) else {
            continue;
        };
        if oend > end {
            continue; // its name hasn't arrived yet
        }
        let name = py_strip(&raw[ns..ne]).to_string();
        if !seen.insert(name.clone()) {
            continue; // [F2] a parameter given twice keeps its first value
        }
        out.push((name, (oend, end), k + 1 == starts.len()));
    }
    out
}

/// `_nemotron4_arg_converter(raw, partial)`, as the dict it dumps.
pub(crate) fn convert(raw: &str, partial: bool) -> Vec<(String, PyVal)> {
    let mut params = Vec::new();
    for (name, (a, b), last) in parameters(raw) {
        let region = &raw[a..b];
        let closes = finditer(region, match_close);
        let value = if let Some(&(start, _)) = closes.last() {
            &region[..start] // [F3] the last </parameter> before the next parameter
        } else if partial && last {
            &region[..region.len() - arriving_tag(region).len()]
        } else {
            // [F1] no </parameter>: the value runs to the next parameter, </function> or the end
            region
        };
        PyVal::dict_insert(
            &mut params,
            name,
            PyVal::Str(trim_wrapping_newlines(value).to_string()),
        );
    }
    params
}

/// [F10] `_OpenString`: the open last value of a call's arguments, a string, streamed by extension.
#[derive(Debug, Clone)]
pub(crate) struct OpenString {
    pending: String,
    lead: bool,
}

impl OpenString {
    /// The state after `region` (the open value's raw text so far), or `None` if a `</parameter>`
    /// may have closed it.
    pub(crate) fn start(region: &str) -> Option<Self> {
        if !finditer(region, match_close).is_empty() {
            return None;
        }
        let mut s = Self {
            pending: String::new(),
            lead: true,
        };
        s.advance(region, false);
        Some(s)
    }

    /// The JSON text to append for `delta`, or `None` when the delta may do more than extend the
    /// value.
    pub(crate) fn extend(&mut self, delta: &str) -> Option<String> {
        self.advance(delta, true)
    }

    fn advance(&mut self, delta: &str, check: bool) -> Option<String> {
        let mut text = format!("{}{}", self.pending, delta);
        if self.lead {
            if text.is_empty() {
                return Some(String::new());
            }
            self.lead = false;
            if let Some(rest) = text.strip_prefix('\n') {
                text = rest.to_string();
            }
        }
        if check
            && text.contains('<')
            && (!finditer(&text, match_close).is_empty()
                || !finditer(&text, match_open_start).is_empty())
        {
            return None;
        }
        let window = if text.contains('<') {
            arriving_tag(&text).to_string()
        } else {
            String::new()
        };
        let mut emit = &text[..text.len() - window.len()];
        let newline = emit.ends_with('\n');
        if newline {
            emit = &emit[..emit.len() - 1];
        }
        let out = if emit.is_empty() {
            String::new()
        } else {
            let mut s = String::new();
            python::escape_str_into(emit, &mut s);
            s
        };
        self.pending = format!("{}{}", if newline { "\n" } else { "" }, window);
        Some(out)
    }

    /// The JSON text the converter streams for this value so far (after its opening quote).
    pub(crate) fn streamed(&self, region: &str) -> String {
        let text = region.strip_prefix('\n').unwrap_or(region);
        let mut s = String::new();
        python::escape_str_into(&text[..text.len() - self.pending.len()], &mut s);
        s
    }
}

/// `ParserEngine._safe_arg_prefix(json_str, string_keys)`: the prefix of the JSON up to its last
/// top-level value, with a trailing string value of a streamable key included as far as it goes.
pub(crate) fn safe_arg_prefix(json: &str, string_keys: Option<&HashSet<String>>) -> String {
    let chars: Vec<(usize, char)> = json.char_indices().collect();
    let mut last_colon: Option<usize> = None; // char index
    let mut last_key: Option<String> = None;
    let mut pending_key: Option<String> = None;
    let (mut in_string, mut escape) = (false, false);
    let mut string_start: Option<usize> = None;
    let mut depth = 0i64;
    for (ci, &(bi, c)) in chars.iter().enumerate() {
        if escape {
            escape = false;
            continue;
        }
        if in_string {
            if c == '\\' {
                escape = true;
            } else if c == '"' {
                in_string = false;
                if depth == 1
                    && let Some(s) = string_start
                {
                    pending_key = Some(json[chars[s].0 + 1..bi].to_string());
                }
            }
            continue;
        }
        match c {
            '"' => {
                in_string = true;
                string_start = Some(ci);
            }
            '{' | '[' => depth += 1,
            '}' | ']' => depth -= 1,
            ':' if depth == 1 => {
                last_colon = Some(ci);
                last_key = pending_key.take();
            }
            _ => {}
        }
    }
    let Some(colon) = last_colon else {
        return String::new();
    };
    let mut end = colon + 1;
    while end < chars.len() && matches!(chars[end].1, ' ' | '\t' | '\n' | '\r') {
        end += 1;
    }
    let byte = |ci: usize| chars.get(ci).map_or(json.len(), |&(b, _)| b);
    if end >= chars.len() || chars[end].1 != '"' {
        return json[..byte(end)].to_string();
    }
    if let Some(keys) = string_keys
        && !last_key.as_ref().is_some_and(|k| keys.contains(k))
    {
        return json[..byte(end)].to_string();
    }
    let mut escape = false;
    for ci in end + 1..chars.len() {
        let c = chars[ci].1;
        if escape {
            escape = false;
            continue;
        }
        if c == '\\' {
            escape = true;
            continue;
        }
        if c == '"' {
            return json[..byte(ci)].to_string();
        }
    }
    json.to_string()
}

// ── Schemas ──────────────────────────────────────────────────────────────────────────────────────

const TYPE_ALIASES: &[(&str, &str)] = &[
    ("str", "string"),
    ("text", "string"),
    ("varchar", "string"),
    ("char", "string"),
    ("enum", "string"),
    ("int", "integer"),
    ("int32", "integer"),
    ("int64", "integer"),
    ("uint", "integer"),
    ("uint32", "integer"),
    ("uint64", "integer"),
    ("long", "integer"),
    ("short", "integer"),
    ("unsigned", "integer"),
    ("float", "number"),
    ("float32", "number"),
    ("float64", "number"),
    ("double", "number"),
    ("bool", "boolean"),
    ("dict", "object"),
    ("arr", "array"),
    ("list", "array"),
    ("sequence", "array"),
];

fn normalize_type(t: &str) -> String {
    let key = py_strip(t).to_lowercase();
    TYPE_ALIASES
        .iter()
        .find(|(k, _)| *k == key)
        .map_or(key, |(_, v)| (*v).to_string())
}

/// vLLM's `extract_types_from_schema`: every type a schema allows (`type`, `enum` values, `anyOf` /
/// `oneOf` / `allOf` recursively), or `string`.
pub(crate) fn extract_types(schema: &Value) -> Vec<String> {
    let Some(obj) = schema.as_object() else {
        return vec!["string".into()];
    };
    let mut types: Vec<String> = Vec::new();
    let add = |t: &str, types: &mut Vec<String>| {
        if !types.iter().any(|x| x == t) {
            types.push(t.to_string());
        }
    };
    match obj.get("type") {
        Some(Value::String(t)) => add(t, &mut types),
        Some(Value::Array(ts)) => {
            for t in ts {
                if let Value::String(t) = t {
                    add(t, &mut types);
                }
            }
        }
        _ => {}
    }
    if let Some(Value::Array(values)) = obj.get("enum")
        && !values.is_empty()
    {
        for v in values {
            let t = match v {
                Value::Null => "null",
                Value::Bool(_) => "boolean",
                Value::Number(n) if n.is_i64() || n.is_u64() => "integer",
                Value::Number(_) => "number",
                Value::String(_) => "string",
                Value::Array(_) => "array",
                Value::Object(_) => "object",
            };
            add(t, &mut types);
        }
    }
    for comb in ["anyOf", "oneOf", "allOf"] {
        if let Some(Value::Array(choices)) = obj.get(comb) {
            for c in choices {
                for t in extract_types(c) {
                    add(&t, &mut types);
                }
            }
        }
    }
    if types.is_empty() {
        vec!["string".into()]
    } else {
        types
    }
}

/// The result of `coerce_to_schema_type`: the very string it was given (`Same`), or a new value.
enum Coerced {
    Same,
    New(PyVal),
}

/// vLLM's `coerce_to_schema_type(value, types)`.
fn coerce_to_schema_type(value: &str, types: &[String]) -> Coerced {
    let norm: HashSet<String> = types.iter().map(|t| normalize_type(t)).collect();
    for candidate in [
        "null", "integer", "number", "boolean", "object", "array", "string",
    ] {
        if !norm.contains(candidate) {
            continue;
        }
        match candidate {
            "null" => {
                if value.to_lowercase() == "null" {
                    return Coerced::New(PyVal::Null);
                }
            }
            "string" => return Coerced::Same,
            "integer" => {
                if let Some(d) = python::py_int(value) {
                    return Coerced::New(PyVal::Int(d));
                }
            }
            "number" => {
                if let Some(f) = python::py_float(value)
                    && f.is_finite()
                {
                    return Coerced::New(if f.fract() != 0.0 {
                        PyVal::Float(f)
                    } else {
                        PyVal::Int(python::float_to_int_digits(f))
                    });
                }
            }
            "boolean" => {
                let lower = value.to_lowercase();
                match py_strip(&lower) {
                    "true" | "1" => return Coerced::New(PyVal::Bool(true)),
                    "false" | "0" => return Coerced::New(PyVal::Bool(false)),
                    _ => {}
                }
            }
            _ => {
                if let Some(v) = python::loads_finite(value) {
                    return Coerced::New(v);
                }
            }
        }
    }
    match python::loads(value) {
        Some(v)
            if !matches!(v, PyVal::Float(f) if !f.is_finite())
                && python::loads_finite(value).is_some() =>
        {
            Coerced::New(v)
        }
        _ => Coerced::Same,
    }
}

/// The nemotron4 plugin's `_coerce(value, schema)`: stock's coercion, then [F8] and [F9]. `None`:
/// unchanged.
fn coerce(value: &str, schema: &Value) -> Option<PyVal> {
    let types = extract_types(schema);
    let mut out = coerce_to_schema_type(value, &types);
    let norm: HashSet<String> = types.iter().map(|t| normalize_type(t)).collect();
    if matches!(out, Coerced::Same)
        && !norm.contains("string")
        && (norm.contains("object") || norm.contains("array"))
        && let Some(lit) = python::python_literal(value)
    {
        let fits = matches!(lit, PyVal::Dict(_)) && norm.contains("object")
            || matches!(lit, PyVal::List(_)) && norm.contains("array");
        if fits {
            out = Coerced::New(lit);
        }
    }
    if let Coerced::New(PyVal::Float(f)) = out
        && norm.contains("integer")
        && !norm.contains("number")
        && f.fract() == 0.0
    {
        out = Coerced::New(PyVal::Int(python::float_to_int_digits(f)));
    }
    match out {
        Coerced::Same => None,
        Coerced::New(v) => Some(v),
    }
}

/// [F11] `_typed_schema` for each tool, as the parser reads them: local `$ref`s resolved, the
/// properties a schema's `allOf` / `anyOf` / `oneOf` branches declare merged into its own.
#[derive(Debug, Clone, Default)]
pub(crate) struct ToolSchemas {
    tools: Vec<(String, Value)>,
}

const MAX_SCHEMA_DEPTH: usize = 16;

fn resolve_ref(schema: &Map<String, Value>, root: &Value) -> Map<String, Value> {
    let Some(Value::String(r)) = schema.get("$ref") else {
        return schema.clone();
    };
    let Some(path) = r.strip_prefix("#/") else {
        return schema.clone();
    };
    let mut target = root;
    for part in path.split('/') {
        let part = part.replace("~1", "/").replace("~0", "~");
        match target.as_object().and_then(|o| o.get(&part)) {
            Some(t) => target = t,
            None => return schema.clone(),
        }
    }
    let Some(target) = target.as_object() else {
        return schema.clone();
    };
    let mut out = target.clone();
    for (k, v) in schema {
        if k != "$ref" {
            out.insert(k.clone(), v.clone());
        }
    }
    out
}

fn typed_schema(schema: &Value, root: &Value, depth: usize) -> Value {
    let Some(obj) = schema.as_object() else {
        return schema.clone();
    };
    if depth > MAX_SCHEMA_DEPTH {
        return schema.clone();
    }
    let obj = resolve_ref(obj, root);
    let mut declared: Vec<(String, Vec<Value>)> = Vec::new();
    let mut prop_sets: Vec<Value> = Vec::new();
    if let Some(p) = obj.get("properties") {
        prop_sets.push(p.clone());
    }
    for comb in ["allOf", "anyOf", "oneOf"] {
        if let Some(Value::Array(branches)) = obj.get(comb) {
            for b in branches {
                let tb = typed_schema(b, root, depth + 1);
                if let Some(o) = tb.as_object() {
                    prop_sets.push(o.get("properties").cloned().unwrap_or(Value::Null));
                }
            }
        }
    }
    for props in prop_sets {
        let Value::Object(props) = props else {
            continue;
        };
        for (name, sub) in props {
            let sub = typed_schema(&sub, root, depth + 1);
            let slot = match declared.iter_mut().position(|(n, _)| *n == name) {
                Some(i) => &mut declared[i].1,
                None => {
                    declared.push((name.clone(), Vec::new()));
                    &mut declared.last_mut().unwrap().1
                }
            };
            if !slot.contains(&sub) {
                slot.push(sub);
            }
        }
    }
    let mut out = obj.clone();
    if !declared.is_empty() {
        let mut props = Map::new();
        for (name, mut subs) in declared {
            let v = if subs.len() == 1 {
                subs.pop().unwrap()
            } else {
                let mut m = Map::new();
                m.insert("anyOf".into(), Value::Array(subs));
                Value::Object(m)
            };
            props.insert(name, v);
        }
        out.insert("properties".into(), Value::Object(props));
    }
    if let Some(items @ Value::Object(_)) = obj.get("items") {
        out.insert("items".into(), typed_schema(items, root, depth + 1));
    }
    Value::Object(out)
}

fn needs_typing(params: &Value) -> bool {
    let dumped = serde_json::to_string(params).unwrap_or_default();
    ["\"allOf\"", "\"anyOf\"", "\"oneOf\"", "\"$ref\""]
        .iter()
        .any(|k| dumped.contains(k))
}

impl ToolSchemas {
    /// The request's tools; `schema_types` is [F11] (`NEMOTRON4_SCHEMA_TYPES`, on by default).
    pub(crate) fn new(tools: &[Tool], schema_types: bool) -> Self {
        let tools = tools
            .iter()
            .map(|t| {
                let params =
                    if schema_types && t.parameters.is_object() && needs_typing(&t.parameters) {
                        typed_schema(&t.parameters, &t.parameters, 0)
                    } else {
                        t.parameters.clone()
                    };
                (t.name.clone(), params)
            })
            .collect();
        Self { tools }
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    pub(crate) fn declares(&self, name: &str) -> bool {
        self.tools.iter().any(|(n, _)| n == name)
    }

    /// vLLM's `find_tool_properties`: the first tool of that name's `properties`, or none.
    fn properties(&self, name: &str) -> Option<&Map<String, Value>> {
        let (_, params) = self.tools.iter().find(|(n, _)| n == name)?;
        params.get("properties")?.as_object().filter(|p| !p.is_empty())
    }

    /// `_streamable_string_keys`: `None` without a schema; else the keys whose values can only be
    /// strings.
    pub(crate) fn string_keys(&self, name: &str) -> Option<HashSet<String>> {
        let props = self.properties(name)?;
        Some(
            props
                .iter()
                .filter(|(_, s)| {
                    let t = extract_types(s);
                    t.len() == 1 && t[0] == "string"
                })
                .map(|(k, _)| k.clone())
                .collect(),
        )
    }

    /// `_fix_arg_types` on the converter's dict: each string value coerced by its property's
    /// schema.
    pub(crate) fn fix_types(
        &self,
        name: &str,
        mut args: Vec<(String, PyVal)>,
    ) -> Vec<(String, PyVal)> {
        if self.tools.is_empty() || name.is_empty() {
            return args;
        }
        let Some(props) = self.properties(name) else {
            return args;
        };
        for (key, value) in args.iter_mut() {
            let Some(schema @ Value::Object(_)) = props.get(key) else {
                continue;
            };
            if let PyVal::Str(s) = value
                && let Some(new) = coerce(s, schema)
            {
                *value = new;
            }
        }
        args
    }
}

/// The final arguments of a call: `_fix_arg_types(converter(raw, False))`.
pub(crate) fn final_arguments(schemas: &ToolSchemas, name: &str, raw: &str) -> String {
    python::dumps(&PyVal::Dict(schemas.fix_types(name, convert(raw, false))))
}

/// The streamed form so far: `_fix_arg_types(converter(raw, True))`.
pub(crate) fn partial_arguments(schemas: &ToolSchemas, name: &str, raw: &str) -> String {
    python::dumps(&PyVal::Dict(schemas.fix_types(name, convert(raw, true))))
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn tool(name: &str, params: Value) -> Tool {
        Tool {
            name: name.into(),
            description: None,
            parameters: params,
            strict: None,
            defer_loading: None,
        }
    }

    #[test]
    fn converter_rules() {
        let c = |raw: &str| python::dumps(&PyVal::Dict(convert(raw, false)));
        assert_eq!(
            c("\n<parameter=cmd>\nls -la\n</parameter>\n"),
            "{\"cmd\": \"ls -la\"}"
        );
        // [F1] no </parameter>: the value runs to the end
        assert_eq!(c("\n<parameter=cmd>\nls\n"), "{\"cmd\": \"ls\"}");
        // [F2] the first value
        assert_eq!(
            c("<parameter=cmd>ls</parameter>\n<parameter=cmd>pwd</parameter>"),
            "{\"cmd\": \"ls\"}"
        );
        // [F3] the last </parameter> before the next parameter
        assert_eq!(
            c("<parameter=patch>\n+a\n</parameter>\n+b\n</parameter>\n"),
            "{\"patch\": \"+a\\n</parameter>\\n+b\"}"
        );
        // [F3] an opener mid-line is text
        assert_eq!(
            c("<parameter=patch>\n+<parameter=cmd>\n+ls\n</parameter>\n"),
            "{\"patch\": \"+<parameter=cmd>\\n+ls\"}"
        );
        assert_eq!(c("<parameter =x>v</parameter>"), "{\"x\": \"v\"}");
    }

    #[test]
    fn coercion_rules() {
        let s = ToolSchemas::new(
            &[tool(
                "f",
                json!({"type": "object", "properties": {
                    "n": {"type": "integer"}, "o": {"type": "object"}, "a": {"type": "array"},
                    "x": {"type": "number"}, "b": {"type": "boolean"}, "s": {"type": "string"}}}),
            )],
            true,
        );
        let fix = |k: &str, v: &str| {
            python::dumps(&PyVal::Dict(
                s.fix_types("f", vec![(k.into(), PyVal::Str(v.into()))]),
            ))
        };
        assert_eq!(fix("n", "3.0"), "{\"n\": 3}"); // [F9]
        assert_eq!(fix("n", " 7 "), "{\"n\": 7}");
        assert_eq!(
            fix("o", "{'a': None, 'b': True}"),
            "{\"o\": {\"a\": null, \"b\": true}}"
        ); // [F8]
        assert_eq!(fix("a", "['a', 'b']"), "{\"a\": [\"a\", \"b\"]}");
        assert_eq!(fix("x", "1e3"), "{\"x\": 1000}");
        assert_eq!(fix("x", "0.5"), "{\"x\": 0.5}");
        assert_eq!(fix("b", "False"), "{\"b\": false}");
        assert_eq!(fix("s", "5"), "{\"s\": \"5\"}");
        assert_eq!(fix("n", "abc"), "{\"n\": \"abc\"}");
    }

    #[test]
    fn safe_prefix_holds_the_last_value() {
        let keys: HashSet<String> = ["cmd".to_string()].into();
        assert_eq!(
            safe_arg_prefix("{\"cmd\": \"ls -l", Some(&keys)),
            "{\"cmd\": \"ls -l"
        );
        assert_eq!(safe_arg_prefix("{\"n\": 3", Some(&keys)), "{\"n\": ");
        assert_eq!(
            safe_arg_prefix("{\"cmd\": \"ls\"}", Some(&keys)),
            "{\"cmd\": \"ls"
        );
        assert_eq!(safe_arg_prefix("{}", Some(&keys)), "");
    }
}
