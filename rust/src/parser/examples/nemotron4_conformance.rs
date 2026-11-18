// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Run the nemotron4 parser over reference cases (the nemotron4 package's `eval/rust_conformance.py
//! export`) and write its results in the same layout, for `rust_conformance.py compare`:
//!
//!     cargo run --example nemotron4_conformance -- REF.json OUT.json
//!
//! Each case's output ids are decoded as the Rust frontend decodes them (the stop token left out,
//! special tokens kept since the parser asks for them) and fed to the parser all at once, not
//! streamed ("all"), and streamed at 1 and 3 ids per delta.

use std::collections::{BTreeMap, HashMap};
use std::path::Path;
use std::sync::Arc;

use serde_json::{Value, json};
use thiserror_ext::AsReport;
use vllm_parser::tool::Tool;
use vllm_parser::unified::{
    Nemotron4Options, Nemotron4ToolChoice, Nemotron4UnifiedParser, UnifiedParser,
    UnifiedParserEvent, UnifiedParserOutput,
};
use vllm_tokenizer::{DynTokenizer, HuggingFaceTokenizer};

fn tools_of(v: &Value) -> Vec<Tool> {
    v.as_array()
        .map(|a| {
            a.iter()
                .filter_map(|t| {
                    let f = t.get("function")?;
                    Some(Tool {
                        name: f.get("name")?.as_str()?.to_string(),
                        description: f
                            .get("description")
                            .and_then(Value::as_str)
                            .map(str::to_string),
                        parameters: f.get("parameters").cloned().unwrap_or(json!({})),
                        strict: f.get("strict").and_then(Value::as_bool),
                        defer_loading: f.get("defer_loading").and_then(Value::as_bool),
                    })
                })
                .collect()
        })
        .unwrap_or_default()
}

fn tool_choice_of(v: &Value) -> Nemotron4ToolChoice {
    match v {
        Value::String(s) if s == "none" => Nemotron4ToolChoice::None,
        Value::String(s) if s == "required" => Nemotron4ToolChoice::Required,
        Value::Object(o) => Nemotron4ToolChoice::Named(
            o.get("function")
                .and_then(|f| f.get("name"))
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string(),
        ),
        _ => Nemotron4ToolChoice::Auto,
    }
}

struct Acc {
    reasoning: String,
    content: String,
    calls: BTreeMap<usize, (String, String)>,
    reasoning_tokens: usize,
}

impl Acc {
    fn add(&mut self, out: UnifiedParserOutput) {
        for e in out.events {
            match e {
                UnifiedParserEvent::Text(t) => self.content.push_str(&t),
                UnifiedParserEvent::Reasoning(dt) => {
                    self.reasoning_tokens += dt.attributions.len();
                    self.reasoning.push_str(&dt.text);
                }
                UnifiedParserEvent::ToolCall(c) => {
                    let slot = self.calls.entry(c.tool_index).or_default();
                    if let Some(n) = c.name {
                        slot.0.push_str(&n);
                    }
                    slot.1.push_str(&c.arguments);
                }
            }
        }
    }
}

fn run(
    tok: &DynTokenizer,
    case: &Value,
    special_ids: &[u32],
    eos_ids: &[u32],
    chunk: Option<usize>,
) -> Result<(Value, usize), String> {
    let tools = tools_of(&case["tools"]);
    let mut parser =
        Nemotron4UnifiedParser::new(&tools, tok.clone()).map_err(|e| e.to_report_string())?;
    let kwargs: HashMap<String, Value> = case["chat_template_kwargs"]
        .as_object()
        .map(|o| o.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
        .unwrap_or_default();
    parser
        .set_options(Nemotron4Options {
            tool_choice: tool_choice_of(&case["tool_choice"]),
            template_kwargs: kwargs,
            chat_template: None,
            special_ids: Some(special_ids.to_vec()),
            stream: chunk.is_some(),
            end_ids: Some(eos_ids.to_vec()),
        })
        .map_err(|e| e.to_report_string())?;
    let prompt: Vec<u32> = case["prompt_ids"]
        .as_array()
        .map(|a| a.iter().filter_map(|x| x.as_u64().map(|x| x as u32)).collect())
        .unwrap_or_default();
    if case["prompt_ids"].is_array() {
        parser.initialize(&prompt).map_err(|e| e.to_report_string())?;
    }
    let mut out: Vec<u32> = case["output_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap() as u32)
        .collect();
    if case["stop"].as_bool() == Some(true) {
        out.pop(); // the Rust frontend leaves the stop token out of the decoded text
    }
    let mut decoder = tok.create_decode_stream(&prompt, !parser.preserve_special_tokens(), 0);
    let mut acc = Acc {
        reasoning: String::new(),
        content: String::new(),
        calls: BTreeMap::new(),
        reasoning_tokens: 0,
    };
    let n = chunk.unwrap_or(out.len().max(1));
    for c in out.chunks(n) {
        for &id in c {
            decoder.push_token(id).map_err(|e| e.to_report_string())?;
        }
        if let Some(dt) = decoder.next_chunk() {
            let mut o = UnifiedParserOutput::default();
            parser.parse_into(dt, &mut o).map_err(|e| e.to_report_string())?;
            acc.add(o);
        }
    }
    let (last, _) = decoder.flush(None).map_err(|e| e.to_report_string())?;
    if let Some(dt) = last {
        let mut o = UnifiedParserOutput::default();
        parser.parse_into(dt, &mut o).map_err(|e| e.to_report_string())?;
        acc.add(o);
    }
    acc.add(parser.finish().map_err(|e| e.to_report_string())?);
    let opt = |s: String| {
        if s.is_empty() {
            Value::Null
        } else {
            Value::String(s)
        }
    };
    let calls: Vec<Value> = acc.calls.into_values().map(|(n, a)| json!([n, a])).collect();
    Ok((
        json!([opt(acc.reasoning), opt(acc.content), calls]),
        acc.reasoning_tokens,
    ))
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (ref_path, out_path) = (&args[1], &args[2]);
    let data: Value = serde_json::from_str(&std::fs::read_to_string(ref_path).unwrap()).unwrap();
    let tok_path = data["source"]["tokenizer_json"].as_str().unwrap();
    let tok: DynTokenizer = Arc::new(HuggingFaceTokenizer::new_hf(Path::new(tok_path)).unwrap());
    let special_ids: Vec<u32> = data["special_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap() as u32)
        .collect();
    // the stop tokens the Python parsers had (the reference's)
    let eos_ids: Vec<u32> = data["eos_ids"]
        .as_array()
        .map(|a| a.iter().filter_map(|x| x.as_u64().map(|x| x as u32)).collect())
        .unwrap_or_default();
    let mut cases = Vec::new();
    let mut errors = 0;
    for case in data["cases"].as_array().unwrap() {
        let mut results = serde_json::Map::new();
        let mut counts = serde_json::Map::new();
        for (mode, chunk) in [("all", None), ("1", Some(1)), ("3", Some(3))] {
            match run(&tok, case, &special_ids, &eos_ids, chunk) {
                Ok((r, n)) => {
                    results.insert(mode.into(), r);
                    counts.insert(mode.into(), json!(n));
                }
                Err(e) => {
                    errors += 1;
                    results.insert("error".into(), json!(e));
                }
            }
        }
        cases.push(json!({"id": case["id"], "results": results, "reasoning_tokens": counts}));
    }
    std::fs::write(
        out_path,
        serde_json::to_string(&json!({"cases": cases})).unwrap(),
    )
    .unwrap();
    eprintln!("wrote {out_path}: {} cases, {errors} errors", cases.len());
}
