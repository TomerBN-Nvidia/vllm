// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Render reference cases (the nemotron4 package's `eval/rust_render_conformance.py export`) with
//! the Rust structural encoder and write its results in the same layout, for
//! `rust_render_conformance.py compare`:
//!
//!     cargo run --example nemotron4_render_conformance -- REF_DIR OUT_DIR
//!
//! With `BENCH=N`, each case is also rendered N times and its best time printed.

use std::path::Path;

use serde_json::{Value, json};
use vllm_chat::nemotron4_render::{RenderInput, StructuralRenderError, StructuralRenderer};
use vllm_tokenizer::HuggingFaceTokenizer;

fn added_tokens(tokenizer_json: &Path) -> Vec<(String, u32)> {
    let spec: Value =
        serde_json::from_str(&std::fs::read_to_string(tokenizer_json).unwrap()).unwrap();
    spec["added_tokens"]
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|t| {
                    Some((t["content"].as_str()?.to_string(), t["id"].as_u64()? as u32))
                })
                .collect()
        })
        .unwrap_or_default()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (ref_dir, out_dir) = (Path::new(&args[1]), Path::new(&args[2]));
    std::fs::create_dir_all(out_dir).unwrap();
    let mut files: Vec<_> = std::fs::read_dir(ref_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .collect();
    files.sort();
    for path in files.into_iter().filter(|p| p.extension().is_some_and(|e| e == "json")) {
        let data: Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        let tok_dir = Path::new(data["tokenizer_dir"].as_str().unwrap());
        let tokenizer = HuggingFaceTokenizer::new_hf(&tok_dir.join("tokenizer.json")).unwrap();
        let template = std::fs::read_to_string(data["template_path"].as_str().unwrap()).unwrap();
        let special: Vec<(String, Value)> = data["special_tokens"]
            .as_object()
            .map(|o| o.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
            .unwrap_or_default();
        let renderer = StructuralRenderer::new(
            &template,
            &added_tokens(&tok_dir.join("tokenizer.json")),
            special,
        );
        let mut cases = Vec::new();
        let mut ok = 0;
        for case in data["cases"].as_array().unwrap() {
            let result = match &renderer {
                Err(e) => json!({"error": "Compile", "message": e.to_string()}),
                Ok(r) => {
                    let empty = serde_json::Map::new();
                    let input = RenderInput {
                        messages: case["messages"].as_array().map(Vec::as_slice).unwrap_or(&[]),
                        tools: case["tools"].as_array().map(Vec::as_slice),
                        documents: case["kwargs"]["documents"].as_array().map(Vec::as_slice),
                        add_generation_prompt: case["add_generation_prompt"]
                            .as_bool()
                            .unwrap_or(true),
                        continue_final_message: case["continue_final_message"]
                            .as_bool()
                            .unwrap_or(false),
                        template_kwargs: case["kwargs"].as_object().unwrap_or(&empty),
                    };
                    let repeats: usize =
                        std::env::var("BENCH").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
                    let mut best = f64::INFINITY;
                    for _ in 0..repeats {
                        let t0 = std::time::Instant::now();
                        let _ = r.render(&input, &tokenizer);
                        best = best.min(t0.elapsed().as_secs_f64() * 1000.0);
                    }
                    if repeats > 0 {
                        eprintln!(
                            "  {}: {best:.1} ms (python {:.1} ms)",
                            case["id"].as_str().unwrap_or("?"),
                            case["python_ms"].as_f64().unwrap_or(f64::NAN)
                        );
                    }
                    match r.render(&input, &tokenizer) {
                        Ok(out) => json!({"ids": out.ids, "text": out.text}),
                        Err(e) => json!({
                            "error": match e {
                                StructuralRenderError::Compile(_) => "Compile",
                                StructuralRenderError::UnknownOrigin(_) => "UnknownOrigin",
                                StructuralRenderError::Template(_) => "Template",
                            },
                            "message": e.to_string(),
                        }),
                    }
                }
            };
            if result.get("ids") == case["python"].get("ids") && result.get("ids").is_some() {
                ok += 1;
            }
            cases.push(json!({"id": case["id"], "rust": result}));
        }
        eprintln!(
            "{}: {} cases, {ok} identical ids",
            data["name"].as_str().unwrap_or("?"),
            cases.len()
        );
        std::fs::write(
            out_dir.join(path.file_name().unwrap()),
            serde_json::to_string(&json!({"cases": cases})).unwrap(),
        )
        .unwrap();
    }
}
