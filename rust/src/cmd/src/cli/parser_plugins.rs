// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! `--tool-parser-plugin` / `--reasoning-parser-plugin`. vllm-rs has no Python plugin mechanism:
//! it accepts the nemotron4 plugin files, whose parsers it has built in (a Rust port of them), so
//! that one `vllm serve` line works with either frontend, and refuses any other file.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Deserializer};
use sha2::{Digest, Sha256};
use thiserror_ext::AsReport as _;
use tracing::{info, warn};
use vllm_chat::NEMOTRON4_VERSION;

/// The Python plugins' revision whose rules the built-in port implements.
const PORTED_REVISIONS: [&str; 1] = [NEMOTRON4_VERSION];

/// One of the two plugin flags.
struct PluginFlag {
    flag: &'static str,
    /// The file's `PLUGIN_NAME`.
    plugin_name: &'static str,
    /// The parser selection that uses the built-in port.
    selection: &'static str,
}

const TOOL: PluginFlag = PluginFlag {
    flag: "--tool-parser-plugin",
    plugin_name: "nemotron4_tool_parser_plugin",
    selection: "--tool-call-parser nemotron4_tool",
};

const REASONING: PluginFlag = PluginFlag {
    flag: "--reasoning-parser-plugin",
    plugin_name: "nemotron4_reasoning_parser",
    selection: "--reasoning-parser nemotron4",
};

/// A module-level string constant of the Python file, `NAME = "value"`.
fn constant(text: &str, name: &str) -> Option<String> {
    let prefix = format!("{name} = \"");
    text.lines()
        .find_map(|line| line.strip_prefix(prefix.as_str())?.strip_suffix('"'))
        .map(str::to_owned)
}

/// Accept `path` if it is the nemotron4 plugin file for `flag`, and log what serves it.
fn check(flag: &PluginFlag, path: &Path) -> Result<PathBuf, String> {
    let bytes = std::fs::read(path).map_err(|error| {
        format!(
            "{} {}: cannot read the file: {}",
            flag.flag,
            path.display(),
            error.as_report()
        )
    })?;
    let text = String::from_utf8_lossy(&bytes);
    if constant(&text, "PLUGIN_NAME").as_deref() != Some(flag.plugin_name) {
        return Err(format!(
            "{} {}: vllm-rs loads no Python plugins; it accepts only the nemotron4 plugin \
             file ({}.py), whose parser it has built in",
            flag.flag,
            path.display(),
            flag.plugin_name
        ));
    }
    let revision = constant(&text, "REVISION").unwrap_or_else(|| "?".to_owned());
    let sha256: String = Sha256::digest(&bytes)[..6].iter().map(|b| format!("{b:02x}")).collect();
    info!(
        path = %path.display(),
        plugin = flag.plugin_name,
        revision,
        sha256,
        "{}: served by the built-in Rust port of the nemotron4 parsers ({NEMOTRON4_VERSION}) \
         with {}; the Python file is not loaded by the Rust frontend",
        flag.flag,
        flag.selection
    );
    if !PORTED_REVISIONS.contains(&revision.as_str()) {
        warn!(
            revision,
            "{}: the built-in nemotron4 parsers implement the Python plugins' {} rules; results \
             may differ from this file's revision",
            flag.flag,
            NEMOTRON4_VERSION
        );
    }
    Ok(path.to_path_buf())
}

/// clap value parser for `--tool-parser-plugin`.
pub(super) fn parse_tool_plugin(value: &str) -> Result<PathBuf, String> {
    check(&TOOL, Path::new(value))
}

/// clap value parser for `--reasoning-parser-plugin`.
pub(super) fn parse_reasoning_plugin(value: &str) -> Result<PathBuf, String> {
    check(&REASONING, Path::new(value))
}

/// `--args-json` field `tool_parser_plugin`.
pub(super) fn deserialize_tool_plugin<'de, D>(deserializer: D) -> Result<Option<PathBuf>, D::Error>
where
    D: Deserializer<'de>,
{
    let path = Option::<PathBuf>::deserialize(deserializer)?;
    path.map(|path| check(&TOOL, &path))
        .transpose()
        .map_err(serde::de::Error::custom)
}

/// `--args-json` field `reasoning_parser_plugin`.
pub(super) fn deserialize_reasoning_plugin<'de, D>(
    deserializer: D,
) -> Result<Option<PathBuf>, D::Error>
where
    D: Deserializer<'de>,
{
    let path = Option::<PathBuf>::deserialize(deserializer)?;
    path.map(|path| check(&REASONING, &path))
        .transpose()
        .map_err(serde::de::Error::custom)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plugin_file(dir: &Path, name: &str, body: &str) -> PathBuf {
        let path = dir.join(name);
        std::fs::write(&path, body).unwrap();
        path
    }

    #[test]
    fn accepts_only_the_nemotron4_plugins() {
        let dir = tempfile::tempdir().unwrap();
        let tool = plugin_file(
            dir.path(),
            "nemotron4_tool_parser_plugin.py",
            "\"\"\"doc\"\"\"\nPLUGIN_NAME = \"nemotron4_tool_parser_plugin\"\nREVISION = \"1.2.0rc1\"\n",
        );
        let reasoning = plugin_file(
            dir.path(),
            "nemotron4_reasoning_parser.py",
            "PLUGIN_NAME = \"nemotron4_reasoning_parser\"\nREVISION = \"1.3.0rc0\"\n",
        );
        let other = plugin_file(dir.path(), "my_parser.py", "PLUGIN_NAME = \"mine\"\n");

        assert_eq!(parse_tool_plugin(tool.to_str().unwrap()).unwrap(), tool);
        assert_eq!(
            parse_reasoning_plugin(reasoning.to_str().unwrap()).unwrap(),
            reasoning
        );
        // each flag takes its own file only
        assert!(parse_tool_plugin(reasoning.to_str().unwrap()).is_err());
        let error = parse_reasoning_plugin(other.to_str().unwrap()).unwrap_err();
        assert!(error.contains("vllm-rs loads no Python plugins"), "{error}");
        assert!(parse_tool_plugin(dir.path().join("missing.py").to_str().unwrap()).is_err());
    }
}
