// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::fmt;
use std::str::FromStr;

use itertools::Itertools;
use serde_with::{DeserializeFromStr, SerializeDisplay};
use strum::{EnumIter, IntoEnumIterator};

/// Specify which chat renderer implementation to use.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, DeserializeFromStr, SerializeDisplay, EnumIter,
)]
pub enum RendererSelection {
    /// Use model-based auto-detection.
    #[default]
    Auto,
    /// Force the generic Hugging Face chat-template renderer.
    Hf,
    /// Force the DeepSeek V3.2 renderer.
    DeepSeekV32,
    /// Force the DeepSeek V4 renderer.
    DeepSeekV4,
    /// Force the DeepSeek V4.1 renderer.
    DeepSeekV41,
    /// Force the GPT-OSS Harmony renderer.
    Harmony,
    /// Force the Inkling native token renderer.
    Inkling,
    /// Force the Kimi K3 XTML renderer.
    KimiK3,
    /// Force the native GLM-5.3 / GLM-5.3-Flash renderer.
    Glm5Next,
    /// The nemotron4 structural renderer: the HF chat template compiled to a symbolic encoder
    /// (selected with the nemotron4 parsers).
    Nemotron4,
}

impl RendererSelection {
    pub const AUTO_LITERAL: &str = "auto";
    pub const DEEPSEEK_V32_LITERAL: &str = "deepseek_v32";
    pub const DEEPSEEK_V4_LITERAL: &str = "deepseek_v4";
    pub const DEEPSEEK_V41_LITERAL: &str = "deepseek_v41";
    pub const GPT_OSS_MODEL_TYPE: &str = "gpt_oss";
    pub const HARMONY_LITERAL: &str = "harmony";
    pub const HF_LITERAL: &str = "hf";
    pub const INKLING_LITERAL: &str = "inkling";
    pub const INKLING_MODEL_TYPE: &str = "inkling_mm_model";
    pub const KIMI_K3_LITERAL: &str = "kimi_k3";
    pub const KIMI_K31_MODEL_TYPE: &str = "kimi_k31";
    pub const GLM5_NEXT_LITERAL: &str = "glm5_next";
    pub const NEMOTRON4_LITERAL: &str = "nemotron4";

    /// The renderer for these parser selections: `Auto` becomes the nemotron4 structural renderer
    /// when the nemotron4 parsers are selected (as the Python plugin's input hook comes with its
    /// tool parser), unless `NEMOTRON4_STRUCTURAL_CHAT=0`.
    pub fn for_parsers(
        self,
        tool_call_parser: &crate::parser::ParserSelection,
        reasoning_parser: &crate::parser::ParserSelection,
    ) -> Self {
        use crate::parser::ParserSelection;
        if self != Self::Auto || !crate::renderer::nemotron4::structural_chat_enabled() {
            return self;
        }
        let name = |s: &ParserSelection| match s {
            ParserSelection::Explicit(name) => Some(name.clone()),
            _ => None,
        };
        let (tool, reasoning) = (name(tool_call_parser), name(reasoning_parser));
        let nemotron4 =
            crate::parser::unified::paired_unified_name(tool.as_deref(), reasoning.as_deref())
                .is_some()
                || tool.as_deref() == Some(crate::parser::unified::names::NEMOTRON4)
                || tool
                    .as_deref()
                    .is_some_and(|t| crate::parser::unified::is_paired_name("tool", t));
        if nemotron4 { Self::Nemotron4 } else { self }
    }

    /// Resolve the renderer selection using the given model type string, if
    /// it's `Auto`.
    pub fn resolve(self, model_type: &str) -> Self {
        match self {
            Self::Auto => match model_type {
                Self::DEEPSEEK_V32_LITERAL => Self::DeepSeekV32,
                Self::DEEPSEEK_V4_LITERAL => Self::DeepSeekV4,
                Self::DEEPSEEK_V41_LITERAL => Self::DeepSeekV41,
                Self::GPT_OSS_MODEL_TYPE => Self::Harmony,
                Self::INKLING_MODEL_TYPE => Self::Inkling,
                Self::KIMI_K3_LITERAL | Self::KIMI_K31_MODEL_TYPE => Self::KimiK3,
                Self::GLM5_NEXT_LITERAL => Self::Glm5Next,
                _ => Self::Hf,
            },
            selection => selection,
        }
    }
}

impl FromStr for RendererSelection {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        if value.eq_ignore_ascii_case(Self::AUTO_LITERAL) {
            Ok(Self::Auto)
        } else if value.eq_ignore_ascii_case(Self::HF_LITERAL) {
            Ok(Self::Hf)
        } else if value.eq_ignore_ascii_case(Self::DEEPSEEK_V32_LITERAL) {
            Ok(Self::DeepSeekV32)
        } else if value.eq_ignore_ascii_case(Self::DEEPSEEK_V4_LITERAL) {
            Ok(Self::DeepSeekV4)
        } else if value.eq_ignore_ascii_case(Self::DEEPSEEK_V41_LITERAL) {
            Ok(Self::DeepSeekV41)
        } else if value.eq_ignore_ascii_case(Self::HARMONY_LITERAL) {
            Ok(Self::Harmony)
        } else if value.eq_ignore_ascii_case(Self::INKLING_LITERAL) {
            Ok(Self::Inkling)
        } else if value.eq_ignore_ascii_case(Self::KIMI_K3_LITERAL) {
            Ok(Self::KimiK3)
        } else if value.eq_ignore_ascii_case(Self::GLM5_NEXT_LITERAL) {
            Ok(Self::Glm5Next)
        } else if value.eq_ignore_ascii_case(Self::NEMOTRON4_LITERAL) {
            Ok(Self::Nemotron4)
        } else {
            Err(format!(
                "unknown renderer `{value}` (expected one of: {})",
                Self::iter().join(", ")
            ))
        }
    }
}

impl fmt::Display for RendererSelection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Auto => f.write_str(Self::AUTO_LITERAL),
            Self::Hf => f.write_str(Self::HF_LITERAL),
            Self::DeepSeekV32 => f.write_str(Self::DEEPSEEK_V32_LITERAL),
            Self::DeepSeekV4 => f.write_str(Self::DEEPSEEK_V4_LITERAL),
            Self::DeepSeekV41 => f.write_str(Self::DEEPSEEK_V41_LITERAL),
            Self::Harmony => f.write_str(Self::HARMONY_LITERAL),
            Self::Inkling => f.write_str(Self::INKLING_LITERAL),
            Self::KimiK3 => f.write_str(Self::KIMI_K3_LITERAL),
            Self::Glm5Next => f.write_str(Self::GLM5_NEXT_LITERAL),
            Self::Nemotron4 => f.write_str(Self::NEMOTRON4_LITERAL),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::str::FromStr as _;

    use strum::IntoEnumIterator;

    use super::RendererSelection;

    #[test]
    fn renderer_selection_display_round_trips() {
        for selection in RendererSelection::iter() {
            assert_eq!(
                selection.to_string().parse::<RendererSelection>().unwrap(),
                selection
            );
        }
    }

    #[test]
    fn auto_resolves_glm5_next_from_model_type_only() {
        // GLM-5.3 shares `glm_moe_dsa` with GLM-5 through GLM-5.2, whose
        // templates differ, so it needs an explicit `glm5_next`.
        assert_eq!(
            RendererSelection::Auto.resolve("glm5_next"),
            RendererSelection::Glm5Next
        );
        assert_eq!(
            RendererSelection::Auto.resolve("glm_moe_dsa"),
            RendererSelection::Hf
        );
    }

    #[test]
    fn renderer_selection_expected_error_message() {
        let err = RendererSelection::from_str("unknown").unwrap_err();
        expect_test::expect![
            "unknown renderer `unknown` (expected one of: auto, hf, deepseek_v32, deepseek_v4, deepseek_v41, harmony, inkling, kimi_k3, glm5_next, nemotron4)"
        ]
        .assert_eq(&err);
    }
}
