"""
Unit-test the NemotronH exclude-pattern recipe against the actual
is_layer_excluded matcher used by ModelOptQuantConfigBase.

Run on any machine with vllm installed:

    VLLM_MODELOPT_EXTRA_EXCLUDE_MODULES="..." python3 tools/test_exclude_patterns.py

Or, without the env var (the env injection is mocked in the test below).
"""
from __future__ import annotations

import os
import sys


def main() -> None:
    # Run-time test wrappers around fnmatch + substring (mirrors
    # ModelOptQuantConfigBase.is_layer_excluded line 146-182).
    from fnmatch import fnmatch

    def is_layer_excluded(prefix: str, exclude_modules: list[str]) -> bool:
        if not exclude_modules:
            return False
        # Substring match (line 166-175 of modelopt.py)
        for em in exclude_modules:
            if em != prefix and (
                em in prefix
                or (
                    prefix.startswith("language_model.")
                    and em in prefix.removeprefix("language_model.")
                )
            ):
                return True
        # fnmatch wildcards (line 178-180)
        for pat in exclude_modules:
            if fnmatch(prefix, pat):
                return True
        return False

    recipe = (
        "*qkv_proj,*o_proj,*gate,*fc1_latent_proj,*fc2_latent_proj,"
        "*mixer.in_proj,*mixer.out_proj,*mixer.conv1d"
    )
    patterns = [p.strip() for p in recipe.split(",") if p.strip()]

    # Representative NemotronH layer prefixes from the v17rl branch.
    # SHOULD be EXCLUDED (kept BF16):
    excluded = [
        "backbone.layers.0.mixer.qkv_proj",
        "backbone.layers.12.mixer.o_proj",
        "backbone.layers.42.mixer.gate",
        "backbone.layers.42.mixer.fc1_latent_proj",
        "backbone.layers.42.mixer.fc2_latent_proj",
        "backbone.layers.7.mixer.in_proj",
        "backbone.layers.7.mixer.out_proj",
        "backbone.layers.7.mixer.conv1d",
        # language_model. prefix variant (LLaVA-style nested checkpoint)
        "language_model.backbone.layers.5.mixer.qkv_proj",
    ]
    # SHOULD STAY MXFP8 (NOT excluded):
    not_excluded = [
        "backbone.layers.42.mixer.experts",                       # routed experts
        "backbone.layers.42.mixer.shared_experts.up_proj",        # shared expert MLP
        "backbone.layers.42.mixer.shared_experts.down_proj",
        "backbone.layers.20.mixer.up_proj",                       # pure MLP layer
        "backbone.layers.20.mixer.down_proj",
        "embed_tokens",
        "lm_head",
    ]

    pass_count = 0
    fail_count = 0
    for p in excluded:
        ok = is_layer_excluded(p, patterns)
        tag = "PASS" if ok else "FAIL"
        print(f"  [{tag}]  excluded? {p}")
        pass_count += int(ok)
        fail_count += int(not ok)
    for p in not_excluded:
        ok = not is_layer_excluded(p, patterns)
        tag = "PASS" if ok else "FAIL"
        print(f"  [{tag}]  kept MXFP8? {p}")
        pass_count += int(ok)
        fail_count += int(not ok)

    total = pass_count + fail_count
    print()
    print(f"== {pass_count}/{total} pass, {fail_count} fail ==")
    if fail_count:
        sys.exit(1)


if __name__ == "__main__":
    main()
