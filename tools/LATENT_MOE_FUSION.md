# Fix D — latent-MoE input pre-quantize fusion

`ModelOptMxFp8FusedMoE.apply_monolithic` quantizes its bf16 input on
every call (modelopt.py:2046, post-Fix-D the unconditional path is now
the `else` branch). For NemotronH this is wasted work: the previous
Linear (`fc1_latent_proj`, MXFP8) already had the value in fp8+scale
form internally before dequantizing to bf16 to return through the
standard `Linear.forward` interface.

Phase 1 (landed this commit): `apply_monolithic` accepts an optional
pre-quantized input.

```python
out = quant_method.apply_monolithic(
    layer=layer,
    x=hidden_states_bf16,
    router_logits=router_logits,
    routing_replay_out=...,
    # NEW (keyword-only):
    x_mxfp8=hidden_states_fp8,        # [M, K] fp8_e4m3fn
    x_mxfp8_scale=hidden_states_scale # [M, K//32] uint8 linear layout
)
```

When `x_mxfp8 is None` (default), behaviour is bit-identical to the
pre-Fix-D path. Microbench `tools/microbench_fix_d.py` confirms
byte-equality at M ∈ {32, 128, 256, 512}.

Phase 2 (not landed; profile-driven follow-up): wire the path through
`DefaultMoERunner` and add an MXFP8 fast-path on `fc1_latent_proj` that
emits fp8 directly instead of dequantizing.

## Phase 2 wiring sketch

### 1. Linear-side: expose a pre-quantized output

`ModelOptMxFp8LinearMethod.apply` returns bf16 today. Add:

```python
class ModelOptMxFp8LinearMethod(LinearMethodBase):
    def apply_returning_mxfp8(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Like `apply`, but return the result already in MXFP8 + scale
        form (linear / un-swizzled layout). Eliminates one dequant→requant
        round-trip when the downstream consumer is also MXFP8."""
        # Internally this is a small variant of `_apply_flashinfer_cutlass`
        # that calls mxfp8_quantize on the bf16 mm output, then returns
        # the (fp8, scale) pair instead of casting to bf16.
        bf16_out = self.apply(layer, x, bias)
        out_fp8, out_scale = mxfp8_e4m3_quantize(
            bf16_out, is_sf_swizzled_layout=False
        )
        if out_scale.ndim == 1 and out_fp8.ndim == 2:
            out_scale = out_scale.view(out_fp8.size(0), -1)
        return out_fp8, out_scale
```

The naive implementation above (`apply` then quantize) is wrong — it
re-introduces the round-trip we wanted to skip. The real win requires
modifying `_apply_flashinfer_cutlass` itself to bypass the
mm_mxfp8's internal dequantize back to bf16, returning the kernel's
fp8 result directly. That depends on whether `flashinfer.mm_mxfp8` can
return fp8 output natively — probably no on 0.6.10. Either:

- Wait for a future flashinfer API that exposes the fp8 output, OR
- Land Phase 2A: the naive wrapper (no win) and gate Phase 2B on the
  kernel update.

For now, Phase 2 is **paused pending Tomer's confirmation that
mm_mxfp8 can return fp8 output**.

### 2. Runner-side: detect and propagate

`DefaultMoERunner.forward` currently calls
`apply_routed_input_transform` (default_moe_runner.py:410) which
invokes `self.routed_input_transform(hidden_states)` and returns bf16.

Phase 2 plumbing:

```python
def forward(self, hidden_states, router_logits):
    ...
    # Detect pre-quantize fast path.
    rit = self.routed_input_transform
    can_prequantize = (
        rit is not None
        and getattr(rit, "linear_method", None) is not None
        and isinstance(rit.linear_method, ModelOptMxFp8LinearMethod)
        and self.quant_method.is_monolithic
        and isinstance(self.quant_method, ModelOptMxFp8FusedMoE)
    )
    if can_prequantize:
        # Inline the transform + quantize.
        hidden_states_fp8, hidden_states_scale = (
            rit.linear_method.apply_returning_mxfp8(rit, hidden_states)
        )
        # bf16 view is still needed for shape/replay slicing.
        hidden_states_bf16 = torch.empty(
            hidden_states_fp8.shape,
            dtype=torch.bfloat16,
            device=hidden_states_fp8.device,
        )
        ...
        final_hidden_states = self.quant_method.apply_monolithic(
            layer=layer,
            x=hidden_states_bf16,
            router_logits=router_logits,
            routing_replay_out=routing_replay_out,
            x_mxfp8=hidden_states_fp8,
            x_mxfp8_scale=hidden_states_scale,
        )
    else:
        hidden_states = self.apply_routed_input_transform(hidden_states)
        # existing path
        ...
```

Two call sites in default_moe_runner.py:506 and :697 both need the
same plumbing.

### 3. NemotronH opt-in

Set a flag on the `experts` layer in NemotronHMoE.__init__:

```python
self.experts._prequantize_routed_input = True  # opt into Fix D
```

Then in DefaultMoERunner.forward, only run Phase 2 when this flag is
present:

```python
can_prequantize = (
    getattr(self.layer, "_prequantize_routed_input", False)
    and ... # the conditions above
)
```

Keeps the optimization NemotronH-only until other models opt in.

## Why phase 2 is paused

The 0.5-2% E2E impact (revised down from §7.2's loose 5-10%) doesn't
justify the cross-cutting plumbing change in `DefaultMoERunner` until a
profile (Fix A, when oci-hsg returns) confirms the latent-MoE quantize
boundary is actually a measurable hot path. If the profile shows
quantize is < 1% of step time, Phase 2 is descoped.

Phase 1 (the API surface in `apply_monolithic`) is shipped because it's
a backwards-compatible 20-line change with a byte-equality microbench
that costs nothing to keep.

## How to validate phase 1

```bash
python3 tools/microbench_fix_d.py
```

Confirms at M ∈ {32, 128, 256, 512} that the new pre-quantized path
gives byte-identical output to the current quantize-inside path on
GB200 + flashinfer 0.6.10.
