# Nemotron Shortcut for RL

Use the checkpoint's config, processor, tokenizer and trained weights together.
Source-of-truth checkpoints declare their architecture, `index_topk`, MTP depth
and repeated-layer setting. No `--hf-overrides` is needed to replace those fields.
In particular, do not disable `index_topk` to bypass missing indexer weights:
that changes sparse attention to full attention.

## Weight updates and MTP

An initial sparse checkpoint load requires indexer weights. After that model
instance has validated its indexer, RL may send refit buckets that contain no
indexer tensors, including empty buckets. The update must still transfer all
weights that changed; this check does not validate completeness of a refit.

Enable MTP with the normal speculative-decoding option:

```bash
vllm serve /path/to/source-of-truth-checkpoint \
    --speculative-config '{"method":"mtp","num_speculative_tokens":2}'
```

vLLM derives a separate draft config from the checkpoint's text config, retaining
its quantization settings. This internal conversion does not require a user
`--hf-overrides` argument and leaves the target config unchanged.

## Video prompts

The Python processor formats both T=1 and T=2 videos with the training header
and timestamped frame labels. Processor caching retains the metadata used for
those labels. When using the training-style video loader, its patch size must
match `video_temporal_patch_size` in the checkpoint (absent means T=1):

```bash
vllm serve /path/to/t1-checkpoint \
    --media-io-kwargs '{"video":{"video_backend":"nemotron_shortcut_t2","temporal_patch_size":1,"fps":2,"max_frames":64}}'
```

For a T=2 checkpoint, use `"temporal_patch_size":2`. This loader option controls
frame sampling; it does not override the model config. The engine's `max_frames`
also bounds the video memory profile.

## MoE memory

FlashInfer builds that support caller-owned CUTLASS MoE workspaces use persistent
scratch managed by vLLM. Memory profiling runs again after persistent buffers
have grown, accounting for the target model's activations on top of shared MTP
buffers. Other FlashInfer builds retain their existing allocation path.
