# Nemotron Shortcut for RL

Use the checkpoint's config, processor, tokenizer and weights together; no
`--hf-overrides` is needed. MTP uses a separate draft config and retains the
checkpoint's quantization settings.

An initial sparse load requires trained indexer weights. Later RL refit buckets
may omit indexer tensors. This check does not verify that a refit is complete.

T=1 and T=2 video prompts use training-style timestamp labels. To use training
frame sampling for a T=1 checkpoint, set
`--media-io-kwargs '{"video":{"video_backend":"nemotron_shortcut_t2","temporal_patch_size":1}}'`.
The loader patch size must match the checkpoint; its default is 2. The engine's
`max_frames` also sets the frame cap for memory profiling.
