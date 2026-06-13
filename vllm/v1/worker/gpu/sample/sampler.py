# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

import numpy as np
import torch

import vllm.envs as envs
from vllm.config.model import LogprobsMode
from vllm.sampling_params import SamplingParams
from vllm.v1.worker.gpu.metrics.logits import get_num_nans
from vllm.v1.worker.gpu.sample.bad_words import BadWordsState
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
from vllm.v1.worker.gpu.sample.logit_bias import LogitBiasState
from vllm.v1.worker.gpu.sample.logprob import compute_topk_logprobs
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.gpu.sample.penalties import PenaltiesState
from vllm.v1.worker.gpu.sample.states import NO_LOGPROBS, SamplingStates
from vllm.v1.worker.gpu.states import RequestState


class Sampler:
    def __init__(
        self,
        max_num_reqs: int,
        vocab_size: int,
        device: torch.device,
        req_states: RequestState,
        logprobs_mode: LogprobsMode = "raw_logprobs",
        num_speculative_tokens: int = 1,
    ):
        if logprobs_mode not in ("processed_logprobs", "raw_logprobs"):
            raise NotImplementedError(f"Unsupported logprobs_mode: {logprobs_mode}")
        self.logprobs_mode = logprobs_mode
        self.compute_nans = envs.VLLM_COMPUTE_NANS_IN_LOGITS  # False by default.
        self.debug_logprobs = os.getenv("VLLM_NEMO_DEBUG_LOGPROBS") == "1"
        self._debug_logprobs_prints = 0
        try:
            self._debug_logprobs_max_prints = int(
                os.getenv("VLLM_NEMO_DEBUG_LOGPROBS_MAX_PRINTS", "32")
            )
        except ValueError:
            self._debug_logprobs_max_prints = 32

        self.sampling_states = SamplingStates(max_num_reqs, vocab_size)
        self.penalties_state = PenaltiesState(req_states)
        self.logit_bias_state = LogitBiasState(max_num_reqs, device)
        self.bad_words_state = BadWordsState(req_states)
        self.num_speculative_tokens = num_speculative_tokens

    def add_request(
        self, req_idx: int, prompt_len: int, sampling_params: SamplingParams
    ) -> None:
        self.sampling_states.add_request(req_idx, sampling_params)
        self.penalties_state.add_request(req_idx, sampling_params)
        self.logit_bias_state.add_request(req_idx, prompt_len, sampling_params)
        self.bad_words_state.add_request(req_idx, sampling_params)

    def apply_staged_writes(self) -> None:
        self.sampling_states.apply_staged_writes()
        self.penalties_state.apply_staged_writes()
        self.logit_bias_state.apply_staged_writes()
        self.bad_words_state.apply_staged_writes()

    def __call__(
        self,
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
        cu_num_logits_np: np.ndarray,
        pos: torch.Tensor,
        input_ids: torch.Tensor,
        expanded_local_pos: torch.Tensor,
    ) -> SamplerOutput:
        # NOTE(woosuk): We intentionally compute num_nans before sampling to make clear
        # that num_nans is computed before applying penalties and temperature.
        num_nans = get_num_nans(logits) if self.compute_nans else None
        sampled, processed_logits = self.sample(
            logits,
            expanded_idx_mapping,
            idx_mapping_np,
            pos,
            input_ids,
            expanded_local_pos,
        )

        max_num_logprobs = self.sampling_states.max_num_logprobs(idx_mapping_np)
        if max_num_logprobs != NO_LOGPROBS:
            if self.logprobs_mode == "processed_logprobs":
                logits = processed_logits
            expanded_logits = logits.shape[0] != idx_mapping_np.shape[0]
            cu_num_logits = cu_num_logits_np.tolist() if expanded_logits else None
            logprobs_tensors = compute_topk_logprobs(
                logits, max_num_logprobs, sampled, cu_num_logits
            )
            if self.debug_logprobs:
                self._debug_logprobs_path(
                    logits_for_logprobs=logits,
                    processed_logits=processed_logits,
                    logprobs=logprobs_tensors.logprobs,
                    sampled=sampled,
                    idx_mapping_np=idx_mapping_np,
                    max_num_logprobs=max_num_logprobs,
                )
        else:
            logprobs_tensors = None

        # These are GPU tensors.
        sampler_output = SamplerOutput(
            # The sampled tokens are expanded to 2D tensor with shape
            # [num_requests, 1], where each row represents one generated
            # token per request.
            sampled_token_ids=sampled.view(-1, 1),
            logprobs_tensors=logprobs_tensors,
            num_nans=num_nans,
        )
        return sampler_output

    def _debug_logprobs_path(
        self,
        logits_for_logprobs: torch.Tensor,
        processed_logits: torch.Tensor,
        logprobs: torch.Tensor,
        sampled: torch.Tensor,
        idx_mapping_np: np.ndarray,
        max_num_logprobs: int,
    ) -> None:
        def tensor_counts(tensor: torch.Tensor) -> tuple[int, int, int, int]:
            nan_count = torch.isnan(tensor).sum().item()
            inf_count = torch.isinf(tensor).sum().item()
            finite_count = torch.isfinite(tensor).sum().item()
            return tensor.numel(), finite_count, nan_count, inf_count

        _, _, logits_nan, logits_inf = tensor_counts(logits_for_logprobs)
        _, _, processed_nan, processed_inf = tensor_counts(processed_logits)
        _, _, logprob_nan, logprob_inf = tensor_counts(logprobs)
        has_nonfinite = any(
            count
            for count in (
                logits_nan,
                logits_inf,
                processed_nan,
                processed_inf,
                logprob_nan,
                logprob_inf,
            )
        )
        should_print = has_nonfinite or self._debug_logprobs_prints < 2
        if not should_print:
            return
        if self._debug_logprobs_prints >= self._debug_logprobs_max_prints:
            return

        logits_total, logits_finite, _, _ = tensor_counts(logits_for_logprobs)
        proc_total, proc_finite, _, _ = tensor_counts(processed_logits)
        lp_total, lp_finite, _, _ = tensor_counts(logprobs)

        bad_logprob_rows = torch.nonzero(
            ~torch.isfinite(logprobs).all(dim=1), as_tuple=False
        ).flatten()
        bad_logits_rows = torch.nonzero(
            ~torch.isfinite(logits_for_logprobs).all(dim=1), as_tuple=False
        ).flatten()

        first_bad_logprob_rows = (
            bad_logprob_rows[:8].detach().cpu().tolist()
            if bad_logprob_rows.numel()
            else []
        )
        first_bad_logits_rows = (
            bad_logits_rows[:8].detach().cpu().tolist()
            if bad_logits_rows.numel()
            else []
        )
        sampled_preview = sampled[:8].detach().cpu().tolist()
        req_preview = idx_mapping_np[:8].tolist()

        print(
            "[vllm-nemo-debug-logprobs] "
            f"mode={self.logprobs_mode} max_num_logprobs={max_num_logprobs} "
            f"reqs={len(idx_mapping_np)} req_idx_preview={req_preview} "
            f"sampled_preview={sampled_preview} "
            f"logits_shape={tuple(logits_for_logprobs.shape)} "
            f"logits_finite={logits_finite}/{logits_total} "
            f"logits_nan={logits_nan} logits_inf={logits_inf} "
            f"processed_shape={tuple(processed_logits.shape)} "
            f"processed_finite={proc_finite}/{proc_total} "
            f"processed_nan={processed_nan} processed_inf={processed_inf} "
            f"logprobs_shape={tuple(logprobs.shape)} "
            f"logprobs_finite={lp_finite}/{lp_total} "
            f"logprobs_nan={logprob_nan} logprobs_inf={logprob_inf} "
            f"bad_logits_rows={first_bad_logits_rows} "
            f"bad_logprob_rows={first_bad_logprob_rows}",
            flush=True,
        )
        self._debug_logprobs_prints += 1

    def sample(
        self,
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
        pos: torch.Tensor,
        input_ids: torch.Tensor,
        expanded_local_pos: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Copy logits to a new FP32 tensor.
        logits = torch.empty_like(logits, dtype=torch.float32).copy_(logits)

        # Apply logit bias (e.g., allowed_token_ids, min_tokens) in place.
        self.logit_bias_state.apply_logit_bias(
            logits, expanded_idx_mapping, idx_mapping_np, pos
        )

        # Apply penalties in place.
        self.penalties_state.apply_penalties(
            logits,
            expanded_idx_mapping,
            idx_mapping_np,
            input_ids,
            expanded_local_pos,
            self.num_speculative_tokens,
        )

        # Apply bad words masking in place.
        self.bad_words_state.apply_bad_words(
            logits,
            expanded_idx_mapping,
            idx_mapping_np,
            input_ids,
            expanded_local_pos,
        )

        # Apply temperature in place.
        self.sampling_states.apply_temperature(
            logits, expanded_idx_mapping, idx_mapping_np
        )

        # Apply min_p in place.
        self.sampling_states.apply_min_p(logits, expanded_idx_mapping, idx_mapping_np)

        # Apply top_k and/or top_p. This might or might not return a new tensor.
        logits = self.sampling_states.apply_top_k_top_p(
            logits, expanded_idx_mapping, idx_mapping_np
        )

        # Sample the next token.
        sampled = gumbel_sample(
            logits,
            expanded_idx_mapping,
            self.sampling_states.temperature.gpu,
            self.sampling_states.seeds.gpu,
            pos,
            apply_temperature=False,
        )
        return sampled, logits
