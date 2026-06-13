# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A layer that samples the next tokens from the model's outputs."""

import os

import torch
import torch.nn as nn

from vllm.config.model import LogprobsMode
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.v1.outputs import LogprobsTensors, SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.ops.bad_words import apply_bad_words
from vllm.v1.sample.ops.logprobs import batched_count_greater_than
from vllm.v1.sample.ops.penalties import apply_all_penalties
from vllm.v1.sample.ops.topk_topp_sampler import TopKTopPSampler

_SAMPLING_EPS = 1e-5


class Sampler(nn.Module):
    """
    A layer that samples the next tokens from the model's outputs
    with the following steps in order:

    1. If logprobs are requested:
        a) If `logprobs_mode` is `raw_logprobs`, compute logprobs
           as the final logprobs to return.
        b) If `logprobs_mode` is `raw_logits`, clone the logits
           as the final logprobs to return.
    2. Convert logits to float32.
    3. Apply allowed token ids whitelist.
    4. Apply bad words exclusion.
    5. Apply logit processors which are not argmax-invariant,
       i.e. that can impact greedy sampling.
        a) Min tokens processor
        b) Logit bias processor
    6. Apply penalties
        a) Repetition penalty
        b) Frequency penalty
        c) Presence penalty
    7. Sample the next tokens. `sample` method performs the following steps:
        a) If not `all_random`, perform greedy sampling. If `all_greedy`,
           return the greedily sampled tokens and final logprobs if requested.
        b) Apply temperature.
        c) Apply logit processors which are argmax-invariant, by default
           the min_p processor.
        d) Apply top_k and/or top_p.
        e) Sample the next tokens with the probability distribution.
        f) If `all_random` or temperature >= epsilon (1e-5), return the
           randomly sampled tokens and final logprobs if requested. Else,
           return the greedily sampled tokens and logprobs if requested.
    8. Gather the logprobs of the top `max_num_logprobs` and sampled token
       (if requested). Note that if the sampled token is within the top
       `max_num_logprobs`, the logprob will be eventually merged in
       `LogprobsProcessor` during output processing. Therefore, the
       final output may contain either `max_num_logprobs + 1` or
       `max_num_logprobs` logprobs.
    9. Return the final `SamplerOutput`.
    """

    def __init__(self, logprobs_mode: LogprobsMode = "raw_logprobs"):
        super().__init__()
        self.topk_topp_sampler = TopKTopPSampler(logprobs_mode)
        self.pin_memory = is_pin_memory_available()
        self.logprobs_mode = logprobs_mode
        self.debug_logprobs = os.getenv("VLLM_NEMO_DEBUG_LOGPROBS") == "1"
        self._debug_logprobs_prints = 0
        try:
            self._debug_logprobs_max_prints = int(
                os.getenv("VLLM_NEMO_DEBUG_LOGPROBS_MAX_PRINTS", "32")
            )
        except ValueError:
            self._debug_logprobs_max_prints = 32

    def forward(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        predict_bonus_token: bool = False,
        logprobs_mode_override: LogprobsMode | None = None,
    ) -> SamplerOutput:
        logprobs_mode = logprobs_mode_override or self.logprobs_mode
        raw_logits_for_debug = logits if self.debug_logprobs else None
        raw_logprobs = None
        # NOTE(woosuk): Use the original logits (before any penalties or
        # temperature scaling) for the top-k logprobs.
        # This is different from the V0 sampler, which uses the logits that
        # is used for sampling (after penalties and temperature scaling).
        num_logprobs = sampling_metadata.max_num_logprobs
        if num_logprobs is not None:
            if logprobs_mode == "raw_logprobs":
                raw_logprobs = self.compute_logprobs(logits)
            elif logprobs_mode == "raw_logits":
                if logits.dtype == torch.float32:
                    raw_logprobs = logits.clone()
                else:
                    raw_logprobs = logits.to(torch.float32)

        # Use float32 for the logits.
        logits = logits.to(torch.float32)

        logits = self.apply_logits_processors(
            logits, sampling_metadata, predict_bonus_token
        )
        # Sample the next token.
        sampled, processed_logprobs = self.sample(logits, sampling_metadata)
        if processed_logprobs is not None:
            raw_logprobs = processed_logprobs
        # Convert sampled token ids to int64 (long) type to ensure compatibility
        # with subsequent operations that may use these values as indices.
        # This conversion is necessary because FlashInfer sampling operations
        # return int32 (while PyTorch argmax and topk return int64).
        sampled = sampled.long()

        if num_logprobs is None:
            logprobs_tensors = None
        elif num_logprobs == -1:
            # Return the full unsorted and unranked logprobs.
            logprobs_tensors = LogprobsTensors(
                torch.empty(0), raw_logprobs, torch.empty(0)
            )
        else:
            # Gather the logprobs and ranks of the topk and sampled token.
            logprobs_tensors = self.gather_logprobs(
                raw_logprobs, num_logprobs, token_ids=sampled
            )
            if self.debug_logprobs:
                self._debug_logprobs_path(
                    logprobs_mode=logprobs_mode,
                    raw_logits=raw_logits_for_debug,
                    raw_logprobs=raw_logprobs,
                    processed_logits=logits,
                    processed_logprobs=processed_logprobs,
                    gathered_logprobs=logprobs_tensors.logprobs,
                    sampled=sampled,
                    sampling_metadata=sampling_metadata,
                    num_logprobs=num_logprobs,
                )

        # Use int32 to reduce the tensor size.
        sampled = sampled.to(torch.int32)

        # These are GPU tensors.
        sampler_output = SamplerOutput(
            # The sampled tokens are expanded to 2D tensor with shape
            # [num_requests, 1], where each row represents one generated
            # token per request.
            sampled_token_ids=sampled.unsqueeze(-1),
            logprobs_tensors=logprobs_tensors,
        )
        return sampler_output

    def _debug_logprobs_path(
        self,
        logprobs_mode: LogprobsMode,
        raw_logits: torch.Tensor | None,
        raw_logprobs: torch.Tensor | None,
        processed_logits: torch.Tensor,
        processed_logprobs: torch.Tensor | None,
        gathered_logprobs: torch.Tensor,
        sampled: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        num_logprobs: int,
    ) -> None:
        def tensor_counts(tensor: torch.Tensor | None) -> tuple[int, int, int, int]:
            if tensor is None:
                return 0, 0, 0, 0
            nan_count = torch.isnan(tensor).sum().item()
            inf_count = torch.isinf(tensor).sum().item()
            finite_count = torch.isfinite(tensor).sum().item()
            return tensor.numel(), finite_count, nan_count, inf_count

        raw_total, raw_finite, raw_nan, raw_inf = tensor_counts(raw_logits)
        rlp_total, rlp_finite, rlp_nan, rlp_inf = tensor_counts(raw_logprobs)
        proc_total, proc_finite, proc_nan, proc_inf = tensor_counts(processed_logits)
        plp_total, plp_finite, plp_nan, plp_inf = tensor_counts(processed_logprobs)
        glp_total, glp_finite, glp_nan, glp_inf = tensor_counts(gathered_logprobs)

        has_nonfinite = any(
            count
            for count in (
                raw_nan,
                raw_inf,
                rlp_nan,
                rlp_inf,
                proc_nan,
                proc_inf,
                plp_nan,
                plp_inf,
                glp_nan,
                glp_inf,
            )
        )
        should_print = has_nonfinite or self._debug_logprobs_prints < 2
        if not should_print:
            return
        if self._debug_logprobs_prints >= self._debug_logprobs_max_prints:
            return

        def bad_rows(tensor: torch.Tensor | None) -> list[int]:
            if tensor is None or tensor.ndim < 2:
                return []
            rows = torch.nonzero(~torch.isfinite(tensor).all(dim=1), as_tuple=False)
            return rows.flatten()[:8].detach().cpu().tolist()

        sampled_preview = sampled[:8].detach().cpu().tolist()
        output_len_preview = [len(ids) for ids in sampling_metadata.output_token_ids[:8]]

        print(
            "[vllm-nemo-debug-logprobs] "
            f"mode={logprobs_mode} num_logprobs={num_logprobs} "
            f"batch={processed_logits.shape[0]} "
            f"all_greedy={sampling_metadata.all_greedy} "
            f"all_random={sampling_metadata.all_random} "
            f"sampled_preview={sampled_preview} "
            f"output_len_preview={output_len_preview} "
            f"raw_logits_shape={tuple(raw_logits.shape) if raw_logits is not None else None} "
            f"raw_logits_finite={raw_finite}/{raw_total} "
            f"raw_logits_nan={raw_nan} raw_logits_inf={raw_inf} "
            f"raw_logprobs_shape={tuple(raw_logprobs.shape) if raw_logprobs is not None else None} "
            f"raw_logprobs_finite={rlp_finite}/{rlp_total} "
            f"raw_logprobs_nan={rlp_nan} raw_logprobs_inf={rlp_inf} "
            f"processed_logits_shape={tuple(processed_logits.shape)} "
            f"processed_logits_finite={proc_finite}/{proc_total} "
            f"processed_logits_nan={proc_nan} processed_logits_inf={proc_inf} "
            f"processed_logprobs_shape={tuple(processed_logprobs.shape) if processed_logprobs is not None else None} "
            f"processed_logprobs_finite={plp_finite}/{plp_total} "
            f"processed_logprobs_nan={plp_nan} processed_logprobs_inf={plp_inf} "
            f"gathered_logprobs_shape={tuple(gathered_logprobs.shape)} "
            f"gathered_logprobs_finite={glp_finite}/{glp_total} "
            f"gathered_logprobs_nan={glp_nan} gathered_logprobs_inf={glp_inf} "
            f"bad_raw_logits_rows={bad_rows(raw_logits)} "
            f"bad_raw_logprobs_rows={bad_rows(raw_logprobs)} "
            f"bad_processed_logits_rows={bad_rows(processed_logits)} "
            f"bad_processed_logprobs_rows={bad_rows(processed_logprobs)} "
            f"bad_gathered_logprobs_rows={bad_rows(gathered_logprobs)}",
            flush=True,
        )
        self._debug_logprobs_prints += 1

    @staticmethod
    def apply_temperature(
        logits: torch.Tensor,
        temp: torch.Tensor,
        all_random: bool,
    ) -> torch.Tensor:
        # Use in-place division to avoid creating a new tensor.
        # Avoid division by zero if there are greedy requests.
        if not all_random:
            temp = torch.where(temp < _SAMPLING_EPS, 1.0, temp)
        return logits.div_(temp.unsqueeze(dim=1))

    @staticmethod
    def greedy_sample(logits: torch.Tensor) -> torch.Tensor:
        return logits.argmax(dim=-1).view(-1)

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        logprobs_mode_override: LogprobsMode | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Sample logits based on sampling metadata.

        The various logits processing functions called in this method
        may update the logits tensor in-place.
        """

        logprobs_mode = logprobs_mode_override or self.logprobs_mode
        assert not (sampling_metadata.all_greedy and sampling_metadata.all_random)
        if sampling_metadata.all_random:
            greedy_sampled = None
        else:
            greedy_sampled = self.greedy_sample(logits)
            if sampling_metadata.all_greedy:
                processed_logprobs = None
                if sampling_metadata.max_num_logprobs is not None:
                    if logprobs_mode == "processed_logits":
                        processed_logprobs = logits
                    elif logprobs_mode == "processed_logprobs":
                        processed_logprobs = self.compute_logprobs(logits)
                return greedy_sampled, processed_logprobs

        assert sampling_metadata.temperature is not None

        # Apply temperature.
        logits = self.apply_temperature(
            logits, sampling_metadata.temperature, sampling_metadata.all_random
        )

        # Apply logits processors that only apply to random sampling
        # (argmax invariant)
        for processor in sampling_metadata.logitsprocs.argmax_invariant:
            logits = processor.apply(logits)

        # Apply top_k and/or top_p.
        random_sampled, processed_logprobs = self.topk_topp_sampler(
            logits,
            sampling_metadata.generators,
            sampling_metadata.top_k,
            sampling_metadata.top_p,
        )

        if greedy_sampled is None:
            return random_sampled, processed_logprobs

        sampled = torch.where(
            sampling_metadata.temperature < _SAMPLING_EPS,
            greedy_sampled,
            random_sampled,
            out=greedy_sampled,  # Reuse tensor
        )
        return sampled, processed_logprobs

    @staticmethod
    def compute_logprobs(logits: torch.Tensor) -> torch.Tensor:
        return logits.log_softmax(dim=-1, dtype=torch.float32)

    @staticmethod
    def gather_logprobs(
        logprobs: torch.Tensor,
        num_logprobs: int,
        token_ids: torch.Tensor,
    ) -> LogprobsTensors:
        """
        Gather logprobs for topk and sampled/prompt token.

        Args:
          logprobs: (num tokens) x (vocab) tensor
          num_logprobs: maximum number of logprobs to
                        retain per token
          token_ids: prompt tokens (if prompt logprobs)
                     or sampled tokens (if sampled
                     logprobs); 1D token ID tensor
                     with (num tokens) elements
                     Must be int64.

        Returns:
          Top-k int indices tensor, (num tokens) x (num_logprobs + 1)
          Top-k float logprobs tensor, (num tokens) x (num_logprobs + 1)
          Sampled token rank tensor, (num tokens)
        """
        assert token_ids.dtype == torch.int64
        # Find the topK values.
        topk_logprobs, topk_indices = torch.topk(logprobs, num_logprobs, dim=-1)

        # Get with the logprob of the prompt or sampled token.
        token_ids = token_ids.unsqueeze(-1)
        token_logprobs = logprobs.gather(-1, token_ids)

        # Compute the ranks of the actual token.
        token_ranks = batched_count_greater_than(logprobs, token_logprobs)

        # Concatenate together with the topk.
        indices = torch.cat((token_ids, topk_indices), dim=1)
        logprobs = torch.cat((token_logprobs, topk_logprobs), dim=1)

        # Use int32 to reduce the tensor size.
        indices = indices.to(torch.int32)

        return LogprobsTensors(indices, logprobs, token_ranks)

    @staticmethod
    def _combine_outputs_with_spec_tokens(
        output_token_ids: list[list[int]],
        spec_token_ids: list[list[int]] | None = None,
    ) -> list[list[int]]:
        if spec_token_ids is None:
            return output_token_ids

        return [
            [*out, *spec] if spec else out
            for out, spec in zip(output_token_ids, spec_token_ids)
        ]

    def apply_logits_processors(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        predict_bonus_token: bool,
    ) -> torch.Tensor:
        bad_words_token_ids = sampling_metadata.bad_words_token_ids
        any_penalties_or_bad_words = (
            bool(bad_words_token_ids) or not sampling_metadata.no_penalties
        )

        output_token_ids = sampling_metadata.output_token_ids
        if predict_bonus_token and any_penalties_or_bad_words:
            # Combine base outputs with spec tokens when speculative decoding
            # is enabled.
            output_token_ids = self._combine_outputs_with_spec_tokens(
                output_token_ids,
                sampling_metadata.spec_token_ids,
            )

        # Apply allowed token ids.
        if sampling_metadata.allowed_token_ids_mask is not None:
            logits.masked_fill_(sampling_metadata.allowed_token_ids_mask, float("-inf"))

        # Apply bad words exclusion.
        if bad_words_token_ids:
            apply_bad_words(logits, bad_words_token_ids, output_token_ids)

        # Apply logits processors which can impact greedy sampling.
        for processor in sampling_metadata.logitsprocs.non_argmax_invariant:
            logits = processor.apply(logits)

        # Apply penalties (e.g., freq_penalties).
        logits = self.apply_penalties(logits, sampling_metadata, output_token_ids)
        return logits

    @staticmethod
    def apply_penalties(
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        output_token_ids: list[list[int]],
    ) -> torch.Tensor:
        if sampling_metadata.no_penalties:
            return logits

        assert sampling_metadata.prompt_token_ids is not None
        return apply_all_penalties(
            logits,
            sampling_metadata.prompt_token_ids,
            sampling_metadata.presence_penalties,
            sampling_metadata.frequency_penalties,
            sampling_metadata.repetition_penalties,
            output_token_ids,
        )
