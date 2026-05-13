"""
Trigger one torch.profiler capture on a running vLLM server.

Assumes the server was launched with:
    vllm serve ... --profiler torch \
                   --torch-profiler-dir <abs-path> \
                   --delay-iterations <N>  --max-iterations <M>

After /start_profile is called, vLLM's profiler will skip `delay_iterations`
warm-up iterations, capture the next `max_iterations`, and auto-stop (no need
to /stop_profile, but we still call it as a safety net). The trace lands in
<torch_profiler_dir>/<pid>_<hostname>_<ts>.json{,.gz} per worker process.

Usage:
    python3 profile_one_burst.py \
        --base-url http://localhost:8000 \
        --model /lustre/.../mxfp8 \
        --num-requests 64 --max-tokens 256 --concurrency 64
"""
from __future__ import annotations

import argparse
import asyncio
import time

import aiohttp


async def _one(session: aiohttp.ClientSession, base_url: str, model: str,
               prompt: str, max_tokens: int, timeout_s: float) -> int:
    """Fire one completion request and return tokens generated."""
    url = f"{base_url.rstrip('/')}/v1/completions"
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
    }
    try:
        async with session.post(
            url, json=payload, timeout=aiohttp.ClientTimeout(total=timeout_s)
        ) as r:
            if r.status != 200:
                return 0
            data = await r.json()
            usage = (data or {}).get("usage") or {}
            return int(usage.get("completion_tokens") or 0)
    except Exception:
        return 0


async def _post(session: aiohttp.ClientSession, url: str) -> int:
    async with session.post(url) as r:
        return r.status


async def main_async(args: argparse.Namespace) -> None:
    base = args.base_url.rstrip("/")
    timeout_s = float(args.timeout_s)
    connector = aiohttp.TCPConnector(limit=max(64, args.concurrency * 2))
    async with aiohttp.ClientSession(connector=connector) as session:
        # Warm-up so the profiler isn't catching cold-start overhead.
        if args.warmup > 0:
            print(f"[warmup] sending {args.warmup} requests")
            await asyncio.gather(*(
                _one(session, base, args.model, args.warmup_prompt,
                     args.warmup_max_tokens, timeout_s)
                for _ in range(args.warmup)
            ))

        # Trigger profile start.
        print("[profile] POST /start_profile")
        status = await _post(session, f"{base}/start_profile")
        print(f"           status={status}")

        # Drive the burst.
        prompts = [args.prompt + f" (req {i})" for i in range(args.num_requests)]
        sem = asyncio.Semaphore(args.concurrency)

        async def go(p: str) -> int:
            async with sem:
                return await _one(session, base, args.model, p,
                                  args.max_tokens, timeout_s)

        print(f"[burst]   {args.num_requests} reqs × max_tokens={args.max_tokens} "
              f"concurrency={args.concurrency}")
        t0 = time.perf_counter()
        tokens = await asyncio.gather(*(go(p) for p in prompts))
        elapsed = time.perf_counter() - t0
        total_tok = sum(tokens)
        print(f"[burst]   elapsed={elapsed:.2f}s tokens={total_tok} "
              f"throughput={total_tok / elapsed:.1f} tok/s")

        # Trigger stop (safety net — profiler may already have auto-stopped).
        print("[profile] POST /stop_profile")
        status = await _post(session, f"{base}/stop_profile")
        print(f"           status={status}")

        print("\nTrace files land under the --torch-profiler-dir given to "
              "`vllm serve`. Pull them off the cluster and run "
              "summarize_profile.py.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True,
                    help="value of `model` for the OpenAI completions API "
                         "(usually the served-model path)")
    ap.add_argument("--num-requests", type=int, default=64)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--prompt", default="Write a short paragraph about silicon.")
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--warmup-prompt", default="Hello.")
    ap.add_argument("--warmup-max-tokens", type=int, default=16)
    ap.add_argument("--timeout-s", type=float, default=300.0)
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
