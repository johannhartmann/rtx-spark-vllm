#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Measure the three single-request, 512-token non-thinking code prompts.

Run against an already healthy local service. The JSON result includes the
generated text, so output validity can be reviewed alongside throughput.
"""

import argparse
import http.client
import json
import statistics
import time
from pathlib import Path


PROMPTS = (
    "Write a complete Python module implementing a thread-safe LRU cache with "
    "a per-entry TTL. Include docstrings and unittest-based tests.",
    "Write a TypeScript Express middleware that implements token-bucket rate "
    "limiting per API key backed by Redis. Include types and comments.",
    "Explain how to implement Dijkstra's algorithm with a binary heap in Go, "
    "then give the complete code with a worked example in main.",
)
OUTPUT_MARKERS = (
    ("lru", "ttl", "class"),
    ("redis", "token", "express"),
    ("dijkstra", "heap", "go"),
)
SPEC_COUNTERS = {
    "drafts": "vllm:spec_decode_num_drafts_total",
    "accepted": "vllm:spec_decode_num_accepted_tokens_total",
}


def counters(host, port):
    conn = http.client.HTTPConnection(host, port, timeout=30)
    conn.request("GET", "/metrics")
    response = conn.getresponse()
    if response.status != 200:
        raise RuntimeError(f"Metrics HTTP {response.status}")
    lines = response.read().decode().splitlines()
    conn.close()
    result = {key: 0.0 for key in SPEC_COUNTERS}
    for line in lines:
        if line.startswith("#"):
            continue
        for key, name in SPEC_COUNTERS.items():
            if line.startswith(name + "{") or line.startswith(name + " "):
                result[key] += float(line.rsplit(" ", 1)[1])
    return result


def request(host, port, model, prompt, max_tokens):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    conn = http.client.HTTPConnection(host, port, timeout=3600)
    start = time.perf_counter()
    conn.request("POST", "/v1/chat/completions", json.dumps(body),
                 {"Content-Type": "application/json"})
    response = conn.getresponse()
    if response.status != 200:
        raise RuntimeError(f"Completion HTTP {response.status}: {response.read()[:500]!r}")
    first = None
    usage = None
    finish = None
    output = []
    reasoning = []
    while line := response.readline():
        line = line.strip()
        if not line.startswith(b"data:"):
            continue
        data = line[5:].strip()
        if data == b"[DONE]":
            break
        chunk = json.loads(data)
        if chunk.get("usage"):
            usage = chunk["usage"]
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}
            content = delta.get("content") or ""
            thought = delta.get("reasoning_content") or delta.get("reasoning") or ""
            if first is None and (content or thought):
                first = time.perf_counter()
            output.append(content)
            reasoning.append(thought)
            finish = choice.get("finish_reason") or finish
    end = time.perf_counter()
    conn.close()
    return {
        "ttft_s": (first or end) - start,
        "decode_s": end - (first or end),
        "usage": usage or {},
        "finish_reason": finish,
        "content": "".join(output),
        "reasoning": "".join(reasoning),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--model", default="p16-local")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be positive")

    # Warm the request path, as in the original reference measurement.
    request(args.host, args.port, args.model, "Say hi.", 16)
    results = []
    for run in range(1, args.runs + 1):
        rows = []
        for index, prompt in enumerate(PROMPTS, 1):
            before = counters(args.host, args.port)
            sample = request(args.host, args.port, args.model, prompt, 512)
            after = counters(args.host, args.port)
            drafts = after["drafts"] - before["drafts"]
            accepted = after["accepted"] - before["accepted"]
            output_tokens = sample["usage"].get("completion_tokens", 0)
            decode_s = sample.pop("decode_s")
            row = {
                "prompt_id": index,
                "prompt": prompt,
                "prompt_tokens": sample["usage"].get("prompt_tokens", 0),
                "completion_tokens": output_tokens,
                "decode_s": round(decode_s, 4),
                "decode_tps": round((output_tokens - 1) / decode_s, 3)
                if output_tokens > 1 and decode_s > 0 else None,
                "drafts": drafts,
                "accepted": accepted,
                "acceptance_length": round(1 + accepted / drafts, 3)
                if drafts else None,
                # Wall time includes scheduler, draft, verify, and streaming.
                # This is not a CUDA kernel or GPU-only timing.
                "estimated_wall_step_ms": round(1000 * decode_s / drafts, 3)
                if drafts else None,
                **sample,
            }
            text = row["content"].lower()
            row["valid_output"] = (
                output_tokens == 512 and sample["finish_reason"] == "length"
                and not row["reasoning"]
                and all(marker in text for marker in OUTPUT_MARKERS[index - 1])
            )
            rows.append(row)
            print(f"run {run} prompt {index}: {row['decode_tps']} tok/s, "
                  f"wall step ~{row['estimated_wall_step_ms']} ms, "
                  f"valid={row['valid_output']}", flush=True)
        results.append({"run": run, "mean_decode_tps": round(statistics.mean(
            row["decode_tps"] for row in rows), 3), "prompts": rows})
        args.output.write_text(json.dumps(results, indent=2) + "\n")
    return 0 if all(row["valid_output"] for result in results
                    for row in result["prompts"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
