# P16 Qwen3.8-27B performance record, 2026-11-17

## Reproduction inputs

The machine is an ASUS ProArt P16 H7607BA with one NVIDIA RTX Spark N1X.
The running kernel was `7.2.8-36-rtx-spark-npu` on AArch64, with the
`615.78.08` NVIDIA open kernel module and CUDA 13.4 userspace driver. The
selected ACPI power profile was `performance`, AC power was connected, and
normal desktop applications stayed open. No physical EC reset, power-profile
change, GPU clock request, or driver replacement was used for these results.

The target is `unsloth/Qwen3.8-27B-NVFP4` at revision
`f0b7c9e722f5565102fff8481c99e4d86ae099c7`, prepared as the pinned
sharded checkpoint. DFlash2 is `z-lab/Qwen3.8-27B-DFlash2` at revision
`50307d4c4cde6860d4eee73e2547cd786fe8e8a4`. The draft weight SHA-256
is `67fc76d68dc5a9415511a4f394ef744d67510cd20e93b37cc2cc7d28e4bab65c`.
The native runtime is vLLM `0.33.0rc1.dev1011+g57d8e025e.p16.1`, PyTorch
`2.14.0+cu130`, and FlashInfer `0.7.3`. The target retains its mixed
NVFP4/FP8 checkpoint; the draft uses online `fp8_per_tensor` and the KV
cache uses FP8.

The tested service uses Model Runner V2, dynamic NVFP4 with both qualified
A16 cutoffs at zero, CUTLASS FP8, cuDNN BF16, FlashInfer autotuning, eight
probabilistically sampled DFlash2 draft tokens, standard rejection sampling,
and the launcher's published graph, cache, and batching settings. The kernel
qualification passed numerical checks before model startup. The user service
was disabled for boot autostart.

## Independent GPU control

With vLLM stopped, the saved 8192³ cuBLAS GemmEx control used FP16 inputs and
outputs with FP32 accumulation, 384 MiB of matrix allocations, and checked
sampled results in every window. Its median GPU throughput, excluding the
first window, was **78.28 TFLOP/s** across 22 windows. The older slow run on
kernel `7.2.8-23-rtx-spark-npu` measured 4.65 TFLOP/s without vLLM and
10.56 decode tokens/s on the three code prompts. The current control is close
to the recorded earlier healthy 75.94 TFLOP/s run. These are separate-session
comparisons, so they establish current recovery rather than an EC or kernel
causal explanation. The physical EC reset proposed for a persistently slow
control was unnecessary.

## Matched full-model results

Each pass streamed the same three code prompts in order: Python TTL LRU
cache, TypeScript/Redis token bucket middleware, and Go Dijkstra with a
binary heap. Each request used 512 output tokens, one concurrent request,
temperature zero, and `enable_thinking=false`. One short request warmed the
server before measurement. Decode rate uses the 511 tokens after the first
response token divided by elapsed time after the first token. Time to first
token is recorded separately. Outputs were nonempty, on topic, and had no
reasoning content; 512-token responses ended at the requested length.

| DFlash2 profile | Pass | Python | TypeScript | Go | Mean decode tok/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| 7 tokens, probabilistic | 1 | 44.22 | 41.41 | 41.65 | 42.43 |
| 7 tokens, probabilistic | 2 | 43.93 | 41.28 | 39.17 | 41.46 |
| 7 tokens, probabilistic | 3 | 41.17 | 38.89 | 38.87 | 39.64 |
| 7 tokens, greedy | 1 | 41.67 | 38.81 | 39.05 | 39.84 |
| 7 tokens, greedy | 2 | 41.55 | 38.95 | 39.03 | 39.84 |
| 7 tokens, greedy | 3 | 41.55 | 38.89 | 38.98 | 39.81 |
| 5 tokens, probabilistic | 1 | 36.11 | 34.29 | 33.88 | 34.76 |
| 5 tokens, probabilistic | 2 | 35.79 | 34.21 | 34.00 | 34.67 |
| 5 tokens, probabilistic | 3 | 36.07 | 34.32 | 33.80 | 34.73 |
| **8 tokens, probabilistic** | **1** | **42.74** | **40.86** | **39.59** | **41.06** |
| **8 tokens, probabilistic** | **2** | **42.93** | **40.73** | **39.46** | **41.04** |
| **8 tokens, probabilistic** | **3** | **42.32** | **40.73** | **39.84** | **40.97** |

For the selected eight-token profile, the three prompts' speculative
acceptance lengths were 5.340, 5.059, and 4.876 tokens per draft. The
benchmark script's estimated wall times per draft/verification step ranged
from 122.2 to 124.5 ms across the nine requests. This wall measure includes
scheduling, draft, verification, and response streaming; it is not a
GPU-only kernel duration. During the selected profile's measured window,
reported SM clocks had a median of 2320 MHz, GPU temperature ranged from
69 to 74 °C, and reported GPU power had a median of 27.49 W. Unified-memory
GPU memory capacity and usage were unavailable through `nvidia-smi`; host
available memory was about 22 GiB at the end of the final installed run.

A separate PyTorch/CUPTI trace profiled one 512-token Python request with the
same installed eight-token profile. Its 98 GPU `generation_1(9)` annotations
had a **103.92 ms median GPU step duration** (mean 105.59 ms, sampled 10th to
90th percentile 97.47–113.28 ms). Those 98 GPU intervals totaled 10.35 s;
the profiled HTTP request took 12.70 s. This is one instrumented request,
not the uninstrumented nine-request throughput sample. The trace is stored
outside Git at
`/home/johann/p16/benchmarks/qwen38-27b-vllm/plan-gpu-profile-20261117/`.

The selected profile exceeded the **40 decode tokens/s pass-mean target in
three consecutive passes**. One individual Go prompt remained below 40; the
plan's threshold applies to each three-prompt pass mean.

After installing the profile through `install.sh config`, the ordinary service
was started without a runtime override. A second, stricter three-pass check
required 512 output tokens, a length finish, no reasoning text, and topic
markers in each saved response. It gave:

| Installed pass | Python | TypeScript | Go | Mean decode tok/s |
| --- | ---: | ---: | ---: | ---: |
| 1 | 43.60 | 41.38 | 40.76 | **41.91** |
| 2 | 42.71 | 40.94 | 38.98 | **40.88** |
| 3 | 41.73 | 39.41 | 39.12 | **40.09** |

All nine response checks passed. The third pass has little margin above the
threshold. In this run, reported SM clocks had a median near 2250 MHz, GPU
temperature remained between 61 and 70 °C, and median reported GPU power
was about 23.6 W. The service remained active with autostart disabled and
no runtime drop-in. The results are saved outside Git at
`/home/johann/p16/benchmarks/qwen38-27b-vllm/plan-final-20261117-171613/`.

## Driver warning and measurement limits

No NVRM, Xid, or GSP fault was logged during any of the measured prompt
windows. Some model starts emitted `NV_ERR_NO_MEMORY` from the NVIDIA module's
`_memdescAllocInternal` path, despite successful later model startup and
available host memory. The first guarded start was intentionally stopped by
the telemetry watcher on those warnings. The first selected eight-token start
logged seven such warnings before serving; the final installed start logged
two. They are unresolved; the
requirement for **no new NVIDIA errors across startup and operation** is
therefore not fully met. No Xid or GSP timeout was observed. The separate
GPU trace above provides a device-side step duration; it must not be confused
with the wall-step estimate.

## Online NVFP4 weight-only drafter A/B

On 2026-11-17, an isolated copy of the installed vLLM wheel was patched with
[vLLM #67006](https://github.com/vllm-project/vllm/pull/67006) (downloaded PR
patch SHA-256 `331e1c610ae67c0894796ce46dabed32ef3c35f6605fae9138f856e5868f5092`).
The three code commits applied to the wheel copy; the fourth, a docstring-only
change, was already present in the local code. The patched package was loaded
through `PYTHONPATH`, leaving the installed wheel and service files untouched.
The same pinned target and drafter, eight draft tokens, model settings, power
profile, and benchmark script were used. The drafter's online quantization
changed from `fp8_per_tensor` to `nvfp4_weight_only`. Because the existing
global `nvfp4_dynamic` backend rejects an online W4A16 drafter, the candidate
also set `linear_backend_per_quant.nvfp4_w4a16=marlin`. The first candidate
start without that override failed before serving; the measured candidate
started with Marlin for W4A16 and Dynamic NVFP4 for the target.

| Arm | Pass 1 | Pass 2 | Pass 3 | Nine-request mean | Mean acceptance length |
| --- | ---: | ---: | ---: | ---: | ---: |
| Installed FP8-per-tensor drafter | 42.219 | 42.275 | 41.919 | **42.138 tok/s** | 5.092 |
| #67006 NVFP4 weight-only drafter | 36.182 | 35.928 | 36.120 | **36.077 tok/s** | 4.524 |

The NVFP4 candidate was **14.4% slower** on this workload. Its Go-prompt
acceptance length fell from 4.876 to 3.909; the mean wall time per draft step
changed from 120.2 to 124.1 ms. All 18 measured responses had 512 output
tokens, a `length` finish, no reasoning text, and the prompt-specific topic
markers. These checks are not a full quality evaluation. vLLM reported 21.68
GiB model-load memory for the candidate versus 22.42 GiB for the earlier
installed start, but unified-memory availability differed between starts.
Three already-known `NV_ERR_NO_MEMORY` warnings appeared during candidate
startup; none appeared during its measured requests, and there was no Xid or
GSP error. The FP8-per-tensor service was restored after the A/B; one further
three-prompt control pass measured 41.067 tok/s with the original acceptance
lengths (5.340, 5.059, 4.876) and three valid responses. Raw responses,
per-request metrics, candidate log, and the experimental launcher
are under `/home/johann/p16/benchmarks/qwen38-27b-vllm/nvfp4-weight-only-20261117/`.
The NVFP4 drafter is not selected for the maintained service.

## Reproduce

Use the pinned checkpoint and native runtime from the [setup guide](vllm-qwen38.md).
Install the repository's configuration on the existing `main` branch while
the service is stopped; the installer backs up user configuration and leaves
boot autostart disabled:

```sh
bash config/vllm-qwen38/install.sh check
bash config/vllm-qwen38/install.sh config
systemctl --user start vllm-qwen38.service
systemctl --user is-enabled vllm-qwen38.service
python3 tests/benchmark_code.py --runs 3 --output /path/to/results.json
```

`tests/benchmark_code.py` saves every output and per-prompt token count,
time to first token, decode rate, acceptance length, and estimated wall step
time. Review the saved text and startup log. Record `uname -r`,
`nvidia-smi`, `powerprofilesctl get`, and `journalctl -k -b` in the same
session. The cuBLAS control source is in the P16 installation repository's
`experiments/gpu-syspl3/gemm-stream.cu`; run it only with vLLM stopped.
