# Native vLLM profile for RTX Spark

Pinned ARM64 vLLM setup for `unsloth/Qwen3.8-27B-NVFP4` on RTX Spark N1X,
including DFlash2/MTP launch profiles, SM12x Python backports, checkpoint
preparation, configuration checks and a user systemd service.

The [setup guide](docs/vllm-qwen38.md) documents dependencies, immutable input
revisions, installation stages and qualification limits. The
[November 2026 performance record](docs/performance-2026-11-17.md) contains
full-model output checks, matched decode measurements and remaining driver
warnings on the P16.

## Setup

```sh
git clone https://github.com/johannhartmann/rtx-spark-vllm.git
cd rtx-spark-vllm
```

Follow the [installation guide](docs/vllm-qwen38.md#installation-without-starting-the-service)
to prepare the environment and checkpoints. The installer stages are
`runtime`, `models`, `check` and `config`. The service is installed stopped
and disabled; starting a serving workload is a separate operation.

The existing `config/vllm-qwen38/` layout, service name and pinned `.p16.1`
wheel version are retained so extraction does not change the installation
interface or invalidate the recorded build/checkpoint hashes.

## Run the qualified P16 profile

The measured profile uses eight DFlash2 draft tokens with probabilistic
drafting. The installer preserves an existing `~/.config/vllm-qwen38.env`, so
check that it contains `VLLM_DRAFT_SAMPLE_METHOD=probabilistic` before running
the service. The [setup guide](docs/vllm-qwen38.md) covers the pinned models,
runtime, and initial installation.

From this repository on `main`, with the service stopped:

```sh
bash config/vllm-qwen38/install.sh check
bash config/vllm-qwen38/install.sh config
systemctl --user start vllm-qwen38.service
curl -fsS http://127.0.0.1:8080/health
python3 tests/benchmark_code.py --runs 3 --output /path/to/results.json
systemctl --user is-enabled vllm-qwen38.service
```

`config` installs the launcher and user unit, leaves the service stopped, and
disables boot autostart; the explicit `start` command is required afterward.
`is-enabled` should report `disabled`. Use
`systemctl --user stop vllm-qwen38.service` to stop serving. The
[P16 performance record](docs/performance-2026-11-17.md) gives exact measured
rates, GPU step timing, telemetry, and the remaining startup driver warning.

## Tests

Run the offline configuration tests without model weights or a GPU:

```sh
python3 -m unittest discover -s tests -p test_vllm_configuration.py -v
bash -n config/vllm-qwen38/install.sh config/vllm-qwen38/serve.sh
```

Two optional CPU tensor/backport checks require `VLLM_TEST_PYTHON` to point to
an existing Python environment with the matching dependencies. Kernel and
serving qualification are described separately in the setup guide; passing
configuration tests does not establish working inference.

## Origin and licenses

Extracted from
[`asus-proart-p16-rtx-spark-linux`](https://github.com/johannhartmann/asus-proart-p16-rtx-spark-linux)
with the path-specific Git history and original notices retained.
[`ORIGIN.json`](ORIGIN.json) records the public source commit, original paths
and blob identities. The 18 profile/documentation/test files were transferred
without content changes. The original MIT license moved to [`LICENSE`](LICENSE).

Helpers, configuration, tests and documentation retain their original MIT
license. The backport derives from upstream vLLM code; see
[`NOTICE`](NOTICE), [`licenses/vllm-Apache-2.0.txt`](licenses/vllm-Apache-2.0.txt)
and the pinned upstream references in
[`build-profile.json`](config/vllm-qwen38/build-profile.json).
