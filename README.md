# Alice

Alice is an engineering project for a functioning modular cognitive system,
informed by research and existing technical solutions. Its direction is an independently owned, changeable
cognitive state and acquired experience that can survive replacement of a language
backbone. **The cognitive runtime and its new abilities remain planned.**

> Use science to design the architecture; use engineering evidence to determine
> whether Alice works.

## What is here

The working implementation is the small **NMCM v2 Foundation**: versioned neural
module contracts, a registry and graph runner, independent state/checkpoints,
compatibility checks, telemetry and a separate GPU control plane. The synthetic
A1/A2/A3/B/C modules exercise those boundaries; they are not a neural memory or an
Alice language runtime. Historical Foundation CPU and target-GPU qualification
each passed 15 gates within their original bounded scope.

The approved next direction uses a **frozen, replaceable Qwen3.5-2B-Base** for
language processing and response formation, a **planned independent plastic
neural memory**, and a **planned replaceable trainable Memory Bridge**. ComfyUI
is the chosen future external visual graph/UI/API layer. These integrations are
**not implemented in this repository**. See [ARCHITECTURE.md](ARCHITECTURE.md) for
the precise boundaries and [ROADMAP.md](ROADMAP.md) for proposed functional work.

[Engineering First Policy v2](docs/ENGINEERING_FIRST_POLICY_V2_RU.md) is active:
new cognitive functions must improve useful runtime behavior in matched MODULE
OFF / MODULE ON checks. There is no mandatory M1-style scientific qualification
or separate network for each historical M0–M13 function. The replacement roadmap
awaits owner approval; this PR starts no training, downloads or implementation.
Start with [CODEX.md](CODEX.md), the [engineering contract](ENGINEERING_MODULE_CONTRACT.md)
and [validation rules](ENGINEERING_VALIDATION.md). [Policy scope](docs/POLICY_SCOPE.md)
separates active rules from the unchanged Foundation research records.

## Basic local check

Use CPython 3.12 on Linux. From the repository root:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-cpu.lock
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=2 \
  .venv/bin/python -B -m pytest -q -p no:cacheprovider \
  tests/test_foundation0_cpu.py::test_01_manifest_and_schema_validation \
  tests/test_foundation0_cpu.py::test_02_registry_graph_and_semantics_blind_routing
```

These two unchanged tests validate manifests/schemas and registry/graph/routing
with a small synthetic forward pass. They do not train, download models, start
CUDA or use M1 data. The rest of the original CPU suite is supplied for inspection
and separate deliberate reproduction; several of its fixtures **do train small
synthetic modules**. It is not part of the basic check or default CI job.

The source checkout is the supported layout; schemas are resolved relative to
that layout. No standalone wheel portability, Windows runtime qualification or
fresh full Foundation qualification is claimed. See [the Foundation guide](docs/FOUNDATION.md)
and [the small protocol](experiments/foundation-smoke/README.md).

## M1 research is separate

M1 is **RESEARCH ARCHIVED / NOT_FROZEN**, remains open to external contributions,
and is no longer required active research for Alice. Its results and limitations
are preserved in [M1-Event-Understanding](https://github.com/XRedMzK/M1-Event-Understanding)
and the [24 October 2026 research release](https://github.com/XRedMzK/M1-Event-Understanding/releases/tag/research-snapshot-20261024).
M1 is not an active runtime dependency of this source tree. Archiving it does not
turn its failed or incomplete qualifications into PASS.

Project-owned source and documentation use [MIT](LICENSE). Dependencies and
separately obtained model assets retain [their own terms](THIRD_PARTY_NOTICES.md).
No pretrained weights, datasets, historical runs, private logs or conversations
are distributed here. Contributions: [CONTRIBUTING.md](CONTRIBUTING.md).
