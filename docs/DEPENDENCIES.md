# Dependency and environment boundary

Python: CPython **3.12**, Linux. The original project metadata and paths are
source-checkout oriented. CPU and GPU dependency profiles remain separate; this
publication installs/runs no GPU environment and downloads no model.

| Direct requirement | Historical CPU version | Purpose |
| --- | --- | --- |
| torch | 2.13.0+cpu | Dense tensors and small Foundation neural modules |
| numpy | 2.5.2 | Retained original environment profile |
| safetensors | 0.8.0 | Explicit tensor checkpoint format |
| jsonschema | 4.26.0 | Contract/schema validation |
| pytest | 9.1.1 | Original tests |

`requirements-cpu.in` and `requirements-cpu.lock` retain the original exact
top-level/resolved specifications. The optional original GPU profile uses
`torch==2.13.0+cu130` and `nvidia-ml-py==13.610.43`; the corresponding specification
does not authorize a GPU qualification or establish portability to other devices.
Transitive dependencies remain pinned in the lock files and separately licensed.
Packages are imported, not redistributed; their supplied notices remain applicable.

The two basic checks have been run against the existing pinned project-local CPU
environment. A clean download/install and GitHub-hosted execution are separate
operations; local PASS is not a claim that a hosted workflow already ran.
No Transformers, M1, neural-memory, Bridge, ComfyUI or external-client dependency
is required by the basic CPU path. The generic GPU package imports without
initializing CUDA, but actual monitoring requires its separately installed NVML
binding and compatible host access.
