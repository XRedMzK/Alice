# Rights, provenance and third-party dependencies

Original project code and documentation in this source publication are offered
under [MIT](LICENSE), copyright 2026 XRedMzK. They come from the owner's NMCM v2
Foundation workspace and clean publication work authorized by the owner. The
earlier [research snapshot](https://github.com/XRedMzK/M1-Event-Understanding/releases/tag/research-snapshot-20261024)
already licensed selected unchanged Foundation source under the same project MIT
notice. No third-party repository implementation, dataset, wheel or pretrained
weight is vendored here. A file-level origin/hash manifest accompanies this tree.

Installed dependencies retain their own rights. Direct historical environment
profiles and resolved locks are preserved; their presence is not a relicensing of
any installed distribution. Source-license references are:

| Dependency | Upstream source license | Reference |
| --- | --- | --- |
| PyTorch | BSD-3-Clause; distributed wheels include additional licenses | [Official source LICENSE](https://github.com/pytorch/pytorch/blob/v2.13.0/LICENSE) |
| NumPy | BSD-3-Clause; distributions include additional licenses | [Official source LICENSE](https://github.com/numpy/numpy/blob/v2.5.2/LICENSE.txt) |
| safetensors | Apache-2.0 | [Official source LICENSE](https://github.com/huggingface/safetensors/blob/v0.8.0/LICENSE) |
| jsonschema | MIT | [Official source COPYING](https://github.com/python-jsonschema/jsonschema/blob/v4.26.0/COPYING) |
| pytest (tests) | MIT | [Official source LICENSE](https://github.com/pytest-dev/pytest/blob/9.1.1/LICENSE) |
| nvidia-ml-py (optional GPU) | BSD | [Upstream package](https://pypi.org/project/nvidia-ml-py/13.610.43/) |

NumPy, PyTorch and CUDA distributions can bundle other code and NVIDIA software
under additional terms. Preserve all notices shipped with distributions when
redistributing them. This repo distributes only dependency specifications, not
those binaries. See [DEPENDENCIES.md](docs/DEPENDENCIES.md) for tested versions and
environment limits.

The future language/response donor is `Qwen/Qwen3.5-2B-Base`; the identified
historical model revision is `b1485b2fa6dfa1287294f269f5fb618e03d52d7c`.
Its [upstream Apache-2.0 license](https://huggingface.co/Qwen/Qwen3.5-2B-Base/blob/b1485b2fa6dfa1287294f269f5fb618e03d52d7c/LICENSE)
governs separately obtained assets. No Qwen/tokenizer assets or Alice language
runtime are included; no rights over its upstream pretraining data are claimed.

ComfyUI is a selected future external application, not a vendored or linked
implementation in this tree. Any future custom nodes, dependency distribution
or integration must retain the relevant upstream terms. This publication makes
no license claim over external ComfyUI or future client software.
