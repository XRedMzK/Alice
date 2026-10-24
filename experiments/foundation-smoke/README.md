# Foundation basic-check protocol

Purpose: check that the source-only publication can validate original manifests,
schemas, registry/graph structure and semantics-blind routing without M1 assets,
checkpoints, a donor, CUDA or optimization steps. This is ordinary engineering
validation, not a new cognitive experiment or full Foundation qualification.

Precondition: Linux CPython 3.12 and `requirements-cpu.lock` in a separate environment.
Run from the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=2 \
  .venv/bin/python -B -m pytest -q -p no:cacheprovider \
  tests/test_foundation0_cpu.py::test_01_manifest_and_schema_validation \
  tests/test_foundation0_cpu.py::test_02_registry_graph_and_semantics_blind_routing
```

PASS: exactly these two original tests pass, with no failure/error/skip. Test 02
performs a small synthetic forward pass. It uses no learning fixture. The tests
and acceptance thresholds have not been rewritten. Historical CPU-03…15 and GPU
qualification are outside this check; no inference about memory or Bridge follows.
