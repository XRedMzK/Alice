# Foundation source publication and limits

The reusable implementation is the project-authored NMCM v2 Foundation. Its
unchanged code is in `src/nmcm` and `src/nmcm_gpu`; schemas/manifests/graphs and CPU
protocol are in `configs`, with generic GPU schemas in `gpu_configs`. The original
CPU test file and fixtures are included unchanged. No M1 package is imported.

## Accepted baseline

- [Module contract](../MODULE_CONTRACT.md), original baseline `v1alpha1`.
- [Frozen original architecture](FOUNDATION_ARCHITECTURE_V0.md).
- [Research protocol](../RESEARCH_PROTOCOL.md) and [resource policy](../RESOURCE_POLICY.md).
- [CPU plan](FOUNDATION_0_IMPLEMENTATION_PLAN.md), [CPU acceptance](FOUNDATION_0_CPU_ACCEPTANCE.md)
  and [target-GPU acceptance](FOUNDATION_0_GPU_ACCEPTANCE.md).

The historical CPU result reports **15 passed in 24.928 seconds**. The accepted
GPU attempt reports **15/15 PASS**, official wall time **32.141692678 seconds**.
These are historical results in their original synthetic scope, not results
rerun during this publication or evidence of a cognitive Alice. The CPU acceptance
descriptor is retained byte-for-byte. Its `runs/...` paths are immutable historical
bindings, not included payloads or active runtime dependencies of the basic checks.

## Source-only boundaries

The clean tree intentionally excludes original `runs`, learned checkpoints,
machine-bound GPU profile/acceptance descriptors, GPU qualification tests, private
work-package authority and environment outputs. Generic GPU code/schemas are
preserved without rewriting the original hardware policy to fit a new machine.
`nmcm_gpu.artifacts.load_accepted_cpu_artifacts` and the historical GPU qualification
runner require the excluded bound inputs; they are not usable as an admitted GPU
qualification from this clone alone. Obtain and verify the historical inputs or
prepare a separately scoped successor; never fabricate or weaken their hashes.

The [M1 research source archive](https://github.com/XRedMzK/M1-Event-Understanding/releases/tag/research-snapshot-20261024)
also preserves selected historical Foundation context. It is a separate immutable
source record, not a dependency to install here. This source publication does not
claim it supplies every original Foundation run or an automatic full replay.

The two basic tests need no accepted checkpoint, private path or old cache. They
cover syntax/semantics of manifests, module parameter counts, schemas, registry,
fixed graphs and routing. Test 02 performs a small synthetic forward pass. Other
original test functions train small modules and require deliberate invocation.

Linux CPython 3.12 is the tested basic environment. `resource` and POSIX checkpoint
semantics are used; Windows and packaged-wheel execution are not qualified. The
unchanged qualification CLI can generate fresh evidence and synthetic learning
when deliberately invoked; importing the packages does not launch a qualification.

## Historical reference interpretation

The retained old documents are evidence, not current architecture instructions.
Their original relative links, workload stops and authority filenames remain
unaltered. The new top-level ARCHITECTURE.md describes the owner-approved direction;
the exact old architecture is stored here as FOUNDATION_ARCHITECTURE_V0.md. The
public source provenance manifest records original/public hashes. This relocation
does not change any original accepted contract or threshold.
