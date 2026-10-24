# Contributing

Start with the [architecture status](ARCHITECTURE.md) and
[Foundation scope](docs/FOUNDATION.md). Describe the concrete behavior your change
provides, why it is needed, affected interfaces, validation and limitations.

Run the two basic tests listed in [README.md](README.md) for Foundation changes.
They perform no training. Larger qualification or training work needs its own
bounded task and resource/evidence plan. Keep historical results immutable and
label new results with their actual scope. Reuse existing evidence when relevant
dependencies have not changed; do not turn every ordinary correction into a new
scientific qualification.

Do not silently change frozen Foundation schemas, thresholds or accepted evidence.
Version intentional contract changes and explain consumer/state migration risks.
Do not present planned memory, Bridge, ComfyUI, Collab or cognitive properties as
implemented. M1 contributions belong in its [separate project](https://github.com/XRedMzK/M1-Event-Understanding).

Contribute only material you have the right to publish. Project-owned contributions
are submitted under [MIT](LICENSE); preserve separate upstream notices and identify
reused code, data/model terms and provenance. Do not commit model weights, caches,
training runs, credentials, private logs, personal paths or conversations. Report
sensitive findings privately to the maintainer instead of posting the secret.

Automated CI is intentionally limited to unprivileged CPU basic tests. GPU access,
downloads of model assets and cognitive training are not required to contribute
documentation or ordinary engineering fixes. Windows compatibility is not
qualified; do not bypass exactness/ownership gates to obtain a platform PASS.
