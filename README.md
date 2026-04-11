# hermes-mempalace-plugin

Development workspace for a Hermes Agent memory provider backed by MemPalace.

This repository now mirrors the final Hermes plugin directory layout:

```text
plugins/memory/mempalace/
```

The Hermes runtime currently discovers memory providers only from in-tree
`plugins/memory/<name>/` directories, so the production landing point is still
the Hermes repository. This repo is the implementation workspace that can be
copied into Hermes with minimal reshaping.

This repo is used to:

- capture architecture and delivery decisions
- implement the provider against the Hermes `MemoryProvider` contract
- run local tests against explicit path resolution, session scoping, write gating, and wing filtering
- keep migration notes clear before landing the code into Hermes

Key implementation files:

- `plugins/memory/mempalace/__init__.py` — provider implementation
- `plugins/memory/mempalace/plugin.yaml` — Hermes plugin metadata
- `plugins/memory/mempalace/cli.py` — `hermes mempalace ...` CLI hooks
- `plugins/memory/mempalace/README.md` — provider-specific setup and migration notes
- `tests/test_mempalace_provider.py` — local verification suite

See [docs/plan.md](docs/plan.md) for the planning history behind this workspace.
