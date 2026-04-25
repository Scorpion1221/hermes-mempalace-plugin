# hermes-mempalace-plugin (ARCHIVED — moved 2026-04-25)

**This repository has been merged into the main MemPalace repo** at
[`Scorpion1221/mempalace`](https://github.com/Scorpion1221/mempalace)
under `integrations/hermes/`.

The Hermes plugin imports directly from `mempalace.recall_llm`,
`mempalace.palace`, and `mempalace.hooks_cli`, so every cross-cutting
change had to land atomically on both repos. Keeping them separate
made coordinated refactors error-prone. They now share one commit,
one push, one deploy.

## Where to go

- **Plugin code**: `~/git/mempalace/integrations/hermes/plugins/memory/mempalace/`
- **Tests**: `~/git/mempalace/integrations/hermes/tests/`
- **Deploy**: `bash ~/git/mempalace/scripts/sync-plugins.sh`

## Old content

The files in this repo are the state at commit `c6b7f80` (2026-04-25).
They stay here as a historical reference only — all new work happens in
the merged repo.

---

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
- stage runtime rollouts when the Hermes in-tree plugin is not symlinked to this repo

Key implementation files:

- `plugins/memory/mempalace/__init__.py` — provider implementation
- `plugins/memory/mempalace/plugin.yaml` — Hermes plugin metadata
- `plugins/memory/mempalace/cli.py` — `hermes mempalace ...` CLI hooks
- `plugins/memory/mempalace/README.md` — provider-specific setup and migration notes
- `tests/test_mempalace_provider.py` — local verification suite

See [docs/plan.md](docs/plan.md) for the planning history behind this workspace.

## Current recall behavior

The Hermes MemPalace provider now carries forward the **previous
assistant reply** as structured recall context:

- `sync_turn()` stores the cleaned assistant reply in session memory
- the same reply is also written to a session cache file under the
  active Hermes profile so gateway restarts can recover it
- recall uses the tail of that reply (500 chars) when rewriting and
  reranking the next user query

This makes short follow-ups like “why?” and “continue” recall the right
MemPalace drawers without relying on transcript scraping at recall time.
