# MemPalace Memory Provider

Local Hermes memory provider backed by the published `mempalace` Python package.

This implementation is designed to be copied into the Hermes monorepo as:

```text
plugins/memory/mempalace/
```

The current repository is only the development workspace. The final production
landing point remains the Hermes repository.

## What V1 Does

- stores completed turns as MemPalace drawers in the active wing
- keeps all provider paths explicit and profile-scoped by default
- mirrors Hermes built-in memory writes into the `builtin_memory` room
- exposes `mempalace_search`, `mempalace_kg_query`, and `mempalace_remember`
- keeps session caches keyed by `session_id` to avoid concurrent-session bleed
- blocks durable writes in `subagent`, `cron`, and `flush` contexts

## Default Paths

Unless overridden in `$HERMES_HOME/mempalace.json`, the provider resolves:

- `palace_path`: `$HERMES_HOME/mempalace/palace`
- `identity_path`: `$HERMES_HOME/mempalace/identity.txt`
- `kg_path`: `$HERMES_HOME/mempalace/knowledge_graph.sqlite3`
- `config_path`: `$HERMES_HOME/mempalace.json`

The provider always passes resolved explicit paths into MemPalace entrypoints. It
does not intentionally fall back to `~/.mempalace/*`.

## Setup

```bash
hermes memory setup
```

Or configure manually:

```bash
hermes config set memory.provider mempalace
```

Optional non-secret config file:

```json
{
  "palace_path": "/absolute/or/profile-relative/palace",
  "identity_path": "/absolute/or/profile-relative/identity.txt",
  "kg_path": "/absolute/or/profile-relative/knowledge_graph.sqlite3"
}
```

Relative config paths are resolved from the active `HERMES_HOME`.

## Tools

| Tool | Description |
|------|-------------|
| `mempalace_search` | semantic search over drawers in the active wing |
| `mempalace_kg_query` | query temporal facts from the profile-scoped KG |
| `mempalace_remember` | explicitly persist a durable memory drawer |

## CLI

```bash
hermes mempalace status
```

Shows the resolved storage paths for the active Hermes profile.

## Migration Into Hermes

When this workspace is ready to land:

1. Copy `plugins/memory/mempalace/` into the Hermes repository.
2. Keep the directory layout unchanged.
3. Re-run the Hermes-side memory plugin tests against the in-tree plugin.
