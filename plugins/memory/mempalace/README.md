# MemPalace Memory Provider v2.0

Local Hermes memory provider backed by the published `mempalace>=3.3.0` Python package.

This implementation is designed to be copied into the Hermes monorepo as:

```text
plugins/memory/mempalace/
```

The current repository is only the development workspace. The final production
landing point remains the Hermes repository.

## What V2 Does

- stores completed turns as MemPalace drawers in the active wing
- keeps all provider paths explicit and profile-scoped by default
- mirrors Hermes built-in memory writes into the `builtin_memory` room
- exposes **31 tools** covering search, knowledge graph, palace graph, drawers, diary, and more
- keeps session caches keyed by `session_id` to avoid concurrent-session bleed
- blocks durable writes in `subagent`, `cron`, and `flush` contexts
- `on_pre_compress` hook saves recent turns before context compression discards them
- dispatch-table architecture for tool handling (no if/elif chains)
- `_safe_content()` helper using `sanitize_content` from mempalace
- `_bounded_int()` helper for clamped integer parsing
- `_get_collection(create=False)` returns `None` instead of raising when collection doesn't exist
- `_json_result` uses `default=str` for safe datetime serialization
- carries forward the previous assistant reply as recall context for the next turn
- uses memory-first + file-fallback session caches for previous assistant context

## Default Paths

Unless overridden in `$HERMES_HOME/mempalace.json`, the provider resolves:

- `palace_path`: `$HERMES_HOME/mempalace/palace`
- `identity_path`: `$HERMES_HOME/mempalace/identity.txt`
- `kg_path`: `$HERMES_HOME/mempalace/knowledge_graph.sqlite3`
- `config_path`: `$HERMES_HOME/mempalace.json`

The provider always passes resolved explicit paths into MemPalace entrypoints. It
does not intentionally fall back to `~/.mempalace/*`.

Previous-assistant context cache files live under the profile-scoped base dir:

- `$HERMES_HOME/mempalace/session_state/<slug>_<hash>_last_assistant.txt`

The runtime flow is:

1. `sync_turn()` stores the assistant reply in `SessionState.last_assistant_reply`
2. the same reply is written to the session-state file
3. recall reads memory first, then the file cache if the process restarted
4. `on_session_end()` deletes the cache file, while bare process shutdown leaves
   it in place so the next process can recover continuity

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

## Tools (31 total)

### Search & Recall
| Tool | Description |
|------|-------------|
| `mempalace_search` | Semantic search over drawers in the active wing |
| `mempalace_kg_query` | Query temporal facts from the profile-scoped KG |
| `mempalace_remember` | Explicitly persist a durable memory drawer |

Recall for normal turns uses:

- current user message as the primary signal
- previous assistant reply tail (500 chars) as optional structured context

Short follow-up prompts like “why?” and “continue” are allowed into recall when
previous assistant context exists; pure acknowledgements like “ok” still skip.

### Knowledge Graph
| Tool | Description |
|------|-------------|
| `mempalace_kg_add` | Add a triple (subject, predicate, object) to the KG |
| `mempalace_kg_invalidate` | End-date a triple in the KG |
| `mempalace_kg_timeline` | Temporal timeline of an entity |
| `mempalace_kg_stats` | KG statistics |

### Palace Status & Taxonomy
| Tool | Description |
|------|-------------|
| `mempalace_status` | Current provider status, paths, wing info |
| `mempalace_list_wings` | List all wings with counts |
| `mempalace_list_rooms` | List rooms, optionally filtered by wing |
| `mempalace_get_taxonomy` | Full wing/room taxonomy with counts |

### Palace Graph (Tunnels)
| Tool | Description |
|------|-------------|
| `mempalace_traverse` | Graph traversal from a starting room |
| `mempalace_find_tunnels` | Find tunnels connecting two wings |
| `mempalace_graph_stats` | Palace graph statistics |
| `mempalace_create_tunnel` | Create a tunnel between rooms |
| `mempalace_list_tunnels` | List tunnels, filtered by wing |
| `mempalace_delete_tunnel` | Delete a tunnel by ID |
| `mempalace_follow_tunnels` | Follow tunnels from a wing/room |

### Drawer Management
| Tool | Description |
|------|-------------|
| `mempalace_add_drawer` | Add a new drawer |
| `mempalace_delete_drawer` | Delete a drawer by ID |
| `mempalace_get_drawer` | Retrieve a specific drawer |
| `mempalace_list_drawers` | List drawers with optional filters |
| `mempalace_update_drawer` | Update drawer content |

### Diary
| Tool | Description |
|------|-------------|
| `mempalace_diary_write` | Write a diary entry (appends to same-day) |
| `mempalace_diary_read` | Read recent diary entries |

### Utilities
| Tool | Description |
|------|-------------|
| `mempalace_check_duplicate` | Check if content already exists |
| `mempalace_check_facts` | Fact-check text against stored knowledge |
| `mempalace_hook_settings` | Get/set hook settings |
| `mempalace_reconnect` | Reconnect ChromaDB client |
| `mempalace_get_aaak_spec` | AAAK specification |
| `mempalace_memories_filed_away` | Count of memories filed this session |

## Hooks

| Hook | Description |
|------|-------------|
| `on_turn_start` | Set up session state for the new turn |
| `on_session_end` | Flush queued writes |
| `on_memory_write` | Mirror built-in memory writes |
| `on_pre_compress` | Save recent turns before context compression |

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
