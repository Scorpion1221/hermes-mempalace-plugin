# Hermes MemPalace Plugin Plan

## Goal

Build a Hermes Agent external memory provider named `mempalace` that uses the
published `mempalace` Python package for local, profile-scoped long-term memory.

The provider should give Hermes:

- cross-session recall
- local semantic search over prior turns
- optional factual retrieval through MemPalace's temporal knowledge graph
- explicit write paths for durable memories
- zero cloud dependency in the default path

## Key Constraints

### 1. Hermes memory providers are in-tree, not externally installed

Hermes currently discovers memory providers by scanning `plugins/memory/<name>/`
inside the Hermes repository. It does not load external pip packages as memory
providers.

Implication:

- this repo is a development workspace only
- the final provider must be implemented as `plugins/memory/mempalace/` inside Hermes
- if we want true external memory-provider loading later, that requires a separate Hermes platform change

### 2. Do not bridge through MCP for the provider runtime

MemPalace already exposes an MCP server, but the Hermes memory-provider contract is
native Python and in-process. Running MemPalace through MCP inside a Hermes memory
provider would add avoidable complexity:

- extra subprocess lifecycle management
- duplicated tool schemas
- slower read/write path
- more failure modes

Decision:

- V1 uses direct Python imports from the `mempalace` package
- MCP remains a separate integration surface, not the implementation path for the Hermes plugin

### 3. Respect Hermes profile isolation

Hermes memory plugins are expected to use `hermes_home` for profile-scoped state.
MemPalace defaults to `~/.mempalace`, which is global.

Decision:

- plugin default storage lives under `HERMES_HOME/mempalace/`
- allow override to an existing global palace path via plugin config
- never hardcode `~/.mempalace` as the only storage path

## Architecture Decision

### Runtime model

Provider implementation will be a new in-tree Hermes plugin:

```text
plugins/memory/mempalace/
├── __init__.py
├── plugin.yaml
├── README.md
└── cli.py
```

The provider will import and wrap these MemPalace primitives directly:

- `mempalace.layers.MemoryStack` for first-turn or low-cost recall context
- `mempalace.searcher.search_memories` for semantic retrieval
- `mempalace.knowledge_graph.KnowledgeGraph` for temporal fact queries and writes
- direct collection writes patterned after MemPalace's `tool_add_drawer` flow for async turn persistence

### Storage layout

Default paths:

- palace path: `$(HERMES_HOME)/mempalace/palace`
- identity path: `$(HERMES_HOME)/mempalace/identity.txt`
- plugin config: `$(HERMES_HOME)/mempalace.json`

Optional override:

- user can set `palace_path` in plugin config to point at an existing shared MemPalace store

### Memory model

V1 uses a two-track storage approach.

Track A: raw exchange persistence

- every completed Hermes turn is stored asynchronously as a verbatim drawer
- drawer ids should be deterministic by session id + turn index to avoid duplicates
- metadata should include `session_id`, `turn_number`, `platform`, `user_id`, `agent_identity`, `filed_at`

Track B: structured recall helpers

- semantic search uses the MemPalace Chroma collection directly
- factual lookups use the temporal knowledge graph
- optional end-of-session extraction can classify memories into room types such as `decisions`, `preferences`, `problems`, `milestones`

### Wing and room strategy

V1 should keep routing simple and deterministic.

Default wing:

- CLI: `wing_<agent_identity_or_profile>`
- gateway / chat platforms: `wing_<user_id>`

Default room strategy:

- raw turn persistence goes to `general`
- explicit remembered facts can go to `facts`
- built-in memory mirroring can go to `builtin_memory`
- later session-end extraction may append to `decisions`, `preferences`, `problems`, `milestones`

This avoids over-engineering room inference in the first implementation while leaving room for richer organization later.

## Provider Behavior

### `is_available()`

Should succeed when:

- `mempalace` import works
- required local dependencies are present

Should not perform network calls.

### `initialize(session_id, **kwargs)`

Responsibilities:

- resolve `hermes_home`
- load plugin config
- choose effective `palace_path` and `identity_path`
- ensure local directories exist
- create or reuse `KnowledgeGraph`
- initialize lightweight runtime state for prefetch and async writes

### `system_prompt_block()`

Keep this small. It should only say:

- MemPalace provider is active
- storage path in use
- how the model should use the exposed tools

Do not inject large memory payloads here.

### `prefetch()` and `queue_prefetch()`

Decision:

- first turn: return `MemoryStack.wake_up()` result
- later turns: return cached search results prepared by `queue_prefetch()`

`queue_prefetch()` should run semantic recall in a background thread using the last user query.

V1 target:

- fast, bounded prefetched context
- no unbounded dump of the palace

### `sync_turn()`

Decision:

- required
- non-blocking
- stores raw paired turn content in the collection

Format:

```text
[role: user]
...

[role: assistant]
...
```

Rules:

- strip any injected memory context fences before persistence
- skip trivial turns like `ok` / `thanks` unless configured otherwise
- never block the main request path on collection writes

### `on_session_end()`

V1 should do lightweight finalization only:

- flush pending writes
- optionally append a compact diary-style summary entry if we find it valuable during review

Do not make this hook depend on heavyweight LLM extraction in the initial version.

### `on_memory_write()`

Mirror Hermes built-in memory updates into MemPalace so explicit remembered facts are not split across two systems.

V1 behavior:

- store built-in memory additions as drawers in `builtin_memory`
- optionally map well-formed facts into KG writes in a follow-up iteration, not the first cut

## Tool Surface

V1 should avoid importing the entire 19-tool MemPalace MCP surface into Hermes.

Recommended initial tools:

1. `mempalace_search`
   Semantic search over stored drawers.

2. `mempalace_profile`
   Return wake-up or profile-style context for the active user/profile.

3. `mempalace_kg_query`
   Query temporal facts for an entity.

4. `mempalace_remember`
   Explicitly store a durable memory drawer.

Possible V1.1 additions after real usage:

- `mempalace_kg_add`
- `mempalace_kg_invalidate`
- `mempalace_diary_read`

Non-goal for V1:

- exposing every MemPalace MCP operation through Hermes

## Config Design

Keep setup minimal.

`get_config_schema()` should initially expose:

- `palace_path`
  Description: optional custom path; blank means profile-scoped default under `HERMES_HOME`

- `identity_path`
  Description: optional custom identity file path; blank means profile-scoped default

Do not prompt for every MemPalace internal option in `hermes memory setup`.

`save_config()` writes non-secret config to `$(HERMES_HOME)/mempalace.json`.

`plugin.yaml` should include:

- `name: mempalace`
- `pip_dependencies: [mempalace]`
- hook declarations for whichever lifecycle hooks we actually implement

## Implementation Plan

### Phase 0: repository and review setup

- create this local git workspace
- capture architecture plan
- get engineering review before any provider code is written

### Phase 1: provider skeleton in Hermes

- add `plugins/memory/mempalace/__init__.py`
- add `plugin.yaml`
- add provider README
- add minimal `cli.py` with `status` and `config` visibility
- wire config loading and default path resolution

### Phase 2: core memory flow

- implement `initialize`
- implement `system_prompt_block`
- implement `sync_turn`
- implement `prefetch` + `queue_prefetch`
- implement bounded result formatting

### Phase 3: tool surface

- implement `mempalace_search`
- implement `mempalace_profile`
- implement `mempalace_kg_query`
- implement `mempalace_remember`

### Phase 4: test coverage

- discovery and load tests
- config and profile-isolation tests
- async sync-turn persistence tests
- prefetch formatting and first-turn behavior tests
- tool routing tests
- real temp-dir integration tests with local MemPalace collection and KG

### Phase 5: manual validation

- run `hermes memory setup`
- activate `mempalace`
- verify dependency install through `plugin.yaml`
- run a short CLI conversation across multiple sessions
- confirm recall and explicit memory writes work

## Test Plan

Hermes-side tests:

- `tests/agent/test_memory_provider.py`
  Add coverage only if shared manager behavior needs extension.

- add a dedicated provider test file:
  `tests/plugins/memory/test_mempalace_provider.py`

Suggested cases:

- provider loads and reports available when `mempalace` is installed
- blank config resolves to `HERMES_HOME` paths
- custom `palace_path` override wins
- `sync_turn()` writes non-trivial exchanges and skips trivial ones
- prefetch returns wake-up context on first turn
- queued semantic recall is returned on subsequent turns
- tool handlers serialize clean JSON results
- KG query tool returns structured facts
- shutdown joins background threads safely

Manual acceptance:

- a fact remembered in session A can be found in session B
- built-in memory writes are mirrored into MemPalace
- switching Hermes profiles does not cross-contaminate default local storage

## Risks

### Risk 1: MemPalace APIs are not optimized for Hermes turn-by-turn ingestion

Mitigation:

- write raw drawers directly through collection operations instead of shelling out to the CLI
- keep async writes isolated behind provider helper methods

### Risk 2: `Layer1.generate()` and some MemPalace status flows fetch broadly

Mitigation:

- keep prefetch bounded
- avoid calling unbounded status/taxonomy routines on the hot path
- use wake-up text only where it materially helps

### Risk 3: tool surface bloat

Mitigation:

- start with 3 to 4 high-value tools only
- keep advanced MemPalace operations out of V1

### Risk 4: confusion between standalone repo and real Hermes integration target

Mitigation:

- keep this repo explicitly labeled as a development workspace
- implement actual provider code only in a Hermes working tree

## Review Questions For Engineering

1. Is direct-library integration the right choice, or is there any strong reason to bridge through MCP anyway?
2. Is the proposed default storage under `HERMES_HOME/mempalace/` the right default, with optional override to a shared global palace?
3. Is the proposed V1 tool surface small enough, or should `mempalace_profile` be omitted and handled purely through prefetch?
4. Should session-end extraction into typed rooms ship in V1, or wait until the raw-turn path is proven stable?
5. Do we want `on_memory_write()` to translate explicit built-in memories into KG facts in V1, or keep that as a follow-up?

## Recommended Next Step

Assign an engineering review task against this plan first.

Only after review feedback is incorporated should implementation be assigned.
