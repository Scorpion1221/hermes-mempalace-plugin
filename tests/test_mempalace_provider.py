from __future__ import annotations

import json
import warnings
from pathlib import Path

import chromadb
import pytest
from mempalace.knowledge_graph import KnowledgeGraph

import plugins.memory.mempalace as mempalace_plugin
from plugins.memory.mempalace import MemPalaceMemoryProvider, register, resolve_paths


def _get_collection(palace_path: Path):
    import os
    os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
    client = chromadb.PersistentClient(path=str(palace_path))
    return client.get_or_create_collection("mempalace_drawers")


def _collection_count(palace_path: Path) -> int:
    try:
        return _get_collection(palace_path).count()
    except Exception:
        return 0


def _provider(hermes_home: Path, *, session_id: str = "session-1", **kwargs) -> MemPalaceMemoryProvider:
    provider = MemPalaceMemoryProvider()
    provider.initialize(
        session_id=session_id,
        hermes_home=str(hermes_home),
        platform=kwargs.pop("platform", "cli"),
        agent_identity=kwargs.pop("agent_identity", "coder"),
        **kwargs,
    )
    return provider


# ---------------------------------------------------------------------------
# Original 11 tests (unchanged names and behaviour)
# ---------------------------------------------------------------------------


def test_register_registers_provider() -> None:
    class DummyContext:
        def __init__(self) -> None:
            self.providers = []

        def register_memory_provider(self, provider) -> None:
            self.providers.append(provider)

    ctx = DummyContext()
    register(ctx)
    assert len(ctx.providers) == 1
    assert isinstance(ctx.providers[0], MemPalaceMemoryProvider)


def test_default_paths_resolve_under_hermes_home(tmp_path: Path) -> None:
    hermes_home = tmp_path / "profile"
    paths = resolve_paths(hermes_home)

    assert paths.base_dir == hermes_home / "mempalace"
    assert paths.palace_path == hermes_home / "mempalace" / "palace"
    assert paths.identity_path == hermes_home / "mempalace" / "identity.txt"
    assert paths.kg_path == hermes_home / "mempalace" / "knowledge_graph.sqlite3"


def test_custom_paths_from_config_are_respected(tmp_path: Path) -> None:
    hermes_home = tmp_path / "profile"
    provider = MemPalaceMemoryProvider()
    provider.save_config(
        {
            "palace_path": "shared/palace",
            "identity_path": "custom/identity.txt",
            "kg_path": "kg/graph.sqlite3",
        },
        str(hermes_home),
    )

    paths = resolve_paths(hermes_home)
    assert paths.palace_path == hermes_home / "shared" / "palace"
    assert paths.identity_path == hermes_home / "custom" / "identity.txt"
    assert paths.kg_path == hermes_home / "kg" / "graph.sqlite3"


def test_sync_turn_does_not_create_default_home_mempalace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_home = tmp_path / "fake-home"
    monkeypatch.setenv("HOME", str(fake_home))
    hermes_home = tmp_path / "profile"
    provider = _provider(hermes_home)

    provider.on_turn_start(0, "remember this", session_id="session-1")
    provider.sync_turn("remember this", "stored", session_id="session-1")
    provider.shutdown()

    assert not (fake_home / ".mempalace").exists()
    # Per-turn raw drawers were removed in favour of Haiku-extracted batches
    # (see _haiku_save_recent_turns); without an LLM configured nothing is
    # written until on_session_end fires the auto-diary path.
    assert _collection_count(provider.resolved_paths.palace_path) == 0


def test_prefetch_cache_is_session_keyed_and_wing_scoped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hermes_home = tmp_path / "profile"
    monkeypatch.setenv("MEMPAL_RECALL_LLM", "0")
    provider = _provider(hermes_home, user_id="alice")
    collection = _get_collection(provider.resolved_paths.palace_path)
    collection.upsert(
        ids=["alice-1", "bob-1"],
        documents=["Alice likes espresso", "Bob prefers tea"],
        metadatas=[
            {"wing": "wing_alice", "room": "facts", "source_file": "", "chunk_index": 0, "added_by": "test", "filed_at": "2026-04-11T00:00:00"},
            {"wing": "wing_bob", "room": "facts", "source_file": "", "chunk_index": 0, "added_by": "test", "filed_at": "2026-04-11T00:00:00"},
        ],
    )

    provider.on_turn_start(1, "espresso", session_id="session-a", user_id="alice")
    provider.on_turn_start(1, "tea", session_id="session-b", user_id="bob")
    provider.queue_prefetch("espresso", session_id="session-a")
    provider.queue_prefetch("tea", session_id="session-b")

    if provider._sessions["session-a"].prefetch_future is not None:
        provider._sessions["session-a"].prefetch_future.result(timeout=10)
    if provider._sessions["session-b"].prefetch_future is not None:
        provider._sessions["session-b"].prefetch_future.result(timeout=10)

    recall_a = provider.prefetch("espresso", session_id="session-a")
    recall_b = provider.prefetch("tea", session_id="session-b")

    # Global palace design: search spans all wings, so both results appear
    # in both sessions.  The key property is that prefetch caches are
    # session-keyed (each session ran its own query).
    assert "Alice likes espresso" in recall_a
    assert "Bob prefers tea" in recall_b


def test_sync_turn_is_idempotent_for_same_session_and_turn(tmp_path: Path) -> None:
    provider = _provider(tmp_path / "profile")
    provider.on_turn_start(7, "keep it", session_id="session-1")

    provider.sync_turn("keep it", "stored once", session_id="session-1")
    provider.sync_turn("keep it", "stored once", session_id="session-1")
    provider.shutdown()

    # Per-turn writes were replaced by Haiku-extracted periodic saves, which
    # require an LLM config. Without one, repeated sync_turn calls simply
    # accumulate in the in-memory buffer and never reach Chroma.
    collection = _get_collection(provider.resolved_paths.palace_path)
    stored = collection.get(include=["documents", "metadatas"])
    assert stored["ids"] == []


@pytest.mark.parametrize("context_name", ["subagent", "cron", "flush"])
def test_non_primary_contexts_do_not_write(tmp_path: Path, context_name: str) -> None:
    provider = _provider(tmp_path / context_name, agent_context=context_name)
    provider.on_turn_start(0, "secret", session_id="session-1", agent_context=context_name)
    provider.sync_turn("secret", "should not persist", session_id="session-1")
    provider.on_memory_write("add", "memory", "also blocked")
    provider.shutdown()

    assert _collection_count(provider.resolved_paths.palace_path) == 0


def test_shared_palace_search_spans_all_wings(tmp_path: Path) -> None:
    """Search is global by design — a single palace, wings are organisational tags."""
    shared_palace = tmp_path / "shared" / "palace"
    provider_a = MemPalaceMemoryProvider()
    provider_a.save_config({"palace_path": str(shared_palace)}, str(tmp_path / "alice"))
    provider_a.initialize("session-a", hermes_home=str(tmp_path / "alice"), user_id="alice", agent_identity="coder")

    provider_b = MemPalaceMemoryProvider()
    provider_b.save_config({"palace_path": str(shared_palace)}, str(tmp_path / "bob"))
    provider_b.initialize("session-b", hermes_home=str(tmp_path / "bob"), user_id="bob", agent_identity="coder")

    json.loads(provider_a.handle_tool_call("mempalace_remember", {"content": "favorite coffee is espresso"}))
    json.loads(provider_b.handle_tool_call("mempalace_remember", {"content": "favorite tea is oolong"}))

    search_a = json.loads(provider_a.handle_tool_call("mempalace_search", {"query": "favorite"}))
    search_b = json.loads(provider_b.handle_tool_call("mempalace_search", {"query": "favorite"}))

    # Both providers see both drawers because search uses wing=None (global).
    texts_a = {result["text"] for result in search_a["results"]}
    texts_b = {result["text"] for result in search_b["results"]}
    assert "favorite coffee is espresso" in texts_a
    assert "favorite tea is oolong" in texts_a
    assert "favorite coffee is espresso" in texts_b
    assert "favorite tea is oolong" in texts_b

    provider_a.shutdown()
    provider_b.shutdown()


def test_chromadb_collection_access_does_not_emit_model_fields_warning(tmp_path: Path) -> None:
    _provider(tmp_path / "profile")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _get_collection(tmp_path / "profile" / "mempalace" / "palace")

    assert not [warning for warning in caught if "model_fields" in str(warning.message)]


def test_first_turn_prefetch_is_bounded(tmp_path: Path) -> None:
    provider = _provider(tmp_path / "profile", user_id="alice")
    collection = _get_collection(provider.resolved_paths.palace_path)
    collection.upsert(
        ids=[f"id-{index}" for index in range(6)],
        documents=[f"Alice memory #{index}" for index in range(6)],
        metadatas=[
            {"wing": "wing_alice", "room": "facts", "source_file": "", "chunk_index": index, "added_by": "test", "filed_at": f"2026-04-11T00:00:0{index}"}
            for index in range(6)
        ],
    )

    provider.on_turn_start(0, "Alice memory", session_id="session-1", user_id="alice")
    recall = provider.prefetch("Alice memory", session_id="session-1")

    assert recall.startswith("## MemPalace Recall")
    assert recall.count("\n- [") <= mempalace_plugin.FIRST_TURN_RECALL_LIMIT


def test_tool_outputs_are_json_and_kg_query_is_structured(tmp_path: Path) -> None:
    provider = _provider(tmp_path / "profile", user_id="alice")
    remember = json.loads(
        provider.handle_tool_call("mempalace_remember", {"content": "Alice likes espresso"})
    )
    assert remember["success"] is True

    search = json.loads(provider.handle_tool_call("mempalace_search", {"query": "espresso"}))
    assert search["results"][0]["text"] == "Alice likes espresso"

    kg = KnowledgeGraph(db_path=str(provider.resolved_paths.kg_path))
    kg.add_triple("Alice", "likes", "espresso", valid_from="2026-04-11")
    kg.close()

    kg_result = json.loads(
        provider.handle_tool_call(
            "mempalace_kg_query",
            {"entity": "Alice", "direction": "outgoing", "as_of": "2026-04-11"},
        )
    )
    assert kg_result["entity"] == "Alice"
    assert kg_result["results"][0]["predicate"] == "likes"
    assert kg_result["results"][0]["object"] == "espresso"

    provider.shutdown()


# ---------------------------------------------------------------------------
# New V2 tests
# ---------------------------------------------------------------------------


def test_kg_add(tmp_path: Path) -> None:
    provider = _provider(tmp_path / "profile")
    result = json.loads(
        provider.handle_tool_call("mempalace_kg_add", {
            "subject": "Alice",
            "predicate": "likes",
            "object": "coffee",
            "valid_from": "2026-01-01",
        })
    )
    assert result["success"] is True
    assert result["subject"] == "Alice"
    assert result["predicate"] == "likes"
    assert result["object"] == "coffee"

    # Verify it's queryable
    query = json.loads(
        provider.handle_tool_call("mempalace_kg_query", {"entity": "Alice"})
    )
    assert any(r["predicate"] == "likes" for r in query["results"])
    provider.shutdown()


def test_kg_invalidate(tmp_path: Path) -> None:
    provider = _provider(tmp_path / "profile")
    provider.handle_tool_call("mempalace_kg_add", {
        "subject": "Bob",
        "predicate": "works_at",
        "object": "Acme",
    })
    result = json.loads(
        provider.handle_tool_call("mempalace_kg_invalidate", {
            "subject": "Bob",
            "predicate": "works_at",
            "object": "Acme",
            "ended": "2026-04-15",
        })
    )
    assert result["success"] is True
    provider.shutdown()


def test_kg_timeline(tmp_path: Path) -> None:
    provider = _provider(tmp_path / "profile")
    provider.handle_tool_call("mempalace_kg_add", {
        "subject": "Eve",
        "predicate": "visited",
        "object": "Paris",
        "valid_from": "2026-03-01",
    })
    result = json.loads(
        provider.handle_tool_call("mempalace_kg_timeline", {"entity": "Eve"})
    )
    assert "timeline" in result
    provider.shutdown()


def test_kg_stats(tmp_path: Path) -> None:
    provider = _provider(tmp_path / "profile")
    result = json.loads(provider.handle_tool_call("mempalace_kg_stats", {}))
    assert "stats" in result
    provider.shutdown()


def test_status(tmp_path: Path) -> None:
    provider = _provider(tmp_path / "profile")
    result = json.loads(provider.handle_tool_call("mempalace_status", {}))
    assert result["provider"] == "mempalace"
    assert result["version"] == "2.0.0"
    assert "wing" in result
    assert "drawer_count" in result
    provider.shutdown()


def test_list_wings(tmp_path: Path) -> None:
    provider = _provider(tmp_path / "profile")
    provider.handle_tool_call("mempalace_remember", {"content": "test data"})
    result = json.loads(provider.handle_tool_call("mempalace_list_wings", {}))
    assert "wings" in result
    assert len(result["wings"]) >= 1
    provider.shutdown()


def test_list_rooms(tmp_path: Path) -> None:
    provider = _provider(tmp_path / "profile")
    provider.handle_tool_call("mempalace_remember", {"content": "test data", "room": "myroom"})
    result = json.loads(provider.handle_tool_call("mempalace_list_rooms", {}))
    assert "rooms" in result
    assert "myroom" in result["rooms"]
    provider.shutdown()


def test_get_taxonomy(tmp_path: Path) -> None:
    provider = _provider(tmp_path / "profile")
    provider.handle_tool_call("mempalace_remember", {"content": "taxonomy test"})
    result = json.loads(provider.handle_tool_call("mempalace_get_taxonomy", {}))
    assert "taxonomy" in result
    assert len(result["taxonomy"]) >= 1
    provider.shutdown()


def test_add_drawer(tmp_path: Path) -> None:
    provider = _provider(tmp_path / "profile")
    result = json.loads(
        provider.handle_tool_call("mempalace_add_drawer", {
            "content": "drawer content",
            "room": "test_room",
        })
    )
    assert result["success"] is True
    assert result["room"] == "test_room"
    assert "drawer_id" in result
    provider.shutdown()


def test_delete_drawer(tmp_path: Path) -> None:
    provider = _provider(tmp_path / "profile")
    add_result = json.loads(
        provider.handle_tool_call("mempalace_add_drawer", {"content": "to be deleted"})
    )
    drawer_id = add_result["drawer_id"]

    delete_result = json.loads(
        provider.handle_tool_call("mempalace_delete_drawer", {"drawer_id": drawer_id})
    )
    assert delete_result["success"] is True
    assert delete_result["deleted"] == drawer_id

    # Verify it's gone
    get_result = json.loads(
        provider.handle_tool_call("mempalace_get_drawer", {"drawer_id": drawer_id})
    )
    assert "error" in get_result
    provider.shutdown()


def test_get_drawer(tmp_path: Path) -> None:
    provider = _provider(tmp_path / "profile")
    add_result = json.loads(
        provider.handle_tool_call("mempalace_add_drawer", {"content": "retrieve me"})
    )
    drawer_id = add_result["drawer_id"]

    get_result = json.loads(
        provider.handle_tool_call("mempalace_get_drawer", {"drawer_id": drawer_id})
    )
    assert get_result["drawer_id"] == drawer_id
    assert get_result["content"] == "retrieve me"
    assert "metadata" in get_result
    provider.shutdown()


def test_update_drawer(tmp_path: Path) -> None:
    provider = _provider(tmp_path / "profile")
    add_result = json.loads(
        provider.handle_tool_call("mempalace_add_drawer", {"content": "original"})
    )
    drawer_id = add_result["drawer_id"]

    update_result = json.loads(
        provider.handle_tool_call("mempalace_update_drawer", {
            "drawer_id": drawer_id,
            "content": "updated",
        })
    )
    assert update_result["success"] is True

    get_result = json.loads(
        provider.handle_tool_call("mempalace_get_drawer", {"drawer_id": drawer_id})
    )
    assert get_result["content"] == "updated"
    provider.shutdown()


def test_list_drawers(tmp_path: Path) -> None:
    provider = _provider(tmp_path / "profile")
    provider.handle_tool_call("mempalace_add_drawer", {"content": "item 1", "room": "listing"})
    provider.handle_tool_call("mempalace_add_drawer", {"content": "item 2", "room": "listing"})

    result = json.loads(
        provider.handle_tool_call("mempalace_list_drawers", {"room": "listing"})
    )
    assert "drawers" in result
    assert len(result["drawers"]) >= 2
    provider.shutdown()


def test_diary_write(tmp_path: Path) -> None:
    provider = _provider(tmp_path / "profile", agent_identity="hermes")
    result = json.loads(
        provider.handle_tool_call("mempalace_diary_write", {"entry": "Today was productive"})
    )
    assert result["success"] is True
    assert "diary_" in result["room"]
    assert result["date"] is not None
    provider.shutdown()


def test_diary_read(tmp_path: Path) -> None:
    provider = _provider(tmp_path / "profile", agent_identity="hermes")
    provider.handle_tool_call("mempalace_diary_write", {"entry": "Morning entry"})
    provider.handle_tool_call("mempalace_diary_write", {"entry": "Evening entry"})

    result = json.loads(provider.handle_tool_call("mempalace_diary_read", {}))
    assert "entries" in result
    assert len(result["entries"]) >= 1
    # Same-day entries are appended, so we should have 1 entry with both texts
    entry_text = result["entries"][0]["content"]
    assert "Morning entry" in entry_text
    assert "Evening entry" in entry_text
    provider.shutdown()


def test_check_duplicate(tmp_path: Path) -> None:
    provider = _provider(tmp_path / "profile")
    provider.handle_tool_call("mempalace_remember", {"content": "unique fact"})

    result = json.loads(
        provider.handle_tool_call("mempalace_check_duplicate", {"content": "unique fact"})
    )
    assert "is_duplicate" in result
    # Should find the existing exact match
    assert result["similar"] >= 1
    provider.shutdown()


def test_reconnect(tmp_path: Path) -> None:
    provider = _provider(tmp_path / "profile")
    provider.handle_tool_call("mempalace_remember", {"content": "before reconnect"})

    result = json.loads(provider.handle_tool_call("mempalace_reconnect", {}))
    assert result["success"] is True
    assert result["drawer_count"] >= 1

    # Verify data survives reconnect
    search = json.loads(
        provider.handle_tool_call("mempalace_search", {"query": "before reconnect"})
    )
    assert len(search["results"]) >= 1
    provider.shutdown()


def test_on_pre_compress(tmp_path: Path) -> None:
    provider = _provider(tmp_path / "profile")
    messages = [
        {"role": "user", "content": "Tell me about quantum computing"},
        {"role": "assistant", "content": "Quantum computing uses qubits instead of classical bits..."},
        {"role": "user", "content": "How does entanglement work?"},
        {"role": "assistant", "content": "Entanglement is a quantum phenomenon where particles become correlated..."},
    ]
    provider.on_pre_compress(messages)

    # Should have saved at least one turn as a drawer in the "compressed" room
    collection = _get_collection(provider.resolved_paths.palace_path)
    stored = collection.get(include=["documents", "metadatas"])
    compressed = [m for m in stored["metadatas"] if m.get("room") == "compressed"]
    assert len(compressed) >= 1
    provider.shutdown()


def test_on_pre_compress_blocked_when_writes_disabled(tmp_path: Path) -> None:
    provider = _provider(tmp_path / "profile", agent_context="subagent")
    messages = [
        {"role": "user", "content": "save this"},
        {"role": "assistant", "content": "I saved it for you."},
    ]
    provider.on_pre_compress(messages)
    assert _collection_count(provider.resolved_paths.palace_path) == 0
    provider.shutdown()


# Write-blocked context tests for new write tools
@pytest.mark.parametrize("tool_name,args", [
    ("mempalace_kg_add", {"subject": "A", "predicate": "B", "object": "C"}),
    ("mempalace_kg_invalidate", {"subject": "A", "predicate": "B", "object": "C"}),
    ("mempalace_add_drawer", {"content": "blocked"}),
    ("mempalace_delete_drawer", {"drawer_id": "x"}),
    ("mempalace_update_drawer", {"drawer_id": "x", "content": "blocked"}),
    ("mempalace_diary_write", {"entry": "blocked"}),
    ("mempalace_create_tunnel", {"source_wing": "a", "source_room": "b", "target_wing": "c", "target_room": "d"}),
    ("mempalace_delete_tunnel", {"tunnel_id": "x"}),
])
def test_write_tools_blocked_in_subagent_context(tmp_path: Path, tool_name: str, args: dict) -> None:
    provider = _provider(tmp_path / "blocked", agent_context="subagent")
    result = json.loads(provider.handle_tool_call(tool_name, args))
    assert result.get("success") is False or result.get("reason") == "writes_disabled"
    provider.shutdown()


def test_memories_filed_away(tmp_path: Path) -> None:
    provider = _provider(tmp_path / "profile")
    provider.handle_tool_call("mempalace_remember", {"content": "fact one"})
    provider.handle_tool_call("mempalace_remember", {"content": "fact two"})

    result = json.loads(provider.handle_tool_call("mempalace_memories_filed_away", {}))
    assert result["memories_filed"] == 2
    provider.shutdown()


def test_get_aaak_spec(tmp_path: Path) -> None:
    provider = _provider(tmp_path / "profile")
    result = json.loads(provider.handle_tool_call("mempalace_get_aaak_spec", {}))
    assert result["spec"] == "AAAK/1.0"
    assert result["provider"] == "mempalace"
    assert len(result["capabilities"]) == 31
    provider.shutdown()


def test_system_prompt_lists_all_tools(tmp_path: Path) -> None:
    provider = _provider(tmp_path / "profile")
    prompt = provider.system_prompt_block()
    assert "mempalace_search" in prompt
    assert "mempalace_memories_filed_away" in prompt
    provider.shutdown()


def test_31_tool_schemas_returned(tmp_path: Path) -> None:
    provider = _provider(tmp_path / "profile")
    schemas = provider.get_tool_schemas()
    assert len(schemas) == 31
    names = {s["name"] for s in schemas}
    assert "mempalace_search" in names
    assert "mempalace_memories_filed_away" in names
    provider.shutdown()


def test_unknown_tool_returns_error(tmp_path: Path) -> None:
    provider = _provider(tmp_path / "profile")
    result = json.loads(provider.handle_tool_call("mempalace_nonexistent", {}))
    assert "error" in result
    assert "unknown_tool" in result["error"]
    provider.shutdown()


def test_hook_settings_read(tmp_path: Path) -> None:
    provider = _provider(tmp_path / "profile")
    result = json.loads(provider.handle_tool_call("mempalace_hook_settings", {}))
    # Should return current settings without error
    assert "hook_silent_save" in result or "error" in result
    provider.shutdown()


def test_sync_turn_caches_previous_assistant_reply_for_session(tmp_path: Path) -> None:
    provider = _provider(tmp_path / "profile")
    provider.on_turn_start(0, "why?", session_id="session-1")
    provider.sync_turn(
        "why?",
        "<<MEMORY_CONTEXT>>\nignore this\n<</MEMORY_CONTEXT>>\nReal assistant reply",
        session_id="session-1",
    )

    assert provider._sessions["session-1"].last_assistant_reply == "Real assistant reply"
    cache_dir = provider.resolved_paths.base_dir / "session_state"
    cache_files = list(cache_dir.glob("*_last_assistant.txt"))
    assert len(cache_files) == 1
    assert cache_files[0].read_text(encoding="utf-8") == "Real assistant reply"
    provider.shutdown()


def test_render_recall_includes_previous_assistant_context_in_fallback_search(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    provider = _provider(tmp_path / "profile")
    provider.on_turn_start(0, "why?", session_id="session-1")
    provider._sessions["session-1"].last_assistant_reply = (
        "Earlier I explained the MemPalace Claude and Codex hooks."
    )

    captured = {}

    def fake_search_memories(**kwargs):
        captured.update(kwargs)
        return {
            "results": [
                {"wing": "wing_default", "room": "decisions", "text": "Matching memory"}
            ]
        }

    monkeypatch.setattr(mempalace_plugin, "search_memories", fake_search_memories)
    monkeypatch.setenv("MEMPAL_RECALL_LLM", "0")

    recall = provider.prefetch("why?", session_id="session-1")

    assert "## MemPalace Recall" in recall
    assert captured["query"] == (
        "Earlier I explained the MemPalace Claude and Codex hooks.\n\nwhy?"
    )
    provider.shutdown()


def test_render_recall_uses_file_fallback_when_memory_state_is_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = _provider(tmp_path / "profile")
    provider.on_turn_start(0, "why?", session_id="session-1")
    cache_path = provider._assistant_cache_path("session-1")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        "Earlier I explained the MemPalace Claude and Codex hooks.",
        encoding="utf-8",
    )

    captured = {}

    def fake_search_memories(**kwargs):
        captured.update(kwargs)
        return {
            "results": [
                {"wing": "wing_default", "room": "decisions", "text": "Matching memory"}
            ]
        }

    monkeypatch.setattr(mempalace_plugin, "search_memories", fake_search_memories)
    monkeypatch.setenv("MEMPAL_RECALL_LLM", "0")

    recall = provider.prefetch("why?", session_id="session-1")

    assert "## MemPalace Recall" in recall
    assert provider._sessions["session-1"].last_assistant_reply == (
        "Earlier I explained the MemPalace Claude and Codex hooks."
    )
    assert captured["query"] == (
        "Earlier I explained the MemPalace Claude and Codex hooks.\n\nwhy?"
    )
    provider.shutdown()


def test_render_recall_passes_previous_assistant_context_to_llm_rewrite_and_rerank(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = _provider(tmp_path / "profile")
    provider.on_turn_start(0, "why?", session_id="session-1")
    provider._sessions["session-1"].last_assistant_reply = (
        "Earlier I explained the MemPalace Claude and Codex hooks."
    )

    calls = {}

    def fake_search_memories(**kwargs):
        return {
            "results": [
                {"wing": "wing_default", "room": "decisions", "text": f"hit {i}"}
                for i in range(6)
            ]
        }

    def fake_decide_recall(
        query,
        config=None,
        previous_assistant_context=None,
        active_context=None,
    ):
        calls["rewrite"] = previous_assistant_context
        calls["active_context"] = active_context
        return {
            "should_recall": True,
            "reason": "short_followup_depends_on_previous_assistant",
            "query": "mempalace hooks",
            "after": None,
        }

    def fake_rerank(query, hits, top_k=5, config=None, previous_assistant_context=None):
        calls["rerank"] = previous_assistant_context
        return hits[:top_k]

    monkeypatch.setattr(mempalace_plugin, "search_memories", fake_search_memories)
    monkeypatch.setattr("mempalace.recall_llm.is_enabled", lambda: True)
    monkeypatch.setattr("mempalace.recall_llm._get_llm_config", lambda: {"backend": "stub"})
    monkeypatch.setattr("mempalace.recall_llm.decide_recall", fake_decide_recall)
    monkeypatch.setattr("mempalace.recall_llm.rerank", fake_rerank)

    recall = provider.prefetch("why?", session_id="session-1")

    assert "## MemPalace Recall" in recall
    assert calls["rewrite"] == {
        "tail": "Earlier I explained the MemPalace Claude and Codex hooks."
    }
    # active_context now carries the Hermes session hints (wing/platform)
    # plus, when available, palace taxonomy + KG entities (see
    # MemPalaceMemoryProvider._build_recall_active_context). The exact
    # taxonomy/entities depend on whatever's in the user's palace at test
    # time, so we only assert the stable fields the gate actually relies
    # on for filter selection.
    active_ctx = calls["active_context"]
    assert isinstance(active_ctx, dict)
    assert active_ctx["wing"] == "wing_coder"
    assert active_ctx["platform"] == "cli"
    assert calls["rerank"] == {
        "tail": "Earlier I explained the MemPalace Claude and Codex hooks."
    }
    provider.shutdown()


def test_render_recall_llm_can_skip_recall_entirely(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = _provider(tmp_path / "profile")
    provider.on_turn_start(0, "format this json", session_id="session-1")
    provider._sessions["session-1"].last_assistant_reply = "Earlier I explained the hooks."

    def fake_decide_recall(*args, **kwargs):
        return {
            "should_recall": False,
            "reason": "direct_local_task_no_memory_needed",
            "query": None,
            "after": None,
        }

    monkeypatch.setattr("mempalace.recall_llm.is_enabled", lambda: True)
    monkeypatch.setattr("mempalace.recall_llm._get_llm_config", lambda: {"backend": "stub"})
    monkeypatch.setattr("mempalace.recall_llm.decide_recall", fake_decide_recall)
    monkeypatch.setattr(
        mempalace_plugin,
        "search_memories",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("search should not run")),
    )

    assert provider.prefetch("format this json", session_id="session-1") == ""
    provider.shutdown()


def test_render_recall_session_local_continue_skips_without_search(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = _provider(tmp_path / "profile")
    provider.on_turn_start(0, "继续推进，直到完全修复完成", session_id="session-1")
    provider._sessions["session-1"].last_assistant_reply = "Earlier I explained the hooks."

    monkeypatch.setattr(
        mempalace_plugin,
        "search_memories",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("search should not run")),
    )
    monkeypatch.setenv("MEMPAL_RECALL_LLM", "0")

    assert provider.prefetch("继续推进，直到完全修复完成", session_id="session-1") == ""
    provider.shutdown()


def test_render_recall_history_continue_can_still_search(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = _provider(tmp_path / "profile")
    provider.on_turn_start(0, "按之前那个方案继续推进", session_id="session-1")
    provider._sessions["session-1"].last_assistant_reply = "Earlier I explained the hooks."

    captured = {}

    def fake_search_memories(**kwargs):
        captured.update(kwargs)
        return {
            "results": [
                {"wing": "wing_default", "room": "decisions", "text": "Matching memory"}
            ]
        }

    monkeypatch.setattr(mempalace_plugin, "search_memories", fake_search_memories)
    monkeypatch.setenv("MEMPAL_RECALL_LLM", "0")

    recall = provider.prefetch("按之前那个方案继续推进", session_id="session-1")

    assert "## MemPalace Recall" in recall
    assert captured["query"].endswith("\n\n按之前那个方案继续推进")
    provider.shutdown()


def test_queue_prefetch_allows_short_followup_when_previous_assistant_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = _provider(tmp_path / "profile")
    provider.on_turn_start(0, "why?", session_id="session-1")
    provider._sessions["session-1"].last_assistant_reply = "Earlier I explained the hooks."

    submitted = {}

    def fake_submit(fn, query, state, limit):
        submitted["query"] = query
        submitted["session_id"] = state.session_id
        submitted["limit"] = limit

        class DummyFuture:
            def done(self):
                return False

        return DummyFuture()

    monkeypatch.setattr(
        provider,
        "_executor",
        type(
            "Exec",
            (),
            {
                "submit": staticmethod(fake_submit),
                "shutdown": staticmethod(lambda **kwargs: None),
            },
        )(),
    )

    provider.queue_prefetch("why?", session_id="session-1")

    assert submitted["query"] == "why?"
    assert submitted["session_id"] == "session-1"
    assert submitted["limit"] == mempalace_plugin.PREFETCH_RECALL_LIMIT
    provider.shutdown()


def test_queue_prefetch_still_skips_acknowledgement_even_with_previous_assistant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = _provider(tmp_path / "profile")
    provider.on_turn_start(0, "ok", session_id="session-1")
    provider._sessions["session-1"].last_assistant_reply = "Earlier I explained the hooks."

    called = {"submit": False}

    def fake_submit(*args, **kwargs):
        called["submit"] = True
        raise AssertionError("submit should not be called for pure ack")

    monkeypatch.setattr(
        provider,
        "_executor",
        type(
            "Exec",
            (),
            {
                "submit": staticmethod(fake_submit),
                "shutdown": staticmethod(lambda **kwargs: None),
            },
        )(),
    )

    provider.queue_prefetch("ok", session_id="session-1")

    assert called["submit"] is False
    provider.shutdown()


def test_on_session_end_clears_assistant_cache_file(tmp_path: Path) -> None:
    provider = _provider(tmp_path / "profile")
    provider.on_turn_start(0, "why?", session_id="session-1")
    provider.sync_turn("why?", "Assistant reply to cache", session_id="session-1")
    cache_path = provider._assistant_cache_path("session-1")
    assert cache_path.exists()

    provider.on_session_end(
        [
            {"role": "user", "content": "why?"},
            {"role": "assistant", "content": "Assistant reply to cache"},
        ]
    )

    assert not cache_path.exists()


def test_shutdown_preserves_assistant_cache_file_for_process_restart(tmp_path: Path) -> None:
    hermes_home = tmp_path / "profile"
    provider = _provider(hermes_home)
    provider.on_turn_start(0, "why?", session_id="session-1")
    provider.sync_turn("why?", "Assistant reply to persist", session_id="session-1")
    cache_path = provider._assistant_cache_path("session-1")
    assert cache_path.exists()
    provider.shutdown()

    provider2 = _provider(hermes_home, session_id="session-1")
    provider2.on_turn_start(0, "why?", session_id="session-1")

    captured = {}

    def fake_search_memories(**kwargs):
        captured.update(kwargs)
        return {
            "results": [
                {"wing": "wing_default", "room": "decisions", "text": "Matching memory"}
            ]
        }

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(mempalace_plugin, "search_memories", fake_search_memories)
        mp.setenv("MEMPAL_RECALL_LLM", "0")
        recall = provider2.prefetch("why?", session_id="session-1")

    assert "## MemPalace Recall" in recall
    assert captured["query"] == "Assistant reply to persist\n\nwhy?"
    provider2.shutdown()


# ---------------------------------------------------------------------------
# New tests: recall filter forwarding, active_context enrichment, Haiku save
# ---------------------------------------------------------------------------


def test_recall_filter_forwarding_valid_room_and_hall(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If decide_recall returns valid filters, search_memories receives them."""
    provider = _provider(tmp_path / "profile")
    provider.on_turn_start(0, "show notes", session_id="session-1")
    provider._sessions["session-1"].last_assistant_reply = "Sure."

    captured = {}

    def fake_search_memories(**kwargs):
        captured.update(kwargs)
        # Note: post-search filter strips room=='diary' from the rendered
        # output, so we return a non-diary hit here. The test asserts on
        # what was passed INTO search_memories, not what came out.
        return {
            "results": [
                {"wing": "wing_default", "room": "decisions", "text": "Past decisions log"}
            ]
        }

    def fake_decide_recall(query, config=None, previous_assistant_context=None, active_context=None):
        return {
            "should_recall": True,
            "reason": "history_reference",
            "query": "decisions",
            "after": None,
            "filters": {"room": "decisions", "hall": "hall_decisions"},
        }

    def fake_build_active_context(self_arg, state):
        return {
            "wing": state.wing,
            "platform": state.platform,
            "palace": {
                "rooms": ["decisions", "general", "facts"],
                "halls": ["hall_decisions", "general"],
            },
        }

    monkeypatch.setattr(mempalace_plugin, "search_memories", fake_search_memories)
    monkeypatch.setattr("mempalace.recall_llm.is_enabled", lambda: True)
    monkeypatch.setattr("mempalace.recall_llm._get_llm_config", lambda: {"backend": "stub"})
    monkeypatch.setattr("mempalace.recall_llm.decide_recall", fake_decide_recall)
    monkeypatch.setattr(
        mempalace_plugin.MemPalaceMemoryProvider,
        "_build_recall_active_context",
        fake_build_active_context,
    )

    recall = provider.prefetch("show notes", session_id="session-1")

    assert "## MemPalace Recall" in recall
    assert captured.get("room") == "decisions"
    assert captured.get("hall") == "hall_decisions"
    provider.shutdown()


def test_recall_filter_validation_drops_hallucinated_room(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Filters with non-existent values are stripped (no room in search_memories)."""
    provider = _provider(tmp_path / "profile")
    provider.on_turn_start(0, "show fictional", session_id="session-1")
    provider._sessions["session-1"].last_assistant_reply = "Sure."

    captured = {}

    def fake_search_memories(**kwargs):
        captured.update(kwargs)
        return {
            "results": [
                {"wing": "wing_default", "room": "general", "text": "some result"}
            ]
        }

    def fake_decide_recall(query, config=None, previous_assistant_context=None, active_context=None):
        return {
            "should_recall": True,
            "reason": "history_reference",
            "query": "fictional room query",
            "after": None,
            "filters": {"room": "fictional_room"},
        }

    def fake_build_active_context(self_arg, state):
        return {
            "wing": state.wing,
            "platform": state.platform,
            "palace": {"rooms": ["diary", "general", "facts"], "halls": []},
        }

    monkeypatch.setattr(mempalace_plugin, "search_memories", fake_search_memories)
    monkeypatch.setattr("mempalace.recall_llm.is_enabled", lambda: True)
    monkeypatch.setattr("mempalace.recall_llm._get_llm_config", lambda: {"backend": "stub"})
    monkeypatch.setattr("mempalace.recall_llm.decide_recall", fake_decide_recall)
    monkeypatch.setattr(
        mempalace_plugin.MemPalaceMemoryProvider,
        "_build_recall_active_context",
        fake_build_active_context,
    )

    recall = provider.prefetch("show fictional", session_id="session-1")

    assert "## MemPalace Recall" in recall
    # "fictional_room" was not in the taxonomy, so it should NOT be
    # forwarded to search_memories.
    assert "room" not in captured
    provider.shutdown()


def test_active_context_enrichment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """decide_recall receives active_context with palace + entities keys."""
    provider = _provider(tmp_path / "profile")
    provider.on_turn_start(0, "where is the config?", session_id="session-1")
    provider._sessions["session-1"].last_assistant_reply = "Let me look."

    ctx_captured = {}

    def fake_search(**kwargs):
        return {"results": [{"wing": "w", "room": "r", "text": "t"}]}

    def fake_decide(query, config=None, previous_assistant_context=None, active_context=None):
        ctx_captured.update(active_context if isinstance(active_context, dict) else {"raw": active_context})
        return {
            "should_recall": True,
            "reason": "test",
            "query": "config",
            "after": None,
        }

    def fake_active_ctx(self_arg, state):
        return {
            "wing": state.wing,
            "platform": state.platform,
            "palace": {"rooms": ["general"], "halls": ["general"]},
            "entities": ["Alice", "Bob"],
        }

    monkeypatch.setattr(mempalace_plugin, "search_memories", fake_search)
    monkeypatch.setattr("mempalace.recall_llm.is_enabled", lambda: True)
    monkeypatch.setattr("mempalace.recall_llm._get_llm_config", lambda: {"backend": "stub"})
    monkeypatch.setattr("mempalace.recall_llm.decide_recall", fake_decide)
    monkeypatch.setattr(
        mempalace_plugin.MemPalaceMemoryProvider,
        "_build_recall_active_context",
        fake_active_ctx,
    )

    provider.prefetch("where is the config?", session_id="session-1")

    assert "palace" in ctx_captured
    assert "rooms" in ctx_captured["palace"]
    assert "entities" in ctx_captured
    assert "Alice" in ctx_captured["entities"]
    provider.shutdown()


def test_haiku_save_trigger_after_n_turns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """_haiku_save_recent_turns runs after MEMPAL_HERMES_SAVE_INTERVAL turns."""
    monkeypatch.setenv("MEMPAL_HERMES_SAVE_INTERVAL", "3")
    monkeypatch.setenv("MEMPAL_RECALL_LLM", "1")

    provider = _provider(tmp_path / "profile")

    # Stub LLM to return a known JSON
    fixed_json = json.dumps({
        "diary": "User discussed project alpha and decided on a rewrite.",
        "drawers": [
            {"wing": "wing_alpha", "room": "decisions", "content": "Decided to rewrite the auth module from scratch."},
        ],
        "kg": [
            {"subject": "user", "predicate": "decided", "object": "auth_rewrite"},
        ],
    })

    def fake_call_llm(config, prompt, max_tokens=None, timeout=None, json_mode=False):
        return fixed_json

    monkeypatch.setattr("mempalace.recall_llm.is_enabled", lambda: True)
    monkeypatch.setattr(
        "mempalace.recall_llm._get_llm_config",
        lambda: {"backend": "stub", "model": "h", "key": "k", "endpoint": "e"},
    )
    monkeypatch.setattr("mempalace.recall_llm._call_llm", fake_call_llm)

    # Simulate 3 sync_turn calls
    for i in range(3):
        provider.on_turn_start(i, f"user message {i}", session_id="session-1")
        provider.sync_turn(
            f"user message {i}",
            f"assistant reply {i} which is long enough to not be trivial",
            session_id="session-1",
        )

    provider.shutdown()

    # Verify drawers were written
    collection = _get_collection(provider.resolved_paths.palace_path)
    stored = collection.get(include=["documents", "metadatas"])
    haiku_entries = [
        m for m in stored["metadatas"] if m.get("added_by") == "haiku_async_save"
    ]
    assert len(haiku_entries) >= 1, f"Expected Haiku-extracted drawers, got {stored['ids']}"
    # At least the diary + 1 drawer
    rooms_written = {m.get("room") for m in haiku_entries}
    assert "diary" in rooms_written or "decisions" in rooms_written

    # Verify KG triple was written
    kg = KnowledgeGraph(db_path=str(provider.resolved_paths.kg_path))
    try:
        facts = kg.query_entity("user", direction="outgoing")
        assert any(f.get("predicate") == "decided" for f in facts)
    finally:
        kg.close()

    provider.shutdown()


def test_haiku_save_malformed_json_dumps_failure_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the LLM returns invalid JSON, a hermes_save_fail_*.txt is created."""
    monkeypatch.setenv("MEMPAL_HERMES_SAVE_INTERVAL", "1")
    monkeypatch.setenv("MEMPAL_RECALL_LLM", "1")

    provider = _provider(tmp_path / "profile")

    def fake_call_llm(config, prompt, max_tokens=None, timeout=None, json_mode=False):
        return "This is not JSON { broken {"

    monkeypatch.setattr("mempalace.recall_llm.is_enabled", lambda: True)
    monkeypatch.setattr(
        "mempalace.recall_llm._get_llm_config",
        lambda: {"backend": "stub", "model": "h", "key": "k", "endpoint": "e"},
    )
    monkeypatch.setattr("mempalace.recall_llm._call_llm", fake_call_llm)

    provider.on_turn_start(0, "user message", session_id="session-1")
    provider.sync_turn(
        "user message",
        "assistant reply that is substantial enough",
        session_id="session-1",
    )

    provider.shutdown()

    # Verify the dump file was written
    hook_state_dir = provider.resolved_paths.base_dir / "hook_state"
    dump_files = list(hook_state_dir.glob("hermes_save_fail_*.txt"))
    assert len(dump_files) >= 1, f"Expected failure dump file, found: {list(hook_state_dir.iterdir()) if hook_state_dir.exists() else '(dir missing)'}"
    content = dump_files[0].read_text(encoding="utf-8")
    assert "ERROR:" in content
    assert "PROMPT" in content
    assert "RAW RESPONSE" in content


def test_recall_filtered_search_retries_without_filters_on_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When filtered search returns 0 hits, retry without filters."""
    provider = _provider(tmp_path / "profile")
    provider.on_turn_start(0, "find diary", session_id="session-1")
    provider._sessions["session-1"].last_assistant_reply = ""

    call_count = {"n": 0}

    def fake_search_memories(**kwargs):
        call_count["n"] += 1
        if kwargs.get("room") == "diary":
            return {"results": []}  # filtered search: empty
        return {
            "results": [
                {"wing": "wing_default", "room": "general", "text": "Fallback result"}
            ]
        }

    def fake_decide_recall(query, config=None, previous_assistant_context=None, active_context=None):
        return {
            "should_recall": True,
            "reason": "history_reference",
            "query": "diary entries",
            "after": None,
            "filters": {"room": "diary"},
        }

    def fake_build_active_context(self_arg, state):
        return {
            "wing": state.wing,
            "platform": state.platform,
            "palace": {"rooms": ["diary", "general"], "halls": []},
        }

    monkeypatch.setattr(mempalace_plugin, "search_memories", fake_search_memories)
    monkeypatch.setattr("mempalace.recall_llm.is_enabled", lambda: True)
    monkeypatch.setattr("mempalace.recall_llm._get_llm_config", lambda: {"backend": "stub"})
    monkeypatch.setattr("mempalace.recall_llm.decide_recall", fake_decide_recall)
    monkeypatch.setattr(
        mempalace_plugin.MemPalaceMemoryProvider,
        "_build_recall_active_context",
        fake_build_active_context,
    )

    recall = provider.prefetch("find diary", session_id="session-1")

    assert "## MemPalace Recall" in recall
    assert "Fallback result" in recall
    # search_memories should have been called twice: once with filter, once without
    assert call_count["n"] == 2
    provider.shutdown()


def test_on_session_end_flushes_buffered_turns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """on_session_end runs Haiku save on remaining buffered turns."""
    monkeypatch.setenv("MEMPAL_HERMES_SAVE_INTERVAL", "99")  # no periodic trigger
    monkeypatch.setenv("MEMPAL_RECALL_LLM", "1")

    provider = _provider(tmp_path / "profile")

    fixed_json = json.dumps({
        "diary": "Session-end flush test diary entry for testing.",
        "drawers": [{"wing": "wing_coder", "room": "general", "content": "Session-end drawer content for test purposes."}],
        "kg": [],
    })

    def fake_call_llm(config, prompt, max_tokens=None, timeout=None, json_mode=False):
        return fixed_json

    monkeypatch.setattr("mempalace.recall_llm.is_enabled", lambda: True)
    monkeypatch.setattr(
        "mempalace.recall_llm._get_llm_config",
        lambda: {"backend": "stub", "model": "h", "key": "k", "endpoint": "e"},
    )
    monkeypatch.setattr("mempalace.recall_llm._call_llm", fake_call_llm)

    # 2 turns — won't trigger periodic save at interval=99
    for i in range(2):
        provider.on_turn_start(i, f"message {i}", session_id="session-1")
        provider.sync_turn(
            f"message {i}",
            f"reply {i} which is long enough for substance",
            session_id="session-1",
        )

    # Buffer should be non-empty
    assert len(provider._sessions["session-1"].recent_turns) == 2

    # Provide a minimal messages list for on_session_end (under threshold
    # for auto_diary but still triggers Haiku flush).
    provider.on_session_end([
        {"role": "user", "content": "message 0"},
        {"role": "assistant", "content": "reply 0"},
    ])

    collection = _get_collection(provider.resolved_paths.palace_path)
    stored = collection.get(include=["metadatas"])
    haiku_entries = [
        m for m in stored["metadatas"] if m.get("added_by") == "haiku_async_save"
    ]
    assert len(haiku_entries) >= 1, "on_session_end should have flushed buffered turns"
