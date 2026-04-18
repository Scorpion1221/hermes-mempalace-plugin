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
    assert _collection_count(provider.resolved_paths.palace_path) == 1


def test_prefetch_cache_is_session_keyed_and_wing_scoped(tmp_path: Path) -> None:
    hermes_home = tmp_path / "profile"
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

    collection = _get_collection(provider.resolved_paths.palace_path)
    stored = collection.get(include=["documents", "metadatas"])
    assert len(stored["ids"]) == 1
    assert stored["metadatas"][0]["turn_number"] == 7


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

    def fake_rewrite_query(query, config=None, previous_assistant_context=None):
        calls["rewrite"] = previous_assistant_context
        return {"query": "mempalace hooks", "after": None}

    def fake_rerank(query, hits, top_k=5, config=None, previous_assistant_context=None):
        calls["rerank"] = previous_assistant_context
        return hits[:top_k]

    monkeypatch.setattr(mempalace_plugin, "search_memories", fake_search_memories)
    monkeypatch.setattr("mempalace.recall_llm.is_enabled", lambda: True)
    monkeypatch.setattr("mempalace.recall_llm._get_llm_config", lambda: {"backend": "stub"})
    monkeypatch.setattr("mempalace.recall_llm.rewrite_query", fake_rewrite_query)
    monkeypatch.setattr("mempalace.recall_llm.rerank", fake_rerank)

    recall = provider.prefetch("why?", session_id="session-1")

    assert "## MemPalace Recall" in recall
    assert calls["rewrite"] == {
        "tail": "Earlier I explained the MemPalace Claude and Codex hooks."
    }
    assert calls["rerank"] == {
        "tail": "Earlier I explained the MemPalace Claude and Codex hooks."
    }
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
