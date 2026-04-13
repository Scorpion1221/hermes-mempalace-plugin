from __future__ import annotations

import json
import warnings
from pathlib import Path

import chromadb
import pytest
from mempalace.knowledge_graph import KnowledgeGraph

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

    provider._sessions["session-a"].prefetch_future.result(timeout=10)
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
    assert recall.count("\n- [") <= 3


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
