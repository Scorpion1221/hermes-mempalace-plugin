"""Hermes MemPalace memory provider.

This repository mirrors the final in-tree Hermes plugin layout so the directory
can be copied into ``plugins/memory/mempalace/`` in the Hermes monorepo.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import chromadb
except ImportError:  # pragma: no cover - handled by is_available()
    chromadb = None

try:
    from mempalace.config import sanitize_name
    from mempalace.knowledge_graph import KnowledgeGraph
    from mempalace.searcher import search_memories
except ImportError:  # pragma: no cover - handled by is_available()
    KnowledgeGraph = None
    sanitize_name = None
    search_memories = None

try:  # Hermes runtime import.
    from agent.memory_provider import MemoryProvider
except ImportError:  # pragma: no cover - local workspace fallback.
    class MemoryProvider:  # type: ignore[no-redef]
        """Fallback base class for local development outside Hermes."""


logger = logging.getLogger(__name__)

COLLECTION_NAME = "mempalace_drawers"
DEFAULT_BASE_DIR_NAME = "mempalace"
CONFIG_FILE_NAME = "mempalace.json"
WRITE_BLOCKED_CONTEXTS = {"subagent", "cron", "flush"}
FIRST_TURN_RECALL_LIMIT = 3
PREFETCH_RECALL_LIMIT = 5
MAX_RECALL_SNIPPET_CHARS = 240
MAX_SYSTEM_PROMPT_PATH_CHARS = 120
TRIVIAL_USER_MESSAGES = {
    "ok",
    "okay",
    "thanks",
    "thank you",
    "cool",
    "nice",
    "great",
    "sounds good",
}


def _patch_chromadb_pydantic_compat() -> None:
    """Backport Chroma's Pydantic 2.11 compatibility fix for older 0.6.x installs."""

    if chromadb is None:
        return

    try:
        collection_cls = chromadb.types.Collection
    except AttributeError:
        return

    if getattr(collection_cls, "_mempalace_pydantic_compat", False):
        return

    def _get_model_fields(self) -> Dict[Any, Any]:
        try:
            return type(self).model_fields
        except AttributeError:
            return self.__fields__

    collection_cls.get_model_fields = _get_model_fields
    collection_cls._mempalace_pydantic_compat = True


_patch_chromadb_pydantic_compat()

SEARCH_TOOL_SCHEMA = {
    "name": "mempalace_search",
    "description": (
        "Semantic search over stored MemPalace drawers for the active user/profile. "
        "Results are always filtered to the current active wing."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "What to search for in the active wing.",
            },
            "room": {
                "type": "string",
                "description": "Optional room filter inside the active wing.",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of results to return (default 5, max 10).",
            },
        },
        "required": ["query"],
    },
}

KG_QUERY_TOOL_SCHEMA = {
    "name": "mempalace_kg_query",
    "description": (
        "Query structured temporal facts from the MemPalace knowledge graph "
        "backing the active Hermes profile."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "entity": {
                "type": "string",
                "description": "Entity name to query.",
            },
            "direction": {
                "type": "string",
                "description": "Query direction: outgoing, incoming, or both.",
                "enum": ["outgoing", "incoming", "both"],
            },
            "as_of": {
                "type": "string",
                "description": "Optional ISO date filter (YYYY-MM-DD).",
            },
        },
        "required": ["entity"],
    },
}

REMEMBER_TOOL_SCHEMA = {
    "name": "mempalace_remember",
    "description": (
        "Persist a durable user-approved memory into the active MemPalace wing. "
        "Use this for explicit long-term facts or preferences."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "Verbatim memory content to store.",
            },
            "room": {
                "type": "string",
                "description": "Optional room name. Defaults to facts.",
            },
        },
        "required": ["content"],
    },
}


@dataclass(frozen=True)
class ResolvedPaths:
    """All storage paths resolved against the active Hermes profile."""

    hermes_home: Path
    base_dir: Path
    palace_path: Path
    identity_path: Path
    kg_path: Path
    config_path: Path


@dataclass
class SessionState:
    """Per-session runtime state to prevent cross-session cache bleed."""

    session_id: str
    turn_number: int = 0
    wing: str = "wing_default"
    platform: str = "cli"
    user_id: str = ""
    agent_identity: str = ""
    agent_context: str = "primary"
    allow_writes: bool = True
    last_user_query: str = ""
    prefetched_text: str = ""
    prefetch_future: Optional[Future[str]] = None
    pending_write_futures: List[Future[Any]] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)


def _slug(value: str) -> str:
    text = (value or "").strip().lower()
    chars: List[str] = []
    last_was_sep = False
    for char in text:
        if char.isalnum():
            chars.append(char)
            last_was_sep = False
            continue
        if not last_was_sep:
            chars.append("_")
            last_was_sep = True
    slug = "".join(chars).strip("_")
    return slug or "default"


def _resolve_optional_path(raw_value: str | None, default_path: Path, hermes_home: Path) -> Path:
    if not raw_value:
        return default_path
    path = Path(str(raw_value)).expanduser()
    if not path.is_absolute():
        path = hermes_home / path
    return path


def resolve_paths(hermes_home: str | Path) -> ResolvedPaths:
    """Resolve profile-scoped MemPalace paths from ``$HERMES_HOME`` config."""

    hermes_home_path = Path(hermes_home).expanduser()
    config_path = hermes_home_path / CONFIG_FILE_NAME
    config_values: Dict[str, Any] = {}
    if config_path.exists():
        try:
            config_values = json.loads(config_path.read_text())
        except json.JSONDecodeError:
            logger.warning("Invalid JSON in %s; falling back to defaults", config_path)
        except OSError as exc:
            logger.warning("Could not read %s: %s", config_path, exc)

    base_dir = hermes_home_path / DEFAULT_BASE_DIR_NAME
    palace_path = _resolve_optional_path(config_values.get("palace_path"), base_dir / "palace", hermes_home_path)
    identity_path = _resolve_optional_path(
        config_values.get("identity_path"),
        base_dir / "identity.txt",
        hermes_home_path,
    )
    kg_path = _resolve_optional_path(
        config_values.get("kg_path"),
        base_dir / "knowledge_graph.sqlite3",
        hermes_home_path,
    )
    return ResolvedPaths(
        hermes_home=hermes_home_path,
        base_dir=base_dir,
        palace_path=palace_path,
        identity_path=identity_path,
        kg_path=kg_path,
        config_path=config_path,
    )


def _safe_room_name(room: str | None, default: str) -> str:
    candidate = (room or default).strip() or default
    if sanitize_name is not None:
        try:
            return sanitize_name(candidate, "room")
        except ValueError:
            pass
    return _slug(candidate)


def _safe_wing_name(value: str) -> str:
    candidate = value.strip()
    if sanitize_name is not None:
        try:
            return sanitize_name(candidate, "wing")
        except ValueError:
            pass
    return _slug(candidate)


def _truncate(text: str, limit: int = MAX_RECALL_SNIPPET_CHARS) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."


def _normalize_user_message(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def _is_trivial_turn(user_content: str, assistant_content: str) -> bool:
    user_text = _normalize_user_message(user_content)
    assistant_text = _normalize_user_message(assistant_content)
    if not user_text or not assistant_text:
        return True
    return user_text in TRIVIAL_USER_MESSAGES and len(assistant_text) <= 120


def _strip_injected_memory(text: str) -> str:
    lines = []
    skipping = False
    for line in text.splitlines():
        if line.strip() == "<<MEMORY_CONTEXT>>":
            skipping = True
            continue
        if line.strip() == "<</MEMORY_CONTEXT>>":
            skipping = False
            continue
        if skipping:
            continue
        lines.append(line)
    return "\n".join(lines).strip()


class MemPalaceMemoryProvider(MemoryProvider):
    """Hermes memory provider backed by local MemPalace storage."""

    def __init__(self) -> None:
        self._paths: Optional[ResolvedPaths] = None
        self._default_session_id = ""
        self._current_session_id = ""
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="mempalace")
        self._sessions: Dict[str, SessionState] = {}
        self._sessions_lock = threading.Lock()
        self._agent_context = "primary"
        self._agent_identity = ""
        self._platform = "cli"
        self._user_id = ""

    @property
    def name(self) -> str:
        return "mempalace"

    @property
    def resolved_paths(self) -> Optional[ResolvedPaths]:
        return self._paths

    def is_available(self) -> bool:
        """Check that the local MemPalace dependencies are importable."""

        return all(
            dependency is not None
            for dependency in (chromadb, KnowledgeGraph, sanitize_name, search_memories)
        )

    def get_config_schema(self) -> List[Dict[str, Any]]:
        return [
            {
                "key": "palace_path",
                "description": (
                    "Optional custom MemPalace Chroma directory. Blank uses "
                    "$HERMES_HOME/mempalace/palace."
                ),
            },
            {
                "key": "identity_path",
                "description": (
                    "Optional custom MemPalace identity file. Blank uses "
                    "$HERMES_HOME/mempalace/identity.txt."
                ),
            },
            {
                "key": "kg_path",
                "description": (
                    "Optional custom MemPalace knowledge graph SQLite path. Blank uses "
                    "$HERMES_HOME/mempalace/knowledge_graph.sqlite3."
                ),
            },
        ]

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        hermes_home_path = Path(hermes_home).expanduser()
        hermes_home_path.mkdir(parents=True, exist_ok=True)
        config_path = hermes_home_path / CONFIG_FILE_NAME
        existing: Dict[str, Any] = {}
        if config_path.exists():
            try:
                existing = json.loads(config_path.read_text())
            except json.JSONDecodeError:
                existing = {}
        existing.update(
            {
                key: value
                for key, value in values.items()
                if key in {"palace_path", "identity_path", "kg_path"} and value
            }
        )
        config_path.write_text(json.dumps(existing, indent=2, sort_keys=True) + "\n")

    def initialize(self, session_id: str, **kwargs) -> None:
        if not self.is_available():
            logger.debug("MemPalace provider inactive; dependencies are unavailable")
            return

        hermes_home = kwargs.get("hermes_home")
        if not hermes_home:
            raise ValueError("initialize() requires hermes_home")

        self._paths = resolve_paths(hermes_home)
        self._paths.base_dir.mkdir(parents=True, exist_ok=True)
        self._paths.palace_path.mkdir(parents=True, exist_ok=True)
        self._paths.identity_path.parent.mkdir(parents=True, exist_ok=True)
        self._paths.kg_path.parent.mkdir(parents=True, exist_ok=True)
        self._paths.identity_path.touch(exist_ok=True)

        self._default_session_id = session_id
        self._current_session_id = session_id
        self._agent_context = str(kwargs.get("agent_context") or "primary")
        self._agent_identity = str(kwargs.get("agent_identity") or "")
        self._platform = str(kwargs.get("platform") or "cli")
        self._user_id = str(kwargs.get("user_id") or "")

        self._ensure_session_state(session_id, **kwargs)

    def system_prompt_block(self) -> str:
        if self._paths is None:
            return ""
        state = self._get_session_state()
        wing = state.wing if state else "wing_default"
        return "\n".join(
            [
                "## MemPalace Memory",
                f"- active wing: `{wing}`",
                f"- palace path: `{_truncate(str(self._paths.palace_path), MAX_SYSTEM_PROMPT_PATH_CHARS)}`",
                f"- kg path: `{_truncate(str(self._paths.kg_path), MAX_SYSTEM_PROMPT_PATH_CHARS)}`",
                "- tools: mempalace_search, mempalace_kg_query, mempalace_remember",
                "- use mempalace_remember only for durable facts worth preserving across sessions",
            ]
        )

    def on_turn_start(self, turn_number: int, message: str, **kwargs) -> None:
        session_id = str(kwargs.get("session_id") or self._default_session_id)
        state_kwargs = dict(kwargs)
        state_kwargs.pop("session_id", None)
        state = self._ensure_session_state(session_id, **state_kwargs)
        with state.lock:
            state.turn_number = int(turn_number)
            state.last_user_query = message or ""
            state.prefetched_text = ""
            state.pending_write_futures = [
                future for future in state.pending_write_futures if not future.done()
            ]
        self._current_session_id = session_id

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        state = self._get_session_state(session_id)
        if state is None:
            return ""
        if query:
            state.last_user_query = query

        with state.lock:
            future = state.prefetch_future
            cached = state.prefetched_text

        if future is not None and future.done():
            try:
                cached = future.result()
            except Exception as exc:  # pragma: no cover - defensive logging.
                logger.warning("Prefetch future failed for %s: %s", state.session_id, exc)
                cached = ""
            with state.lock:
                state.prefetched_text = cached
                state.prefetch_future = None

        if cached:
            return cached

        if state.turn_number <= 0:
            return self._render_recall(query or state.last_user_query, state, FIRST_TURN_RECALL_LIMIT)

        return ""

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        state = self._get_session_state(session_id)
        if state is None or not query.strip():
            return

        state.last_user_query = query
        future = self._executor.submit(
            self._render_recall,
            query,
            state,
            PREFETCH_RECALL_LIMIT,
        )
        with state.lock:
            state.prefetch_future = future

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        state = self._get_session_state(session_id)
        if state is None or not state.allow_writes:
            return
        if _is_trivial_turn(user_content, assistant_content):
            return

        with state.lock:
            if state.turn_number < 0:
                state.turn_number = 0
            turn_number = state.turn_number

        future = self._executor.submit(
            self._write_turn_drawer,
            state,
            turn_number,
            user_content,
            assistant_content,
        )
        with state.lock:
            state.pending_write_futures.append(future)

    def on_memory_write(self, action: str, target: str, content: str) -> None:
        state = self._get_session_state()
        if state is None or not state.allow_writes:
            return
        if action not in {"add", "replace"} or not content.strip():
            return

        room = "builtin_memory"
        title = f"[builtin-memory {target}/{action}]"
        document = f"{title}\n{content.strip()}"
        drawer_id = self._content_drawer_id(state.wing, room, document)
        future = self._executor.submit(
            self._upsert_drawer,
            drawer_id,
            document,
            room,
            state,
        )
        with state.lock:
            state.pending_write_futures.append(future)

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        del messages  # The V1 provider only flushes queued writes.
        session_id = self._current_session_id or self._default_session_id
        if not session_id:
            return
        self._flush_session(session_id)
        with self._sessions_lock:
            self._sessions.pop(session_id, None)

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [SEARCH_TOOL_SCHEMA, KG_QUERY_TOOL_SCHEMA, REMEMBER_TOOL_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        state = self._get_session_state(kwargs.get("session_id", ""))
        if state is None:
            return self._json_result({"error": "provider_not_initialized"})

        if tool_name == "mempalace_search":
            limit = self._bounded_limit(args.get("limit"), default=5)
            room = _safe_room_name(args.get("room"), "general") if args.get("room") else None
            # Search across all wings for broader recall; wing info is
            # returned per-result so the caller can still distinguish origin.
            result = search_memories(
                str(args.get("query", "")),
                palace_path=str(self._paths.palace_path),
                wing=None,
                room=room,
                n_results=limit,
            )
            return self._json_result(result)

        if tool_name == "mempalace_kg_query":
            direction = str(args.get("direction") or "outgoing")
            if direction not in {"outgoing", "incoming", "both"}:
                direction = "outgoing"
            kg = KnowledgeGraph(db_path=str(self._paths.kg_path))
            try:
                result = {
                    "entity": str(args.get("entity", "")),
                    "direction": direction,
                    "as_of": args.get("as_of"),
                    "results": kg.query_entity(
                        str(args.get("entity", "")),
                        as_of=args.get("as_of"),
                        direction=direction,
                    ),
                }
            finally:
                kg.close()
            return self._json_result(result)

        if tool_name == "mempalace_remember":
            if not state.allow_writes:
                return self._json_result({"success": False, "reason": "writes_disabled"})
            content = str(args.get("content", "")).strip()
            if not content:
                return self._json_result({"success": False, "error": "content is required"})
            room = _safe_room_name(args.get("room"), "facts")
            drawer_id = self._content_drawer_id(state.wing, room, content)
            self._upsert_drawer(drawer_id, content, room, state)
            return self._json_result(
                {
                    "success": True,
                    "drawer_id": drawer_id,
                    "wing": state.wing,
                    "room": room,
                }
            )

        return self._json_result({"error": f"unknown_tool:{tool_name}"})

    def shutdown(self) -> None:
        for session_id in list(self._sessions):
            self._flush_session(session_id)
        self._executor.shutdown(wait=True, cancel_futures=False)

    def _ensure_session_state(self, session_id: str, **kwargs) -> SessionState:
        if not session_id:
            session_id = self._default_session_id or "default-session"
        if self._paths is None:
            raise RuntimeError("Provider not initialized")

        with self._sessions_lock:
            state = self._sessions.get(session_id)
            if state is None:
                state = SessionState(session_id=session_id)
                self._sessions[session_id] = state

        user_id = str(kwargs.get("user_id") or state.user_id or self._user_id or "")
        agent_identity = str(
            kwargs.get("agent_identity")
            or state.agent_identity
            or self._agent_identity
            or self._paths.hermes_home.name
        )
        agent_context = str(kwargs.get("agent_context") or state.agent_context or self._agent_context)
        platform = str(kwargs.get("platform") or state.platform or self._platform)

        # Allow wing override from mempalace.json config (e.g. "default_wing": "solvely_web")
        config_wing = None
        if self._paths and self._paths.config_path.exists():
            try:
                _cfg = json.loads(self._paths.config_path.read_text())
                config_wing = _cfg.get("default_wing")
            except (json.JSONDecodeError, OSError):
                pass
        if config_wing:
            wing = _safe_wing_name(config_wing)
        else:
            wing_seed = user_id or agent_identity or session_id or self._paths.hermes_home.name
            wing = _safe_wing_name(f"wing_{_slug(wing_seed)}")

        with state.lock:
            state.wing = wing
            state.user_id = user_id
            state.agent_identity = agent_identity
            state.agent_context = agent_context
            state.platform = platform
            state.allow_writes = agent_context not in WRITE_BLOCKED_CONTEXTS

        return state

    def _get_session_state(self, session_id: str = "") -> Optional[SessionState]:
        if self._paths is None:
            return None
        target_session_id = str(session_id or self._current_session_id or self._default_session_id)
        if not target_session_id:
            return None
        state = self._sessions.get(target_session_id)
        if state is not None:
            return state
        return self._ensure_session_state(target_session_id)

    def _bounded_limit(self, raw_value: Any, default: int) -> int:
        try:
            value = int(raw_value)
        except (TypeError, ValueError):
            value = default
        return max(1, min(value, 10))

    def _render_recall(self, query: str, state: SessionState, limit: int) -> str:
        if not query.strip() or self._paths is None:
            return ""

        result = search_memories(
            query=query,
            palace_path=str(self._paths.palace_path),
            wing=None,  # search all wings for broader recall
            n_results=limit,
        )
        hits = result.get("results", []) if isinstance(result, dict) else []
        if not hits:
            return ""

        lines = ["## MemPalace Recall"]
        for hit in hits[:limit]:
            room = hit.get("room", "general")
            snippet = _truncate(hit.get("text", ""))
            if snippet:
                lines.append(f"- [{room}] {snippet}")
        return "\n".join(lines)

    def _write_turn_drawer(
        self,
        state: SessionState,
        turn_number: int,
        user_content: str,
        assistant_content: str,
    ) -> None:
        room = "general"
        drawer_id = self._turn_drawer_id(state.wing, state.session_id, turn_number, room)
        cleaned_user = _strip_injected_memory(user_content)
        cleaned_assistant = _strip_injected_memory(assistant_content)
        document = (
            "[role: user]\n"
            f"{cleaned_user}\n\n"
            "[role: assistant]\n"
            f"{cleaned_assistant}"
        ).strip()
        self._upsert_drawer(drawer_id, document, room, state, turn_number=turn_number)

    def _upsert_drawer(
        self,
        drawer_id: str,
        content: str,
        room: str,
        state: SessionState,
        *,
        turn_number: Optional[int] = None,
    ) -> None:
        collection = self._get_collection(create=True)
        metadata = {
            "wing": state.wing,
            "room": room,
            "source_file": "",
            "chunk_index": 0,
            "added_by": "hermes-mempalace",
            "filed_at": datetime.now().isoformat(),
            "session_id": state.session_id,
            "turn_number": int(turn_number if turn_number is not None else state.turn_number),
            "platform": state.platform,
            "user_id": state.user_id,
            "agent_identity": state.agent_identity,
            "agent_context": state.agent_context,
        }
        collection.upsert(ids=[drawer_id], documents=[content], metadatas=[metadata])

    def _get_collection(self, *, create: bool):
        if chromadb is None or self._paths is None:
            raise RuntimeError("MemPalace provider is unavailable")
        client = chromadb.PersistentClient(path=str(self._paths.palace_path))
        if create:
            return client.get_or_create_collection(COLLECTION_NAME)
        return client.get_collection(COLLECTION_NAME)

    def _content_drawer_id(self, wing: str, room: str, content: str) -> str:
        digest = hashlib.sha256(f"{wing}:{room}:{content}".encode("utf-8")).hexdigest()[:24]
        return f"drawer_{_slug(wing)}_{_slug(room)}_{digest}"

    def _turn_drawer_id(self, wing: str, session_id: str, turn_number: int, room: str) -> str:
        digest = hashlib.sha256(
            f"{session_id}:{turn_number}:{wing}:{room}".encode("utf-8")
        ).hexdigest()[:24]
        return f"drawer_{_slug(wing)}_{_slug(room)}_{digest}"

    def _flush_session(self, session_id: str) -> None:
        state = self._sessions.get(session_id)
        if state is None:
            return
        with state.lock:
            futures = list(state.pending_write_futures)
            prefetch_future = state.prefetch_future
            state.pending_write_futures = []
            state.prefetch_future = None
        if prefetch_future is not None:
            try:
                prefetch_future.result(timeout=10)
            except Exception as exc:  # pragma: no cover - defensive logging.
                logger.debug("Ignoring prefetch shutdown failure for %s: %s", session_id, exc)
        for future in futures:
            try:
                future.result(timeout=10)
            except Exception as exc:  # pragma: no cover - defensive logging.
                logger.warning("Write flush failed for %s: %s", session_id, exc)

    def _json_result(self, payload: Dict[str, Any]) -> str:
        return json.dumps(payload, ensure_ascii=True, sort_keys=True)


def register(ctx) -> None:
    """Hermes plugin discovery entrypoint."""

    ctx.register_memory_provider(MemPalaceMemoryProvider())


__all__ = ["MemPalaceMemoryProvider", "ResolvedPaths", "register", "resolve_paths"]
