"""Hermes MemPalace memory provider v2.0.

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
    from mempalace.config import sanitize_content, sanitize_name
    from mempalace.knowledge_graph import KnowledgeGraph
    from mempalace.searcher import search_memories
except ImportError:  # pragma: no cover - handled by is_available()
    KnowledgeGraph = None
    sanitize_name = None
    sanitize_content = None
    search_memories = None

try:
    from mempalace.palace_graph import (
        create_tunnel,
        delete_tunnel,
        find_tunnels,
        follow_tunnels,
        graph_stats,
        list_tunnels,
        traverse,
    )
except ImportError:  # pragma: no cover
    create_tunnel = delete_tunnel = find_tunnels = follow_tunnels = None
    graph_stats = list_tunnels = traverse = None

try:
    from mempalace.fact_checker import check_text as fact_check_text
except ImportError:  # pragma: no cover
    fact_check_text = None

try:
    from mempalace.config import MempalaceConfig
except ImportError:  # pragma: no cover
    MempalaceConfig = None

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
FIRST_TURN_RECALL_LIMIT = 5
PREFETCH_RECALL_LIMIT = 5
RECALL_POOL = 10  # over-fetch for LLM reranking
MAX_RECALL_SNIPPET_CHARS = 400
PREVIOUS_ASSISTANT_TAIL_CHARS = 500
MAX_SYSTEM_PROMPT_PATH_CHARS = 120
TRIVIAL_USER_MESSAGES = frozenset({
    # Greetings
    "hi", "hello", "hey", "嗨", "你好",
    # Acknowledgement
    "ok", "okay", "好", "好的", "行", "嗯", "对",
    "cool", "nice", "great", "sounds good",
    # Affirmation / negation
    "yes", "no", "是", "是的", "不", "不是",
    # Continuation
    "continue", "go", "go on", "next", "继续",
    # Gratitude
    "thanks", "thank you", "thx", "谢谢",
    # Completion / exit
    "done", "完成", "搞定", "stop", "quit", "exit",
})
MIN_RECALL_QUERY_LEN = 6  # skip very short prompts from recall
CONTEXTUAL_FOLLOWUP_MESSAGES = frozenset({
    "continue", "go", "go on", "next", "继续",
})
HARD_SKIP_USER_MESSAGES = TRIVIAL_USER_MESSAGES - CONTEXTUAL_FOLLOWUP_MESSAGES

ALL_TOOL_NAMES = [
    "mempalace_search",
    "mempalace_kg_query",
    "mempalace_remember",
    "mempalace_kg_add",
    "mempalace_kg_invalidate",
    "mempalace_kg_timeline",
    "mempalace_kg_stats",
    "mempalace_status",
    "mempalace_list_wings",
    "mempalace_list_rooms",
    "mempalace_get_taxonomy",
    "mempalace_traverse",
    "mempalace_find_tunnels",
    "mempalace_graph_stats",
    "mempalace_create_tunnel",
    "mempalace_list_tunnels",
    "mempalace_delete_tunnel",
    "mempalace_follow_tunnels",
    "mempalace_add_drawer",
    "mempalace_delete_drawer",
    "mempalace_get_drawer",
    "mempalace_list_drawers",
    "mempalace_update_drawer",
    "mempalace_diary_write",
    "mempalace_diary_read",
    "mempalace_check_duplicate",
    "mempalace_check_facts",
    "mempalace_hook_settings",
    "mempalace_reconnect",
    "mempalace_get_aaak_spec",
    "mempalace_memories_filed_away",
]


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

# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

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

KG_ADD_TOOL_SCHEMA = {
    "name": "mempalace_kg_add",
    "description": "Add a triple (subject, predicate, object) to the knowledge graph.",
    "parameters": {
        "type": "object",
        "properties": {
            "subject": {"type": "string", "description": "Subject entity."},
            "predicate": {"type": "string", "description": "Relationship predicate."},
            "object": {"type": "string", "description": "Object entity."},
            "valid_from": {"type": "string", "description": "Optional ISO date start."},
        },
        "required": ["subject", "predicate", "object"],
    },
}

KG_INVALIDATE_TOOL_SCHEMA = {
    "name": "mempalace_kg_invalidate",
    "description": "Invalidate (end-date) a triple in the knowledge graph.",
    "parameters": {
        "type": "object",
        "properties": {
            "subject": {"type": "string", "description": "Subject entity."},
            "predicate": {"type": "string", "description": "Relationship predicate."},
            "object": {"type": "string", "description": "Object entity."},
            "ended": {"type": "string", "description": "Optional ISO date of invalidation."},
        },
        "required": ["subject", "predicate", "object"],
    },
}

KG_TIMELINE_TOOL_SCHEMA = {
    "name": "mempalace_kg_timeline",
    "description": "Return the temporal timeline of an entity in the knowledge graph.",
    "parameters": {
        "type": "object",
        "properties": {
            "entity": {"type": "string", "description": "Optional entity name. If omitted, returns full timeline."},
        },
        "required": [],
    },
}

KG_STATS_TOOL_SCHEMA = {
    "name": "mempalace_kg_stats",
    "description": "Return statistics about the knowledge graph.",
    "parameters": {"type": "object", "properties": {}, "required": []},
}

STATUS_TOOL_SCHEMA = {
    "name": "mempalace_status",
    "description": "Return current MemPalace provider status, paths, and wing info.",
    "parameters": {"type": "object", "properties": {}, "required": []},
}

LIST_WINGS_TOOL_SCHEMA = {
    "name": "mempalace_list_wings",
    "description": "List all wings present in the palace collection.",
    "parameters": {"type": "object", "properties": {}, "required": []},
}

LIST_ROOMS_TOOL_SCHEMA = {
    "name": "mempalace_list_rooms",
    "description": "List all rooms, optionally filtered by wing.",
    "parameters": {
        "type": "object",
        "properties": {
            "wing": {"type": "string", "description": "Optional wing filter."},
        },
        "required": [],
    },
}

GET_TAXONOMY_TOOL_SCHEMA = {
    "name": "mempalace_get_taxonomy",
    "description": "Return the full wing/room taxonomy with counts.",
    "parameters": {"type": "object", "properties": {}, "required": []},
}

TRAVERSE_TOOL_SCHEMA = {
    "name": "mempalace_traverse",
    "description": "Graph traversal from a starting room.",
    "parameters": {
        "type": "object",
        "properties": {
            "start_room": {"type": "string", "description": "Room to start traversal from."},
            "max_hops": {"type": "integer", "description": "Max hops (default 2)."},
        },
        "required": ["start_room"],
    },
}

FIND_TUNNELS_TOOL_SCHEMA = {
    "name": "mempalace_find_tunnels",
    "description": "Find tunnels connecting two wings.",
    "parameters": {
        "type": "object",
        "properties": {
            "wing_a": {"type": "string", "description": "First wing."},
            "wing_b": {"type": "string", "description": "Second wing."},
        },
        "required": [],
    },
}

GRAPH_STATS_TOOL_SCHEMA = {
    "name": "mempalace_graph_stats",
    "description": "Return palace graph statistics (tunnels, rooms, connections).",
    "parameters": {"type": "object", "properties": {}, "required": []},
}

CREATE_TUNNEL_TOOL_SCHEMA = {
    "name": "mempalace_create_tunnel",
    "description": "Create a tunnel between two rooms.",
    "parameters": {
        "type": "object",
        "properties": {
            "source_wing": {"type": "string", "description": "Source wing."},
            "source_room": {"type": "string", "description": "Source room."},
            "target_wing": {"type": "string", "description": "Target wing."},
            "target_room": {"type": "string", "description": "Target room."},
            "label": {"type": "string", "description": "Optional tunnel label."},
            "source_drawer_id": {"type": "string", "description": "Optional source drawer ID."},
            "target_drawer_id": {"type": "string", "description": "Optional target drawer ID."},
        },
        "required": ["source_wing", "source_room", "target_wing", "target_room"],
    },
}

LIST_TUNNELS_TOOL_SCHEMA = {
    "name": "mempalace_list_tunnels",
    "description": "List tunnels, optionally filtered by wing.",
    "parameters": {
        "type": "object",
        "properties": {
            "wing": {"type": "string", "description": "Optional wing filter."},
        },
        "required": [],
    },
}

DELETE_TUNNEL_TOOL_SCHEMA = {
    "name": "mempalace_delete_tunnel",
    "description": "Delete a tunnel by ID.",
    "parameters": {
        "type": "object",
        "properties": {
            "tunnel_id": {"type": "string", "description": "Tunnel ID to delete."},
        },
        "required": ["tunnel_id"],
    },
}

FOLLOW_TUNNELS_TOOL_SCHEMA = {
    "name": "mempalace_follow_tunnels",
    "description": "Follow tunnels from a wing/room.",
    "parameters": {
        "type": "object",
        "properties": {
            "wing": {"type": "string", "description": "Wing to start from."},
            "room": {"type": "string", "description": "Room to start from."},
        },
        "required": ["wing", "room"],
    },
}

ADD_DRAWER_TOOL_SCHEMA = {
    "name": "mempalace_add_drawer",
    "description": "Add a new drawer to the palace.",
    "parameters": {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "Drawer content."},
            "room": {"type": "string", "description": "Room name (default: general)."},
            "wing": {"type": "string", "description": "Optional wing override."},
            "drawer_id": {"type": "string", "description": "Optional custom drawer ID."},
        },
        "required": ["content"],
    },
}

DELETE_DRAWER_TOOL_SCHEMA = {
    "name": "mempalace_delete_drawer",
    "description": "Delete a drawer by ID.",
    "parameters": {
        "type": "object",
        "properties": {
            "drawer_id": {"type": "string", "description": "Drawer ID to delete."},
        },
        "required": ["drawer_id"],
    },
}

GET_DRAWER_TOOL_SCHEMA = {
    "name": "mempalace_get_drawer",
    "description": "Retrieve a specific drawer by ID.",
    "parameters": {
        "type": "object",
        "properties": {
            "drawer_id": {"type": "string", "description": "Drawer ID to retrieve."},
        },
        "required": ["drawer_id"],
    },
}

LIST_DRAWERS_TOOL_SCHEMA = {
    "name": "mempalace_list_drawers",
    "description": "List drawers, optionally filtered by wing and room.",
    "parameters": {
        "type": "object",
        "properties": {
            "wing": {"type": "string", "description": "Optional wing filter."},
            "room": {"type": "string", "description": "Optional room filter."},
            "limit": {"type": "integer", "description": "Max results (default 20)."},
        },
        "required": [],
    },
}

UPDATE_DRAWER_TOOL_SCHEMA = {
    "name": "mempalace_update_drawer",
    "description": "Update an existing drawer's content.",
    "parameters": {
        "type": "object",
        "properties": {
            "drawer_id": {"type": "string", "description": "Drawer ID to update."},
            "content": {"type": "string", "description": "New content."},
        },
        "required": ["drawer_id", "content"],
    },
}

DIARY_WRITE_TOOL_SCHEMA = {
    "name": "mempalace_diary_write",
    "description": "Write a diary entry for today. Appends to same-day entry if one exists.",
    "parameters": {
        "type": "object",
        "properties": {
            "entry": {"type": "string", "description": "Diary entry text."},
        },
        "required": ["entry"],
    },
}

DIARY_READ_TOOL_SCHEMA = {
    "name": "mempalace_diary_read",
    "description": "Read recent diary entries.",
    "parameters": {
        "type": "object",
        "properties": {
            "limit": {"type": "integer", "description": "Max entries to return (default 5)."},
        },
        "required": [],
    },
}

CHECK_DUPLICATE_TOOL_SCHEMA = {
    "name": "mempalace_check_duplicate",
    "description": "Check if content already exists as a drawer (duplicate detection).",
    "parameters": {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "Content to check for duplicates."},
            "room": {"type": "string", "description": "Optional room filter."},
        },
        "required": ["content"],
    },
}

CHECK_FACTS_TOOL_SCHEMA = {
    "name": "mempalace_check_facts",
    "description": "Fact-check text against stored knowledge.",
    "parameters": {
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "Text to fact-check."},
        },
        "required": ["text"],
    },
}

HOOK_SETTINGS_TOOL_SCHEMA = {
    "name": "mempalace_hook_settings",
    "description": "Get or set MemPalace hook settings (silent_save, desktop_toast).",
    "parameters": {
        "type": "object",
        "properties": {
            "key": {"type": "string", "description": "Setting key to set. Omit to list all."},
            "value": {"type": "boolean", "description": "Value to set."},
        },
        "required": [],
    },
}

RECONNECT_TOOL_SCHEMA = {
    "name": "mempalace_reconnect",
    "description": "Reconnect the ChromaDB client (useful after errors).",
    "parameters": {"type": "object", "properties": {}, "required": []},
}

GET_AAAK_SPEC_TOOL_SCHEMA = {
    "name": "mempalace_get_aaak_spec",
    "description": "Return the AAAK (Agent-Accessible API Keys) specification.",
    "parameters": {"type": "object", "properties": {}, "required": []},
}

MEMORIES_FILED_AWAY_TOOL_SCHEMA = {
    "name": "mempalace_memories_filed_away",
    "description": "Return the count of memories filed in this session.",
    "parameters": {"type": "object", "properties": {}, "required": []},
}

_ALL_TOOL_SCHEMAS = [
    SEARCH_TOOL_SCHEMA,
    KG_QUERY_TOOL_SCHEMA,
    REMEMBER_TOOL_SCHEMA,
    KG_ADD_TOOL_SCHEMA,
    KG_INVALIDATE_TOOL_SCHEMA,
    KG_TIMELINE_TOOL_SCHEMA,
    KG_STATS_TOOL_SCHEMA,
    STATUS_TOOL_SCHEMA,
    LIST_WINGS_TOOL_SCHEMA,
    LIST_ROOMS_TOOL_SCHEMA,
    GET_TAXONOMY_TOOL_SCHEMA,
    TRAVERSE_TOOL_SCHEMA,
    FIND_TUNNELS_TOOL_SCHEMA,
    GRAPH_STATS_TOOL_SCHEMA,
    CREATE_TUNNEL_TOOL_SCHEMA,
    LIST_TUNNELS_TOOL_SCHEMA,
    DELETE_TUNNEL_TOOL_SCHEMA,
    FOLLOW_TUNNELS_TOOL_SCHEMA,
    ADD_DRAWER_TOOL_SCHEMA,
    DELETE_DRAWER_TOOL_SCHEMA,
    GET_DRAWER_TOOL_SCHEMA,
    LIST_DRAWERS_TOOL_SCHEMA,
    UPDATE_DRAWER_TOOL_SCHEMA,
    DIARY_WRITE_TOOL_SCHEMA,
    DIARY_READ_TOOL_SCHEMA,
    CHECK_DUPLICATE_TOOL_SCHEMA,
    CHECK_FACTS_TOOL_SCHEMA,
    HOOK_SETTINGS_TOOL_SCHEMA,
    RECONNECT_TOOL_SCHEMA,
    GET_AAAK_SPEC_TOOL_SCHEMA,
    MEMORIES_FILED_AWAY_TOOL_SCHEMA,
]

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


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
    last_assistant_reply: str = ""
    prefetched_text: str = ""
    prefetch_future: Optional[Future[str]] = None
    pending_write_futures: List[Future[Any]] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)
    memories_filed: int = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


def _tail_chars(text: str, limit: int) -> str:
    if not text or limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    return text[-limit:]


def _session_cache_filename(session_id: str) -> str:
    slug = _slug(session_id)
    digest = hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:12]
    return f"{slug}_{digest}_last_assistant.txt"


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


def _safe_content(value: str, max_length: int = 100_000) -> str:
    """Sanitize content using mempalace's sanitize_content or fallback truncation."""
    if sanitize_content is not None:
        try:
            return sanitize_content(value, max_length=max_length)
        except (ValueError, TypeError):
            pass
    # Fallback: simple truncation
    if len(value) > max_length:
        return value[:max_length]
    return value


def _bounded_int(raw: Any, default: int, lo: int, hi: int) -> int:
    """Parse an integer with bounds clamping."""
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = default
    return max(lo, min(value, hi))


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


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
        self._chroma_client: Any = None
        self._chroma_lock = threading.Lock()
        self._cached_config: Optional[Dict[str, Any]] = None

        # Build dispatch table
        self._tool_dispatch: Dict[str, Any] = {
            "mempalace_search": self._tool_search,
            "mempalace_kg_query": self._tool_kg_query,
            "mempalace_remember": self._tool_remember,
            "mempalace_kg_add": self._tool_kg_add,
            "mempalace_kg_invalidate": self._tool_kg_invalidate,
            "mempalace_kg_timeline": self._tool_kg_timeline,
            "mempalace_kg_stats": self._tool_kg_stats,
            "mempalace_status": self._tool_status,
            "mempalace_list_wings": self._tool_list_wings,
            "mempalace_list_rooms": self._tool_list_rooms,
            "mempalace_get_taxonomy": self._tool_get_taxonomy,
            "mempalace_traverse": self._tool_traverse,
            "mempalace_find_tunnels": self._tool_find_tunnels,
            "mempalace_graph_stats": self._tool_graph_stats,
            "mempalace_create_tunnel": self._tool_create_tunnel,
            "mempalace_list_tunnels": self._tool_list_tunnels,
            "mempalace_delete_tunnel": self._tool_delete_tunnel,
            "mempalace_follow_tunnels": self._tool_follow_tunnels,
            "mempalace_add_drawer": self._tool_add_drawer,
            "mempalace_delete_drawer": self._tool_delete_drawer,
            "mempalace_get_drawer": self._tool_get_drawer,
            "mempalace_list_drawers": self._tool_list_drawers,
            "mempalace_update_drawer": self._tool_update_drawer,
            "mempalace_diary_write": self._tool_diary_write,
            "mempalace_diary_read": self._tool_diary_read,
            "mempalace_check_duplicate": self._tool_check_duplicate,
            "mempalace_check_facts": self._tool_check_facts,
            "mempalace_hook_settings": self._tool_hook_settings,
            "mempalace_reconnect": self._tool_reconnect,
            "mempalace_get_aaak_spec": self._tool_get_aaak_spec,
            "mempalace_memories_filed_away": self._tool_memories_filed_away,
        }

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

        # Cache config from mempalace.json once at init time.
        self._cached_config = {}
        if self._paths.config_path.exists():
            try:
                self._cached_config = json.loads(self._paths.config_path.read_text())
            except (json.JSONDecodeError, OSError):
                pass

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
        tools_line = ", ".join(ALL_TOOL_NAMES)
        return "\n".join(
            [
                "## MemPalace Memory",
                f"- active wing: `{wing}`",
                f"- palace path: `{_truncate(str(self._paths.palace_path), MAX_SYSTEM_PROMPT_PATH_CHARS)}`",
                f"- kg path: `{_truncate(str(self._paths.kg_path), MAX_SYSTEM_PROMPT_PATH_CHARS)}`",
                f"- tools: {tools_line}",
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

        # Always fall back to synchronous recall when no prefetch cache is
        # available.  Previously this was gated on turn_number <= 0 which
        # meant later turns with no cache silently returned "".  The query
        # passed here is the *current* user message (or the enriched
        # persist_user_message that includes reply-to context), so recall
        # is always relevant to what the user just said.
        return self._render_recall(query or state.last_user_query, state, FIRST_TURN_RECALL_LIMIT)

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        state = self._get_session_state(session_id)
        if state is None or not query.strip():
            return

        # Skip recall for trivial prompts
        normalized = _normalize_user_message(query)
        previous_assistant_tail = self._previous_assistant_tail(state)
        if normalized in HARD_SKIP_USER_MESSAGES:
            return
        if len(normalized) < MIN_RECALL_QUERY_LEN and not previous_assistant_tail:
            return
        try:
            from mempalace.recall_llm import local_recall_decision

            local_decision = local_recall_decision(
                query,
                previous_assistant_context={"tail": previous_assistant_tail},
                active_context={"wing": state.wing, "platform": state.platform},
            )
            if local_decision and not local_decision.get("should_recall"):
                return
        except Exception:
            pass

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
        cleaned_assistant = _strip_injected_memory(assistant_content)
        with state.lock:
            state.last_assistant_reply = cleaned_assistant
        self._write_assistant_cache(state.session_id, cleaned_assistant)
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
        session_id = self._current_session_id or self._default_session_id
        if not session_id:
            return

        # --- Auto-diary: write a natural language session summary ---
        state = self._sessions.get(session_id)
        if state and state.allow_writes and messages and len(messages) > 2:
            try:
                self._auto_diary(state, messages)
            except Exception as exc:  # pragma: no cover - best-effort
                logger.debug("auto-diary failed: %s", exc)

        self._flush_session(session_id)
        self._clear_assistant_cache(session_id)
        with self._sessions_lock:
            self._sessions.pop(session_id, None)

    # ------------------------------------------------------------------
    # Auto-diary: write a compact session summary on session end
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_msg_text(msg: dict) -> str:
        """Extract plain text from a message, handling str and list content."""
        content = msg.get("content", "")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            return " ".join(
                b.get("text", "") for b in content if isinstance(b, dict)
            ).strip()
        return ""

    @staticmethod
    def _is_user_content(text: str) -> bool:
        """Check if text is meaningful user content (not injected memory/preamble)."""
        return bool(
            text
            and not text.startswith("## MemPalace")
            and not text.startswith("<<MEMORY_CONTEXT>>")
            and not text.startswith("<mempalace-recall>")
            and len(text) > 5
        )

    @staticmethod
    def _extract_keywords(texts: list, max_keywords: int = 10) -> list:
        """Extract salient keywords from user messages for searchability.

        Looks for file paths, technical terms, CJK phrases, and action verbs.
        """
        import re as _re

        keywords = []
        seen = set()
        for text in texts:
            # File paths (e.g. src/foo/bar.py, ~/.hermes/config.json)
            for m in _re.finditer(r'[~/.]?[\w._-]+/[\w._/-]+', text):
                token = m.group().rstrip("/")
                if token not in seen and len(token) > 4:
                    keywords.append(token)
                    seen.add(token)
            # Backtick-quoted identifiers (e.g. `feishu.py`, `hook_stop`)
            for m in _re.finditer(r'`([^`]{2,60})`', text):
                token = m.group(1)
                if token not in seen:
                    keywords.append(token)
                    seen.add(token)
            # CJK key phrases (2-8 chars surrounded by punctuation/space)
            for m in _re.finditer(r'[\u4e00-\u9fff]{2,8}', text):
                token = m.group()
                if token not in seen:
                    keywords.append(token)
                    seen.add(token)
        return keywords[:max_keywords]

    def _auto_diary(self, state: "SessionState", messages: List[Dict[str, Any]]) -> None:
        """Write a natural language diary entry summarising the session.

        Extracts: tool calls, user topics, keywords, decisions, turn count.
        No LLM needed — pure extraction from message history.
        Mirrors the Claude Code Stop-hook pattern but runs server-side.
        Output is plain natural language for better vector search recall.
        """
        agent_name = state.agent_identity or "hermes"
        today = datetime.now().strftime("%Y-%m-%d")
        now_ts = datetime.now().strftime("%H%M")

        # Pass 1: collect structured data from all messages
        user_turns = 0
        tool_names: list = []
        user_texts: list = []
        assistant_texts: list = []
        first_user_msg = ""
        last_user_msg = ""

        for msg in messages:
            role = msg.get("role", "")
            if role == "user":
                user_turns += 1
                text = self._extract_msg_text(msg)
                if self._is_user_content(text):
                    user_texts.append(text)
                    if not first_user_msg:
                        first_user_msg = text[:120]
                    last_user_msg = text[:120]
            elif role == "assistant":
                text = self._extract_msg_text(msg)
                if text and len(text) > 20:
                    assistant_texts.append(text)
                # Collect tool calls
                tc = msg.get("tool_calls") or []
                for call in tc:
                    fn = call.get("function", {}).get("name", "")
                    if fn and fn not in tool_names:
                        tool_names.append(fn)

        # Skip trivial sessions (< 3 user turns)
        if user_turns < 3:
            return

        # Pass 2: extract searchable keywords from user messages
        keywords = self._extract_keywords(user_texts)

        # Pass 3: extract key actions/decisions from assistant messages
        actions: list = []
        import re as _re
        action_patterns = [
            _re.compile(r'(?:已|完成|修复|修好|创建|添加|删除|更新|部署|重启|提交|推送)了?\s*[`\u4e00-\u9fff\w._/-]{2,40}'),
            _re.compile(r'(?:✅|✓|☑)\s*.{5,60}'),
        ]
        for text in assistant_texts[-6:]:  # focus on recent assistant turns
            for pat in action_patterns:
                for m in pat.finditer(text):
                    action = m.group().strip()
                    if action not in actions:
                        actions.append(action)
                        if len(actions) >= 5:
                            break

        # Build natural language entry
        topic_slug = (
            first_user_msg[:80]
            .replace("\n", " ")
            .replace("|", "/")
            .strip()
        )
        last_slug = (
            last_user_msg[:80]
            .replace("\n", " ")
            .replace("|", "/")
            .strip()
        )

        parts = [f"Session on {today} at {now_ts}"]
        if state.platform:
            parts.append(f"via {state.platform}")
        parts.append(f"({user_turns} turns)")
        entry_header = " ".join(parts) + "."

        body_parts = []
        if topic_slug:
            body_parts.append(f"User asked: {topic_slug}")
        if last_slug and last_slug != topic_slug:
            body_parts.append(f"Last topic: {last_slug}")
        if actions:
            body_parts.append(f"Actions taken: {'; '.join(actions[:5])}")
        if keywords:
            body_parts.append(f"Keywords: {', '.join(keywords[:10])}")
        if tool_names:
            abbrev = [t.replace("mempalace_", "mp:") for t in tool_names[:8]]
            body_parts.append(f"Tools used: {', '.join(abbrev)}")
            if len(tool_names) > 8:
                body_parts.append(f"(+{len(tool_names) - 8} more)")

        entry = entry_header + " " + ". ".join(body_parts) + "."

        # Use the same diary mechanism as _tool_diary_write
        room = "diary"
        drawer_id = f"diary_{_slug(agent_name)}_{today}_{now_ts}"

        collection = self._get_collection(create=True)
        if collection is None:
            return

        meta = {
            "wing": state.wing,
            "room": room,
            "type": "diary_entry",
            "agent": agent_name,
            "date": today,
            "filed_at": datetime.now().isoformat(),
        }
        try:
            collection.upsert(
                ids=[drawer_id],
                documents=[entry],
                metadatas=[meta],
            )
            state.filed_count += 1
            logger.info("auto-diary: %s → %s", drawer_id, entry[:120])
        except Exception as exc:  # pragma: no cover
            logger.warning("auto-diary upsert failed: %s", exc)

    def on_pre_compress(self, messages: List[Dict[str, Any]]) -> None:
        """Save the last few turns as drawers before context compression discards them."""
        state = self._get_session_state()
        if state is None or not state.allow_writes:
            return

        # Extract the last few user/assistant pairs
        pairs: List[tuple] = []
        i = len(messages) - 1
        while i >= 0 and len(pairs) < 3:
            msg = messages[i]
            if msg.get("role") == "assistant":
                assistant_text = msg.get("content", "")
                # Look for the preceding user message
                if i > 0 and messages[i - 1].get("role") == "user":
                    user_text = messages[i - 1].get("content", "")
                    if not _is_trivial_turn(user_text, assistant_text):
                        pairs.append((user_text, assistant_text))
                    i -= 2
                    continue
            i -= 1

        for user_text, assistant_text in reversed(pairs):
            room = "compressed"
            cleaned_user = _strip_injected_memory(user_text)
            cleaned_assistant = _strip_injected_memory(assistant_text)
            document = (
                "[role: user]\n"
                f"{cleaned_user}\n\n"
                "[role: assistant]\n"
                f"{cleaned_assistant}"
            ).strip()
            drawer_id = self._content_drawer_id(state.wing, room, document)
            try:
                self._upsert_drawer(drawer_id, document, room, state)
            except Exception as exc:  # pragma: no cover
                logger.warning("on_pre_compress write failed: %s", exc)

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return list(_ALL_TOOL_SCHEMAS)

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        state = self._get_session_state(kwargs.get("session_id", ""))
        if state is None:
            return self._json_result({"error": "provider_not_initialized"})

        handler = self._tool_dispatch.get(tool_name)
        if handler is None:
            return self._json_result({"error": f"unknown_tool:{tool_name}"})

        try:
            return handler(state, args)
        except Exception as exc:
            logger.warning("Tool %s failed: %s", tool_name, exc, exc_info=True)
            return self._json_result({"error": str(exc), "tool": tool_name})

    def shutdown(self) -> None:
        for session_id in list(self._sessions):
            self._flush_session(session_id)
        self._executor.shutdown(wait=True, cancel_futures=False)
        self._chroma_client = None

    # ------------------------------------------------------------------
    # Tool handlers
    # ------------------------------------------------------------------

    def _tool_search(self, state: SessionState, args: Dict) -> str:
        limit = _bounded_int(args.get("limit"), default=5, lo=1, hi=10)
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

    def _tool_kg_query(self, state: SessionState, args: Dict) -> str:
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

    def _tool_remember(self, state: SessionState, args: Dict) -> str:
        if not state.allow_writes:
            return self._json_result({"success": False, "reason": "writes_disabled"})
        content = _safe_content(str(args.get("content", "")).strip())
        if not content:
            return self._json_result({"success": False, "error": "content is required"})
        room = _safe_room_name(args.get("room"), "facts")
        drawer_id = self._content_drawer_id(state.wing, room, content)
        self._upsert_drawer(drawer_id, content, room, state)
        with state.lock:
            state.memories_filed += 1
        return self._json_result(
            {
                "success": True,
                "drawer_id": drawer_id,
                "wing": state.wing,
                "room": room,
            }
        )

    def _tool_kg_add(self, state: SessionState, args: Dict) -> str:
        if not state.allow_writes:
            return self._json_result({"success": False, "reason": "writes_disabled"})
        subject = str(args.get("subject", "")).strip()
        predicate = str(args.get("predicate", "")).strip()
        obj = str(args.get("object", "")).strip()
        if not all([subject, predicate, obj]):
            return self._json_result({"success": False, "error": "subject, predicate, object required"})
        kg = KnowledgeGraph(db_path=str(self._paths.kg_path))
        try:
            kg.add_triple(subject, predicate, obj, valid_from=args.get("valid_from"))
        finally:
            kg.close()
        return self._json_result({"success": True, "subject": subject, "predicate": predicate, "object": obj})

    def _tool_kg_invalidate(self, state: SessionState, args: Dict) -> str:
        if not state.allow_writes:
            return self._json_result({"success": False, "reason": "writes_disabled"})
        subject = str(args.get("subject", "")).strip()
        predicate = str(args.get("predicate", "")).strip()
        obj = str(args.get("object", "")).strip()
        if not all([subject, predicate, obj]):
            return self._json_result({"success": False, "error": "subject, predicate, object required"})
        kg = KnowledgeGraph(db_path=str(self._paths.kg_path))
        try:
            kg.invalidate(subject, predicate, obj, ended=args.get("ended"))
        finally:
            kg.close()
        return self._json_result({"success": True, "subject": subject, "predicate": predicate, "object": obj})

    def _tool_kg_timeline(self, state: SessionState, args: Dict) -> str:
        entity = args.get("entity")
        entity = str(entity).strip() if entity else None
        kg = KnowledgeGraph(db_path=str(self._paths.kg_path))
        try:
            result = kg.timeline(entity_name=entity)
        finally:
            kg.close()
        return self._json_result({"timeline": result})

    def _tool_kg_stats(self, state: SessionState, args: Dict) -> str:
        kg = KnowledgeGraph(db_path=str(self._paths.kg_path))
        try:
            result = kg.stats()
        finally:
            kg.close()
        return self._json_result({"stats": result})

    def _tool_status(self, state: SessionState, args: Dict) -> str:
        collection = self._get_collection(create=False)
        count = collection.count() if collection else 0
        return self._json_result({
            "provider": "mempalace",
            "version": "2.0.0",
            "wing": state.wing,
            "palace_path": str(self._paths.palace_path),
            "kg_path": str(self._paths.kg_path),
            "drawer_count": count,
            "session_id": state.session_id,
        })

    def _tool_list_wings(self, state: SessionState, args: Dict) -> str:
        wings = self._scan_metadata_field("wing")
        return self._json_result({"wings": sorted(wings.keys()), "counts": wings})

    def _tool_list_rooms(self, state: SessionState, args: Dict) -> str:
        wing_filter = args.get("wing")
        rooms = self._scan_metadata_field("room", wing_filter=wing_filter)
        return self._json_result({"rooms": sorted(rooms.keys()), "counts": rooms})

    def _tool_get_taxonomy(self, state: SessionState, args: Dict) -> str:
        collection = self._get_collection(create=False)
        if collection is None:
            return self._json_result({"taxonomy": {}})

        taxonomy: Dict[str, Dict[str, int]] = {}
        total = collection.count()
        offset = 0
        batch_size = 1000
        while offset < total:
            batch = collection.get(
                offset=offset,
                limit=batch_size,
                include=["metadatas"],
            )
            for meta in (batch.get("metadatas") or []):
                wing = (meta or {}).get("wing", "unknown")
                room = (meta or {}).get("room", "unknown")
                if wing not in taxonomy:
                    taxonomy[wing] = {}
                taxonomy[wing][room] = taxonomy[wing].get(room, 0) + 1
            offset += batch_size

        return self._json_result({"taxonomy": taxonomy})

    def _tool_traverse(self, state: SessionState, args: Dict) -> str:
        if traverse is None:
            return self._json_result({"error": "traverse not available"})
        start_room = str(args.get("start_room", ""))
        max_hops = _bounded_int(args.get("max_hops"), default=2, lo=1, hi=10)
        col = self._get_collection(create=False)
        result = traverse(start_room, col=col, config=None, max_hops=max_hops)
        return self._json_result({"traversal": result})

    def _tool_find_tunnels(self, state: SessionState, args: Dict) -> str:
        if find_tunnels is None:
            return self._json_result({"error": "find_tunnels not available"})
        col = self._get_collection(create=False)
        result = find_tunnels(
            wing_a=args.get("wing_a"),
            wing_b=args.get("wing_b"),
            col=col,
            config=None,
        )
        return self._json_result({"tunnels": result})

    def _tool_graph_stats(self, state: SessionState, args: Dict) -> str:
        if graph_stats is None:
            return self._json_result({"error": "graph_stats not available"})
        col = self._get_collection(create=False)
        result = graph_stats(col=col, config=None)
        return self._json_result({"stats": result})

    def _tool_create_tunnel(self, state: SessionState, args: Dict) -> str:
        if not state.allow_writes:
            return self._json_result({"success": False, "reason": "writes_disabled"})
        if create_tunnel is None:
            return self._json_result({"error": "create_tunnel not available"})
        result = create_tunnel(
            source_wing=str(args.get("source_wing", "")),
            source_room=str(args.get("source_room", "")),
            target_wing=str(args.get("target_wing", "")),
            target_room=str(args.get("target_room", "")),
            label=str(args.get("label", "")),
            source_drawer_id=args.get("source_drawer_id"),
            target_drawer_id=args.get("target_drawer_id"),
        )
        return self._json_result({"success": True, "tunnel": result})

    def _tool_list_tunnels(self, state: SessionState, args: Dict) -> str:
        if list_tunnels is None:
            return self._json_result({"error": "list_tunnels not available"})
        result = list_tunnels(wing=args.get("wing"))
        return self._json_result({"tunnels": result})

    def _tool_delete_tunnel(self, state: SessionState, args: Dict) -> str:
        if not state.allow_writes:
            return self._json_result({"success": False, "reason": "writes_disabled"})
        if delete_tunnel is None:
            return self._json_result({"error": "delete_tunnel not available"})
        result = delete_tunnel(tunnel_id=str(args.get("tunnel_id", "")))
        return self._json_result({"success": True, "result": result})

    def _tool_follow_tunnels(self, state: SessionState, args: Dict) -> str:
        if follow_tunnels is None:
            return self._json_result({"error": "follow_tunnels not available"})
        col = self._get_collection(create=False)
        result = follow_tunnels(
            wing=str(args.get("wing", "")),
            room=str(args.get("room", "")),
            col=col,
            config=None,
        )
        return self._json_result({"tunnels": result})

    def _tool_add_drawer(self, state: SessionState, args: Dict) -> str:
        if not state.allow_writes:
            return self._json_result({"success": False, "reason": "writes_disabled"})
        content = _safe_content(str(args.get("content", "")).strip())
        if not content:
            return self._json_result({"success": False, "error": "content is required"})
        room = _safe_room_name(args.get("room"), "general")
        wing_override = args.get("wing")
        drawer_id = args.get("drawer_id")
        if not drawer_id:
            drawer_id = self._content_drawer_id(
                wing_override or state.wing, room, content
            )
        self._upsert_drawer(
            drawer_id, content, room, state, wing_override=wing_override
        )
        with state.lock:
            state.memories_filed += 1
        return self._json_result({
            "success": True,
            "drawer_id": drawer_id,
            "wing": wing_override or state.wing,
            "room": room,
        })

    def _tool_delete_drawer(self, state: SessionState, args: Dict) -> str:
        if not state.allow_writes:
            return self._json_result({"success": False, "reason": "writes_disabled"})
        drawer_id = str(args.get("drawer_id", "")).strip()
        if not drawer_id:
            return self._json_result({"success": False, "error": "drawer_id is required"})
        collection = self._get_collection(create=False)
        if collection is None:
            return self._json_result({"success": False, "error": "collection not found"})
        collection.delete(ids=[drawer_id])
        return self._json_result({"success": True, "deleted": drawer_id})

    def _tool_get_drawer(self, state: SessionState, args: Dict) -> str:
        drawer_id = str(args.get("drawer_id", "")).strip()
        if not drawer_id:
            return self._json_result({"error": "drawer_id is required"})
        collection = self._get_collection(create=False)
        if collection is None:
            return self._json_result({"error": "collection not found"})
        result = collection.get(ids=[drawer_id], include=["documents", "metadatas"])
        if not result["ids"]:
            return self._json_result({"error": "drawer not found", "drawer_id": drawer_id})
        return self._json_result({
            "drawer_id": result["ids"][0],
            "content": result["documents"][0],
            "metadata": result["metadatas"][0],
        })

    def _tool_list_drawers(self, state: SessionState, args: Dict) -> str:
        collection = self._get_collection(create=False)
        if collection is None:
            return self._json_result({"drawers": []})
        limit = _bounded_int(args.get("limit"), default=20, lo=1, hi=100)
        wing_filter = args.get("wing")
        room_filter = args.get("room")

        where: Optional[Dict] = None
        if wing_filter and room_filter:
            where = {"$and": [{"wing": wing_filter}, {"room": room_filter}]}
        elif wing_filter:
            where = {"wing": wing_filter}
        elif room_filter:
            where = {"room": room_filter}

        kwargs: Dict[str, Any] = {"limit": limit, "include": ["documents", "metadatas"]}
        if where:
            kwargs["where"] = where

        result = collection.get(**kwargs)
        drawers = []
        for i, did in enumerate(result.get("ids", [])):
            drawers.append({
                "drawer_id": did,
                "content": (result.get("documents") or [])[i] if i < len(result.get("documents") or []) else "",
                "metadata": (result.get("metadatas") or [])[i] if i < len(result.get("metadatas") or []) else {},
            })
        return self._json_result({"drawers": drawers})

    def _tool_update_drawer(self, state: SessionState, args: Dict) -> str:
        if not state.allow_writes:
            return self._json_result({"success": False, "reason": "writes_disabled"})
        drawer_id = str(args.get("drawer_id", "")).strip()
        content = _safe_content(str(args.get("content", "")).strip())
        if not drawer_id:
            return self._json_result({"success": False, "error": "drawer_id is required"})
        if not content:
            return self._json_result({"success": False, "error": "content is required"})
        collection = self._get_collection(create=False)
        if collection is None:
            return self._json_result({"success": False, "error": "collection not found"})
        # Get existing metadata to preserve it
        existing = collection.get(ids=[drawer_id], include=["metadatas"])
        if not existing["ids"]:
            return self._json_result({"success": False, "error": "drawer not found"})
        metadata = existing["metadatas"][0]
        metadata["filed_at"] = datetime.now().isoformat()
        collection.update(ids=[drawer_id], documents=[content], metadatas=[metadata])
        return self._json_result({"success": True, "drawer_id": drawer_id})

    def _tool_diary_write(self, state: SessionState, args: Dict) -> str:
        if not state.allow_writes:
            return self._json_result({"success": False, "reason": "writes_disabled"})
        entry = _safe_content(str(args.get("entry", "")).strip())
        if not entry:
            return self._json_result({"success": False, "error": "entry is required"})

        agent_name = state.agent_identity or "default"
        room = f"diary_{_slug(agent_name)}"
        today = datetime.now().strftime("%Y-%m-%d")
        drawer_id = f"diary_{_slug(agent_name)}_{today}"

        # Check if same-day entry exists; if so, append
        collection = self._get_collection(create=True)
        existing = collection.get(ids=[drawer_id], include=["documents", "metadatas"])
        if existing["ids"]:
            old_content = existing["documents"][0] or ""
            new_content = f"{old_content}\n\n---\n\n{entry}"
            metadata = existing["metadatas"][0]
            metadata["filed_at"] = datetime.now().isoformat()
            collection.update(ids=[drawer_id], documents=[new_content], metadatas=[metadata])
        else:
            self._upsert_drawer(drawer_id, entry, room, state)

        with state.lock:
            state.memories_filed += 1
        return self._json_result({
            "success": True,
            "drawer_id": drawer_id,
            "room": room,
            "date": today,
        })

    def _tool_diary_read(self, state: SessionState, args: Dict) -> str:
        collection = self._get_collection(create=False)
        if collection is None:
            return self._json_result({"entries": []})

        agent_name = state.agent_identity or "default"
        room = f"diary_{_slug(agent_name)}"
        limit = _bounded_int(args.get("limit"), default=5, lo=1, hi=50)

        result = collection.get(
            where={"room": room},
            include=["documents", "metadatas"],
        )
        entries = []
        for i, did in enumerate(result.get("ids", [])):
            entries.append({
                "drawer_id": did,
                "content": (result.get("documents") or [])[i] if i < len(result.get("documents") or []) else "",
                "metadata": (result.get("metadatas") or [])[i] if i < len(result.get("metadatas") or []) else {},
            })
        # Sort by drawer_id descending (dates sort lexicographically)
        entries.sort(key=lambda e: e["drawer_id"], reverse=True)
        return self._json_result({"entries": entries[:limit]})

    def _tool_check_duplicate(self, state: SessionState, args: Dict) -> str:
        content = str(args.get("content", "")).strip()
        if not content:
            return self._json_result({"error": "content is required"})
        room = args.get("room")
        room_name = _safe_room_name(room, "general") if room else None

        # Check by content hash match
        result = search_memories(
            content,
            palace_path=str(self._paths.palace_path),
            wing=None,
            room=room_name,
            n_results=3,
            max_distance=0.0,
        )
        hits = result.get("results", []) if isinstance(result, dict) else []
        exact_matches = [h for h in hits if h.get("text", "").strip() == content]
        return self._json_result({
            "is_duplicate": len(exact_matches) > 0,
            "matches": len(exact_matches),
            "similar": len(hits),
        })

    def _tool_check_facts(self, state: SessionState, args: Dict) -> str:
        text = str(args.get("text", "")).strip()
        if not text:
            return self._json_result({"error": "text is required"})
        if fact_check_text is None:
            return self._json_result({"error": "fact_checker not available"})
        issues = fact_check_text(text, palace_path=str(self._paths.palace_path), config=None)
        return self._json_result({"issues": issues, "count": len(issues)})

    def _tool_hook_settings(self, state: SessionState, args: Dict) -> str:
        if MempalaceConfig is None:
            return self._json_result({"error": "MempalaceConfig not available"})
        config = MempalaceConfig()
        key = args.get("key")
        value = args.get("value")
        if key is not None and value is not None:
            if not state.allow_writes:
                return self._json_result({"success": False, "reason": "writes_disabled"})
            config.set_hook_setting(key, value)
            return self._json_result({"success": True, "key": key, "value": value})
        # List current settings
        return self._json_result({
            "hook_silent_save": config.hook_silent_save,
            "hook_desktop_toast": config.hook_desktop_toast,
        })

    def _tool_reconnect(self, state: SessionState, args: Dict) -> str:
        with self._chroma_lock:
            self._chroma_client = None
            # Clear ChromaDB's global singleton registry so PersistentClient
            # can be re-created with fresh settings (avoids "different settings" error).
            try:
                from chromadb.api.shared_system_client import SharedSystemClient
                SharedSystemClient.clear_system_cache()
            except Exception:
                pass
        # Re-create on next access
        try:
            col = self._get_collection(create=True)
            count = col.count() if col else 0
        except Exception as exc:
            return self._json_result({"success": False, "error": str(exc)})
        return self._json_result({"success": True, "drawer_count": count})

    def _tool_get_aaak_spec(self, state: SessionState, args: Dict) -> str:
        return self._json_result({
            "spec": "AAAK/1.0",
            "provider": "mempalace",
            "version": "2.0.0",
            "capabilities": ALL_TOOL_NAMES,
        })

    def _tool_memories_filed_away(self, state: SessionState, args: Dict) -> str:
        with state.lock:
            count = state.memories_filed
        return self._json_result({"memories_filed": count, "session_id": state.session_id})

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

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
        # Uses cached config read at initialize() time — no per-call file I/O.
        config_wing = (self._cached_config or {}).get("default_wing")
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

        previous_assistant_tail = self._previous_assistant_tail(state)
        try:
            from mempalace.recall_llm import local_recall_decision

            local_decision = local_recall_decision(
                query,
                previous_assistant_context={"tail": previous_assistant_tail},
                active_context={"wing": state.wing, "platform": state.platform},
            )
            if local_decision and not local_decision.get("should_recall"):
                logger.info(
                    "Recall: local skip reason=%s",
                    local_decision.get("reason", "unknown"),
                )
                return ""
        except Exception:
            pass

        # --- LLM-enhanced recall (opt-in via MEMPAL_RECALL_LLM=1) ---
        llm_config = None
        search_query = query
        if previous_assistant_tail:
            search_query = f"{previous_assistant_tail}\n\n{query}"
        time_after = None
        try:
            from mempalace.recall_llm import is_enabled, _get_llm_config, decide_recall, rerank
            if is_enabled():
                llm_config = _get_llm_config()
            if llm_config:
                recall_decision = decide_recall(
                    query,
                    config=llm_config,
                    previous_assistant_context={"tail": previous_assistant_tail},
                    active_context={"wing": state.wing, "platform": state.platform},
                )
                if recall_decision:
                    if not recall_decision.get("should_recall"):
                        logger.info(
                            "Recall: LLM skipped recall reason=%s",
                            recall_decision.get("reason", "unknown"),
                        )
                        return ""
                    search_query = recall_decision["query"]
                    time_after = recall_decision.get("after")
                    logger.info(
                        "Recall: LLM decided recall reason=%s, query=%r, after=%s",
                        recall_decision.get("reason", "unknown"),
                        search_query[:80],
                        time_after,
                    )
        except Exception as e:
            logger.info("Recall: decide+rewrite failed (%s), using fallback", e)

        pool_size = RECALL_POOL if llm_config else limit
        result = search_memories(
            query=search_query,
            palace_path=str(self._paths.palace_path),
            wing=None,  # search all wings for broader recall
            preferred_wing=state.wing,  # soft-boost results from the active wing
            n_results=pool_size,
            after=time_after,
        )
        hits = result.get("results", []) if isinstance(result, dict) else []

        # Filter out diary entries — session summaries pollute auto-recall
        hits = [h for h in hits if h.get("room") != "diary"]

        if not hits:
            return ""

        # LLM rerank + relevance filter
        if llm_config and len(hits) > limit:
            try:
                reranked = rerank(
                    query,
                    hits,
                    top_k=limit,
                    config=llm_config,
                    previous_assistant_context={"tail": previous_assistant_tail},
                )
                if reranked is not None:
                    if len(reranked) == 0:
                        logger.info("Recall: LLM filtered all %d hits as irrelevant", len(hits))
                        return ""
                    logger.info("Recall: LLM reranked %d → %d", len(hits), len(reranked))
                    hits = reranked
            except Exception as e:
                logger.info("Recall: LLM rerank failed (%s), using BM25 order", e)

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
        wing_override: Optional[str] = None,
    ) -> None:
        collection = self._get_collection(create=True)
        wing = wing_override or state.wing
        metadata = {
            "wing": wing,
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
            if create:
                raise RuntimeError("MemPalace provider is unavailable")
            return None
        with self._chroma_lock:
            if self._chroma_client is None:
                import os
                os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
                try:
                    self._chroma_client = chromadb.PersistentClient(
                        path=str(self._paths.palace_path),
                    )
                except ValueError:
                    # ChromaDB singleton conflict — clear global cache and retry.
                    try:
                        from chromadb.api.shared_system_client import SharedSystemClient
                        SharedSystemClient.clear_system_cache()
                    except Exception:
                        pass
                    self._chroma_client = chromadb.PersistentClient(
                        path=str(self._paths.palace_path),
                    )
        if create:
            return self._chroma_client.get_or_create_collection(COLLECTION_NAME)
        try:
            return self._chroma_client.get_collection(COLLECTION_NAME)
        except Exception:
            return None

    def _assistant_cache_dir(self) -> Path:
        if self._paths is None:
            raise RuntimeError("Provider not initialized")
        return self._paths.base_dir / "session_state"

    def _assistant_cache_path(self, session_id: str) -> Path:
        return self._assistant_cache_dir() / _session_cache_filename(session_id)

    def _write_assistant_cache(self, session_id: str, assistant_reply: str) -> None:
        if self._paths is None or not session_id:
            return
        path = self._assistant_cache_path(session_id)
        if not assistant_reply:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(assistant_reply, encoding="utf-8")
        except OSError:
            logger.debug("assistant cache write failed for %s", session_id, exc_info=True)

    def _read_assistant_cache(self, session_id: str) -> str:
        if self._paths is None or not session_id:
            return ""
        path = self._assistant_cache_path(session_id)
        if not path.is_file():
            return ""
        try:
            return path.read_text(encoding="utf-8")
        except OSError:
            logger.debug("assistant cache read failed for %s", session_id, exc_info=True)
            return ""

    def _clear_assistant_cache(self, session_id: str) -> None:
        if self._paths is None or not session_id:
            return
        path = self._assistant_cache_path(session_id)
        try:
            path.unlink(missing_ok=True)
        except OSError:
            logger.debug("assistant cache clear failed for %s", session_id, exc_info=True)

    def _previous_assistant_tail(self, state: SessionState) -> str:
        with state.lock:
            assistant_reply = state.last_assistant_reply
        if not assistant_reply:
            assistant_reply = self._read_assistant_cache(state.session_id)
            if assistant_reply:
                with state.lock:
                    state.last_assistant_reply = assistant_reply
        return _tail_chars(assistant_reply, PREVIOUS_ASSISTANT_TAIL_CHARS)

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
        return json.dumps(payload, ensure_ascii=True, sort_keys=True, default=str)

    def _scan_metadata_field(self, field_name: str, *, wing_filter: Optional[str] = None) -> Dict[str, int]:
        """Scan ChromaDB metadata in batches, counting distinct values of a field."""
        collection = self._get_collection(create=False)
        if collection is None:
            return {}

        counts: Dict[str, int] = {}
        total = collection.count()
        offset = 0
        batch_size = 1000
        while offset < total:
            batch = collection.get(
                offset=offset,
                limit=batch_size,
                include=["metadatas"],
            )
            for meta in (batch.get("metadatas") or []):
                if meta is None:
                    continue
                if wing_filter and meta.get("wing") != wing_filter:
                    continue
                value = meta.get(field_name, "unknown")
                counts[value] = counts.get(value, 0) + 1
            offset += batch_size

        return counts


def register(ctx) -> None:
    """Hermes plugin discovery entrypoint."""

    ctx.register_memory_provider(MemPalaceMemoryProvider())


__all__ = ["MemPalaceMemoryProvider", "ResolvedPaths", "register", "resolve_paths"]
