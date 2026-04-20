"""Test isolation — ensure tests use default (local) embedding and no LLM recall."""

import os

for _var in (
    "MEMPAL_EMBEDDING_MODEL", "MEMPALACE_EMBEDDING_MODEL",
    "MEMPAL_RECALL_LLM",
):
    os.environ.pop(_var, None)

try:
    from mempalace.embedding import reset_cache
    from mempalace.palace import _reset_embedding_cache

    reset_cache()
    _reset_embedding_cache()
except ImportError:
    pass
