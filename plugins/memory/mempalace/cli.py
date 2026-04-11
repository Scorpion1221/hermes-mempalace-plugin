"""CLI helpers for the Hermes MemPalace provider."""

from __future__ import annotations

import json
from pathlib import Path

from . import resolve_paths


def _status_payload(hermes_home: str) -> dict:
    paths = resolve_paths(hermes_home)
    return {
        "provider": "mempalace",
        "hermes_home": str(paths.hermes_home),
        "config_path": str(paths.config_path),
        "base_dir": str(paths.base_dir),
        "palace_path": str(paths.palace_path),
        "identity_path": str(paths.identity_path),
        "kg_path": str(paths.kg_path),
    }


def cmd_status(args) -> None:
    payload = _status_payload(args.hermes_home)
    print(json.dumps(payload, indent=2, sort_keys=True))


def register_cli(subparser) -> None:
    """Register ``hermes mempalace ...`` CLI commands."""

    subs = subparser.add_subparsers(dest="mempalace_command")

    status = subs.add_parser("status", help="Show resolved MemPalace paths")
    status.add_argument(
        "--hermes-home",
        default=str(Path("~/.hermes").expanduser()),
        help="Hermes profile home to inspect",
    )
    status.set_defaults(func=cmd_status)
