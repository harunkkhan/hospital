"""``python -m hospital.api.codegen dump-schema`` — the TS-contract schema dump.

Emits the app's OpenAPI document (which already carries the merged wire
manifest — see ``main.HospitalAPI.openapi``) as canonical JSON: sorted keys,
2-space indent, trailing newline. The committed artifact lives at
``apps/api/schema/openapi.json``; the drift gate is ``dump -> diff`` — any
pydantic contract change not regenerated fails the byte-stability test.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from hospital.api.main import create_app

# Repo-relative home of the committed artifact (resolved from this file).
SCHEMA_PATH = Path(__file__).resolve().parents[3] / "schema" / "openapi.json"


def build_schema_document() -> dict[str, Any]:
    """The merged OpenAPI + wire-manifest document (one schema, two exits)."""
    return create_app(scenario_dir=None).openapi()


def render_schema() -> str:
    """Canonical, diff-stable JSON for the committed artifact."""
    return json.dumps(build_schema_document(), indent=2, sort_keys=True) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m hospital.api.codegen",
        description="Dump the merged OpenAPI + wire-manifest JSON schema",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    dump = sub.add_parser("dump-schema", help="write the schema to stdout (or --out)")
    dump.add_argument("--out", default=None, help="output path (default: stdout)")
    args = parser.parse_args(argv)
    text = render_schema()
    if args.out is not None:
        Path(args.out).write_text(text)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
