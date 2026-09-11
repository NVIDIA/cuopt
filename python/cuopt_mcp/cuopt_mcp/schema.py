# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Access to the generated solver-settings JSON Schema.

The schema is emitted from ``cpp/src/grpc/codegen/field_registry.yaml`` by
``./build.sh codegen`` — the same single source of truth that drives the
proto and the C++ conversion code. Nothing here hand-maintains a second
copy of the settings surface.
"""

import functools
import json
from pathlib import Path

SCHEMA_FILENAME = "cuopt_mcp_schema.json"

# In an installed wheel the schema is packaged alongside this module. In a
# source checkout it lives in the codegen output directory; fall back to that
# so the server runs from the repo without a build step.
_PACKAGED = Path(__file__).parent / "_generated" / SCHEMA_FILENAME
_IN_TREE = (
    Path(__file__).resolve().parents[3]
    / "cpp"
    / "src"
    / "grpc"
    / "codegen"
    / "generated"
    / SCHEMA_FILENAME
)


def schema_path() -> Path:
    for candidate in (_PACKAGED, _IN_TREE):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"{SCHEMA_FILENAME} not found. In a source checkout, run "
        "`./build.sh codegen` to generate it."
    )


@functools.lru_cache(maxsize=1)
def load() -> dict:
    """Return the full generated schema document."""
    return json.loads(schema_path().read_text())


def settings_schema(kind: str) -> dict:
    """Return the JSON Schema for ``pdlp_settings`` or ``mip_settings``."""
    return load()["settings"][kind]


def known_parameters(kind: str) -> set:
    return set(settings_schema(kind)["properties"])


def validate_settings(kind: str, settings: dict) -> None:
    """Reject unknown or wrongly-typed settings before any gRPC traffic.

    The schema declares ``additionalProperties: false``, so a typo like
    ``time_limt`` fails here with the near-miss named rather than being
    silently dropped by the solver.
    """
    if not settings:
        return
    schema = settings_schema(kind)
    unknown = set(settings) - set(schema["properties"])
    if unknown:
        import difflib

        hints = []
        for name in sorted(unknown):
            close = difflib.get_close_matches(
                name, schema["properties"], n=1, cutoff=0.7
            )
            hints.append(
                f"{name}" + (f" (did you mean {close[0]}?)" if close else "")
            )
        raise ValueError(
            f"unknown {kind} parameter(s): {', '.join(hints)}. "
            f"Call cuopt_list_settings('{kind}') for the full list."
        )
    for name, value in settings.items():
        prop = schema["properties"][name]
        expected = prop.get("type")
        if expected == "string" and "enum" in prop:
            if value not in prop["enum"]:
                raise ValueError(
                    f"{name} must be one of {prop['enum']}, got {value!r}"
                )
        elif expected == "integer" and not isinstance(value, int):
            raise ValueError(f"{name} must be an integer, got {value!r}")
        elif expected == "number" and not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be a number, got {value!r}")
        elif expected == "boolean" and not isinstance(value, bool):
            raise ValueError(f"{name} must be a boolean, got {value!r}")
