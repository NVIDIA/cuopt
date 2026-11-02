#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Best-effort lint: flag field_registry.yaml `default:` strings that look
stale against the C++ member initializer they describe.

Not part of CI -- run manually:
    python cpp/src/grpc/codegen/lint_registry_defaults.py

`default:` is documented (field_registry.yaml's header comment,
FIELD_REGISTRY_REFERENCE.md sec 2.1) as free text the generator doesn't
derive from the C++ struct; it only checks, for a non-optional settings
field, that this string textually matches the proto3 zero value -- it can't
catch a C++ default that changed with no matching registry update. This
tool cannot parse arbitrary C++ either -- it only compares members with a
simple `= VALUE;` or `{VALUE};` initializer on one line, and it prints
warnings rather than failing, since a false positive here should never
block a PR. Treat its output as a prompt to double-check, not a verdict.
"""

import re
import sys
from decimal import Decimal
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent))
import generate_conversions as gc  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[4]

# (registry section, header file, nested-struct name or None for top-level,
#  member prefix)
SOURCES = [
    (
        "pdlp_settings",
        "cpp/include/cuopt/mathematical_optimization/pdlp/solver_settings.hpp",
        "tolerances_t",
        "tolerances.",
    ),
    (
        "pdlp_settings",
        "cpp/include/cuopt/mathematical_optimization/pdlp/solver_settings.hpp",
        None,
        "",
    ),
    (
        "mip_settings",
        "cpp/include/cuopt/mathematical_optimization/mip/solver_settings.hpp",
        "tolerances_t",
        "tolerances.",
    ),
    (
        "mip_settings",
        "cpp/include/cuopt/mathematical_optimization/mip/solver_settings.hpp",
        None,
        "",
    ),
    (
        "mip_settings",
        "cpp/include/cuopt/mathematical_optimization/mip/heuristics_hyper_params.hpp",
        "mip_heuristics_hyper_params_t",
        "heuristic_params.",
    ),
]

_INIT_RE = re.compile(
    r"^\s*(?:f_t|i_t|bool)\s+(\w+)\s*(?:=\s*(?P<eq>[^;]+)|\{(?P<br>[^}]+)\})\s*;",
    re.MULTILINE,
)
_NUMBER_RE = re.compile(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?")


def _extract_initializers(text):
    """Return {member_name: raw_initializer_text} for single-line members."""
    out = {}
    for m in _INIT_RE.finditer(text):
        name = m.group(1)
        value = (m.group("eq") or m.group("br")).split("//")[0].strip()
        out[name] = value
    return out


def _cpp_defaults(header_path, struct_name):
    text = (REPO_ROOT / header_path).read_text()
    if struct_name is None:
        return _extract_initializers(text)
    block = re.search(
        rf"struct {re.escape(struct_name)}\b[^{{]*\{{(.*?)\n\s*\}};",
        text,
        re.DOTALL,
    )
    if not block:
        print(f"  (could not locate struct {struct_name} in {header_path})")
        return {}
    return _extract_initializers(block.group(1))


def _sentinel_kind(cpp_value):
    if "numeric_limits" not in cpp_value:
        return None
    return "infinity" if "infinity" in cpp_value else "max"


def _mismatch(cpp_value, registry_default):
    if registry_default is None:
        return None  # nothing to compare against
    reg_lower = str(registry_default).lower()

    kind = _sentinel_kind(cpp_value)
    if kind == "infinity":
        ok = any(w in reg_lower for w in ("infinity", "no limit", "inf"))
        return (
            None
            if ok
            else "C++ is infinity-sentineled, default: doesn't say so"
        )
    if kind == "max":
        ok = any(w in reg_lower for w in ("no limit", "max"))
        return None if ok else "C++ is max-sentineled, default: doesn't say so"

    if cpp_value in ("true", "false"):
        ok = cpp_value in reg_lower
        return (
            None
            if ok
            else f"C++ default is {cpp_value}, default: doesn't match"
        )

    cpp_num = _NUMBER_RE.search(cpp_value)
    if cpp_num is None:
        return None  # unrecognized expression (e.g. a function call) -- skip
    reg_num = _NUMBER_RE.search(reg_lower)
    if reg_num is None:
        return None  # default: is pure prose (e.g. context-dependent) -- skip
    if Decimal(cpp_num.group()) != Decimal(reg_num.group()):
        return f"C++ default is {cpp_num.group()}, default: says {reg_num.group()}"
    return None


def main():
    registry = yaml.safe_load(
        (Path(__file__).parent / "field_registry.yaml").read_text()
    )
    warnings = []
    checked = 0
    for section, header_path, struct_name, prefix in SOURCES:
        cpp_defaults = _cpp_defaults(header_path, struct_name)
        fields = registry.get(section, {}).get("fields", [])
        for f in gc.parse_settings_fields(fields):
            member = f["member"]
            if not member.startswith(prefix):
                continue
            local_name = member[len(prefix) :]
            if "." in local_name or local_name not in cpp_defaults:
                continue
            checked += 1
            reason = _mismatch(cpp_defaults[local_name], f.get("default"))
            if reason:
                warnings.append(
                    f"{section}.{member}: {reason} "
                    f"(C++: {cpp_defaults[local_name]!r}, "
                    f"registry default: {f.get('default')!r})"
                )

    print(
        f"Checked {checked} fields with a simple single-line C++ initializer."
    )
    if warnings:
        print(f"\n{len(warnings)} possible drift(s) -- review by hand:\n")
        for w in warnings:
            print(f"  WARN: {w}")
    else:
        print("No drift found among the fields this tool could parse.")


if __name__ == "__main__":
    main()
