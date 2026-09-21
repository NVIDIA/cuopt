# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Every advertised setting must be one cuOpt actually accepts.

The registry field name is the *proto* field name, which frequently differs
from the ``CUOPT_*`` string parameter (``relative_mip_gap`` vs
``mip_relative_gap``, ``mir_cuts`` vs ``mip_mixed_integer_rounding_cuts``).
Without this check a renamed or newly added field reaches an agent as a
setting that fails at solve time with "Invalid parameter".

Parsed from the in-repo constants.h rather than the installed cuOpt, so the
check stays consistent with the registry it is validating even when the
environment has a different cuOpt version installed.
"""

import re
from pathlib import Path

import pytest

from cuopt_mcp import schema

CONSTANTS_H = (
    Path(__file__).resolve().parents[3]
    / "cpp"
    / "include"
    / "cuopt"
    / "mathematical_optimization"
    / "constants.h"
)

pytestmark = pytest.mark.skipif(
    not CONSTANTS_H.is_file(), reason="not a source checkout"
)


def cuopt_parameter_names() -> set:
    text = CONSTANTS_H.read_text()
    # #define CUOPT_X \<newline>  "x" — join continuations before matching.
    text = re.sub(r"\\\s*\n\s*", " ", text)
    return {
        m.group(2)
        for m in re.finditer(r'#define\s+(CUOPT_\w+)\s+"(\w+)"', text)
    }


@pytest.mark.parametrize("kind", ["pdlp_settings", "mip_settings"])
def test_every_advertised_setting_is_a_real_cuopt_parameter(kind):
    valid = cuopt_parameter_names()
    assert valid, "failed to parse any parameter names from constants.h"
    bad = {}
    for name, prop in schema.settings_schema(kind)["properties"].items():
        resolved = prop.get("x-parameter-name", name)
        if resolved not in valid:
            bad[name] = resolved
    assert not bad, (
        f"{kind} advertises settings cuOpt will reject: {bad}. Add or fix "
        "`param_name:` in field_registry.yaml, then ./build.sh codegen."
    )


def test_known_divergent_names_are_mapped():
    """Spot-check the renames that motivated param_name."""
    mip = schema.settings_schema("mip_settings")["properties"]
    assert mip["relative_mip_gap"]["x-parameter-name"] == "mip_relative_gap"
    assert mip["mir_cuts"]["x-parameter-name"] == (
        "mip_mixed_integer_rounding_cuts"
    )
    assert mip["seed"]["x-parameter-name"] == "random_seed"
    pdlp = schema.settings_schema("pdlp_settings")["properties"]
    assert pdlp["detect_infeasibility"]["x-parameter-name"] == (
        "infeasibility_detection"
    )


def test_unsettable_field_is_not_advertised():
    """presolve_absolute_tolerance has no CUOPT_* constant, so it is omitted."""
    assert (
        "presolve_absolute_tolerance"
        not in schema.settings_schema("mip_settings")["properties"]
    )
