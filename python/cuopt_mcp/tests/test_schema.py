# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the generated settings schema and its validation."""

import pytest

from cuopt_mcp import schema


def test_both_settings_sections_present():
    doc = schema.load()
    assert set(doc["settings"]) == {"pdlp_settings", "mip_settings"}


@pytest.mark.parametrize("kind", ["pdlp_settings", "mip_settings"])
def test_every_parameter_is_documented(kind):
    """Every generated property carries a description.

    Guards the registry: a field added without `description:`/`default:`
    would reach an agent as a bare name and type, which is unusable for
    deciding whether to set it.
    """
    props = schema.settings_schema(kind)["properties"]
    assert props
    undocumented = [n for n, p in props.items() if not p.get("description")]
    assert undocumented == []


@pytest.mark.parametrize("kind", ["pdlp_settings", "mip_settings"])
def test_every_parameter_states_a_default(kind):
    props = schema.settings_schema(kind)["properties"]
    missing = [
        n
        for n, p in props.items()
        if "Default:" not in p.get("description", "")
    ]
    assert missing == []


def test_enum_parameters_expose_their_values():
    mode = schema.settings_schema("pdlp_settings")["properties"][
        "pdlp_solver_mode"
    ]
    assert mode["type"] == "string"
    assert "Stable3" in mode["enum"]


def test_sentinel_field_tells_client_to_omit_not_send_minus_one():
    """iteration_limit's -1 encoding must not leak into agent-facing text.

    The wire encodes "no limit" as -1, but a client should express that by
    omitting the field; advertising -1 invites sending it as a literal
    iteration count.
    """
    prop = schema.settings_schema("pdlp_settings")["properties"][
        "iteration_limit"
    ]
    assert "Omit" in prop["description"]
    assert "-1" not in prop["description"]


def test_settings_schema_is_closed():
    for kind in ("pdlp_settings", "mip_settings"):
        assert schema.settings_schema(kind)["additionalProperties"] is False


def test_validate_accepts_known_settings():
    schema.validate_settings("pdlp_settings", {"time_limit": 5.0})
    schema.validate_settings("pdlp_settings", {"method": "Barrier"})


def test_validate_rejects_unknown_setting_with_suggestion():
    with pytest.raises(ValueError, match="did you mean time_limit"):
        schema.validate_settings("pdlp_settings", {"time_limt": 5.0})


def test_validate_rejects_bad_enum_value():
    with pytest.raises(ValueError, match="must be one of"):
        schema.validate_settings("pdlp_settings", {"method": "Simplex"})


def test_validate_rejects_wrong_type():
    with pytest.raises(ValueError, match="must be a number"):
        schema.validate_settings("pdlp_settings", {"time_limit": "fast"})


def test_enum_parameters_carry_a_name_to_integer_mapping():
    """Enum settings must ship the integer cuOpt's parameter interface wants.

    Callers show a model the readable name ("Barrier") but set_parameter
    rejects it with "value Barrier is not an integer", so the schema has to
    carry the mapping rather than leave each client to hand-write it.
    """
    method = schema.settings_schema("pdlp_settings")["properties"]["method"]
    mapping = method["x-enum-values"]
    assert set(mapping) == set(method["enum"])
    assert all(isinstance(v, int) for v in mapping.values())
    # Must agree with constants.h (CUOPT_METHOD_CONCURRENT 0, PDLP 1, ...)
    assert mapping["Concurrent"] == 0
    assert mapping["PDLP"] == 1
    assert mapping["DualSimplex"] == 2
    assert mapping["Barrier"] == 3


def test_non_enum_parameters_have_no_mapping():
    prop = schema.settings_schema("pdlp_settings")["properties"]["time_limit"]
    assert "x-enum-values" not in prop
