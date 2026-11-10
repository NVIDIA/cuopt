# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small tests for the C++ test filter generator."""

import pytest
import yaml

from ci.utils.generate_test_paths import UniqueKeySafeLoader, generate

WORKFLOW = """# Keep this comment.
name: pr
jobs:
  changed-files:
    with:
      files_yaml: |
        test_cpp:
          - '**'
        test_cpp_routing:
          - '**'
        test_cpp_mathopt:
          - '**'
        test_python:
          - 'python/**'
  other-job:
    runs-on: ubuntu-latest
"""


@pytest.fixture
def rules():
    """Provide a small set of shared and solver-specific paths."""
    return {
        "common_exclusions": ["docs/**"],
        "mathopt_only": ["cpp/src/pdlp/**"],
        "routing_only": ["cpp/src/routing/**"],
    }


def test_generated_groups(rules):
    """Check which exclusions each generated group receives."""

    workflow = yaml.safe_load(generate(WORKFLOW, rules))
    groups = yaml.safe_load(
        workflow["jobs"]["changed-files"]["with"]["files_yaml"]
    )

    assert groups["test_cpp"] == ["**", "!docs/**"]
    assert groups["test_cpp_routing"] == ["**", "!docs/**", "!cpp/src/pdlp/**"]
    assert groups["test_cpp_mathopt"] == [
        "**",
        "!docs/**",
        "!cpp/src/routing/**",
    ]


def test_preserves_other_text(rules):
    """Preserve unrelated workflow text byte for byte."""
    result = generate(WORKFLOW, rules)

    for path in ("docs/**", "cpp/src/pdlp/**", "cpp/src/routing/**"):
        result = result.replace(f"          - '!{path}'\n", "")
    assert result == WORKFLOW


def test_second_run_changes_nothing(rules):
    """Keep generation idempotent."""
    result = generate(WORKFLOW, rules)
    assert generate(result, rules) == result


@pytest.mark.parametrize(
    "header",
    ["        renamed_group:\n", "        test_cpp: []\n        test_cpp:\n"],
)
def test_missing_or_duplicate_group(rules, header):
    """Reject missing or ambiguous workflow groups."""
    workflow = WORKFLOW.replace("        test_cpp:\n", header)

    with pytest.raises(ValueError, match="Expected exactly one test_cpp key"):
        generate(workflow, rules)


@pytest.mark.parametrize(
    "paths, message",
    [
        ("docs/**", "expected a list"),
        ([42], "expected a non-empty string"),
        ([""], "expected a non-empty string"),
        (["!docs/**"], "omit the leading"),
        (["docs/\npage"], "fit on one line"),
        (["docs/**", "docs/**"], "duplicate paths"),
    ],
)
def test_invalid_rules(rules, paths, message):
    """Reject malformed paths and duplicate exclusions."""

    rules["common_exclusions"] = paths

    with pytest.raises(ValueError, match=message):
        generate(WORKFLOW, rules)


@pytest.mark.parametrize(
    "section", ["common_exclusions", "mathopt_only", "routing_only"]
)
def test_duplicate_yaml_keys(rules, section):
    """Reject a repeated section instead of silently replacing its paths."""

    text = yaml.safe_dump(rules) + f"{section}: []\n"

    with pytest.raises(
        yaml.constructor.ConstructorError, match="duplicate key"
    ):
        yaml.load(text, Loader=UniqueKeySafeLoader)


def test_valid_yaml_rules(rules):
    """Load unique sections without changing their contents."""
    assert (
        yaml.load(yaml.safe_dump(rules), Loader=UniqueKeySafeLoader) == rules
    )


def test_unsafe_yaml_tag():
    """Keep rejecting Python object tags with the custom safe loader."""

    with pytest.raises(yaml.constructor.ConstructorError):
        yaml.load("!!python/tuple [1, 2]", Loader=UniqueKeySafeLoader)
