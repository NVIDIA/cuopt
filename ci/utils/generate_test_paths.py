# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate the C++ path filters in pr.yaml from ci/test-paths.yaml."""

import argparse
import re
from pathlib import Path

import yaml


class UniqueKeySafeLoader(yaml.SafeLoader):
    """Load safe YAML while rejecting duplicate mapping keys."""

    def construct_mapping(self, node, deep=False):
        """Reject repeated keys before constructing a mapping."""
        keys = set()

        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)

            if key in keys:
                raise yaml.constructor.ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    f"duplicate key: {key!r}",
                    key_node.start_mark,
                )

            keys.add(key)

        return super().construct_mapping(node, deep=deep)


def mapping_value(node, name):
    """Find exactly one key in a YAML mapping, rejecting ambiguous input."""

    if not isinstance(node, yaml.MappingNode):
        raise ValueError(f"Expected a mapping containing {name}")

    matches = [value for key, value in node.value if key.value == name]

    if len(matches) != 1:
        raise ValueError(f"Expected exactly one {name} key")

    return matches[0]


def replace_filters(workflow, groups):
    """Locate files_yaml structurally and replace its named list blocks."""

    node = yaml.compose(workflow)

    for name in ("jobs", "changed-files", "with", "files_yaml"):
        node = mapping_value(node, name)

    if not isinstance(node, yaml.ScalarNode) or node.style != "|":
        raise ValueError("Expected files_yaml to be a literal YAML block")

    lines = workflow.splitlines(keepends=True)
    start = node.start_mark.line + 1
    end = node.end_mark.line
    block = "".join(lines[start:end])
    parsed = yaml.compose(node.value)

    for name, paths in groups.items():
        if not isinstance(mapping_value(parsed, name), yaml.SequenceNode):
            raise ValueError(f"Expected {name} to contain a list")

        pattern = re.compile(
            rf"^(?P<indent> +){name}:[ \t]*\n"
            rf"(?:(?P=indent) +- [^\n]*\n)+",
            re.MULTILINE,
        )

        matches = list(pattern.finditer(block))

        if len(matches) != 1:
            raise ValueError(f"Expected one plain list block for {name}")

        match = matches[0]
        indent = match["indent"]
        replacement = [f"{indent}{name}:\n", f"{indent}  - '**'\n"]

        for path in paths:
            escaped = path.replace("'", "''")
            replacement.append(f"{indent}  - '!{escaped}'\n")

        block = (
            block[: match.start()]
            + "".join(replacement)
            + block[match.end() :]
        )

    generated = yaml.safe_load(block)

    for name, paths in groups.items():
        if generated[name] != ["**", *("!" + path for path in paths)]:
            raise ValueError(f"Unsupported list layout for {name}")

    return "".join(lines[:start]) + block + "".join(lines[end:])


def generate(workflow, rules):
    """Replace the three C++ filters, preserving other workflow text."""

    expected = {"common_exclusions", "mathopt_only", "routing_only"}

    if not isinstance(rules, dict) or set(rules) != expected:
        raise ValueError(
            f"Expected exactly these rule sections: {sorted(expected)}"
        )

    for name, paths in rules.items():
        if not isinstance(paths, list):
            raise ValueError(f"{name}: expected a list of paths")

        for path in paths:
            if not isinstance(path, str) or not path:
                raise ValueError(f"{name}: expected a non-empty string")

            if path.startswith("!"):
                raise ValueError(f"{name}: omit the leading '!': {path!r}")

            if "\n" in path or "\r" in path:
                raise ValueError(
                    f"{name}: path must fit on one line: {path!r}"
                )

        if len(paths) != len(set(paths)):
            raise ValueError(f"{name}: duplicate paths")

    groups = {
        "test_cpp": rules["common_exclusions"],
        "test_cpp_routing": rules["common_exclusions"] + rules["mathopt_only"],
        "test_cpp_mathopt": rules["common_exclusions"] + rules["routing_only"],
    }

    return replace_filters(workflow, groups)


def main():
    """Load path rules and update or check the workflow."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="Check without writing"
    )

    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]

    path = root / ".github/workflows/pr.yaml"
    rules = yaml.load(
        (root / "ci/test-paths.yaml").read_text(), Loader=UniqueKeySafeLoader
    )

    original = path.read_text()
    updated = generate(original, rules)

    if original == updated:
        return 0

    if args.check:
        print(
            "C++ filters are stale; run pre-commit run generate-test-paths --all-files"
        )
        return 1

    path.write_text(updated)
    print("Updated C++ test filters in .github/workflows/pr.yaml")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
