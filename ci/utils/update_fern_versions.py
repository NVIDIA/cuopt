#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Add a new version entry to fern/docs.yml when cutting a release.

Called by ci/release/update-version.sh. Inserts the new version at the top of the
versions list and demotes the previous entry (removes "(Latest)" suffix). If the
version already exists, exits cleanly without modifying the file.

The script uses line-based editing to preserve the existing YAML formatting and
the rapids-pre-commit-hooks suppression comment on the latest entry.
"""

import re
from pathlib import Path


def main():
    repo_root = Path(__file__).parent.parent.parent
    version = (repo_root / "VERSION").read_text().strip()
    if not version:
        raise ValueError("VERSION file is empty")

    # rapids-pre-commit-hooks: disable-next-line
    # Derive short tag (e.g. "26.08.00" → "26.08") and docs-yml path suffix
    parts = version.split(".")
    major, minor = int(parts[0]), int(parts[1]) if len(parts) > 1 else 0
    short = f"{major:02d}.{minor:02d}"
    display = f"{short} (Latest)"
    path_val = f"docs-v{short}.yml"

    docs_yml = repo_root / "fern/docs.yml"
    text = docs_yml.read_text()

    # Bail out if this version already exists
    if f'display-name: "{short}' in text:
        print(f"Version {short} already present in fern/docs.yml")
        return

    # Demote existing "(Latest)" entry: strip the suffix and remove the suppress comment
    text = re.sub(
        r"  # rapids-pre-commit-hooks: disable-next-line\n"
        r'  - display-name: "([^"]+) \(Latest\)"',
        r'  - display-name: "\1"',
        text,
    )

    # Insert new entry after the `versions:` key
    new_entry = (
        f"  # rapids-pre-commit-hooks: disable-next-line\n"
        f'  - display-name: "{display}"\n'
        f"    path: {path_val}\n"
    )
    text = re.sub(r"(versions:\n)", r"\1" + new_entry, text, count=1)

    docs_yml.write_text(text)
    print(f"Added {display} to fern/docs.yml")


if __name__ == "__main__":
    main()
