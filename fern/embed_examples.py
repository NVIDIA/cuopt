#!/usr/bin/env python3
"""
Embed example source files into *-examples.mdx pages as fenced code blocks.

How it works
------------
MDX example pages contain either:

  (a) a bare link stub on its own line:
        [filename.py](examples/filename.py)

  (b) a previously-generated embed block:
        <!-- embed: examples/filename.py -->
        ```python
        ...code...
        ```
        <!-- /embed -->

On each run this script:
  - Converts case (a) to case (b).
  - Refreshes the code inside any existing case (b) marker pair.

The example source files live alongside the MDX pages in each
  fern/docs/pages/<domain>/<section>/examples/
directory.

Usage:
    python fern/embed_examples.py
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
PAGES = REPO_ROOT / "fern/docs/pages"

# MDX files to process, each with the examples directory next to it
EXAMPLE_PAGES = [
    PAGES / "cuopt-python/convex/convex-examples.mdx",
    PAGES / "cuopt-python/mip/mip-examples.mdx",
    PAGES / "cuopt-python/routing/routing-examples.mdx",
    PAGES / "cuopt-c/convex/convex-examples.mdx",
    PAGES / "cuopt-c/mip/mip-examples.mdx",
]

_EXT_LANG = {".py": "python", ".c": "c", ".sh": "bash", ".mps": "text", ".lp": "text"}

# Matches a standalone link to an example file:
#   [any\_text\_with\_escapes](examples/filename.ext)
_LINK_RE = re.compile(r'^\[([^\]]+)\]\(examples/([^)]+)\)\s*$')

# Matches the opening embed marker:  <!-- embed: examples/filename.ext -->
_OPEN_RE = re.compile(r'^<!--\s*embed:\s*examples/(\S+)\s*-->$')
_CLOSE = "<!-- /embed -->"


def _code_block(src_file: Path, rel: str) -> str:
    lang = _EXT_LANG.get(Path(rel).suffix, "text")
    code = src_file.read_text(encoding="utf-8").rstrip()
    return f"<!-- embed: examples/{rel} -->\n```{lang}\n{code}\n```\n{_CLOSE}"


def _process(mdx_path: Path) -> bool:
    """Embed or refresh code blocks in one MDX file. Returns True if changed."""
    examples_dir = mdx_path.parent / "examples"
    original = mdx_path.read_text(encoding="utf-8")
    lines = original.splitlines(keepends=True)
    out = []
    changed = False
    i = 0
    while i < len(lines):
        line = lines[i].rstrip("\n")

        # Case (a): bare link stub → embed
        m = _LINK_RE.match(line)
        if m:
            rel = m.group(2)  # e.g. "simple_lp_example.py"
            src = examples_dir / rel
            if src.exists():
                block = _code_block(src, rel)
                out.append(block + "\n")
                changed = True
                i += 1
                continue
            # No source file found — leave as-is
            out.append(lines[i])
            i += 1
            continue

        # Case (b): existing embed marker → refresh content
        mo = _OPEN_RE.match(line)
        if mo:
            rel = mo.group(1)
            src = examples_dir / rel
            # Skip everything up to and including <!-- /embed -->
            i += 1
            while i < len(lines) and lines[i].rstrip("\n") != _CLOSE:
                i += 1
            i += 1  # skip the closing marker line
            if src.exists():
                block = _code_block(src, rel)
                out.append(block + "\n")
                changed = True
            else:
                # Source gone — emit a warning comment and leave a placeholder
                out.append(f"<!-- embed: examples/{rel} -->\n")
                out.append(f"*Example `{rel}` not found.*\n")
                out.append(f"{_CLOSE}\n")
                changed = True
            continue

        out.append(lines[i])
        i += 1

    result = "".join(out)
    if result != original:
        mdx_path.write_text(result, encoding="utf-8")
        return True
    return False


def embed_all():
    for mdx in EXAMPLE_PAGES:
        if not mdx.exists():
            print(f"  [SKIP] {mdx.relative_to(REPO_ROOT)} (not found)")
            continue
        changed = _process(mdx)
        status = "updated" if changed else "unchanged"
        print(f"  {status}: {mdx.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    print("Embedding examples into MDX pages...")
    embed_all()
    print("Done.")
