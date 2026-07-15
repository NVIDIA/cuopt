#!/usr/bin/env python3
"""
Extract Python API docstrings via AST (no imports, no GPU) and write MDX pages.

Usage:
    python fern/extract_python_api.py

Generates MDX for the three Python API reference pages that were previously
handled by Sphinx autodoc (.. autoclass:: / .. autofunction::).
"""

import ast
import inspect
import re
import textwrap
from pathlib import Path

try:
    from numpydoc.docscrape import NumpyDocString
    _NUMPYDOC_AVAILABLE = True
except ImportError:
    _NUMPYDOC_AVAILABLE = False

REPO_ROOT = Path(__file__).parent.parent
PYTHON_SRC = REPO_ROOT / "python/cuopt/cuopt"
PAGES = REPO_ROOT / "fern/docs/pages"

# ---------------------------------------------------------------------------
# Page configuration — maps output MDX path to source directories.
# Each directory is scanned non-recursively (*.py files only, no subdirs).
# Adding a new .py file to a listed directory makes it appear automatically.
# To exclude a specific file, add its stem to the page's "exclude" set.
# ---------------------------------------------------------------------------
PAGE_SOURCES = [
    {
        "output": "cuopt-python/convex/convex-api.mdx",
        "title": "Convex Optimization Python API Reference",
        "dirs": [
            "linear_programming",
            "linear_programming/solver_settings",
            "linear_programming/io",
        ],
        "exclude": set(),
    },
    {
        "output": "cuopt-python/mip/mip-api.mdx",
        "title": "MIP Python API Reference",
        "dirs": ["linear_programming/mip"],
        "exclude": set(),
    },
    {
        "output": "cuopt-python/routing/routing-api.mdx",
        "title": "cuOpt Routing Python API Reference",
        "dirs": ["routing"],
        "exclude": set(),
    },
]


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------

def _get_docstring(node) -> str:
    """Extract the first string literal from a class/function body."""
    if (
        node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
        and isinstance(node.body[0].value.value, str)
    ):
        return inspect.cleandoc(node.body[0].value.value)
    return ""


def _parse_module(path: Path) -> dict:
    """
    Parse a Python file and return:
        {
          "module_doc": str,
          "classes": [{"name", "doc", "methods": [{"name", "doc", "signature"}]}],
          "functions": [{"name", "doc", "signature"}],
        }
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError as e:
        print(f"  [WARN] syntax error in {path}: {e}")
        return {"module_doc": "", "classes": [], "functions": []}

    module_doc = ast.get_docstring(tree) or ""
    classes = []
    functions = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.col_offset == 0:
            methods = []
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if item.name.startswith("_") and item.name != "__init__":
                        continue
                    sig = _format_sig(item)
                    doc = _get_docstring(item)
                    methods.append({"name": item.name, "doc": doc, "signature": sig})
            classes.append({
                "name": node.name,
                "doc": _get_docstring(node),
                "methods": methods,
            })

        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.col_offset == 0 and not node.name.startswith("_"):
                sig = _format_sig(node)
                doc = _get_docstring(node)
                functions.append({"name": node.name, "doc": doc, "signature": sig})

    return {"module_doc": module_doc, "classes": classes, "functions": functions}


def _format_sig(node) -> str:
    """Build a simplified function signature string."""
    args = node.args
    parts = []
    # positional args
    n_defaults = len(args.defaults)
    n_args = len(args.args)
    for i, arg in enumerate(args.args):
        default_idx = i - (n_args - n_defaults)
        if arg.arg == "self":
            continue
        annot = ""
        if arg.annotation:
            try:
                annot = ": " + ast.unparse(arg.annotation)
            except Exception:
                pass
        if default_idx >= 0:
            try:
                default = " = " + ast.unparse(args.defaults[default_idx])
            except Exception:
                default = " = ..."
            parts.append(f"{arg.arg}{annot}{default}")
        else:
            parts.append(f"{arg.arg}{annot}")
    # *args
    if args.vararg:
        parts.append(f"*{args.vararg.arg}")
    # **kwargs
    if args.kwarg:
        parts.append(f"**{args.kwarg.arg}")
    ret = ""
    if node.returns:
        try:
            ret = " -> " + ast.unparse(node.returns)
        except Exception:
            pass
    return f"{node.name}({', '.join(parts)}){ret}"


# ---------------------------------------------------------------------------
# MDX renderers
# ---------------------------------------------------------------------------

_EXAMPLE_SECTIONS = {"Examples", "Example"}
_LIST_SECTIONS = {"Parameters", "Returns", "Raises", "Attributes", "Other Parameters", "Yields"}


def _doc_to_mdx(doc: str) -> str:
    """Minimal docstring → MDX: preserve code blocks, wrap numpy-style sections."""
    if not doc:
        return ""
    lines = doc.splitlines()
    out = []
    i = 0
    while i < len(lines):
        line = lines[i]
        # Numpy-style section header: "Name\n---------"
        if (
            i + 1 < len(lines)
            and lines[i + 1].strip()
            and all(c == "-" for c in lines[i + 1].strip())
        ):
            section = line.strip()
            out.append(f"\n**{section}**\n")
            i += 2

            # Collect the raw body lines until we hit the next section or EOF
            body_lines = []
            while i < len(lines):
                pline = lines[i]
                if (
                    i + 1 < len(lines)
                    and pline.strip()
                    and not pline[0].isspace()
                    and lines[i + 1].strip()
                    and all(c == "-" for c in lines[i + 1].strip())
                ):
                    break
                body_lines.append(pline)
                i += 1

            if section in _EXAMPLE_SECTIONS:
                # Wrap example code in a fenced code block
                code = "\n".join(body_lines).strip()
                if code:
                    out.append("```python")
                    out.append(code)
                    out.append("```")
                out.append("")
            elif section in _LIST_SECTIONS:
                # Render numpy-style entries as a markdown list.
                # Each unindented line that looks like a param ("name : type")
                # starts a new entry; its indented follow-on lines are the desc.
                # Stop list collection when an unindented line looks like prose
                # (e.g., "Note:") — pass remaining lines through as-is.
                entries = []
                current_name = None
                current_desc = []
                remaining = []
                in_list = True
                for pline in body_lines:
                    if not in_list:
                        remaining.append(pline)
                        continue
                    if not pline.strip():
                        # blank line — keep going
                        continue
                    if pline[0].isspace():
                        current_desc.append(pline.strip())
                    else:
                        stripped = pline.strip()
                        # Param entries have " : " or are bare identifiers.
                        # Prose lines (like "Note:", "Warning:") end with ":"
                        # and don't have " : " (type separator).
                        if stripped.endswith(":") and " : " not in stripped:
                            if current_name is not None:
                                entries.append((current_name, current_desc))
                                current_name = None
                            in_list = False
                            remaining.append(pline)
                        else:
                            if current_name is not None:
                                entries.append((current_name, current_desc))
                            current_name = stripped
                            current_desc = []
                if current_name is not None:
                    entries.append((current_name, current_desc))
                for name, desc in entries:
                    desc_text = " ".join(desc).strip()
                    if desc_text:
                        out.append(f"- **`{name}`** — {desc_text}")
                    else:
                        out.append(f"- **`{name}`**")
                if remaining:
                    out.append("")
                    out.extend(remaining)
                out.append("")
            else:
                # Other sections (Notes, See Also, etc.) — pass through as-is
                out.extend(body_lines)
                out.append("")
            continue

        out.append(line)
        i += 1
    return "\n".join(out)


def _doc_to_mdx_numpydoc(doc: str) -> str:
    """
    Parse a docstring with numpydoc and render it as MDX markdown.
    Falls back to _doc_to_mdx() if numpydoc is not available or parsing
    produces no useful content.
    """
    if not doc:
        return ""
    if not _NUMPYDOC_AVAILABLE:
        return _doc_to_mdx(doc)

    try:
        parsed = NumpyDocString(doc)
    except Exception:
        return _doc_to_mdx(doc)

    # Check if numpydoc extracted anything useful
    summary = parsed["Summary"]
    has_content = (
        summary
        or parsed["Extended Summary"]
        or parsed["Parameters"]
        or parsed["Returns"]
        or parsed["Raises"]
        or parsed["Examples"]
        or parsed["Notes"]
    )
    if not has_content:
        return _doc_to_mdx(doc)

    out = []

    # Summary
    if summary:
        out.append(" ".join(summary))
        out.append("")

    # Extended Summary
    ext = parsed["Extended Summary"]
    if ext:
        out.append(" ".join(ext))
        out.append("")

    # Parameters
    params = parsed["Parameters"]
    if params:
        out.append("**Parameters**")
        out.append("")
        for p in params:
            name = p.name or ""
            ptype = p.type or ""
            desc = " ".join(p.desc).strip()
            if ptype:
                header = f"- **`{name}`** (`{ptype}`)"
            else:
                header = f"- **`{name}`**"
            if desc:
                out.append(f"{header} — {desc}")
            else:
                out.append(header)
        out.append("")

    # Returns
    returns = parsed["Returns"]
    if returns:
        out.append("**Returns**")
        out.append("")
        for r in returns:
            name = r.name or ""
            rtype = r.type or ""
            desc = " ".join(r.desc).strip()
            if name and rtype:
                header = f"- **`{name}`** (`{rtype}`)"
            elif name:
                header = f"- **`{name}`**"
            elif rtype:
                header = f"- (`{rtype}`)"
            else:
                header = "-"
            if desc:
                out.append(f"{header} — {desc}")
            else:
                out.append(header)
        out.append("")

    # Raises
    raises = parsed["Raises"]
    if raises:
        out.append("**Raises**")
        out.append("")
        for r in raises:
            name = r.name or ""
            desc = " ".join(r.desc).strip()
            if desc:
                out.append(f"- **`{name}`** — {desc}")
            else:
                out.append(f"- **`{name}`**")
        out.append("")

    # Notes
    notes = parsed["Notes"]
    if notes:
        out.append("**Notes**")
        out.append("")
        out.extend(notes)
        out.append("")

    # Examples
    examples = parsed["Examples"]
    if examples:
        out.append("**Examples**")
        out.append("")
        # numpydoc returns example lines as-is; wrap in a fenced code block
        # if they look like code, otherwise pass through
        code_lines = [ln for ln in examples if ln.strip()]
        if code_lines:
            out.append("```python")
            out.extend(examples)
            out.append("```")
        out.append("")

    return "\n".join(out)


def _render_class(cls: dict, level: int = 2) -> str:
    h = "#" * level
    h2 = "#" * (level + 1)
    lines = ["<hr />\n", f"{h} class `{cls['name']}`\n"]
    if cls["doc"]:
        lines.append(_doc_to_mdx_numpydoc(cls["doc"]))
        lines.append("")
    for method in cls["methods"]:
        name = method["name"]
        sig = method["signature"]
        lines.append(f"{h2} def `{name}`\n")
        lines.append(f"```python\n{sig}\n```\n")
        if method["doc"]:
            lines.append(_doc_to_mdx_numpydoc(method["doc"]))
            lines.append("")
    return "\n".join(lines)


def _render_function(fn: dict, level: int = 2) -> str:
    h = "#" * level
    lines = ["<hr />\n", f"{h} def `{fn['name']}`\n"]
    lines.append(f"```python\n{fn['signature']}\n```\n")
    if fn["doc"]:
        lines.append(_doc_to_mdx_numpydoc(fn["doc"]))
        lines.append("")
    return "\n".join(lines)


def write_api_page(title: str, dest: Path, sections: list):
    """
    sections: list of (heading, parsed_module_dict | None, [extra_mdx_lines])
    """
    lines = [f'---\ntitle: "{title}"\n---\n']
    for heading, parsed, extras in sections:
        if heading:
            lines.append(f"\n## {heading}\n")
        if parsed:
            if parsed.get("module_doc"):
                lines.append(_doc_to_mdx_numpydoc(parsed["module_doc"]) + "\n")
            for cls in parsed.get("classes", []):
                lines.append(_render_class(cls, level=3))
            for fn in parsed.get("functions", []):
                lines.append(_render_function(fn, level=3))
        for extra in (extras or []):
            lines.append(extra)
    content = "\n".join(lines) + "\n"
    # MDX parses `<` as a JSX tag start; escape `<=` comparison operators.
    content = re.sub(r'\\?<=', '&lt;=', content)
    dest.write_text(content, encoding="utf-8")
    print(f"  Written: {dest.relative_to(REPO_ROOT)}")


# ---------------------------------------------------------------------------
# Page generator — driven by PAGE_SOURCES config above
# ---------------------------------------------------------------------------

def _scan_dir(src_dir: Path, exclude: set) -> list[tuple[str, dict]]:
    """
    Return (section_heading, parsed_module) for every public .py file in
    src_dir (non-recursive). Skips __pycache__, _private.py, and excluded stems.
    __init__.py is included when it contains public symbols.
    """
    if not src_dir.exists():
        return []
    results = []
    for py_file in sorted(src_dir.glob("*.py")):
        # Skip private modules (single leading underscore, e.g. _internal.py)
        # but include __init__.py which may expose module-level API.
        if py_file.name.startswith("_") and not py_file.name.startswith("__"):
            continue
        if py_file.stem in exclude:
            continue
        parsed = _parse_module(py_file)
        if not (parsed["classes"] or parsed["functions"]):
            continue
        if py_file.name == "__init__.py":
            # Use parent directory name as heading for init modules
            heading = src_dir.name.replace("_", " ").title()
        else:
            heading = py_file.stem.replace("_", " ").title()
        results.append((heading, parsed))
    return results


def generate_pages():
    for page in PAGE_SOURCES:
        sections = []
        for dir_rel in page["dirs"]:
            src_dir = PYTHON_SRC / dir_rel
            for heading, parsed in _scan_dir(src_dir, page.get("exclude", set())):
                sections.append((heading, parsed, []))

        if not sections:
            sections = [("", None, ["<Note>No public API symbols found in the configured source directories.</Note>"])]

        dest = PAGES / page["output"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        write_api_page(title=page["title"], dest=dest, sections=sections)


if __name__ == "__main__":
    print("Extracting Python API docstrings (AST, no imports)...")
    generate_pages()
    print("Done.")
