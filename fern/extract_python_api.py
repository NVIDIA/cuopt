#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Extract Python API docstrings via AST (no imports, no GPU) and write MDX pages.

Usage:
    python fern/extract_python_api.py

Generates MDX for the Python API reference pages. Pure-Python files are parsed
with ast.parse(); Cython .pyx files are parsed with a regex line-scanner that
handles 'class Foo(Base):' while skipping 'cdef class' extension types.
"""

import ast
import inspect
import re
from pathlib import Path

try:
    from numpydoc.docscrape import NumpyDocString

    _NUMPYDOC_AVAILABLE = True
except ImportError:
    _NUMPYDOC_AVAILABLE = False

REPO_ROOT = Path(__file__).parent.parent
PYTHON_SRC = REPO_ROOT / "python/cuopt/cuopt"
PAGES = REPO_ROOT / "fern/docs/pages"
DOCS_V26_08 = REPO_ROOT / "fern/docs-v26-08.yml"

# ---------------------------------------------------------------------------
# Page configuration — maps output MDX path to source directories.
# Each directory is scanned non-recursively (*.py + *.pyx, no subdirs).
# ---------------------------------------------------------------------------
PAGE_SOURCES = [
    {
        "output": "cuopt-python/convex/convex-api.mdx",
        "title": "Convex Optimization Python API Reference",
        # RST order: Problem → Variable → LinearExpression → QuadraticExpression →
        #            Constraint → SolverSettings → CType → sense → Read
        "dirs": [
            # problem.py: Problem, Variable, LinearExpression, QuadraticExpression,
            #             Constraint, VType, CType, sense — all in source order
            {"path": "linear_programming", "use_init_all": False},
            # SolverSettings, SolverMethod, PDLPSolverMode
            {
                "path": "linear_programming/solver_settings",
                "use_init_all": True,
            },
            # Read, ParseMps
            {
                "path": "linear_programming/io",
                "use_init_all": True,
                "exclude": {"parser_wrapper"},
            },
            # DataModel (low-level; after main API)
            {
                "path": "linear_programming/data_model",
                "use_init_all": True,
                "exclude": {"data_model_wrapper"},
            },
            # Solution, PDLPWarmStartData
            {"path": "linear_programming/solution", "use_init_all": True},
            # Solve, BatchSolve
            {
                "path": "linear_programming/solver",
                "use_init_all": True,
                "exclude": {"solver_wrapper"},
            },
        ],
    },
    {
        "output": "cuopt-python/mip/mip-api.mdx",
        "title": "MIP Python API Reference",
        "dirs": [
            {
                "path": "linear_programming/data_model",
                "use_init_all": True,
                "exclude": {"data_model_wrapper"},
            },
            {"path": "linear_programming/solution", "use_init_all": True},
            {
                "path": "linear_programming/solver",
                "use_init_all": True,
                "exclude": {"solver_wrapper"},
            },
        ],
    },
    {
        "output": "cuopt-python/routing/routing-api.mdx",
        "title": "cuOpt Routing Python API Reference",
        # RST order: WaypointMatrix → SolutionStatus → DataModel → SolverSettings
        #            → Solve → BatchSolve → Assignment
        "dirs": [
            # WaypointMatrix comes first per existing RST
            {
                "path": "distance_engine",
                "use_init_all": True,
                "exclude": {"waypoint_matrix_wrapper"},
            },
            # routing/__init__.py __all__ filters internals; wrapper pyx get no heading
            {
                "path": "routing",
                "use_init_all": True,
                "exclude": {"validation", "waypoint_matrix_wrapper"},
                "order": [
                    "vehicle_routing",  # DataModel, SolverSettings, Solve, BatchSolve
                    "assignment",  # SolutionStatus, Assignment
                    "utils",  # generate_dataset + utility functions
                    "utils_wrapper",  # DatasetDistribution (inline, no heading)
                    "vehicle_routing_wrapper",  # ErrorStatus, NodeType, Objective (inline)
                ],
            },
        ],
    },
]


# ---------------------------------------------------------------------------
# AST helpers (.py files)
# ---------------------------------------------------------------------------


def _get_docstring(node) -> str:
    if (
        node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
        and isinstance(node.body[0].value.value, str)
    ):
        return inspect.cleandoc(node.body[0].value.value)
    return ""


def _format_sig(node) -> str:
    args = node.args
    parts = []
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
    if args.vararg:
        parts.append(f"*{args.vararg.arg}")
    if args.kwarg:
        parts.append(f"**{args.kwarg.arg}")
    ret = ""
    if node.returns:
        try:
            ret = " -> " + ast.unparse(node.returns)
        except Exception:
            pass
    return f"{node.name}({', '.join(parts)}){ret}"


def _parse_module(path: Path) -> dict:
    """Parse a .py file via AST and return classes, functions, module_doc."""
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
            members = []
            # Detect IntEnum subclasses: collect their member names
            bases = [
                ast.unparse(b) for b in node.bases if isinstance(b, ast.expr)
            ]
            is_enum = any("Enum" in b or "enum" in b.lower() for b in bases)
            for item in node.body:
                if is_enum and isinstance(item, ast.Assign):
                    for t in item.targets:
                        if isinstance(t, ast.Name) and not t.id.startswith(
                            "_"
                        ):
                            members.append(t.id)
                elif isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if item.name.startswith("_") and item.name != "__init__":
                        continue
                    sig = _format_sig(item)
                    doc = _get_docstring(item)
                    methods.append(
                        {"name": item.name, "doc": doc, "signature": sig}
                    )
            classes.append(
                {
                    "name": node.name,
                    "doc": _get_docstring(node),
                    "methods": methods,
                    "members": members,
                }
            )

        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.col_offset == 0 and not node.name.startswith("_"):
                sig = _format_sig(node)
                doc = _get_docstring(node)
                functions.append(
                    {"name": node.name, "doc": doc, "signature": sig}
                )

    return {
        "module_doc": module_doc,
        "classes": classes,
        "functions": functions,
    }


# ---------------------------------------------------------------------------
# Regex helpers (.pyx files)
# ---------------------------------------------------------------------------


def _collect_pyx_docstring(lines: list[str], start: int) -> str:
    """Collect a docstring starting at `start`, skipping blank lines."""
    i = start
    while i < len(lines) and not lines[i].strip():
        i += 1
    if i >= len(lines):
        return ""
    stripped = lines[i].strip()
    for q in ('"""', "'''"):
        if stripped.startswith(q):
            # Check for single-line docstring
            rest = stripped[len(q) :]
            if q in rest:
                return inspect.cleandoc(rest[: rest.index(q)])
            # Multi-line
            doc_lines = [rest]
            i += 1
            while i < len(lines):
                s = lines[i]
                if q in s:
                    doc_lines.append(s[: s.index(q)])
                    break
                doc_lines.append(s)
                i += 1
            return inspect.cleandoc("\n".join(doc_lines))
    return ""


def _parse_module_pyx(path: Path) -> dict:
    """
    Parse a Cython .pyx file for user-facing symbols:
      - Regular Python classes (class Foo(Base):) — enums, thin wrappers
      - Top-level def functions (not cpdef/cdef; those are C-layer internals)
    Skips 'cdef class' extension types (not importable as plain Python objects).
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return {"module_doc": "", "classes": [], "functions": []}

    lines = text.splitlines()
    classes = []
    functions = []

    i = 0
    while i < len(lines):
        raw = lines[i]
        stripped = raw.strip()
        indent = len(raw) - len(raw.lstrip())

        if indent != 0:
            i += 1
            continue

        # Skip cdef class / cpdef / cdef (C-layer definitions)
        if re.match(r"(cdef|cpdef)\s", stripped):
            i += 1
            continue

        # Regular Python class at module level
        m = re.match(r"class\s+([A-Za-z_]\w*)\s*(?:\([^)]*\))?\s*:", stripped)
        if m:
            name = m.group(1)
            if name.startswith("_"):
                i += 1
                continue
            # Base classes — detect enums
            base_match = re.match(r"class\s+\w+\s*\(([^)]*)\)", stripped)
            bases = base_match.group(1) if base_match else ""
            is_enum = "Enum" in bases or "enum" in bases.lower()

            doc = _collect_pyx_docstring(lines, i + 1)

            # Collect body: enum members and def methods
            members = []
            methods = []
            j = i + 1
            while j < len(lines):
                bline = lines[j]
                bindent = len(bline) - len(bline.lstrip())
                bstripped = bline.strip()

                # End of class body: unindented non-blank line
                if bstripped and bindent == 0:
                    break

                if bindent > 0 and bstripped:
                    if is_enum:
                        # Match both SCREAMING_SNAKE_CASE and PascalCase enum members
                        em = re.match(r"([A-Za-z_]\w*)\s*=", bstripped)
                        if em and not em.group(1).startswith("_"):
                            members.append(em.group(1))
                    # def methods (skip cdef/cpdef)
                    dm = re.match(r"def\s+([A-Za-z_]\w*)\s*\(", bstripped)
                    if dm:
                        mname = dm.group(1)
                        if not mname.startswith("_") or mname == "__init__":
                            mdoc = _collect_pyx_docstring(lines, j + 1)
                            # Collect the signature (single line for simplicity)
                            sig_m = re.match(
                                r"def\s+(\w+\s*\([^)]*\))", bstripped
                            )
                            sig = sig_m.group(1) if sig_m else f"{mname}(...)"
                            methods.append(
                                {"name": mname, "doc": mdoc, "signature": sig}
                            )
                j += 1

            classes.append(
                {
                    "name": name,
                    "doc": doc,
                    "methods": methods,
                    "members": members,
                }
            )
            i = j
            continue

        # Top-level def (not cpdef/cdef — already skipped above)
        fm = re.match(r"def\s+([A-Za-z_]\w*)\s*\(", stripped)
        if fm:
            fname = fm.group(1)
            if not fname.startswith("_"):
                doc = _collect_pyx_docstring(lines, i + 1)
                # Collect full signature (may span multiple lines until ":")
                sig_lines = [stripped.rstrip(":")]
                j = i + 1
                while j < len(lines) and ":" not in sig_lines[-1]:
                    sig_lines.append(lines[j].strip())
                    j += 1
                sig = " ".join(sig_lines).rstrip(":")
                functions.append({"name": fname, "doc": doc, "signature": sig})

        i += 1

    return {"module_doc": "", "classes": classes, "functions": functions}


# ---------------------------------------------------------------------------
# MDX renderers
# ---------------------------------------------------------------------------

_EXAMPLE_SECTIONS = {"Examples", "Example"}
_LIST_SECTIONS = {
    "Parameters",
    "Returns",
    "Raises",
    "Attributes",
    "Other Parameters",
    "Yields",
}


def _doc_to_mdx(doc: str) -> str:
    """Minimal docstring → MDX: preserve code blocks, wrap numpy-style sections."""
    if not doc:
        return ""
    lines = doc.splitlines()
    out = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if (
            i + 1 < len(lines)
            and lines[i + 1].strip()
            and all(c == "-" for c in lines[i + 1].strip())
        ):
            section = line.strip()
            out.append(f"\n**{section}**\n")
            i += 2
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
                code = "\n".join(body_lines).strip()
                if code:
                    out.append("```python")
                    out.append(code)
                    out.append("```")
                out.append("")
            elif section in _LIST_SECTIONS:
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
                        continue
                    if pline[0].isspace():
                        current_desc.append(pline.strip())
                    else:
                        s = pline.strip()
                        if s.endswith(":") and " : " not in s:
                            if current_name is not None:
                                entries.append((current_name, current_desc))
                                current_name = None
                            in_list = False
                            remaining.append(pline)
                        else:
                            if current_name is not None:
                                entries.append((current_name, current_desc))
                            current_name = s
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
                out.extend(body_lines)
                out.append("")
            continue
        out.append(line)
        i += 1
    return "\n".join(out)


def _doc_to_mdx_numpydoc(doc: str) -> str:
    """Parse a NumPy-style docstring and render as MDX. Falls back to _doc_to_mdx."""
    if not doc:
        return ""
    if not _NUMPYDOC_AVAILABLE:
        return _doc_to_mdx(doc)
    try:
        parsed = NumpyDocString(doc)
    except Exception:
        return _doc_to_mdx(doc)

    has_content = (
        parsed["Summary"]
        or parsed["Extended Summary"]
        or parsed["Parameters"]
        or parsed["Returns"]
        or parsed["Raises"]
        or parsed["Examples"]
        or parsed["Notes"]
        or parsed["Attributes"]
    )
    if not has_content:
        return _doc_to_mdx(doc)

    out = []
    if parsed["Summary"]:
        out.append(" ".join(parsed["Summary"]))
        out.append("")
    if parsed["Extended Summary"]:
        out.append(" ".join(parsed["Extended Summary"]))
        out.append("")

    def _render_params(label, items):
        if not items:
            return
        out.append(f"**{label}**")
        out.append("")
        for p in items:
            name = p.name or ""
            ptype = p.type or ""
            desc = " ".join(p.desc).strip()
            header = (
                f"- **`{name}`** (`{ptype}`)" if ptype else f"- **`{name}`**"
            )
            out.append(f"{header} — {desc}" if desc else header)
        out.append("")

    _render_params("Parameters", parsed["Parameters"])
    _render_params("Attributes", parsed["Attributes"])
    _render_params("Returns", parsed["Returns"])

    if parsed["Raises"]:
        out.append("**Raises**")
        out.append("")
        for r in parsed["Raises"]:
            name = r.name or ""
            desc = " ".join(r.desc).strip()
            out.append(
                f"- **`{name}`** — {desc}" if desc else f"- **`{name}`**"
            )
        out.append("")
    if parsed["Notes"]:
        out.append("**Notes**")
        out.append("")
        out.extend(parsed["Notes"])
        out.append("")
    if parsed["Examples"]:
        out.append("**Examples**")
        out.append("")
        code_lines = [ln for ln in parsed["Examples"] if ln.strip()]
        if code_lines:
            out.append("```python")
            out.extend(parsed["Examples"])
            out.append("```")
        out.append("")

    return "\n".join(out)


def _render_class(cls: dict, level: int = 2) -> str:
    h = "#" * level
    h2 = "#" * (level + 1)
    lines = ["<hr />\n", f"{h} class `{cls['name']}`\n"]
    if cls.get("doc"):
        lines.append(_doc_to_mdx_numpydoc(cls["doc"]))
        lines.append("")
    if cls.get("members"):
        lines.append(
            "**Members:** " + ", ".join(f"`{m}`" for m in cls["members"])
        )
        lines.append("")
    for method in cls.get("methods", []):
        name = method["name"]
        sig = method["signature"]
        lines.append(f"{h2} def `{name}`\n")
        lines.append(f"```python\n{sig}\n```\n")
        if method.get("doc"):
            lines.append(_doc_to_mdx_numpydoc(method["doc"]))
            lines.append("")
    return "\n".join(lines)


def _render_function(fn: dict, level: int = 2) -> str:
    h = "#" * level
    lines = ["<hr />\n", f"{h} def `{fn['name']}`\n"]
    lines.append(f"```python\n{fn['signature']}\n```\n")
    if fn.get("doc"):
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
        for extra in extras or []:
            lines.append(extra)
    content = "\n".join(lines) + "\n"
    content = re.sub(r"\\?<=", "&lt;=", content)
    dest.write_text(content, encoding="utf-8")
    print(f"  Written: {dest.relative_to(REPO_ROOT)}")


# ---------------------------------------------------------------------------
# Directory scanner
# ---------------------------------------------------------------------------


def _read_init_all(directory: Path) -> set[str] | None:
    """Read __all__ from directory/__init__.py via ast.literal_eval. Returns None if absent."""
    init_path = directory / "__init__.py"
    if not init_path.exists():
        return None
    try:
        tree = ast.parse(init_path.read_text(encoding="utf-8"))
    except (SyntaxError, OSError):
        return None
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    try:
                        return set(ast.literal_eval(node.value))
                    except Exception:
                        return None
    return None


def _scan_dir(
    src_dir: Path,
    exclude: set,
    use_init_all: bool = False,
    order: list | None = None,
) -> list[tuple[str, dict]]:
    """
    Return (section_heading, parsed_module) for every public .py and .pyx
    file in src_dir (non-recursive).

    - Private modules (single leading underscore) and excluded stems are skipped.
    - .py wins over .pyx for the same stem (Python wrapper preferred over Cython).
    - When use_init_all=True, symbols not in src_dir/__init__.py __all__ are dropped.
    - Cross-file deduplication: if the same symbol name appears in multiple files,
      only the first occurrence (order-then-alphabetical) is kept.
    - order: list of stem names controlling processing sequence; unlisted stems
      go at the end in alphabetical order.
    - *_wrapper.pyx files never emit a section heading (symbols appear inline).
    """
    if not src_dir.exists():
        return []

    allowed_symbols: set[str] | None = (
        _read_init_all(src_dir) if use_init_all else None
    )

    # Collect .py and .pyx files, deduplicated by stem (.py wins over .pyx)
    seen_stems: dict[str, Path] = {}
    for path in sorted(src_dir.iterdir()):
        if path.suffix not in (".py", ".pyx"):
            continue
        if path.name.startswith("_") and not path.name.startswith("__"):
            continue
        if path.stem in exclude:
            continue
        if path.stem in seen_stems:
            if path.suffix == ".py":
                seen_stems[path.stem] = path
        else:
            seen_stems[path.stem] = path

    # Sort stems: items in `order` come first (in list order), rest alphabetically.
    _order = order or []

    def _sort_key(stem: str) -> tuple:
        try:
            return (0, _order.index(stem), stem)
        except ValueError:
            return (1, 0, stem)

    seen_symbol_names: set[str] = set()
    results = []
    for stem, path in sorted(
        seen_stems.items(), key=lambda kv: _sort_key(kv[0])
    ):
        parsed = (
            _parse_module_pyx(path)
            if path.suffix == ".pyx"
            else _parse_module(path)
        )

        # Wrapper .pyx files expose C-layer internals as functions — drop them;
        # only enums/classes defined there are public API.
        if path.suffix == ".pyx" and "_wrapper" in path.stem:
            parsed["functions"] = []

        # Filter by __all__ if requested
        if allowed_symbols is not None:
            parsed["classes"] = [
                c for c in parsed["classes"] if c["name"] in allowed_symbols
            ]
            parsed["functions"] = [
                f for f in parsed["functions"] if f["name"] in allowed_symbols
            ]

        # Cross-file dedup: drop symbols already emitted by an earlier file
        parsed["classes"] = [
            c for c in parsed["classes"] if c["name"] not in seen_symbol_names
        ]
        parsed["functions"] = [
            f
            for f in parsed["functions"]
            if f["name"] not in seen_symbol_names
        ]
        seen_symbol_names.update(c["name"] for c in parsed["classes"])
        seen_symbol_names.update(f["name"] for f in parsed["functions"])

        if not (parsed["classes"] or parsed["functions"]):
            continue

        # Wrapper .pyx files don't get their own ## section heading — their
        # symbols appear inline without a heading break.
        is_wrapper_pyx = path.suffix == ".pyx" and "_wrapper" in path.stem
        heading = (
            ""
            if is_wrapper_pyx
            else (
                src_dir.name.replace("_", " ").title()
                if path.name == "__init__.py"
                else stem.replace("_", " ").title()
            )
        )
        results.append((heading, parsed))
    return results


# ---------------------------------------------------------------------------
# Navigation updater
# ---------------------------------------------------------------------------

_PY_NAV_START = "      - section: Routing Optimization\n"
_PY_NAV_SECTION = "      - section: Python API Overview\n"

# Marker comments in docs-v26-08.yml that bracket the auto-generated Python API pages.
# Insert these markers manually around the Python API page entries if you want
# update_python_api_navigation() to manage them. Without markers the function is a no-op.
_NAV_START_MARKER = "        # BEGIN auto-generated python-api pages\n"
_NAV_END_MARKER = "        # END auto-generated python-api pages\n"


def update_python_api_navigation(generated_pages: list[dict]) -> None:
    """
    Rewrite the auto-generated Python API page entries in docs-v26-08.yml.
    Only operates if the BEGIN/END marker comments are present in the file.
    """
    if not DOCS_V26_08.exists():
        return
    text = DOCS_V26_08.read_text(encoding="utf-8")
    if _NAV_START_MARKER not in text or _NAV_END_MARKER not in text:
        return  # Markers not present; leave navigation alone

    start = text.index(_NAV_START_MARKER) + len(_NAV_START_MARKER)
    end = text.index(_NAV_END_MARKER)

    new_entries = []
    for page in generated_pages:
        rel_path = Path(page["output"])
        title = page["title"]
        nav_path = f"docs/pages/{rel_path}"
        new_entries.append(f"              - page: {title}\n")
        new_entries.append(f"                path: {nav_path}\n")

    new_block = "".join(new_entries)
    updated = text[:start] + new_block + text[end:]
    DOCS_V26_08.write_text(updated, encoding="utf-8")
    print(f"  Updated navigation in {DOCS_V26_08.relative_to(REPO_ROOT)}")


# ---------------------------------------------------------------------------
# Page generator
# ---------------------------------------------------------------------------


def generate_pages():
    for page in PAGE_SOURCES:
        sections = []
        for dir_cfg in page["dirs"]:
            # dirs entries are either plain strings or dicts with path + options
            if isinstance(dir_cfg, str):
                dir_path = dir_cfg
                exclude = set()
                use_init_all = False
            else:
                dir_path = dir_cfg["path"]
                exclude = set(dir_cfg.get("exclude", set()))
                use_init_all = bool(dir_cfg.get("use_init_all", False))

            order = (
                list(dir_cfg.get("order", []))
                if isinstance(dir_cfg, dict)
                else []
            )
            src_dir = PYTHON_SRC / dir_path
            for heading, parsed in _scan_dir(
                src_dir, exclude, use_init_all, order
            ):
                sections.append((heading, parsed, []))

        if not sections:
            sections = [
                (
                    "",
                    None,
                    [
                        "<Note>No public API symbols found in the configured source directories.</Note>"
                    ],
                )
            ]

        dest = PAGES / page["output"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        write_api_page(title=page["title"], dest=dest, sections=sections)

    update_python_api_navigation(PAGE_SOURCES)


if __name__ == "__main__":
    print(
        "Extracting Python API docstrings (AST + Cython regex, no imports)..."
    )
    generate_pages()
    print("Done.")
