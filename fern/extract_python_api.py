#!/usr/bin/env python3
"""
Extract Python API docstrings via AST (no imports, no GPU) and write MDX pages.

Usage:
    python fern/extract_python_api.py

Generates MDX for the three Python API reference pages that were previously
handled by Sphinx autodoc (.. autoclass:: / .. autofunction::).
"""

import ast
import re
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
PYTHON_SRC = REPO_ROOT / "python/cuopt/cuopt"
PAGES = REPO_ROOT / "fern/docs/pages"


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
        return textwrap.dedent(node.body[0].value.value).strip()
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

def _doc_to_mdx(doc: str) -> str:
    """Minimal docstring → MDX: preserve code blocks, wrap numpy-style sections."""
    if not doc:
        return ""
    # Detect numpy-style sections (Parameters, Returns, etc.)
    lines = doc.splitlines()
    out = []
    i = 0
    while i < len(lines):
        line = lines[i]
        # Section header: "Parameters\n----------"
        if (
            i + 1 < len(lines)
            and lines[i + 1].strip()
            and all(c == "-" for c in lines[i + 1].strip())
        ):
            out.append(f"\n**{line.strip()}**\n")
            i += 2
            continue
        out.append(line)
        i += 1
    return "\n".join(out)


def _render_class(cls: dict, level: int = 2) -> str:
    h = "#" * level
    lines = [f"{h} {cls['name']}\n"]
    if cls["doc"]:
        lines.append(_doc_to_mdx(cls["doc"]))
        lines.append("")
    for method in cls["methods"]:
        lines.append(f"{'#' * (level + 1)} `{method['signature']}`\n")
        if method["doc"]:
            lines.append(_doc_to_mdx(method["doc"]))
            lines.append("")
    return "\n".join(lines)


def _render_function(fn: dict, level: int = 2) -> str:
    h = "#" * level
    lines = [f"{h} `{fn['signature']}`\n"]
    if fn["doc"]:
        lines.append(_doc_to_mdx(fn["doc"]))
        lines.append("")
    return "\n".join(lines)


def write_api_page(title: str, dest: Path, sections: list):
    """
    sections: list of (heading, parsed_module_dict | None, [extra_mdx_lines])
    """
    lines = [f'---\ntitle: "{title}"\n---\n\n# {title}\n']
    for heading, parsed, extras in sections:
        if heading:
            lines.append(f"\n## {heading}\n")
        if parsed:
            if parsed.get("module_doc"):
                lines.append(_doc_to_mdx(parsed["module_doc"]) + "\n")
            for cls in parsed.get("classes", []):
                lines.append(_render_class(cls, level=3))
            for fn in parsed.get("functions", []):
                lines.append(_render_function(fn, level=3))
        for extra in (extras or []):
            lines.append(extra)
    dest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  Written: {dest.relative_to(REPO_ROOT)}")


# ---------------------------------------------------------------------------
# Page definitions — mirrors the autodoc directives in the RST source
# ---------------------------------------------------------------------------

def generate_pages():
    # 1. Convex Optimization (LP/QP) Python API
    lp_problem = _parse_module(PYTHON_SRC / "linear_programming/problem.py")
    lp_settings = _parse_module(PYTHON_SRC / "linear_programming/solver_settings/__init__.py")
    lp_io_path = PYTHON_SRC / "linear_programming/io/__init__.py"
    lp_io = _parse_module(lp_io_path) if lp_io_path.exists() else None

    # Filter to exposed classes matching the autodoc RST
    exposed_lp_classes = {
        "Problem", "Variable", "LinearExpression", "QuadraticExpression",
        "Constraint", "CType", "sense",
    }
    lp_problem["classes"] = [
        c for c in lp_problem["classes"] if c["name"] in exposed_lp_classes
    ]
    exposed_settings = {"SolverSettings"}
    lp_settings["classes"] = [
        c for c in lp_settings["classes"] if c["name"] in exposed_settings
    ]

    write_api_page(
        title="Convex Optimization Python API Reference",
        dest=PAGES / "cuopt-python/convex/convex-api.mdx",
        sections=[
            ("Problem & Variables", lp_problem, []),
            ("Solver Settings", lp_settings, []),
            ("I/O", lp_io, []) if lp_io else ("", None, []),
        ],
    )

    # 2. MIP Python API
    mip_path = PYTHON_SRC / "linear_programming/mip"
    if mip_path.exists():
        mip_files = list(mip_path.glob("*.py"))
        mip_sections = []
        for f in sorted(mip_files):
            if f.name.startswith("_"):
                continue
            parsed = _parse_module(f)
            if parsed["classes"] or parsed["functions"]:
                mip_sections.append((f.stem.replace("_", " ").title(), parsed, []))
        if not mip_sections:
            mip_sections = [("", None, ["<Note>MIP Python API shares classes with the LP API. See [Convex Optimization API](convex-api).</Note>"])]
    else:
        mip_sections = [("", None, ["<Note>MIP Python API shares classes with the LP API. See [Convex Optimization API](convex-api).</Note>"])]

    write_api_page(
        title="MIP Python API Reference",
        dest=PAGES / "cuopt-python/mip/mip-api.mdx",
        sections=mip_sections,
    )

    # 3. Routing Python API
    routing_init = _parse_module(PYTHON_SRC / "routing/__init__.py")
    routing_main = _parse_module(PYTHON_SRC / "routing/vehicle_routing.py")
    routing_assignment = _parse_module(PYTHON_SRC / "routing/assignment.py")

    # Expose only documented public API
    exposed_routing = {"DataModel", "SolverSettings", "Assignment", "SolutionStatus"}
    routing_main["classes"] = [
        c for c in routing_main["classes"] if c["name"] in exposed_routing
    ]

    write_api_page(
        title="cuOpt Routing Python API Reference",
        dest=PAGES / "cuopt-python/routing/routing-api.mdx",
        sections=[
            ("Core Classes", routing_main, []),
            ("Assignment & Results", routing_assignment, []),
            ("Module Functions (Solve, BatchSolve)", routing_init, []),
        ],
    )

    # 4. Server thin client API (sh-cli-api)
    sc_path = PYTHON_SRC.parent.parent.parent / "python/cuopt_self_hosted"
    sc_client = sc_path / "cuopt_self_hosted/cuopt_sh_client" if sc_path.exists() else None
    if sc_client and sc_client.exists():
        sc_init = _parse_module(sc_client / "__init__.py")
        write_api_page(
            title="Self-Hosted Service Client API Reference",
            dest=PAGES / "cuopt-server/client-api/sh-cli-api.mdx",
            sections=[("Client API", sc_init, [])],
        )
    else:
        (PAGES / "cuopt-server/client-api/sh-cli-api.mdx").write_text(
            '---\ntitle: "Self-Hosted Service Client API Reference"\n---\n\n'
            "# Self-Hosted Service Client API Reference\n\n"
            "<Note>Install `cuopt-sh-client` to access the thin client API. See [Build Your Own Client](sh-cli-build).</Note>\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    print("Extracting Python API docstrings (AST, no imports)...")
    generate_pages()
    print("Done.")
