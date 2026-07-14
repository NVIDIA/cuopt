#!/usr/bin/env python3
"""
Parse cuopt_c.h and constants.h and regenerate MDX pages for the C API
reference, replacing the Doxygen-stub <Note> placeholders.

Usage:
    python fern/extract_c_api.py

Regenerates:
    fern/docs/pages/cuopt-c/convex/convex-c-api.mdx
    fern/docs/pages/cuopt-c/mip/mip-c-api.mdx
"""

import re
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
HEADER = REPO_ROOT / "cpp/include/cuopt/mathematical_optimization/cuopt_c.h"
CONSTANTS = REPO_ROOT / "cpp/include/cuopt/mathematical_optimization/constants.h"
RST_CONVEX = REPO_ROOT / "docs/cuopt/source/cuopt-c/convex/convex-c-api.rst"
RST_MIP = REPO_ROOT / "docs/cuopt/source/cuopt-c/mip/mip-c-api.rst"
PAGES = REPO_ROOT / "fern/docs/pages"
OUT_CONVEX = PAGES / "cuopt-c/convex/convex-c-api.mdx"
OUT_MIP = PAGES / "cuopt-c/mip/mip-c-api.mdx"


# ---------------------------------------------------------------------------
# Header parser
# ---------------------------------------------------------------------------

def _strip_comment_markers(text: str) -> str:
    """Strip /** ... */ and /* ... */ markers and leading * from lines."""
    # Remove opening /** or /*
    text = re.sub(r"^/\*+\s*", "", text.strip())
    # Remove closing */
    text = re.sub(r"\s*\*/$", "", text.strip())
    # Strip leading " * " from each line
    lines = []
    for line in text.splitlines():
        line = re.sub(r"^\s*\*\s?", "", line)
        lines.append(line)
    return "\n".join(lines).strip()


def _clean_brief(text: str) -> str:
    """Remove @brief tag and return the cleaned text."""
    text = re.sub(r"@brief\s*", "", text).strip()
    # Collapse interior whitespace runs (doxygen wrapping)
    return text


def _parse_doxygen_block(raw_comment: str) -> dict:
    """
    Parse a doxygen /** ... */ block into:
        {
          "brief": str,
          "params": [{"name": str, "dir": str, "desc": str}],
          "return": str,
          "note": str,
          "deprecated": str,
        }
    """
    text = _strip_comment_markers(raw_comment)

    result = {"brief": "", "params": [], "return": "", "note": "", "deprecated": ""}

    # Split on @ tags at the start of a line (may be multiline)
    # We'll process line by line
    lines = text.splitlines()

    current_tag = "brief"
    current_content: list[str] = []

    def _flush(tag, content):
        val = " ".join(content).strip()
        if not val:
            return
        if tag == "brief":
            result["brief"] += (" " if result["brief"] else "") + val
        elif tag in ("param[in]", "param[out]", "param[in,out]", "param[in, out]",
                     "param[out, in]"):
            # Already stored as structured entry — handled below
            pass
        elif tag == "return":
            result["return"] += (" " if result["return"] else "") + val
        elif tag == "note":
            result["note"] += (" " if result["note"] else "") + val
        elif tag in ("deprecated", "note_deprecated"):
            result["deprecated"] += (" " if result["deprecated"] else "") + val

    param_buf: dict | None = None

    def _flush_param():
        nonlocal param_buf
        if param_buf:
            param_buf["desc"] = " ".join(param_buf["_lines"]).strip()
            del param_buf["_lines"]
            result["params"].append(param_buf)
            param_buf = None

    for line in lines:
        # Match @param[dir] name desc
        m = re.match(r"@param(\[[^\]]*\])?\s+(\S+)\s*(.*)", line)
        if m:
            _flush(current_tag, current_content)
            _flush_param()
            current_content = []
            direction = (m.group(1) or "").strip("[]").strip()
            param_buf = {"name": m.group(2), "dir": direction, "_lines": [m.group(3).strip()]}
            current_tag = "__param__"
            continue

        # Match @return, @brief, @note, @verbatim, @deprecated
        m2 = re.match(r"@(return|brief|note|verbatim|deprecated|attention)(.*)", line)
        if m2:
            if current_tag == "__param__":
                _flush_param()
            else:
                _flush(current_tag, current_content)
            current_content = []
            tag_name = m2.group(1)
            rest = m2.group(2).strip()
            if tag_name in ("verbatim", "attention"):
                tag_name = "note"
            current_tag = tag_name
            if rest:
                current_content.append(rest)
            continue

        # endverbatim / endcode
        if re.match(r"@end(verbatim|code)", line):
            continue

        # Continuation
        if current_tag == "__param__" and param_buf is not None:
            param_buf["_lines"].append(line.strip())
        else:
            current_content.append(line.strip())

    # Flush last
    if current_tag == "__param__":
        _flush_param()
    else:
        _flush(current_tag, current_content)

    return result


def parse_headers() -> dict:
    """
    Parse both header files and return a dict keyed by symbol name:
        {
          "CUOPT_SUCCESS":          {"kind": "define", "value": "0", "brief": "..."},
          "cuopt_float_t":          {"kind": "typedef", "underlying": "double", "brief": "..."},
          "cuOptOptimizationProblem": {"kind": "typedef", "underlying": "void*", "brief": "..."},
          "cuOptMIPGetSolutionCallback": {"kind": "typedef_fn", "brief": "...", "params": [...], "return": "..."},
          "cuOptSolve":             {"kind": "function", "ret": "cuopt_int_t",
                                     "params": [{"name":..., "type":...}],
                                     "brief": "...", "param_docs": [...], "return_doc": "..."},
        }
    """
    symbols: dict = {}

    for hpath in [CONSTANTS, HEADER]:
        text = hpath.read_text(encoding="utf-8")
        _parse_file(text, symbols)

    return symbols


def _parse_file(text: str, symbols: dict):
    """Mutate symbols dict by parsing text."""

    # We scan token by token: look for a comment block immediately preceding
    # a #define, typedef, or function declaration.

    # Normalize line endings
    lines = text.splitlines()
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]

        # ----------------------------------------------------------------
        # #define constants
        # ----------------------------------------------------------------
        m = re.match(r'^#define\s+(\w+)\s+(.*)', line)
        if m:
            name = m.group(1)
            value = m.group(2).strip()
            # Skip internal/instantiation guards
            if name.startswith("CUOPT_INSTANTIATE_") or name in (
                "CUOPT_C_API_H", "CUOPT_CONSTANTS_H"
            ):
                i += 1
                continue
            # Handle line continuation
            while value.endswith("\\") and i + 1 < n:
                i += 1
                value = value.rstrip("\\").strip() + " " + lines[i].strip()
            value = value.strip()

            # Look back for a per-symbol comment.
            # Only use /** ... */ (double-star) comments for defines;
            # single /* ... */ comments are group/section labels, not per-symbol docs.
            comment = _lookback_comment(lines, i - 1 if i > 0 else 0)
            brief = ""
            if comment and comment.strip().startswith("/**"):
                block = _parse_doxygen_block(comment)
                brief = block["brief"]
            # For string defines the value includes quotes
            symbols[name] = {"kind": "define", "value": value, "brief": brief}
            i += 1
            continue

        # ----------------------------------------------------------------
        # typedef void* Name;  (simple opaque handle)
        # ----------------------------------------------------------------
        m = re.match(r'^typedef\s+(.*?)\s+(\w+)\s*;', line)
        if m:
            underlying = m.group(1).strip()
            name = m.group(2)
            comment = _lookback_comment(lines, i - 1)
            brief = ""
            if comment:
                block = _parse_doxygen_block(comment)
                brief = block["brief"]
            symbols[name] = {"kind": "typedef", "underlying": underlying, "brief": brief}
            i += 1
            continue

        # typedef void (*Name)(...) — function pointer typedef (may span lines)
        m = re.match(r'^typedef\s+\w[\w\s\*]*\s+\(\s*\*\s*(\w+)\s*\)', line)
        if m:
            name = m.group(1)
            # Collect the whole declaration until ;
            decl_lines = [line]
            j = i
            while ";" not in line and j + 1 < n:
                j += 1
                line = lines[j]
                decl_lines.append(line)
            decl = " ".join(decl_lines)
            comment = _lookback_comment(lines, i - 1)
            brief = ""
            param_docs = []
            ret_doc = ""
            note = ""
            if comment:
                block = _parse_doxygen_block(comment)
                brief = block["brief"]
                param_docs = block["params"]
                ret_doc = block["return"]
                note = block["note"]
            symbols[name] = {
                "kind": "typedef_fn",
                "brief": brief,
                "param_docs": param_docs,
                "return_doc": ret_doc,
                "note": note,
            }
            i = j + 1
            continue

        # ----------------------------------------------------------------
        # Function declarations (may span multiple lines, end with ;)
        # ----------------------------------------------------------------
        # Start: return_type cuOptFunctionName( ...
        m = re.match(r'^([\w\s\*]+?)\s+(cuOpt\w+)\s*\(', line)
        if m:
            ret_type = m.group(1).strip()
            func_name = m.group(2)
            # Collect until closing );
            decl_lines = [line]
            j = i
            while ");" not in lines[j] and j + 1 < n:
                j += 1
                decl_lines.append(lines[j])
            full_decl = " ".join(l.strip() for l in decl_lines)

            # Parse parameters from declaration
            param_match = re.search(r'\((.*)\)', full_decl, re.DOTALL)
            raw_params = param_match.group(1).strip() if param_match else ""
            decl_params = _parse_decl_params(raw_params)

            comment = _lookback_comment(lines, i - 1)
            brief = ""
            param_docs = []
            ret_doc = ""
            note = ""
            deprecated = ""
            if comment:
                block = _parse_doxygen_block(comment)
                brief = block["brief"]
                param_docs = block["params"]
                ret_doc = block["return"]
                note = block["note"]
                deprecated = block["deprecated"]

            symbols[func_name] = {
                "kind": "function",
                "ret": ret_type,
                "decl_params": decl_params,
                "brief": brief,
                "param_docs": param_docs,
                "return_doc": ret_doc,
                "note": note,
                "deprecated": deprecated,
            }
            i = j + 1
            continue

        i += 1


def _lookback_comment(lines: list[str], start: int) -> str | None:
    """
    Search backwards from `start` for a /** or /* comment block.
    Returns the raw comment string or None.
    """
    i = start
    # Skip blank lines
    while i >= 0 and lines[i].strip() == "":
        i -= 1
    if i < 0:
        return None
    # Check for end of block comment */
    if lines[i].strip().endswith("*/"):
        end = i
        # Walk back to find /*
        while i >= 0 and not re.search(r'/\*', lines[i]):
            i -= 1
        if i < 0:
            return None
        return "\n".join(lines[i:end + 1])
    # Check for single-line /** @brief ... */
    m = re.match(r'\s*/\*\*.*\*/', lines[i])
    if m:
        return lines[i].strip()
    return None


def _parse_decl_params(raw: str) -> list[dict]:
    """
    Parse 'type1 name1, const type2* name2, ...' into
    [{"type": "type1", "name": "name1"}, ...]
    """
    if not raw or raw.strip() in ("void", ""):
        return []
    params = []
    # Split by commas not inside <>
    parts = re.split(r',\s*', raw)
    for part in parts:
        part = part.strip()
        if not part:
            continue
        # Last word is the name (possibly with *)
        tokens = part.rsplit(None, 1)
        if len(tokens) == 2:
            ptype = tokens[0].strip()
            pname = tokens[1].lstrip("*").strip()
            # absorb pointer into type
            if "*" in tokens[1]:
                ptype = ptype + "*"
                pname = tokens[1].replace("*", "").strip()
            params.append({"type": ptype, "name": pname})
        else:
            params.append({"type": part, "name": ""})
    return params


# ---------------------------------------------------------------------------
# MDX renderers
# ---------------------------------------------------------------------------

def _escape_mdx(text: str) -> str:
    """Escape characters that MDX/JSX can't handle in prose."""
    # <= → &lt;=
    text = re.sub(r'<=', '&lt;=', text)
    # Curly braces
    text = text.replace("{", "(").replace("}", ")")
    return text


def _clean_desc(text: str) -> str:
    """Post-process a doxygen description string for MDX output."""
    text = text.strip()
    # Strip leading "- " that doxygen authors sometimes add after the param name
    text = re.sub(r'^-\s+', '', text)
    # Convert double-backtick ``foo`` to single-backtick `foo`
    text = re.sub(r'``([^`]+)``', r'`\1`', text)
    return text


def _render_typedef(name: str, info: dict) -> str:
    brief = _escape_mdx(_clean_desc(info.get("brief", "") or ""))
    underlying = info.get("underlying", "")
    # Horizontal rule + "typedef" label for visual scanning.
    if underlying:
        return f"<hr />\n\n**`typedef`** **`{name}`** — {brief} (`typedef {underlying}`)\n"
    return f"<hr />\n\n**`typedef`** **`{name}`** — {brief}\n"


def _render_typedef_fn(name: str, info: dict) -> str:
    lines = []
    brief = _escape_mdx(_clean_desc(info.get("brief", "") or ""))
    lines.append(f"<hr />\n\n**`typedef`** **`{name}`** — {brief}\n")
    note = info.get("note", "")
    if note:
        lines.append(f"\n<Note>\n{_escape_mdx(_clean_desc(note))}\n</Note>\n")
    param_docs = info.get("param_docs", [])
    if param_docs:
        lines.append("\n**Parameters**\n")
        for p in param_docs:
            desc = _escape_mdx(_clean_desc(p.get("desc", "") or ""))
            direction = p.get("dir", "")
            dir_str = f" `[{direction}]`" if direction else ""
            lines.append(f"- **`{p['name']}`**{dir_str} — {desc}")
        lines.append("")
    ret = info.get("return_doc", "")
    if ret:
        lines.append(f"\n**Returns** {_escape_mdx(_clean_desc(ret))}\n")
    return "\n".join(lines)


def _render_define(name: str, info: dict) -> str:
    value = info.get("value", "")
    brief = _escape_mdx(_clean_desc(info.get("brief", "") or ""))
    # String values already include quotes; numeric are bare
    if brief:
        return f"- `{name}` (`{value}`) — {brief}"
    return f"- `{name}` (`{value}`)"


def _render_function(name: str, info: dict) -> str:
    lines = []

    # Build compact one-line signature
    ret = info.get("ret", "")
    decl_params = info.get("decl_params", [])
    if decl_params:
        param_str = ", ".join(
            f"{p['type']} {p['name']}".strip() for p in decl_params
        )
    else:
        param_str = "void"

    sig = f"{name}({param_str})"
    if ret and ret != "void":
        sig = f"{sig} -> {ret}"

    lines.append(f"<hr />\n\n#### `{sig}`\n")

    brief = _escape_mdx(_clean_desc(info.get("brief", "") or ""))
    if brief:
        lines.append(f"{brief}\n")

    deprecated = info.get("deprecated", "")
    if deprecated:
        lines.append(f"<Warning>\n{_escape_mdx(_clean_desc(deprecated))}\n</Warning>\n")

    note = info.get("note", "")
    if note:
        lines.append(f"<Note>\n{_escape_mdx(_clean_desc(note))}\n</Note>\n")

    param_docs = info.get("param_docs", [])
    if param_docs:
        lines.append("**Parameters**\n")
        for p in param_docs:
            pname = p.get("name", "")
            direction = p.get("dir", "")
            desc = _escape_mdx(_clean_desc(p.get("desc", "") or ""))
            dir_str = f" `[{direction}]`" if direction else ""
            # Find matching type from decl
            ptype = ""
            for dp in decl_params:
                if dp["name"] == pname:
                    ptype = dp["type"]
                    break
            if ptype:
                lines.append(f"- **`{pname}`** (`{ptype}`){dir_str} — {desc}")
            else:
                lines.append(f"- **`{pname}`**{dir_str} — {desc}")
        lines.append("")

    ret_doc = info.get("return_doc", "")
    if ret_doc:
        lines.append(f"**Returns** {_escape_mdx(_clean_desc(ret_doc))}\n")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# RST structure parser → ordered symbol list
# ---------------------------------------------------------------------------

def _parse_rst_directives(rst_path: Path) -> list[tuple[str, str]]:
    """
    Return ordered list of (directive_type, symbol_name) from an RST file.
    directive_type is one of: "function", "typedef", "define"
    """
    text = rst_path.read_text(encoding="utf-8")
    result = []
    for m in re.finditer(
        r"\.\.\s+doxygen(function|typedef|define)::\s*(\w+)", text
    ):
        kind_map = {"function": "function", "typedef": "typedef", "define": "define"}
        result.append((kind_map[m.group(1)], m.group(2)))
    return result


# ---------------------------------------------------------------------------
# MDX page generator: re-generate from RST structure + parsed headers
# ---------------------------------------------------------------------------

def _render_symbol(kind: str, name: str, symbols: dict) -> str:
    """Render a single symbol to MDX text."""
    info = symbols.get(name)
    if not info:
        return f"*`{name}` — documentation not found in headers.*\n"

    actual_kind = info.get("kind", "")
    if kind == "function":
        if actual_kind == "function":
            return _render_function(name, info)
        if actual_kind in ("typedef_fn",):
            return _render_typedef_fn(name, info)
    if kind == "typedef":
        if actual_kind == "typedef":
            return _render_typedef(name, info)
        if actual_kind == "typedef_fn":
            return _render_typedef_fn(name, info)
    if kind == "define":
        if actual_kind == "define":
            return _render_define(name, info)

    # Fallback
    return f"*`{name}` — (see header)*\n"


def _generate_mdx_from_rst(rst_path: Path, symbols: dict, title: str) -> str:
    """
    Re-build an MDX page by walking the RST line by line.
    Prose text is preserved; doxygen directives are replaced with rendered MDX.
    """
    rst = rst_path.read_text(encoding="utf-8")
    rst_lines = rst.splitlines()

    out_lines: list[str] = [
        f'---',
        f'title: "{title}"',
        f'---',
        f'',
        f'# {title}',
        f'',
    ]

    # Track groups of defines to render as a list
    pending_defines: list[str] = []

    def flush_defines():
        if pending_defines:
            for dname in pending_defines:
                info = symbols.get(dname)
                if info and info.get("kind") == "define":
                    out_lines.append(_render_define(dname, info))
                else:
                    out_lines.append(f"- `{dname}` — (not found)")
            out_lines.append("")
            pending_defines.clear()

    # RST section underline chars
    underline_chars = set("=-~^#*+")

    i = 0
    n = len(rst_lines)
    skip_title = True  # skip the document title (first heading)

    while i < n:
        line = rst_lines[i]

        # ---- Section headings ----
        if (
            i + 1 < n
            and line.strip()
            and not line.startswith(".")
            and len(set(rst_lines[i + 1].strip())) == 1
            and rst_lines[i + 1].strip()[0] in underline_chars
            and len(rst_lines[i + 1].strip()) >= len(line.strip())
        ):
            if skip_title:
                skip_title = False
                i += 2
                # skip blank line after heading
                while i < n and not rst_lines[i].strip():
                    i += 1
                continue
            flush_defines()
            heading = line.strip()
            # RST heading level → MD level (we use ## for top sections)
            char = rst_lines[i + 1].strip()[0]
            level_map = {"-": "##", "~": "###", "^": "####"}
            hashes = level_map.get(char, "##")
            out_lines.append(f"{hashes} {heading}")
            out_lines.append("")
            i += 2
            continue

        # ---- RST label anchors like .. _foo: ----
        if re.match(r'\.\. _[\w\-]+:', line):
            i += 1
            continue

        # ---- doxygen directives ----
        m = re.match(r'\.\.\s+doxygen(function|typedef|define)::\s*(\w+)', line)
        if m:
            kind = m.group(1)
            sym = m.group(2)
            # Skip option lines (indented)
            j = i + 1
            while j < n and rst_lines[j].startswith("   "):
                j += 1
            i = j

            if kind == "define":
                pending_defines.append(sym)
            else:
                flush_defines()
                rendered = _render_symbol(kind, sym, symbols)
                out_lines.append(rendered)
            continue

        # ---- .. note:: ----
        if line.strip().startswith(".. note::"):
            flush_defines()
            # Collect note body
            note_lines = []
            j = i + 1
            # blank line then indented content
            while j < n and (not rst_lines[j].strip() or rst_lines[j].startswith("   ")):
                note_lines.append(rst_lines[j].lstrip("   ").rstrip())
                j += 1
            note_body = "\n".join(note_lines).strip()
            note_body = _escape_mdx(note_body)
            # Convert RST inline code ``foo`` → `foo`
            note_body = re.sub(r'``([^`]+)``', r'`\1`', note_body)
            # Convert ``**bold**``
            out_lines.append(f"<Note>")
            out_lines.append(note_body)
            out_lines.append(f"</Note>")
            out_lines.append("")
            i = j
            continue

        # ---- RST comment lines (.. comment) not doxygen ----
        if re.match(r'\.\.\s+[A-Za-z]', line) and not re.match(r'\.\.\s+doxygen', line):
            # Skip directive and its indented body
            j = i + 1
            while j < n and (rst_lines[j].startswith("   ") or not rst_lines[j].strip()):
                j += 1
            i = j
            continue

        # ---- Blank lines ----
        if not line.strip():
            if pending_defines:
                # don't flush on single blank line between defines
                # but do if there's prose ahead
                if i + 1 < n and not re.match(r'\.\.\s+doxygen(define)::', rst_lines[i + 1]):
                    flush_defines()
            else:
                out_lines.append("")
            i += 1
            continue

        # ---- Plain prose ----
        flush_defines()
        prose = line.rstrip()
        # Convert RST inline markup
        # ``code`` → `code`
        prose = re.sub(r'``([^`]+)``', r'`\1`', prose)
        # `text <ref>`_ → [text](ref)  — RST hyperlinks
        prose = re.sub(r'`([^`<]+)\s*<([^>]+)>`_', r'[\1](\2)', prose)
        # :ref:`label <anchor>` → just the label (within-page anchor)
        prose = re.sub(r':ref:`([^`<]+)\s*<([^>]+)>`', r'\1', prose)
        prose = re.sub(r':ref:`([^`]+)`', r'\1', prose)
        # :c:func:`name` → `name`
        prose = re.sub(r':c:func:`([^`]+)`', r'`\1`', prose)
        # :doc:`label <path>` → [label](../path)
        def _doc_link(m2):
            label = m2.group(1).strip()
            path = m2.group(2).strip()
            # Use just the basename for the URL
            slug = path.split("/")[-1]
            return f"[{label}](../{slug})"
        prose = re.sub(r':doc:`([^`<]+)\s*<([^>]+)>`', _doc_link, prose)
        def _doc_simple(m2):
            p = m2.group(1).strip().lstrip("/")
            slug = p.split("/")[-1]
            label = slug.replace("-", " ").replace("_", " ").title()
            return f"[{label}]({slug})"
        prose = re.sub(r':doc:`([^`]+)`', _doc_simple, prose)
        # `text` → `text` (already fine)
        prose = _escape_mdx(prose)
        # Fix list items (RST uses - or *)
        if re.match(r'^[-*]\s', prose):
            pass  # already list markdown
        out_lines.append(prose)
        i += 1

    flush_defines()

    # Collapse triple+ blank lines
    result = "\n".join(out_lines)
    result = re.sub(r'\n{3,}', '\n\n', result)
    return result.strip() + "\n"


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_pages():
    print("Extracting C API docs from headers...")
    symbols = parse_headers()
    print(f"  Parsed {len(symbols)} symbols")

    # Convex API page
    mdx_convex = _generate_mdx_from_rst(
        RST_CONVEX, symbols, "cuOpt Convex Optimization C API Reference"
    )
    OUT_CONVEX.parent.mkdir(parents=True, exist_ok=True)
    OUT_CONVEX.write_text(mdx_convex, encoding="utf-8")
    print(f"  Wrote {OUT_CONVEX.relative_to(REPO_ROOT)}")

    # MIP API page
    mdx_mip = _generate_mdx_from_rst(
        RST_MIP, symbols, "cuOpt MIP C API Reference"
    )
    OUT_MIP.parent.mkdir(parents=True, exist_ok=True)
    OUT_MIP.write_text(mdx_mip, encoding="utf-8")
    print(f"  Wrote {OUT_MIP.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    generate_pages()
