#!/usr/bin/env python3
"""
Parse cuopt_c.h and constants.h via Doxygen XML and regenerate MDX pages for
the C API reference.

Usage:
    python fern/extract_c_api.py

Regenerates:
    fern/docs/pages/cuopt-c/convex/convex-c-api.mdx
    fern/docs/pages/cuopt-c/mip/mip-c-api.mdx
"""

import re
import subprocess
import xml.etree.ElementTree as ET
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
# Doxygen XML parser — replaces the old regex-based header parser
# ---------------------------------------------------------------------------

_DOXYFILE = REPO_ROOT / "fern/Doxyfile"
_XML_DIR = REPO_ROOT / "fern/.doxygen-xml/xml"


def _xml_text(elem) -> str:
    """Recursively extract plain text from a Doxygen XML element."""
    if elem is None:
        return ""
    parts = []
    if elem.text:
        parts.append(elem.text)
    for child in elem:
        # <computeroutput> → backtick; everything else → plain text
        if child.tag == "computeroutput":
            inner = _xml_text(child)
            parts.append(f"`{inner}`")
        elif child.tag == "ref":
            parts.append(_xml_text(child))
        elif child.tag == "para":
            parts.append(_xml_text(child))
        else:
            parts.append(_xml_text(child))
        if child.tail:
            parts.append(child.tail)
    return "".join(parts).strip()


def _parse_memberdef(member) -> dict | None:
    """
    Parse a <memberdef> element from Doxygen XML into the same dict format
    that the old regex parser produced, keyed by symbol kind.
    """
    kind = member.get("kind", "")
    name_el = member.find("name")
    if name_el is None:
        return None
    name = name_el.text or ""

    # Brief description
    brief_el = member.find("briefdescription")
    brief = ""
    if brief_el is not None:
        para = brief_el.find("para")
        if para is not None:
            brief = _xml_text(para)

    # Detailed description (params, return, notes, deprecated)
    detail_el = member.find("detaileddescription")
    param_docs = []
    return_doc = ""
    note = ""
    deprecated = ""

    if detail_el is not None:
        for para in detail_el.iter("para"):
            # Parameters
            for plist in para.findall("parameterlist"):
                if plist.get("kind") == "param":
                    for item in plist.findall("parameteritem"):
                        pname_list = item.find("parameternamelist")
                        pdesc_el = item.find("parameterdescription")
                        if pname_list is None:
                            continue
                        pname_el = pname_list.find("parametername")
                        pname = pname_el.text or "" if pname_el is not None else ""
                        direction = pname_el.get("direction", "") if pname_el is not None else ""
                        pdesc = ""
                        if pdesc_el is not None:
                            inner = pdesc_el.find("para")
                            pdesc = _xml_text(inner) if inner is not None else _xml_text(pdesc_el)
                        # Strip leading "- " that Doxygen authors sometimes add
                        pdesc = re.sub(r"^-\s+", "", pdesc.strip())
                        param_docs.append({"name": pname, "dir": direction, "desc": pdesc})

            # Return value
            for ss in para.findall("simplesect"):
                if ss.get("kind") == "return":
                    inner = ss.find("para")
                    return_doc = _xml_text(inner) if inner is not None else _xml_text(ss)
                elif ss.get("kind") in ("note", "attention"):
                    inner = ss.find("para")
                    note_text = _xml_text(inner) if inner is not None else _xml_text(ss)
                    note = (note + " " + note_text).strip() if note else note_text
                elif ss.get("kind") == "warning":
                    inner = ss.find("para")
                    dep_text = _xml_text(inner) if inner is not None else _xml_text(ss)
                    deprecated = (deprecated + " " + dep_text).strip() if deprecated else dep_text

    if kind == "define":
        init_el = member.find("initializer")
        value = (init_el.text or "").strip() if init_el is not None else ""
        return {"kind": "define", "value": value, "brief": brief, "name": name}

    elif kind == "typedef":
        type_el = member.find("type")
        args_el = member.find("argsstring")
        underlying = _xml_text(type_el) if type_el is not None else ""
        args = args_el.text or "" if args_el is not None else ""
        # Function pointer typedef: argsstring contains the parameter list
        if args.strip().startswith(")(") or args.strip().startswith("*)("):
            return {
                "kind": "typedef_fn",
                "name": name,
                "brief": brief,
                "param_docs": param_docs,
                "return_doc": return_doc,
                "note": note,
            }
        # Simple typedef: underlying type  (strip backtick wrapping added by _xml_text)
        underlying = underlying.replace("`", "").strip()
        return {"kind": "typedef", "underlying": underlying, "brief": brief, "name": name}

    elif kind == "function":
        type_el = member.find("type")
        ret = _xml_text(type_el) if type_el is not None else ""
        ret = ret.replace("`", "").strip()
        # Collect declaration params
        decl_params = []
        for param in member.findall("param"):
            ptype_el = param.find("type")
            pname_el = param.find("declname")
            ptype = _xml_text(ptype_el) if ptype_el is not None else ""
            ptype = ptype.replace("`", "").strip()
            pname = pname_el.text or "" if pname_el is not None else ""
            decl_params.append({"type": ptype, "name": pname})
        return {
            "kind": "function",
            "name": name,
            "ret": ret,
            "decl_params": decl_params,
            "brief": brief,
            "param_docs": param_docs,
            "return_doc": return_doc,
            "note": note,
            "deprecated": deprecated,
        }

    return None


def parse_headers() -> dict:
    """
    Run Doxygen on the two C headers, parse the XML output, and return a dict
    keyed by symbol name in the same format the old regex parser produced:

        {
          "CUOPT_SUCCESS":               {"kind": "define",    "value": "0",  "brief": "..."},
          "cuopt_float_t":               {"kind": "typedef",   "underlying":  "double", "brief": "..."},
          "cuOptOptimizationProblem":    {"kind": "typedef",   "underlying":  "void *", "brief": "..."},
          "cuOptMIPGetSolutionCallback": {"kind": "typedef_fn","brief": "...", "params": [...], ...},
          "cuOptSolve":                  {"kind": "function",  "ret": "cuopt_int_t", ...},
        }
    """
    # Run doxygen to produce XML
    doxyfile = _DOXYFILE
    if not doxyfile.exists():
        raise FileNotFoundError(f"Doxyfile not found: {doxyfile}")

    result = subprocess.run(
        ["doxygen", str(doxyfile)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"doxygen failed (exit {result.returncode}):\n{result.stderr}"
        )

    xml_dir = _XML_DIR
    if not xml_dir.exists():
        raise FileNotFoundError(f"Doxygen XML output dir not found: {xml_dir}")

    symbols: dict = {}

    # Parse each per-file XML (constants_8h.xml and cuopt__c_8h.xml)
    for xml_file in sorted(xml_dir.glob("*.xml")):
        if xml_file.name in ("index.xml", "combine.xslt"):
            continue
        if not (xml_file.name.startswith("constants_") or
                xml_file.name.startswith("cuopt__c_")):
            continue
        try:
            tree = ET.parse(xml_file)
        except ET.ParseError as e:
            print(f"  [WARN] XML parse error in {xml_file.name}: {e}")
            continue

        root = tree.getroot()
        for compound in root.findall(".//compounddef"):
            for section in compound.findall("sectiondef"):
                for member in section.findall("memberdef"):
                    info = _parse_memberdef(member)
                    if info is None:
                        continue
                    name = info.pop("name")
                    # Skip internal guards
                    if name.startswith("CUOPT_INSTANTIATE_") or name in (
                        "CUOPT_C_API_H", "CUOPT_CONSTANTS_H"
                    ):
                        continue
                    symbols[name] = info

    # Doxygen skips typedefs inside #if preprocessor blocks (like cuopt_int_t
    # and cuopt_float_t).  Supplement with a targeted regex pass on the header.
    _supplement_if_guarded_typedefs(symbols)

    return symbols


def _supplement_if_guarded_typedefs(symbols: dict):
    """
    Regex fallback for typedefs inside #if blocks that Doxygen skips.
    Reads the header, finds active (enabled) #if branches, and adds any
    missing typedef entries to `symbols`.
    """
    text = HEADER.read_text(encoding="utf-8")
    lines = text.splitlines()
    n = len(lines)

    # First pass: collect the instantiate macro values from constants.h
    consts_text = CONSTANTS.read_text(encoding="utf-8")
    macro_values: dict[str, int] = {}
    for m in re.finditer(r'^#define\s+(CUOPT_INSTANTIATE_\w+)\s+(\d+)', consts_text, re.M):
        macro_values[m.group(1)] = int(m.group(2))

    # State machine: walk lines, track active #if CUOPT_INSTANTIATE_* blocks
    in_active_block = False
    pending_comment: list[str] = []
    i = 0
    while i < n:
        line = lines[i]

        m_if = re.match(r'^#if\s+(CUOPT_INSTANTIATE_\w+)', line)
        m_endif = re.match(r'^#endif', line)

        if m_if:
            macro = m_if.group(1)
            val = macro_values.get(macro, 0)
            in_active_block = bool(val)
            pending_comment = []
            i += 1
            continue

        if m_endif:
            in_active_block = False
            pending_comment = []
            i += 1
            continue

        if in_active_block:
            # Accumulate Doxygen comment
            if line.strip().startswith("/**") or line.strip().startswith("*"):
                pending_comment.append(line)
            elif re.match(r'^typedef\s+(.*?)\s+(\w+)\s*;', line):
                m_td = re.match(r'^typedef\s+(.*?)\s+(\w+)\s*;', line)
                underlying = m_td.group(1).strip()
                name = m_td.group(2)
                if name not in symbols:
                    # Extract brief from pending comment
                    brief = ""
                    if pending_comment:
                        raw = "\n".join(pending_comment)
                        # Strip comment markers
                        cleaned = re.sub(r'/\*+', '', raw)
                        cleaned = re.sub(r'\*/', '', cleaned)
                        cleaned = re.sub(r'^\s*\*\s?', '', cleaned, flags=re.M)
                        brief_m = re.search(r'@brief\s+(.*)', cleaned, re.S)
                        if brief_m:
                            brief = re.sub(r'\s+', ' ', brief_m.group(1)).strip()
                        else:
                            brief = re.sub(r'\s+', ' ', cleaned).strip()
                    symbols[name] = {"kind": "typedef", "underlying": underlying, "brief": brief}
                pending_comment = []
            else:
                if not line.strip():
                    pass  # blank line — don't reset comment
                elif not line.strip().startswith("//"):
                    pending_comment = []

        i += 1


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


# Alias expected by generate_api_docs.py
main = generate_pages


if __name__ == "__main__":
    generate_pages()
