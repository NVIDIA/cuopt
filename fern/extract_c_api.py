#!/usr/bin/env python3
"""
Parse cuopt_c.h and constants.h via Doxygen XML and refresh the C API MDX
skeleton pages.

How it works
------------
Each MDX skeleton page (checked into git) contains placeholder markers:

    <!-- symbol: cuOptCreateProblem -->
    <!-- /symbol -->

On each run this script:
  1. Runs Doxygen to extract symbols from the C headers.
  2. For each skeleton file, finds every <!-- symbol: NAME --> marker and
     replaces the content between it and <!-- /symbol --> with freshly
     rendered Doxygen documentation.
  3. Errors if any symbol found in Doxygen XML has no marker in any skeleton.
     This forces developers to consciously place new symbols when they add them
     to the headers.

To add a new symbol to the docs:
  1. Add a <!-- symbol: NEW_SYMBOL --><!-- /symbol --> marker in the right
     location in the appropriate skeleton MDX file.
  2. Re-run the script (or build docs) — the content fills in automatically.

Usage:
    python fern/extract_c_api.py

Regenerates:
    fern/docs/pages/cuopt-c/convex/convex-c-api.mdx
    fern/docs/pages/cuopt-c/mip/mip-c-api.mdx
"""

import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
HEADER = REPO_ROOT / "cpp/include/cuopt/mathematical_optimization/cuopt_c.h"
CONSTANTS = REPO_ROOT / "cpp/include/cuopt/mathematical_optimization/constants.h"
PAGES = REPO_ROOT / "fern/docs/pages"

# MDX skeleton files — checked into git; script refreshes symbol blocks in-place.
SKELETON_PAGES = [
    PAGES / "cuopt-c/convex/convex-c-api.mdx",
    PAGES / "cuopt-c/mip/mip-c-api.mdx",
]

_DOXYFILE = REPO_ROOT / "fern/Doxyfile"
_XML_DIR = REPO_ROOT / "fern/.doxygen-xml/xml"

# Symbols to always suppress (internal guards, macros not for users)
_INTERNAL_SYMBOLS = {
    "CUOPT_C_API_H",
    "CUOPT_CONSTANTS_H",
}
_INTERNAL_PREFIXES = ("CUOPT_INSTANTIATE_",)

# Marker regexes
_OPEN_RE = re.compile(r'^<!--\s*symbol:\s*(\S+)\s*-->$')
_CLOSE = "<!-- /symbol -->"


# ---------------------------------------------------------------------------
# Doxygen XML parser
# ---------------------------------------------------------------------------

def _xml_text(elem) -> str:
    if elem is None:
        return ""
    parts = []
    if elem.text:
        parts.append(elem.text)
    for child in elem:
        if child.tag == "computeroutput":
            parts.append(f"`{_xml_text(child)}`")
        else:
            parts.append(_xml_text(child))
        if child.tail:
            parts.append(child.tail)
    return "".join(parts).strip()


def _parse_memberdef(member) -> dict | None:
    kind = member.get("kind", "")
    name_el = member.find("name")
    if name_el is None:
        return None
    name = name_el.text or ""

    brief_el = member.find("briefdescription")
    brief = ""
    if brief_el is not None:
        para = brief_el.find("para")
        if para is not None:
            brief = _xml_text(para)

    detail_el = member.find("detaileddescription")
    param_docs = []
    return_doc = ""
    note = ""
    deprecated = ""

    if detail_el is not None:
        for para in detail_el.iter("para"):
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
                        pdesc = re.sub(r"^-\s+", "", pdesc.strip())
                        param_docs.append({"name": pname, "dir": direction, "desc": pdesc})

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
        if args.strip().startswith(")(") or args.strip().startswith("*)("):
            return {
                "kind": "typedef_fn",
                "name": name,
                "brief": brief,
                "param_docs": param_docs,
                "return_doc": return_doc,
                "note": note,
            }
        underlying = underlying.replace("`", "").strip()
        return {"kind": "typedef", "underlying": underlying, "brief": brief, "name": name}

    elif kind == "function":
        type_el = member.find("type")
        ret = _xml_text(type_el) if type_el is not None else ""
        ret = ret.replace("`", "").strip()
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
    """Run Doxygen and return all parsed symbols keyed by name."""
    if not _DOXYFILE.exists():
        raise FileNotFoundError(f"Doxyfile not found: {_DOXYFILE}")

    result = subprocess.run(
        ["doxygen", str(_DOXYFILE)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"doxygen failed (exit {result.returncode}):\n{result.stderr}")

    if not _XML_DIR.exists():
        raise FileNotFoundError(f"Doxygen XML output dir not found: {_XML_DIR}")

    symbols: dict = {}
    for xml_file in sorted(_XML_DIR.glob("*.xml")):
        if xml_file.name in ("index.xml", "combine.xslt"):
            continue
        if not (xml_file.name.startswith("constants_") or xml_file.name.startswith("cuopt__c_")):
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
                    if name in _INTERNAL_SYMBOLS:
                        continue
                    if any(name.startswith(p) for p in _INTERNAL_PREFIXES):
                        continue
                    symbols[name] = info

    _supplement_if_guarded_typedefs(symbols)
    return symbols


def _supplement_if_guarded_typedefs(symbols: dict):
    """Regex fallback for typedefs inside #if blocks that Doxygen skips."""
    text = HEADER.read_text(encoding="utf-8")
    lines = text.splitlines()
    n = len(lines)

    consts_text = CONSTANTS.read_text(encoding="utf-8")
    macro_values: dict[str, int] = {}
    for m in re.finditer(r'^#define\s+(CUOPT_INSTANTIATE_\w+)\s+(\d+)', consts_text, re.M):
        macro_values[m.group(1)] = int(m.group(2))

    in_active_block = False
    pending_comment: list[str] = []
    i = 0
    while i < n:
        line = lines[i]
        m_if = re.match(r'^#if\s+(CUOPT_INSTANTIATE_\w+)', line)
        m_endif = re.match(r'^#endif', line)

        if m_if:
            macro = m_if.group(1)
            in_active_block = bool(macro_values.get(macro, 0))
            pending_comment = []
            i += 1
            continue

        if m_endif:
            in_active_block = False
            pending_comment = []
            i += 1
            continue

        if in_active_block:
            if line.strip().startswith("/**") or line.strip().startswith("*"):
                pending_comment.append(line)
            elif re.match(r'^typedef\s+(.*?)\s+(\w+)\s*;', line):
                m_td = re.match(r'^typedef\s+(.*?)\s+(\w+)\s*;', line)
                underlying = m_td.group(1).strip()
                name = m_td.group(2)
                if name not in symbols:
                    brief = ""
                    if pending_comment:
                        raw = "\n".join(pending_comment)
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
                if line.strip() and not line.strip().startswith("//"):
                    pending_comment = []

        i += 1


# ---------------------------------------------------------------------------
# MDX renderers
# ---------------------------------------------------------------------------

def _escape_mdx(text: str) -> str:
    text = re.sub(r'<=', '&lt;=', text)
    text = text.replace("{", "(").replace("}", ")")
    return text


def _clean_desc(text: str) -> str:
    text = text.strip()
    text = re.sub(r'^-\s+', '', text)
    text = re.sub(r'``([^`]+)``', r'`\1`', text)
    return text


def _render_typedef(name: str, info: dict) -> str:
    brief = _escape_mdx(_clean_desc(info.get("brief", "") or ""))
    underlying = info.get("underlying", "")
    if underlying:
        return f"**`typedef`** **`{name}`** — {brief} (`typedef {underlying}`)"
    return f"**`typedef`** **`{name}`** — {brief}"


def _render_typedef_fn(name: str, info: dict) -> str:
    lines = []
    brief = _escape_mdx(_clean_desc(info.get("brief", "") or ""))
    lines.append(f"**`typedef`** **`{name}`** — {brief}")
    note = info.get("note", "")
    if note:
        lines.append(f"\n<Note>\n{_escape_mdx(_clean_desc(note))}\n</Note>")
    param_docs = info.get("param_docs", [])
    if param_docs:
        lines.append("\n**Parameters**\n")
        for p in param_docs:
            desc = _escape_mdx(_clean_desc(p.get("desc", "") or ""))
            direction = p.get("dir", "")
            dir_str = f" `[{direction}]`" if direction else ""
            lines.append(f"- **`{p['name']}`**{dir_str} — {desc}")
    ret = info.get("return_doc", "")
    if ret:
        lines.append(f"\n**Returns** {_escape_mdx(_clean_desc(ret))}")
    return "\n".join(lines)


def _render_define(name: str, info: dict) -> str:
    value = info.get("value", "")
    brief = _escape_mdx(_clean_desc(info.get("brief", "") or ""))
    if brief:
        return f"- `{name}` (`{value}`) — {brief}"
    return f"- `{name}` (`{value}`)"


def _render_function(name: str, info: dict) -> str:
    lines = []
    ret = info.get("ret", "")
    decl_params = info.get("decl_params", [])
    param_str = ", ".join(f"{p['type']} {p['name']}".strip() for p in decl_params) or "void"
    sig = f"{name}({param_str})"
    if ret and ret != "void":
        sig = f"{sig} -> {ret}"
    lines.append(f"#### `{sig}`\n")

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
            ptype = next((dp["type"] for dp in decl_params if dp["name"] == pname), "")
            if ptype:
                lines.append(f"- **`{pname}`** (`{ptype}`){dir_str} — {desc}")
            else:
                lines.append(f"- **`{pname}`**{dir_str} — {desc}")
        lines.append("")

    ret_doc = info.get("return_doc", "")
    if ret_doc:
        lines.append(f"**Returns** {_escape_mdx(_clean_desc(ret_doc))}\n")

    return "\n".join(lines)


def _render_symbol(name: str, symbols: dict) -> str:
    info = symbols.get(name)
    if not info:
        return f"*`{name}` — documentation not found in headers.*"
    kind = info.get("kind", "")
    if kind == "function":
        return _render_function(name, info)
    if kind == "typedef_fn":
        return _render_typedef_fn(name, info)
    if kind == "typedef":
        return _render_typedef(name, info)
    if kind == "define":
        return _render_define(name, info)
    return f"*`{name}` — (see header)*"


# ---------------------------------------------------------------------------
# Skeleton refresher
# ---------------------------------------------------------------------------

def _collect_markers(mdx_path: Path) -> set:
    """Return the set of symbol names with markers in this skeleton file."""
    found = set()
    for line in mdx_path.read_text(encoding="utf-8").splitlines():
        m = _OPEN_RE.match(line.strip())
        if m:
            found.add(m.group(1))
    return found


def _refresh_skeleton(mdx_path: Path, symbols: dict) -> bool:
    """
    Replace content between <!-- symbol: NAME --> and <!-- /symbol --> markers
    with freshly rendered Doxygen output. Returns True if the file changed.
    """
    original = mdx_path.read_text(encoding="utf-8")
    lines = original.splitlines(keepends=True)
    out = []
    i = 0
    while i < len(lines):
        line = lines[i].rstrip("\n")
        m = _OPEN_RE.match(line.strip())
        if m:
            name = m.group(1)
            out.append(lines[i])  # keep the opening marker
            i += 1
            # skip existing content up to closing marker
            while i < len(lines) and lines[i].rstrip("\n").strip() != _CLOSE:
                i += 1
            # render fresh content
            rendered = _render_symbol(name, symbols).rstrip()
            if rendered:
                out.append(rendered + "\n")
            # emit closing marker
            out.append(_CLOSE + "\n")
            i += 1  # skip the closing marker line we consumed
            continue
        out.append(lines[i])
        i += 1

    result = "".join(out)
    if result != original:
        mdx_path.write_text(result, encoding="utf-8")
        return True
    return False


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_pages():
    print("Extracting C API docs from headers (Doxygen XML)...")
    symbols = parse_headers()
    print(f"  Parsed {len(symbols)} symbols")

    # Collect every symbol name referenced in any skeleton
    all_placed: set = set()
    for skeleton in SKELETON_PAGES:
        if skeleton.exists():
            all_placed |= _collect_markers(skeleton)

    # Error on any Doxygen symbol that has no marker in any skeleton
    missing = sorted(
        name for name in symbols
        if name not in all_placed
        and name not in _INTERNAL_SYMBOLS
        and not any(name.startswith(p) for p in _INTERNAL_PREFIXES)
    )
    if missing:
        print(
            f"\nERROR: {len(missing)} symbol(s) found in headers but not placed in any MDX skeleton.\n"
            "Add a <!-- symbol: NAME --><!-- /symbol --> marker to the appropriate skeleton file:\n"
            + "\n".join(f"  - {n}" for n in missing),
            file=sys.stderr,
        )
        sys.exit(1)

    # Refresh each skeleton
    for skeleton in SKELETON_PAGES:
        if not skeleton.exists():
            print(f"  [SKIP] {skeleton.relative_to(REPO_ROOT)} (not found)")
            continue
        changed = _refresh_skeleton(skeleton, symbols)
        status = "updated" if changed else "unchanged"
        print(f"  {status}: {skeleton.relative_to(REPO_ROOT)}")


# Alias expected by generate_api_docs.py
main = generate_pages


if __name__ == "__main__":
    generate_pages()
