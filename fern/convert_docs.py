#!/usr/bin/env python3
"""
Convert cuOpt Sphinx/RST docs to Fern MDX format.

Usage:
    python fern/convert_docs.py

Outputs:
    fern/docs/pages/**/*.mdx  - converted pages
    fern/docs/images/         - copied images
    fern/docs.yml             - navigation + Fern config
"""

import os
import re
import subprocess
import shutil
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
SRC = REPO_ROOT / "docs/cuopt/source"
FERN = REPO_ROOT / "fern"
PAGES = FERN / "docs/pages"
IMAGES = FERN / "docs/images"


# ---------------------------------------------------------------------------
# RST pre-processor (runs before pandoc)
# ---------------------------------------------------------------------------

def _resolve_literalinclude(match, rst_path: Path) -> str:
    """Replace .. literalinclude:: with an RST code-block from the actual file."""
    filepath = match.group(1).strip()
    options_block = match.group(2) or ""

    lang = "text"
    start_after = None
    end_before = None
    for line in options_block.splitlines():
        m = re.match(r"\s+:language:\s+(.+)", line)
        if m:
            lang = m.group(1).strip()
        m = re.match(r"\s+:start-after:\s+(.+)", line)
        if m:
            start_after = m.group(1).strip()
        m = re.match(r"\s+:end-before:\s+(.+)", line)
        if m:
            end_before = m.group(1).strip()

    target = (rst_path.parent / filepath).resolve()
    if not target.exists():
        return f".. code-block:: {lang}\n\n   # File not found: {filepath}\n"

    code = target.read_text()
    if start_after:
        idx = code.find(start_after)
        if idx != -1:
            code = code[idx + len(start_after):]
    if end_before:
        idx = code.find(end_before)
        if idx != -1:
            code = code[:idx]

    indented = textwrap.indent(code.strip(), "   ")
    return f".. code-block:: {lang}\n\n{indented}\n"


def _preprocess_rst(content: str, rst_path: Path) -> str:
    """Replace Sphinx directives that pandoc cannot handle."""

    # 1. literalinclude → code-block
    content = re.sub(
        r"\.\. literalinclude::\s*(\S+)((?:\n[ \t]+:[a-z\-]+:.*)*)",
        lambda m: _resolve_literalinclude(m, rst_path),
        content,
    )

    # 2. install-selector → raw HTML div (JS widget is loaded as a custom script)
    def _install_selector_div(m):
        options = m.group(0)
        iface_m = re.search(r":default-iface:\s*(\w+)", options)
        default_iface = iface_m.group(1) if iface_m else ""
        attr = f' data-default-iface="{default_iface}"' if default_iface else ""
        return (
            ".. raw:: html\n\n"
            f'   <div id="cuopt-install-selector"{attr}></div>\n'
        )

    content = re.sub(
        r"\.\. install-selector::[^\n]*(?:\n[ \t]+[^\n]*)*",
        _install_selector_div,
        content,
    )

    # 3. swagger-plugin → pointer to the native Fern API explorer in the nav
    #    Consume the full directive block (directive line + indented option lines)
    #    so pandoc doesn't pick up the :id: option as stray list content.
    content = re.sub(
        r"\.\. swagger-plugin::[^\n]*(?:\n[ \t]+[^\n]*)*",
        (
            "The interactive REST API explorer is available in the "
            "**REST API Reference** section of the left navigation. "
            "It covers all cuOpt server endpoints for LP, MIP, and "
            "routing optimization problems.\n"
        ),
        content,
    )

    # 4. autodoc / autosummary directives → remove (Python API handled separately)
    content = re.sub(
        r"\.\. auto(?:module|class|function|method|attribute|summary)::[^\n]*\n(?:[ \t]+[^\n]*\n)*",
        "",
        content,
    )

    # 5. breathe (C API doxygen) directives → placeholder
    content = re.sub(
        r"\.\. doxygen(?:file|struct|class|function|namespace|group|enum|typedef|variable|define|union|page)::[^\n]*\n(?:[ \t]+[^\n]*\n)*",
        ".. note::\n\n   C API reference is generated from Doxygen and will be linked here once Fern C++ library support ships.\n",
        content,
    )

    # 6. toctree → remove (navigation built into docs.yml)
    # Match the directive line plus all following blank or indented lines
    # (RST toctrees have a blank line before the entry list)
    content = re.sub(
        r"\.\. toctree::[^\n]*\n(?:(?:[ \t]*)[^\n]*\n)*",
        "",
        content,
    )

    # 7. download role → plain link text
    content = re.sub(r":download:`([^`<]+)\s*<([^>]+)>`", r"`\1 <\2>`_", content)

    # 8. :doc: role → page slug link
    def doc_link(m):
        label = m.group(1).strip() if m.group(1) else None
        path = m.group(2).strip() if m.group(2) else m.group(1).strip()
        # strip leading / and .rst
        slug = re.sub(r"\.rst$", "", path.lstrip("/"))
        display = label if label else slug.split("/")[-1].replace("-", " ").title()
        return f"`{display} <{slug}>`_"

    content = re.sub(r":doc:`([^`<]+)\s*<([^>]+)>`", doc_link, content)
    def doc_link_simple(m):
        path = m.group(1).strip()
        slug = re.sub(r"\.rst$", "", path.lstrip("/"))
        display = slug.split("/")[-1].replace("-", " ").title()
        return f"`{display} <{slug}>`_"

    content = re.sub(r":doc:`([^`]+)`", doc_link_simple, content)

    # 9. :ref: role → anchor link
    content = re.sub(r":ref:`([^`<]+)\s*<([^>]+)>`", r"`\1 <#\2>`_", content)
    content = re.sub(r":ref:`([^`]+)`", r"`\1 <#\1>`_", content)

    # 10. :func: / :class: / :meth: / :attr: / :obj: → inline code
    content = re.sub(r":(?:func|class|meth|attr|obj|data|mod|exc|const):`([^`]+)`", r"``\1``", content)

    # 11. :abbr: → just the term
    content = re.sub(r":abbr:`([^`(]+)\s*\([^)]+\)`", r"\1", content)

    # 12. .. dropdown:: Title → <Accordion title="Title">...</Accordion>
    #     Must run before pandoc so content inside is properly converted.
    def _dropdown_to_accordion(m):
        title = m.group(1).strip().replace('"', '&quot;')
        body_raw = m.group(2)
        # Drop :class: / :open: option lines at the start of body
        body_lines = body_raw.splitlines()
        non_option_lines = []
        in_options = True
        for line in body_lines:
            if in_options and re.match(r"[ \t]+:[a-z\-]+:", line):
                continue
            in_options = False
            non_option_lines.append(line)
        body = "\n".join(non_option_lines)
        # Dedent to column 0 so pandoc sees unindented RST content.
        body = textwrap.dedent(body)
        return (
            f".. raw:: html\n\n   <Accordion title=\"{title}\">\n\n"
            + body.strip()
            + "\n\n.. raw:: html\n\n   </Accordion>\n\n"
        )

    content = re.sub(
        r"\.\. dropdown::[ \t]+([^\n]+)\n((?:(?:[ \t][^\n]*)?\n)*)",
        _dropdown_to_accordion,
        content,
    )

    return content


# ---------------------------------------------------------------------------
# MDX post-processor (runs after pandoc)
# ---------------------------------------------------------------------------

def _postprocess_mdx(md: str, title: str) -> str:
    """Clean up pandoc output and add Fern MDX components."""

    # 1. Strip the pandoc-generated H1 that will duplicate the frontmatter title,
    #    then prepend frontmatter.  Must strip BEFORE prepending so that `^` anchors
    #    at the real start of the pandoc output (not after the `---` block).
    md = re.sub(r'^# [^\n]+\n+', '', md, count=1)
    md = f"---\ntitle: \"{title}\"\n---\n\n" + md

    # 2. pandoc admonition formats → Fern components
    #    pandoc 3.x: > [!NOTE] blockquote style
    def admonition_block(m):
        kind = m.group(1).upper()
        body = re.sub(r"^> ?", "", m.group(2), flags=re.MULTILINE).strip()
        tag = {"NOTE": "Note", "WARNING": "Warning", "IMPORTANT": "Note", "TIP": "Tip"}.get(kind, "Note")
        return f"<{tag}>\n{body}\n</{tag}>"

    md = re.sub(
        r"> \[!(NOTE|WARNING|IMPORTANT|TIP)\]\n((?:>.*\n?)*)",
        admonition_block,
        md,
        flags=re.IGNORECASE,
    )

    # pandoc 2.9 / RST-native: <div class="note"> style
    def admonition_div(m):
        kind = m.group(1).strip().lower()
        body = m.group(2).strip()
        tag = {"note": "Note", "warning": "Warning", "important": "Note", "tip": "Tip"}.get(kind, "Note")
        return f"<{tag}>\n{body}\n</{tag}>"

    md = re.sub(
        r'<div class="(note|warning|important|tip)">\s*<div class="title">\s*\w+\s*</div>\s*(.*?)\s*</div>',
        admonition_div,
        md,
        flags=re.DOTALL | re.IGNORECASE,
    )

    # Remaining blockquote "> **Note**" style (some pandoc versions)
    def admonition_simple(m):
        kind = m.group(1)
        rest = re.sub(r"^> ?", "", m.group(2), flags=re.MULTILINE).strip()
        tag = {"Note": "Note", "Warning": "Warning", "Important": "Note"}.get(kind, "Note")
        return f"<{tag}>\n{rest}\n</{tag}>"

    md = re.sub(
        r"> \*\*(Note|Warning|Important)\*\*\n((?:>.*\n?)*)",
        admonition_simple,
        md,
    )

    # 3b-pre. Strip blockquote lines that are purely RST toctree remnants
    #   (e.g. "> routing-api.rst routing-examples.rst" from unconverted toctrees)
    md = re.sub(
        r"^>[ \t]+(?:[\w./\-]+\.rst[ \t]*)+\n",
        "",
        md,
        flags=re.MULTILINE,
    )
    # Also strip inline .rst links left in prose like "(api.rst)" → remove .rst extension
    md = re.sub(r"\(([^)]+)\.rst\)", r"(\1)", md)

    # 3. Convert any remaining <div class="dropdown"> blocks to <Accordion>
    #    These come from .. dropdown:: directives not caught by the preprocessor.
    def _div_dropdown_to_accordion(m):
        inner = m.group(1).strip()
        # First non-empty paragraph is the title
        parts = re.split(r"\n\n+", inner, maxsplit=1)
        title = parts[0].strip().replace('"', '&quot;')
        body = parts[1].strip() if len(parts) > 1 else ""
        return f'<Accordion title="{title}">\n\n{body}\n\n</Accordion>'

    md = re.sub(
        r'<div class="dropdown">\s*(.*?)\s*</div>',
        _div_dropdown_to_accordion,
        md,
        flags=re.DOTALL,
    )

    # 3c. MDX can't parse JSX components inside list items when the body/closing
    #     tag is at column 0. Un-indent the opening tag so the whole block sits
    #     between list items instead.
    md = re.sub(
        r"^ {4}<(Note|Warning|Tip|Accordion)",
        r"<\1",
        md,
        flags=re.MULTILINE,
    )

    # 3d. Inside fenced code blocks (including blockquote-prefixed ones), escape
    #     patterns that MDX's JSX parser would misread as JSX tags:
    #     - `</foo`  → `< /foo`  (shell redirects; space is valid in shell)
    #     - `<word>` → `&lt;word&gt;` (placeholder names like <ip>, <image>)
    #     Both must be escaped even inside `> ``` ` blockquote code fences.
    KNOWN_HTML = {'a','abbr','b','br','code','div','em','h1','h2','h3','h4','h5','h6',
                  'hr','i','img','li','ol','p','pre','span','strong','table','td','th',
                  'tr','ul','sup','sub','s','del','ins','blockquote','details','summary'}
    MDX_COMPONENTS = {'Note','Warning','Tip','Accordion','AccordionGroup','CodeBlock',
                      'Card','CardGroup','Tabs','Tab','Frame','Steps','Step',
                      'Callout','Info','Check','Cross','Info'}

    def _escape_code_block(m):
        block = m.group(0)
        # Escape </word → < /word (shell redirect)
        block = re.sub(r'</(?=[a-z/])', r'< /', block)
        # Escape <word> placeholder tags (not known HTML, not MDX components)
        def _escape_placeholder(pm):
            tag = pm.group(1)
            if tag.lower() in KNOWN_HTML or tag in MDX_COMPONENTS:
                return pm.group(0)
            return f'&lt;{tag}&gt;'
        block = re.sub(r'<([a-zA-Z][a-zA-Z0-9_-]*)>', _escape_placeholder, block)
        return block

    md = re.sub(
        r'(?m)^([ \t]*(?:>[ \t]*)*)```[^\n]*\n.*?^\1```',
        _escape_code_block,
        md,
        flags=re.DOTALL,
    )

    # 3e. Angle-bracket autolinks are valid Markdown but MDX parses them as JSX:
    #     - URL autolinks: <https://...> → [url](url)
    #     - Email autolinks: <user@domain> → [user@domain](mailto:user@domain)
    md = re.sub(r'<(https?://[^>]+)>', r'[\1](\1)', md)
    md = re.sub(r'<([a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,})>', r'[\1](mailto:\1)', md)

    # 3f. MDX parses `<` as a JSX tag start even in plain text.
    #     Escape `<=` (and pandoc's `\<=`) to `&lt;=` outside code spans.
    #     Curly braces were already converted to parentheses in the RST
    #     pre-processor (step 13), so no further `{...}` handling needed here.
    def _escape_lt_outside_code(text):
        segments = re.split(r'(`[^`\n]+`|```[\s\S]*?```)', text)
        result = []
        for i, seg in enumerate(segments):
            if i % 2 == 1:  # code span or block — leave unchanged
                result.append(seg)
            else:
                seg = re.sub(r'\\?<=', '&lt;=', seg)
                result.append(seg)
        return ''.join(result)
    md = _escape_lt_outside_code(md)

    # 3g. Escape bare { and } in prose so MDX doesn't parse them as JSX expressions.
    #     Only applies outside fenced code blocks and inline code spans.
    def _escape_braces_outside_code(text):
        # Split on fenced code blocks first, then inline code spans within prose
        fence_parts = re.split(r'(?ms)(^[ \t]*(?:>[ \t]*)*)```[^\n]*\n.*?^\1```', text)
        out = []
        for i, part in enumerate(fence_parts):
            if i % 2 == 1:  # fenced code block — leave unchanged
                out.append(part)
                continue
            # Within prose, split on inline code spans
            segs = re.split(r'(`[^`\n]+`)', part)
            escaped = []
            for j, seg in enumerate(segs):
                if j % 2 == 1:  # inline code span
                    escaped.append(seg)
                else:
                    seg = seg.replace('{', r'\{').replace('}', r'\}')
                    escaped.append(seg)
            out.append(''.join(escaped))
        return ''.join(out)
    md = _escape_braces_outside_code(md)

    # 3b. Fix image paths → point into /docs/images/ (flat)
    # Matches: images/foo.png, ../images/foo.png, ../../foo/images/foo.png
    md = re.sub(r"!\[([^\]]*)\]\((?:[^)]*\/)?images\/([^)]+)\)", r"![\1](/docs/images/\2)", md)

    # 4. Clean up RST label anchors that pandoc leaves as raw HTML
    md = re.sub(r"\[]{#[^}]+}", "", md)
    md = re.sub(r'\[\\]{#[^}]+}', "", md)

    # 5. Remove empty HTML comment artifacts
    md = re.sub(r"<!--.*?-->", "", md, flags=re.DOTALL)

    # 6. Collapse excessive blank lines
    md = re.sub(r"\n{3,}", "\n\n", md)

    return md.strip() + "\n"


# ---------------------------------------------------------------------------
# Title extractor from RST
# ---------------------------------------------------------------------------

def _extract_title(rst_content: str, fallback: str) -> str:
    """Return the first RST section title."""
    lines = rst_content.splitlines()
    underline_chars = set("=-~^#*+")
    for i, line in enumerate(lines):
        if (
            i + 1 < len(lines)
            and len(line.strip()) > 0
            and not line.startswith(".")
            and len(set(lines[i + 1].strip())) == 1
            and lines[i + 1].strip()[0] in underline_chars
            and len(lines[i + 1].strip()) >= len(line.strip())
        ):
            return line.strip()
        # overline + title + underline pattern
        if (
            i >= 1
            and len(set(line.strip())) == 1
            and line.strip()[0] in underline_chars
            and i + 1 < len(lines)
            and len(set(lines[i + 1].strip())) == 1
            and lines[i + 1].strip()[0] in underline_chars
        ):
            return lines[i - 1].strip() if i >= 1 and lines[i - 1].strip() else fallback
    return fallback


# ---------------------------------------------------------------------------
# Single file converter
# ---------------------------------------------------------------------------

def convert_rst(rst_path: Path, dest_mdx: Path):
    """Convert one RST file to MDX and write to dest_mdx."""
    content = rst_path.read_text(encoding="utf-8")
    fallback = dest_mdx.stem.replace("-", " ").title()
    title = _extract_title(content, fallback)

    preprocessed = _preprocess_rst(content, rst_path)

    result = subprocess.run(
        ["pandoc", "--from=rst", "--to=gfm", "--wrap=none"],
        input=preprocessed,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"  [WARN] pandoc error on {rst_path.name}: {result.stderr.strip()}")

    md = result.stdout
    mdx = _postprocess_mdx(md, title)

    dest_mdx.parent.mkdir(parents=True, exist_ok=True)
    dest_mdx.write_text(mdx, encoding="utf-8")


def convert_md(md_path: Path, dest_mdx: Path):
    """Copy an existing .md file and rename to .mdx, adding frontmatter."""
    content = md_path.read_text(encoding="utf-8")
    m = re.match(r"#\s+(.+)", content)
    title = m.group(1).strip() if m else dest_mdx.stem.replace("-", " ").title()
    content = _postprocess_mdx(content, title)
    dest_mdx.parent.mkdir(parents=True, exist_ok=True)
    dest_mdx.write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# Toctree parser → navigation tree
# ---------------------------------------------------------------------------

def _parse_toctree(rst_content: str, base_path: Path) -> list:
    """
    Parse all toctree directives in an RST file.
    Returns list of dicts: {"caption": str|None, "entries": [str, ...]}
    Each entry is a relative path (without .rst) relative to source root.
    """
    groups = []
    # Capture everything from .. toctree:: until a line starting with a
    # non-space/non-empty character (next top-level element).
    pattern = re.compile(
        r"\.\. toctree::([^\n]*)\n((?:(?:[ \t][^\n]*)?\n)*)",
        re.MULTILINE,
    )
    for m in pattern.finditer(rst_content):
        options_raw = m.group(2)
        caption = None
        entries = []
        for line in options_raw.splitlines():
            line = line.strip()
            if not line:
                continue
            cm = re.match(r":caption:\s*(.+)", line)
            if cm:
                caption = cm.group(1).strip()
                continue
            if line.startswith(":"):
                continue
            entries.append(line)
        if entries:
            groups.append({"caption": caption, "entries": entries})
    return groups


def _resolve_entry(entry: str, rst_dir: Path, src: Path) -> Path | None:
    """Resolve a toctree entry to an absolute normalized path (rst or md).

    We normalize (collapse ../) but do NOT follow symlinks — this lets
    relative_to(src) work for files like release-notes.md that are symlinks
    pointing outside the source tree.
    """
    for suffix in ("", ".rst", ".md", "/index.rst", "/index.md"):
        raw = rst_dir / (entry + suffix)
        # normpath collapses ../../ without following symlinks
        candidate = Path(os.path.normpath(raw))
        if candidate.exists() and candidate.suffix in (".rst", ".md"):
            return candidate
    return None


def _build_nav_tree(rst_path: Path, src: Path, visited: set | None = None) -> list:
    """
    Recursively build a navigation tree from toctree directives.
    Returns a list of navigation items for docs.yml.
    """
    if visited is None:
        visited = set()
    if rst_path in visited:
        return []
    visited.add(rst_path)

    content = rst_path.read_text(encoding="utf-8")
    groups = _parse_toctree(content, rst_path.parent)

    nav = []
    for group in groups:
        items = []
        for entry_raw in group["entries"]:
            entry_raw = entry_raw.strip()
            # strip display label: "Display Text <path>"
            dm = re.match(r".+<([^>]+)>", entry_raw)
            entry = dm.group(1).strip() if dm else entry_raw

            child_rst = _resolve_entry(entry, rst_path.parent, src)
            if child_rst is None:
                print(f"  [WARN] toctree entry not found: {entry} (from {rst_path.name})")
                continue

            rel = child_rst.relative_to(src)
            page_path = f"docs/pages/{rel.with_suffix('.mdx').as_posix()}"
            child_content = child_rst.read_text(encoding="utf-8")
            child_groups = _parse_toctree(child_content, child_rst.parent)
            child_title = _extract_title(child_content, child_rst.stem.replace("-", " ").title())

            if child_groups:
                # Section index: recurse, then use the child's sections directly
                sub_items = _build_nav_tree(child_rst, src, visited)
                # Include the index page itself only if it has meaningful prose
                body = re.sub(r"\.\. toctree::.*?(?=\n\S|\Z)", "", child_content, flags=re.DOTALL).strip()
                contents = []
                if len(body) > 100:
                    # Use "Overview" so the section header and the page inside it
                    # don't display the same name twice in the left nav.
                    contents.append({"page": "Overview", "path": page_path})
                contents.extend(sub_items)
                items.append({"section": child_title, "contents": contents})
            else:
                items.append({"page": child_title, "path": page_path})

        if group["caption"]:
            # Flatten: if caption has one section index child with same-ish name,
            # hoist its contents up to avoid double-wrapping.
            if len(items) == 1 and "section" in items[0]:
                inner = items[0]
                nav.append({"section": group["caption"], "contents": inner["contents"]})
            else:
                nav.append({"section": group["caption"], "contents": items})
        else:
            nav.extend(items)

    return nav


# ---------------------------------------------------------------------------
# Collect all RST files that need converting
# ---------------------------------------------------------------------------

def _collect_rst_files(src: Path) -> list[Path]:
    files = []
    for rst in sorted(src.rglob("*.rst")):
        # skip hidden/ (excluded in Sphinx conf.py too)
        if "hidden" in rst.parts:
            continue
        files.append(rst)
    for md in sorted(src.rglob("*.md")):
        if "hidden" in md.parts:
            continue
        files.append(md)
    return files


# ---------------------------------------------------------------------------
# docs.yml builder
# ---------------------------------------------------------------------------

DOCS_YML_HEADER = """\
instances:
  - url: nvidia-cuopt.docs.buildwithfern.com

title: NVIDIA cuOpt

global-theme: nvidia

navbar-links:
  - type: github
    value: https://github.com/NVIDIA/cuopt

logo:
  dark: docs/images/nvidia-cuopt-logo-dark.svg
  light: docs/images/nvidia-cuopt-logo-light.svg
  href: https://docs.nvidia.com/cuopt/

colors:
  accentPrimary:
    dark: "#76B900"
    light: "#76B900"

libraries:
  cuopt-c-api:
    input:
      path: ../cpp
    output:
      path: ./static/cpp-docs
    lang: cpp
    config:
      doxyfile: ./Doxyfile.fern

  cuopt-python-api:
    input:
      path: ../python/cuopt/cuopt
    output:
      path: ./static/python-docs
    lang: python

css:
  - docs/scripts/install-selector.css

js:
  - path: docs/scripts/cuopt-install-version.js
  - path: docs/scripts/install-selector.js

tabs:
  docs:
    display-name: Documentation
    icon: book

versions:
  - display-name: "26.08 (Latest)"
    path: docs-v26-08.yml
  - display-name: "26.06"
    path: docs-v26-06.yml
  - display-name: "26.04"
    path: docs-v26-04.yml
  - display-name: "26.02"
    path: docs-v26-02.yml
  - display-name: "25.12"
    path: docs-v25-12.yml
  - display-name: "25.08"
    path: docs-v25-08.yml
  - display-name: "25.05"
    path: docs-v25-05.yml
"""

def _inject_nav(items: list, path: list, entry: dict) -> bool:
    """Walk the nav tree along `path` (section names) and append `entry` to the
    matching section's contents.  Returns True if the target was found."""
    if not path:
        items.append(entry)
        return True
    target = path[0]
    for item in items:
        if item.get("section") == target:
            return _inject_nav(item.setdefault("contents", []), path[1:], entry)
    return False


def _inject_nav_after(items: list, section_path: list, after_page: str, entry: dict) -> bool:
    """Walk to section_path, then insert `entry` immediately after the page
    whose display name is `after_page`.  Falls back to appending if not found."""
    if not section_path:
        # We're at the target level — find after_page and insert after it.
        for i, item in enumerate(items):
            if item.get("page") == after_page or item.get("api") == after_page:
                items.insert(i + 1, entry)
                return True
        # after_page not found — just append
        items.append(entry)
        return True
    target = section_path[0]
    for item in items:
        if item.get("section") == target:
            return _inject_nav_after(
                item.setdefault("contents", []), section_path[1:], after_page, entry
            )
    return False


def _rename_page(items: list, section_path: list, old_name: str, new_name: str) -> bool:
    """Walk to section_path, then rename the first page whose display name
    matches `old_name` to `new_name`."""
    if not section_path:
        for item in items:
            if item.get("page") == old_name:
                item["page"] = new_name
                return True
        return False
    target = section_path[0]
    for item in items:
        if item.get("section") == target:
            return _rename_page(item.get("contents", []), section_path[1:], old_name, new_name)
    return False


def _remove_page(items: list, path_value: str) -> bool:
    """Remove the first nav entry whose `path` field equals `path_value`
    from anywhere in the tree (depth-first)."""
    for i, item in enumerate(items):
        if item.get("path") == path_value:
            del items[i]
            return True
        if "contents" in item:
            if _remove_page(item["contents"], path_value):
                return True
    return False


def _nav_to_yaml(items: list, indent: int = 0) -> str:
    """Render a nav item list to YAML lines."""
    pad = "  " * indent
    lines = []
    for item in items:
        if "section" in item:
            lines.append(f"{pad}- section: {_yaml_str(item['section'])}")
            lines.append(f"{pad}  contents:")
            lines.append(_nav_to_yaml(item["contents"], indent + 2))
        elif "page" in item:
            lines.append(f"{pad}- page: {_yaml_str(item['page'])}")
            lines.append(f"{pad}  path: {item['path']}")
        elif "folder" in item:
            lines.append(f"{pad}- folder: {item['folder']}")
            if "title" in item:
                lines.append(f"{pad}  title: {_yaml_str(item['title'])}")
        elif "api" in item:
            lines.append(f"{pad}- api: {_yaml_str(item['api'])}")
            if "spec" in item:
                lines.append(f"{pad}  spec: {item['spec']}")
    return "\n".join(lines)


def _yaml_str(s: str) -> str:
    if any(c in s for c in ':{}[]|>&*!,#?-"\'\n'):
        return '"' + s.replace('"', '\\"') + '"'
    return s


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    PAGES.mkdir(parents=True, exist_ok=True)
    IMAGES.mkdir(parents=True, exist_ok=True)

    # 1. Copy images
    src_images = SRC / "images"
    if src_images.exists():
        for img in src_images.iterdir():
            shutil.copy2(img, IMAGES / img.name)
    # Also copy sub-directory images (routing, grpc, etc.)
    for img in SRC.rglob("images/*"):
        if img.is_file() and "hidden" not in img.parts:
            shutil.copy2(img, IMAGES / img.name)
    print(f"Copied images to {IMAGES}")

    # 2. Convert RST/MD files
    files = _collect_rst_files(SRC)
    print(f"Converting {len(files)} files...")
    for f in files:
        rel = f.relative_to(SRC)
        dest = PAGES / rel.with_suffix(".mdx")
        if f.suffix == ".rst":
            convert_rst(f, dest)
        else:
            convert_md(f, dest)
        print(f"  {rel} → fern/docs/pages/{rel.with_suffix('.mdx')}")

    # 3. Build navigation from toctree
    index_rst = SRC / "index.rst"
    nav_items = _build_nav_tree(index_rst, SRC)

    # 4. Inject optional extras and apply curated nav fixes.

    # 4a. Python distance engine library docs.
    python_lib_entry = {}
    if (FERN / "static/python-docs").exists() and any((FERN / "static/python-docs").rglob("*.mdx")):
        python_lib_entry = {"folder": "./static/python-docs", "title": "Python Distance Engine"}
    if python_lib_entry:
        _inject_nav(nav_items, ["Python API", "Routing Optimization"], python_lib_entry)

    # 4b. REST API explorer: insert directly after "cuOpt Server CLI" in Server API Overview.
    _inject_nav_after(
        nav_items,
        section_path=["Server", "Server API", "Server API Overview"],
        after_page="cuOpt Server CLI",
        entry={"api": "REST API Reference"},
    )

    # 4c. Settings pages under Python API (same shared MDX as C API / Server).
    _inject_nav_after(
        nav_items,
        section_path=["Python API", "Convex Optimization (LP/QP/QCQP/SOCP)", "LP/QP/QCQP/SOCP Python API"],
        after_page="Convex Optimization API Reference",
        entry={"page": "Convex Optimization Settings", "path": "docs/pages/convex-settings.mdx"},
    )
    _inject_nav_after(
        nav_items,
        section_path=["Python API", "Mixed Integer Programming (MIP)", "MIP Python API"],
        after_page="MIP API Reference",
        entry={"page": "MIP Settings", "path": "docs/pages/mip-settings.mdx"},
    )

    # 4d. Rename "Resources" page inside the "Resources" section to avoid the
    #     sidebar showing the same name twice (section label + page label).
    _rename_page(nav_items, ["Resources"], "Resources", "External Resources")

    # 4e. Same fix for "Third-Party Modeling Languages" section.
    _rename_page(nav_items, ["Third-Party Modeling Languages"], "Third-Party Modeling Languages", "Overview")

    # 4f. Remove the stub Swagger page (REST API Reference replaces it).
    _remove_page(nav_items, "docs/pages/open-api.mdx")

    all_nav = nav_items
    # Write current-version navigation to its own file (Fern requires navigation
    # and versions to be in separate files when versioning is enabled).
    nav_yaml = _nav_to_yaml(all_nav, indent=1)
    current_nav_yml = "navigation:\n" + nav_yaml + "\n"
    (FERN / "docs-v26-08.yml").write_text(current_nav_yml, encoding="utf-8")
    print(f"Wrote fern/docs-v26-08.yml")

    # Write the global docs.yml (settings + versions list, no navigation).
    (FERN / "docs.yml").write_text(DOCS_YML_HEADER, encoding="utf-8")
    print(f"Wrote fern/docs.yml")

    # 7. Generate OpenAPI spec from FastAPI app (no GPU required)
    _generate_openapi_spec()

    # 8. Extract Python API docstrings via AST
    _extract_python_api()

    # 9. Extract C API docs from headers and regenerate C API MDX pages
    _extract_c_api()

    print("\nDone. Run: fern check")


def _generate_openapi_spec():
    """Generate cuopt_spec.yaml from the FastAPI app without a GPU, and write the version script."""
    try:
        import sys as _sys
        _sys.path.insert(0, str(REPO_ROOT / "python/cuopt_server"))
        from cuopt_server.webserver import app
        from fastapi.openapi.utils import get_openapi
        import yaml

        spec = get_openapi(
            title=app.title,
            version=app.version,
            openapi_version=app.openapi_version,
            description=app.description,
            summary=getattr(app, "summary", None),
            routes=app.routes,
        )
        # IdModel has additionalProperties:false, but the GET /cuopt/solution/{id}
        # 200 response uses anyOf[IdModel, SolutionModelWithId, ...] with examples that
        # include a 'response' key.  Fern's dev server (stricter than fern check) treats
        # this as a fatal error.  Relax IdModel for docs purposes — the schema is trivial
        # (only reqId) so this doesn't change what users see.
        schemas = spec.get("components", {}).get("schemas", {})
        # Relax IdModel: its additionalProperties:false causes Fern dev-server to reject
        # anyOf examples that include a 'response' key (valid for SolutionModelWithId).
        if "IdModel" in schemas:
            schemas["IdModel"].pop("additionalProperties", None)

        # Fix health-endpoint 500 examples: DetailModel requires error_result (bool)
        # but the generated examples only include 'error'.
        for path_item in spec.get("paths", {}).values():
            for op in path_item.values():
                if not isinstance(op, dict):
                    continue
                for resp in op.get("responses", {}).values():
                    for media in resp.get("content", {}).values():
                        for ex in media.get("examples", {}).values():
                            val = ex.get("value", {})
                            if isinstance(val, dict) and "error" in val and "error_result" not in val:
                                val["error_result"] = False

        # Fix incumbents examples: cost/bound must be number|null, not a string;
        # bound is required but missing from the generated examples.
        incumbents_path = "/cuopt/solution/{id}/incumbents"
        inc_op = spec.get("paths", {}).get(incumbents_path, {}).get("get", {})
        inc_examples = (
            inc_op.get("responses", {})
            .get("200", {})
            .get("content", {})
            .get("application/json", {})
            .get("examples", {})
        )
        for ex in inc_examples.values():
            items = ex.get("value") or []
            for item in items:
                if isinstance(item, dict):
                    if not isinstance(item.get("cost"), (int, float)):
                        item["cost"] = None
                    if "bound" not in item:
                        item["bound"] = None

        out = FERN / "openapi/cuopt_spec.yaml"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            yaml.dump(spec, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        print(f"Generated {out.relative_to(REPO_ROOT)} ({len(spec['paths'])} paths)")

        # Write the version injection script for the install selector widget
        ver = app.version  # e.g. "26.08"
        parts = ver.split(".")
        major, minor = int(parts[0]), int(parts[1]) if len(parts) > 1 else 0
        pip_ver = f"{major}.{minor}"
        conda_ver = f"{major:02d}.{minor:02d}"
        version_js = (
            FERN / "docs/scripts/cuopt-install-version.js"
        )
        version_js.parent.mkdir(parents=True, exist_ok=True)
        version_js.write_text(
            f'window.CUOPT_INSTALL_VERSION = {{"conda": "{conda_ver}", "pip": "{pip_ver}"}};\n'
        )
        print(f"Generated {version_js.relative_to(REPO_ROOT)} (conda={conda_ver}, pip={pip_ver})")
    except Exception as e:
        print(f"  [WARN] Could not generate OpenAPI spec or version script: {e}")
        print("         Run manually: python fern/convert_docs.py (with cuopt_server installed)")


def _extract_python_api():
    """Delegate to extract_python_api.py for Python API MDX generation."""
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "extract_python_api", FERN / "extract_python_api.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.generate_pages()
    except Exception as e:
        print(f"  [WARN] Python API extraction failed: {e}")


def _extract_c_api():
    """Delegate to extract_c_api.py for C API MDX generation."""
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "extract_c_api", FERN / "extract_c_api.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.generate_pages()
    except Exception as e:
        print(f"  [WARN] C API extraction failed: {e}")


if __name__ == "__main__":
    main()
