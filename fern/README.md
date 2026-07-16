# cuOpt Fern Docs

This directory contains the [Fern](https://buildwithfern.com)-based documentation for NVIDIA cuOpt.

## Structure

```
fern/
├── docs.yml                  # Top-level Fern config (tabs, versions, nav)
├── docs-v26-08.yml           # Per-version nav and content config
├── docs/pages/               # All MDX content pages
│   ├── cuopt-python/         # Python API docs (generated + examples)
│   ├── cuopt-c/              # C API docs (generated + examples)
│   └── ...                   # Other feature/guide pages
├── generate_api_docs.py      # Master build script (calls all extractors)
├── extract_python_api.py     # Python API → MDX (AST + Cython regex)
├── extract_c_api.py          # C API → MDX (Doxygen XML)
├── embed_examples.py         # Embeds example .py/.c files as code blocks
├── build_docs.sh             # Local preview helper
└── openapi/                  # OpenAPI spec (auto-generated)
```

## Local preview

```bash
bash fern/build_docs.sh
```

Opens at http://localhost:3000. Regenerates all API docs before starting.

## How generated content works

Three scripts run on every build (`generate_api_docs.py` calls all three):

| Script | Input | Output |
|---|---|---|
| `extract_python_api.py` | Python source + docstrings | `docs/pages/cuopt-python/*/api.mdx` |
| `extract_c_api.py` | Doxygen XML from C headers | `docs/pages/cuopt-c/*/api.mdx` |
| `embed_examples.py` | `examples/*.py` files next to each MDX page | code blocks in `*-examples.mdx` |

Never hand-edit generated MDX files — changes are overwritten on the next build.

## Adding a Python API symbol

1. Add a docstring (NumPy style) to the class or function in `python/cuopt/`.
2. Make sure the symbol is in the module's `__all__`.
3. Run `python fern/generate_api_docs.py` to regenerate.

To control which symbols appear or in what order, edit `PAGE_SOURCES` at the top of `extract_python_api.py`.

## Adding a C API symbol

1. Add a Doxygen comment to the header in `cpp/include/cuopt/mathematical_optimization/`.
2. Place a `{/* symbol: SYMBOL_NAME */}{/* /symbol */}` marker in the right section of the skeleton MDX file (`docs/pages/cuopt-c/convex/convex-c-api.mdx` or `mip-c-api.mdx`).
3. Run `python fern/generate_api_docs.py` — the marker fills in automatically.

## Adding an example

1. Drop a `.py` or `.c` file into the `examples/` subdirectory next to the relevant `*-examples.mdx` page.
2. Add a bare link in the MDX file: `[filename.py](examples/filename.py)`
3. Run `python fern/embed_examples.py` — the link is replaced with a fenced code block.

## Editing guide pages

All non-generated pages under `docs/pages/` are plain MDX and can be edited directly. Images go in `docs/images/`.

## Navigation

Navigation is defined in `docs-v26-08.yml` (current version). The Python API page entries between the `BEGIN / END auto-generated` markers are managed automatically by `extract_python_api.py`.
