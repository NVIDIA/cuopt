# Fern Docs — Developer Reference

The cuOpt docs live in `fern/`. Content is a mix of hand-edited MDX pages and auto-generated API reference pages. `fern/README.md` has the quick-start; this file covers the patterns you need when modifying or extending the system.

## Generated vs hand-edited pages

| Page type | Location | Edit how |
|---|---|---|
| Python API reference | `fern/docs/pages/cuopt-python/*/api.mdx` | Edit docstrings in source; regenerate |
| C API reference | `fern/docs/pages/cuopt-c/*/api.mdx` | Edit Doxygen comments + skeleton markers; regenerate |
| Example code blocks | `*-examples.mdx` | Edit source files in `examples/`; regenerate |
| Feature/guide pages | Everything else in `docs/pages/` | Edit MDX directly |

Run `python fern/generate_api_docs.py` after any change to source docstrings or C headers. Never hand-edit generated files.

## Python API: adding or changing a symbol

**Show a new symbol:** Add it to the module's `__all__` and add a NumPy-style docstring. Regenerate.

**Control ordering:** Edit the `order` list in the relevant `PAGE_SOURCES` dir config in `extract_python_api.py`. Stems listed there are processed first; unlisted stems go alphabetically after.

**Hide a symbol:** Remove it from `__all__`, or add its module stem to the `exclude` set in `PAGE_SOURCES`.

**Wrapper `.pyx` files** (`*_wrapper.pyx`): only enums/classes surface — functions are always suppressed. Use the parent `.py` file for any functions you want documented.

**Enum member descriptions:** Use a NumPy `Attributes` section in the class docstring. Plain `MEMBER = value` with no docstring renders as a bare member list.

## C API: adding a new symbol

1. Add a Doxygen block comment above the declaration in `cuopt_c.h` or `constants.h`.
2. Place `{/* symbol: SYMBOL_NAME */}{/* /symbol */}` in the right section of the skeleton MDX.
3. Regenerate — the content fills in automatically.

If the script errors saying a symbol has no marker, add the marker first.

## MDX comment markers

Embed and symbol markers use JSX comment syntax (not HTML comments — Fern's MDX parser rejects `<!-- -->`):

```
{/* embed: examples/filename.py */}
...code block...
{/* /embed */}

{/* symbol: cuOptSolve */}
...rendered docs...
{/* /symbol */}
```

## Navigation

`docs-v26-08.yml` defines the nav tree. The Python API entries between `# BEGIN auto-generated python-api pages` and `# END auto-generated python-api pages` are rewritten automatically by `extract_python_api.py`. Everything outside those markers is hand-managed.

## Pre-commit notes

- `fern/openapi/cuopt_spec.yaml` and `fern/docs/scripts/cuopt-install-version.js` are auto-generated from `VERSION` and are excluded from `verify-hardcoded-version`.
- The `26.08 (Latest)` entry in `docs.yml` carries a `# rapids-pre-commit-hooks: disable-next-line` suppression.
- Run `pre-commit run --all-files` before pushing; `check-yaml` and `RAPIDS dependency file generator` failures on `dependencies.yaml` are pre-existing on `main` and unrelated to docs changes.
