# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Generate dynamic API reference content for the Fern docs.

Regenerates on every `./build.sh docs` run:
  - fern/openapi/cuopt_spec.yaml       (from FastAPI app)
  - fern/docs/scripts/cuopt-install-version.js  (version string for install widget)
  - fern/docs/pages/cuopt-python/      (Python API MDX via AST)
  - fern/docs/pages/cuopt-c/           (C API MDX from headers)
  - *-examples.mdx pages               (example .py/.c files embedded as code blocks)

Hand-edited prose in MDX pages is preserved; only code-embed markers are refreshed.
"""

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
FERN = REPO_ROOT / "fern"


def _generate_openapi_spec():
    try:
        sys.path.insert(0, str(REPO_ROOT / "python/cuopt_server"))
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

        schemas = spec.get("components", {}).get("schemas", {})
        if "IdModel" in schemas:
            schemas["IdModel"].pop("additionalProperties", None)

        for path_item in spec.get("paths", {}).values():
            for op in path_item.values():
                if not isinstance(op, dict):
                    continue
                for resp in op.get("responses", {}).values():
                    for media in resp.get("content", {}).values():
                        for ex in media.get("examples", {}).values():
                            val = ex.get("value", {})
                            if (
                                isinstance(val, dict)
                                and "error" in val
                                and "error_result" not in val
                            ):
                                val["error_result"] = False

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
            for item in ex.get("value") or []:
                if isinstance(item, dict):
                    if not isinstance(item.get("cost"), (int, float)):
                        item["cost"] = None
                    if "bound" not in item:
                        item["bound"] = None

        out = FERN / "openapi/cuopt_spec.yaml"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            yaml.dump(
                spec,
                f,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
            )
        print(
            f"Generated {out.relative_to(REPO_ROOT)} ({len(spec['paths'])} paths)"
        )

        ver = app.version
        parts = ver.split(".")
        major, minor = int(parts[0]), int(parts[1]) if len(parts) > 1 else 0
        pip_ver = f"{major}.{minor}"
        conda_ver = f"{major:02d}.{minor:02d}"
        version_js = FERN / "docs/scripts/cuopt-install-version.js"
        version_js.parent.mkdir(parents=True, exist_ok=True)
        version_js.write_text(
            f'window.CUOPT_INSTALL_VERSION = {{"conda": "{conda_ver}", "pip": "{pip_ver}"}};\n'
        )
        print(
            f"Generated {version_js.relative_to(REPO_ROOT)} (conda={conda_ver}, pip={pip_ver})"
        )

    except Exception as e:
        print(f"  [WARN] Could not generate OpenAPI spec: {e}")
        print(
            "         Ensure cuopt_server is installed: pip install -e python/cuopt_server --no-deps"
        )


def _run_module(path: Path, fn_name: str):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    getattr(mod, fn_name)()


def main():
    print("Generating OpenAPI spec and version script...")
    _generate_openapi_spec()

    print("Extracting Python API docstrings...")
    _run_module(FERN / "extract_python_api.py", "generate_pages")

    print("Extracting C API from headers...")
    _run_module(FERN / "extract_c_api.py", "main")

    print("Embedding examples into MDX pages...")
    _run_module(FERN / "embed_examples.py", "embed_all")

    print("\nDone. Run: fern check")


if __name__ == "__main__":
    main()
