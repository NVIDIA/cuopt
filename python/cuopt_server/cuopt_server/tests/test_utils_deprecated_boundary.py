# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
import os
import subprocess
import sys
from pathlib import Path

import cuopt_server.utils as utils_pkg


def _utils_root():
    return Path(utils_pkg.__file__).resolve().parent


def _permanent_python_files():
    root = _utils_root()
    for path in root.rglob("*.py"):
        try:
            path.relative_to(root / "deprecated")
        except ValueError:
            yield path
        else:
            continue


def _imported_names(tree):
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            prefix = "." * node.level
            names.append(prefix + module)
            for alias in node.names:
                if module:
                    names.append(prefix + module + "." + alias.name)
                else:
                    names.append(prefix + alias.name)
    return names


def _is_deprecated_import(name, file_path, utils_root):
    if name.startswith("cuopt_server.utils.deprecated"):
        return True
    if name == "cuopt_server.utils.deprecated":
        return True
    if not name.startswith("."):
        return False
    rel = file_path.parent.relative_to(utils_root)
    parts = list(rel.parts)
    dots = len(name) - len(name.lstrip("."))
    remainder = name[dots:]
    up = dots - 1
    if up > len(parts):
        return False
    base = parts[: len(parts) - up]
    target = list(base)
    if remainder:
        target.extend(p for p in remainder.split(".") if p)
    return bool(target) and target[0] == "deprecated"


def test_permanent_utils_do_not_import_deprecated():
    utils_root = _utils_root()
    violations = []
    for path in _permanent_python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for name in _imported_names(tree):
            if _is_deprecated_import(name, path, utils_root):
                violations.append(f"{path.relative_to(utils_root)}: {name}")
    assert violations == []


def test_importing_cuopt_server_does_not_load_legacy_service():
    env = os.environ.copy()
    env["PYTHONPATH"] = str(_utils_root().parents[2]) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    probe = (
        "import sys\n"
        "import cuopt_server\n"
        "loaded = [m for m in sys.modules if m.startswith('cuopt_server')]\n"
        "assert 'cuopt_server.cuopt_service' not in sys.modules, loaded\n"
        "assert not any('utils.deprecated' in m for m in sys.modules), loaded\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
