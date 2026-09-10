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
    # Resolve relative to the cuopt_server package so imports that leave
    # utils/ (e.g. from ..utils.deprecated) are still detected.
    pkg_root = utils_root.parent
    parts = list(file_path.parent.relative_to(pkg_root).parts)
    dots = len(name) - len(name.lstrip("."))
    remainder = name[dots:]
    up = dots - 1
    if up > len(parts):
        return False
    base = parts[: len(parts) - up]
    target = list(base)
    if remainder:
        target.extend(p for p in remainder.split(".") if p)
    return (
        len(target) >= 2 and target[0] == "utils" and target[1] == "deprecated"
    )


def test_relative_import_of_deprecated_is_detected():
    utils_root = _utils_root()
    in_utils = utils_root / "http_codec.py"
    in_routing = utils_root / "routing" / "conversion.py"
    assert _is_deprecated_import(".deprecated", in_utils, utils_root)
    assert _is_deprecated_import("..utils.deprecated", in_utils, utils_root)
    assert _is_deprecated_import(
        "..deprecated.job_queue", in_routing, utils_root
    )
    assert not _is_deprecated_import("..logutil", in_utils, utils_root)


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
    env["PYTHONPATH"] = str(_utils_root().parents[1]) + (
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
