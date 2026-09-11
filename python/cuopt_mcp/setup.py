# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Copy the generated settings schema into the package at build time.

The schema is a codegen artifact owned by cpp/src/grpc/codegen/generated/.
Copying it here at build time rather than committing a second copy keeps
field_registry.yaml the single source of truth: there is no checked-in file
that can drift from it.

A source checkout needs no copy — cuopt_mcp.schema falls back to the codegen
output directory directly.
"""

import shutil
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py

SCHEMA = "cuopt_mcp_schema.json"
SOURCE = (
    Path(__file__).resolve().parents[2]
    / "cpp"
    / "src"
    / "grpc"
    / "codegen"
    / "generated"
    / SCHEMA
)


class BuildPyWithSchema(build_py):
    def run(self):
        if not SOURCE.is_file():
            raise SystemExit(
                f"{SOURCE} is missing. Run `./build.sh codegen` before "
                "building cuopt_mcp."
            )
        target = Path(self.build_lib) / "cuopt_mcp" / "_generated"
        target.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SOURCE, target / SCHEMA)
        super().run()


setup(cmdclass={"build_py": BuildPyWithSchema})
