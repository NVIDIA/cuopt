# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Extension-based file format dispatch for LP I/O."""

_MPS_SUFFIXES = (".mps", ".qps")
_LP_SUFFIXES = (".lp",)
_SUPPORTED_WRITE_SUFFIXES = _MPS_SUFFIXES + _LP_SUFFIXES


def file_format_from_path(file_path: str) -> str:
    """Infer the output format from a file path (case-insensitive).

    Mirrors the extension dispatch used by :func:`Read` and the C++
    ``file_format_from_path`` helper:

    - ``.lp`` → ``"lp"``
    - ``.mps``, ``.qps`` → ``"mps"``

    Compressed output is not supported; paths ending in ``.gz`` or ``.bz2``
    are rejected.

    Parameters
    ----------
    file_path : str
        Output file path.

    Returns
    -------
    str
        ``"lp"`` or ``"mps"``.

    Raises
    ------
    RuntimeError
        If the file extension is not one of the supported suffixes, including
        compressed output suffixes.
    """
    lower = file_path.lower()
    for suffix in _LP_SUFFIXES:
        if lower.endswith(suffix):
            return "lp"
    for suffix in _MPS_SUFFIXES:
        if lower.endswith(suffix):
            return "mps"
    supported = ", ".join(_SUPPORTED_WRITE_SUFFIXES)
    raise RuntimeError(
        "write: unrecognized output file extension. "
        f"Supported (case-insensitive): {supported}. "
        "Compressed output is not supported. "
        f"Given path: {file_path}"
    )
