# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Host-filesystem helpers for cuOpt HTTP: stage a problem under
# CUOPT_DATA_DIR and optionally write a result under CUOPT_RESULT_DIR
# when the serialized response meets the maxresult threshold.

import json
import logging
import os
import time
import uuid
import zlib

import msgpack
from fastapi import HTTPException

import cuopt_server.utils.settings as settings
from cuopt_server.utils.http_codec import (
    PickleForbidden,
    cuopt_pickle_load,
)


def get_output_name(resultdir, CUOPT_DATA_FILE, CUOPT_RESULT_FILE):
    # Reject paths that escape resultdir using canonicalized containment check.
    if CUOPT_RESULT_FILE and resultdir:
        root = os.path.realpath(resultdir)
        candidate = os.path.realpath(os.path.join(root, CUOPT_RESULT_FILE))
        if (
            os.path.isabs(CUOPT_RESULT_FILE)
            or os.path.commonpath([root, candidate]) != root
        ):
            CUOPT_RESULT_FILE = ""
    if not resultdir:
        res = ""
    elif CUOPT_RESULT_FILE:
        res = CUOPT_RESULT_FILE
    elif CUOPT_DATA_FILE:
        res = os.path.basename(CUOPT_DATA_FILE) + ".result"
    else:
        res = str(uuid.uuid4())
    return res


def validate_file_path(cuopt_data_file):
    ddir = settings.get_data_dir()
    if not ddir:
        logging.error("cuopt data directory not set!")
        raise HTTPException(
            status_code=400,
            detail="cuopt data directory not set",
        )

    if os.path.isabs(cuopt_data_file):
        raise HTTPException(
            status_code=400,
            detail="cuopt-data-file must be relative to CUOPT_DATA_DIR",
        )

    root = os.path.realpath(ddir)
    file_path = os.path.realpath(os.path.join(root, cuopt_data_file))
    if os.path.commonpath([root, file_path]) != root:
        raise HTTPException(
            status_code=400,
            detail="cuopt-data-file must stay inside CUOPT_DATA_DIR",
        )

    if not os.path.exists(file_path):
        logging.error("cuopt-data-file does not exist")
        raise HTTPException(
            status_code=400,
            detail=f"specified data file does not exist: {cuopt_data_file}",
        )

    if not os.path.isfile(file_path):
        logging.error("cuopt-data-file is not a regular file")
        raise HTTPException(
            status_code=400,
            detail=(
                f"specified data file is not a regular file: {cuopt_data_file}"
            ),
        )

    return file_path


def result_meets_threshold(resultfile, resultdir, data_size, maxresult):
    return bool(resultfile and resultdir and data_size >= maxresult * 1000)


def write_result_file(resultdir, resultfile, buf, mode=None):
    op = os.path.join(resultdir, resultfile)
    logging.debug(f"Writing large result to disk {op}")
    with open(op, "wb") as out:
        out.write(buf)
    if mode:
        os.chmod(op, mode)
    return op


def file_result_message(resultfile, warnings=None, notes=None):
    r = {"result_file": resultfile}
    if warnings:
        logging.debug("adding warnings to file result")
        r["warnings"] = warnings
    if notes:
        logging.debug("adding notes to file result")
        r["notes"] = notes
    return r


def decode_file_bytes(ext, raw_data, warnings=None):
    if ext == "zlib":
        data = json.loads(zlib.decompress(raw_data))
        logging.debug("zlib data")
    elif ext == "msgpack":
        data = msgpack.loads(raw_data, strict_map_key=False)
        logging.debug("msgpack serialized data")
    elif ext == "json":
        data = json.loads(raw_data)
        logging.debug("uncompressed data")
    elif ext == "pickle":
        data = cuopt_pickle_load(raw_data, kind="")
        if warnings is not None:
            warnings.append(
                "Pickle data format is deprecated. "
                "Use zlib, msgpack, or plain JSON"
            )
        logging.warning("pickle data is deprecated")
        logging.debug("pickle data")
    else:
        raise ValueError(
            f"File extension {ext} is unsupported. "
            "Supported file extensions are "
            ".json, .zlib, .msgpack, or .pickle"
        )
    return data


def load_optimization_file(file_path, warnings=None):
    # read the data from the file
    # if we have an extension, use it otherwise try everything
    try:
        ext = file_path.split(".")[-1] if "." in file_path else ""
        read_begin = time.time()
        with open(file_path, "rb") as f:
            raw_data = f.read()
            if ext:
                data = decode_file_bytes(ext, raw_data, warnings)
            else:
                for e in ["msgpack", "json", "zlib", "pickle"]:
                    try:
                        data = decode_file_bytes(e, raw_data, warnings)
                        break
                    except PickleForbidden:
                        # In this case we know it loaded as pickle but
                        # it failed the class restrictions, no reason
                        # to try anything else
                        raise

                    except Exception:
                        pass
                else:
                    raise HTTPException(
                        status_code=422,
                        detail="unable to read "
                        "optimization data file, "
                        "no file extension present and failed to load "
                        "as any supported format",
                    )
            logging.debug(f"Total file load time {time.time() - read_begin}")

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=422,
            detail="unable to read optimization data file, %s" % (str(e)),
        )

    return data
