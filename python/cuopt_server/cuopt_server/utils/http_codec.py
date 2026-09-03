# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Shared codec for cuOpt HTTP request and response bodies.
# This module holds the mime types cuOpt speaks and the JSON, msgpack, zlib,
# and pickle serialization used by every HTTP entry point. It deliberately
# knows nothing about jobs, results, caches, shared memory, or files so that
# any HTTP front end can use it.

import io
import json
import logging
import pickle
import time
import zlib

import msgpack
import msgpack_numpy
import numpy
import numpy.core.multiarray
from fastapi import HTTPException
from fastapi.responses import JSONResponse, Response

msgpack_numpy.patch()


mime_json = "application/json"
mime_msgpack = "application/vnd.msgpack"
mime_zlib = "application/zlib"
mime_pickle = "application/octet-stream"
mime_wild = ["application/*", "*/*"]


class PickleForbidden(Exception):
    pass


class SafeUnpickler(pickle.Unpickler):
    def __init__(self, file, kind, allowed={}):
        self.allowed = allowed
        self.kind = kind
        super().__init__(file)

    def find_class(self, module, name):
        if (
            module not in self.allowed
            or name not in self.allowed[module]["names"]
        ):
            raise PickleForbidden(
                f"{module}.{name} is forbidden "
                f"in a cuopt {self.kind}pickle file"
            )
        else:
            return getattr(self.allowed[module]["mod"], name)


# LP pickle allow is superset of VRP, so allow the kind
# to be set to "" for messaging and this routine to be
# used when we don't pre-know the problem type
def cuopt_pickle_load(s, kind="LP "):
    allowed_LP = {
        "numpy.core.multiarray": {
            "names": ["_reconstruct"],
            "mod": numpy.core.multiarray,
        },
        "numpy": {"names": ["ndarray", "dtype"], "mod": numpy},
    }

    return SafeUnpickler(io.BytesIO(s), kind, allowed_LP).load()


def cuopt_pickle_load_VRP(s):
    return SafeUnpickler(io.BytesIO(s), "VRP ").load()


def get_format(mime_type):
    f = {
        mime_json: "json",
        mime_zlib: "zlib",
        mime_msgpack: "msgpack",
        mime_pickle: "pickle",
    }
    return f[mime_type]


def decode(ctype, buf):
    # Any content type other than json, zlib, or pickle is treated as
    # msgpack, matching the behavior of the original request handling
    if ctype == mime_json:
        logging.debug("decode as json")
        data = json.loads(buf)
    elif ctype == mime_zlib:
        logging.debug("decode as zlib compressed json")
        data = json.loads(zlib.decompress(buf))
    elif ctype == mime_pickle:
        logging.debug("decode as pickle")
        data = cuopt_pickle_load(buf, kind="")
    else:
        logging.debug("decode as msgpack")
        data = msgpack.loads(buf, strict_map_key=False)
    return data


def deserialize(ctype, buf):
    # decode with a client error on bad data
    try:
        data = decode(ctype, buf)
    except Exception as e:
        raise HTTPException(
            status_code=422,
            detail="unable to load optimization data stream, %s" % (str(e)),
        )
    return data


def encode_bytes(data, mime_type):
    # Write data to a byte array based on mime type
    if mime_type in [mime_json, mime_zlib]:
        d = bytes(json.dumps(data), encoding="utf-8")
        if mime_type == mime_zlib:
            now = time.time()
            d = zlib.compress(d, zlib.Z_BEST_SPEED)
            logging.debug(
                f"Time for zlib compression of result {time.time() - now}"
            )
    else:
        d = msgpack.dumps(data)
    return d


def encode(result, accept, job_result=False):
    if accept not in [mime_json, mime_msgpack, mime_zlib] + mime_wild:
        accept = mime_json

    # This is an exception packaged up elsewhere
    if isinstance(result, JSONResponse):
        status_code = result.status_code
        result = json.loads(result.body)
        result["error_result"] = job_result
        if accept == mime_json:
            return JSONResponse(result, status_code)
    else:
        status_code = 200

    # Expect a dictionary at this point
    if accept == mime_json:
        logging.debug("job_result returning json")
        r = result
    elif accept == mime_zlib:
        logging.debug("job_result returning zlib")
        d = bytes(json.dumps(result), encoding="utf-8")
        r = Response(
            content=zlib.compress(d, zlib.Z_BEST_SPEED),
            media_type=mime_zlib,
            status_code=status_code,
        )
    else:
        logging.debug("job_result returning msgpack")
        r = Response(
            content=msgpack.dumps(result),
            media_type=mime_msgpack,
            status_code=status_code,
        )
    return r
