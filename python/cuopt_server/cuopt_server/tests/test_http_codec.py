# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import pickle
import zlib

import msgpack
import numpy as np
import pytest
from fastapi import HTTPException
from fastapi.responses import JSONResponse, Response

from cuopt_server.utils.http_codec import (
    PickleForbidden,
    decode,
    deserialize,
    encode,
    encode_bytes,
    get_format,
    mime_json,
    mime_msgpack,
    mime_pickle,
    mime_wild,
    mime_zlib,
)

body_mime_types = [mime_json, mime_msgpack, mime_zlib]

sample_data = {
    "cost_matrix_data": {"data": {"1": [[0, 1], [1, 0]]}},
    "task_data": {"task_locations": [1, 1]},
    "solver_config": {"time_limit": 1.0},
}


@pytest.mark.parametrize("mime_type", body_mime_types)
def test_round_trip(mime_type):
    d = encode_bytes(sample_data, mime_type)
    assert isinstance(d, bytes)
    assert decode(mime_type, d) == sample_data
    assert deserialize(mime_type, d) == sample_data


def test_encode_bytes_wire_format():
    assert json.loads(encode_bytes(sample_data, mime_json)) == sample_data
    assert (
        json.loads(zlib.decompress(encode_bytes(sample_data, mime_zlib)))
        == sample_data
    )
    assert (
        msgpack.loads(
            encode_bytes(sample_data, mime_msgpack), strict_map_key=False
        )
        == sample_data
    )


def test_decode_unknown_content_type_is_msgpack():
    # Anything that is not json, zlib, or pickle is decoded as msgpack
    assert decode("application/unknown", msgpack.dumps(sample_data)) == (
        sample_data
    )


def test_msgpack_numpy_round_trip():
    arrays = {
        "primal": np.array([1.5, 2.5]),
        "offsets": np.array([0, 2], dtype=np.int32),
    }
    result = decode(mime_msgpack, encode_bytes(arrays, mime_msgpack))
    for key, value in arrays.items():
        assert np.array_equal(result[key], value)
        assert result[key].dtype == value.dtype


@pytest.mark.parametrize("mime_type", body_mime_types)
def test_deserialize_bad_data(mime_type):
    with pytest.raises(HTTPException) as e:
        deserialize(mime_type, b"this is not valid in any format")
    assert e.value.status_code == 422
    assert "unable to load optimization data stream" in e.value.detail


def test_encode_json_returns_result():
    assert encode(sample_data, mime_json) == sample_data


def test_encode_unsupported_accept_returns_json():
    assert encode(sample_data, "application/unknown") == sample_data


@pytest.mark.parametrize("accept", [mime_msgpack, mime_zlib])
def test_encode_binary_accept(accept):
    r = encode(sample_data, accept)
    assert isinstance(r, Response)
    assert r.media_type == accept
    assert r.status_code == 200
    assert decode(accept, r.body) == sample_data


@pytest.mark.parametrize("accept", mime_wild)
def test_encode_wildcard_accept_is_msgpack(accept):
    # Callers resolve wildcards before encoding, an unresolved
    # wildcard falls through to msgpack
    r = encode(sample_data, accept)
    assert r.media_type == mime_msgpack
    assert decode(mime_msgpack, r.body) == sample_data


@pytest.mark.parametrize("accept", body_mime_types)
def test_encode_error_result(accept):
    expected = {"error": "something failed", "error_result": True}
    r = encode(JSONResponse({"error": "something failed"}, 500), accept, True)
    assert r.status_code == 500
    if accept == mime_json:
        assert isinstance(r, JSONResponse)
        assert json.loads(r.body) == expected
    else:
        assert r.media_type == accept
        assert decode(accept, r.body) == expected


def test_get_format():
    formats = [
        get_format(m)
        for m in [mime_json, mime_zlib, mime_msgpack, mime_pickle]
    ]
    assert formats == ["json", "zlib", "msgpack", "pickle"]


@pytest.mark.parametrize("mime_type", body_mime_types)
def test_job_queue_uses_codec(mime_type):
    # job_queue re-exports the shared mime types and defers to the codec
    from cuopt_server.utils import job_queue

    assert job_queue.mime_json == mime_json
    assert job_queue.mime_msgpack == mime_msgpack
    assert job_queue.mime_pickle == mime_pickle
    assert job_queue.mime_wild == mime_wild
    assert job_queue.mime_zlib == mime_zlib
    assert (
        job_queue.deserialize(mime_type, encode_bytes(sample_data, mime_type))
        == sample_data
    )


def test_pickle_round_trip():
    encoded = pickle.dumps(sample_data)
    assert decode(mime_pickle, encoded) == sample_data
    assert deserialize(mime_pickle, encoded) == sample_data


def test_pickle_forbidden_class():
    encoded = pickle.dumps({"obj": object()})
    with pytest.raises(PickleForbidden):
        decode(mime_pickle, encoded)
    with pytest.raises(HTTPException) as e:
        deserialize(mime_pickle, encoded)
    assert e.value.status_code == 422


def test_job_queue_pickle_uses_codec():
    from cuopt_server.utils import job_queue
    from cuopt_server.utils import http_codec as codec

    assert job_queue.deserialize is codec.deserialize
    assert job_queue.SafeUnpickler is codec.SafeUnpickler
    assert job_queue.cuopt_pickle_load is codec.cuopt_pickle_load
