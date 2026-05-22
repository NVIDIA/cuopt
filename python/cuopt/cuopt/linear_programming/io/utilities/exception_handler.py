# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import functools
import json


class InputValidationError(Exception):
    pass


class InputRuntimeError(Exception):
    pass


class OutOfMemoryError(Exception):
    pass


def catch_io_exception(f):
    """Translate the C++ parser's JSON-tagged RuntimeError to a typed Python
    exception. The error tag string ("MPS_PARSER_ERROR_TYPE") is preserved
    verbatim because it's produced by the C++ mps_parser_expects() macro,
    which is shared between the MPS and LP parsers; renaming it would be a
    C++-side change with a larger blast radius.
    """

    @functools.wraps(f)
    def func(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except RuntimeError as e:
            err_msg = str(e)
            if "MPS_PARSER_ERROR_TYPE" in err_msg:
                err = json.loads(err_msg.split("\n")[0])
                if err["MPS_PARSER_ERROR_TYPE"] == "ValidationError":
                    raise InputValidationError(err["msg"])
                elif err["MPS_PARSER_ERROR_TYPE"] == "RuntimeError":
                    raise InputRuntimeError(err["msg"])
                elif err["MPS_PARSER_ERROR_TYPE"] == "OutOfMemoryError":
                    raise OutOfMemoryError(err["msg"])
                else:
                    raise RuntimeError(err["msg"])
            else:
                raise e
        except Exception as e:
            raise e

    return func
