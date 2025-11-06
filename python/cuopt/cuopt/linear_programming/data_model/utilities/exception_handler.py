# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import functools
import json


class InputValidationError(Exception):
    pass


class InputRuntimeError(Exception):
    pass


class OutOfMemoryError(Exception):
    pass


def catch_cuopt_exception(f):
    @functools.wraps(f)
    def func(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except RuntimeError as e:
            err_msg = str(e)
            if "CUOPT_ERROR_TYPE" in err_msg:
                err = json.loads(err_msg.split("\n")[0])
                if err["CUOPT_ERROR_TYPE"] == "ValidationError":
                    raise InputValidationError(err["msg"])
                elif err["CUOPT_ERROR_TYPE"] == "RuntimeError":
                    raise InputRuntimeError(err["msg"])
                elif err["CUOPT_ERROR_TYPE"] == "OutOfMemoryError":
                    raise OutOfMemoryError(err["msg"])
                else:
                    raise RuntimeError(err["msg"])
            elif "MPS_PARSER_ERROR_TYPE" in err_msg:
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
