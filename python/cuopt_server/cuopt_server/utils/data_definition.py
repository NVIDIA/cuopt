# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.  # noqa
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import json
import typing
import uuid
from enum import Enum
from typing import List, Literal, Optional, Union

import jsonref
from pydantic import BaseModel, Extra, Field

from .linear_programming.data_definition import (  # noqa
    IncumbentSolution,
    LPData,
    LPSolve,
    LPTupleData,
    WarmStartData,
    lp_example_data,
    lp_msgpack_example_data,
    lp_response,
    lp_zlib_example_data,
    lpschema,
    managed_lp_example_data,
    managed_lp_response,
    managed_milp_response,
    milp_response,
)
from .routing.data_definition import *  # noqa

# INPUT DATA DEFINITIONS

# Valid cuOpt actions
cuopt_actions = Literal[
    "cuOpt_OptimizedRouting",
    "cuOpt_RoutingValidator",
    "cuOpt_LP",
    "cuOpt_LPValidator",
    "cuOpt_Solver",
    "cuOpt_Validator",
]


def get_valid_actions():
    return list(typing.get_args(cuopt_actions))


class StrictModel(BaseModel):
    class Config:
        extra = Extra.forbid


class cuoptData(StrictModel):
    action: Optional[cuopt_actions] = Field(
        default="cuOpt_OptimizedRouting",
        description=(
            "Action to be performed by the service, "
            "validator action just validates input against format and "
            "base rules."
        ),
    )
    data: Optional[
        Union[OptimizedRoutingData, LPData, List[LPData]]  # noqa
    ] = Field(
        ...,
        description=("Data to be processed by the service"),
    )
    client_version: Optional[str] = Field(
        default="",
        description=(
            "cuOpt client version. Set to 'custom' to skip version check."
        ),
    )


class cuoptDataInternal(cuoptData):
    data: Optional[dict] = Field(
        ..., description=("Data to be processed by the service")
    )


# We need this for managed service endpoint, since the
# request id is at the NVCF level and we don't return one
class SolutionModel(StrictModel):
    response: Union[FeasibleSolve, InFeasibleSolve, LPSolve] = Field(  # noqa
        default=..., description="Solution"
    )
    warnings: List[str] = Field(
        default=[], description="List of warnings for users to handle issues"
    )
    notes: List[str] = Field(default=[], description="Any notes for users")


class SolutionModelWithId(SolutionModel):
    reqId: str = Field(
        default=..., description="Id of request", example=str(uuid.uuid4())
    )


class IdModel(StrictModel):
    reqId: str = Field(
        default=...,
        description="Id to poll for result",
        example=str(uuid.uuid4()),
    )


class RequestStatusModel(Enum):
    completed = "completed"
    aborted = "aborted"
    running = "running"
    queued = "queued"


class SolutionModelInFile(StrictModel):
    result_file: str = Field(
        default=..., description=("result_file is a file path to result")
    )
    warnings: List[str] = Field(
        default=[], description="List of warnings for users to handle issues"
    )
    notes: List[str] = Field(default=[], description="Any notes for users")
    format: str = Field(
        description="Indicator of the format of the result file"
    )


class DeleteRequestModel(StrictModel):
    queued: int = Field(description="Number of queued jobs that were deleted.")
    running: int = Field(
        description="Number of running jobs that were deleted."
    )
    cached: int = Field(description="Number of cached jobs that were deleted.")


class DetailModel(BaseModel):
    error: str = Field(default=..., description=("Error details"))
    error_result: bool = Field(
        default=...,
        description=("Whether or not the error is a result from the solver"),
    )


class EmptyResponseModel(StrictModel):
    class Config:
        extra = "forbid"


class HealthResponseModel(StrictModel):
    status: str = Field(
        description="The status of the service. RUNNING is normal."
    )
    version: str = Field(description="The version of the service.")


class LogResponseModel(StrictModel):
    log: List[str]
    nbytes: int


ErrorResponse = {
    400: {
        "model": DetailModel,
        "description": "Value Error Or Validation Error",
    },
    415: {
        "model": DetailModel,
        "description": "Unsupported mime_type specified",  # noqa
    },
    422: {
        "model": DetailModel,
        "description": "Unprocessable Entity or Runtime Error or Out of memory error",  # noqa
    },
    409: {
        "model": DetailModel,
        "description": "Solver returned invalid status",
    },
    500: {
        "model": DetailModel,
        "description": "Any uncaught cuOpt error or Server errors",
    },
}


IdResponse = copy.copy(ErrorResponse)
IdResponse[200] = {
    "model": IdModel,
    "description": "Successful response",
    "content": {
        "application/json": {
            "examples": {"Id response": reqId_response}  # noqa
        }
    },
}
del IdResponse[409]

IncumbentSolutionResponse = copy.copy(ErrorResponse)
IncumbentSolutionResponse[200] = {
    "content": {
        "application/json": {
            "examples": {
                "Empty response (no current solutions)": {"value": []},
                "Sentinel response (no future solutions).": {
                    "value": [
                        {
                            "solution": [],
                            "cost": "value will be 'null' but openapi docs will not display it",  # noqa
                        }
                    ],
                },
                "Solution response": {
                    "value": [
                        {"solution": [1.0, 2.0], "cost": 3.0},
                        {"solution": [4.0, 5.0], "cost": 6.0},
                    ]
                },
            }
        }
    }
}
IncumbentSolutionResponse[404] = {
    "model": DetailModel,
    "description": "Not found",
}
del IncumbentSolutionResponse[409]
del IncumbentSolutionResponse[400]


SolutionResponse = copy.copy(ErrorResponse)
SolutionResponse[200] = {
    "description": "Successful request response",
    "content": {
        "application/json": {
            "examples": {
                "VRP response": vrp_response,  # noqa
                "LP response": lp_response,  # noqa
                "MILP response": milp_response,  # noqa
                "ID response": reqId_response,  # noqa
            }
        },
        "application/vnd.msgpack": {
            "schema": {"type": "string", "format": "byte"},
            "examples": {
                "VRP response compressed with msgpack": {
                    "value": "\x82\xa8response\x81\xafsolver_response\x86\xa6status\x00\xacnum_vehicles\x02\xadsolution_cost\xcb@\x00\x00\x00\x00\x00\x00\x00\xb0objective_values\x81\xa4cost\xcb@\x00\x00\x00\x00\x00\x00\x00\xacvehicle_data\x82\xa5veh-1\x84\xa7task_id\x92\xa5Break\xa6Task-A\xadarrival_stamp\x92\xcb?\xf0\x00\x00\x00\x00\x00\x00\xcb@\x00\x00\x00\x00\x00\x00\x00\xa4type\x92\xa5Break\xa8Delivery\xa5route\x92\x01\x01\xa5veh-2\x84\xa7task_id\x94\xa5Depot\xa5Break\xa6Task-B\xa5Depot\xadarrival_stamp\x94\xcb@\x00\x00\x00\x00\x00\x00\x00\xcb@\x00\x00\x00\x00\x00\x00\x00\xcb@\x10\x00\x00\x00\x00\x00\x00\xcb@\x14\x00\x00\x00\x00\x00\x00\xa4type\x94\xa5Depot\xa5Break\xa8Delivery\xa5Depot\xa5route\x94\x00\x00\x02\x00\xaddropped_tasks\x82\xa7task_id\x90\xaatask_index\x90\xa5reqId\xd9$9147524d-c5fb-4413-976d-752e84a39123".encode(  # noqa
                        "unicode_escape"
                    )
                },
                "LP response compressed with msgpack": {
                    "value": "\x83\xa8response\x81\xafsolver_response\x82\xa6status\x01\xa8solution\x87\xafprimal_solution\x92\xcb\x00\x00\x00\x00\x00\x00\x00\x00\xcb\x00\x00\x00\x00\x00\x00\x00\x00\xaddual_solution\x92\xcb\x00\x00\x00\x00\x00\x00\x00\x00\xcb\x00\x00\x00\x00\x00\x00\x00\x00\xb0primal_objective\xcb\x00\x00\x00\x00\x00\x00\x00\x00\xaedual_objective\xcb\x00\x00\x00\x00\x00\x00\x00\x00\xabsolver_time\xcb@\x08\x00\x00\x00\x00\x00\x00\xa4vars\x80\xadlp_statistics\x84\xafprimal_residual\xcb\x00\x00\x00\x00\x00\x00\x00\x00\xaddual_residual\xcb\x00\x00\x00\x00\x00\x00\x00\x00\xa3gap\xcb\x00\x00\x00\x00\x00\x00\x00\x00\xacreduced_cost\x92\xcb?\xc9\x99\x99\x99\x99\x99\x9a\xcb?\xb9\x99\x99\x99\x99\x99\x9a\xa5reqId\xd9$d47dfa94-e9f4-4207-ba70-a07775a3b044\xa5notes\x91\xa7Optimal".encode(  # noqa
                        "unicode_escape"
                    )
                },
            },
        },
        "application/zlib": {
            "schema": {"type": "string", "format": "byte"},
            "examples": {
                "VRP response compressed with zlib": {
                    "value": "x\x01mQ\xc1n\x830\x0c\xfd\x15\x94s3\x01*\x08v[\xd5\xcb\xee\xbbM(\xf2\x82\xab\xb1R\x92%\x01\xadB\xfc\xfb\xec\xac\xea\xca\xbaKd=?\xfb=\xbf\xcc\xc2\xa1\xb7f\xf0(\x1e\x93Yx\xd3O\xe8\xd4\x1a\x0b\x10FO\xedt\x93\x88a<\xa9\t\xdf;\xdd#C9A43\x86\xce\x0cJ\x1b\x1f\x18{`\xa2y\xfb@\x1d\xba\t\xd5\x04\xfd\x18\xc9\xb3\xf8e,D\xb9\xecQ-\x04\x88\xea\x04\xc8,V\x01\xfcQu-\xd5\xafb\xe7\x10\x8e\x82\xf8/\x04\xca'\xd1P\t\xceu\xb4W\xf9\x00'\xcb\xac\x8cEI\x99\x9b\xe1l\xf9\x9a\x9b\xc9=\xf6\xe4\xc4\x9d\xe3\xac3c\x88\xfdl\x93d\xcd\xc5\x88\xcc\xeft\xf7hM`\xdd\xb5\x81\x1dC?\xbd\x7f\x9d\xc4\xf3\xe3\xb3eO\xc5\x1fO\xf7[\xaf\xe6V{\xaf.i\x07\x9fFo\xb3\xb0\xd9\xd6\x19k\xb1U\x1c\x11\x7f\xc1,n\xc2\x8a\xe7\xc7\xec\x86\x16\xbf8\x04\x1a\xe2)\x87\x9f\xcf\x1c\xa78\x949b\x05Z\x96\x95\xd6r[\xa7 \xeb\xbc,dq\xc0\xba\xd6e\x9eBQ\x89\xe5\x1b\xe6o\xa1\x12".encode(  # noqa
                        "unicode_escape"
                    )
                },
                "LP response compressed with zlib": {
                    "value": 'x\x01}OK\n\x830\x14\xbc\x8adm\x8a\x9fjKo\xd0U\x0f "\xf9<K\x8a5i\x12\xdd\x88w\xef\x8b\x95R\x11\x9a\xc5#\xcc\x87\x99\x99\x88\x05gt\xef\x80\\\xa2\x898\xdd\x8d`\x9b-\xe6\x99\x1f\x1c\xd2i\x1c\x05\xc1\xe0\x95\xee\x17\xb5\xb1\xea\xc9\xba\xe6\x07\xab\x92C\x12Gxj\xd4\xca\xe1\x0f\xb9z5\x7f\x80\xf0j\x0c\xf1\x8b\xf7\xe3\xda\xc3k3\xaf\x9eA\x9a\x87\x1822\x1bzM3\xfe;\xd38,\xaa\x9cWb\x01\xc9\x9a\x80[Th\xb2\r\xd8\xa1wf\xbe\n\x0br\x10 \x1b\xa1\x9dG\x10GeaTZ\xcf\xf80\xcb\xc2\xeb*\x91 \xe2T\x14\xb2\xcd[*\xce-\xa7\xc7,\x01\xcaE\x96S^B)\x99hy\n)AC\xaf=\x84R\x15\xb9\x19\\\x80e\xea\xf9\r%\xb1vx'.encode(  # noqa
                        "unicode_escape"
                    )
                },
            },
        },
    },
}
SolutionResponse[404] = {"model": DetailModel, "description": "Not found"}


RequestResponse = copy.copy(ErrorResponse)
RequestResponse[200] = {
    "description": "Successful request response",
    "content": {
        "application/json": {
            "examples": {
                "completed request": {"value": RequestStatusModel.completed},
                "queued request": {"value": RequestStatusModel.queued},
                "aborted request": {"value": RequestStatusModel.aborted},
                "running request": {"value": RequestStatusModel.running},
            }
        }
    },
}
RequestResponse[404] = {"model": DetailModel, "description": "Not found"}
del RequestResponse[409]
del RequestResponse[400]

# cuopt/cuopt only returns results as JSON currently,
# we might change that in the future
ManagedRequestResponse = copy.copy(ErrorResponse)
ManagedRequestResponse[200] = {
    "description": "Successful request response",
    "content": {
        "application/json": {
            "examples": {
                "VRP response": managed_vrp_response,  # noqa
                "LP response": managed_lp_response,  # noqa
                "MILP response": managed_milp_response,  # noqa
            }
        },
    },
}

ValidationErrorResponse = {
    404: {"model": DetailModel, "description": "Not found"},
    422: {"model": DetailModel, "description": "Unprocessable Entity"},
    500: {
        "model": DetailModel,
        "description": "Any uncaught cuOpt error or Server errors",
    },
}

DeleteResponse = copy.copy(ValidationErrorResponse)
DeleteResponse[200] = {
    "model": EmptyResponseModel,
    "description": "Successful request response (no content)",
}

HealthResponse = {
    200: {
        "model": HealthResponseModel,
        "description": "Successful health check",
        "content": {
            "application/json": {
                "examples": {
                    "Healthy response": {
                        "value": {"status": "RUNNING", "version": "25.10"}
                    }
                }
            }
        },
    },
    500: {
        "model": DetailModel,
        "description": "Unsuccessful health check",
        "content": {
            "application/json": {
                "examples": {
                    "Unhealthy response": {
                        "value": {
                            "error": "\nStatus : Broken\nThe server will be restarted and will be available in 15 mins !!!\nERROR : Server is BROKEN"  # noqa
                        }
                    }
                }
            }
        },
    },
}

LogResponse = copy.copy(ErrorResponse)
LogResponse[404] = {"model": DetailModel, "description": "Not found"}
del LogResponse[409]
del LogResponse[400]

cuoptdataschema = jsonref.loads(
    json.dumps(cuoptData.model_json_schema()), proxies=False
)
del cuoptdataschema["$defs"]

solutionschema = jsonref.loads(
    json.dumps(SolutionModelWithId.model_json_schema()), proxies=False
)
del solutionschema["$defs"]
