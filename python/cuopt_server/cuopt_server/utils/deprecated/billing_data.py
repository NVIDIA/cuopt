# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import uuid
from datetime import datetime
from enum import IntEnum


class SolverStatus(IntEnum):
    SUCCESS = 0
    INFEASIBLE = 1
    TIMEOUT = 2
    ERROR = -1


class Properties:
    def __init__(self) -> None:
        # SKU, may be number of tasks and other elements to measure
        self.sku = None

        # Tier of the request which restricts access to features
        self.request_tier = None

        # Solution status
        #  0 - SUCCESS
        #  1 - INFEASIBLE
        # -1 - ERROR
        self.solution_status = SolverStatus.ERROR

        # GPU optimization time
        self.solve_time = None

        # Type of GPU in the system
        self.gpu_type = None

        # Region
        self.region = None


class BillingDataModel:
    def __init__(self, nspect_id, nca_id, request_id, request_type) -> None:
        # Nspect Id for the service
        self.nspect_id = nspect_id

        # Client's Nvidia accound ID
        self.nca_id = nca_id

        # Unique Id to represent each solve request
        self.transaction_id = str(uuid.uuid4())

        self.set_request_type(request_type)

        # Time stamp when request received
        self.timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")

        # Properties of each request
        self.properties = Properties()

    def set_request_type(self, request_type):
        # This is for future to specify type of request like,
        # vehicle routing, bin-packing and other options
        if request_type:
            self.event_type = request_type[0].upper() + request_type[1:]
        else:
            self.event_type = "unknown"

    def is_request_type_known(self):
        return self.event_type != "unknown"

    def set_request_properties(
        self,
        sku,
        request_tier=None,
        solution_status=SolverStatus.ERROR,
        solve_time_ms=0,
        gpu_type=None,
        region=None,
    ):
        self.properties.sku = sku
        self.properties.request_tier = request_tier
        self.properties.solution_status = solution_status
        self.properties.solve_time = solve_time_ms
        self.properties.gpu_type = gpu_type
        self.properties.region = region

    def get_billing(self, env):
        billing_event = {"msg": "metering event", "env": env}
        # The __dict__ method does not recurse so we
        # make 2 calls.
        d = self.__dict__
        d["properties"] = self.properties.__dict__
        billing_event["data"] = d

        return billing_event
