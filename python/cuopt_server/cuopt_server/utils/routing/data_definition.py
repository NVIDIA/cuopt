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
from enum import Enum
from typing import Dict, List, Optional, Union

import jsonref
from pydantic import BaseModel, Extra, Field, RootModel, root_validator


class LocationTypeEnum(str, Enum):
    Depot = "Depot"
    Pickup = "Pickup"
    Delivery = "Delivery"
    Break = "Break"
    w = "w"


class StrictModel(BaseModel):
    class Config:
        extra = Extra.forbid


class Objective(StrictModel):
    cost: Optional[float] = Field(
        default=None,
        examples=[1],
        description=(
            "dtype: float32. "
            " \n\n "
            "The weight assigned to minimizing the cost for a given solution, default value is 1"  # noqa
        ),
    )
    travel_time: Optional[float] = Field(
        default=None,
        examples=[0],
        description=(
            "dtype: float32."
            " \n\n "
            "The weight assigned to minimizing total travel time for a given solution (includes drive, service and wait time)"  # noqa
        ),
    )
    variance_route_size: Optional[float] = Field(
        default=None,
        examples=[0],
        description=(
            "dtype: float32."
            " \n\n "
            "The weight assigned to the variance in the number of orders served by each route."  # noqa
        ),
    )
    variance_route_service_time: Optional[float] = Field(
        default=None,
        examples=[0],
        description=(
            "dtype: float32."
            " \n\n "
            "The weight assigned to the variance in the accumulated service times of each route"  # noqa
        ),
    )
    prize: Optional[float] = Field(
        default=None,
        examples=[0],
        description=(
            "dtype: float32."
            " \n\n "
            "The weight assigned to the prize in accumulated prizes for each job fulfilled."  # noqa
            "This will be negated from overall values accumulated with other objectives."  # noqa
            "For example, if cost accumulated is 10 and objective value for it is 1, and if the prize accumulated is 3 and objective is 2, "  # noqa
            "then total cost would look something like this 10 x 1 - 3 x 2 = 4."  # noqa
            "Note: When this value is zero the prize objective is ignored."  # noqa
        ),
    )
    vehicle_fixed_cost: Optional[float] = Field(
        default=None,
        examples=[0],
        description=(
            "dtype: float32."
            " \n\n "
            "The weight assigned to the accumulated fixed costs of each vehicle used in solution"  # noqa
        ),
    )


class VehicleBreak(StrictModel):
    vehicle_id: int = Field(
        ...,
        description=(
            "dtype: int32, vehicle_id >= 0."
            " \n\n "
            "Vehicle id as an integer denoting the vehicle index for which the break is added"  # noqa
        ),
    )
    earliest: int = Field(
        ...,
        description=(
            "dtype: int32, earliest >= 0." " \n\n " "Earliest break time"
        ),
    )
    latest: int = Field(
        ...,
        description=("dtype: int32, latest > 0." " \n\n " "Latest break time"),
    )
    duration: int = Field(
        ...,
        description=(
            "dtype: int32, duration >= 0."
            " \n\n "
            "Duration of the break time"
        ),
    )
    locations: Optional[List[int]] = Field(
        ...,
        description=(
            "dtype: int32, location_id >= 0."
            " \n\n "
            "Location ids where this break can be taken."
        ),
    )


class VehicleOrderMatch(StrictModel):
    vehicle_id: int = Field(
        ...,
        description=(
            "dtype: int32, vehicle_id >= 0."
            " \n\n "
            "Vehicle id as an integer, and can serve all the "
            "order listed in order_ids."
        ),
    )
    order_ids: List[int] = Field(
        ...,
        description=(
            "dtype: int32, order_id >= 0."
            " \n\n "
            "Indices of orders which can be served by this particular vehicle"  # noqa
        ),
    )


class OrderVehicleMatch(StrictModel):
    order_id: int = Field(
        ...,
        description=(
            "dtype: int32, order_id >= 0."
            " \n\n "
            "Indices of orders which can be served by this particular vehicle"  # noqa
            "Order id as an integer"
        ),
    )
    vehicle_ids: List[int] = Field(
        ...,
        description=(
            "dtype: int32, vehicle_id >= 0."
            " \n\n "
            "Indices of the vehicles which can serve this particular order. \n"  # noqa
        ),
    )


class WaypointGraph(StrictModel):
    edges: List[int] = Field(
        ...,
        description=(
            "dtype: int32, edge >= 0."
            " \n\n "
            "Vertices of all the directed edges."
        ),
    )
    offsets: List[int] = Field(
        ...,
        description=(
            "dtype: int32, offset >= 0."
            " \n\n "
            "Offsets which provide number of edges from the source vertex signified by the index."  # noqa
        ),
    )
    weights: Optional[List[float]] = Field(
        default=None,
        description=(
            "dtype: float32, weight >= 0." " \n\n " "Weights of each edges."
        ),
    )


class WaypointGraphData(StrictModel):
    waypoint_graph: Optional[Dict[int, WaypointGraph]] = Field(default=None)


class WaypointGraphWeights(StrictModel):
    weights: Dict[int, List[float]] = Field(
        ...,
        description=(
            "dtype: float32, weight >= 0." " \n\n " "Weights of each edges"
        ),
    )


class CostMatrices(StrictModel):
    data: Optional[Dict[int, List[List[float]]]] = Field(
        default=None,
        description=(
            "dtype : vehicle-type (uint8), cost (float32), cost >= 0.\n"
            " \n\n "
            "Sqaure matrix with cost to travel from A to B and B to A. \n"
            "If there different types of vehicles which have different \n"
            "cost matrices, they can be provided with key value pair \n"
            "where key is vehicle-type and value is cost matrix. Value of \n"
            "vehicle type should be within [0, 255]"
        ),
    )


class FleetData(StrictModel):
    vehicle_locations: List[List[int]] = Field(
        ...,
        examples=[[[0, 0], [0, 0]]],
        description=(
            "dtype: int32, vehicle_location >= 0."
            " \n\n "
            "Start and end location of the vehicles in the given set of locations in WayPointGraph or CostMatrices.\n"  # noqa
            "Example: For 2 vehicles, "
            " \n\n "
            "    ["
            " \n\n "
            "        [veh_1_start_loc, veh_1_end_loc],"
            " \n\n "
            "        [veh_2_start_loc, veh_2_end_loc]"
            " \n\n "
            "    ]"
        ),
    )
    vehicle_ids: Optional[List[str]] = Field(
        default=None,
        examples=[["veh-1", "veh-2"]],
        description=("List of the vehicle ids or names provided as a string."),
    )
    capacities: Optional[List[List[int]]] = Field(
        default=None,
        examples=[[[2, 2], [4, 1]]],
        description=(
            "dtype: int32, capacity >= 0."
            " \n\n "
            "Note: For this release number of capacity dimensions are limited to 3."  # noqa
            " \n\n "
            "Lists of capacities of each vehicle.\n"
            "Multiple capacities can be added and each list will represent "
            "one kind of capacity. Order of kind of the capacities "
            "should match order of the demands.\n"
            "Total capacity for each type "
            "should be sufficient to complete all demand of that type."
            "Example: In case of two sets of capacities per vehicle with 3 vehicles, "  # noqa
            " \n\n "
            "    ["
            " \n\n "
            "        [cap_1_veh_1, cap_1_veh_2, cap_1_veh_3],"
            " \n\n "
            "        [cap_2_veh_1, cap_2_veh_2, cap_2_veh_3]"
            " \n\n "
            "    ]"
        ),
    )
    vehicle_time_windows: Optional[List[List[int]]] = Field(
        default=None,
        examples=[[[0, 10], [0, 10]]],
        description=(
            "dtype: int32, time >= 0."
            " \n\n "
            "Earliest and Latest time window pairs for each vehicle,\n"
            "for example the data would look as follows for 2 vehicles, \n"
            " \n\n "
            "    ["
            " \n\n "
            "        [veh_1_earliest, veh_1_latest],"
            " \n\n "
            "        [veh_2_earliest, veh_2_latest]"
            " \n\n "
            "    ]"
        ),
    )
    vehicle_break_time_windows: Optional[List[List[List[int]]]] = Field(
        default=None,
        examples=[[[[1, 2], [2, 3]]]],
        description=(
            "dtype: int32, time >= 0."
            " \n\n "
            "Multiple break time windows can be added for each vehicle."
            "Earliest and Latest break time window pairs for each vehicle,\n"
            "For example, in case of 2 sets of breaks for each vehicle which translates to 2 dimensions of breaks,\n"  # noqa
            " \n\n "
            "    ["
            " \n\n "
            "        [[brk_1_veh_1_earliest, brk_1_veh_1_latest], [brk_1_veh_2_earliest, brk_1_veh_2_latest]]"  # noqa
            " \n\n "
            "        [[brk_2_veh_1_earliest, brk_2_veh_1_latest], [brk_2_veh_2_earliest, brk_2_veh_2_latest]]"  # noqa
            " \n\n "
            "    ]"
            " \n\n "
            "The break duration within this time window is provided through "
            "vehicle_break_durations."
        ),
    )
    vehicle_break_durations: Optional[List[List[int]]] = Field(
        default=None,
        examples=[[[1, 1]]],
        description=(
            "dtype: int32, time >= 0."
            " \n\n "
            "Break duration for each vehicle. "
            "vehicle_break_time_windows should be provided to use this option."
            "For example, in case of having 2 breaks for each vehicle, "
            " \n\n "
            "    ["
            " \n\n "
            "        [brk_1_veh_1_duration, brk_1_veh_2_duration],"
            " \n\n "
            "        [brk_2_veh_1_duration, brk_2_veh_2_duration],"
            " \n\n "
            "    ]"
        ),
    )
    vehicle_break_locations: Optional[List[int]] = Field(
        default=None,
        examples=[[0, 1]],
        description=(
            "dtype: int32, location >= 0."
            " \n\n "
            "Break location where vehicles can take breaks. "
            "If not set, any location can be used for the break."
        ),
    )
    vehicle_breaks: Optional[List[VehicleBreak]] = Field(
        default=None,
        examples=[
            [
                {
                    "vehicle_id": 0,
                    "earliest": 0,
                    "latest": 10,
                    "duration": 2,
                    "locations": [2],
                },  # noqa
                {
                    "vehicle_id": 1,
                    "earliest": 10,
                    "latest": 15,
                    "duration": 3,
                    "locations": [3, 5],
                },  # noqa
                {
                    "vehicle_id": 1,
                    "earliest": 0,
                    "latest": 5,
                    "duration": 2,
                },  # noqa
            ]
        ],
        description=(
            "A list of Vehicle Breaks where vehicle id can take a break "
            "between earliest and latest time for specified duration "
            "in the specified locations. By default any location can "
            "be used."
        ),
    )
    vehicle_types: Optional[List[int]] = Field(
        default=None,
        examples=[[1, 2]],
        description=(
            "dtype: uint8."
            " \n\n "
            "Types of vehicles in the fleet given as positive integers."
        ),
    )
    vehicle_order_match: Optional[List[VehicleOrderMatch]] = Field(
        default=None,
        examples=[
            [
                {"vehicle_id": 0, "order_ids": [0]},
                {"vehicle_id": 1, "order_ids": [1]},
            ]
        ],
        description=(
            "A list of vehicle order match, where the match would contain "
            "a vehicle id and a list of orders that vehicle can serve."
        ),
    )
    skip_first_trips: Optional[List[bool]] = Field(
        default=None,
        examples=[[True, False]],
        description="Drop the cost of trip to first location for that vehicle.",  # noqa
    )
    drop_return_trips: Optional[List[bool]] = Field(
        default=None,
        examples=[[True, False]],
        description="Drop cost of return trip for each vehicle.",
    )
    min_vehicles: Optional[int] = Field(
        default=None,
        examples=[2],
        description=(
            "dtype: int32, min_vehicles >= 1."
            " \n\n "
            "Solution should consider minimum number of vehicles"  # noqa
        ),
    )
    vehicle_max_costs: Optional[List[float]] = Field(
        default=None,
        examples=[[7, 10]],
        description=(
            "dtype: float32, max_costs >= 0."
            " \n\n "
            "Maximum cost a vehicle can incur and it is based on cost matrix/cost waypoint graph."  # noqa
        ),
    )
    vehicle_max_times: Optional[List[float]] = Field(
        default=None,
        examples=[[7, 10]],
        description=(
            "dtype: float32, max_time >= 0."
            " \n\n "
            "Maximum time a vehicle can operate (includes drive, service and wait time), this is based on travel time matrix/travel time waypoint graph."  # noqa
        ),
    )
    vehicle_fixed_costs: Optional[List[float]] = Field(
        default=None,
        examples=[[15, 5]],
        description=(
            "dtype: float32, fixed_cost >= 0."
            " \n\n "
            "Cost of each vehicle."
            "This helps in routing where may be 2 vehicles with less cost "
            "is effective compared to 1 vehicle with huge cost. As example "
            "shows veh-0 (15) > veh-1 (5) + veh-2 (5)"
        ),
    )


class TaskData(StrictModel):
    task_locations: List[int] = Field(
        ...,
        examples=[[1, 2]],
        description=(
            "dtype: int32, location >= 0."
            " \n\n "
            "Location where the task has been requested."  # noqa
        ),
    )
    task_ids: Optional[List[str]] = Field(
        default=None,
        examples=[["Task-A", "Task-B"]],
        description=("List of the task ids or names provided as a string."),
    )
    demand: Optional[List[List[int]]] = Field(
        default=None,
        examples=[[[1, 1], [3, 1]]],
        description=(
            "dtype: int32"
            " \n\n "
            "Note: For this release number of demand dimensions are limited to 3."  # noqa
            " \n\n "
            "Lists of demands of each tasks.\n"
            "Multiple demands can be added and each list represents "
            "one kind of demand. Order of these demands should match the "
            "type of vehicle capacities provided."
            "Example: In case of two sets of demands per vehicle with 3 vehicles, "  # noqa
            " \n\n "
            "    ["
            " \n\n "
            "        [dem_1_tsk_1, dem_1_tsk_2, dem_1_tsk_3],"
            " \n\n "
            "        [dem_2_tsk_1, dem_2_tsk_2, dem_2_tsk_3]"
            " \n\n "
            "    ]"
        ),
    )
    pickup_and_delivery_pairs: Optional[List[List[int]]] = Field(
        default=None,
        examples=[None],
        description=(
            "dtype: int32, pairs >= 0."
            " \n\n "
            "List of Pick-up and delivery index pairs from task locations.\n"
            "In case we have the following pick-up and delivery locations, "
            "2->1, 4->5, 3->4, then task locations would look something like, "
            "task_locations = [0, 2, 1, 4, 5, 3, 4] and "
            "pick-up and delivery pairs would be index of those locations "
            "in task location and would look like "
            "[[1, 2], [3, 4], [5, 6]], 1 is pickup index for location 2 and "
            "it should be delivered to location 1 which is at index 2."
            "Example schema: "
            " \n\n "
            "    ["
            " \n\n "
            "        [pcikup_1_idx_to_task, drop_1_idx_to_task],"
            " \n\n "
            "        [pcikup_2_idx_to_task, drop_2_idx_to_task],"
            " \n\n "
            "    ]"
        ),
    )
    task_time_windows: Optional[List[List[int]]] = Field(
        default=None,
        examples=[[[0, 5], [3, 9]]],
        description=(
            "dtype: int32, time >= 0."
            " \n\n "
            "Earliest and Latest time windows for each tasks.\n"
            "For example the data would look as follows, \n"
            " \n\n "
            "    ["
            " \n\n "
            "        [tsk_1_earliest, tsk_1_latest],"
            " \n\n "
            "        [tsk_2_earliest, tsk_2_latest]"
            " \n\n "
            "    ]"
        ),
    )
    service_times: Optional[
        Union[List[int], Dict[int, List[int]]]
    ] = Field(  # noqa
        default=None,
        examples=[[0, 0]],
        description=(
            "dtype: int32, time >= 0."
            " \n\n "
            "Service time for each task. Accepts a list of service times for "
            "all vehicles. In case of vehicle specific service times, accepts "
            "a dict with key as vehicle id and value as list of service times."
            "Example schema: In case all vehicles have same service times, "
            " \n\n "
            "    [tsk_1_srv_time, tsk_2_srv_time, tsk_3_srv_time]"
            " \n\n "
            " \n\n "
            "In case, there are 2 vehicles and each of them have different service times,"  # noqa
            " \n\n "
            "    {"
            " \n\n "
            "        vehicle-id-1: [tsk_1_veh_1_srv_time, tsk_2_veh_1_srv_time, tsk_3_veh_1_srv_time],"  # noqa
            " \n\n "
            "        vehicle-id-2: [tsk_1_veh_2_srv_time, tsk_2_veh_2_srv_time, tsk_3_veh_2_srv_time],"  # noqa
            " \n\n "
            "    }"
        ),
    )
    prizes: Optional[List[float]] = Field(
        default=None,
        examples=[None],
        description=(
            "dtype: float32, prizes >= 0."
            " \n\n "
            "List of values which signifies prizes that are collected "
            "for fulfilling each task. This can be used effectively in case "
            "solution is infeasible and need to drop few tasks to get "
            "feasible solution. Solver will prioritize for higher prize tasks "
        ),
    )
    order_vehicle_match: Optional[List[OrderVehicleMatch]] = Field(
        default=None,
        examples=[
            [
                {"order_id": 0, "vehicle_ids": [0]},
                {"order_id": 1, "vehicle_ids": [1]},
            ]
        ],
        description=(
            "A list of order vehicle match, where the match would contain "
            "a order id and a list of vehicle ids that can serve this order."
        ),
    )


class SolverSettingsConfig(StrictModel):
    time_limit: Optional[float] = Field(
        default=None, examples=[5.0], description="SolverSettings time limit"
    )
    objectives: Optional[Objective] = Field(
        default=None,
        description=(
            "Values provided dictate the linear combination of factors used to evaluate solution quality."  # noqa
            "Only prize will be negated, all others gets accumulated. That's why sometime you might come across negative value as solution cost."  # noqa
        ),
    )
    config_file: Optional[str] = Field(
        default=None,
        examples=[None],
        description=("Dump configuration information in a given file as yaml"),
    )
    verbose_mode: Optional[bool] = Field(
        default=False,
        examples=[False],
        description=(
            "Displaying internal information during the solver execution."
        ),
    )
    error_logging: Optional[bool] = Field(
        default=True,
        examples=[True],
        description=(
            "Displaying constraint error information during the "
            "solver execution."
        ),
    )


class VehicleSolData(BaseModel):
    task_id: List[str] = Field(
        default=[],
        description=(
            "task_ids being assigned to vehicle along with depot and breaks"
        ),
    )
    type: List[LocationTypeEnum] = Field(
        default=[],
        description=(
            "Type of routing point, whether it is Depot, Waypoint - w \n"
            "Delivery, Break, Pickup \n"
        ),
    )


class InitialSolution(RootModel):
    root: Dict[str, VehicleSolData] = Field(
        default={},
        examples=[
            {
                "veh-1": {
                    "task_id": ["Break", "Task-A"],
                    "type": ["Break", "Delivery"],
                },
                "veh-2": {
                    "task_id": ["Depot", "Break", "Task-B", "Depot"],
                    "type": ["Depot", "Break", "Delivery", "Depot"],
                },
            }
        ],
        description=("Details of initial solution routes"),
    )


# Class holds Task, Fleet and Cost information for the service endpoint for
# Route optimization.


class OptimizedRoutingData(StrictModel):
    # We use a Union for top-level fields below instead of Optional
    # so that we can allow empty list (ie []) as a null value.
    # Pydantic v1 allowed this implicitly while v2 does not, and some legacy
    # cuopt data files have this, so we support it for backwards compat.
    # Any list that is not [] is screened out in the field_validator below.

    cost_waypoint_graph_data: Optional[WaypointGraphData] = Field(
        default=WaypointGraphData(),
        examples=[None],
        description=(
            "Waypoint graph with weights as cost to travel from A to B \n"
            "and B to A. If there are different types of vehicles \n"
            "they can be provided with key value pair \n"
            "where key is vehicle-type and value is the graph. Value of \n"
            "vehicle type should be within [0, 255]"
        ),
    )
    travel_time_waypoint_graph_data: Optional[WaypointGraphData] = Field(
        default=WaypointGraphData(),
        examples=[None],
        description=(
            "Waypoint graph with weights as time to travel from A to B \n"
            "and B to A. If there are different types of vehicles \n"
            "they can be provided with key value pair \n"
            "where key is vehicle-type and value is the graph. Value of \n"
            "vehicle type should be within [0, 255]"
        ),
    )
    cost_matrix_data: Optional[CostMatrices] = Field(
        default=CostMatrices(),
        examples=[
            {
                "cost_matrix": {
                    1: [[0, 1, 1], [1, 0, 1], [1, 1, 0]],
                    2: [[0, 1, 1], [1, 0, 1], [1, 2, 0]],
                }
            }
        ],
        description=(
            "Sqaure matrix with cost to travel from A to B and B to A. \n"
            "Cost is defined by user, it can be distance/fuel/time or \n"
            "a function of several factors."
            "If there are different types of vehicles which have different \n"
            "cost matrices, they can be provided with key value pair \n"
            "where key is vehicle-type and value is cost matrix. Value of \n"
            "vehicle type should be within [0, 255]"
        ),
    )
    travel_time_matrix_data: Optional[CostMatrices] = Field(
        default=CostMatrices(),
        examples=[
            {
                "cost_matrix": {
                    1: [[0, 1, 1], [1, 0, 1], [1, 1, 0]],
                    2: [[0, 1, 1], [1, 0, 1], [1, 2, 0]],
                }
            }
        ],
        description=(
            "Sqaure matrix with time to travel from A to B and B to A. \n"
            "If there are different types of vehicles which have different \n"
            "travel time matrices, they can be provided with key value pair \n"
            "where key is vehicle-type and value is time matrix. Value of \n"
            "vehicle type should be within [0, 255]"
        ),
    )
    fleet_data: FleetData = Field(..., description=("All Fleet information"))
    task_data: TaskData = Field(..., description=("All Task information"))
    initial_solution: Optional[List[InitialSolution]] = None
    solver_config: Optional[SolverSettingsConfig] = None

    # We need this validator to preserve backward compat for passing {}
    # for data in the local file case for get_routes
    # Without this the value must be None, which is a client change
    @root_validator(skip_on_failure=True, pre=True)
    def allow_empty_dict(cls, values):
        if values == {}:
            return {
                "travel_time_waypoint_graph_data": None,
                "cost_waypoint_graph_data": None,
                "travel_time_waypoint_graph_data": None,
                "cost_matrix_data": None,
                "travel_time_matrix_data": None,
                "fleet_data": FleetData(vehicle_locations=[]),
                "task_data": TaskData(task_locations=[]),
                "solver_config": None,
            }
        return values


class VehicleData(StrictModel):
    task_id: List[str] = Field(
        default=[],
        description=(
            "task_ids being assigned to vehicle along with depot and breaks"
        ),
    )
    arrival_stamp: List[float] = Field(
        default=[], description=("arrival stamps at each task locations")
    )
    route: List[int] = Field(
        default=[],
        description=(
            "Route indices as per waypoint graph or cost matrix provided"
        ),
    )
    type: List[LocationTypeEnum] = Field(
        default=[],
        description=(
            "Type of routing point, whether it is Depot, Waypoint - w \n"
            "Delivery, Break, Pickup \n"
        ),
    )


class DroppedTasks(StrictModel):
    task_id: Union[List[int], List[str]] = Field(
        default=[],
        description=(
            "With prize collection enabled, there is a chance of "
            "tasks being dropped to make a feasible solution. "
            "This list contains infeasible task ids which are dropped."
        ),
    )

    task_index: List[int] = Field(
        default=[],
        description=(
            "With prize collection enabled, there is a chance of "
            "tasks being dropped to make a feasible solution. "
            "This list contains infeasible task indices into task locations which are dropped."  # noqa
        ),
    )


class FeasibleResultData(StrictModel):
    status: int = Field(
        default=0,
        examples=[0],
        description=(
            "0 - Solution is available \n"
            "1 - Infeasible solution is available \n"
        ),
    )
    num_vehicles: int = Field(
        default=-1,
        examples=[2],
        description="Number of vehicle being used for the solution",
    )
    solution_cost: float = Field(
        default=-1.0, examples=[2], description="Total cost of the solution"
    )
    objective_values: Dict[str, float] = Field(
        default={},
        examples=[
            {
                "objective_values": {
                    "cost": 100.0,
                    "travel_time": 245.0,
                    "prize": -1000.0,
                }
            }
        ],
        description=("Individual objective values"),
    )
    vehicle_data: Dict[str, VehicleData] = Field(
        default={},
        examples=[
            {
                "vehicle_data": {
                    "veh-1": {
                        "task_id": ["Break", "Task-A"],
                        "arrival_stamp": [1, 2],
                        "route": [1, 1],
                        "type": ["Break", "Delivery"],
                    },
                    "veh-2": {
                        "task_id": ["Depot", "Break", "Task-B", "Depot"],
                        "arrival_stamp": [2, 2, 4, 5],
                        "route": [0, 0, 2, 0],
                        "type": ["Depot", "Break", "Delivery", "Depot"],
                    },
                }
            }
        ],
        description=("All the details of vehicle routes and timestamps"),
    )
    initial_solutions: List[str] = Field(
        default=[],
        description=(
            "Indicates whether each initial solution was accepted, not accepted or "  # noqa
            "not evaluated by the solver in case initial solutions were provided in request."  # noqa
        ),
    )
    dropped_tasks: DroppedTasks = Field(
        default=[],
        description=(
            "Contains details of dropped tasks when prize collection is enabled"  # noqa
        ),
    )
    msg: Optional[str] = Field(
        default="", description="Any information pertaining to the run."
    )


class InfeasibleResultData(StrictModel):
    status: int = Field(
        default=1,
        examples=[1],
        description=("1 - Infeasible solution is available \n"),
    )
    num_vehicles: int = Field(
        default=-1,
        examples=[2],
        description="Number of vehicle being used for the solution",
    )
    solution_cost: float = Field(
        default=-1.0, examples=[2], description="Total cost of the solution"
    )
    objective_values: Dict[str, float] = Field(
        default={},
        examples=[
            {
                "objective_values": {
                    "cost": 100.0,
                    "travel_time": 245.0,
                    "prize": -1000.0,
                }
            }
        ],
        description=("Individual objective values"),
    )
    vehicle_data: Dict[str, VehicleData] = Field(
        default={},
        examples=[
            {
                "vehicle_data": {
                    "veh-1": {
                        "task_id": ["Break", "Task-A"],
                        "arrival_stamp": [1, 2],
                        "route": [1, 1],
                        "type": ["Break", "Delivery"],
                    },
                    "veh-2": {
                        "task_id": ["Depot", "Break", "Task-B", "Depot"],
                        "arrival_stamp": [2, 2, 4, 5],
                        "route": [0, 0, 2, 0],
                        "type": ["Depot", "Break", "Delivery", "Depot"],
                    },
                }
            }
        ],
        description=("All the details of vehicle routes and timestamps"),
    )
    initial_solutions: List[str] = Field(
        default=[],
        description=(
            "Indicates whether each initial solution was accepted, not accepted or "  # noqa
            "not evaluated by the solver in case initial solutions were provided in request."  # noqa
        ),
    )
    dropped_tasks: DroppedTasks = Field(
        default=[],
        description=(
            "Note: This is just a place holder since there will not be any dropped tasks in infeasible solution."  # noqa
            "Contains details of dropped tasks when prize collection is enabled."  # noqa
        ),
    )
    msg: Optional[str] = Field(
        default="", description="Any information pertaining to the run."
    )


class FeasibleSolve(StrictModel):
    solver_response: FeasibleResultData = Field(
        default=FeasibleResultData(), description="Feasible solution"
    )
    perf_times: Optional[Dict] = Field(
        default=None, description=("Etl and Solve times of the solve call")
    )


class InFeasibleSolve(StrictModel):
    solver_infeasible_response: InfeasibleResultData = Field(
        default=InfeasibleResultData(),
        description=(
            "Infeasible solution, this can mean the problem itself is infeasible or "  # noqa
            "solver requires more time to find a solution. Setting default solve time is "  # noqa
            "suggested in case you are not aware of the expected time."
        ),
    )
    perf_times: Optional[Dict] = Field(
        default=None, description=("Etl and Solve times of the solve call")
    )


vrp_example_data = {
    "cost_waypoint_graph_data": None,
    "travel_time_waypoint_graph_data": None,
    "cost_matrix_data": {
        "data": {
            "1": [[0, 1, 1], [1, 0, 1], [1, 1, 0]],
            "2": [[0, 1, 1], [1, 0, 1], [1, 2, 0]],
        }
    },
    "travel_time_matrix_data": {
        "data": {
            "1": [[0, 1, 1], [1, 0, 1], [1, 1, 0]],
            "2": [[0, 1, 1], [1, 0, 1], [1, 2, 0]],
        }
    },
    "fleet_data": {
        "vehicle_locations": [[0, 0], [0, 0]],
        "vehicle_ids": ["veh-1", "veh-2"],
        "capacities": [[2, 2], [4, 1]],
        "vehicle_time_windows": [[0, 10], [0, 10]],
        "vehicle_break_time_windows": [[[1, 2], [2, 3]]],
        "vehicle_break_durations": [[1, 1]],
        "vehicle_break_locations": [0, 1],
        "vehicle_types": [1, 2],
        "vehicle_order_match": [
            {"order_ids": [0], "vehicle_id": 0},
            {"order_ids": [1], "vehicle_id": 1},
        ],
        "skip_first_trips": [True, False],
        "drop_return_trips": [True, False],
        "min_vehicles": 2,
        "vehicle_max_costs": [7, 10],
        "vehicle_max_times": [7, 10],
        "vehicle_fixed_costs": [15, 5],
    },
    "task_data": {
        "task_locations": [1, 2],
        "task_ids": ["Task-A", "Task-B"],
        "demand": [[1, 1], [3, 1]],
        "pickup_and_delivery_pairs": None,
        "task_time_windows": [[0, 5], [3, 9]],
        "service_times": [0, 0],
        "prizes": None,
        "order_vehicle_match": [
            {"order_id": 0, "vehicle_ids": [0]},
            {"order_id": 1, "vehicle_ids": [1]},
        ],
    },
    "solver_config": {
        "time_limit": 1,
        "objectives": {
            "cost": 1,
            "travel_time": 0,
            "variance_route_size": 0,
            "variance_route_service_time": 0,
            "prize": 0,
            "vehicle_fixed_cost": 0,
        },
        "config_file": None,
        "verbose_mode": False,
        "error_logging": True,
    },
}

# fmt: off
vrp_msgpack_example_data = "\x85\xb0cost_matrix_data\x81\xa4data\x82\xa11\x93\x93\x00\x01\x01\x93\x01\x00\x01\x93\x01\x01\x00\xa12\x93\x93\x00\x01\x01\x93\x01\x00\x01\x93\x01\x02\x00\xb7travel_time_matrix_data\x81\xa4data\x82\xa11\x93\x93\x00\x01\x01\x93\x01\x00\x01\x93\x01\x01\x00\xa12\x93\x93\x00\x01\x01\x93\x01\x00\x01\x93\x01\x02\x00\xaafleet_data\x8f\xb1vehicle_locations\x92\x92\x00\x00\x92\x00\x00\xabvehicle_ids\x92\xa5veh-1\xa5veh-2\xaacapacities\x92\x92\x02\x02\x92\x04\x01\xb4vehicle_time_windows\x92\x92\x00\n\x92\x00\n\xbavehicle_break_time_windows\x91\x92\x92\x01\x02\x92\x02\x03\xb7vehicle_break_durations\x91\x92\x01\x01\xb7vehicle_break_locations\x92\x00\x01\xadvehicle_types\x92\x01\x02\xb3vehicle_order_match\x92\x82\xa9order_ids\x91\x00\xaavehicle_id\x00\x82\xa9order_ids\x91\x01\xaavehicle_id\x01\xb0skip_first_trips\x92\xc3\xc2\xb1drop_return_trips\x92\xc3\xc2\xacmin_vehicles\x02\xb1vehicle_max_costs\x92\x07\n\xb1vehicle_max_times\x92\x07\n\xb3vehicle_fixed_costs\x92\x0f\x05\xa9task_data\x86\xaetask_locations\x92\x01\x02\xa8task_ids\x92\xa6Task-A\xa6Task-B\xa6demand\x92\x92\x01\x01\x92\x03\x01\xb1task_time_windows\x92\x92\x00\x05\x92\x03\t\xadservice_times\x92\x00\x00\xb3order_vehicle_match\x92\x82\xa8order_id\x00\xabvehicle_ids\x91\x00\x82\xa8order_id\x01\xabvehicle_ids\x91\x01\xadsolver_config\x84\xaatime_limit\x01\xaaobjectives\x86\xa4cost\x01\xabtravel_time\x00\xb3variance_route_size\x00\xbbvariance_route_service_time\x00\xa5prize\x00\xb2vehicle_fixed_cost\x00\xacverbose_mode\xc2\xaderror_logging\xc3".encode("unicode_escape") # noqa
# fmt: on


managed_vrp_example_data = {
    "action": "cuOpt_OptimizedRouting",
    "data": vrp_example_data,
    "client_version": "25.08",
}

# cut and pasted from actual run of VRP example data.
# don't reformat :)
vrp_response = {
    "value": {
        "response": {
            "solver_response": {
                "status": 0,
                "num_vehicles": 2,
                "solution_cost": 2.0,
                "objective_values": {"cost": 2.0},
                "vehicle_data": {
                    "veh-1": {
                        "task_id": ["Break", "Task-A"],
                        "arrival_stamp": [1.0, 2.0],
                        "type": ["Break", "Delivery"],
                        "route": [1, 1],
                    },
                    "veh-2": {
                        "task_id": ["Depot", "Break", "Task-B", "Depot"],
                        "arrival_stamp": [2.0, 2.0, 4.0, 5.0],
                        "type": ["Depot", "Break", "Delivery", "Depot"],
                        "route": [0, 0, 2, 0],
                    },
                },
                "dropped_tasks": {"task_id": [], "task_index": []},
            }
        },
        "reqId": "e8421e9e-e42e-4511-8da2-314253667dcf",
    }  # noqa
}

managed_vrp_response = copy.deepcopy(vrp_response)
del managed_vrp_response["value"]["reqId"]

reqId_response = {"value": {"reqId": "e8421e9e-e42e-4511-8da2-314253667dcf"}}

vrpschema = jsonref.loads(
    json.dumps(OptimizedRoutingData.model_json_schema()), proxies=False
)
del vrpschema["$defs"]
