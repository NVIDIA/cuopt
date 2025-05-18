# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.  # noqa
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

from enum import Enum

from cuopt.utilities import catch_cuopt_exception


class SolutionStatus(Enum):
    SUCCESS = 0
    FAIL = 1
    TIMEOUT = 2
    EMPTY = 3


class Assignment:
    """
    A container of vehicle routing solver output

    Parameters
    ----------
    vehicle_count : Integer
        Number of vehicles in the solution
    total_objective_value : Float64
        Objective value calculated as per objective functions and weights
    objective_values : dict[Objective, Float64]
    route_df: cudf.DataFrame
        Contains route, vehicle_id, arrival_stamp.
    accepted: cudf.Series
    status: Integer
        Solver status 0 - SUCCESS, 1 - FAIL, 2 - TIMEOUT and 3 - EMPTY.
    message: String
        Any error message if there is failure
    undeliverable_orders: cudf.Series
        Orders which can not be served
    """

    @catch_cuopt_exception
    def __init__(
        self,
        vehicle_count,
        total_objective_value,
        objective_values,
        route_df,
        accepted,
        status,
        message,
        error_status,
        error_message,
        undeliverable_orders,
    ):
        self.vehicle_count = vehicle_count
        self.total_objective_value = total_objective_value
        self.objective_values = objective_values
        self.route = route_df
        self.total_objective_value = total_objective_value
        self.accepted = accepted
        self.status = status
        self.message = message
        self.error_status = error_status
        self.error_message = error_message
        self.undeliverable_orders = undeliverable_orders

    @catch_cuopt_exception
    def get_vehicle_count(self):
        """
        Returns the number of vehicle needed for this routing assignment.
        """
        return self.vehicle_count

    @catch_cuopt_exception
    def get_total_objective(self):
        """
        Returns the objective value calculated based on the user
        provided objective function and the routes found by the solver.
        """
        return self.total_objective_value

    @catch_cuopt_exception
    def get_objective_values(self):
        """
        Returns the individual objective_values as dictionary
        """
        return self.objective_values

    @catch_cuopt_exception
    def get_route(self):
        """
        Returns the route, truck ids for each stop and the arrival stamp
        as cudf.DataFrame.

        Examples
        --------
        >>> import cuopt
        >>> import cudf
        >>> locations = [0, 1, 2, 3]
        >>> vehicles  = [0, 1]
        >>> cost_mat  = [
        ...   [0, 1, 5, 2],
        ...   [2, 0, 7, 4],
        ...   [1, 5, 0, 9],
        ...   [5, 6, 2, 0]
        ... ]
        >>> cost_mat = cudf.DataFrame(cost_mat)
        >>> cost_mat
           0  1  2  3
        0  0  1  5  2
        1  2  0  7  4
        2  1  5  0  9
        3  5  6  2  0
        >>> data_model = routing.DataModel(len(locations), len(vehicles))
        >>> data_model.set_matrix(cost_mat)
        >>> solver = cuopt.Solver(data_model)
        >>> solver.set_min_vehicles(len(vehicles))
        >>> solver.set_min_vehicles(len(vehicles))
        >>> solution = solver.solve()
        >>> solution.get_route()
        >>> solution.get_route()
           route  arrival_stamp  truck_id  location
        0      0            0.0         1         0
        1      3            2.0         1         3
        2      2            4.0         1         2
        3      0            5.0         1         0
        4      0            0.0         0         0
        5      1            1.0         0         1
        6      0            3.0         0         0
        """
        return self.route

    @catch_cuopt_exception
    def get_status(self):
        """
        Returns the final solver status as per SolutionStatus.
        """
        return self.status

    @catch_cuopt_exception
    def get_message(self):
        """
        Returns the final solver message as per SolutionStatus.
        """
        return self.message

    @catch_cuopt_exception
    def get_error_status(self):
        """
        Returns the error status as per ErrorStatus.
        """
        return self.error_status

    @catch_cuopt_exception
    def get_error_message(self):
        """
        Returns the error message as per ErrorMessage.
        """
        return self.error_message

    @catch_cuopt_exception
    def get_infeasible_orders(self):
        """
        Returns the infeasible order numbers as cudf.Series.
        """
        return self.undeliverable_orders

    @catch_cuopt_exception
    def get_accepted_solutions(self):
        """ """
        return self.accepted

    @catch_cuopt_exception
    def display_routes(self):
        """
        Display the solution in human readable format.
        Intended for relatively small inputs.

        Examples
        --------
        >>> import cuopt
        >>> import cudf
        >>> locations = [0, 1, 2, 3]
        >>> vehicles  = [0, 1]
        >>> cost_mat  = [
        ...   [0, 1, 5, 2],
        ...   [2, 0, 7, 4],
        ...   [1, 5, 0, 9],
        ...   [5, 6, 2, 0]
        ... ]
        >>> cost_mat = cudf.DataFrame(cost_mat)
        >>> cost_mat
           0  1  2  3
        0  0  1  5  2
        1  2  0  7  4
        2  1  5  0  9
        3  5  6  2  0
        >>> data_model = routing.DataModel(len(locations), len(vehicles))
        >>> data_model.set_matrix(cost_mat)
        >>> solver = cuopt.Solver(data_model)
        >>> solver.set_min_vehicles(len(vehicles))
        >>> solver.set_min_vehicles(len(vehicles))
        >>> solution = solver.solve()
        >>> solution.display_routes()
        Vehicle-0 starts at: 0.0, completes at: 3.0, travel time: 3.0,  Route :
        0->1->0
        Vehicle-1 starts at: 0.0, completes at: 5.0, travel time: 5.0,  Route :
        0->3->2->0
        This results in a travel time of 8.0 to deliver all routes
        """
        solution_cudf = self.route

        total_times = []
        for i, assign in enumerate(
            solution_cudf["truck_id"].unique().to_arrow().to_pylist()
        ):
            solution_vehicle_x = solution_cudf[
                solution_cudf["truck_id"] == assign
            ]
            vehicle_x_start_time = round(
                float(solution_vehicle_x["arrival_stamp"].min()), 2
            )
            vehicle_x_final_time = round(
                float(solution_vehicle_x["arrival_stamp"].max()), 2
            )
            vehicle_x_total_time = round(
                vehicle_x_final_time - vehicle_x_start_time, 2
            )
            total_times.append(vehicle_x_total_time)
            print(
                "Vehicle-{} starts at: {}, completes at: {}, travel time: {}, ".format(  # noqa
                    assign,
                    vehicle_x_start_time,
                    vehicle_x_final_time,
                    vehicle_x_total_time,
                ),
                "Route : ",
            )

            route_string = ""
            if "type" in solution_cudf.columns:
                loc_ids = solution_vehicle_x["route"].to_arrow().to_pylist()
                loc_types = solution_vehicle_x["type"].to_arrow().to_pylist()
                for stop, t in zip(loc_ids, loc_types):
                    if stop is not None:
                        if t == "Depot":
                            loc_type = "Dpt"
                        else:
                            loc_type = t[0]
                        route_string += str(stop) + "(" + loc_type + ")" + "->"
                print(route_string[:-2], "\n")
            else:
                for stop in solution_vehicle_x["route"].to_arrow().to_pylist():
                    if stop is not None:
                        route_string += str(stop) + "->"
                print(route_string[:-2], "\n")

        print(
            "This results in a travel time of {} to deliver all routes\n\n".format(  # noqa
                sum(total_times)
            )
        )
