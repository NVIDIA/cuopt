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

import numpy as np

import cudf

from cuopt import routing
from cuopt.routing import vehicle_routing_wrapper
from cuopt.utilities import catch_cuopt_exception

from .validation import (
    validate_matrix,
    validate_non_negative,
    validate_positive,
    validate_range,
    validate_size,
    validate_time_windows,
)


class DataModel(vehicle_routing_wrapper.DataModel):
    """

    DataModel(n_locations, n_fleet, n_orders: int = -1, session_id=None)

    Initialize a Data Model.

    Parameters
    ----------
    n_locations : Integer
        number of locations to visit, including vehicle/technician location.
    n_fleet : Integer
        number of vehicles/technician in the fleet.
    n_orders : Integer
        number of orders.
    session_id : Integer
        This is used with dask for Multi GPU scenario.

    Note:
      - A cost matrix must be set before passing
        this object to the solver.

      - If vehicle locations is not set, then by default 0th index in
        cost/transit time matrix, time windows, capacity dimension,
        order location is considered as start and end location of all the
        vehicles.

    Examples
    --------
    >>> from cuopt import routing
    >>> locations = [0, 1, 2, 3, 4, 5, 6]
    >>> vehicles  = [0, 1, 2, 3]
    >>> data_model = routing.DataModel(len(locations), len(vehicles))
    """

    @catch_cuopt_exception
    def __init__(
        self,
        n_locations,
        n_fleet,
        n_orders: int = -1,
        session_id=None,
    ):
        super().__init__(
            n_locations, n_fleet, n_orders=n_orders, session_id=session_id
        )

    @catch_cuopt_exception
    def add_cost_matrix(self, cost_mat, vehicle_type=0):
        """
        Add a matrix for all locations (vehicle/technician locations included)
        at once.

        A cost matrix is a square matrix containing the cost of travel which
        can be distance, time or any other metric, taken pairwise, between all
        locations. Diagonal elements should be 0.

        This cost matrix will be used to find the routes through all the
        locations.
        The user can call add_cost_matrix multiple times. Setting the
        vehicle type will enable heterogeneous fleet. It can model traveling
        distances for different vehicles (bicycles, bikes, trucks).

        Note:
          - If vehicle locations is not set, then by default 0th index
            and column are considered start and end location for
            all vehicles.

        Parameters
        ----------
        cost_mat : cudf.DataFrame dtype - float32
            cudf.DataFrame representing floating point square matrix with
            num_location rows and columns.
        vehicle_type : uint8
            Identifier of the vehicle.

        Examples
        --------
        >>> from cuopt import routing
        >>> import cudf
        >>> locations = [0, 1, 2, 3]
        >>> vehicles  = [0, 1]
        >>> cost_mat_bikes  = [
        ...   [0, 1, 5, 2],
        ...   [2, 0, 7, 4],
        ...   [1, 5, 0, 9],
        ...   [5, 6, 2, 0]
        ... ]
        >>> cost_mat_bikes = cudf.DataFrame(cost_mat_bikes)
        >>> cost_mat_bikes
           0  1  2  3
        0  0  1  5  2
        1  2  0  7  4
        2  1  5  0  9
        3  5  6  2  0
        >>> data_model = routing.DataModel(len(locations), len(vehicles))
        >>> data_model.add_cost_matrix(cost_mat_bikes, 1)
        >>> cost_mat_car  = [
        ...   [0, 1, 2, 1],
        ...   [1, 0, 3, 2],
        ...   [1, 2, 0, 3],
        ...   [1, 3, 9, 0]
        ... ]
        >>> cost_mat_car = cudf.DataFrame(cost_mat_bikes)
        >>> cost_mat_car
           0  1  2  3
        0  0  1  2  1
        1  1  0  3  2
        2  1  2  0  3
        3  1  3  9  0
        >>> data_model.add_cost_matrix(cost_mat_car, 2)
        """

        if vehicle_type in self.costs:
            raise ValueError("Vehicle type matrix has already been added")

        validate_matrix(cost_mat, "cost matrix", self.get_num_locations())

        super().add_cost_matrix(cost_mat, vehicle_type)

    @catch_cuopt_exception
    def add_transit_time_matrix(self, mat, vehicle_type=0):
        """
        Add transit time matrix for all locations
        (vehicle/technician locations included) at once.

        This matrix is used to check constraints satisfiability rather
        than participating in cost optimization.

        For instance, this matrix can be used to model the time to
        travel between locations with time windows referring to it while the
        solver could optimize for cost/distance. A transit time matrix is
        defined as a square matrix containing the cost, taken pairwise,
        between all locations.
        Users should pre-compute time between each pair of locations
        with their own technique before calling this function. Entries in
        this matrix could represent time, miles, meters or any metric that
        can be stored as a real number and satisfies the property above.

        The user can call add_transit_time_matrix multiple times. Setting the
        vehicle type will enable heterogeneous fleet. It can model traveling
        speeds for different vehicles (bicycles, bikes, trucks).

        Time windows specified in set_order_time_windows will validate the time
        to travel with secondary matrix if it is available, else primary matrix
        is used to validate the constraint.

        Note:
          - The values provided are considered as units and it is user's
            responsibility to ensure all time related entries are normalized
            to one common unit (hours/minutes/seconds/any).

          - If vehicle locations is not set, then by default 0th index
            and column are considered start and end location for
            all vehicles.

        Parameters
        ----------
        mat : cudf.DataFrame dtype - float32
            cudf.DataFrame representing floating point square matrix with
            num_location rows and columns.
        vehicle_type : uint8
            Identifier of the vehicle.

        Examples
        --------
        >>> from cuopt import routing
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
        >>> data_model.add_cost_matrix(cost_mat, 0)
        >>> time_mat = [
        ...   [0, 10, 50, 20],
        ...   [20, 0, 70, 40],
        ...   [10, 50, 0, 90],
        ...   [50, 60, 20, 0]
        ... ]
        >>> time_mat = cudf.DataFrame(time_mat)
        >>> data_model.add_transit_time_matrix(time_mat, 0)
        """
        if vehicle_type in self.transit_times:
            raise ValueError("Vehicle type matrix has already been added")

        validate_matrix(mat, "transit time matrix", self.get_num_locations())

        super().add_transit_time_matrix(mat, vehicle_type)

    @catch_cuopt_exception
    def set_break_locations(self, break_locations):
        """
        The vehicle is allowed to stop at specific locations during a break.
        It can be at a customer node or another location representing for
        instance a gas station.
        The solver will pick the best stop out of all break nodes.
        The same break node can appear on several routes and satisfy
        multiple break constraints.

        Note: If the break locations are not set, every location can
        be used as a break location

        Parameters
        ----------
        break_locations: cudf.Series dtype-int32
            representing the designated locations that can be used for breaks.
            The break locations should be numbered
            in between 0 and nlocations - 1.

        Examples
        --------
        >>> from cuopt import routing
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
        >>> data_model.add_cost_matrix(cost_mat)
        >>> data_model.set_break_locations(cudf.Series([1, 3]))
        """
        validate_range(
            break_locations, "break_locations", 0, self.get_num_locations()
        )
        super().set_break_locations(break_locations)

    @catch_cuopt_exception
    def add_break_dimension(
        self, break_earliest, break_latest, break_duration
    ):
        """
        Add break time windows to model the Vehicle Routing Problem with Time
        Windows (VRPTW).
        The vehicles have break time windows within which
        the breaks must be taken. And multiple breaks can be added
        using the same api as another dimension, check the example.

        Note: The values provided are considered as units and it is user's
        responsibility to ensure all time related entries are normalized to
        one common unit (hours/minutes/seconds/any).

        Note: This function cannot be used in conjuction with add_vehicle_break

        Parameters
        ----------
        break_earliest: cudf.Series dtype - int32
            Earliest time a vehicle can be at a break location.
        break_latest: cudf.Series dtype - int32
            Latest time a vehicle can be at a break location.
        break_duration: cudf.Series dtype - int32
            Time spent at the break location, internally equivalent
            to service time.

        Examples
        --------
        >>> from cuopt import routing
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
        >>> data_model.add_cost_matrix(cost_mat)
        >>> time_mat = [
        ...   [0, 10, 50, 20],
        ...   [20, 0, 70, 40],
        ...   [10, 50, 0, 90],
        ...   [50, 60, 20, 0]
        ... ]
        >>> time_mat = cudf.DataFrame(time_mat)
        >>> data_model.add_transit_time_matrix(time_mat)
        >>> # Considering vehicles need to take two breaks
        >>> lunch_break_earliest = [20, 25]
        >>> lunch_break_latest   = [40, 45]
        >>> lunch_break_service  = [5,   5]
        >>> data_model.add_break_dimension(
        ...   cudf.Series(lunch_break_earliest),
        ...   cudf.Series(lunch_break_latest),
        ...   cudf.Series(lunch_break_service)
        ... )
        >>> snack_break_earliest = [40, 45]
        >>> snack_break_latest   = [60, 65]
        >>> snack_break_service  = [5,   5]
        >>> data_model.add_break_dimension(
        ...   cudf.Series(snack_break_earliest),
        ...   cudf.Series(snack_break_latest),
        ...   cudf.Series(snack_break_service)
        """

        n_fleet = self.get_fleet_size()
        validate_size(
            break_duration, "break duration", n_fleet, "number of vehicles"
        )
        validate_time_windows(
            break_earliest, break_latest, n_fleet, "number of vehicles"
        )
        super().add_break_dimension(
            break_earliest, break_latest, break_duration
        )

    @catch_cuopt_exception
    def add_vehicle_break(
        self, vehicle_id, earliest, latest, duration, locations=cudf.Series()
    ):
        """
        Specify a break for a given vehicle. Use this api to specify
        non-homogenous breaks. For example, different number of breaks can be
        speficied for each vehicle by calling this function different number of
        times for each vehicle. Furthermore, this function provides more
        flexibility in specifying locations for each break.

        Note: This function cannot be used in conjection with
        add_break_dimension

        Parameters
        ----------
        vehicle_id: integer
            Vehicle Id for which the break is being specified
        earliest:  integer
            Earliest time the vehicle can start the break
        latest:    integer
            Latest time the vehicle can start the break
        duration:  ingteger
            Time spent at the break location
        locations: cudf.Series dtype - int32
            List of locations where this break can be taken. By default
            any location can be used

        Examples
        --------
        >>> from cuopt import routing
        >>> vehicle_num = 2
        >>> d = routing.DataModel(nodes, vehicle_num)
        >>> d.add_vehicle_break(0, 10, 20, 5, cudf.Series([3, 6, 8]))
        >>> d.add_vehicle_break(0, 60, 70, 5, cudf.Series([1, 4, 7]))
        >>> d.add_vehicle_break(1, 30, 40, 5)
        """
        validate_range(vehicle_id, "vehicle id", 0, self.get_fleet_size())
        if len(locations) > 0:
            validate_range(
                locations, "break locations", 0, self.get_num_locations()
            )

        super().add_vehicle_break(
            vehicle_id, earliest, latest, duration, locations
        )

    @catch_cuopt_exception
    def set_objective_function(self, objectives, objective_weights):
        """
        The objective function can be defined as a linear combination of
        the different objectives. Solver optimizes for vehicle
        count first and then the total objective. The default value of
        1 is used for COST objective weight and 0 for other objective weights

        Parameters
        ----------
        objectives : cudf.Series dtype - cuopt.routing.Objective
            Series of Objective criteria
        objective_weights : cudf.Series dtype - float32
            Series to the weighs associated with the objectives.
            Series will be cast to float32.

        Examples
        --------
        >>> from cuopt import routing
        >>> d = routing.DataModel(nodes, vehicle_num)
        >>> d.set_objective_function(
        >>> cudf.Series([routing.Objective.PRIZE, routing.Objective.COST]),
        >>>             cudf.Series([2**32, 1]))
        """
        validate_size(
            objectives, "objectives", objective_weights, "objective weights"
        )

        super().set_objective_function(objectives, objective_weights)

    def add_initial_solutions(self, vehicle_ids, routes, types, sol_offsets):
        """ """
        validate_size(vehicle_ids, "vehicle_ids", routes, "routes")
        validate_size(routes, "routes", types, "types")
        super().add_initial_solutions(vehicle_ids, routes, types, sol_offsets)

    @catch_cuopt_exception
    def set_order_locations(self, order_locations):
        """
        Set a location for each order.

        This allows the cases with multiple orders per locations run
        efficiently.
        Consider an example with 4 locations and 10 orders serving to these 4
        locations the order_locations series can look like:
        [0, 2, 3, 1, 3, 1, 2, 1, 3, 2]. In this case, the ith entry in
        the series represents the location id of the ith order. Using this,
        the distance matrix is represented as size 4x4 instead of 10x10.

        Parameters
        ----------
        order_locations : cudf.Series dtype - int32
            cudf.Series representing location id of each order
            given as positive integers

        Examples
        --------
        >>> from cuopt import routing
        >>> import cudf
        >>> locations = [0, 1, 2, 3]
        >>> vehicles  = [0, 1]
        >>> orders    = [0, 2, 3, 1, 3, 1, 2, 1, 3, 2]
        >>> data_model = routing.DataModel(
        ...   len(locations),
        ...   len(vehicles),
        ...   len(orders)
        ... )
        >>> data_model.set_order_locations(cudf.Series(orders))
        """
        validate_size(
            order_locations,
            "order locations",
            self.get_num_orders(),
            "number of orders",
        )
        validate_range(
            order_locations, "order locations", 0, self.get_num_locations()
        )
        super().set_order_locations(order_locations)

    @catch_cuopt_exception
    def set_vehicle_types(self, vehicle_types):
        """
        Set vehicle types in the fleet.

        When multiple matrices are given as input the solver
        is enabling heterogeneous cost matrix and time matrix
        optimization. We thus need the corresponding vehicle
        type id for all vehicles in the data model.

        Parameters
        ----------
        vehicle_types : cudf.Series dtype - uint8
            cudf.Series representing types of vehicles in
            the fleet given as positive integers.

        Examples
        --------
        >>> from cuopt import routing
        >>> import cudf
        >>> locations = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        >>> vehicles     = [0, 1, 2, 3, 4]
        >>> vehicle_tpes = [0, 1, 1, 0, 0] # 0 - Car 1 - bike
        >>> data_model = routing.DataModel(
        ...   len(locations),
        ...   len(vehicles),
        ... )
        >>> data_model.set_vehicle_types(cudf.Series(vehicle_types))
        """
        validate_size(
            vehicle_types,
            "vehicle types",
            self.get_fleet_size(),
            "number of vehicles",
        )
        validate_non_negative(vehicle_types, "vehicle types")
        super().set_vehicle_types(vehicle_types)

    @catch_cuopt_exception
    def set_pickup_delivery_pairs(self, pickup_indices, delivery_indices):
        """
        Set pick-up delivery pairs given by indices to the orders.

        Currently mixed pickup and delivery is not supported, meaning that all
        the orders should be a included in the pick-up delivery pair indices.
        These indices are indices to order locations set using
        set_order_locations.

        Parameters
        ----------
        pickup_indices : cudf.Series dtype - int32
            int cudf.Series representing the indices of pickup orders.
        delivery_indices : cudf.Series dtype - int32
            int cudf.Series representing the indices of delivery orders.

        Examples
        --------
        >>> from cuopt import routing
        >>> import cudf
        >>> locations = [0, 1, 2, 3, 4]
        >>> vehicles  = [0, 1]
        >>> order_locations = [2, 1, 3, 4, 1, 4]
        >>> pickup_indices   = [0, 2, 4]
        >>> delivery_indices = [1, 3, 5] # 2 -> 1, 3 -> 4 and 1->4
        >>> data_model = routing.DataModel(
        ...   len(locations),
        ...   len(vehicles),
        ... )
        >>> data_model.set_order_locations(order_locations)
        >>> data_model.set_pickup_delivery_pairs(
        ...   cudf.Series(pickup_indices),
        ...   cudf.Series(delivery_indices)
        ... )
        """
        super().set_pickup_delivery_pairs(pickup_indices, delivery_indices)

    @catch_cuopt_exception
    def set_vehicle_time_windows(self, earliest_time, latest_time):
        """
        Set vehicle time windows in the fleet.

        The earliest time is the time vehicle can leave the starting location.
        The latest time is the time vehicle must be free.
        In case of drop_return_trip, latest time specifies the service end
        time with the last customer.
        The size of this array must be equal to fleet_size.

        Note: The values provided are considered as units and it is user's
        responsibility to ensure all time related entries are normalized to
        one common unit (hours/minutes/seconds/any).

        This would help users to solve for routes which consider
        vehicle availability time window for each vehicle.
        If secondary matrix has been set using add_transit_time_matrix,
        then that will be used for time validation,
        else primary matrix is used.

        Parameters
        ----------
        earliest_time : cudf.Series dtype - int32
            cudf.Series representing earliest available times of vehicles
        latest_time : cudf.Series dtype - int32
            cudf.Series representing latest time vehicle must be free.

        Examples
        --------
        >>> from cuopt import routing
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
        >>> data_model.add_cost_matrix(cost_mat)
        >>> time_mat = [
        ...   [0, 10, 50, 20],
        ...   [20, 0, 70, 40],
        ...   [10, 50, 0, 90],
        ...   [50, 60, 20, 0]
        ... ]
        >>> time_mat = cudf.DataFrame(time_mat)
        >>> data_model.add_transit_time_matrix(time_mat)
        >>> veh_earliest = [  0,  20] # earliest a vehicle/tech start
        >>> veh_latest   = [200, 180] # end of the vehicle/tech shift
        >>> data_model.set_vehicle_time_windows(
        ...   cudf.Series(veh_earliest),
        ...   cudf.Series(veh_latest),
        ... )

        """
        validate_time_windows(
            earliest_time,
            latest_time,
            self.get_fleet_size(),
            "number of vehicles",
        )

        super().set_vehicle_time_windows(earliest_time, latest_time)

    @catch_cuopt_exception
    def set_vehicle_locations(self, start_locations, return_locations):
        """
        Set start and return locations for vehicles in the fleet.

        The start location is a point of start for that vehicle, and
        return location is designated return location for that vehicle.
        These can be depot, home or any other locations.
        The size of these arrays must be equal to fleet_size. When these
        arrays are not set, all the vehicles in the fleet are assumed
        to be starting from and returning to depot location, which is
        zero indexed location.

        Parameters
        ----------
        start_locations  : cudf.Series dtype - int32
            cudf.Series representing starting locations of vehicles
        return_locations : cudf.Series dtype - int32
            cudf.Series representing return locations of vehicles

        Examples
        --------
        >>> from cuopt import routing
        >>> from cuopt import routing
        >>> import cudf
        >>> locations = [0, 1, 2, 3]
        >>> vehicles  = [0, 1]
        >>> vehicle_start_location = [0, 0]
        >>> vehicle_end_location   = [2, 3]
        >>> data_model = routing.DataModel(len(locations), len(vehicles))
        >>> data_model.set_vehicle_locations(
        ...   cudf.Series(vehicle_start_location),
        ...   cudf.Series(vehicle_end_location)
        ... )
        """

        validate_size(
            start_locations,
            "start locations",
            self.get_fleet_size(),
            "number of locations",
        )
        validate_size(
            return_locations,
            "return locations",
            self.get_fleet_size(),
            "number of locations",
        )
        validate_non_negative(start_locations, "start locations")
        validate_non_negative(return_locations, "return locations")

        super().set_vehicle_locations(start_locations, return_locations)

    @catch_cuopt_exception
    def set_order_time_windows(self, earliest, latest):
        """
        Add order time windows to model the Vehicle Routing Problem
        with Time Windows (VRPTW)

        The locations have time windows within which the visits must be made.
        If transit time matrix has been set using add_transit_time_matrix,
        then that will be used to validate time windows,
        else primary matrix is used.

        Note:
          - The values provided are considered as units and it is user's
            responsibility to ensure all time related entries are normalized to
            one common unit (hours/minutes/seconds/any).

          - If vehicle locations is not set, then by default 0th index
            in all columns are considered start and end location for
            all vehicles. So may be you need to provide big time window
            for completion of all jobs/depot time window with may be with
            service time to be 0.

        Parameters
        ----------
        earliest : cudf.Series dtype - int32
            cudf.Series containing the earliest visit time for each location
            including the depot. Order is implicit and should be consistent
            with the data model.
        latest : cudf.Series dtype - int32
            cudf.Series containing the latest visit time for each location
            including the depot. Order is implicit and should be consistent
            with the data model.

        Examples
        --------
        >>> from cuopt import routing
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
        >>> data_model.add_cost_matrix(cost_mat)
        >>> time_mat = [
        ...   [0, 10, 50, 20],
        ...   [20, 0, 70, 40],
        ...   [10, 50, 0, 90],
        ...   [50, 60, 20, 0]
        ... ]
        >>> time_mat = cudf.DataFrame(time_mat)
        >>> data_model.add_transit_time_matrix(time_mat)
        >>> earliest = [  0,  15,  60,   0] # earliest a job can be started
        >>> latest   = [500, 180, 150, 180] # latest a job can be started
        >>> data_model.set_order_time_windows(
        ...   cudf.Series(earliest),
        ...   cudf.Series(latest)
        ... )
        """
        n_orders = self.get_num_orders()

        validate_time_windows(earliest, latest, n_orders, "number of orders")

        super().set_order_time_windows(earliest, latest)

    @catch_cuopt_exception
    def set_order_prizes(self, prizes):
        """
        Set prizes for orders

        Parameters
        ----------
        prizes : cudf.Series dtype - float32
            cudf.Series containing prizes for each order including
            the depot (if depot is included in the order list). Order is
            implicit and should be consistent with the data model.
            Size of this series must be equal to num_orders in data model.

        Examples
        --------
        >>> from cuopt import routing
        >>> import cudf
        >>> locations = [0, 1, 2, 3]
        >>> vehicles  = [0, 1]
        >>> prizes = [20, 10, 0, 30]
        >>> data_model.set_order_prizes(cudf.Series(prizes))
        """

        validate_size(
            prizes, "prizes", self.get_num_orders(), "number of orders"
        )

        super().set_order_prizes(prizes)

    @catch_cuopt_exception
    def set_drop_return_trips(self, set_drop_return_trips):
        """
        Control if individual vehicles in the fleet return to the
        end location after the last stop.

        End location is where vehicles will return after completing
        all the tasks assigned.

        Parameters
        ----------
        set_drop_return_trips : cudf.Series dtype - bool
            Set True to the drop return trip to end location for each vehicle.

        Examples
        --------
        >>> from cuopt import routing
        >>> import cudf
        >>> locations   = [0, 1, 2, 3]
        >>> vehicles    = [   0,     1]
        >>> drop_return = [True, False] # Drop the return for the first vehicle
        >>> data_model = routing.DataModel(len(locations), len(vehicles))
        >>> data_model.set_drop_return_trips(cudf.Series(drop_return))
        """

        validate_size(
            set_drop_return_trips,
            "drop return trips",
            self.get_fleet_size(),
            "number of vehicles",
        )

        super().set_drop_return_trips(set_drop_return_trips)

    @catch_cuopt_exception
    def set_skip_first_trips(self, set_skip_first_trips):
        """
        Skips/neglects cost of travel to first task location,
        implicitly skipping the travel to location.

        Parameters
        ----------
        set_skip_first_trips : cudf.Series dtype - bool
            Set True to skip the trip cost to first task location.

        Examples
        --------
        >>> from cuopt import routing
        >>> import cudf
        >>> locations       = [0, 1, 2, 3]
        >>> vehicles        = [   0,     1]
        >>> skip_first_trip = [False, True]
        >>> data_model = routing.DataModel(len(locations), len(vehicles))
        >>> data_model.set_skip_first_trips(cudf.Series(skip_first_trip))
        """

        validate_size(
            set_skip_first_trips,
            "skip first trips",
            self.get_fleet_size(),
            "number of vehicles",
        )
        super().set_skip_first_trips(set_skip_first_trips)

    @catch_cuopt_exception
    def add_vehicle_order_match(self, vehicle_id, orders):
        """
        Control if a vehicle can only serve a subset of orders

        Parameters
        ----------
        vehicle_id  : Integer
            vehicle id of the vehicle that has restriction on orders
        orders      : cudf.Series dtype - int32
            cudf.Series contains the orders that can be fulfilled by
            vehicle with vehicle_id

        Note: A user can set this multiple times. However, if it is
              set more than once for same vehicle, the order list will
              be overridden with the most recent function call.

              The orders in the give list allowed to be served by other
              vehicles. To make any order served only by a particular
              vehicle, use add_order_vehicle_match function

        Examples
        --------
        >>> n_locations = 4
        >>> n_vehicles = 3
        >>> d = routing.DataModel(n_locations, n_vehicles)
        >>> distance = [
        >>>    [0., 1., 5., 2.], [2., 0., 7., 4.],
        >>>    [1., 5., 0., 9.], [5., 6., 2., 0.]]
        >>> d.add_cost_matrix(cudf.DataFrame(distances))
        >>> # vehicle 0 serves order 1, vehicle 1 serves order 2,
        >>> # vehicle 2 serves order 3
        >>> d.add_vehicle_order_match(0, cudf.Series([1]))
        >>> d.add_vehicle_order_match(1, cudf.Series([2]))
        >>> d.add_vehicle_order_match(2, cudf.Series([3]))
        >>> cuopt_solution = routing.Solve(d)
        """
        validate_range(orders, "orders served", 0, self.get_num_orders())
        validate_range(
            len(orders), "number of orders served", 0, self.get_num_orders()
        )
        super().add_vehicle_order_match(vehicle_id, orders)

    @catch_cuopt_exception
    def add_order_vehicle_match(self, order_id, vehicles):
        """
        Control if an order should only be served by a subset of vehicles

        Parameters
        ----------
        order_id  : Integer
            order id of the order that has restriction on vehicles
        vehicles  : cudf.Series dtype - int32
            cudf.Series contains the vehicles that can fulfill the
            order with order_id

        Note: A user can set this multiple times. However, if it is
              set more than once for same order, the vehicle list will
              be overridden with the most recent function call

              The vehicles in the give list can serve other orders as well.
              To make a vehicle serve only a subset of orders use
              add_vehicle_order_match function

        Examples
        --------
        >>> n_locations = 4
        >>> n_vehicles = 3
        >>> d = routing.DataModel(n_locations, n_vehicles)
        >>> distance = [
        >>>    [0., 1., 5., 2.], [2., 0., 7., 4.],
        >>>    [1., 5., 0., 9.], [5., 6., 2., 0.]]
        >>> d.add_cost_matrix(cudf.DataFrame(distances))
        >>> # order 1 can be served only by vehicle 0,
        >>> # order 2 can be served only by vehicle 1,
        >>> # order 3 can be served only by vehicle 2
        >>> d.add_order_vehicle_match(1, cudf.Series([0]))
        >>> d.add_order_vehicle_match(2, cudf.Series([1]))
        >>> d.add_order_vehicle_match(3, cudf.Series([2]))
        >>> cuopt_solution = routing.Solve(d)
        """
        validate_range(
            len(vehicles), "Number of vehicles", 0, self.get_fleet_size() + 1
        )
        validate_range(
            vehicles,
            "vehicles that can fulfill the order",
            0,
            self.get_fleet_size(),
        )
        super().add_order_vehicle_match(order_id, vehicles)

    @catch_cuopt_exception
    def set_order_service_times(self, service_times, vehicle_id=-1):
        """
        In fully heterogeneous fleet mode, vehicle can take different
        amount of times to complete a task based on their profile
        and the order being served. Here we enable that
        ability to the user by setting for each vehicle id
        the corresponding service times. They can be the same for
        all orders per vehicle/vehicle type or unique.

        The service times are defaulted for all vehicles unless
        vehicle id is specified. If no default service times are
        given then the solver expects all vehicle ids up to fleet
        size to be specified.

        Parameters
        ----------
        service_times : cudf.Series dtype - int32
            service times of size number of orders
        vehicle_id  : int32
            Vehicle id

        Note: A user can set this multiple times. However, if it is
              set more than once for same vehicle, the service times list will
              be overridden with the most recent function call

        Examples
        --------
        >>> n_locations = 4
        >>> n_vehicles = 3
        >>> d = routing.DataModel(n_locations, n_vehicles)
        >>> distance = [
        >>>    [0., 1., 5., 2.], [2., 0., 7., 4.],
        >>>    [1., 5., 0., 9.], [5., 6., 2., 0.]]
        >>> d.add_cost_matrix(cudf.DataFrame(distances))
        >>> # default for all
        >>> d.set_order_service_times(cudf.Series([0, 1, 1, 1]))
        >>> # override vehicle 1
        >>> d.set_order_service_times(cudf.Series([0, 2, 4, 5]), 1)
        >>> cuopt_solution = routing.Solve(d)
        """

        validate_size(
            service_times,
            "service times",
            self.get_num_orders(),
            "number of orders",
        )
        validate_non_negative(service_times, "service times")
        super().set_order_service_times(service_times, vehicle_id)

    @catch_cuopt_exception
    def add_capacity_dimension(self, name, demand, capacity):
        """
        Add capacity dimensions to model the Capacitated Vehicle Routing
        Problem (CVRP)

        The vehicles have a limited carrying capacity of the
        goods that must be delivered. This function can be called more than
        once to model multiple capacity dimensions (weight, volume, number
        of orders, skills). After solving the problem, the demands on each
        route will not exceed the vehicle capacities.

        Note:
          - If vehicle locations is not set, then by default 0th index
            in demand column is considered start and end location of
            all the vehicles. May be it is better to keep demand to be 0.

        Parameters
        ----------
        name : str
            user-specified name for the dimension
        demand : cudf.Series dtype - int32
            cudf.Series containing integer demand value for each locations,
            including the depot. Order is implicit and should be consistent
            with the data model.
        capacity : cudf.Series dtype - int32
            cudf.Series containing integer capacity value for each vehicle
            in the fleet.
            Size of this series must be equal to fleet_size in data model

        Examples
        --------
        >>> from cuopt import routing
        >>> import cudf
        >>> locations      = [0,  1,  2,  3]
        >>> demand_weight  = [0, 10, 20, 40]
        >>> skill_x        = [0,  1,  0,  1] # 0 - skill not needed, 1 - needed
        >>> skill_y        = [0,  0,  1,  1] # 0 - skill not needed, 1 - needed
        >>> vehicles        = [ 0,     1]
        >>> # Vehicle 0 can carry at max 50 units and vehicle 1 100 units
        >>> capacity_weight = [50,   100]
        >>> # If vehicle has skill keep a high value > number of orders, else 0
        >>> veh_skill_x     = [0,    1000] # vehicle-0 doesn't have the skill
        >>> veh_skill_y     = [1000, 1000] # both vehicles have the skill
        >>> data_model = routing.DataModel(len(locations), len(vehicles))
        >>> # Add weight capacity dimension
        >>> data_model.add_capacity_dimension(
        ...   "weight",
        ...   cudf.Series(demand_weight),
        ...   cudf.Series(capacity_weight)
        ... )
        >>> # Add skill x as capacity
        >>> data_model.add_capacity_dimension(
        ...   "skill_x",
        ...   cudf.Series(skill_x),
        ...   cudf.Series(veh_skill_x)
        ... )
        >>> # Add skill y as capacity
        >>> data_model.add_capacity_dimension(
        ...   "skill_y",
        ...   cudf.Series(skill_y),
        ...   cudf.Series(veh_skill_y)
        ... )
        """

        validate_size(
            demand, "demand", self.get_num_orders(), "number of orders"
        )
        validate_size(
            capacity, "capacity", self.get_fleet_size(), "number of vehicles"
        )
        validate_non_negative(capacity, "capacity")
        super().add_capacity_dimension(name, demand, capacity)

    @catch_cuopt_exception
    def set_vehicle_max_costs(self, vehicle_max_costs):
        """
        Limits per vehicle primary matrix cost accumulated along a route.

        Parameters
        ----------
        vehicle_max_costs : cudf.Series dtype - float32
            Upper bound per vehicle for max distance cumulated on a route

        Examples
        --------
        >>> from cuopt import routing
        >>> locations = [0,  1,  2,  3]
        >>> vehicles  = [0, 1]
        >>> vehicle_max_costs = [150, 200]
        >>> data_model = routing.DataModel(len(locations), len(vehicles))
        >>> data_model.set_vehicle_max_costs(cudf.Series(vehicle_max_costs))
        """
        validate_size(
            vehicle_max_costs,
            "vehicle max costs",
            self.get_fleet_size(),
            "number of vehicles",
        )
        validate_positive(vehicle_max_costs, "vehicle max costs")
        super().set_vehicle_max_costs(vehicle_max_costs)

    @catch_cuopt_exception
    def set_vehicle_max_times(self, vehicle_max_times):
        """
        Limits per vehicle the time accumulated along a route. This limit
        accounts for both travel, service and wait time.

        Parameters
        ----------
        vehicle_max_times : cudf.Series dtype - float32
            Upper bound per vehicle for max duration cumulated on a route

        Examples
        --------
        >>> from cuopt import routing
        >>> locations = [0,  1,  2,  3]
        >>> vehicles  = [0, 1]
        >>> vehicle_max_times = [150, 200]
        >>> data_model = routing.DataModel(len(locations), len(vehicles))
        >>> data_model.set_vehicle_max_times(cudf.Series(vehicle_max_times))
        """

        validate_size(
            vehicle_max_times,
            "vehicle max times",
            self.get_fleet_size(),
            "number of vehicles",
        )
        validate_positive(vehicle_max_times, "vehicle max times")
        super().set_vehicle_max_times(vehicle_max_times)

    @catch_cuopt_exception
    def set_vehicle_fixed_costs(self, vehicle_fixed_costs):
        """
        Limits per vehicle primary matrix cost accumulated along a route.
        Lets the solver find the optimal fleet according to vehicle costs.
        In a heterogeneous setting, not all vehicles will have the same cost.
        Sometimes it may be optimal to use two vehicles with lower cost
        compared to one vehicle with a huge cost.

        Parameters
        ----------
        vehicle_fixed_costs : cudf.Series dtype - float32
            Cost of each vehicle

        Examples
        --------
        >>> from cuopt import routing
        >>> locations = [0,  1,  2,  3]
        >>> vehicles  = [0, 1]
        >>> vehicle_fixed_costs = [16, 1]
        >>> data_model = routing.DataModel(len(locations), len(vehicles))
        >>> data_model.set_vehicle_fixed_costs(
        >>>     cudf.Series(vehicle_fixed_costs)
        >>> )
        """
        validate_size(
            vehicle_fixed_costs,
            "vehicle costs",
            self.get_fleet_size(),
            "number of vehicles",
        )
        validate_range(
            vehicle_fixed_costs, "vehicle costs", 0, np.finfo(np.float32).max
        )
        super().set_vehicle_fixed_costs(vehicle_fixed_costs)

    @catch_cuopt_exception
    def set_min_vehicles(self, min_vehicles):
        """
        Request a minimum number of vehicles to be used for routing.
        Note: The resulting solution may not be optimal.

        Parameters
        ----------
        min_vehicles : Integer
            The minimum number of vehicle to use.

        Examples
        --------
        >>> from cuopt import routing
        >>> locations = [0,  1,  2,  3]
        >>> vehicles  = [0, 1]
        >>> data_model = routing.DataModel(len(locations), len(vehicles))
        >>> # Set minimum vehicles that needs to be used to find the solution
        >>> data_model.set_min_vehicles(2)
        """
        validate_positive(min_vehicles, "minimum number of vehicles")
        super().set_min_vehicles(min_vehicles)

    @catch_cuopt_exception
    def get_num_locations(self):
        """
        Returns the number of locations
        (vehicle start/end locations + task locations).
        """
        return super().get_num_locations()

    @catch_cuopt_exception
    def get_fleet_size(self):
        """
        Returns the number of vehicles in the fleet.
        """
        return super().get_fleet_size()

    @catch_cuopt_exception
    def get_num_orders(self):
        """
        Return number of orders.
        """
        return super().get_num_orders()

    @catch_cuopt_exception
    def get_cost_matrix(self, vehicle_type=0):
        """
        Returns cost matrix as 2D DeviceNDArray in row major format.
        """
        return super().get_cost_matrix(vehicle_type)

    @catch_cuopt_exception
    def get_transit_time_matrix(self, vehicle_type=0):
        """
        Returns transit time matrix as 2D DeviceNDArray in row major format.
        """
        return super().get_transit_time_matrix(vehicle_type)

    @catch_cuopt_exception
    def get_transit_time_matrices(self):
        """
        Returns all transit time matrices as 2D DeviceNDArray
        in row major format as dictionary with vehicle types as keys.
        """
        return super().get_transit_time_matrices()

    @catch_cuopt_exception
    def get_initial_solutions(self):
        """ """
        return super().get_initial_solutions()

    @catch_cuopt_exception
    def get_order_locations(self):
        """
        Returns order locations as cudf.Series with int type.
        """
        return super().get_order_locations()

    @catch_cuopt_exception
    def get_vehicle_types(self):
        """
        Returns types of vehicles in the fleet
        as cudf.Series with uint8 type
        """
        return super().get_vehicle_types()

    @catch_cuopt_exception
    def get_pickup_delivery_pairs(self):
        """
        Returns pick up and delivery order indices as
        cudf.Series with int type.
        """
        return super().get_pickup_delivery_pairs()

    @catch_cuopt_exception
    def get_vehicle_time_windows(self):
        """
        Returns earliest and latest time windows as cudf.Series with int type.
        """
        return super().get_vehicle_time_windows()

    @catch_cuopt_exception
    def get_vehicle_locations(self):
        """
        Returns start and return locations as cudf.Series with int type.
        """
        return super().get_vehicle_locations()

    @catch_cuopt_exception
    def get_drop_return_trips(self):
        """
        Returns drop return trips as cudf.Series with bool type.
        """
        return super().get_drop_return_trips()

    @catch_cuopt_exception
    def get_skip_first_trips(self):
        """
        Returns skip first trips as cudf.Series with bool type.
        """
        return super().get_skip_first_trips()

    @catch_cuopt_exception
    def get_capacity_dimensions(self):
        """
        Returns a dictionary containing demands and capacity under demand name.
        """
        return super().get_capacity_dimensions()

    @catch_cuopt_exception
    def get_order_time_windows(self):
        """
        Returns earliest and latest time
        as cudf.Series with int type.
        """
        return super().get_order_time_windows()

    @catch_cuopt_exception
    def get_order_prizes(self):
        """
        Returns order prizes as cudf.Series with float32 type
        """
        return super().get_order_prizes()

    @catch_cuopt_exception
    def get_break_locations(self):
        """
        Returns break locations as cudf.Series with int type.
        """
        return super().get_break_locations()

    @catch_cuopt_exception
    def get_break_dimensions(self):
        """
        Returns a dictionary containing break earliest, latest, duration
        under demand name.
        """
        return super().get_break_dimensions()

    @catch_cuopt_exception
    def get_non_uniform_breaks(self):
        """
        Returns a dictionary containing breaks
        """
        return super().get_non_uniform_breaks()

    @catch_cuopt_exception
    def get_objective_function(self):
        """
        Returns objectives as cudf.Series with int type and
        weights as cudf.Series with float type.
        """
        return super().get_objective_function()

    @catch_cuopt_exception
    def get_vehicle_max_costs(self):
        """
        Returns max costs per vehicles
        """
        return super().get_vehicle_max_costs()

    @catch_cuopt_exception
    def get_vehicle_max_times(self):
        """
        Returns max times per vehicles
        """
        return super().get_vehicle_max_times()

    @catch_cuopt_exception
    def get_vehicle_fixed_costs(self):
        """
        Returns fixed costs per vehicles
        """
        return super().get_vehicle_fixed_costs()

    @catch_cuopt_exception
    def get_vehicle_order_match(self):
        """
        Returns a dictionary containing the orders that can be fulfilled
        for specified vehicles
        """
        return super().get_vehicle_order_match()

    @catch_cuopt_exception
    def get_order_vehicle_match(self):
        """
        Returns a dictionary containing the vehicles that can fulfill
        specified orders
        """
        return super().get_order_vehicle_match()

    @catch_cuopt_exception
    def get_order_service_times(self, vehicle_id=-1):
        """
        Returns a dictionary containing the vehicles and their associated
        service times
        """
        return super().get_order_service_times(vehicle_id)

    @catch_cuopt_exception
    def get_min_vehicles(self):
        """
        Returns minimum vehicles set.
        """
        return super().get_min_vehicles()


class SolverSettings(vehicle_routing_wrapper.SolverSettings):
    """
    SolverSettings()

    Initialize a SolverSettings.
    """

    @catch_cuopt_exception
    def __init__(self):
        super().__init__()

    @catch_cuopt_exception
    def set_time_limit(self, seconds):
        """
        Set a fixed solving time in seconds, the timer starts when `Solve`
        is called.

        Accuracy may be impacted. Problem under 100 locations may be solved
        with reasonable accuracy under a second. Larger problems may need a few
        minutes. A generous upper bond is to set the number of seconds to
        num_locations. By default it is set to num_locations/5.
        If increased accuracy is desired, this needs to set to higher numbers.

        Parameters
        ----------
        seconds : Float
            The number of seconds

        Examples
        --------
        >>> from cuopt import routing
        >>> locations = [0,  1,  2,  3]
        >>> vehicles  = [0, 1]
        >>> data_model = routing.DataModel(len(locations), len(vehicles))
        ...........
        >>> settings = routing.SolverSettings()
        >>> settings.set_time_limit(0.1)
        """
        validate_positive(seconds, "time limit")
        super().set_time_limit(seconds)

    @catch_cuopt_exception
    def set_verbose_mode(self, verbose):
        """
        Displaying internal information during the solver execution.

        Parameters
        ----------
        verbose : bool
            Set True to display information. Execution time may be impacted.
        """
        super().set_verbose_mode(verbose)

    @catch_cuopt_exception
    def set_error_logging_mode(self, logging):
        """
        Displaying constraint error information during the solver execution.

        Parameters
        ----------
        logging : bool
            Set True to display information. Execution time may be impacted.
        """
        super().set_error_logging_mode(logging)

    @catch_cuopt_exception
    def dump_best_results(self, file_path, interval):
        """
        Dump best results in a given file as csv, reports in given intervals.

        Parameters
        ----------
        file_path : Absolute path of file to be written
        interval : Report intervals in seconds

        Examples
        --------
        >>> from cuopt import routing
        >>> locations = [0,  1,  2,  3]
        >>> vehicles  = [0, 1]
        >>> data_model = routing.DataModel(len(locations), len(vehicles))
        ...........
        >>> settings = routing.SolverSettings()
        >>> settings.dump_best_results("results.csv", 20)
        """
        super().dump_best_results(file_path, interval)

    @catch_cuopt_exception
    def dump_config_file(self, file_name):
        """
        Dump configuration information in a given file as yaml

        Parameters
        ----------
        file_name : Absolute path of file to be written
        """
        super().dump_config_file(file_name)

    @catch_cuopt_exception
    def get_time_limit(self):
        """
        Returns solving time set.
        """
        return super().get_time_limit()

    @catch_cuopt_exception
    def get_best_results_file_path(self):
        """
        Returns file path where result will be stored,
        if not set, it will return None.
        """
        return super().get_best_results_file_path()

    @catch_cuopt_exception
    def get_config_file_name(self):
        """
        Returns the full abs path of the file including the filename
        where the configuration data will be dumped
        if not set, it will return None
        """
        return super().get_config_file_name()

    @catch_cuopt_exception
    def get_best_results_interval(self):
        """
        Returns interval set, if not set, it will return None
        """
        return super().get_best_results_interval()


@catch_cuopt_exception
def Solve(data_model, solver_settings=None):
    """
    Solves the routing problem.

    Parameters
    ----------
    data_model: DataModel
        data model containing orders, vehicles and constraints.
    solver_settings: SolverSettings
        settings to configure solver configurations.
        By default, it uses default solver settings to solve.

    Returns
    -------
    assignment : Assignment
        Assignment object containing the solver output.
    """
    if solver_settings is None:
        solver_settings = SolverSettings()

    solution = vehicle_routing_wrapper.Solve(data_model, solver_settings)
    if solver_settings.get_config_file_name() is not None:
        routing.utils.save_data_model_to_yaml(
            data_model,
            solver_settings,
            solution,
            solver_settings.get_config_file_name(),
        )
    return solution
