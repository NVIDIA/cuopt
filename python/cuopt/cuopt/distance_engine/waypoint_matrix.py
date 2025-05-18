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

from cuopt.distance_engine import waypoint_matrix_wrapper
from cuopt.utilities import catch_cuopt_exception


class WaypointMatrix(waypoint_matrix_wrapper.WaypointMatrix):
    """

    WaypointMatrix(offsets, indices, weights)

    Initialize a Waypoint Matrix.

    Parameters
    ----------
    offsets : numpy.ndarray
        numpy.ndarray of size V + 1 (V: number of vertices).
        It contains the offsets for the vertices in this graph.
        Offsets must be in the range [0, E] (E: number of edges).
    indices : numpy.ndarray
        numpy.ndarray of size E (E: number of edges).
        It contains the destination index for each edge.
        Destination indices must be in the range [0, V)
        (V: number of vertices).
    weights : numpy.ndarray
        numpy.ndarray of size E (E: number of edges).
        It contains the weight value for each edge.
        The expected type is floating point number.

    Examples
    --------
    >>> import cuopt
    >>> import numpy as np
    >>> offsets= np.array([0,       3,    5,    7, 8, 9])
    >>> edges=   np.array([1, 2, 3, 0, 2, 0, 3, 4, 0])
    >>> weights= np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> w_matrix = routing.WaypointMatrix(offsets, edges, weights)
    """

    @catch_cuopt_exception
    def __init__(self, offsets, indices, weights):
        if len(offsets.shape) != 1:
            raise ValueError("""Shape of offsets needs to be of length 1""")

        if len(indices.shape) != 1:
            raise ValueError("""Shape of offsets needs to be of length 1""")

        if len(weights.shape) != 1:
            raise ValueError("""Shape of offsets needs to be of length 1""")

        if offsets.shape[0] <= 0:
            raise ValueError("""Offsets length must be positive""")

        if indices.shape[0] <= 0:
            raise ValueError("""Indices length must be positive""")

        if weights.shape[0] <= 0:
            raise ValueError("""Weights length must be positive""")

        if len(indices) != len(weights):
            raise ValueError(
                """Number of indices doesn't match number of weights"""
            )
        super().__init__(offsets, indices, weights)

    @catch_cuopt_exception
    def compute_cost_matrix(self, target_locations):
        """
        Compute the cost matrix over the passed graph and target locations.

        This function can be used when the cost matrix is not acquirable due
        to an incomplete graph. The cost matrix is computed then returned.
        It can later be used for the Solver (see DataModel.set_matrix).

        Parameters
        ----------
        target_locations : numpy.ndarray
            numpy.ndarray representing the target locations indices
            with respect to the graph.
            Target locations indices must be in the range [0, V-1]
            (V: number of vertices).

        Raises
        ------
        ValueError
            Shape of target_locations needs to be of length 1.
        ValueError
            Target_locations length must be positive.

        Returns
        -------
        cudf.DataFrame
            cudf.DataFrame representing the cost matrix

        Examples
        --------
        >>> import cuopt
        >>> import numpy as np
        >>> offsets= np.array([0,       3,    5,    7, 8, 9])
        >>> edges=   np.array([1, 2, 3, 0, 2, 0, 3, 4, 0])
        >>> weights= np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
        >>> w_matrix = routing.WaypointMatrix(offsets, edges, weights)
        >>> target_locations = np.array([1, 3])
        >>> cost_matrix = w_matrix.compute_cost_matrix(target_locations)
        >>> cost_matrix
              0    1
        0   0.0  7.0
        1  18.0  0.0
        >>> data_model = routing.DataModel(cost_matrix.shape[0], 2)
        >>> data_model.set_matrix(cost_matrix)
        """
        if len(target_locations.shape) != 1:
            raise ValueError(
                """Shape of target_locations needs to be of length 1"""
            )

        if target_locations.shape[0] <= 0:
            raise ValueError("""Target_locations length must be positive""")

        return super().compute_cost_matrix(target_locations)

    @catch_cuopt_exception
    def compute_waypoint_sequence(self, target_locations, route_df):
        """
        Compute the waypoint sequence over the whole route.

        The waypoint sequence is an extend version of the route.
        Between each route target locations, all the intermediate waypoints
        are added.
        Waypoints and target locations ids are based on the graph.

        A new field is added to route_df to browse through the returned
        waypoint sequence.
        The 'sequence_offset' field associates an offset to each element
        in the route.

        Notes
        -----
        Calling this function before compute_cost_matrix is an error.

        Parameters
        ----------
        target_locations : numpy.ndarray
            numpy.ndarray representing the target locations indices
            with respect to the graph.
            Target locations indices must be in the range [0, V]
            (V: number of vertices).
        route_df: cudf.DataFrame
            Contains route, vehicle_id, arrival_stamp,
            and locations. Contains an extra 'sequence_offset' field
            after this call.

        Raises
        ------
        ValueError
            Shape of target_locations needs to be of length 1
        ValueError
            Target_locations length must be positive
        ValueError
            Route length must be positive

        Returns
        -------
        cudf.DataFrame
            waypoint_sequence cudf.Series representing the waypoint_sequence
            waypoint_type     cudf.Series representing type of waypoint
            Start    - Start location of the vehicle/tech/robot
            End      - End location of the vehicle/tech/robot
            w        - Location passing through to get to delivery location
            Delivery - Location where delivery needs to be made
            "-"      - Separates vehicle sequence from another

        Examples
        --------
        >>> import cuopt
        >>> import numpy as np
        >>> offsets= np.array([0,       3,    5,    7, 8, 9])
        >>> edges=   np.array([1, 2, 3, 0, 2, 0, 3, 4, 0])
        >>> weights= np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
        >>> w_matrix = routing.WaypointMatrix(offsets, edges, weights)
        >>> # Starting node is considered as depot
        >>> target_locations = np.array([0, 1, 3, 4])
        >>> cost_matrix = w_matrix.compute_cost_matrix(target_locations)
        >>> cost_matrix
              0     1     2     3
        0   0.0   1.0   3.0  11.0
        1   4.0   0.0   7.0  15.0
        2  17.0  18.0   0.0   8.0
        3   9.0  10.0  12.0   0.0
        >>> data_model = routing.DataModel(cost_matrix.shape[0], 2)
        >>> data_model.set_matrix(cost_matrix)
        >>> solver = cuopt.Solver(data_model)
        >>> solver.set_min_vehicles(2)
        >>> solution = solver.solve()
        >>> route = solution.get_route()
        >>> # Location in this routes is actually ids positional indices of
        >>> # target location
        >>> route
           route  arrival_stamp  truck_id  location
        0      0            0.0         1         0
        1      2            3.0         1         2
        2      3           11.0         1         3
        3      0           20.0         1         0
        4      0            0.0         0         0
        5      1            1.0         0         1
        6      0            5.0         0         0
        >>> # Call to this will also update location in route
        >>> # with actual target ids along with sequence offset
        >>> w_matrix.compute_waypoint_sequence(target_locations, route)
            waypoint_sequence waypoint_type
        0                   0         Start
        1                   3          Task
        2                   3             w
        3                   4          Task
        4                   4             w
        5                   0           End
        6                   0             -
        7                   0         Start
        8                   1          Task
        9                   1             w
        10                  0           End
        >>> # Difference in location and sequence offset in route can be seen
        >>> route
           route  arrival_stamp  truck_id  location  sequence_offset
        0      0            0.0         1         0                0
        1      2            3.0         1         3                2
        2      3           11.0         1         4                4
        3      0           20.0         1         0                6
        4      0            0.0         0         0                7
        5      1            1.0         0         1                9
        6      0            5.0         0         0               11

        """
        if len(target_locations.shape) != 1:
            raise ValueError(
                """Shape of target_locations needs to be of length 1"""
            )

        if target_locations.shape[0] <= 0:
            raise ValueError("""Target_locations length must be positive""")

        if route_df.shape[0] <= 0:
            raise ValueError("""Route length must be positive""")

        return super().compute_waypoint_sequence(target_locations, route_df)

    @catch_cuopt_exception
    def compute_shortest_path_costs(self, target_locations, weights):
        """
        Compute a custom matrix over the passed weights and target locations.

        This is applied on shortest paths found during previous
        compute_cost_matrix call.

        This function allows setting a custom cost between waypoints (for
        example, time) and then getting the total cost it takes to go from one
        target location to all the others. The shortest paths are **not**
        recomputed. The path found from compute_cost_matrix between target
        locations stays the same but the new weight set is used to compute the
        output matrix.

        Notes
        -----
        Giving an edge ordering for weights different from the one given
        during waypoint matrix instanciation will lead to incorrect results.

        Parameters
        ----------
        target_locations : numpy.ndarray
            numpy.ndarray representing the target locations indices
            with respect to the graph.
            Target locations indices must be in the range [0, V)
            (V: number of vertices).
        weights : numpy.ndarray
            numpy.ndarray of size E (E: number of edges).
            It contains the weight value for each edge.
            The expected type is floating point number.

        Raises
        ------
        ValueError
            Shape of target_locations needs to be of length 1
        ValueError
            Target_locations length must be positive
        ValueError
            Shape of weights needs to be of length 1
        ValueError
            Weights length must be positive
        ValueError
            Weights length must be positive
        ValueError
            Given weights and previously set weights length mismatch

        Returns
        -------
        cudf.DataFrame
            cudf.DataFrame representing the custom cost matrix

        Examples
        --------
        >>> import cuopt
        >>> import numpy as np
        >>> offsets=         np.array([ 0,          3,      5,      7, 8, 9])
        >>> edges=           np.array([ 1,  2,  3,  0,  2,  0,  3,  4, 0])
        >>> weights=         np.array([ 1,  2,  3,  4,  5,  6,  7,  8, 9])
        >>> time_to_travel = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90])
        >>> w_matrix = routing.WaypointMatrix(offsets, edges, weights)
        >>> target_locations = np.array([1, 3])
        >>> cost_matrix = w_matrix.compute_cost_matrix(target_locations)
        >>> time_matrix = w_matrix.compute_shortest_path_costs(
        >>>    target_locations, time_to_travel
        >>> )
        >>> cost_matrix
              0    1
        0   0.0  7.0
        1  18.0  0.0
        >>> time_matrix
               0     1
        0    0.0  70.0
        1  180.0   0.0
        >>> data_model = routing.DataModel(cost_matrix.shape[0], 2)
        >>> data_model.set_matrix(cost_matrix)
        >>> data_model.add_transit_time_matrix(time_matrix)
        """
        if len(target_locations.shape) != 1:
            raise ValueError(
                """Shape of target_locations needs to be of length 1"""
            )

        if target_locations.shape[0] <= 0:
            raise ValueError("""Target_locations length must be positive""")

        if len(weights.shape) != 1:
            raise ValueError("""Shape of offsets needs to be of length 1""")

        if weights.shape[0] <= 0:
            raise ValueError("""Weights length must be positive""")

        if self.weights.shape[0] != weights.shape[0]:
            raise ValueError(
                """Given weights and previously set weights length mismatch"""
            )

        return super().compute_shortest_path_costs(target_locations, weights)
