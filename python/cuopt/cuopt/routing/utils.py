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

import glob
import os

import numpy as np
import pandas as pd
import yaml

import cudf

from cuopt import routing
from cuopt.routing import utils_wrapper

# Enable once cupy 13.4.0 is available
# from cupyx.scipy.spatial import distance


def generate_dataset(
    locations=100,
    asymmetric=True,
    min_demand=cudf.Series(),
    max_demand=cudf.Series(),
    min_capacities=cudf.Series(),
    max_capacities=cudf.Series(),
    min_service_time=0,
    max_service_time=0,
    tw_tightness=0.0,
    drop_return_trips=0.0,
    shifts=1,
    n_vehicle_types=1,
    n_matrix_types=1,
    distribution=utils_wrapper.DatasetDistribution.CLUSTERED,
    center_box=None,
    seed=0,
):
    """
    Constructs a dataset from the given parameters.
    locations:

    Parameters
    ----------
    locations: int32
        Number of locations to visit, including the depot.

    asymmetric : bool
        Specifies if the matrices (both cost and time)
        should be asymmetric.

    min_demand : cudf.Series
        Series will be cast to int32.
        Minimum demand value along each dimension.

    max_demand : cudf.Series
        Series will be cast to int32.
        Maximum demand value along each dimension.

    min_capacities : cudf.Series
        Series will be cast to int32.
        Minimum capacity value along each dimension.

    max_capacities : cudf.Series
        Series will be cast to int32.
        Maximum capacity value along each dimension.

    min_service_time : int32
        Lower bound for generated service time.

    max_service_time : int32
        Upper bound for generated service time.

    tw_tightness : float32
        Can make a problem more or less difficult to solve.
        Feasibility is ensured because we made sure the latest time of
        a node is larger than its depot-node distance.

    drop_return_trips : float32
        Percentage of vehicles in the fleet that shouldn't return to the depot.

    shifts : int32
        The number of shifts in the dataset. This will create vehicle
        time windows that split the fleet into multiple shifts.

    n_vehicle_types : int32
        Multiple vehicle types can be generated each with a corresponding
        cost matrix or time matrix.

    n_matrix_types : int32
        There can be one cost and time matrix per vehicle type.
        The cupy array generated is of size dim4(n_vehicle_types,
        n_matrix_types, nloc, nloc). In the case the n_matrix_types is 1,
        the cost matrix is used for time.

    distribution : DatasetDistribution
        The distribution of datapoints similar to what is done for the
        homberger cvrptw dataset.
        https://www.sintef.no/projectweb/top/vrptw/homberger-benchmark/
        Clustered datasets are usually faster to generate and solve.

    center_box : tuple
        The bounding box which constrains all the centroids.

    seed : int32
        Random seed for the raft generator.

    Returns
    -------
    pos_list : cudf.DataFrame
        x and y coordinates.
    matrices : cupy.ndarray
        Vehicle cost and time matrices as
        dim4(n_vehicle_types, n_matrix_types, nloc, nloc).
    orders : cudf.DataFrame
        Time windows and multi dimension demand for each location.
    vehicles : cudf.DataFrame
        Time windows and multi dimension
        capacity for each vehicle.
    """
    if (
        min_demand.shape[0] != max_demand.shape[0]
        or max_demand.shape[0] != min_capacities.shape[0]
        or min_capacities.shape[0] != max_capacities.shape[0]
    ):
        raise ValueError(
            "Demand and capacities arrays have different dimensions"
        )

    return utils_wrapper.generate_dataset(
        locations,
        asymmetric,
        min_demand,
        max_demand,
        min_capacities,
        max_capacities,
        min_service_time,
        max_service_time,
        tw_tightness,
        drop_return_trips,
        shifts,
        n_vehicle_types,
        n_matrix_types,
        distribution,
        center_box,
        seed,
    )


def update_routes_and_vehicles(
    output_df, route, order_pdf, order_data, vehicle_constraints
):
    """
    Helper function to update the final routes and vehicle time windows
    when running cuopt in batches.
    Routes are updated in output_df and vehicle time windows are
    updated in vehicle_constraints.
    """

    def edit_truck_ids(x):
        """
        Get original truck ids. This is added to facilitate temporary
        workaround for vehicle break functionality.
        """
        while x["truck_id"] in vehicle_constraints.node_dict.keys():
            x["truck_id"] = vehicle_constraints.node_dict[x["truck_id"]]
        return x

    def get_truck_ids(x):
        """
        Get original truck ids. This is added to facilitate temporary
        workaround for vehicle break functionality.
        """
        y = x
        while y in vehicle_constraints.node_dict.keys():
            y = vehicle_constraints.node_dict[y]
        return y

    nodes = route["route"].to_arrow().to_pylist()
    ordernum = []
    p_d = []
    for node in nodes:
        if node == 0:
            ordernum.append(-1)
            p_d.append("depot")
        elif node > len(order_data.nodes_to_ordernum):
            ordernum.append(-1)
            p_d.append("break")
        else:
            ordernum.append(order_data.nodes_to_ordernum[node - 1])
            p_d.append(order_data.pickup_delivery[node - 1])
    route["#"] = ordernum
    route["service_type"] = p_d
    add_vehicles = []
    truck_ids = route["truck_id"].to_arrow().to_pylist()
    routes = route["route"].to_arrow().to_pylist()
    arrival_times = route["arrival_stamp"].to_arrow().to_pylist()
    orders = route["#"].to_arrow().to_pylist()
    o_l = order_data.order_locations.to_arrow().to_pylist()
    vehicle_locations = {}
    add_v = 1
    for i in range(0, route.shape[0]):
        v = truck_ids[i]
        o = routes[i]

        if orders[i] != -1:
            st = order_pdf[order_pdf["#"] == orders[i]][
                "delivery_service_time"
            ].iloc[0]
            t = arrival_times[i] + st
            vehicle_locations[v] = (o_l[o], t)
        else:  # Add new vehicle
            if arrival_times[i] - vehicle_constraints.vehicle_earliest[v] > 30:
                vehicle_constraints.node_dict[
                    vehicle_constraints.num_vehicles - 1 + add_v
                ] = v
                add_v = add_v + 1
                add_vehicles.append(
                    (
                        vehicle_constraints.vehicle_earliest[v],
                        arrival_times[i],
                        0,
                    )
                )
    vehicle_constraints.num_vehicles = vehicle_constraints.num_vehicles + len(
        add_vehicles
    )
    for i in add_vehicles:
        vehicle_constraints.vehicle_earliest.append(i[0])
        vehicle_constraints.vehicle_latest.append(i[1])
    for v, loc_and_time in vehicle_locations.items():
        # l = loc_and_time[0]
        t = loc_and_time[1]
        vehicle_constraints.vehicle_earliest[v] = t
    if len(output_df) == 0:
        output_df = route[
            ["truck_id", "#", "service_type", "arrival_stamp"]
        ].to_pandas()
    else:
        output_df = pd.concat(
            [
                output_df,
                route[
                    ["truck_id", "#", "service_type", "arrival_stamp"]
                ].to_pandas(),
            ]
        ).reset_index(drop=True)
    output_df = output_df.apply(lambda x: edit_truck_ids(x), axis=1)
    return output_df


def add_vehicle_constraints(
    num_vehicles,
    vehicle_capacity,
    break_earliest=None,
    break_latest=None,
    vehicle_speed=1,
):
    """
    Helper function to add vehicle constraints when running
    cuopt in batches. Vehicle constraints that can be added
    include number of vehicles, vehicle capacity, speed of vehicles
    and earliest and latest vehicle break times.
    """

    vehicle_constraints = {}
    vehicle_constraints["vehicle_earliest"] = [0] * num_vehicles
    vehicle_constraints["vehicle_latest"] = [2147483647] * num_vehicles
    vehicle_constraints["node_dict"] = {}
    if break_earliest is not None and break_latest is not None:
        for i in range(num_vehicles):
            for j in range(1, len(break_earliest) + 1):
                vehicle_constraints["node_dict"][i + num_vehicles * j] = i
        for j in range(len(break_earliest)):
            vehicle_constraints["vehicle_earliest"] = (
                vehicle_constraints["vehicle_earliest"]
                + [break_latest[j]] * num_vehicles
            )
            vehicle_constraints["vehicle_latest"] = [
                break_earliest[-(j + 1)]
            ] * num_vehicles + vehicle_constraints["vehicle_latest"]
    vehicle_constraints["num_vehicles"] = len(
        vehicle_constraints["vehicle_earliest"]
    )
    vehicle_constraints["speed"] = vehicle_speed
    vehicle_constraints["capacity"] = vehicle_capacity
    return type("vehicle_constraints_class", (object,), vehicle_constraints)()


def create_pickup_delivery_data(
    matrix_pdf, raw_order_pdf, depot, vehicle_constraints
):
    """
    Creates and returns travel matrix, order information
    and vehicle information for pickup and deliveries.
    """

    matrix_df_cols = matrix_pdf.index.tolist()
    matrix_pdf = matrix_pdf.values.tolist()
    nodes_to_ordernum = (
        raw_order_pdf["#"].tolist() + raw_order_pdf["#"].tolist()
    )

    pickup_delivery = ["pickup"] * len(raw_order_pdf) + ["delivery"] * len(
        raw_order_pdf
    )

    raw_order_df = cudf.DataFrame(
        raw_order_pdf[["earliest_time", "latest_time"]]
    ).reset_index(drop=True)
    raw_order_df["earliest_time"] = raw_order_df["earliest_time"]
    raw_order_df["latest_time"] = raw_order_df["latest_time"]
    temp_df = cudf.DataFrame()
    temp_df["earliest_time"] = raw_order_df["earliest_time"]
    temp_df["latest_time"] = raw_order_df["latest_time"]
    raw_order_df = cudf.concat([raw_order_df, temp_df])
    raw_order_df["service_time"] = (
        raw_order_pdf["pickup_service_time"].tolist()
        + raw_order_pdf["delivery_service_time"].tolist()
    )
    raw_order_df["demand"] = (raw_order_pdf["demand"]).tolist() + (
        -raw_order_pdf["demand"]
    ).tolist()
    constraints_df = cudf.DataFrame()
    constraints_df = cudf.concat([constraints_df, raw_order_df]).reset_index(
        drop=True
    )

    pickup = raw_order_pdf["source"].tolist()
    delivery = raw_order_pdf["sink"].tolist()

    orders = pickup + delivery

    unique_locations = {}
    order_locations = []
    locations = [depot]
    unique_locations[depot] = 0
    cnt = len(unique_locations)
    for order in orders:
        if unique_locations.get(order, -1) == -1:
            unique_locations[order] = cnt
            order_locations.append(cnt)
            locations.append(order)
            cnt = cnt + 1
        else:
            order_locations.append(unique_locations.get(order, -1))

    vehicle_constraints_df = cudf.DataFrame()
    vehicle_constraints_df["vehicle_earliest"] = cudf.Series(
        vehicle_constraints.vehicle_earliest
    ).astype(np.int32)
    vehicle_constraints_df["vehicle_latest"] = cudf.Series(
        vehicle_constraints.vehicle_latest
    ).astype(np.int32)

    order_locations = cudf.Series(order_locations)

    # num_orders = len(orders)
    num_locations = cnt

    matrix = np.empty(shape=(num_locations, num_locations))
    matrix.fill(0.0)

    for i in range(num_locations):
        for j in range(num_locations):
            my_x = locations[i]
            my_y = locations[j]
            my_ix = matrix_df_cols.index(my_x)
            my_iy = matrix_df_cols.index(my_y)
            matrix[i][j] = matrix_pdf[my_ix][my_iy] / vehicle_constraints.speed

    pdf = pd.DataFrame(matrix)
    matrix_df = cudf.DataFrame.from_pandas(pdf).astype("float32")

    pickup_indices = cudf.Series(
        i for i in range(0, int(len(raw_order_df) / 2))
    )
    delivery_indices = cudf.Series(
        i for i in range(int(len(raw_order_df) / 2), len(raw_order_df))
    )

    order_data = {
        "order_locations": order_locations,
        "pickup_indices": pickup_indices,
        "delivery_indices": delivery_indices,
        "nodes_to_ordernum": nodes_to_ordernum,
        "pickup_delivery": pickup_delivery,
        "order_constraints": constraints_df,
    }
    vehicle_data = {
        "vehicle_constraints": vehicle_constraints_df,
        "vehicle_capacity": vehicle_constraints.capacity,
    }
    order_data = type("order_class", (object,), order_data)()
    vehicle_data = type("vehicle_class", (object,), vehicle_data)()
    return matrix_df, order_data, vehicle_data


def create_data_model(filename, num_vehicles=None, run_nodes=None):
    """
    Construct a data model with reduced number of vehicles and/or
    nodes from a standard Homberger and Li&Lim dataset

    This function is used mainly to create smaller tests for testing
    API functionalities

    Parameters
    ----------
    filename :  filename corresponding to the test in benchmark format
    num_vehicles: number of vehicles to be used in a modified test.
        If it is defaulted the number of vehicles is not modified
    run_nodes:  number of nodes to consider in a modified test.
        If it is defaulted the number of nodes/tasks is not modified
    """
    service_list, vehicle_capacity, vehicle_num = create_from_file(filename)
    if num_vehicles is None:
        num_vehicles = vehicle_num

    if run_nodes:
        service_list = service_list.head(run_nodes + 1)
    distances = build_matrix(service_list)
    distances = distances.astype(np.float32)

    nodes = service_list["demand"].shape[0]
    d = routing.DataModel(nodes, num_vehicles, nodes - 1)
    d.add_cost_matrix(distances)

    order_locations = [i + 1 for i in range(nodes - 1)]
    d.set_order_locations(cudf.Series(order_locations))

    # extract depot node info
    vehicle_earliest = [service_list["earliest_time"].iloc[0]] * num_vehicles
    vehicle_latest = [service_list["latest_time"].iloc[0]] * num_vehicles
    d.set_vehicle_time_windows(
        cudf.Series(vehicle_earliest), cudf.Series(vehicle_latest)
    )

    # Remove depot node info
    service_list = service_list.tail(nodes - 1)

    demand = service_list["demand"].astype(np.int32)
    capacity_list = [vehicle_capacity] * num_vehicles
    capacity_series = cudf.Series(capacity_list)
    d.add_capacity_dimension("demand", demand, capacity_series)

    earliest = service_list["earliest_time"].astype(np.int32)
    latest = service_list["latest_time"].astype(np.int32)
    service = service_list["service_time"].astype(np.int32)
    d.set_order_time_windows(earliest, latest)
    d.set_order_service_times(service)

    return d


def create_data_model_from_yaml(file_path):
    with open(file_path, "rt") as f:
        try:
            yaml_dict = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(exc)
            return None

    n_vehicles = yaml_dict["n_vehicles"]

    n_locations = yaml_dict["n_locations"]

    n_orders = yaml_dict["n_orders"]

    data_model = routing.DataModel(
        n_locations,
        n_vehicles,
        n_orders=n_orders,
    )

    veh_types = yaml_dict["vehicle_types"]
    cost_matrix = yaml_dict["cost_matrix"]
    for idx, veh_type in enumerate(veh_types):
        data_model.add_cost_matrix(cudf.DataFrame(cost_matrix[idx]), veh_type)

    if "time_matrix" in yaml_dict:
        time_matrix = yaml_dict["time_matrix"]
        for idx, veh_type in enumerate(veh_types):
            data_model.add_transit_time_matrix(
                cudf.DataFrame(time_matrix[idx], veh_type)
            )

    if "vehicle_start_locations" in yaml_dict:
        data_model.set_vehicle_locations(
            cudf.Series(yaml_dict["vehicle_start_locations"]),
            cudf.Series(yaml_dict["vehicle_end_locations"]),
        )

    if "vehicle_max_costs" in yaml_dict:
        data_model.set_vehicle_max_costs(
            cudf.Series(yaml_dict["vehicle_max_costs"])
        )

    if "veh_tw_earliest" in yaml_dict:
        data_model.set_vehicle_time_windows(
            cudf.Series(yaml_dict["veh_tw_earliest"]),
            cudf.Series(yaml_dict["veh_tw_latest"]),
        )

    if "skip_first_trips" in yaml_dict:
        data_model.set_skip_first_trips(
            cudf.Series(yaml_dict["skip_first_trips"])
        )

    if "drop_return_trips" in yaml_dict:
        data_model.set_drop_return_trips(
            cudf.Series(yaml_dict["drop_return_trips"])
        )

    if "break_locations" in yaml_dict:
        data_model.set_break_locations(
            cudf.Series(yaml_dict["break_locations"])
        )

    if "break_time_windows" in yaml_dict:
        brk_time_windows = yaml_dict["break_time_windows"]
        for cnt, brk_time_window in enumerate(brk_time_windows):  # noqa
            data_model.add_break_dimension(
                cudf.Series(brk_time_window[0]),
                cudf.Series(brk_time_window[1]),
                cudf.Series(brk_time_window[2]),
            )

    if "order_locations" in yaml_dict:
        data_model.set_order_locations(
            cudf.Series(yaml_dict["order_locations"])
        )

    if "order_windows" in yaml_dict:
        order_windows = yaml_dict["order_windows"]
        order_earliest = [ele[0] for ele in order_windows]
        order_latest = [ele[1] for ele in order_windows]
        order_service = [ele[2] for ele in order_windows]
        data_model.set_order_time_windows(
            cudf.Series(order_earliest),
            cudf.Series(order_latest),
        )
        data_model.set_order_service_times(cudf.Series(order_service))

    if "pickup_delivery_pairs" in yaml_dict:
        pdp_pairs = yaml_dict["pickup_delivery_pairs"]
        pickup = [ele[0] for ele in pdp_pairs]
        delivery = [ele[1] for ele in pdp_pairs]
        data_model.set_pickup_delivery_pairs(
            cudf.Series(pickup), cudf.Series(delivery)
        )

    if "capacity_dimensions" in yaml_dict:
        cap_dims = yaml_dict["capacity_dimensions"]
        for cnt, cap_dim in enumerate(cap_dims):  # noqa
            data_model.add_capacity_dimension(
                cap_dim[0], cudf.Series(cap_dim[1]), cudf.Series(cap_dim[2])
            )

    obj_funcs = []
    obj_weights = []
    if "objective_function_weights" in yaml_dict:
        obj_weights = yaml_dict["objective_function_weights"]
    if "objective_function_types" in yaml_dict:
        obj_funcs = yaml_dict["objective_function_types"]
        data_model.set_objective_function(
            cudf.Series(obj_funcs), cudf.Series(obj_weights)
        )

    if "vehicle_order_match" in yaml_dict:
        vehicle_order_match = yaml_dict["vehicle_order_match"]
        for veh_id, orders in vehicle_order_match.items():
            data_model.set_vehicle_order_match(
                cudf.DataFrame(veh_id, cudf.Series(orders))
            )

    if "order_vehicle_match" in yaml_dict:
        order_vehicle_match = yaml_dict["order_vehicle_match"]
        for ord_id, vehicles in order_vehicle_match.items():
            data_model.set_order_vehicle_match(
                cudf.DataFrame(ord_id, cudf.Series(vehicles))
            )

    if "min_vehicles" in yaml_dict:
        data_model.set_min_vehicles(yaml_dict["min_vehicles"])

    solver_settings = routing.SolverSettings()

    if "solve_time" in yaml_dict:
        solver_settings.set_time_limit(yaml_dict["solve_time"])

    if "best_result_path" in yaml_dict:
        if "best_result_interval" in yaml_dict:
            solver_settings.dump_best_results(
                yaml_dict["best_result_path"],
                yaml_dict["best_result_interval"],
            )

    return data_model, solver_settings


def write_yaml_dict_if_not_empty(yamldict, keyname, data):
    if data is None:
        return
    if isinstance(data, list):
        if len(data) > 0:
            yamldict.update({keyname: data})
            return
        return
    if not data.empty:
        yamldict.update({keyname: data.to_arrow().to_pylist()})
    return


def save_data_model_to_yaml(data_model, solver_settings, solution, fname):
    """
    Writes Solver.DataModel object in yaml format in a given file name

    Parameters
    ----------
    solver_settings : Solver object that contains the data model information
    solution: Solution object that contains routing information
    fname      : file name with yaml extension used for the output
    """
    # write attributes of the model
    yamldict = dict(
        {
            "n_vehicles": data_model.get_fleet_size(),
            "n_locations": data_model.get_num_locations(),
            "n_orders": data_model.get_num_orders(),
        }
    )
    # write cost matrix
    veh_types = data_model.get_vehicle_types().unique().to_arrow().to_pylist()
    if not veh_types:
        veh_types = [0]
    write_yaml_dict_if_not_empty(yamldict, "vehicle_types", veh_types)

    cost_matrices = []
    for idx, veh_type in enumerate(veh_types):
        cost_matrices.append(data_model.get_cost_matrix(veh_type).tolist())
    yamldict.update({"cost_matrix": cost_matrices})

    transit_time_matrices = data_model.get_transit_time_matrices()
    if len(transit_time_matrices) != 0:
        time_matrices = []
        for idx, veh_type in enumerate(veh_types):
            time_matrix = data_model.get_transit_time_matrix(veh_type)
            if time_matrix is not None:
                time_matrices.append(time_matrix.tolist())
        write_yaml_dict_if_not_empty(yamldict, "time_matrix", time_matrices)

    (
        vehicle_start_locations,
        vehicle_return_locations,
    ) = data_model.get_vehicle_locations()

    write_yaml_dict_if_not_empty(
        yamldict, "vehicle_start_locations", vehicle_start_locations
    )
    write_yaml_dict_if_not_empty(
        yamldict, "vehicle_end_locations", vehicle_return_locations
    )

    vehicle_max_costs = data_model.get_vehicle_max_costs()
    write_yaml_dict_if_not_empty(
        yamldict, "vehicle_max_costs", vehicle_max_costs
    )

    (
        vehicle_earliest,
        vehicle_latest,
    ) = data_model.get_vehicle_time_windows()
    write_yaml_dict_if_not_empty(yamldict, "veh_tw_earliest", vehicle_earliest)
    write_yaml_dict_if_not_empty(yamldict, "veh_tw_latest", vehicle_latest)

    drop_return_trips = data_model.get_drop_return_trips()
    skip_first_trips = data_model.get_skip_first_trips()
    write_yaml_dict_if_not_empty(
        yamldict, "skip_first_trips", skip_first_trips
    )
    write_yaml_dict_if_not_empty(
        yamldict, "drop_return_trips", drop_return_trips
    )

    order_locations = data_model.get_order_locations()
    write_yaml_dict_if_not_empty(yamldict, "order_locations", order_locations)

    (
        pickup_indices,
        delivery_indices,
    ) = data_model.get_pickup_delivery_pairs()
    if not pickup_indices.empty:
        pickup_delivery_pairs = list(
            tuple(
                zip(
                    pickup_indices.to_arrow().to_pylist(),
                    delivery_indices.to_arrow().to_pylist(),
                )
            )
        )
        write_yaml_dict_if_not_empty(
            yamldict, "pickup_delivery_pairs", pickup_delivery_pairs
        )

    (
        order_earliest,
        order_latest,
    ) = data_model.get_order_time_windows()
    order_service = data_model.get_order_service_times()
    if not order_earliest.empty:
        order_windows = list(
            tuple(
                zip(
                    order_earliest.to_arrow().to_pylist(),
                    order_latest.to_arrow().to_pylist(),
                    order_service.to_arrow().to_pylist(),
                )
            )
        )
        write_yaml_dict_if_not_empty(yamldict, "order_windows", order_windows)

    write_yaml_dict_if_not_empty(
        yamldict,
        "break_locations",
        data_model.get_break_locations(),
    )
    break_dims = data_model.get_break_dimensions()
    break_dims_lst = []
    for name, break_dim in break_dims.items():
        break_dim_tuple = (
            break_dim["earliest"].to_arrow().to_pylist(),
            break_dim["latest"].to_arrow().to_pylist(),
            break_dim["duration"].to_arrow().to_pylist(),
        )
        break_dims_lst.append(break_dim_tuple)
    write_yaml_dict_if_not_empty(
        yamldict, "break_time_windows", break_dims_lst
    )

    cap_dims = data_model.get_capacity_dimensions()
    cap_dims_lst = []
    for name, cap_dim in cap_dims.items():
        cap_tuple = (
            name,
            cap_dim["demand"].to_arrow().to_pylist(),
            cap_dim["capacity"].to_arrow().to_pylist(),
        )
        cap_dims_lst.append(cap_tuple)
    write_yaml_dict_if_not_empty(yamldict, "capacity_dimensions", cap_dims_lst)

    # objectives info
    objectives, weights = data_model.get_objective_function()
    write_yaml_dict_if_not_empty(
        yamldict, "objective_function_types", objectives
    )
    write_yaml_dict_if_not_empty(
        yamldict, "objective_function_weights", weights
    )

    vh_order_match = data_model.get_vehicle_order_match()
    if len(vh_order_match) > 0:
        vh_order_match = {
            veh_id: orders.to_arrow().to_pylist()
            for veh_id, orders in vh_order_match.items()
        }
        write_yaml_dict_if_not_empty(
            yamldict,
            "vehicle_order_match",
            vh_order_match,
        )

    order_vh_match = data_model.get_order_vehicle_match()
    if len(order_vh_match) > 0:
        order_vh_match = {
            ord_id: vehs.to_arrow().to_pylist()
            for ord_id, vehs in order_vh_match.items()
        }
        write_yaml_dict_if_not_empty(
            yamldict,
            "order_vehicle_match",
            order_vh_match,
        )

    min_v = data_model.get_min_vehicles()
    if min_v:
        yamldict.update({"min_vehicles": min_v})

    sol_t = solver_settings.get_time_limit()
    if sol_t:
        yamldict.update({"solve_time": sol_t})

    b_r_fp = solver_settings.get_best_results_file_path()
    b_r_i = solver_settings.get_best_results_interval()
    if b_r_fp:
        if b_r_i:
            yamldict.update({"best_result_path", b_r_fp})
            yamldict.update({"best_result_interval", b_r_i})

    if solution.get_status() == 0:
        sol_df = solution.get_route()
        write_yaml_dict_if_not_empty(
            yamldict, "route_solution", sol_df.to_pandas().to_numpy().tolist()
        )

    fo = open(fname, "w")
    yaml.safe_dump(yamldict, fo, default_flow_style=False)
    fo.close()


RAPIDS_DATASET_ROOT_DIR = os.getenv("RAPIDS_DATASET_ROOT_DIR")
if RAPIDS_DATASET_ROOT_DIR is None:
    RAPIDS_DATASET_ROOT_DIR = os.path.dirname(os.getcwd())
    RAPIDS_DATASET_ROOT_DIR = os.path.join(RAPIDS_DATASET_ROOT_DIR, "datasets")

SOLOMON_PATH = os.path.join(RAPIDS_DATASET_ROOT_DIR, "solomon", "In", "*.txt")
REF_PATH = os.path.join(RAPIDS_DATASET_ROOT_DIR, "ref")
DATASETS_SOLOMON = glob.glob(SOLOMON_PATH)[:3]


def read_ref_tsp(file, test_type):
    ref_file = f"{test_type}.txt"
    file_path = os.path.join(REF_PATH, ref_file)
    base_name = os.path.basename(file)
    with open(file_path, "rt") as f:
        for line in f:
            refs = [x.strip() for x in line.split(",")]
            ref_path = refs[0]
            if base_name == os.path.basename(ref_path):
                return float(refs[1]), int(refs[2])
    raise ValueError(f"Reference for {base_name} could not be found!")


def read_ref(file, test_type, nodes):
    ref_file = f"{test_type}_{nodes}.txt"
    file_path = os.path.join(REF_PATH, ref_file)
    base_name = os.path.basename(file)
    with open(file_path, "rt") as f:
        for line in f:
            refs = [x.strip() for x in line.split(",")]
            ref_path = refs[0]
            if base_name == os.path.basename(ref_path):
                return float(refs[1]), int(refs[2])
    raise ValueError(f"Reference for {base_name} could not be found!")


def create_from_file_tsp(file_path):
    node_list = []
    with open(file_path, "rt") as f:
        data_section = False
        for line in f:
            if line.split()[0].isdecimal():
                data_section = True
            if line.split()[0] == "EOF":
                break
            if data_section:
                node_list.append(line.split())

    df = cudf.DataFrame(columns=["vertex", "xcord", "ycord"])

    for item in node_list:
        row = {
            "vertex": int(item[0]) - 1,
            "xcord": float(item[1]),
            "ycord": float(item[2]),
        }
        df = cudf.concat(
            [df, cudf.DataFrame(row)],
            ignore_index=True,
        )

    return df


def create_from_file(file_path, is_pdp=False):
    node_list = []
    with open(file_path, "rt") as f:
        count = 1
        for line in f:
            if is_pdp and count == 1:
                vehicle_num, vehicle_capacity, speed = line.split()
            elif not is_pdp and count == 5:
                vehicle_num, vehicle_capacity = line.split()
            elif is_pdp:
                node_list.append(line.split())
            elif count >= 10:
                node_list.append(line.split())
            count += 1
            # if count == 36:
            #     break

    vehicle_num = int(vehicle_num)
    vehicle_capacity = int(vehicle_capacity)
    df = cudf.DataFrame(
        columns=[
            "vertex",
            "xcord",
            "ycord",
            "demand",
            "earliest_time",
            "latest_time",
            "service_time",
            "pickup_index",
            "delivery_index",
        ]
    )

    for item in node_list:
        row = {
            "vertex": int(item[0]),
            "xcord": float(item[1]),
            "ycord": float(item[2]),
            "demand": int(item[3]),
            "earliest_time": int(item[4]),
            "latest_time": int(item[5]),
            "service_time": int(item[6]),
        }
        if is_pdp:
            row["pickup_index"] = int(item[7])
            row["delivery_index"] = int(item[8])
        df = cudf.concat(
            [df, cudf.DataFrame(row)],
            ignore_index=True,
        )

    return df, vehicle_capacity, vehicle_num


# returns cudf Dataframe of vertices, loc coordnates, demand, time windows
# and a list of vehicle capacities, and an integer which represents vehicle
# number
def create_from_yaml_file(file_path):
    with open(file_path, "rt") as f:
        try:
            yaml_dict = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(exc)
            return None
        df = cudf.DataFrame(
            columns=[
                "vertex",
                "xcord",
                "ycord",
                "demand",
                "earliest_time",
                "latest_time",
                "service_time",
            ]
        )
        vehicle_capacity = yaml_dict["vehicle_capacity"]
        vehicle_num = int(yaml_dict["n_vehicles"])
        loc_coord = yaml_dict["location_coordinates"]
        vertexes = yaml_dict["vertexes"]
        orders = yaml_dict["orders"]
        e_t = yaml_dict["time_win_earliest"]
        l_t = yaml_dict["time_win_latest"]
        s_t = yaml_dict["time_win_service"]
        num_loc = len(loc_coord)
        for idx in range(num_loc):
            loc = loc_coord[idx]
            df = cudf.concat(
                [
                    df,
                    cudf.DataFrame(
                        {
                            "vertex": int(vertexes[idx]),
                            "xcord": float(loc[0]),
                            "ycord": float(loc[1]),
                            "demand": int(orders[idx]),
                            "earliest_time": int(e_t[idx]),
                            "latest_time": int(l_t[idx]),
                            "service_time": int(s_t[idx]),
                        }
                    ),
                ],
                ignore_index=True,
            )

    return df, vehicle_capacity, vehicle_num


# This is temporary till cupy releases new version > 13.3.0
# which has fix with cuvs for cdist
def euclidean_distance(coord):
    coord = np.array(coord)
    n_coord = len(coord)
    distance_matrix = np.zeros((n_coord, n_coord))

    for i in range(n_coord):
        for j in range(n_coord):
            if i != j:
                distance_matrix[i, j] = np.sqrt(
                    (coord[i, 0] - coord[j, 0]) ** 2
                    + (coord[i, 1] - coord[j, 1]) ** 2
                )

    return distance_matrix


def build_matrix(df):
    coords = list(
        zip(
            df["xcord"].to_arrow().to_pylist(),
            df["ycord"].to_arrow().to_pylist(),
        )
    )
    # Enable once cupy 13.4.0 is available
    # return cudf.DataFrame(distance.cdist(coords, coords))
    return cudf.DataFrame(euclidean_distance(coords))


def fill_demand(
    df, data_model, vehicle_capacity, n_vehicles, use_order_loc=False
):
    demand = df["demand"].astype(np.int32)
    if use_order_loc:
        demand = demand[1::]
    capacity_list = [vehicle_capacity] * n_vehicles
    capacity = cudf.Series(capacity_list)
    data_model.add_capacity_dimension("demand", demand, capacity)


def get_time_limit(nodes):
    return 10 + nodes / 6


def set_limits(solver_settings, nodes):
    solver_settings.set_time_limit(get_time_limit(nodes))


def set_limits_for_quality(solver_settings, nodes):
    solver_settings.set_time_limit(2 * get_time_limit(nodes))


def fill_tw(data_model, df, use_order_loc=False):
    earliest = df["earliest_time"].astype(np.int32)
    latest = df["latest_time"].astype(np.int32)
    service = df["service_time"].astype(np.int32)
    if use_order_loc:
        earliest = earliest[1::]
        latest = latest[1::]
        service = service[1::]
    data_model.set_order_time_windows(earliest, latest)
    data_model.set_order_service_times(service)


def fill_pdp_index(data_model, df, use_order_loc=False):
    n_pd_requests = (df.shape[0] - 1) // 2

    pickup_indices = [0] * n_pd_requests
    delivery_indices = [0] * n_pd_requests
    counter = 0
    added = set()
    for i, row in df.to_pandas().iterrows():
        if (
            i == 0
            or row["pickup_index"] in added
            or row["delivery_index"] in added
            or i in added
        ):
            continue
        if row["pickup_index"] != 0:
            delivery_indices[counter] = i
            pickup_indices[counter] = int(row["pickup_index"])
            added.add(int(row["pickup_index"]))
        else:
            delivery_indices[counter] = int(row["delivery_index"])
            pickup_indices[counter] = i
            added.add(int(row["delivery_index"]))
        added.add(i)
        counter += 1

    assert n_pd_requests == counter
    if use_order_loc:
        pickup_indices = [x - 1 for x in pickup_indices]
        delivery_indices = [x - 1 for x in delivery_indices]
    cu_pickup = cudf.Series(pickup_indices, dtype=np.int32)
    cu_delivery = cudf.Series(delivery_indices, dtype=np.int32)
    data_model.set_pickup_delivery_pairs(cu_pickup, cu_delivery)


# Reads Solomon input file and creates a YAML dict
# which is then dumped to an output file


def create_from_solomon_inp_file(file_path):
    node_list = []
    vertexes = []
    locs = []
    orders = []
    e_times = []
    l_times = []
    s_times = []
    cap = []
    with open(file_path, "rt") as f:
        count = 1
        for line in f:
            if count == 5:
                vehicle_num, vehicle_capacity = line.split()
                vehicle_num = int(vehicle_num)
                vehicle_capacity = int(vehicle_capacity)
            elif count >= 10:
                node_list.append(line.split())
            count += 1

    vertexes = [int(item[0]) for item in node_list]
    locs = [tuple((float(item[1]), float(item[2]))) for item in node_list]
    orders = [int(item[3]) for item in node_list]
    e_times = [int(item[4]) for item in node_list]
    l_times = [int(item[5]) for item in node_list]
    s_times = [int(item[6]) for item in node_list]

    cap.extend([vehicle_capacity] * vehicle_num)
    yamldict = dict(
        {
            "vertexes": vertexes,
            "location_coordinates": locs,
            "orders": orders,
            "time_win_earliest": e_times,
            "time_win_latest": l_times,
            "time_win_service": s_times,
            "n_locations": len(locs),
            "n_vehicles": vehicle_num,
            "vehicle_capacity": cap,
        }
    )
    return yamldict


def convert_solomon_inp_file_to_yaml(file_nm):
    inp = file_nm
    op = inp.replace(".txt", ".yaml")
    fo = open(op, "w")
    yamldict = create_from_solomon_inp_file(inp)
    yaml.safe_dump(yamldict, fo, default_flow_style=False)
    fo.close()


def create_model_dictionary_from_file(filename, is_pdp=False):
    """
    Creates a dictionary (json compatible) of the model by reading
    a standard Homberber or Li&Lim test. This dictionary is eventually
    used in testing the SDK or service
    """
    service_list, vehicle_capacity, vehicle_num = create_from_file(
        filename, is_pdp
    )

    distances = build_matrix(service_list)

    # Extract cost matrix data
    cost_matrix_data = {}
    cost_matrix_data["data"] = {}
    cost_matrix_data["data"]["0"] = (
        distances.to_pandas().astype(np.float32).values.tolist()
    )

    # Extract fleet data
    fleet_data = {}
    fleet_data["vehicle_locations"] = [[0, 0]] * vehicle_num
    fleet_data["capacities"] = [[vehicle_capacity] * vehicle_num] * 1

    depot_earliest = int(service_list["earliest_time"].iloc[0])
    depot_latest = int(service_list["latest_time"].iloc[0])
    vehicle_tw = [[depot_earliest, depot_latest]] * vehicle_num
    fleet_data["vehicle_time_windows"] = vehicle_tw

    # Extract task data
    nodes_including_depot = len(service_list["demand"])
    task_nodes = nodes_including_depot - 1
    service_list = service_list.tail(task_nodes)

    task_data = {}
    task_locations = [i + 1 for i in range(task_nodes)]
    earliest = service_list["earliest_time"].to_arrow().to_pylist()
    latest = service_list["latest_time"].to_arrow().to_pylist()
    service = service_list["service_time"].to_arrow().to_pylist()
    demand = service_list["demand"].to_arrow().to_pylist()
    task_tw = [[int(earliest[i]), int(latest[i])] for i in range(task_nodes)]

    # exctract pdp pairs
    if is_pdp:
        p_index = service_list["pickup_index"].to_arrow().to_pylist()
        d_index = service_list["delivery_index"].to_arrow().to_pylist()

        pairs = []
        for i in range(task_nodes):
            p = int(p_index[i] - 1)
            d = int(d_index[i] - 1)
            if p == -1:
                pairs.append([i, d])

    task_data["task_locations"] = task_locations
    task_data["task_time_windows"] = task_tw
    task_data["service_times"] = service
    task_data["demand"] = [demand] * 1
    if is_pdp:
        task_data["pickup_and_delivery_pairs"] = pairs

    model_dict = {}
    model_dict["cost_matrix_data"] = cost_matrix_data
    model_dict["fleet_data"] = fleet_data
    model_dict["task_data"] = task_data

    return model_dict
