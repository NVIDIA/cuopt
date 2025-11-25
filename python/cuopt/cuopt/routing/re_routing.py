# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import cudf

from cuopt.routing import DataModel


def construct_rerouting_model(
    original_model,
    optimized_route,
    reroute_from_time,
    new_order_data,
    new_distances,
    print_debug_info=False,
):
    """
    Creates a new data model for re-routing/re-optimization

    Parameters
    ----------
    original_model:  routing.DataModel object
        the model that used earlier for generating optimized route
    optimized_route: cudf.DataFrame representing the optimized route
        generated with original_model
    reroute_from_time: Float
        time from which the re-routing should start from
    new_order_data: cudf.DataFrame object
        new batch of orders
    new_distances: updated distance/cost matrix
    print_debug_info: option to display debug messages

    Examples
    --------
    >>> original_route = cuopt.Solver(original_model).solve()
    >>> rescheduling_from = 60
    >>> new_order_data = {}
    >>> new_order_data['order_locations'] = [10, 12, 19, 23]
    >>> new_order_data['pickup_indices'] = [0, 1]
    >>> new_order_data['delivery_indices'] = [2, 3]
    >>> new_order_data['earliest_time'] = [65, 75, 90, 85]
    >>> new_order_data['latest_time'] = [75, 80, 95, 95]
    >>> new_order_data['service_time] = [1, 2, 1, 1]
    >>> new_order_data['demand'] = [5, 5, -5, -5]
    >>> rescheduled_model = construct_rerouting_model(
    >>>    original_model, original_route, rescheduling_from,
    >>>    new_order_data, distances)
    >>> rescheduled_route = cuopt.Solver(rescheduled_model).solve()

    Assumptions
    -----------
    1. Re-routing is done from a specified time
    2. Fleet is not changing
    3. Vehicles en route to service a particular order will finish
       that order first
    4. The original optimized plan is going according to the plan, so we can
       determine the finished orders just based on the time
    5. The problem is pickup and delivery only, for now
    6. There is only one capacity demand dimension

    Approach
    --------
    1. Using the optimized route and the time of re-optimization, we figure
       out which orders are fulfilled (picked up and delivered), partially
       fulfilled (picked up but not delivered), and not initiated
    2. We remove the orders that are fulfilled while keeping the orders that
       are not initiated. For the partially fulfilled orders, we create dummy
       pickup orders at vehicle start locations
    """

    # Error checking
    if new_order_data is not None:
        expected_entries_in_new_order_data = [
            "order_locations",
            "earliest_time",
            "latest_time",
            "service_time",
            "pickup_indices",
            "delivery_indices",
            "demand",
        ]
        for entry in expected_entries_in_new_order_data:
            if entry not in new_order_data:
                raise ValueError(f"{entry} is missing in new order data")

        for entry, value in new_order_data.items():
            if entry not in expected_entries_in_new_order_data:
                raise NotImplementedError(
                    f"{entry} is not implemented for re-optimization"
                )

    # Extract the information from previous data model
    vehicle_num = original_model.get_fleet_size()
    nlocations = original_model.get_num_locations()
    norders = original_model.get_num_orders()

    order_locations = original_model.get_order_locations()
    order_locations_h = order_locations.to_arrow().to_pylist()
    if not order_locations_h:
        order_locations_h = list(range(norders))

    (
        earliest_time,
        latest_time,
    ) = original_model.get_order_time_windows()

    # vehicle dependent service time is not supported
    service_time = original_model.get_order_service_times()
    earliest_time_h = earliest_time.to_arrow().to_pylist()
    latest_time_h = latest_time.to_arrow().to_pylist()
    service_time_h = service_time.to_arrow().to_pylist()

    (
        pickup_indices,
        delivery_indices,
    ) = original_model.get_pickup_delivery_pairs()
    pickup_indices_h = pickup_indices.to_arrow().to_pylist()
    delivery_indices_h = delivery_indices.to_arrow().to_pylist()

    """
    Rescheduling logic adds two capacity constraints to force vehicle
    order match. If we are rescheduling an already rescheduled route,
    we might encounter these from the previous model. So we remove here
    and recreate new ones according to the current status
    """
    vehicle_order_matching_constraints = [
        "force-vehicle-order-pair-1",
        "force-vehicle-order-pair-2",
    ]
    capacity_dimensions = original_model.get_capacity_dimensions().copy()
    for name in vehicle_order_matching_constraints:
        if name in capacity_dimensions:
            del capacity_dimensions[name]

    if len(capacity_dimensions) > 1:
        raise NotImplementedError(
            "Multiple capacity dimensions are not supported in re-routing"
        )

    demand_name = list(capacity_dimensions.keys())[0]

    # Remove the fake demands and capacities.
    # This is needed when we re-re-optimize
    capacity_dimensions_h = {}
    # assume that fleet is not changing
    vehicle_cap = (
        capacity_dimensions[demand_name]["capacity"].to_arrow().to_pylist()
    )
    order_demand = (
        capacity_dimensions[demand_name]["demand"].to_arrow().to_pylist()
    )
    capacity_dimensions_h[name] = {
        "demand": order_demand,
        "capacity": vehicle_cap,
    }

    # Generate the information of vehicle locations at the time of re-routing
    routes = {i: [] for i in range(vehicle_num)}
    times = {i: [] for i in range(vehicle_num)}

    truck_ids = optimized_route["truck_id"].to_arrow().to_pylist()
    route_node_ids = optimized_route["route"].to_arrow().to_pylist()
    arrival_stamps = optimized_route["arrival_stamp"].to_arrow().to_pylist()
    for i in range(0, optimized_route.shape[0]):
        v = truck_ids[i]
        o = route_node_ids[i]
        t = arrival_stamps[i]
        routes[v].append(o)
        times[v].append(t)

    if print_debug_info:
        print(f"routes:{routes}")
        for v in range(vehicle_num):
            if len(routes[v]) > 0:
                print(f"vehicle:{v} route:{routes[v]}")
                print(f"            times:{times[v]}\n")

    # Create a mapping btw pickup and delivery
    pickup_of_delivery = {
        delivery_indices_h[i]: pickup_indices_h[i]
        for i in range(len(pickup_indices_h))
    }

    (
        vehicle_earliest,
        vehicle_latest,
    ) = original_model.get_vehicle_time_windows()
    vehicle_earliest_h = vehicle_earliest.to_arrow().to_pylist()
    vehicle_latest_h = vehicle_latest.to_arrow().to_pylist()
    if not vehicle_earliest_h:
        vehicle_earliest_h = [0] * vehicle_num
        vehicle_latest_h = [2147483647] * vehicle_num

    # vehicles can not start before rerouting time
    for i in range(len(vehicle_earliest_h)):
        vehicle_earliest_h[i] = max(vehicle_earliest_h[i], reroute_from_time)

    (
        vehicle_start_locations,
        vehicle_return_locations,
    ) = original_model.get_vehicle_locations()
    vehicle_start_locations_h = vehicle_start_locations.to_arrow().to_pylist()
    vehicle_return_locations_h = (
        vehicle_return_locations.to_arrow().to_pylist()
    )

    if len(vehicle_start_locations_h) == 0:
        vehicle_start_locations_h = [0] * vehicle_num
        vehicle_return_locations_h = [0] * vehicle_num

    completed_orders = []
    pickedup_but_not_delivered = {}

    pickedup_order_to_vehicle = {}
    for vehicle in range(vehicle_num):
        route_size = len(routes[vehicle])
        if route_size > 0:
            if times[vehicle][route_size - 1] <= reroute_from_time:
                intra_route_id = route_size - 1
                time = times[vehicle][intra_route_id]
            else:
                # find the first order with reroute_from_time
                intra_route_id, time = next(
                    (i, el)
                    for i, el in enumerate(times[vehicle])
                    if el > reroute_from_time
                )  # noqa E127

            if intra_route_id > 0:
                last_order = routes[vehicle][intra_route_id]
                vehicle_start_locations_h[vehicle] = order_locations_h[
                    last_order
                ]
                vehicle_earliest_h[vehicle] = time + service_time_h[last_order]

            pickedup_but_not_delivered[vehicle] = []
            if intra_route_id < route_size - 1:
                for j in range(1, intra_route_id + 1):
                    order = routes[vehicle][j]
                    is_pickup_order = order in pickup_indices_h
                    if is_pickup_order:
                        pickedup_but_not_delivered[vehicle].append(order)
                        pickedup_order_to_vehicle[order] = vehicle
                    else:
                        corresponding_pickup = pickup_of_delivery[order]
                        pickedup_but_not_delivered[vehicle].remove(
                            corresponding_pickup
                        )
                        completed_orders.append(corresponding_pickup)
                        completed_orders.append(order)
                        pickedup_order_to_vehicle.pop(corresponding_pickup)
            else:
                completed_orders.extend(
                    [routes[vehicle][j] for j in range(1, route_size)]
                )

    if print_debug_info:
        print(f"vehicle_locations:{vehicle_start_locations_h}")
        print(f"completed_orders:{completed_orders}")

    new_order_locations_h = []
    new_pickup_indices_h = []
    new_delivery_indices_h = []
    new_earliest_time_h = []
    new_latest_time_h = []
    new_service_time_h = []
    new_capacity_dimensions_h = {}

    # assume that fleet is not changing
    vehicle_cap = capacity_dimensions[demand_name]["capacity"]
    new_capacity_dimensions_h[demand_name] = {
        "demand": [],
        "capacity": vehicle_cap,
    }

    """
    If there are unfinished orders (picked up but not delivered yet) at
    the time of re-routing, create new pickup orders from the vehicle
    locations at time zero (i.e. re-routing start time)
    """
    cnt = 0
    new_order_to_old_order = {}
    old_order_to_new_order = {}
    for order in range(0, norders):
        if completed_orders.count(order) == 0:
            new_order_to_old_order[cnt] = order
            old_order_to_new_order[order] = cnt
            cnt = cnt + 1

    # Convert pickup and delivery indices to new numbering
    for i in range(len(pickup_indices_h)):
        pickup = pickup_indices_h[i]
        delivery = delivery_indices_h[i]
        if pickup not in completed_orders:
            new_pickup = old_order_to_new_order[pickup]
            new_delivery = old_order_to_new_order[delivery]
            new_pickup_indices_h.append(new_pickup)
            new_delivery_indices_h.append(new_delivery)

    # Convert order to vehicle to new numbering
    new_pickedup_order_to_vehicle = {
        old_order_to_new_order[pickup]: vehicle
        for pickup, vehicle in pickedup_order_to_vehicle.items()
    }

    # Extract the order info of incomplete orders
    for i in range(0, len(new_order_to_old_order)):
        old_id = new_order_to_old_order[i]
        is_pickup_order = i in new_pickup_indices_h

        # If this order is already picked up, make sure that in the new problem
        # it is picked up at time zero
        if is_pickup_order and i in new_pickedup_order_to_vehicle:
            vehicle = pickedup_order_to_vehicle[old_id]
            new_loc = vehicle_start_locations_h[vehicle]
            pickup_time = vehicle_earliest_h[vehicle]
            new_order_locations_h.append(new_loc)
            new_earliest_time_h.append(pickup_time)
            new_latest_time_h.append(pickup_time)
            new_service_time_h.append(0)
        else:
            new_order_locations_h.append(order_locations_h[old_id])
            new_earliest_time_h.append(earliest_time_h[old_id])
            new_latest_time_h.append(latest_time_h[old_id])
            new_service_time_h.append(service_time_h[old_id])

        for name in capacity_dimensions:
            # assume that fleet is not changing
            demand = capacity_dimensions[name]["demand"][old_id]
            new_capacity_dimensions_h[name]["demand"].append(demand)

    n_leftover_orders = len(new_order_locations_h)

    # append new order data
    new_order_locations_h.extend(new_order_data["order_locations"])
    new_earliest_time_h.extend(new_order_data["earliest_time"])
    new_latest_time_h.extend(new_order_data["latest_time"])
    new_service_time_h.extend(new_order_data["service_time"])

    # new order data consists of indices with respect to new order data
    adjusted_pickup_indices = [
        id + n_leftover_orders for id in new_order_data["pickup_indices"]
    ]
    adjusted_delivery_indices = [
        id + n_leftover_orders for id in new_order_data["delivery_indices"]
    ]

    new_pickup_indices_h.extend(adjusted_pickup_indices)
    new_delivery_indices_h.extend(adjusted_delivery_indices)

    norders = len(new_service_time_h)

    new_d = DataModel(nlocations, vehicle_num, norders)
    new_d.add_cost_matrix(new_distances)

    if print_debug_info:
        print(f"new_order_locations={new_order_locations_h}")
        print(f"new_pickup_indices={new_pickup_indices_h}")
        print(f"new_delivery_indices={new_delivery_indices_h}")
        print(f"new_earliest={new_earliest_time_h}")
        print(f"new_latest={new_latest_time_h}")
        print(f"new_service={new_service_time_h}")

    new_order_locations = cudf.Series(new_order_locations_h)
    new_d.set_order_locations(new_order_locations)

    demand_h = new_capacity_dimensions_h[demand_name]["demand"]
    demand_h.extend(new_order_data["demand"])
    capacity_h = new_capacity_dimensions_h[demand_name]["capacity"]
    new_d.add_capacity_dimension(
        demand_name, cudf.Series(demand_h), cudf.Series(capacity_h)
    )

    constraints = generate_capacity_constraints_for_vehicle_order_match(
        new_pickedup_order_to_vehicle,
        new_pickup_indices_h,
        new_delivery_indices_h,
        vehicle_num,
        len(new_order_locations),
    )

    for name, constraint in constraints.items():
        new_d.add_capacity_dimension(
            name,
            cudf.Series(constraint["demand"]),
            cudf.Series(constraint["capacity"]),
        )

    new_earliest = cudf.Series(new_earliest_time_h)
    new_latest = cudf.Series(new_latest_time_h)
    new_service = cudf.Series(new_service_time_h)
    new_d.set_order_time_windows(new_earliest, new_latest)
    new_d.set_order_service_times(new_service)

    if print_debug_info:
        print(f"new pickup indices: {new_pickup_indices_h}")
        print(f"new delivery indices: {new_delivery_indices_h}")

    new_pickup_indices = cudf.Series(new_pickup_indices_h)
    new_delivery_indices = cudf.Series(new_delivery_indices_h)
    new_d.set_pickup_delivery_pairs(new_pickup_indices, new_delivery_indices)

    new_d.set_vehicle_time_windows(
        cudf.Series(vehicle_earliest_h), cudf.Series(vehicle_latest_h)
    )

    new_d.set_vehicle_locations(
        cudf.Series(vehicle_start_locations_h),
        cudf.Series(vehicle_return_locations_h),
    )

    drop_return = [1] * vehicle_num
    drop_return = cudf.Series(drop_return)
    new_d.set_drop_return_trips(drop_return)

    return new_d


def generate_capacity_constraints_for_vehicle_order_match(
    pickup_order_to_vehicle,
    pickup_indices,
    delivery_indices,
    vehicle_num,
    order_num,
):
    delivery_of_pickup = {
        pickup_indices[i]: delivery_indices[i]
        for i in range(len(pickup_indices))
    }

    orders_per_vehicle = [0] * vehicle_num

    for v in pickup_order_to_vehicle.values():
        orders_per_vehicle[v] = orders_per_vehicle[v] + 1

    demand_1_of_orders_in_vehicle = [0] * vehicle_num
    demand_2_of_orders_in_vehicle = [0] * vehicle_num

    prev_count = 0
    prev_demand = 0
    for v in range(vehicle_num):
        if orders_per_vehicle[v] > 0:
            demand_1_of_orders_in_vehicle[v] = prev_count * prev_demand + 1
            prev_count = orders_per_vehicle[v]
            prev_demand = demand_1_of_orders_in_vehicle[v]

    prev_count = 0
    prev_demand = 0
    for i in range(0, vehicle_num):
        v = vehicle_num - i - 1
        if orders_per_vehicle[v] > 0:
            demand_2_of_orders_in_vehicle[v] = prev_count * prev_demand + 1
            prev_count = orders_per_vehicle[v]
            prev_demand = demand_2_of_orders_in_vehicle[v]

    demand1 = [0] * order_num
    demand2 = [0] * order_num

    for order, v in pickup_order_to_vehicle.items():
        cap1 = demand_1_of_orders_in_vehicle[v]
        cap2 = demand_2_of_orders_in_vehicle[v]

        demand1[order] = cap1
        demand2[order] = cap2

        delivery_order = delivery_of_pickup[order]
        demand1[delivery_order] = -cap1
        demand2[delivery_order] = -cap2

    capacity1 = [
        demand_1_of_orders_in_vehicle[v] * orders_per_vehicle[v]
        for v in range(vehicle_num)
    ]
    capacity2 = [
        demand_2_of_orders_in_vehicle[v] * orders_per_vehicle[v]
        for v in range(vehicle_num)
    ]

    constraint_names = [
        "force-vehicle-order-pair-1",
        "force-vehicle-order-pair-2",
    ]
    capacity_dimensions = {
        constraint_names[0]: {"demand": demand1, "capacity": capacity1},
        constraint_names[1]: {"demand": demand2, "capacity": capacity2},
    }

    return capacity_dimensions
