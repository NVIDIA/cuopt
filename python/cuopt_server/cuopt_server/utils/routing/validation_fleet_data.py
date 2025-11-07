# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


def test_time_window(time_windows, tw_type):
    # All time windows earliest times must be less than latest times
    for time_window in time_windows:
        if len(time_window) != 2:
            return (
                False,
                tw_type
                + ": Time windows for each vehicle must be of length 2. 0: earliest, 1: latest",  # noqa
            )
        if min(time_window) < 0:
            return (
                False,
                tw_type
                + ": All vehicle time window values must be greater than or equal to 0",  # noqa
            )
        if time_window[1] < time_window[0]:
            return (
                False,
                tw_type
                + ": All vehicles time windows must have vehicle_x_time_window[0] < vehicle_x_time_window[1]",  # noqa
            )

    return (True, "")


def validate_fleet_data(
    vehicle_ids,
    vehicle_locations,
    capacities,
    vehicle_time_windows,
    vehicle_breaks,
    vehicle_break_time_windows,
    vehicle_break_durations,
    vehicle_break_locations,
    vehicle_types,
    vehicle_types_dict,
    vehicle_order_match,
    skip_first_trips,
    drop_return_trips,
    min_vehicles,
    vehicle_max_costs,
    vehicle_max_times,
    vehicle_fixed_costs,
    updating=False,
    comparison_locations=None,
):
    if vehicle_locations is not None:
        for loc in vehicle_locations:
            if len(loc) != 2:
                return (
                    False,
                    "Vehicle locations should be list of pairs of start and end location for each vehicle",  # noqa
                )

    if (updating) and (comparison_locations is None):
        return (
            False,
            "No fleet data to update. The set_fleet_data endpoint must be called for the update_fleet_data endpoint to become active",  # noqa
        )

    if vehicle_locations and min(min(vehicle_locations)) < 0:
        return (
            False,
            "Fleet locations represent index locations and must be greater than or equal to 0",  # noqa
        )

    if updating:
        if (vehicle_locations) and (
            len(vehicle_locations) != len(comparison_locations)
        ):
            return (
                False,
                "If updating vehicle locations the length of vehicle locations must be equal to the vehicle location array passed during set_fleet_data. Use set_fleet_data instead.",  # noqa
            )

        n_vehicles = len(comparison_locations)
    else:
        n_vehicles = len(vehicle_locations)

    fleet_length_check_array = [n_vehicles]

    if vehicle_ids is not None:
        fleet_length_check_array.append(len(vehicle_ids))

    if capacities is not None:
        fleet_length_check_array.append(len(capacities[0]))
        # Every capacity dimension must be of length n_vehicles
        for capacity_dim in capacities:
            if min(capacity_dim) < 0:
                return (
                    False,
                    "All capacity dimensions values must be 0 or greater",
                )
            if len(capacity_dim) != n_vehicles:
                return (
                    False,
                    "All capacity dimensions must have length equal to the number of vehicles",  # noqa
                )

    if vehicle_max_costs is not None:
        if min(vehicle_max_costs) <= 0:
            return (
                False,
                "Maximum distance any vehicle can travel must be greater "
                "than 0",
            )
        fleet_length_check_array.append(len(vehicle_max_costs))

    if vehicle_max_times is not None:
        if min(vehicle_max_times) <= 0:
            return (
                False,
                "Maximum time any vehicle can travel must be greater than 0",
            )
        fleet_length_check_array.append(len(vehicle_max_times))

    if vehicle_fixed_costs is not None:
        if min(vehicle_fixed_costs) < 0:
            return (
                False,
                "Fixed cost of vehicle must be greater than or equal to 0",
            )
        fleet_length_check_array.append(len(vehicle_fixed_costs))

    if vehicle_time_windows is not None:
        fleet_length_check_array.append(len(vehicle_time_windows))
        res = test_time_window(vehicle_time_windows, "vehicle_time_windows")
        if not res[0]:
            return res

    if vehicle_breaks is not None:
        if (
            vehicle_break_durations is not None
            or vehicle_break_time_windows is not None
            or vehicle_break_locations is not None
        ):
            return (
                False,
                'vehicle_breaks should not be used together with homogenous break, which is set by "vehicle_break_time_windows", "vehicle_break_durations" and "vehicle_break_locations"',  # noqa
            )

    if (
        vehicle_break_durations is not None
        and vehicle_break_time_windows is None
    ) or (  # noqa
        vehicle_break_durations is None
        and vehicle_break_time_windows is not None
    ):  # noqa
        return (
            False,
            "vehicle_break_time_windows and vehicle_break_durations should be used together",  # noqa
        )

    if vehicle_break_time_windows is not None:
        for time_windows in vehicle_break_time_windows:
            fleet_length_check_array.append(len(time_windows))
            res = test_time_window(time_windows, "vehicle_break_time_windows")
            if not res[0]:
                return res

    if vehicle_break_durations is not None:
        if min(min(vehicle_break_durations)) < 0:
            return (
                False,
                "Vehicle break duration must be greater than or equal to 0",
            )

    if vehicle_break_locations is not None:
        if min(vehicle_break_locations) < 0:
            return (
                False,
                "Vehicle break location must be greater than or equal to 0",
            )

    if vehicle_types is not None:
        unique_vehicle_types = set(vehicle_types)
        for matrix_type, vehicle_ids in vehicle_types_dict.items():
            v_ids = set(vehicle_ids)
            if len(v_ids) > 0 and not unique_vehicle_types.issubset(v_ids):
                return (False, matrix_type + " not set for all vehicle types")
    else:
        for _, vehicle_ids in vehicle_types_dict.items():
            if len(set(vehicle_ids)) > 1:
                return (
                    False,
                    "Set vehicle types when using multiple matrices",
                )

    if vehicle_order_match is not None:
        all_vehicle_ids = [data.vehicle_id for data in vehicle_order_match]
        min_order_id = min(
            [min(data.order_ids) for data in vehicle_order_match]
        )
        if max(all_vehicle_ids) >= n_vehicles or min(all_vehicle_ids) < 0:
            return (
                False,
                "One or more Vehicle IDs provided are not in the expected range, should be within [0,  number of vehicle )",  # noqa
            )
        if min_order_id < 0:
            return (False, "Order Id should be greater than or equal to zero")

    if skip_first_trips is not None:
        fleet_length_check_array.append(len(skip_first_trips))

    if drop_return_trips is not None:
        fleet_length_check_array.append(len(drop_return_trips))

    length_check_set = set(fleet_length_check_array)
    if len(length_check_set) > 1:
        return (
            False,
            "All arrays defining vehicle properties must be of consistent length",  # noqa
        )

    if (updating) and (
        next(iter(length_check_set)) != len(comparison_locations)
    ):
        return (
            False,
            "All arrays updating vehicle properties must be same length as the vehicle data arrays being updated. If overwriting fleet data, use the set_fleet_data endpoint",  # noqa
        )

    if (min_vehicles is not None) and (min_vehicles <= 0):
        return (False, "Minimum vehicles must be greater than 0")

    return (True, "Valid Fleet Data")
