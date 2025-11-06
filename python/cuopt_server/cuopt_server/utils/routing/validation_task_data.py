# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np


def validate_task_data(
    task_ids,
    task_locations,
    demand,
    pickup_and_delivery_pairs,
    task_time_windows,
    task_service_times,
    prizes,
    order_vehicle_match,
    updating=False,
    comparison_locations=None,
):
    if updating:
        if comparison_locations is None:
            return (
                False,
                "No task data to update. The set_task_data endpoint must be called for the update_task_data endpoint to become active",  # noqa
            )
        if (task_locations is not None) and (
            len(task_locations) != len(comparison_locations)
        ):
            return (
                False,
                "Task update location array are not the same length as task set location array. Use set_task_data instead.",  # noqa
            )
        if task_locations is None:
            task_locations = comparison_locations

    if (task_locations is not None) and (task_locations.min() < 0):
        return (
            False,
            "task_locations represent index locations and must be greater than or equal to 0",  # noqa
        )

    # the length of relevant task property arrays must be equal n_tasks
    n_tasks = len(task_locations)

    task_length_check_array = [n_tasks]

    if task_ids is not None:
        task_length_check_array.append(len(task_ids))

    if demand is not None:
        task_length_check_array.append(len(demand[0]))
        # Every demand dimension must be of length n_tasks
        for demand_dim in demand:
            if len(demand_dim) != n_tasks:
                return (
                    False,
                    "All demand dimensions must have length equal to the number of tasks",  # noqa
                )

    # Pickup and delivery checks
    if pickup_and_delivery_pairs is not None:
        pickup_and_delivery_set = set(
            np.array(pickup_and_delivery_pairs).flatten()
        )
        # Check that values are greater than or equal to zero
        if min(pickup_and_delivery_set) < 0:
            return (
                False,
                "pickup_and_delivery_pairs represent order index and must be greater than or equal to 0",  # noqa
            )

        # And since pickup and delivery indices are order indices,
        # they should lie within [0, n_tasks).
        task_index_set = set(range(0, n_tasks))

        if pickup_and_delivery_set != task_index_set:
            return (
                False,
                "pickup_and_delivery_pairs assignments must be in the set of task/order indices and all task location indices must be used",  # noqa
            )

    # Task time windows checks
    if task_time_windows is not None:
        task_length_check_array.append(len(task_time_windows))

        # All time windows earliest times must be less than latest times
        for time_window in task_time_windows:
            # Time window must be of length 2
            if len(time_window) != 2:
                return (
                    False,
                    "All task_time_windows must be of length 2. 0:earliest, 1:latest",  # noqa
                )

            # All time window values must be 0 or greater
            if min(time_window) < 0:
                return (
                    False,
                    "task_time_windows must be greater than or equal to 0",
                )

            # earliest times must be before latest times
            if time_window[1] < time_window[0]:
                return (
                    False,
                    "All task time windows must have task_x_time_window[0] < task_x_time_window[1]",  # noqa
                )

    # Check task service times
    if task_service_times is not None:
        if type(task_service_times) is list:
            if min(task_service_times) < 0:
                return (
                    False,
                    "service_times must be greater than or equal to 0",
                )  # noqa

            task_length_check_array.append(len(task_service_times))
        else:
            for v_id, service_times in task_service_times.items():
                if min(service_times) < 0:
                    return (
                        False,
                        "service_times must be greater than or equal to 0",
                    )  # noqa
                task_length_check_array.append(len(service_times))

    if not (
        all(x == task_length_check_array[0] for x in task_length_check_array)
    ):
        return (
            False,
            "All arrays defining task properties must be of consistent length",
        )

    if prizes is not None:
        if len(prizes) != len(task_locations):
            return (
                False,
                "Size of the task prizes should be equal to number of tasks",  # noqa
            )

    if order_vehicle_match is not None:
        all_order_ids = [data.order_id for data in order_vehicle_match]
        min_vehicle_id = min(
            [min(data.vehicle_ids) for data in order_vehicle_match]
        )
        if max(all_order_ids) >= len(task_locations) or min(all_order_ids) < 0:
            return (
                False,
                "One or more Order IDs provided are not in the expected range in task vehicle match, should be within [0,  len(Task Locations) )",  # noqa
            )
        if min_vehicle_id < 0:
            return (
                False,
                "vehicle Id should be greater than or equal to zero",
            )

    return (True, "Valid Task Data")
