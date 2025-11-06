# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np


def validate_waypoint_graph(
    waypoint_graph,
    is_travel_time=False,
    updating=False,
    comparison_waypoint_graph=None,
):
    if (updating or is_travel_time) and (len(comparison_waypoint_graph) == 0):
        return (
            False,
            "If adding a travel time waypoint graph or updating a waypoint graph, a waypoint_graph must already be set via the set endpoint",  # noqa
        )

    for v_type, graph in waypoint_graph.items():
        # If setting a travel time waypoint_graph
        if (is_travel_time) and (not updating):
            # topology must be consistent edges
            if not np.array_equal(
                graph["edges"], comparison_waypoint_graph[v_type]["edges"]
            ):  # noqa
                return (
                    False,
                    "Graph topology of primary and travel time waypoint graphs must match. Travel Time waypoint graph edges must match primary waypoint graph edges",  # noqa
                )

            # topology must be consistent offsets
            if not np.array_equal(
                graph["offsets"], comparison_waypoint_graph[v_type]["offsets"]
            ):  # noqa
                return (
                    False,
                    "Graph topology of primary and travel time waypoint graphs must match. Travel Time waypoint graph offsets must match primary waypoint graph offsets",  # noqa
                )

            # if setting weights for travel time waypoint_graph
            # they need to be of the right length
            if graph["weights"] is not None:
                if comparison_waypoint_graph[v_type]["weights"] is not None:
                    if len(graph["weights"]) != len(
                        comparison_waypoint_graph[v_type]["weights"]
                    ):
                        return (
                            False,
                            "The length of travel time waypoint graph weights must be equal to the length of primary waypoint graph weights",  # noqa
                        )
                else:
                    if len(graph["weights"]) != len(
                        comparison_waypoint_graph[v_type]["edges"]
                    ):
                        return (
                            False,
                            "The length of travel time waypoint graph weights must be equal to the length of primary waypoint graph edges if primary weights are not set)",  # noqa
                        )

        # Value validation

        if (graph["edges"] is not None) and (min(graph["edges"]) < 0):
            return (False, "edge values must be greater than or equal to 0")

        if (graph["offsets"] is not None) and (min(graph["offsets"]) < 0):
            return (False, "offset values must be greater than or equal to 0")

        if graph["weights"] is not None:
            if min(graph["weights"]) < 0:
                return (
                    False,
                    "weight values must be greater than or equal to 0",
                )

            # Length of weights array must be equal to length of edges
            if updating:
                compare_edges = comparison_waypoint_graph[v_type]["edges"]
            else:
                compare_edges = graph["edges"]

            if len(compare_edges) != len(graph["weights"]):
                return (
                    False,
                    "Length of weights array must be equal to edges array",
                )

        # If setting a primary waypoint_graph edges and
        # offsets need to be of the right proportions
        if (not updating) and (not is_travel_time):
            if len(graph["edges"]) < len(graph["offsets"]):
                return (
                    False,
                    "Length of edges array must be greater than or equal to the length of the offsets array",  # noqa
                )

    return (True, "Valid Waypoint Graph")
