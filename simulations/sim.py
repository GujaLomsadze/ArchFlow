import datetime
import random
import time
from collections import defaultdict
from pprint import pprint

from functions.analysis.main import analyze_hits_dict, calculate_remaining_capacity
from functions.inserts.main import insert_node_hits, insert_remaining_capacity, fetch_node_capacity
from functions.node_data_manipulation.change import update_link_style_parameter_in_redis, \
    increment_link_style_parameter_in_redis, decrement_link_style_parameter_in_redis
from functions.traversal.main import find_edge_ids_for_path, traverse_markov_chain_n_times
from functions.utils.main import fluctuate_integer


def precompute_probabilities(graph):
    """
    Precompute edge weight probabilities for each node's neighbors.

    Args:
        graph: A NetworkX graph object.

    Returns:
        A dict mapping each node to another dict mapping its neighbors to their transition probabilities.
    """
    probabilities = {}
    for node in graph.nodes():
        neighbors = list(graph.neighbors(node))
        if not neighbors:
            probabilities[node] = {}
            continue

        weights = [graph[node][neighbor].get('weight', 1.0) for neighbor in neighbors]
        total_weight = sum(weights)
        probabilities[node] = {neighbor: weight / total_weight for neighbor, weight in zip(neighbors, weights)}

    return probabilities


def traverse_weighted_graph_n_times_optimized(graph, start_node, N, probabilities):
    """
    Optimized version of traversing a weighted graph N times.

    Args:
        graph: A NetworkX graph object.
        start_node: The starting node for the traversal.
        N: The number of times to traverse the graph.
        probabilities: Precomputed probabilities for neighbor selection.

    Yields:
        A list representing a single traversal path.
    """
    for _ in range(N):
        temp_path = [start_node]
        current_node = start_node

        while True:
            neighbors_probs = probabilities.get(current_node)
            if not neighbors_probs:  # No neighbors to move to
                break

            # Choose the next node based on precomputed probabilities
            next_node = random.choices(list(neighbors_probs.keys()), weights=list(neighbors_probs.values()))[0]
            if current_node == next_node and graph.has_edge(current_node, current_node):  # Self-loop check
                break

            temp_path.append(next_node)
            current_node = next_node

        yield temp_path


def traverse_any_given_graph_and_visualize(global_rps, global_rps_fluctuation_percentage, node_capacity_data, pg_conn,
                                           redis_conn, method, graph, edges,
                                           start_node, parameter,
                                           number_of_simulations,
                                           increment_amount, markovian=False):
    r = redis_conn

    edge_hit_count = defaultdict(int)
    node_hit_count = defaultdict(int)

    sim_clock = datetime.datetime.now()

    live = False
    intensity = False
    sim = False

    global_timer = 0
    index_counter = 0

    if method == "live":
        live = True
    if method == "intensity":
        intensity = True
    if method == "sim":
        sim = True

    probabilities = precompute_probabilities(graph)
    traverse_paths = traverse_weighted_graph_n_times_optimized(graph=graph, start_node=start_node,
                                                               N=number_of_simulations if not sim else global_rps * 9999999,
                                                               probabilities=probabilities)

    if markovian:
        traverse_paths = traverse_markov_chain_n_times(graph, start_node, global_rps)

    if sim:
        for path in traverse_paths:
            global_rps_fluctuated = fluctuate_integer(global_rps, global_rps_fluctuation_percentage)

            for node in path:
                node_hit_count[node] += 1

            edge_ids = find_edge_ids_for_path(edges, path)
            r.set("paths_traversed", index_counter)

            if index_counter % global_rps_fluctuated == 0 and index_counter != 0:
                global_timer += 1

                r.set("global_timer", global_timer)

                # print(
                #     f"========================================= ONE SECOND ELAPSED. CURRENT SECOND AFTER START: {global_timer}")

                result = analyze_hits_dict(node_hit_count)

                node_hit_data = result["nodes_info"]

                nodes_in_path = list(result["nodes_info"].keys())

                node_capacity_data = fetch_node_capacity(pg_connection=pg_conn)

                remaining_capacities = calculate_remaining_capacity(node_hit_data=node_hit_data,
                                                                    nodes_in_path=nodes_in_path,
                                                                    initial_capacities=node_capacity_data)

                # TODO: This is hardcoded for the CPU IO/PS. Should be dynamic for any given benchmark for the future
                for node_d, capacity_d in remaining_capacities.items():
                    cpu_remaining = capacity_d["cpu_io_capacity"]
                    if cpu_remaining <= 0 and node_d != start_node:
                        print(f"NEEDS TO BE SCALED UP/OUT: {node_d}")

                node_hit_count = defaultdict(int)
                sim_clock += datetime.timedelta(seconds=1)

                insert_remaining_capacity(remaining_capacities_in=remaining_capacities, insert_timestamp=sim_clock,
                                          pg_connection=pg_conn)
                insert_node_hits(result, sim_clock, pg_connection=pg_conn)

                # print(f"Number of Hits per Edge: {dict(edge_hit_count)}")
                # print(f"Number of Hits per Node: {dict(node_hit_count)}")

            for stage_level, edge_to_color in enumerate(edge_ids):
                stage = stage_level + 1  # TODO: Stage here is Bounce number from a starting node

                edge_hit_count[edge_to_color] += 1

                increment_link_style_parameter_in_redis(r, link_id=edge_to_color,
                                                        parameter=parameter, increment_amount=increment_amount)

            index_counter += 1

    if live:
        for path in traverse_paths:
            edge_ids = find_edge_ids_for_path(edges, path)

            for edge_to_color in edge_ids:
                update_link_style_parameter_in_redis(r, link_id=edge_to_color, parameter=parameter, new_value=1000)

            time.sleep(0.1)
            for edge_to_color in edge_ids:
                update_link_style_parameter_in_redis(r, link_id=edge_to_color, parameter=parameter, new_value=0)

            index_counter += 1

    if intensity:

        # TODO: INDEX IS ONE UNIQUE REQUEST PASSED THROUGH THE GRAPH
        # TODO: INDEX IS ONE UNIQUE REQUEST PASSED THROUGH THE GRAPH
        # TODO: INDEX IS ONE UNIQUE REQUEST PASSED THROUGH THE GRAPH

        # if index % 1000 == 0:
        #     print("DECREMENT TIME")
        #     for stage_level, edge_to_color in enumerate(edge_ids):
        #         decrement_link_style_parameter_in_redis(r, link_id=edge_to_color,
        #                                                 parameter=parameter,
        #
        #                                                 decrement_amount=increment_amount * index)

        for path in traverse_paths:
            edge_ids = find_edge_ids_for_path(edges, path)

            for stage_level, edge_to_color in enumerate(edge_ids):
                stage = stage_level + 1  # TODO: Stage here is Bounce number from a starting node

                increment_link_style_parameter_in_redis(r, link_id=edge_to_color,
                                                        parameter=parameter, increment_amount=increment_amount)

            index_counter += 1


def simulate_errors(graph_in, error_rate, error_weight_in=0.01):
    """
    Simulates errors in a graph by adding self-edges to a random selection of nodes.

    Args:
        graph_in: A NetworkX graph object representing the data architecture.
        error_rate (float): The percentage (0.0 to 1.0) of nodes to introduce errors to.
        error_weight_in (float): Error probability for a given Node

    Returns:
        A new NetworkX graph object with the simulated errors added.
    """

    # Create a copy of the original graph to avoid modifying it directly
    G_with_errors = graph_in.copy()

    new_edges_w_errors = []

    # Select a random set of nodes for error injection
    num_nodes = G_with_errors.number_of_nodes()
    num_errors = int(error_rate * num_nodes)
    error_nodes = random.sample(list(G_with_errors.nodes()), num_errors)

    # Add self-edges to the selected nodes with a chance of error weight
    for node in error_nodes:
        edge_id = f"{node}_{node}"  # f-string for formatted string literal

        G_with_errors.add_edge(node, node, id=edge_id)
        G_with_errors.edges[node, node]['weight'] = error_weight_in
        new_edges_w_errors.append([node, node, edge_id])

    return G_with_errors, new_edges_w_errors
