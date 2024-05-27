import copy
import json
import math
from collections import deque
from pprint import pprint

import community as community_louvain
import networkx as nx
import numpy as np


def create_graph_from_adjacency_matrix(adjacency_matrix):
    return nx.from_numpy_array(adjacency_matrix, create_using=nx.DiGraph)


def calculate_degrees(G):
    in_degrees = dict(G.in_degree())
    out_degrees = dict(G.out_degree())
    return in_degrees, out_degrees


def calculate_shortest_paths(G):
    return dict(nx.shortest_path_length(G))


def calculate_centrality_measures(G):
    betweenness_centrality = nx.betweenness_centrality(G)
    closeness_centrality = nx.closeness_centrality(G)
    return betweenness_centrality, closeness_centrality


def calculate_clustering_coefficient(G):
    # For directed graphs, convert to undirected for clustering calculation
    undirected_G = G.to_undirected()
    return nx.average_clustering(undirected_G)


def calculate_graph_density(G):
    return nx.density(G)


def calculate_connectivity(G):
    # For directed graphs, this will calculate strongly connected components
    return [len(c) for c in sorted(nx.strongly_connected_components(G), key=len, reverse=True)]


def calculate_community_detection(G):
    # This requires converting to undirected graph for the community detection algorithm
    undirected_G = G.to_undirected()
    from networkx.algorithms import community
    communities = community.greedy_modularity_communities(undirected_G)
    return [list(community) for community in communities]


def get_graph_stats(adjacency_matrix):
    """
    Function to get Network basic analysis from Adjacency Matrix
    :param adjacency_matrix: Obvious.
    :return:
    """

    adjacency_matrix = np.array(adjacency_matrix)

    # Create a graph from the adjacency matrix
    G = create_graph_from_adjacency_matrix(adjacency_matrix)

    # Perform calculations
    in_degrees, out_degrees = calculate_degrees(G)
    shortest_paths = calculate_shortest_paths(G)
    betweenness_centrality, closeness_centrality = calculate_centrality_measures(G)
    clustering_coefficient = calculate_clustering_coefficient(G)
    density = calculate_graph_density(G)
    connectivity = calculate_connectivity(G)
    communities = calculate_community_detection(G)

    # Example output
    print(f"In-Degrees: {in_degrees}")
    print(f"Out-Degrees: {out_degrees}")
    print(f"Shortest Paths: {shortest_paths}")
    print(f"Betweenness Centrality: {betweenness_centrality}")
    print(f"Closeness Centrality: {closeness_centrality}")
    print(f"Clustering Coefficient: {clustering_coefficient}")
    print(f"Density: {density}")
    print(f"Connectivity: {connectivity}")
    print(f"Communities: {communities}")

    return in_degrees, out_degrees, shortest_paths, betweenness_centrality, \
        closeness_centrality, clustering_coefficient, density, connectivity, communities


def calculate_transition_probabilities(graph, start_node):
    """
    Calculates transition probabilities from the given start node to all ending nodes.
    This version includes cycle detection to prevent infinite recursion.

    Args:
        graph: A NetworkX DiGraph.
        start_node: The starting node for the transition calculation.

    Returns:
        A dictionary where keys are ending nodes and values are transition probabilities.
    """

    probabilities = {}
    ending_nodes = [node for node in graph.nodes() if graph.out_degree(node) == 0]
    visited = set()  # Keep track of visited nodes to avoid cycles

    def dfs(node, path_probability):
        if node in visited:
            return  # Avoid cycles
        visited.add(node)

        if node in ending_nodes:
            if node in probabilities:
                probabilities[node] += path_probability
            else:
                probabilities[node] = path_probability
            visited.remove(node)  # Backtrack: remove from visited set
            return

        for neighbor in graph.successors(node):
            probability = graph[node][neighbor].get('weight', 1)
            dfs(neighbor, path_probability * probability)

        visited.remove(node)  # Backtrack: remove from visited set

    dfs(start_node, 1.0)

    return probabilities


def calculate_all_transition_probabilities(graph, start_node):
    """
    Calculates transition probabilities from the given start node to all nodes in the graph.
    This version includes cycle detection to prevent infinite recursion and accumulates
    probabilities at each node.

    Args:
        graph: A NetworkX DiGraph.
        start_node: The starting node for the transition calculation.

    Returns:
        A dictionary where keys are node identifiers and values are transition probabilities.
    """

    # Initialize probabilities for all nodes with 0, except the start node with 1
    probabilities = {node: 0 for node in graph.nodes()}
    probabilities[start_node] = 1.0
    visited = set()  # Keep track of visited nodes to avoid cycles

    def dfs(node, path_probability):
        if node in visited:
            return  # Avoid cycles
        visited.add(node)

        # Distribute current path probability to all successors
        total_weight = sum(graph[node][neighbor].get('weight', 1) for neighbor in graph.successors(node))
        for neighbor in graph.successors(node):
            edge_weight = graph[node][neighbor].get('weight', 1)
            # Adjust the path_probability by the edge weight and total outgoing weight
            successor_probability = path_probability * (edge_weight / total_weight if total_weight else 0)
            probabilities[neighbor] += successor_probability
            dfs(neighbor, successor_probability)

        visited.remove(node)  # Backtrack: remove from visited set

    dfs(start_node, 1.0)

    return probabilities


def calculate_relative_depth_probabilities(graph, start_node):
    """
    Calculates transition probabilities from the given start node to all nodes in the graph,
    with each depth level computed relative to that level only.

    Args:
        graph: A NetworkX DiGraph.
        start_node: The starting node for the transition calculation.

    Returns:
        A dictionary where keys are node identifiers and values are transition probabilities.
    """
    probabilities = {node: 0 for node in graph.nodes()}  # Initialize probabilities
    probabilities[start_node] = 1.0  # Start node probability is 100%

    # Store nodes to visit with their associated path probability
    to_visit = [(start_node, 1.0)]

    while to_visit:
        current_level = []
        next_level = []

        # Process nodes at the current level
        for node, path_probability in to_visit:
            # Calculate total weight for normalization
            total_weight = sum(graph[node][neighbor].get('weight', 1) for neighbor in graph.successors(node))
            for neighbor in graph.successors(node):
                edge_weight = graph[node][neighbor].get('weight', 1)
                # Adjust path_probability for the edge weight relative to total outgoing weight
                if total_weight > 0:
                    relative_probability = (edge_weight / total_weight) * path_probability
                else:
                    relative_probability = 0
                probabilities[neighbor] += relative_probability
                next_level.append((neighbor, probabilities[neighbor]))

        to_visit = next_level  # Move to the next level

    return probabilities


def get_relative_color(probabilities):
    """
    Maps each probability to a color on a green-yellow-red scale based on its relative position.

    Args:
    - probabilities: A dictionary of node identifiers to their probabilities.

    Returns:
    - A dictionary of node identifiers to their color codes.
    """
    min_prob = min(probabilities.values())
    max_prob = max(probabilities.values())
    color_map = {}

    for node, prob in probabilities.items():
        if max_prob - min_prob == 0:
            # Avoid division by zero if all probabilities are the same
            relative_position = 0.5
        else:
            relative_position = (prob - min_prob) / (max_prob - min_prob)

        if relative_position <= 0.5:
            # Scale from green (0) to yellow (0.5)
            red = int(2 * relative_position * 255)
            green = 255
        else:
            # Scale from yellow (0.5) to red (1)
            red = 255
            green = int((1 - 2 * (relative_position - 0.5)) * 255)

        blue = 0
        color_map[node] = f"#{red:02x}{green:02x}{blue:02x}"

    return color_map


def calculate_evenness(probabilities):
    """
    Calculates the evenness of the probability distribution.
    Lower values indicate a more uniform (even) distribution.

    Args:
    - probabilities: A dictionary of node identifiers to their probabilities.

    Returns:
    - Coefficient of variation (CV) as a measure of distribution evenness.
    """
    prob_values = list(probabilities.values())
    mean_prob = np.mean(prob_values)
    std_dev_prob = np.std(prob_values)

    # Avoid division by zero in case mean_prob is 0
    if mean_prob > 0:
        cv = std_dev_prob / mean_prob
    else:
        cv = np.inf  # Infinite or undefined when mean is 0

    return cv


def adjust_color_by_level(probabilities, node_levels):
    """
    Adjusts the color mapping to consider probabilities relative to their level.

    Args:
    - probabilities: A dictionary of node identifiers to their probabilities.
    - node_levels: A dictionary of node identifiers to their levels.

    Returns:
    - A dictionary of node identifiers to their color codes.
    """
    # Determine min and max probabilities per level
    level_probabilities = {}
    for node, prob in probabilities.items():
        level = node_levels[node]
        if level in level_probabilities:
            level_probabilities[level]['min'] = min(level_probabilities[level]['min'], prob)
            level_probabilities[level]['max'] = max(level_probabilities[level]['max'], prob)
        else:
            level_probabilities[level] = {'min': prob, 'max': prob}

    # Map probabilities to colors, considering the min and max per level
    color_map = {}
    for node, prob in probabilities.items():
        level = node_levels[node]
        min_prob, max_prob = level_probabilities[level]['min'], level_probabilities[level]['max']
        if max_prob - min_prob == 0:
            relative_position = 0.5
        else:
            relative_position = (prob - min_prob) / (max_prob - min_prob)

        if relative_position <= 0.5:
            red = int(2 * relative_position * 255)
            green = 255
        else:
            red = 255
            green = int((1 - 2 * (relative_position - 0.5)) * 255)
        blue = 0
        color_map[node] = f"#{red:02x}{green:02x}{blue:02x}"

    return color_map


def calculate_levels_and_probabilities(graph_inside, start_node):
    """
    Calculates levels in the graph_inside and transition probabilities from the start node to all nodes,
    with probabilities at each level computed relative to that level only.

    Args:
        graph_inside: A NetworkX DiGraph.
        start_node: The starting node for the transition calculation.

    Returns:
        A tuple containing two items:
        - A dictionary of nodes with their levels.
        - A dictionary where keys are node identifiers and values are transition probabilities.
    """
    levels = {}  # Node to level mapping
    probabilities = {node: 0 for node in graph_inside.nodes()}  # Initialize probabilities
    probabilities[start_node] = 1.0  # Start node probability is 100%

    queue = deque([(start_node, 0)])  # Queue for BFS: (node, level)

    while queue:
        node, level = queue.popleft()
        if node not in levels:  # First visit to node
            levels[node] = level

            total_weight = sum(
                graph_inside[node][neighbor].get('weight', 1) for neighbor in graph_inside.successors(node))
            for neighbor in graph_inside.successors(node):
                edge_weight = graph_inside[node][neighbor].get('weight', 1)
                if total_weight > 0:
                    relative_probability = (edge_weight / total_weight) * probabilities[node]
                else:
                    relative_probability = 0

                probabilities[neighbor] += relative_probability

                queue.append((neighbor, level + 1))

    return levels, probabilities


def identify_nodes_for_replication(graph_in, start_node, replication_percentile=99, min_probability_threshold=0.1):
    _, probabilities = calculate_levels_and_probabilities(graph_in, start_node)

    # Exclude start node from consideration
    probabilities.pop(start_node, None)

    # Calculate threshold based on the specified percentile
    percentile_threshold = np.percentile(list(probabilities.values()), replication_percentile)

    # Ensure the threshold is at least as high as the minimum probability threshold
    final_threshold = max(percentile_threshold, min_probability_threshold)

    # Identify nodes that meet or exceed the final threshold
    nodes_for_replication = [node for node, probability in probabilities.items() if probability >= final_threshold]

    return nodes_for_replication


def distribute_start_node_successor_weights_evenly(graph_cp, start_node):
    successors = list(graph_cp.successors(start_node))
    num_successors = len(successors)
    if num_successors > 0:
        even_weight = 1.0 / num_successors
        for succ in successors:
            # Adjust existing weights or add new edges with even weight
            graph_cp.add_edge(start_node, succ, weight=even_weight)


def replicate_nodes_in_graph_and_track_edges(graph_cp, start_node, replication_percentile=99,
                                             min_probability_threshold=10, rebalance_starting_node=True):
    nodes_for_replication = identify_nodes_for_replication(graph_cp, start_node, replication_percentile,
                                                           min_probability_threshold / 100)

    replicated_node_mapping = {}
    new_edges = []

    for node in nodes_for_replication:
        new_node = f"{node}_replicated"
        graph_cp.add_node(new_node)
        replicated_node_mapping[node] = new_node

        # Handle incoming edges from predecessors
        for pred in graph_cp.predecessors(node):
            edge_data = graph_cp.get_edge_data(pred, node).copy()
            edge_data.pop('id', None)
            new_edge_id = f"{pred}_{new_node}"
            graph_cp.add_edge(pred, new_node, **edge_data)
            new_edges.append([pred, new_node, new_edge_id])

        # Handle outgoing edges to successors
        for succ in graph_cp.successors(node):
            edge_data = graph_cp.get_edge_data(node, succ).copy()
            edge_data.pop('id', None)

            original_weight = edge_data.get('weight', 1)
            adjusted_weight = original_weight / 2  # Assuming 1 replication for simplicity

            graph_cp[node][succ]['weight'] = adjusted_weight

            new_edge_id = f"{new_node}_{succ}"
            edge_data['weight'] = adjusted_weight
            graph_cp.add_edge(new_node, succ, **edge_data)
            new_edges.append([new_node, succ, new_edge_id])

    if rebalance_starting_node:
        distribute_start_node_successor_weights_evenly(graph_cp, start_node)

    # Special handling for the starting node if it's in the replication list
    if start_node in nodes_for_replication:
        new_start_node = replicated_node_mapping[start_node]
        distribute_start_node_successor_weights_evenly(graph_cp, start_node)
        distribute_start_node_successor_weights_evenly(graph_cp, new_start_node)

    return graph_cp, new_edges


def identify_outliers(probabilities):
    """
    Identifies outliers in the node transition probabilities.

    Args:
    - probabilities: A dictionary of node identifiers to their probabilities.

    Returns:
    - A list of node identifiers considered as outliers.
    """

    prob_values = np.array(list(probabilities.values()))

    Q1 = np.percentile(prob_values, 25)
    Q3 = np.percentile(prob_values, 75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = [node for node, prob in probabilities.items() if prob < lower_bound or prob > upper_bound]

    return outliers


def grid_search_replication(graph_in, start_node):
    percentile_range = range(1, 100)  # Example range, adjust as needed
    threshold_range = range(1, 100)  # Example range, adjust as needed

    lowest_cv = np.inf
    best_config = None
    best_new_edges = None
    best_adjusted_graph = None

    # Iterate over all combinations of percentile and threshold values
    for percentile in percentile_range:
        for threshold in threshold_range:
            # Make a deep copy of the graph to avoid modifying the original
            graph_cp = copy.deepcopy(graph_in)
            # Apply replication with the current configuration
            adj_graph, new_edges = replicate_nodes_in_graph_and_track_edges(
                graph_cp, start_node, replication_percentile=percentile, min_probability_threshold=threshold)

            # Calculate the transition probabilities for the adjusted graph
            # Assuming a function exists to do so; you might need to adjust this part
            _, transition_probabilities = calculate_levels_and_probabilities(adj_graph, start_node)

            # Calculate the CV for the current configuration
            cv = calculate_evenness(transition_probabilities)

            print(f"Percentile: {percentile}, Threshold: {threshold}, CV: {cv:.2f}")

            if cv < lowest_cv:
                lowest_cv = cv
                best_config = (percentile, threshold)
                best_new_edges = new_edges
                best_adjusted_graph = adj_graph

    return best_config, lowest_cv, best_adjusted_graph, best_new_edges


def graph_analysis(graph):
    """
    Perform comprehensive graph analysis, including centrality measures, community detection,
    and other statistical analyses.

    Parameters:
    - graph: A NetworkX graph object

    Returns:
    A dictionary containing various analysis results.
    """
    analysis_results = {'degree_centrality': nx.degree_centrality(graph),
                        'betweenness_centrality': nx.betweenness_centrality(graph),
                        'closeness_centrality': nx.closeness_centrality(graph)}

    # Centrality Measures

    # Convert to undirected graph for community detection
    undirected_graph = graph.to_undirected()

    # Community Detection
    partition = community_louvain.best_partition(undirected_graph)
    analysis_results['communities'] = partition

    # Additional Statistics
    analysis_results['average_clustering'] = nx.average_clustering(undirected_graph)

    # Graph-level statistics
    analysis_results['density'] = nx.density(undirected_graph)
    analysis_results['is_connected'] = nx.is_connected(undirected_graph)

    return analysis_results


def results_to_html_table(results, save_to_file=True):
    """
    Convert analysis results to an HTML table.

    Parameters:
    - results: The analysis results dictionary.

    Returns:
    An HTML string representing the analysis results in a table format.
    """
    html_table = "<table border='1'>"
    html_table += "<tr><th>Metric</th><th>Value</th></tr>"

    for key, value in results.items():
        if isinstance(value, dict):
            # For dictionary values, convert dict to string or handle differently as needed
            value_str = "<br>".join(
                [f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in value.items()])
            html_table += f"<tr><td>{key}</td><td>{value_str}</td></tr>"
        else:
            html_table += f"<tr><td>{key}</td><td>{value}</td></tr>"

    html_table += "</table>"

    if save_to_file:
        with open("templates/graph_analysis.html", 'w') as file:
            file.write(html_table)

    return html_table


def analyze_hits_dict(hits_dict):
    """
    Function to calculate detailed analytical information for each node given Hit-rate for nodes
    :param hits_dict: Dictionary of hits. Key:Value = Node:Hit
    :return: Dictionary containing analytical information for each node
    """

    hits_per_node = hits_dict

    total_hits = sum(hits_per_node.values())
    average_hits_per_node = total_hits / len(hits_per_node)
    max_hits_node = max(hits_per_node, key=hits_per_node.get)
    min_hits_node = min(hits_per_node, key=hits_per_node.get)

    # Calculate detailed information for each node
    nodes_info = {}
    for node, hits in hits_per_node.items():
        node_info = {
            "hits": hits,
            "percentage_of_total_hits": (hits / total_hits) * 100
        }
        nodes_info[node] = node_info

    # Output analysis results
    analysis_results = {
        "total_hits": total_hits,
        "average_hits_per_node": average_hits_per_node,
        "max_hits_node": {max_hits_node: hits_per_node[max_hits_node]},
        "min_hits_node": {min_hits_node: hits_per_node[min_hits_node]},
        "nodes_info": nodes_info
    }

    return analysis_results


# Assume this function gets the initial capacities for a specific node
def get_initial_capacities(node):
    # Dummy data, replace with your actual capacity fetching logic
    return {
        'cpu_io_capacity': 10000,  # hypothetical initial capacity for CPU I/O operations
        'disk_io_capacity': 5000,  # hypothetical initial capacity for Disk I/O operations
        'ram_capacity': 16000,  # hypothetical initial capacity for RAM in MB
        'storage_capacity': 100000,  # hypothetical initial storage in MB
        'network_bandwidth_capacity': 10000,  # hypothetical initial network bandwidth in MBps
        'max_cpu_load': 100  # hypothetical maximum CPU load percentage
    }


def calculate_remaining_capacity(node_hit_data, nodes_in_path, initial_capacities):
    remaining_capacities = {}

    for node in nodes_in_path:
        if node in node_hit_data and node in initial_capacities:
            hits = node_hit_data[node]['hits']
            node_capacities = initial_capacities[node]

            # Calculate the remaining capacities only if the initial capacities are valid integers
            remaining_capacities[node] = {
                capacity: value - hits for capacity, value in node_capacities.items() if value is not None
            }

            # Check if the dictionary for the node remains empty after processing
            if not remaining_capacities[node]:
                remaining_capacities[node] = 'All capacities were non-integer or insufficient data'

        else:
            # Provide more specific information on what is missing
            if node not in node_hit_data:
                remaining_capacities[node] = 'No hit data available'
            elif node not in initial_capacities:
                remaining_capacities[node] = 'No initial capacity data available'
    return remaining_capacities


def convert_sets_to_lists(data):
    if isinstance(data, dict):
        return {k: convert_sets_to_lists(v) for k, v in data.items()}
    elif isinstance(data, set):
        return list(data)
    elif isinstance(data, list):
        return [convert_sets_to_lists(i) for i in data]
    else:
        return data


def analyze_graph(G, redis_connection, print_or_not=True):
    r = redis_connection
    # Function to analyze the given directed graph G

    # 1. Calculate centrality measures
    in_degree_centrality = nx.in_degree_centrality(G)
    out_degree_centrality = nx.out_degree_centrality(G)
    closeness_centrality = nx.closeness_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)

    # Perform calculations
    shortest_paths = calculate_shortest_paths(G)
    clustering_coefficient = calculate_clustering_coefficient(G)
    density = calculate_graph_density(G)
    connectivity = calculate_connectivity(G)
    communities = calculate_community_detection(G)

    # Example output

    scc = list(nx.strongly_connected_components(G))

    undirected_G = G.to_undirected()
    articulation_points = list(nx.articulation_points(undirected_G))

    final_analysis = f"""
        Centrality Measures:
        In-Degree Centrality: {in_degree_centrality}
        Out-Degree Centrality: {out_degree_centrality}
        Closeness Centrality: {closeness_centrality}
        Betweenness Centrality: {betweenness_centrality}
        Shortest Paths: {shortest_paths}
        Betweenness Centrality: {betweenness_centrality}
        Closeness Centrality: {closeness_centrality}
        Clustering Coefficient: {clustering_coefficient}
        Density: {density}
        Connectivity: {connectivity}
        Communities: {communities}
        
        Strongly Connected Components: {scc}
        
        Articulation Points (using undirected version of the graph): {articulation_points}
    """

    analysis_data = {
        "In-Degree Centrality": in_degree_centrality,
        "Out-Degree Centrality": out_degree_centrality,
        "Closeness Centrality": closeness_centrality,
        "Betweenness Centrality": betweenness_centrality,
        "Shortest Paths": shortest_paths,
        "Clustering Coefficient": clustering_coefficient,
        "Density": density,
        "Connectivity": connectivity,
        "Communities": communities,
        "Strongly Connected Components": scc,
        "Articulation Points (using undirected version of the graph)": articulation_points
    }
    in_degree_centrality = json.dumps(in_degree_centrality)
    out_degree_centrality = json.dumps(out_degree_centrality)
    closeness_centrality = json.dumps(closeness_centrality)
    betweenness_centrality = json.dumps(betweenness_centrality)
    shortest_paths = json.dumps(shortest_paths)
    shortest_paths = json.dumps(shortest_paths)
    clustering_coefficient = json.dumps(clustering_coefficient)
    density = json.dumps(density)
    connectivity = json.dumps(connectivity)
    communities = json.dumps(communities)
    scc = json.dumps(convert_sets_to_lists(scc))
    articulation_points = json.dumps(articulation_points)

    r.set('in_degree_centrality', in_degree_centrality)
    r.set('out_degree_centrality', out_degree_centrality)
    r.set('closeness_centrality', closeness_centrality)
    r.set('betweenness_centrality', betweenness_centrality)
    r.set('shortest_paths', shortest_paths)
    r.set('shortest_paths', shortest_paths)
    r.set('clustering_coefficient', clustering_coefficient)
    r.set('density', density)
    r.set('connectivity', connectivity)
    r.set('communities', communities)
    r.set('scc', scc)
    r.set('articulation_points', articulation_points)

    if print_or_not:
        print(final_analysis)

    return final_analysis


def analyze_and_recommend(G):
    # Compute centrality measures
    in_degree = nx.in_degree_centrality(G)
    out_degree = nx.out_degree_centrality(G)
    closeness = nx.closeness_centrality(G)
    betweenness = nx.betweenness_centrality(G)

    # Convert to undirected for articulation points
    undirected_G = G.to_undirected()
    articulation_points = list(nx.articulation_points(undirected_G))

    # Strongly connected components
    scc = list(nx.strongly_connected_components(G))

    # Recommendations
    recommendations = []

    # Analyze nodes with high centrality measures for load balancing and redundancy
    critical_nodes = {node for node, value in betweenness.items() if
                      value > 0.01}  # Threshold for 'critical' might be adjusted
    if critical_nodes:
        recommendations.append(
            "Consider load balancing or redundancy for high-betweenness nodes: " + ", ".join(critical_nodes))

    # Check for nodes that are single points of failure
    if articulation_points:
        recommendations.append("Implement redundancy for articulation points to improve fault tolerance: " + ", ".join(
            articulation_points))

    # Evaluate the size of strongly connected components
    if all(len(comp) == 1 for comp in scc):
        recommendations.append(
            "The network may benefit from creating more reciprocal connections to enhance redundancy and resilience.")

    return {
        "In-Degree Centrality": in_degree,
        "Out-Degree Centrality": out_degree,
        "Closeness Centrality": closeness,
        "Betweenness Centrality": betweenness,
        "Articulation Points": articulation_points,
        "Strongly Connected Components": scc,
        "Recommendations": recommendations
    }
