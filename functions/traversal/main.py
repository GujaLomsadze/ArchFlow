import numpy as np


def find_edge_ids_for_path(edges, path):
    """
    Retrieves a list of edge IDs for the given path of nodes, ensuring all transitions
    are accounted for, from the first node to the last.

    :param edges: List of edges where each edge is represented as [from, to, id]
    :param path: List of node IDs representing the path
    :return: List of edge IDs corresponding to the transitions between nodes in the path
    """
    edge_ids = []
    for i in range(len(path) - 1):  # Iterate through the path to get each transition
        from_node = path[i]
        to_node = path[i + 1]

        # Find and append the edge ID for each transition
        edge_id = next((edge[2] for edge in edges if edge[0] == from_node and edge[1] == to_node), None)
        if edge_id:
            edge_ids.append(edge_id)
        else:
            print(f"Could not find Edge ID for transition: {from_node} -> {to_node}")

    return edge_ids


def transition_probability_matrix(graph):
    """
    Compute the transition probability matrix from the weighted directed graph.

    Args:
        graph: A NetworkX directed graph object with weighted edges representing transition probabilities.

    Returns:
        np.array: Transition probability matrix.
    """
    num_nodes = len(graph.nodes)
    transition_matrix = np.zeros((num_nodes, num_nodes))

    for i, node in enumerate(graph.nodes):
        total_weight = sum(data['weight'] for _, data in graph[node].items())
        if total_weight == 0:
            # If total_weight is zero, set transition probabilities to uniform distribution
            transition_matrix[i] = 1 / num_nodes
        else:
            for j, neighbor in enumerate(graph.nodes):
                if neighbor in graph[node]:
                    # Consider only outgoing edges for directed graph
                    if graph.has_edge(node, neighbor):
                        transition_matrix[i][j] = graph[node][neighbor]['weight'] / total_weight

    # Normalize transition probabilities to ensure they sum up to 1
    transition_matrix /= transition_matrix.sum(axis=1, keepdims=True)

    print("Transition Probability Matrix:")
    print(transition_matrix)

    return transition_matrix


def traverse_markov_chain_n_times(graph, start_node, N):
    """
    Generate paths based on Markov chain transitions using matrix exponentiation.

    Args:
        graph: A NetworkX graph object with weighted edges representing transition probabilities.
        start_node: The starting node for the traversal.
        N: The number of times to traverse the graph.

    Yields:
        list: A single traversal path.
    """
    transition_matrix = transition_probability_matrix(graph)
    num_nodes = len(graph.nodes)
    current_state = list(graph.nodes).index(start_node)

    for _ in range(N):
        temp_path = [start_node]

        for _ in range(num_nodes - 1):
            next_state = np.random.choice(range(num_nodes), p=transition_matrix[current_state])
            temp_path.append(list(graph.nodes)[next_state])
            current_state = next_state

        yield temp_path
