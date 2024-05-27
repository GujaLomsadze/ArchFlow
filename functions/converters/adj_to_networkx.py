import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def create_directed_graph_from_adj_matrix(adj_matrix_in, node_names):
    """
    Creates a NetworkX graph from a Matrix
    :param adj_matrix_in: Adjacency Matrix
    :param node_names: List of node names corresponding to the matrix indices

    :return: NetworkX Directed Graph Object
    """
    graph = nx.DiGraph()

    adj_matrix_in = np.array(adj_matrix_in)
    num_nodes = adj_matrix_in.shape[0]

    if len(node_names) != num_nodes:
        raise ValueError("Number of node names must match the size of the adjacency matrix")

    # Add nodes with original names
    for node_name in node_names:
        graph.add_node(node_name)

    # Add edges from adjacency matrix
    for i in range(num_nodes):
        for j in range(num_nodes):
            if adj_matrix_in[i, j] != 0:  # Assuming non-zero values indicate edges
                graph.add_edge(node_names[i], node_names[j], weight=adj_matrix_in[i, j])

    return graph
