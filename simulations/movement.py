import numpy as np


def choose_next_node(G, cur_node):
    """
    Function to choose a next node based on a Probability from weights.
    Simulates Markov's Chain
    :param G: Graph
    :param cur_node: Current node name
    :return: Net node or None
    """
    neighbors = list(G.successors(cur_node))

    if not neighbors:
        return None  # Depth stopped here, no more continuation

    weights = [G[cur_node][neighbor]['weight'] for neighbor in neighbors]

    # Convert weights to probabilities
    total_weight = sum(weights)
    probabilities = [weight / total_weight for weight in weights]

    # Choose next node based on probabilities
    next_node = np.random.choice(neighbors, p=probabilities)
    return next_node
