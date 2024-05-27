def generate_node_info(graph):
    """
    Gets Networkx Directed Graph and shows Dictionary info of it
    :param graph: networkX graph
    :return: Dictionary
    """
    node_info = {}
    for node in graph.nodes():
        successors = list(graph.successors(node))
        weights = {successor: graph[node][successor]['weight'] for successor in successors}
        node_info[node] = {'successors': successors, 'weights': weights}
    return node_info
