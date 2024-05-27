def export_to_mermaid(graph_in):
    """
    Converts Graph to mermaid code so It's easy to import in Draw.io or other software
    :param graph_in: NetworkX graph class
    :return: Mermaid code
    """
    file_name = "graph.mmd"  # Adjust as needed

    mmd_code = "graph TB;\n"

    for edge in graph_in.edges(data=True):
        source = edge[0]
        target = edge[1]
        weight = edge[2]['weight'] if 'weight' in edge[2] else ''
        mmd_code += f"{source} -->|{weight}| {target};\n"

    with open(file_name, "w") as file:
        file.write(mmd_code)

    return mmd_code
