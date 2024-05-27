import json

from functions.converters.json_to_matrix import graph_to_json


def json_to_redis(json_data, redis_conn):
    """

    :param json_data:
    :param redis_conn:
    :return:
    """
    r = redis_conn

    for node in json_data["nodes"]:
        node_key = f"node:{node['id']}"
        for field, value in node.items():
            if isinstance(value, dict):
                for sub_field, sub_value in value.items():
                    r.hset(node_key, f"{field}:{sub_field}", json.dumps(sub_value))
            else:
                r.hset(node_key, field, json.dumps(value))

    for link in json_data["links"]:
        link_key = f"link:{link['id']}"
        for field, value in link.items():
            if isinstance(value, dict):
                for sub_field, sub_value in value.items():
                    r.hset(link_key, f"{field}:{sub_field}", json.dumps(sub_value))
            else:
                r.hset(link_key, field, json.dumps(value))


def reconstruct_entity(entity_data):
    """
    Helper function to reconstruct a single entity (node or link) from Redis hash data.
    """
    entity = {}
    for field, value in entity_data.items():
        field_parts = field.split(":")
        if len(field_parts) == 1:
            # Simple field
            entity[field_parts[0]] = json.loads(value)
        elif len(field_parts) == 2:
            # Nested field
            if field_parts[0] not in entity:
                entity[field_parts[0]] = {}
            entity[field_parts[0]][field_parts[1]] = json.loads(value)
    return entity


def reconstruct_json_from_redis(redis_conn):
    """

    :param redis_conn:
    :return:
    """

    r = redis_conn

    reconstructed_data = {"nodes": [], "links": []}

    # Reconstruct nodes
    node_keys = r.keys(f"node:*")
    for node_key in node_keys:
        node_data = r.hgetall(node_key)
        node = reconstruct_entity(node_data)
        reconstructed_data["nodes"].append(node)

    # Reconstruct links
    link_keys = r.keys(f"link:*")
    for link_key in link_keys:
        link_data = r.hgetall(link_key)
        link = reconstruct_entity(link_data)
        reconstructed_data["links"].append(link)

    return reconstructed_data


def update_redis_with_graph(graph, redis_conn):
    # Generate the updated JSON data from the graph
    updated_json_data = graph_to_json(graph)

    # Use the provided function to update Redis
    # json_to_redis(json_data=updated_json_data, redis_conn=redis_conn)
