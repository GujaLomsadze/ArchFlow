import json


def update_node_style_parameter_in_redis(redis_connection, node_id, parameter, new_value):
    """
    Updates a specific style parameter for a node with the given ID in Redis.

    :param redis_connection: The Redis connection object.
    :param node_id: The ID of the node to update.
    :param parameter: The style parameter to update.
    :param new_value: The new value to set for the parameter.
    """
    node_key = f"node:{node_id}"
    style_key = f"style:{parameter}"

    # Check if the node exists in Redis
    if not redis_connection.exists(node_key):
        print(f"Node {node_id} not found in Redis.")
        return

    # Update the specific style parameter in Redis
    redis_connection.hset(node_key, style_key, json.dumps(new_value))


def update_link_style_parameter_in_redis(redis_connection, link_id, parameter, new_value):
    """
    Updates a specific style parameter for a node with the given ID in Redis.

    :param redis_connection: The Redis connection object.
    :param link_id: The ID of the node to update.
    :param parameter: The style parameter to update.
    :param new_value: The new value to set for the parameter.
    """
    link_key = f"link:{link_id}"
    style_key = f"{parameter}"

    # Check if the node exists in Redis
    if not redis_connection.exists(link_key):
        print(f"update_link_style_parameter_in_redis. Link {link_id} not found in Redis.")
        return

    # Update the specific style parameter in Redis
    redis_connection.hset(link_key, style_key, json.dumps(new_value))


def increment_link_style_parameter_in_redis(redis_connection, link_id, parameter, increment_amount):
    """
    Increments a specific style parameter for a link with the given ID in Redis by a given amount.

    :param redis_connection: The Redis connection object.
    :param link_id: The ID of the link to update.
    :param parameter: The style parameter to increment.
    :param increment_amount: The amount to increment the parameter by.
    """
    link_key = f"link:{link_id}"
    style_key = f"{parameter}"

    # Check if the link exists in Redis
    if not redis_connection.exists(link_key):
        print(f"increment_link_style_parameter_in_redis. Link {link_id} not found in Redis.")
        return

    # Retrieve the current value of the style parameter
    current_value_json = redis_connection.hget(link_key, style_key)
    if current_value_json is not None:
        # Parse the JSON-encoded style value
        current_value = json.loads(current_value_json)
    else:
        # Initialize the value if it doesn't exist
        current_value = 0

    # Ensure the current value and increment amount are of compatible types
    if not isinstance(current_value, (int, float)) or not isinstance(increment_amount, (int, float)):
        print(f"Cannot increment parameter {parameter} as it or the increment amount is not a number.")
        return

    # Increment the value by the specified amount
    updated_value = current_value + increment_amount

    # Update the specific style parameter in Redis with the new incremented value
    redis_connection.hset(link_key, style_key, json.dumps(updated_value))


def decrement_link_style_parameter_in_redis(redis_connection, link_id, parameter, decrement_amount):
    """
    Increments a specific style parameter for a link with the given ID in Redis by a given amount.

    :param redis_connection: The Redis connection object.
    :param link_id: The ID of the link to update.
    :param parameter: The style parameter to increment.
    :param decrement_amount: The amount to increment the parameter by.
    """
    link_key = f"link:{link_id}"
    style_key = f"{parameter}"

    # Check if the link exists in Redis
    if not redis_connection.exists(link_key):
        print(f"increment_link_style_parameter_in_redis. Link {link_id} not found in Redis.")
        return

    # Retrieve the current value of the style parameter
    current_value_json = redis_connection.hget(link_key, style_key)
    if current_value_json is not None:
        # Parse the JSON-encoded style value
        current_value = json.loads(current_value_json)
    else:
        # Initialize the value if it doesn't exist
        current_value = 0

    # Ensure the current value and increment amount are of compatible types
    if not isinstance(current_value, (int, float)) or not isinstance(decrement_amount, (int, float)):
        print(f"Cannot increment parameter {parameter} as it or the increment amount is not a number.")
        return

    # Increment the value by the specified amount
    updated_value = current_value - decrement_amount

    # Update the specific style parameter in Redis with the new incremented value
    redis_connection.hset(link_key, style_key, json.dumps(updated_value))
