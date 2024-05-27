import redis
import json


def set_red_k_j(redis_conn, key, data):
    """
    Small wrapper to set the redis value to a specific key (Node in my case)
    :param redis_conn: Connection object for the Node
    :param key: Redis key
    :param data: value
    :return:
    """

    # TODO: Possible try:except block here
    # but I don't want to reduce speed of this function not a single bit

    # Serialize the Python dictionary to a JSON string
    json_data = json.dumps(data)

    # Set the key in Redis with the JSON string
    redis_conn.set(key, json_data)

    return True


"""
# Example usage

key = 'Node_1'
data = {'name': 'John', 'age': 30, 'city': 'New York'}
    
set_red_k_j(, key, data)

print(f"Data set for key '{key}'")

"""
