import redis
import psycopg2


def get_redis_connection(redis_host='localhost', redis_port=6379, redis_password=None):
    """
    Establish and return a Redis connection.

    Parameters:
    - redis_host (str): Hostname of the Redis server.
    - redis_port (int): Port on which the Redis server is running.
    - redis_password (str): Password for the Redis server, if required.

    Returns:
    - Redis connection object that can be reused for executing commands.
    """
    connection = redis.Redis(host=redis_host, port=redis_port, password=redis_password, decode_responses=True)

    print("Connected to Redis!")

    return connection


def connect_to_postgres(host='localhost', port=5435, database='archflow', user='archflow',
                        password='archflow'):
    """
    Connects to a PostgreSQL database and returns the connection object.

    Parameters:
        host (str): The hostname or IP address of the PostgreSQL server.
        port (int): The port number of the PostgreSQL server.
        database (str): The name of the database to connect to.
        user (str): The username for authentication.
        password (str): The password for authentication.

    Returns:
        connection (psycopg2.extensions.connection): The connection object.
    """
    try:
        connection = psycopg2.connect(
            host=host,
            port=port,
            database=database,
            user=user,
            password=password
        )
        print("Connected to PostgreSQL!")
        return connection

    except Exception as e:
        print("Error connecting to PostgreSQL:", e)
        return None
