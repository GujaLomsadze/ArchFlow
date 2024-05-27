def insert_node_hits(node_data, insert_timestamp, pg_connection):
    """
    Function to insert node hit data into the node_hits table in PostgreSQL.
    :param node_data: List of tuples containing node hit data (node_name, hits, percentage_of_total_hits)
    :param insert_timestamp:
    """

    cur = pg_connection.cursor()

    try:
        # Insert data into the node_hits table
        for node_name, node_info in node_data["nodes_info"].items():
            hits = node_info["hits"]
            percentage_of_total_hits = node_info["percentage_of_total_hits"]

            cur.execute("""
                           INSERT INTO node_hits (node_name, timestamp, hits, percentage_of_total_hits)
                           VALUES (%s, %s, %s, %s)
                       """, (node_name, insert_timestamp, hits, percentage_of_total_hits))

            # Commit the transaction
            # print(f"PG INSERT: {(node_name, insert_timestamp, hits, percentage_of_total_hits)}")

        pg_connection.commit()

    except Exception as e:
        # Rollback the transaction if an error occurs
        pg_connection.rollback()
        print("Error inserting node hit data into node_hits table:", e)
    finally:
        # Close the cursor
        cur.close()


def insert_remaining_capacity(remaining_capacities_in, insert_timestamp, pg_connection):
    """
    Function to insert node remaining capacity data into the node_remaining_capacity table in PostgreSQL.
    :param node_capacity_data: Dictionary containing node names as keys and another dictionary with remaining capacity metrics as values
    :param insert_timestamp: Timestamp at which the data is being inserted
    :param pg_connection: Active PostgreSQL connection object
    """

    cur = pg_connection.cursor()

    try:
        # Insert data into the node_remaining_capacity table
        for node_name, capacity_info in remaining_capacities_in.items():
            cpu_io_capacity = capacity_info.get("cpu_io_capacity")
            disk_io_capacity = capacity_info.get("disk_io_capacity")
            ram_capacity = capacity_info.get("ram_capacity")
            storage_capacity = capacity_info.get("storage_capacity")
            network_bandwidth_capacity = capacity_info.get("network_bandwidth_capacity")
            max_cpu_load = capacity_info.get("max_cpu_load")

            cur.execute("""
                           INSERT INTO node_remaining_capacity (
                               node_name, 
                               timestamp,
                               cpu_io_capacity, 
                               disk_io_capacity, 
                               ram_capacity, 
                               storage_capacity, 
                               network_bandwidth_capacity, 
                               max_cpu_load
                           ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                       """, (
                node_name,
                insert_timestamp,
                cpu_io_capacity,
                disk_io_capacity,
                ram_capacity,
                storage_capacity,
                network_bandwidth_capacity,
                max_cpu_load
            ))

        # Commit the transaction
        pg_connection.commit()

    except Exception as e:
        # Rollback the transaction if an error occurs
        pg_connection.rollback()
        print("Error inserting node remaining capacity data into node_remaining_capacity table:", e)
    finally:
        # Close the cursor
        cur.close()


def insert_node_capacity(node_capacity_data, pg_connection):
    """
    Function to insert node capacity data into the node_capacity table in PostgreSQL.
    :param node_capacity_data: Dictionary containing node names as keys and another
                               dictionary with capacity metrics as values
    :param pg_connection: Active PostgreSQL connection object
    """

    cur = pg_connection.cursor()

    try:
        # Insert data into the node_capacity table
        for node_name, capacity_info in node_capacity_data.items():
            cpu_io_capacity = capacity_info["cpu_io_capacity"]
            disk_io_capacity = capacity_info["disk_io_capacity"]
            ram_capacity = capacity_info["ram_capacity"]
            storage_capacity = capacity_info["storage_capacity"]
            network_bandwidth_capacity = capacity_info.get("network_bandwidth_capacity", None)  # Optional
            max_cpu_load = capacity_info.get("max_cpu_load", None)  # Optional

            cur.execute("""
                           INSERT INTO node_capacity (
                               node_name, 
                               cpu_io_capacity, 
                               disk_io_capacity, 
                               ram_capacity, 
                               storage_capacity, 
                               network_bandwidth_capacity, 
                               max_cpu_load
                           ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                       """, (
                node_name,
                cpu_io_capacity,
                disk_io_capacity,
                ram_capacity,
                storage_capacity,
                network_bandwidth_capacity,
                max_cpu_load
            ))

        # Commit the transaction
        pg_connection.commit()

    except Exception as e:
        # Rollback the transaction if an error occurs
        pg_connection.rollback()
        print("Error inserting node capacity data into node_capacity table:", e)
    finally:
        # Close the cursor
        cur.close()


def fetch_node_capacity(pg_connection):
    """
    Function to fetch node capacity data from the node_capacity table in PostgreSQL.
    :param pg_connection: Active PostgreSQL connection object
    :return: Dictionary with node names as keys and another dictionary with capacity metrics as values
    """
    node_capacity_data = {}

    cur = pg_connection.cursor()
    try:
        # Retrieve data from the node_capacity table
        cur.execute("""
                    SELECT 
                        node_name, 
                        cpu_io_capacity, 
                        disk_io_capacity, 
                        ram_capacity, 
                        storage_capacity, 
                        network_bandwidth_capacity, 
                        max_cpu_load
                    FROM node_capacity
                    """)

        rows = cur.fetchall()
        for row in rows:
            node_name, cpu_io_capacity, disk_io_capacity, ram_capacity, storage_capacity, network_bandwidth_capacity, max_cpu_load = row
            node_capacity_data[node_name] = {
                "cpu_io_capacity": cpu_io_capacity,
                "disk_io_capacity": disk_io_capacity,
                "ram_capacity": ram_capacity,
                "storage_capacity": storage_capacity,
                "network_bandwidth_capacity": network_bandwidth_capacity,
                "max_cpu_load": max_cpu_load
            }

    except Exception as e:
        print("Error fetching node capacity data from node_capacity table:", e)
    finally:
        # Close the cursor
        cur.close()

    return node_capacity_data


def truncate_node_capacity_data(pg_connection):
    """
    Function to insert node capacity data into the node_capacity table in PostgreSQL.
    :param node_capacity_data: Dictionary containing node names as keys and another
                               dictionary with capacity metrics as values
    :param pg_connection: Active PostgreSQL connection object
    """

    cur = pg_connection.cursor()

    try:
        cur.execute("""
                        TRUNCATE TABLE node_capacity
                       """)

        pg_connection.commit()

    except Exception as e:
        # Rollback the transaction if an error occurs
        pg_connection.rollback()
        print("Error inserting node capacity data into node_capacity table:", e)

    finally:
        # Close the cursor
        cur.close()
