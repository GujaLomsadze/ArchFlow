CREATE TABLE node_hits
(
    id                       SERIAL PRIMARY KEY,
    node_name                VARCHAR(255),
    timestamp                TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    hits                     INTEGER,
    percentage_of_total_hits NUMERIC
);

CREATE TABLE archflow_stats
(
    id                    SERIAL PRIMARY KEY,
    elapsed_seconds       INTERVAL,
    total_hits            INTEGER,
    average_hits_per_node NUMERIC,
    max_hits_node         VARCHAR(255),
    max_hits              INTEGER,
    min_hits_node         VARCHAR(255),
    min_hits              INTEGER
);


CREATE TABLE node_capacity
(
    node_id                    SERIAL PRIMARY KEY,    -- A unique identifier for each node
    node_name                  VARCHAR(255) NOT NULL, -- The name or identifier of the node
    cpu_io_capacity            DECIMAL(10, 2),        -- Maximum CPU IO capacity
    disk_io_capacity           DECIMAL(10, 2),        -- Maximum Disk IO capacity
    ram_capacity               DECIMAL(10, 2),        -- Maximum RAM capacity in GB
    storage_capacity           DECIMAL(10, 2),        -- Maximum Storage capacity in GB
    network_bandwidth_capacity DECIMAL(10, 2),        -- Maximum Network bandwidth capacity in Gbps
    max_cpu_load               DECIMAL(5, 2),         -- Maximum CPU load expected
    -- Any other capacity metrics relevant to your nodes
    UNIQUE (node_name)                                -- Ensures node names are unique
);


CREATE TABLE IF NOT EXISTS node_remaining_capacity
(
    record_id                  SERIAL PRIMARY KEY,
    node_name                  VARCHAR(255) NOT NULL,
    timestamp                  TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    cpu_io_capacity            DECIMAL(10,2),
    disk_io_capacity           DECIMAL(10,2),
    ram_capacity               DECIMAL(10,2),
    storage_capacity           DECIMAL(10,2),
    network_bandwidth_capacity DECIMAL(10,2),
    max_cpu_load               DECIMAL(10,2),
    UNIQUE (node_name, timestamp)
);