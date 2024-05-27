from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class Node:
    name: str
    type: Optional[str]
    ip: Optional[str] = None
    port: Optional[int] = None

    cpu_usage: Optional[float] = 0.0  # Percentage
    ram_usage: Optional[float] = 0.0  # GB
    disk_usage: Optional[float] = 0.0  # GB

    disk_capacity: Optional[int] = 50  # GB

    rps: Optional[float] = 1_000  # Requests Per Second
    qps: Optional[float] = 1_000  # Queries Per Second
    tps: Optional[float] = 1_000  # Transactions Per Second

    retention_time_hours: Optional[int] = None  # Retention time in hours

    additional_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class SpringBootMicroservice(Node):
    type: str = "SpringBootMicroservice"
    actuator_health: Optional[str] = None  # Health endpoint status


@dataclass
class NodejsService(Node):
    type: str = "NodejsService"
    event_loop_latency: Optional[float] = None  # Milliseconds


@dataclass
class PostgreSQLInstance(Node):
    type: str = "PostgreSQLInstance"
    active_connections: Optional[int] = None


@dataclass
class RedisInstance(Node):
    type: str = "RedisInstance"
    connected_clients: Optional[int] = None


@dataclass
class MongoDBInstance(Node):
    type: str = "MongoDBInstance"
    collections_count: Optional[int] = None


@dataclass
class KafkaInstance(Node):
    type: str = "KafkaInstance"
    topics_count: Optional[int] = None
    topics_partitions: Optional[int] = None
