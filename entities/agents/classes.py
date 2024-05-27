from dataclasses import dataclass, field
from datetime import datetime
import uuid
import random
from typing import Optional, Dict
import yaml

# Load the configuration from the YAML file
with open("configs/user_action_config.yml", "r") as file:
    config = yaml.safe_load(file)


@dataclass
class UserAction:
    """
    Class which represents a UserAction Agent.

    It can be randomly pre-determined if <this> transactions, action or X is going to fail/succeed
    """

    action_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    action_type: str = field(default_factory=lambda: random.choice(config['defaults']['action_type']))
    timestamp: datetime = field(default_factory=datetime.now)
    user_id: str = field(
        default_factory=lambda: f"{config['user_id_prefix']}{random.randint(config['user_id_range']['min'], config['user_id_range']['max'])}")
    session_id: Optional[str] = None
    action_details: Dict[str, str] = field(default_factory=dict)
    outcome: Optional[str] = field(default_factory=lambda: random.choice(config['defaults']['outcome']))
    metadata: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if not self.session_id:  # Example of setting a random value conditionally
            self.session_id = f"{config['session_id_prefix']}{random.randint(config['session_id_range']['min'], config['session_id_range']['max'])}"


# Example usage without specifying any arguments
random_action = UserAction()
print(random_action)

"""

# Example usage without specifying any arguments
random_action = UserAction()
print(random_action)

"""
