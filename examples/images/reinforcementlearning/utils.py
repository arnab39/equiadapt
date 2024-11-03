import random
from collections import namedtuple
import dotenv
from typing import Optional

# Define Transition as a namedtuple for better structure and readability
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory:
    def __init__(self, capacity):
        """Initialize the ReplayMemory with a fixed capacity.
        
        Args:
            capacity (int): The maximum size of the memory.
        """
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, state, action, next_state, reward):
        """Saves a transition into memory.
        
        Overwrites the oldest transition if memory is at capacity.
        Args:
            state: The state of the environment before taking the action.
            action: The action taken.
            next_state: The state of the environment after taking the action.
            reward: The reward received after taking the action.
        """
        # Create a Transition from the given arguments
        transition = Transition(state, action, next_state, reward)
        
        # Check if there is still room to append a new transition
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        # Overwrite the oldest data if the memory is full
        self.memory[self.position] = transition
        # Move the write position; wraps around to the beginning using modulo
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """Samples a batch of transitions from memory.
        
        Args:
            batch_size (int): Number of transitions to sample.
        
        Returns:
            list: A list of randomly sampled transitions.
        """
        return random.sample(self.memory, batch_size)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

def load_envs(env_file: Optional[str] = None) -> None:
    """
    Load all the environment variables defined in the `env_file`.
    This is equivalent to `. env_file` in bash.

    It is possible to define all the system specific variables in the `env_file`.

    :param env_file: the file that defines the environment variables to use. If None
                     it searches for a `.env` file in the project.
    """
    dotenv.load_dotenv(dotenv_path=env_file, override=True)