from collections import namedtuple, deque
import random
import torch
import numpy as np


class ReplayBuffer(object):
    """Replay buffer to store past experiences that the agent can then use for training data"""

    def __init__(self, buffer_size, batch_size, seed, device=None):

        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple(
            "Experience", field_names=["state", "action", "next_state", "reward"]
        )
        self.seed = random.seed(seed)
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def add_experience(self, states, actions, next_states, rewards):
        """Adds experience(s) into the replay buffer"""
        experience = self.experience(states, actions, next_states, rewards)
        self.memory.append(experience)

    def sample(self, num_experiences=None, separate_out_data_types=True):
        """Draws a random sample of experience from the replay buffer"""
        experiences = self.pick_experiences(num_experiences)
        if separate_out_data_types:
            states, actions, next_states, rewards = self.separate_out_data_types(
                experiences
            )
            return states, actions, next_states, rewards
        return experiences

    def separate_out_data_types(self, experiences):
        """Puts the sampled experience into the correct format for a PyTorch neural network"""
        states = (
            torch.from_numpy(np.stack([e.state for e in experiences if e is not None]))
            .float()
            .to(self.device)
        )
        actions = (
            torch.from_numpy(
                np.stack([e.action for e in experiences if e is not None])
            )
            .float()
            .to(self.device)
        )
        next_states = (
            torch.from_numpy(
                np.stack([e.next_state for e in experiences if e is not None])
            )
            .float()
            .to(self.device)
        )
        rewards = (
            torch.from_numpy(
                np.stack([e.reward for e in experiences if e is not None])
            )
            .float()
            .to(self.device)
        )
        return states, actions, next_states, rewards

    def pick_experiences(self, num_experiences=None):
        if num_experiences is not None:
            batch_size = num_experiences
        else:
            batch_size = self.batch_size
        return random.sample(self.memory, k=batch_size)

    def __len__(self):
        return len(self.memory)
