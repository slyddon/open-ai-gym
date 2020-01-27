import gym
import numpy as np
from collections import namedtuple, deque
import random
import torch

BATCH_SIZE = 64  # minibatch size - sample to train on
BUFFER_SIZE = 10000  # number of experiences to store
UPDATE_EVERY = 4  # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Experience replay
# Store experiences in a large table. Randomly sample from experiences and use them to train rather than
# using the latest experience on the fly. More efficient use of experience


class ReplayMemory:
    def __init__(
        self, buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE, update_every=UPDATE_EVERY
    ):
        self.params = {
            "buffer_size": buffer_size,
            "batch_size": batch_size,
            "update_every": update_every,
        }
        self.memory = deque(maxlen=self.params["buffer_size"])

    def update(self, state, action, reward, next_state, done, t_step):
        self.memory.append((state, action, reward, next_state, done))

        if t_step % self.params["update_every"] == 0:
            if len(self.memory) > self.params["batch_size"]:
                experiences = self._sample_experience()
                return experiences, True  # should the dqn learn?

        return None, False

    def _sample_experience(self):
        experiences = random.sample(self.memory, k=self.params["batch_size"])

        states = (
            torch.from_numpy(np.vstack([e[0] for e in experiences if e is not None]))
            .float()
            .to(device)
        )
        actions = (
            torch.from_numpy(np.vstack([e[1] for e in experiences if e is not None]))
            .long()
            .to(device)
        )
        rewards = (
            torch.from_numpy(np.vstack([e[2] for e in experiences if e is not None]))
            .float()
            .to(device)
        )
        next_states = (
            torch.from_numpy(np.vstack([e[3] for e in experiences if e is not None]))
            .float()
            .to(device)
        )
        dones = (
            torch.from_numpy(
                np.vstack([e[4] for e in experiences if e is not None]).astype(np.uint8)
            )
            .float()
            .to(device)
        )

        return states, actions, rewards, next_states, dones
