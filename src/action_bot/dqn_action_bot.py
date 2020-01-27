from .action_bot import ActionBot
from ..dqn import DQN

import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

GAMMA = 0.98  # discount factor
TAU = 0.001  # for soft update of target parameters
LR = 0.0005  # learning rate
EPSILON_MIN = 0.025


class DQNActionBot(ActionBot):
    def __init__(
        self, env, learn_rate=LR, gamma=GAMMA, tau=TAU, epsilon_min=EPSILON_MIN
    ):
        ActionBot.__init__(self, env)

        self.params = {
            "learn_rate": learn_rate,
            "gamma": gamma,
            "tau": tau,
            "epsilon_min": epsilon_min,
        }

        self.dqn_local = DQN(self.state_space, self.action_space).to(device)
        self.dqn_target = DQN(self.state_space, self.action_space).to(device)

        self.optimizer = optim.Adam(
            self.dqn_local.parameters(), lr=self.params["learn_rate"]
        )

    def get_dq_action(self):

        if (
            np.random.rand()
            < max(1 / np.sqrt(self.episode_n + 1), self.params["epsilon_min"])
        ) and not self.demo:
            return self.get_random_action()

        state = torch.from_numpy(self.obs).float().unsqueeze(0).to(device)
        self.dqn_local.eval()
        with torch.no_grad():
            action_values = self.dqn_local(state)

        self.dqn_local.train()

        return np.argmax(action_values.cpu().data.numpy())

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences

        # get targets
        self.dqn_target.eval()
        with torch.no_grad():
            q_targets_next = torch.max(
                self.dqn_target.forward(next_states), dim=1, keepdim=True
            )[0]

        q_targets = rewards + (self.params["gamma"] * q_targets_next * (1 - dones))

        # get outputs
        self.dqn_local.train()
        q_expected = self.dqn_local.forward(states).gather(1, actions)

        # compute loss
        loss = F.mse_loss(q_expected, q_targets)

        # clear gradients
        self.optimizer.zero_grad()

        # update weights local network
        loss.backward()

        # take one SGD step
        self.optimizer.step()
        # ------------------- update target network ------------------- #
        self.soft_update(self.dqn_local, self.dqn_target, self.params["tau"])

    @staticmethod
    def soft_update(local_model, target_model, tau):
        for target_param, local_param in zip(
            target_model.parameters(), local_model.parameters()
        ):
            target_param.data.copy_(
                tau * local_param.data + (1.0 - tau) * target_param.data
            )

    def load_dqn_weights(self, unique_run):
        self.dqn_local.load_state_dict(
            torch.load(f"../fitted_models/dqn_weights_{unique_run}.pt")
        )
        print("Loaded DQN weights!")

    def save_dqn_weights(self, unique_run):
        torch.save(
            self.dqn_local.state_dict(), f"../fitted_models/dqn_weights_{unique_run}.pt"
        )
        print("Saved DQN weights!")
