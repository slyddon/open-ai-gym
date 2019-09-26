from .action_bot import ActionBot
from ..q_table import QTable
import numpy as np


class QActionBot(ActionBot):
    def __init__(self, env):
        ActionBot.__init__(self, env)
        self.q_table = QTable(env.action_space.n)

    def load_q_table(self, file_path="./fitted_models/q_table_vx_vy.p"):
        self.q_table.load_table(file_path)

    def save_q_table(self, file_path="./fitted_models/q_table_vx_vy.p"):
        self.q_table.save_table(file_path)

    def plot_q_table(self):
        self.q_table.plot_q_table()

    def get_q_action(self):
        if (
            np.random.uniform(0, 1) < max(1 / np.sqrt(self.episode_n + 1), 0.001)
        ) and not self.demo:
            return self.get_random_action()
        else:
            return self.q_table.get_action(self.obs)

    def update_q_table(self, next_obs, action, reward):
        self.q_table.update_table(self.obs, next_obs, action, reward)
