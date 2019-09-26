import numpy as np


class ActionBot:
    def __init__(self, env=None, demo=False):
        self.env = env
        self.state_space = self.env.observation_space.shape[0]
        self.action_space = self.env.action_space.n

        self.obs = None
        self.episode_n = None
        self.demo = demo

    def get_action_by_angle(self):
        _, _, vel_x, vel_y, lander_ang, _, _, _ = self.obs

        dv = np.sqrt(vel_x ** 2 + vel_y ** 2)
        if dv < 0.1:
            return 0

        if lander_ang > 0.1:
            return 3
        if lander_ang < -0.1:
            return 1

        if np.random.uniform() > 0.35:
            return 2
        else:
            return 0

    def get_random_action(self):
        return self.env.action_space.sample()
