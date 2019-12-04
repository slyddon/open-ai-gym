import gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


class EpisodeMemory:
    def __init__(self, env_name=None):
        self.env_name = env_name
        self.seed = np.random.randint(1, 1000000)

        self.action_log = []
        self.total_reward = 0

    def update(self, action, reward):
        self.action_log.append(action)
        self.total_reward += reward

    def replay(self, render=False):
        env = gym.make(self.env_name)
        env.seed(self.seed)
        _ = env.reset()

        x = []
        y = []
        v = []
        for action in self.action_log:
            if render:
                env.render()
            obs, _, _, _ = env.step(action)
            x.append(obs[0])
            y.append(obs[1])
            v.append(np.sqrt(obs[2] ** 2 + obs[3] ** 2))

        fig, ax = plt.subplots(figsize=(10, 7))
        # position
        plt.plot(x, y, "--", lw=1.5)
        # velocity
        sc = plt.scatter(x, y, c=(v / np.max(v)), cmap=plt.cm.get_cmap("Reds"), s=50)
        cbar = plt.colorbar(sc)
        cbar.set_label("Velocity", size=20, rotation=270, labelpad=20)
        # target
        plt.scatter(0, 0, marker="x", c="black", s=100)
        plt.plot([-0.2, 0.2], [0, 0], c="black", ls="dashed")
        plt.plot([-0.2, -0.22, -0.2], [0.1, 0.08, 0.06], c="black")
        plt.plot([0.2, 0.2], [0, 0.1], c="black")
        plt.plot([0.2, 0.22, 0.2], [0.1, 0.08, 0.06], c="black")
        plt.plot([-0.2, -0.2], [0, 0.1], c="black")

        # reward
        plt.text(
            0.05,
            0.95,
            f"Total reward: {round(self.total_reward, 1)}",
            horizontalalignment="left",
            verticalalignment="center",
            size=15,
            fontname="monospace",
            transform=ax.transAxes,
        )

        plt.xlim(-0.5, 0.5)
        plt.ylim(-0.1, 1.5)
        plt.tight_layout()
        plt.show()

        env.close()
