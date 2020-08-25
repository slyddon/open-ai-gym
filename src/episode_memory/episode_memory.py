import gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.collections as mcoll


class EpisodeMemory:
    def __init__(self, env_name=None):
        self.env_name = env_name
        self.seed = np.random.randint(1, 1000000)

        self.action_log = []
        self.total_reward = 0

    def update(self, action, reward):
        self.action_log.append(action)
        self.total_reward += reward


class LanderMemory(EpisodeMemory):
    def __init__(self):
        EpisodeMemory.__init__(self, "LunarLander-v2")

    def replay(self, render=False, plot=False):
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
        env.close()

        if plot:
            self._plot(x, y, v)

    def _plot(self, x, y, v):
        fig, ax = plt.subplots(figsize=(10, 7))
        # position
        plt.plot(x, y, "--", lw=1.5)
        # velocity
        sc = plt.scatter(x, y, c=(v / np.max(v)), cmap=plt.cm.get_cmap("Reds"), s=50)
        cbar = plt.colorbar(sc, pad=0.01)
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


class PoleMemory(EpisodeMemory):
    def __init__(self):
        EpisodeMemory.__init__(self, "CartPole-v0")

    def replay(self, render=False, plot=False):
        env = gym.make(self.env_name)
        env.seed(self.seed)
        _ = env.reset()

        x = []
        v = []
        a = []
        av = []
        for action in self.action_log:
            if render:
                env.render()
            obs, _, _, _ = env.step(action)
            x.append(obs[0])
            v.append(obs[1])
            a.append(obs[2])
            av.append(obs[3])
        env.close()

        if plot:
            self._plot(x, v, a, av)

    def _plot(self, x, v, a, av):
        fig, ax = plt.subplots(figsize=(15, 7))

        # increase the angle when plotting to make it clearer
        angle_multiplier = 6
        # plot every Nth pole
        POLE_SAMPLE = 5
        for i in range(len(x)):
            if i % POLE_SAMPLE == 0:
                plt.plot(
                    [x[i], x[i] + np.sin(angle_multiplier * a[i])],
                    [0, np.cos(angle_multiplier * a[i])],
                    c="black",
                    lw=0.75,
                )
        # angular velocity of the end of the pole
        plt.scatter(
            x + np.sin(angle_multiplier * np.array(a)),
            np.cos(angle_multiplier * np.array(a)),
            c=(np.abs(av) / np.max(av)),
            cmap=plt.cm.get_cmap("Reds"),
            s=50,
        )
        # velocity of the pole base
        sc = plt.scatter(
            x,
            np.zeros(len(x)),
            c=(np.abs(v) / np.max(v)),
            cmap=plt.cm.get_cmap("Reds"),
            s=50,
        )
        cbar = plt.colorbar(sc, pad=0.01)
        cbar.set_label("Velocity", size=20, rotation=270, labelpad=20)
        # reward
        plt.text(
            0.05,
            0.95,
            f"Total reward: {round(self.total_reward, 1)}",
            horizontalalignment="left",
            verticalalignment="center",
            size=20,
            fontname="monospace",
            transform=ax.transAxes,
        )

        plt.xlim(-2.5, 2.5)
        plt.tight_layout()
        plt.show()


class CarMemory(EpisodeMemory):
    def __init__(self):
        EpisodeMemory.__init__(self, "MountainCar-v0")

    def replay(self, render=False, plot=False):
        env = gym.make(self.env_name)
        env.seed(self.seed)
        _ = env.reset()

        x = []
        v = []
        for action in self.action_log:
            if render:
                env.render()
            obs, _, _, _ = env.step(action)
            x.append(obs[0])
            v.append(obs[1])
        env.close()

        if plot:
            self._plot(np.array(x), np.array(v))

    def _colorline(self, x, y, z=None, cmap="copper", linewidth=3, alpha=1.0):
        """
        http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
        http://matplotlib.org/examples/pylab_examples/multicolored_line.html
        Plot a colored line with coordinates x and y
        Optionally specify colors in the array z
        Optionally specify a colormap, a norm function and a line width
        """
        segments = self._make_segments(x, y)
        lc = mcoll.LineCollection(
            segments, array=z, cmap=cmap, linewidth=linewidth, alpha=alpha
        )

        ax = plt.gca()
        ax.add_collection(lc)

        return lc

    @staticmethod
    def _make_segments(x, y):
        """
        Create list of line segments from x and y coordinates, in the correct format
        for LineCollection: an array of the form numlines x (points per line) x 2 (x
        and y) array
        """

        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        return segments

    def _plot(self, x, v):
        fig, ax = plt.subplots(figsize=(15, 7))
        lc = self._colorline(
            x=x, y=np.sin(3 * x) * 0.45 + 0.55, z=np.array(v), cmap="RdBu_r",
        )
        plt.colorbar(lc)
        # success limit
        plt.axhline(np.sin(3 * 0.5) * 0.45 + 0.55, c="green", ls="dashed")
        # reward
        plt.text(
            0.05,
            0.95,
            f"Total reward: {round(self.total_reward, 1)}",
            horizontalalignment="left",
            verticalalignment="center",
            size=20,
            fontname="monospace",
            transform=ax.transAxes,
        )
        plt.xlim(-1.2, 0.6)
        plt.xlabel("Time step", fontname="monospace", size=20)
        plt.ylabel("Height", fontname="monospace", size=20)
        plt.tight_layout()
        plt.show()
