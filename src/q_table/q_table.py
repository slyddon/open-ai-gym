import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as col
import pickle
import os

LEARN_RATE = 0.2
DISCOUNT_RATE = 0.7


class QTable:
    def __init__(
        self, env_action_space, learn_rate=LEARN_RATE, discount_rate=DISCOUNT_RATE
    ):

        self.x_positions = np.linspace(-0.75, 0.75, 30)
        self.y_positions = np.linspace(-0.1, 1.5, 35)
        self.ang_thresh = [-0.5, -0.3, 0, 0.3, 0.5]
        self.vel_thresh = 10 ** np.linspace(-2, 0, 10)
        self.leg_touching = [0, 1]

        self.learn_rate = learn_rate
        self.discount_rate = discount_rate

        # quantize state space
        self.q_table = np.zeros(
            [
                len(self.x_positions),
                len(self.y_positions),
                len(self.ang_thresh),
                len(self.vel_thresh),
                len(self.vel_thresh),
                len(self.leg_touching),
                env_action_space,
            ]
        )

    @staticmethod
    def find_nearest(array, value):
        # get idx of nearest val
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx

    def get_action(self, observation):
        idx_x, idx_y, idx_a, idx_vx, idx_vy, idx_l = self.get_obs_info(observation)
        return np.argmax(self.q_table[idx_x, idx_y, idx_a, idx_vx, idx_vy, idx_l])

    def get_obs_info(self, observation):
        pos_x, pos_y, vel_x, vel_y, angle, _, l_leg, r_leg = observation
        # cartesian position
        idx_x = self.find_nearest(self.x_positions, pos_x)
        idx_y = self.find_nearest(self.y_positions, pos_y)
        # angle
        idx_ang = self.find_nearest(self.ang_thresh, angle)
        # velocity
        idx_vx = self.find_nearest(self.vel_thresh, vel_x)
        idx_vy = self.find_nearest(self.vel_thresh, vel_y)
        # leg touching
        idx_l = 0 if l_leg + r_leg == 0 else 1

        return idx_x, idx_y, idx_ang, idx_vx, idx_vy, idx_l

    def _get_q_factor(self, old_q, reward, action_qspace):
        new_q = (1 - self.learn_rate) * old_q + self.learn_rate * (
            reward + self.discount_rate * np.max(action_qspace)
        )
        return new_q

    def update_table(self, obs, new_obs, action, reward):
        idx_x, idx_y, idx_a, idx_vx, idx_vy, idx_l = self.get_obs_info(obs)
        (
            new_idx_x,
            new_idx_y,
            new_idx_a,
            new_idx_vx,
            new_idx_vy,
            new_idx_l,
        ) = self.get_obs_info(new_obs)

        old_q = self.q_table[idx_x, idx_y, idx_a, idx_vx, idx_vy, idx_l, action]
        action_qspace = self.q_table[
            new_idx_x, new_idx_y, new_idx_a, new_idx_vx, new_idx_vy, new_idx_l
        ]

        new_q = self._get_q_factor(old_q, reward, action_qspace)
        self.q_table[idx_x, idx_y, idx_a, idx_vx, idx_vy, idx_l, action] = new_q

    def plot_q_table(self):

        axis = np.arange(0, len(self.q_table.shape) - 4)  # axes over which to average
        table = np.zeros([len(self.y_positions), len(self.x_positions), 5])

        for ang in range(5):
            for x, _ in enumerate(self.x_positions):
                for y, _ in enumerate(self.y_positions):
                    table[y, x, ang] = np.argmax(
                        np.mean(self.q_table[x, y, ang], axis=tuple(axis))
                    )

        cmap = col.ListedColormap(
            ["whitesmoke", "cornflowerblue", "navajowhite", "lightcoral"]
        )
        bounds = np.arange(0, 5)
        norm = col.BoundaryNorm(bounds, cmap.N)

        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(15, 10))
        # ang = -0.5
        img1 = ax1.imshow(table[:, :, 4], cmap=cmap, norm=norm)
        ax1.set_xticks(
            np.linspace(0, len(self.x_positions), len(self.x_positions))[::10]
        )
        ax1.set_xticklabels([round(x, 2) for x in self.x_positions[::10]])
        ax1.set_yticks(
            np.linspace(0, len(self.y_positions), len(self.y_positions))[::10]
        )
        ax1.set_yticklabels(round(y, 2) for y in self.y_positions[::-10])

        # ang = -0.2
        img2 = ax2.imshow(table[:, :, 3], cmap=cmap, norm=norm)
        ax2.set_xticks(
            np.linspace(0, len(self.x_positions), len(self.x_positions))[::10]
        )
        ax2.set_xticklabels([round(x, 2) for x in self.x_positions[::10]])
        ax2.set_yticks(
            np.linspace(0, len(self.y_positions), len(self.y_positions))[::10]
        )
        ax2.set_yticklabels(round(y, 2) for y in self.y_positions[::-10])

        # ang = 0
        img3 = ax3.imshow(table[:, :, 2], cmap=cmap, norm=norm)
        ax3.set_xticks(
            np.linspace(0, len(self.x_positions), len(self.x_positions))[::10]
        )
        ax3.set_xticklabels([round(x, 2) for x in self.x_positions[::10]])
        ax3.set_yticks(
            np.linspace(0, len(self.y_positions), len(self.y_positions))[::10]
        )
        ax3.set_yticklabels(round(y, 2) for y in self.y_positions[::-10])

        # ang = 0.2
        img4 = ax4.imshow(table[:, :, 1], cmap=cmap, norm=norm)
        ax4.set_xticks(
            np.linspace(0, len(self.x_positions), len(self.x_positions))[::10]
        )
        ax4.set_xticklabels([round(x, 2) for x in self.x_positions[::10]])
        ax4.set_yticks(
            np.linspace(0, len(self.y_positions), len(self.y_positions))[::10]
        )
        ax4.set_yticklabels(round(y, 2) for y in self.y_positions[::-10])

        # ang = 0.5
        img5 = ax5.imshow(table[:, :, 0], cmap=cmap, norm=norm)
        ax5.set_xticks(
            np.linspace(0, len(self.x_positions), len(self.x_positions))[::10]
        )
        ax5.set_xticklabels([round(x, 2) for x in self.x_positions[::10]])
        ax5.set_yticks(
            np.linspace(0, len(self.y_positions), len(self.y_positions))[::10]
        )
        ax5.set_yticklabels(round(y, 2) for y in self.y_positions[::-10])

        cbar = plt.colorbar(
            img5,
            cmap=cmap,
            norm=norm,
            boundaries=bounds,
            ticks=[0.5, 1.5, 2.5, 3.5],
            fraction=0.046,
            pad=0.04,
        )
        cbar.ax.set_yticklabels(["Nothing", "Right", "Main", "Left"])

        plt.tight_layout()
        plt.show()

    def load_table(self, file_path):
        if os.path.exists(file_path):
            self.q_table = pickle.load(open(file_path, "rb"))
            print("Loaded Q table from", file_path)
            print("Q table has shape:", self.q_table.shape)
        else:
            print("ERROR: Table not loaded as does not exist")

    def save_table(self, file_path):
        pickle.dump(self.q_table, open(file_path, "wb"))
        print("Saved Q table!")
