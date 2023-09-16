from pathlib import Path
from typing import cast

from mpl_toolkits.mplot3d import Axes3D
from tensordict import TensorDict
from matplotlib import cm
from matplotlib.colors import Colormap
from matplotlib.markers import MarkerStyle
from matplotlib.figure import Figure
from matplotlib.axes import Axes

import matplotlib.pyplot as plt
import numpy as np
import torch


class MDCVRPVisualizer:
    def __init__(self, td: TensorDict, actions: torch.Tensor, save_path: Path | str | None = None):
        self.td = td
        self.actions = actions
        self.save_path = save_path
        self.n_agents = td["remaining_capacity"].shape[1]
        self.n_custs = td["demand"].shape[1]
        self.dimension = self.td["loc"].shape[-1]

        # if batch_size greater than 0 , we need to select the first batch element
        if self.td.batch_size != torch.Size([]):
            self.td = self.td[0]  # type: ignore
            self.actions = self.actions[0]

        # Unflatten the actions if necessary
        if len(self.actions.shape) == 1:
            self.actions = torch.stack(
                [self.actions // (self.n_agents + self.n_custs), self.actions % (self.n_agents + self.n_custs)], dim=-1
            )

        # detach and move to cpu
        self.td = self.td.detach().cpu()
        self.actions = self.actions.detach().cpu()

    def render(self):
        if self.dimension == 2:
            fig, ax = self.render2d()
        elif self.dimension == 3:
            fig, ax = self.render3d()

        if self.save_path is not None:
            plt.savefig(self.save_path, bbox_inches="tight")
        else:
            plt.show()

        plt.close()

    def render2d(self) -> tuple[Figure, Axes]:
        """Render the 2d environment"""
        colormap: Colormap = cm.get_cmap("nipy_spectral")
        color_list = colormap(np.linspace(0, 1, self.n_agents + 1))
        fig, ax = plt.subplots()

        loc = cast(torch.FloatTensor, self.td["loc"]).numpy()  # (n_agents + n_custs, 2)

        # QUESTION: wanna annotate demand?
        # demand = cast(torch.FloatTensor, td_cpu["demand"]) * vehicle_capacity

        # plot the depots and the customers
        for i in range(loc.shape[0]):
            if i < self.n_agents:
                ax.scatter(loc[i, 0], loc[i, 1], marker=MarkerStyle("*"), s=100, color="r")
                # annotate agent index
                ax.text(loc[i, 0] + 0.01, loc[i, 1] + 0.01, f"{i}", color="k", fontsize=10)
                continue
            ax.scatter(loc[i, 0], loc[i, 1], color="k", s=20)

        # plot the tour of the agent
        agent_loc_idx_before = torch.arange(self.n_agents, dtype=torch.int64)  # (n_agents,)

        for act in self.actions:
            agent_idx, loc_idx = act
            agent_idx, loc_idx = int(agent_idx), int(loc_idx)

            from_loc = loc[agent_loc_idx_before[agent_idx]]
            to_loc = loc[loc_idx]
            ax.plot([from_loc[0], to_loc[0]], [from_loc[1], to_loc[1]], color=color_list[agent_idx], lw=1, alpha=0.5)
            ax.quiver(
                from_loc[0],
                from_loc[1],
                to_loc[0] - from_loc[0],
                to_loc[1] - from_loc[1],
                color=color_list[agent_idx],
                scale_units="xy",
                angles="xy",
                scale=1,
                width=0.005,
                alpha=0.5,
            )

            agent_loc_idx_before[agent_idx] = loc_idx

        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)

        return fig, ax

    def render3d(self):
        """Render the 3d environment"""
        colormap: Colormap = cm.get_cmap("nipy_spectral")
        color_list = colormap(np.linspace(0, 1, self.n_agents + 1))

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        loc = cast(torch.FloatTensor, self.td["loc"]).numpy()  # (n_agents + n_custs, 3)

        # plot the depots and the customers
        for i in range(loc.shape[0]):
            if i < self.n_agents:
                ax.scatter(loc[i, 0], loc[i, 1], loc[i, 2], marker=MarkerStyle("*"), s=100, color="r")
                # annotate agent index
                ax.text(loc[i, 0] + 0.01, loc[i, 1] + 0.01, loc[i, 2] + 0.01, f"{i}", color="k", fontsize=10)
                continue
            ax.scatter(loc[i, 0], loc[i, 1], loc[i, 2], color="k", s=10)

        # plot the tour of the agent
        agent_loc_idx_before = torch.arange(self.n_agents, dtype=torch.int64)  # (n_agents,)

        for act in self.actions:
            agent_idx, loc_idx = act
            agent_idx, loc_idx = int(agent_idx), int(loc_idx)

            from_loc = loc[agent_loc_idx_before[agent_idx]]
            to_loc = loc[loc_idx]
            ax.plot(
                [from_loc[0], to_loc[0]],
                [from_loc[1], to_loc[1]],
                [from_loc[2], to_loc[2]],
                color=color_list[agent_idx],
                lw=1,
                alpha=0.5,
            )
            ax.quiver(
                from_loc[0],
                from_loc[1],
                from_loc[2],
                to_loc[0] - from_loc[0],
                to_loc[1] - from_loc[1],
                to_loc[2] - from_loc[2],
                color=color_list[agent_idx],
                arrow_length_ratio=0.1,
                alpha=0.5,
            )

            agent_loc_idx_before[agent_idx] = loc_idx

        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_zlim(-0.05, 1.05)

        return fig, ax
