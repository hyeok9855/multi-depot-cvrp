"""
Multi-depot CVRP environment
"""
from pathlib import Path
from typing import cast
import warnings

from tensordict.tensordict import TensorDict
import torch
from envs.utils import gather_by_index

warnings.filterwarnings("ignore", category=DeprecationWarning)


class MDCVRPEnv:
    """
    Env for Multi-Depot Capacitated Vehicle Routing Problem

    Args:
        n_nodes: Number of nodes (customers)
        n_agents: Number of agents = Number of depots.
        min_loc: Minimum value for the location of the nodes
        max_loc: Maximum value for the location of the nodes
        min_demand: Minimum value for the demand of the (customer) nodes
        max_demand: Maximum value for the demand of the (customer) nodes
        vehicle_capacity: Capacity of the vehicles
        device: Device to use for torch
    """

    name = "mdcvrp"
    # Capacities of the vehicles at each problem size
    CAPACITIES = {10: 20.0, 20: 30.0, 50: 40.0, 100: 50.0}

    def __init__(
        self,
        n_nodes: int = 20,
        n_agents: int = 2,
        min_loc: float = 0,
        max_loc: float = 1,
        min_demand: int = 1,
        max_demand: int = 9,
        vehicle_capacity: float | None = None,
        device: str = "cpu",
    ):
        self.n_nodes = n_nodes
        self.n_agents = n_agents
        self.min_loc = min_loc
        self.max_loc = max_loc
        self.min_demand = min_demand
        self.max_demand = max_demand
        self.vehicle_capacity = self.CAPACITIES.get(n_nodes, vehicle_capacity)
        assert self.vehicle_capacity is not None, f"Vehicle capacity must be specified for {n_nodes} nodes"
        assert self.vehicle_capacity >= max_demand, "Vehicle capacity must be larger than the maximum demand"
        self.device = device
        self.batch_size: int | None = None

    def reset(self, td: TensorDict | None = None, batch_size: int | None = None) -> TensorDict:
        """Reset the Environment"""
        if td is not None:
            self.batch_size = td.shape[0] if len(td.shape) else None
            return td

        if batch_size is not None:
            self.batch_size = batch_size

        if self.batch_size is not None:
            td = self.generate_data(batch_size=batch_size)  # type: ignore
        else:
            td = self.generate_data(batch_size=1).squeeze(0).to_tensordict()

        return td

    def step(self, td: TensorDict) -> TensorDict:
        """Take a step in the environment"""
        td_shape = td.shape
        if td_shape == torch.Size([]):
            td = td.unsqueeze(0).to_tensordict()
        # Use type cast to avoid type error
        loc = cast(torch.FloatTensor, td["loc"])  # (batch_size, n_agents + n_nodes, 2)
        demand = cast(torch.FloatTensor, td["demand"])  # (batch_size, n_nodes)
        agent_loc = cast(torch.FloatTensor, td["agent_loc"])  # (batch_size, n_agents, 2)
        agent_loc_idx = cast(torch.LongTensor, td["agent_loc_idx"])  # (batch_size, n_agents)
        remaining_capacity = cast(torch.FloatTensor, td["remaining_capacity"])  # (batch_size, n_agents)
        cum_length = cast(torch.FloatTensor, td["cum_length"])  # (batch_size, n_agents)
        action_mask = cast(torch.BoolTensor, td["action_mask"])  # (batch_size, n_agents, n_agents + n_nodes)

        action = cast(torch.LongTensor, td["action"])  # (batch_size, 2)
        selected_agent, selected_action = action.split(1, dim=-1)  # (batch_size, 1) each
        depot_bool = selected_action < self.n_agents  # (batch_size, 1)

        # get the batch index where the agent is at the depot
        at_depot = torch.where(depot_bool)[0]
        at_node = torch.where(~depot_bool)[0]
        selected_node = selected_action[at_node] - self.n_agents  # (n_at_node, 1)

        #### get the new location of the agent
        # cache last location of agents for the cum_length calculation
        agent_loc_before = agent_loc  # (batch_size, n_agents, 2)
        # map the selected node to the node or agent index so that it corresponds to the td["loc"]

        loc_src = gather_by_index(loc, selected_action, dim=1, squeeze=False)  # (batch_size, 1, 2)
        agent_loc = torch.scatter(
            input=agent_loc_before, index=selected_agent.unsqueeze(-1).expand((-1, 1, 2)), dim=1, src=loc_src
        )

        ### get the new node index of the agent
        agent_loc_idx = torch.scatter(input=agent_loc_idx, index=selected_agent, dim=1, src=selected_action)

        ### get the new remaining capacity of the agent
        # if the agent is at the depot, set the remaining capacity to 1.0
        remaining_capacity[at_depot] = torch.scatter(
            input=remaining_capacity[at_depot], index=selected_agent[at_depot], dim=1, value=1.0
        )
        # if the agent is at the node, subtract the demand from the remaining capacity
        capacity_before = gather_by_index(remaining_capacity[at_node], selected_agent[at_node], dim=1, squeeze=False)
        used_capacity = gather_by_index(demand[at_node], selected_node, dim=1, squeeze=False)
        remaining_capacity[at_node] = torch.scatter(
            input=remaining_capacity[at_node],
            index=selected_agent[at_node],
            dim=1,
            src=capacity_before - used_capacity,
        )

        ### get the new demand of the node
        demand[at_node] = torch.scatter(input=demand[at_node], index=selected_node, dim=1, value=0.0)

        ### update the cumulative length of the tour
        # get the distance between the agent location before and after
        dist = torch.norm(agent_loc_before - agent_loc, dim=-1)  # (batch_size, n_agents)
        # add the distance to the cumulative length
        cum_length = cum_length + dist

        ### get the new action mask
        action_mask = torch.full_like(action_mask, False)  # (batch_size, n_agents, n_agents + n_nodes)
        #####################################################################
        # QUESTION: When one agent is delivering, can the other agent move? #
        active_batch_idx, active_agent_idx = torch.where(agent_loc_idx >= self.n_agents)
        # Only the active agent can move, but it cannot go to the other agent's depot
        mask_src = torch.full((active_batch_idx.shape[0], 1, self.n_agents + self.n_nodes), True, device=self.device)
        mask_src[:, :, : self.n_agents] = False
        mask_src = torch.scatter(
            input=mask_src,
            index=active_agent_idx.unsqueeze(-1).unsqueeze(-1).expand((-1, -1, self.n_agents)),
            dim=2,
            value=True,
        )
        action_mask[active_batch_idx] = torch.scatter(
            input=action_mask[active_batch_idx],
            index=active_agent_idx.unsqueeze(-1).unsqueeze(-1).expand((-1, -1, self.n_agents + self.n_nodes)),
            dim=1,
            src=mask_src,
        )
        # if no agent is currently delivering, unmask all nodes except the depot
        inactive_batch_idx = torch.where((agent_loc_idx < self.n_agents).all(dim=1))[0]
        action_mask[inactive_batch_idx, :, self.n_agents :] = True
        #####################################################################

        # mask out the node (or depot) that the agent is currently at
        action_mask = torch.scatter(input=action_mask, index=agent_loc_idx.unsqueeze(-1), dim=2, value=False)
        # if demand is 0, mask out the node
        action_mask[:, :, self.n_agents :] = torch.where(
            demand.unsqueeze(1) == 0, False, action_mask[:, :, self.n_agents :]
        )
        # if the demand of the node is larger than the remaining capacity, mask out the node
        action_mask[:, :, self.n_agents :] = torch.where(
            demand.unsqueeze(1) > remaining_capacity.unsqueeze(2), False, action_mask[:, :, self.n_agents :]
        )

        ### get done and reward
        # done if no action is available for all agents
        done = ~action_mask.any(dim=-1).any(dim=-1, keepdim=True)  # (batch_size, 1)

        # if no action is available for any agent, unmask the first agent's depot
        done_idx = torch.where(done)[0]
        action_mask[done_idx, 0, 0] = True

        # if done, reward is the total length of the tour
        reward = torch.where(done, -torch.sum(cum_length, dim=1, keepdim=True), torch.zeros_like(done))
        # # Intermediate reward: changed length of the tour
        # reward = -torch.sum(dist, dim=1, keepdim=True)  # (batch_size, 1)

        td_step = TensorDict(
            {
                "loc": loc,
                "demand": demand,
                "agent_loc": agent_loc,
                "agent_loc_idx": agent_loc_idx,
                "remaining_capacity": remaining_capacity,
                "cum_length": cum_length,
                "action_mask": action_mask,
                "reward": reward,
                "done": done,
            },
            td.shape,
            device=self.device,
        )
        if td_shape == torch.Size([]):
            td_step = td_step.squeeze(0).to_tensordict()
        return td_step

    def generate_data(self, batch_size: int) -> TensorDict:
        # locations of the nodes (customers)
        loc = torch.rand((batch_size, self.n_agents + self.n_nodes, 2), dtype=torch.float32, device=self.device)

        # demand for each node
        demand = (
            torch.randint(1, int(self.max_demand), (batch_size, self.n_nodes), dtype=torch.float32, device=self.device)
            / self.vehicle_capacity
        )  # Normalize the demand by the vehicle capacity

        # locations of the depots
        agent_loc = loc[:, : self.n_agents, :].clone()

        # index of the agents
        agent_loc_idx = torch.arange(self.n_agents, dtype=torch.int64, device=self.device).repeat(batch_size, 1)

        # capacity for each agent (1.0 because we normalized the demand by the vehicle capacity)
        remaining_capacity = torch.full((batch_size, self.n_agents), 1.0, dtype=torch.float32, device=self.device)

        # cumulative length of the tour for each agent
        cum_length = torch.full((batch_size, self.n_agents), 0.0, dtype=torch.float32, device=self.device)

        # action mask for each (agent, node + 1) pair
        action_mask = torch.full(
            (batch_size, self.n_agents, self.n_agents + self.n_nodes), True, dtype=torch.bool, device=self.device
        )
        action_mask[:, :, : self.n_agents] = False  # Agents cannot visit the depot at the beginning

        return TensorDict(
            {
                "loc": loc,  # (batch_size, n_agents + n_nodes, 2)
                "demand": demand,  # (batch_size, n_nodes)
                "agent_loc": agent_loc,  # (batch_size, n_agents, 2)
                "agent_loc_idx": agent_loc_idx,  # (batch_size, n_agents)
                "remaining_capacity": remaining_capacity,  # (batch_size, n_agents)
                "cum_length": cum_length,  # (batch_size, n_agents)
                "action_mask": action_mask,  # (batch_size, n_agents, n_agents + n_nodes)
            },
            batch_size=torch.Size([batch_size]),
            device=self.device,
        )

    def load_data_from_csv(self, path: dict[str, Path | str], n_agents: int, vehicle_capacity: int) -> TensorDict:
        """Load the data from a csv file"""
        raise NotImplementedError
        import pandas as pd

        # Note that in this case, the batch_size must be 1 (TODO: support batch_size > 1)

        # csv files for `loc` and `demand` are needed
        # Also, n_agents and vehicle_capacity must be specified

        loc = torch.FloatTensor(pd.read_csv(path["loc"]).values, device=self.device).unsqueeze(0)  # batch_size=1
        self.n_agents, self.n_nodes = n_agents, loc.shape[0] - n_agents

        demand = torch.FloatTensor(pd.read_csv(path["demand"]).values, device=self.device) / vehicle_capacity
        assert demand.shape[0] == self.n_nodes, "The number of nodes must be the same as the demand"

        agent_loc = loc[:, : self.n_agents, :].clone()
        agent_loc_idx = torch.full((1, self.n_agents), -1, dtype=torch.int64, device=self.device)
        remaining_capacity = torch.full((1, self.n_agents), 1.0, dtype=torch.float32, device=self.device)
        cum_length = torch.full((1, self.n_agents), 0.0, dtype=torch.float32, device=self.device)
        action_mask = torch.full(
            (1, self.n_agents, self.n_agents + self.n_nodes), True, dtype=torch.bool, device=self.device
        )
        action_mask[:, :, 0] = False  # Agents cannot visit the depot at the beginning

        return TensorDict(
            {
                "loc": loc,  # (1, n_agents + n_nodes, 2)
                "demand": demand,  # (1, n_nodes)
                "agent_loc": agent_loc,  # (1, n_agents, 2)
                "agent_loc_idx": agent_loc_idx,  # (1, n_agents)
                "remaining_capacity": remaining_capacity,  # (1, n_agents)
                "cum_length": cum_length,  # (1, n_agents)
                "action_mask": action_mask,  # (1, n_agents, n_agents + n_nodes)
            },
            batch_size=1,
        )

    def rand_action(self, td: TensorDict) -> TensorDict:
        """Randomly sample an action from the action mask"""
        action_mask = cast(torch.BoolTensor, td["action_mask"])
        n_actions = action_mask.shape[-1]

        action_flat = torch.multinomial(action_mask.flatten(start_dim=1).float(), num_samples=1).squeeze(-1)
        action = torch.stack([action_flat // n_actions, action_flat % n_actions], dim=-1)
        td.update({"action": action})
        return td

    def rand_step(self, td: TensorDict) -> tuple[torch.Tensor, TensorDict]:
        """Randomly sample an action from the action mask and take a step"""
        td = self.rand_action(td)
        action = cast(torch.LongTensor, td["action"])
        next_td = self.step(td)
        return action, next_td

    def render(self, td: TensorDict, actions=None, ax=None, save_path: Path | str | None = None):
        """Render the environment"""
        import matplotlib.pyplot as plt
        import numpy as np

        from matplotlib import cm
        from matplotlib.colors import Colormap
        from matplotlib.markers import MarkerStyle

        colormap: Colormap = cm.get_cmap("nipy_spectral")
        color_list = colormap(np.linspace(0, 1, self.n_agents + 1))
        fit, ax = plt.subplots()

        td_cpu = td.detach().cpu()
        if actions is None:
            actions = td_cpu.get("action", None)
            if actions is None:
                raise ValueError("actions must be specified")

        if actions.shape[-1] != 2:
            # we need to unflatten the actions
            actions = torch.stack(
                [actions // (self.n_agents + self.n_nodes), actions % (self.n_agents + self.n_nodes)], dim=-1
            )

        # if batch_size greater than 0 , we need to select the first batch element
        if td.batch_size != torch.Size([]):
            td_cpu = td_cpu[0]
            actions = actions[0]  # (seq_len, 2)

        loc = cast(torch.FloatTensor, td_cpu["loc"])  # (n_agents + n_nodes, 2)

        # QUESTION: wanna annotate demand?
        # demand = cast(torch.FloatTensor, td_cpu["demand"]) * self.vehicle_capacity

        # # add the depot at the first and last of the action
        # depot_action = torch.cat([torch.arange(self.n_agents).unsqueeze(-1), -torch.ones((self.n_agents, 1))], dim=1)
        # actions = torch.cat([depot_action, actions, depot_action])

        # plot the depot and the nodes
        for i in range(loc.shape[0]):
            if i < self.n_agents:
                ax.scatter(
                    loc[i, 0], loc[i, 1], marker=MarkerStyle("*"), s=50, color=color_list[i], label=f"Depot {i}"
                )
                continue
            ax.scatter(loc[i, 0], loc[i, 1], color="k", s=5)

        # plot the tour of the agent
        agent_loc_idx_before = torch.arange(self.n_agents, dtype=torch.int64)  # (n_agents,)

        for act in actions:
            agent_idx, loc_idx = act
            agent_idx, loc_idx = int(agent_idx), int(loc_idx)

            from_loc = loc[agent_loc_idx_before[agent_idx]]
            to_loc = loc[loc_idx]
            ax.plot([from_loc[0], to_loc[0]], [from_loc[1], to_loc[1]], color=color_list[agent_idx], lw=1)
            ax.annotate(
                "",
                xy=(to_loc[0].item(), to_loc[1].item()),
                xytext=(from_loc[0].item(), from_loc[1].item()),
                arrowprops=dict(arrowstyle="->", color=color_list[agent_idx], lw=1),
                annotation_clip=False,
            )

            agent_loc_idx_before[agent_idx] = loc_idx

        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)

        if save_path is not None:
            plt.savefig(save_path, bbox_inches="tight")
        else:
            plt.show()

    def _set_seed(self, seed: int | None = None):
        """Set the seed for the environment"""
        rng = torch.manual_seed(seed)
        self.rng = rng


if __name__ == "__main__":
    batch_size = 50
    env = MDCVRPEnv(n_nodes=1000, n_agents=20, device="cpu", vehicle_capacity=100.0)
    td = env.reset(batch_size=batch_size)

    actions = torch.empty((batch_size, 0, 2), dtype=torch.int64)
    n_step = 0
    while "done" not in td.keys() or not torch.all(cast(torch.BoolTensor, td["done"])):
        action, td = env.rand_step(td)
        actions = torch.cat([actions, action.unsqueeze(1)], dim=1)
        n_step += 1
        print(f"step! {n_step}")

    env.render(td, actions=actions, save_path="test.png")
