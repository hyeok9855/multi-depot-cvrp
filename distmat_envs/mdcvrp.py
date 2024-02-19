"""
Enviroment for Multi-depot CVRP with both coordinates and distance matrix as input
"""

from pathlib import Path
from typing import Any, cast
import warnings

from tensordict.tensordict import TensorDict
import numpy as np
import pandas as pd
import torch

from distmat_envs.utils import gather_by_index
from distmat_envs.render import MDCVRPVisualizer

warnings.filterwarnings("ignore", category=DeprecationWarning)


class MDCVRPEnv:
    """
    Enviroment for Multi-depot Capacitated Vehicle Routing Problem
    with both coordinates and distance matrix as input

    Args:
        n_custs: Number of customers
        n_agents: Number of agents = Number of depots.
        dimension: Dimension of the location of the nodes
        min_loc: Minimum value for the location of the nodes
        max_loc: Maximum value for the location of the nodes
        dist_perturb_min: Minimum value for the perturbation of the distance matrix
        dist_perturb_max: Maximum value for the perturbation of the distance matrix
        min_demand: Minimum value for the demand of the customers
        max_demand: Maximum value for the demand of the customers
        vehicle_capacity: Capacity of the vehicles
        device: Device to use for torch
        one_by_one: Whether to allow only one agent to move at a time
        no_restart: Whether to disable the restart of the agent
        imbalance_penalty: Whether to penalize the imbalance of the tour length
        intermediate_reward: Whether to give intermediate reward
    """

    name = "mdcvrpdist"
    # Capacities of the vehicles at each problem size
    CAPACITIES = {10: 20.0, 20: 30.0, 50: 40.0, 100: 50.0}

    def __init__(
        self,
        n_custs: int = 20,
        n_agents: int = 2,
        dimension: int = 2,
        min_loc: float = 0,
        max_loc: float = 1,
        dist_mat_min: np.ndarray | None = None,
        dist_mat_max: np.ndarray | None = None,
        dist_perturb_min: float | None = 0.5,
        dist_perturb_max: float | None = 2.0,
        min_demand: int = 1,
        max_demand: int = 9,
        vehicle_capacity: float | None = None,
        device: str = "cpu",
        one_by_one: bool = False,
        no_restart: bool = True,
        imbalance_penalty: bool = True,
        intermediate_reward: bool = False,
        **kwargs,
    ):
        self.n_custs = n_custs
        self.n_agents = n_agents
        self.dimension = dimension
        self.min_loc = min_loc
        self.max_loc = max_loc
        self.dist_mat_min = dist_mat_min
        self.dist_mat_max = dist_mat_max
        self.dist_perturb_min = dist_perturb_min
        self.dist_perturb_max = dist_perturb_max
        self.min_demand = min_demand
        self.max_demand = max_demand
        self.vehicle_capacity = self.CAPACITIES.get(n_custs, vehicle_capacity)
        assert self.vehicle_capacity is not None, f"Vehicle capacity must be specified for {n_custs} customers"
        assert self.vehicle_capacity >= max_demand, "Vehicle capacity must be larger than the maximum demand"
        self.device = device
        self.batch_size: int | None = None
        self.one_by_one = one_by_one
        self.intermediate_reward = intermediate_reward
        self.imbalance_penalty = imbalance_penalty
        self.no_restart = no_restart

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
        loc = cast(torch.FloatTensor, td["loc"])  # (batch_size, n_agents + n_custs, dimension)
        dist_mat = cast(torch.FloatTensor, td["dist_mat"])  # (batch_size, n_agents + n_custs, n_agents + n_custs)
        demand = cast(torch.FloatTensor, td["demand"])  # (batch_size, n_custs)
        agent_loc = cast(torch.FloatTensor, td["agent_loc"])  # (batch_size, n_agents, dimension)
        agent_loc_idx = cast(torch.LongTensor, td["agent_loc_idx"])  # (batch_size, n_agents)
        agent_dist_mat = cast(torch.FloatTensor, td["agent_dist_mat"])  # (batch_size, n_agents, n_agents + n_custs)
        remaining_capacity = cast(torch.FloatTensor, td["remaining_capacity"])  # (batch_size, n_agents)
        cum_length = cast(torch.FloatTensor, td["cum_length"])  # (batch_size, n_agents)
        return_count = cast(torch.LongTensor, td["return_count"])  # (batch_size, n_agents)
        action_mask = cast(torch.BoolTensor, td["action_mask"])  # (batch_size, n_agents, n_agents + n_custs)

        action = cast(torch.LongTensor, td["action"])  # (batch_size, dimension)
        selected_agent, selected_node = action.split(1, dim=-1)  # (batch_size, 1) each
        depot_bool = selected_node < self.n_agents  # (batch_size, 1)

        # get the batch index where the agent is at the depot
        at_depot = torch.where(depot_bool)[0]
        at_cust = torch.where(~depot_bool)[0]
        selected_cust = selected_node[at_cust] - self.n_agents  # (n_at_cust, 1)

        ### get the new location of the agent
        # get the new agent_loc
        loc_src = gather_by_index(loc, selected_node, dim=1, squeeze=False)  # (batch_size, 1, dimension)
        agent_loc = torch.scatter(  # scatter returns a new tensor
            input=agent_loc,
            index=selected_agent.unsqueeze(-1).expand((-1, 1, self.dimension)),
            dim=1,
            src=loc_src,
        )

        ### get the new distance matrix starting from the agent
        # cache the distance matrix before for the cum_length calculation
        agent_dist_mat_before = agent_dist_mat
        # get the new agent_dist_mat
        dist_mat_src = gather_by_index(dist_mat, selected_node, dim=1, squeeze=False)
        # (batch_size, 1, n_agents + n_custs)
        agent_dist_mat = torch.scatter(
            input=agent_dist_mat_before,
            index=selected_agent.unsqueeze(-1).expand((-1, 1, self.n_agents + self.n_custs)),
            dim=1,
            src=dist_mat_src,
        )  # (batch_size, n_agents, n_agents + n_custs)

        ### get the new node index of the agent
        # get the new agent_loc_idx
        agent_loc_idx = torch.scatter(input=agent_loc_idx, index=selected_agent, dim=1, src=selected_node)

        ### get the new remaining capacity of the agent
        # if the agent is at the depot, set the remaining capacity to 1.0
        remaining_capacity[at_depot] = torch.scatter(
            input=remaining_capacity[at_depot], index=selected_agent[at_depot], dim=1, value=1.0
        )
        # if the agent is at the cust, subtract the demand from the remaining capacity
        capacity_before = gather_by_index(remaining_capacity[at_cust], selected_agent[at_cust], dim=1, squeeze=False)
        used_capacity = gather_by_index(demand[at_cust], selected_cust, dim=1, squeeze=False)
        remaining_capacity[at_cust] = torch.scatter(
            input=remaining_capacity[at_cust],
            index=selected_agent[at_cust],
            dim=1,
            src=capacity_before - used_capacity,
        )

        ### get the new demand of the cust
        demand[at_cust] = torch.scatter(input=demand[at_cust], index=selected_cust, dim=1, value=0.0)

        ### update the cumulative length of the tour
        # get the distance between the agent location before and after, using the agent_dist_mat
        dist = gather_by_index(
            gather_by_index(agent_dist_mat_before, selected_agent, dim=1, squeeze=False).squeeze(1),
            selected_node,
            dim=1,
            squeeze=False,
        )  # (batch_size, 1)
        # add the distance to the cumulative length
        cum_length = torch.scatter(
            input=cum_length,
            index=selected_agent,
            dim=1,
            src=gather_by_index(cum_length, selected_agent, dim=1, squeeze=False) + dist,
        )

        ### update the return count of the agent
        # if the agent is at the depot, increment the return count
        return_count[at_depot] = torch.scatter(
            input=return_count[at_depot], index=selected_agent[at_depot], dim=1, src=return_count[at_depot] + 1
        )

        ### get the new action mask
        action_mask = self.get_action_mask(demand, agent_loc_idx, remaining_capacity, return_count)

        ### get done and reward
        # done if no action is available for all agents
        done = ~action_mask.any(dim=-1).any(dim=-1, keepdim=True)  # (batch_size, 1)

        # get reward
        reward = self.get_reward(cum_length, demand, done)  # (batch_size, 1)

        # if no action is available for any agent, unmask the first agent's depot to avoid distribution error
        done_idx = torch.where(done)[0]
        action_mask[done_idx, 0, 0] = True

        td_step = TensorDict(
            {
                "loc": loc,
                "dist_mat": dist_mat,
                "dist_u": td["dist_u"],
                "dist_v": td["dist_v"],
                "demand": demand,
                "agent_loc": agent_loc,
                "agent_loc_idx": agent_loc_idx,
                "agent_dist_mat": agent_dist_mat,
                "remaining_capacity": remaining_capacity,
                "cum_length": cum_length,
                "return_count": return_count,
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

    def get_action_mask(
        self,
        demand: torch.Tensor,
        agent_loc_idx: torch.Tensor,
        remaining_capacity: torch.Tensor,
        return_count: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = demand.shape[0]

        action_mask = torch.full((batch_size, self.n_agents, self.n_agents + self.n_custs), False, device=self.device)
        #####################################################################
        # QUESTION: When one agent is delivering, can the other agent move? #
        if self.one_by_one:
            active_batch_idx, active_agent_idx = torch.where(agent_loc_idx >= self.n_agents)
            # Only the active agent can move, but it cannot go to the other agent's depot
            mask_src = torch.full(
                (active_batch_idx.shape[0], 1, self.n_agents + self.n_custs), True, device=self.device
            )
            mask_src[:, :, : self.n_agents] = False
            mask_src = torch.scatter(
                input=mask_src,
                index=active_agent_idx.unsqueeze(-1).unsqueeze(-1).expand((-1, -1, self.n_agents)),
                dim=2,
                value=True,
            )
            action_mask[active_batch_idx] = torch.scatter(
                input=action_mask[active_batch_idx],
                index=active_agent_idx.unsqueeze(-1).unsqueeze(-1).expand((-1, -1, self.n_agents + self.n_custs)),
                dim=1,
                src=mask_src,
            )
            # if no agent is currently delivering, unmask all nodes except the depot
            inactive_batch_idx = torch.where((agent_loc_idx < self.n_agents).all(dim=1))[0]
            action_mask[inactive_batch_idx, :, self.n_agents :] = True
        else:
            # All agents can move, but they cannot go to the other agent's depot
            action_mask[:, :, : self.n_agents] = (
                torch.eye(self.n_agents, dtype=torch.bool, device=self.device)
                .unsqueeze(0)
                .expand((batch_size, -1, -1))
            )
            action_mask[:, :, self.n_agents :] = True
        #####################################################################

        # if no_restart is enabled, mask out the agent with return_count is 1
        if self.no_restart:
            action_mask = torch.where(return_count.unsqueeze(-1) >= 1, False, action_mask)

        # mask out the nodes that the agent is currently at to mask out the depot if the agent is at the depot
        action_mask = torch.scatter(input=action_mask, index=agent_loc_idx.unsqueeze(-1), dim=2, value=False)
        # mask out the cust with 0 demand; this masks out the nodes that the agent already visited before
        action_mask[:, :, self.n_agents :] = torch.where(
            demand.unsqueeze(1) == 0, False, action_mask[:, :, self.n_agents :]
        )
        # if the demand of the cust is larger than the remaining capacity, mask out the cust
        action_mask[:, :, self.n_agents :] = torch.where(
            demand.unsqueeze(1) > remaining_capacity.unsqueeze(2), False, action_mask[:, :, self.n_agents :]
        )

        return action_mask

    def get_reward(self, cum_length: torch.Tensor, demand: torch.Tensor, done: torch.Tensor) -> torch.Tensor:
        if self.intermediate_reward:
            raise NotImplementedError

        # If not done, reward is 0
        if not torch.all(done):
            return torch.zeros_like(done)

        # If done, reward is the total length of the tour
        reward = -torch.sum(cum_length, dim=1, keepdim=True)  # (batch_size, 1)

        # If some customer is not visited, the sum of demands of unvisited customers is subtracted from the reward
        # This situation can happen only when no_restart is enabled
        if self.no_restart:
            reward = reward - torch.sum(demand, dim=1, keepdim=True) * self.vehicle_capacity  # undo normalization

        # If imbalance penalty is enabled, penalize the imbalance of the tour length
        if self.imbalance_penalty:
            # penalty is the difference between the longest and the shortest tour length
            penalty = torch.max(cum_length, dim=1, keepdim=True)[0] - torch.min(cum_length, dim=1, keepdim=True)[0]
            reward = reward - penalty

        return reward

    def generate_data(self, batch_size: int, seed: int | None = None) -> TensorDict:
        # Use numpy to generate the data as it is easier to set the random seed
        np.random.seed(seed)

        # locations of the customers
        loc = np.random.uniform(self.min_loc, self.max_loc, (batch_size, self.n_agents + self.n_custs, self.dimension))
        loc = torch.from_numpy(loc).float().to(self.device)

        # distance matrix
        euc_dist = torch.norm(loc.unsqueeze(2) - loc.unsqueeze(1), dim=-1)
        # add noise to the distance matrix
        # each distance is multiplied by a random number between [dist_perturb_min, dist_perturb_max]
        if self.dist_mat_min is not None and self.dist_mat_max is not None:
            # `dist_mat_min` and `dist_mat_max` have higher priority
            dist_mat = torch.from_numpy(self.dist_mat_min + (self.dist_mat_max - self.dist_mat_min)).float().to(
                self.device
            ) * torch.rand_like(euc_dist)

        elif self.dist_perturb_min is not None and self.dist_perturb_max is not None:
            dist_mat = euc_dist * (
                self.dist_perturb_min + (self.dist_perturb_max - self.dist_perturb_min) * torch.rand_like(euc_dist)
            )

        else:
            raise ValueError(
                "dist_mat_min and dist_mat_max or dist_perturb_min and dist_perturb_max must be specified."
            )

        # SVD decomposition to get a row/column vectors
        dist_u, _, dist_v = torch.svd(dist_mat)  # rank == (n_agents + n_custs)

        # demand for each customer, Note that the demand is normalized by the vehicle capacity
        demand = np.random.randint(self.min_demand, self.max_demand + 1, (batch_size, self.n_custs))
        demand = torch.from_numpy(demand).float().to(self.device) / self.vehicle_capacity

        # current location of the agents
        agent_loc = loc[:, : self.n_agents, :].clone()

        # node index of the agents
        agent_loc_idx = torch.arange(self.n_agents, dtype=torch.int64).repeat(batch_size, 1)

        # distance matrix starting from the agents
        agent_dist_mat = dist_mat[:, : self.n_agents, :].clone()

        # capacity for each agent (1.0 because we normalized the demand by the vehicle capacity)
        remaining_capacity = torch.full((batch_size, self.n_agents), 1.0, dtype=torch.float32)

        # cumulative length of the tour for each agent
        cum_length = torch.full((batch_size, self.n_agents), 0.0, dtype=torch.float32)

        # count of returning to the depot for each agent
        return_count = torch.zeros((batch_size, self.n_agents), dtype=torch.int64)

        # action mask for each (agent, node) pair
        action_mask = torch.full((batch_size, self.n_agents, self.n_agents + self.n_custs), True, dtype=torch.bool)
        action_mask[:, :, : self.n_agents] = False  # Agents cannot visit the depot at the beginning

        return TensorDict(
            {
                "loc": loc,  # (batch_size, n_agents + n_custs, dimension)
                "dist_mat": dist_mat,  # (batch_size, n_agents + n_custs, n_agents + n_custs)
                "dist_u": dist_u,  # (batch_size, n_agents + n_custs, rank=(n_agents + n_custs))
                "dist_v": dist_v,  # (batch_size, n_agents + n_custs, rank=(n_agents + n_custs))
                "demand": demand,  # (batch_size, n_custs)
                "agent_loc": agent_loc,  # (batch_size, n_agents, dimension)
                "agent_loc_idx": agent_loc_idx,  # (batch_size, n_agents)
                "agent_dist_mat": agent_dist_mat,  # (batch_size, n_agents, n_agents + n_custs)
                "remaining_capacity": remaining_capacity,  # (batch_size, n_agents)
                "cum_length": cum_length,  # (batch_size, n_agents)
                "return_count": return_count,  # (batch_size, n_agents)
                "action_mask": action_mask,  # (batch_size, n_agents, n_agents + n_custs)
            },
            batch_size=torch.Size([batch_size]),
            device=self.device,
        )

    @classmethod
    def from_csv(
        cls, testset_path: Path | str, dist_mat_path: Path | str, **kwargs
    ) -> tuple["MDCVRPEnv", TensorDict, dict[str, Any]]:
        """Generate the environment from a csv file"""
        # Note that in this case, the batch_size must be 1 (TODO: support batch_size > 1)
        df = pd.read_csv(testset_path)
        custs = df[df["type"] == "cust"]
        agents = df[df["type"] == "agent"]
        dimension = 3 if "z" in df.columns else 2

        # Read distance matrix from csv into a numpy array
        dist_mat_np = np.loadtxt(dist_mat_path, delimiter=",", dtype=np.float32)
        # (n_agents + n_custs, n_agents + n_custs)

        n_custs = custs.shape[0]
        n_agents = agents.shape[0]
        min_loc, max_loc = 0, 1  # We assume that the locations are normalized to [0, 1]
        min_demand, max_demand = custs["value"].min().item(), custs["value"].max().item()

        # assert that agent values are all the same
        assert (
            agents["value"].nunique() == 1
        ), "Agent capacity must be all the same"  # TODO: support different capacities
        vehicle_capacity = agents["value"].iloc[0].item()

        ### Generate data from csv ###
        kwargs.update(
            {
                "n_custs": n_custs,
                "n_agents": n_agents,
                "dimension": dimension,
                "min_loc": min_loc,
                "max_loc": max_loc,
                "dist_delta": None,  # dist_delta is not used
                "min_demand": min_demand,
                "max_demand": max_demand,
                "vehicle_capacity": vehicle_capacity,
            }
        )
        env = cls(**kwargs)
        td = env.generate_td_from_data(df=df, dist_mat_np=dist_mat_np)

        return env, td, kwargs

    def generate_td_from_data(self, df: pd.DataFrame, dist_mat_np: np.ndarray) -> TensorDict:
        """Load the data from a user-defined data"""
        # Note that in this case, the batch_size must be 1 (TODO: support batch_size > 1)

        dimension = 3 if "z" in df.columns else 2

        if dimension == 2:
            loc = torch.FloatTensor(df[["x", "y"]].values).unsqueeze(0)
        else:
            loc = torch.FloatTensor(df[["x", "y", "z"]].values).unsqueeze(0)

        dist_mat = torch.from_numpy(dist_mat_np).unsqueeze(0)
        agent_dist_mat = dist_mat[:, : self.n_agents, :].clone()
        # SVD decomposition to get a row/column vectors
        dist_u, _, dist_v = torch.svd(dist_mat)  # rank == (n_agents + n_custs)

        demand = torch.FloatTensor(df.loc[df["type"] == "cust", "value"].values).unsqueeze(0) / self.vehicle_capacity
        agent_loc = loc[:, : self.n_agents, :].clone()
        agent_loc_idx = torch.arange(self.n_agents, dtype=torch.int64).unsqueeze(0)
        remaining_capacity = torch.full((1, self.n_agents), 1.0, dtype=torch.float32)
        cum_length = torch.full((1, self.n_agents), 0.0, dtype=torch.float32)
        return_count = torch.zeros((1, self.n_agents), dtype=torch.int64)
        action_mask = torch.full((1, self.n_agents, self.n_agents + self.n_custs), True, dtype=torch.bool)
        action_mask[:, :, : self.n_agents] = False

        return TensorDict(
            {
                "loc": loc,  # (1, n_agents + n_custs, dimension)
                "dist_mat": dist_mat,  # (1, n_agents + n_custs, n_agents + n_custs)
                "dist_u": dist_u,  # (1, n_agents + n_custs, rank=(n_agents + n_custs))
                "dist_v": dist_v,  # (1, n_agents + n_custs, rank=(n_agents + n_custs))
                "demand": demand,  # (1, n_custs)
                "agent_loc": agent_loc,  # (1, n_agents, dimension)
                "agent_loc_idx": agent_loc_idx,  # (1, n_agents)
                "agent_dist_mat": agent_dist_mat,  # (1, n_agents, n_agents + n_custs)
                "remaining_capacity": remaining_capacity,  # (1, n_agents)
                "cum_length": cum_length,  # (1, n_agents)
                "return_count": return_count,  # (1, n_agents)
                "action_mask": action_mask,  # (1, n_agents, n_agents + n_custs)
            },
            batch_size=torch.Size([1]),
            device=self.device,
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

    def render(self, td: TensorDict, actions: torch.Tensor, ax=None, save_path: Path | str | None = None):
        MDCVRPVisualizer(td, actions, save_path).render()

    def _set_seed(self, seed: int | None = None):
        """Set the seed for the environment"""
        rng = torch.manual_seed(seed)
        self.rng = rng


if __name__ == "__main__":
    batch_size = 50

    n_custs = 20
    n_agents = 2
    n_nodes = n_custs + n_agents

    rand_mat_min = np.random.randint(low=1, high=5, size=(n_nodes, n_nodes))
    rand_mat_min[np.arange(n_nodes), np.arange(n_nodes)] = 0
    print(rand_mat_min)

    rand_mat_max = rand_mat_min + np.random.randint(low=0, high=3, size=(n_nodes, n_nodes))
    rand_mat_max[np.arange(n_nodes), np.arange(n_nodes)] = 0
    print(rand_mat_max)

    env = MDCVRPEnv(
        n_custs=n_custs,
        n_agents=n_agents,
        dimension=2,
        device="cpu",
        vehicle_capacity=20.0,
        dist_mat_min=rand_mat_min,
        dist_mat_max=rand_mat_max,
    )
    td = env.reset(batch_size=batch_size)

    actions = torch.empty((batch_size, 0, 2), dtype=torch.int64)
    n_step = 0
    while "done" not in td.keys() or not torch.all(cast(torch.BoolTensor, td["done"])):
        action, td = env.rand_step(td)
        actions = torch.cat([actions, action.unsqueeze(1)], dim=1)
        n_step += 1
        print(f"step! {n_step}")

    env.render(td, actions=actions, save_path="test.png")
