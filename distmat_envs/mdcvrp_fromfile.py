"""
Enviroment for Multi-depot CVRP loading data from a set of csv files
"""

from pathlib import Path
from typing import cast
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
        input_dir: Directory to load the training data. The directory must contain the following files:
            - nodes.csv: The file containing the node coordinates and the demand
            - distmat_{phase}/{idx}.csv: The files containing the distance matrix
    """

    name = "mdcvrpdist_fromfile"

    def __init__(
        self,
        input_dir: str | Path = "",
        phase: str = "train",
        device: str = "cpu",
        one_by_one: bool = False,
        no_restart: bool = True,
        imbalance_penalty: bool = True,
        intermediate_reward: bool = False,
        **kwargs,
    ):
        self.phase = phase
        self.load_data(input_dir)
        self.device = device
        self.batch_size: int | None = None
        self.one_by_one = one_by_one
        self.intermediate_reward = intermediate_reward
        self.imbalance_penalty = imbalance_penalty
        self.no_restart = no_restart

    def load_data(self, input_dir: str | Path):
        """Load the data from the input directory"""
        assert Path(input_dir).is_dir(), f"{input_dir} is not a directory"
        # nodes.csv
        nodes_path = Path(input_dir) / "nodes.csv"
        assert nodes_path.exists(), f"{nodes_path} does not exist"
        nodes = pd.read_csv(nodes_path)
        agents = nodes[nodes["type"] == "agent"]
        custs = nodes[nodes["type"] == "cust"]

        self.loc = nodes[["x", "y"]].values if "z" not in nodes.columns else nodes[["x", "y", "z"]].values
        self.demand = custs["value"].values

        self.n_agents = agents.shape[0]
        self.n_custs = custs.shape[0]
        self.dimension = 3 if "z" in nodes.columns else 2
        # assert that agent values are all the same
        assert (
            agents["value"].nunique() == 1
        ), "Capacity must be same for all agents"  # TODO: support different capacities
        self.vehicle_capacity = agents["value"].iloc[0].item()

        distmat_paths = list((Path(input_dir) / f"distmat_{self.phase}").glob("*.csv"))
        assert distmat_paths, f"No distmat_*.csv files found in {input_dir}/distmat_{self.phase}"
        self.distmats = [np.loadtxt(p, delimiter=",", dtype=np.float32) for p in distmat_paths]

        # Validation set is needed for training
        self.val_distmats = []
        if self.phase == "train":
            val_distmat_paths = list((Path(input_dir) / "distmat_val").glob("*.csv"))
            assert val_distmat_paths, f"No distmat_*.csv files found in {input_dir}/distmat_val"
            self.val_distmats = [np.loadtxt(p, delimiter=",", dtype=np.float32) for p in val_distmat_paths]

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

    def generate_data(self, batch_size: int, val: bool = False) -> TensorDict:
        # locations of the customers and the agents
        loc = torch.from_numpy(np.tile(self.loc, (batch_size, 1, 1))).float().to(self.device)

        # distance matrix
        # Sample `batch_size` matrices from self.distmats
        distmats = self.val_distmats if self.phase == "train" and val else self.distmats
        dist_mat = (
            torch.from_numpy(np.stack([distmats[i] for i in np.random.choice(len(distmats), batch_size)]))
            .float()
            .to(self.device)
        )

        # SVD decomposition to get a row/column vectors
        dist_u, _, dist_v = torch.svd(dist_mat)  # rank == (n_agents + n_custs)

        # demand for each customer, Note that the demand is normalized by the vehicle capacity
        demand = np.tile(self.demand, (batch_size, 1))  # (batch_size, n_custs)  # type: ignore
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

    env = MDCVRPEnv(
        input_dir="data/N20_M2_D3/",
        device="cpu",
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
