from typing import Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from tensordict import TensorDict

from distmat_envs import MDCVRPEnv
from distmat_models.base import Encoder
from distmat_models.pointer import PtrNet


class Actor(nn.Module):
    """
    Actor for MDCVRP

    Args:
        env: MDCVRP environment
        loc_encoder_params: Location encoder params
        agent_encoder_params: Agent encoder params
        rnn_input_encoder_params: RNN input encoder params
        ptrnet_params: Pointer network for decoding
    """

    def __init__(
        self,
        env: MDCVRPEnv,
        loc_encoder_params: dict[str, Any],
        dist_encoder_params: dict[str, Any],
        rnn_input_encoder_params: dict[str, Any],
        ptrnet_params: dict[str, Any],
    ):
        super().__init__()
        self.env = env
        self.node_encoder = Encoder(**loc_encoder_params)
        self.agent_encoder = Encoder(**rnn_input_encoder_params)

        # Distance encoding
        self.svd_q = dist_encoder_params["input_size"]
        self.arrive_encoder = Encoder(**dist_encoder_params)
        self.depart_encoder = Encoder(**dist_encoder_params)
        self.arrive_map = nn.Sequential(
            nn.Linear(
                loc_encoder_params["hidden_size"] + dist_encoder_params["hidden_size"], ptrnet_params["hidden_size"]
            ),
            nn.ReLU(),
            nn.Linear(ptrnet_params["hidden_size"], ptrnet_params["hidden_size"]),
        )
        self.depart_map = nn.Sequential(
            nn.Linear(
                rnn_input_encoder_params["hidden_size"] + dist_encoder_params["hidden_size"],
                ptrnet_params["hidden_size"],
            ),
            nn.ReLU(),
            nn.Linear(ptrnet_params["hidden_size"], ptrnet_params["hidden_size"]),
        )
        self.bias_map = nn.Sequential(nn.Linear(2, 20), nn.Sigmoid(), nn.Linear(20, 1))

        self.ptrnet = PtrNet(**ptrnet_params)
        self.phase = "train"

        self.n_nodes = self.env.n_agents + self.env.n_custs

    def forward(self, obs_td: TensorDict) -> tuple[TensorDict, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the actor
        """
        batch_size = obs_td["loc"].shape[0]

        n_step = 0
        rnn_last_hidden = None
        actions = log_probs = rewards = torch.empty((batch_size, 0), device=obs_td.device)
        while "done" not in obs_td.keys() or not torch.all(cast(torch.BoolTensor, obs_td["done"])):
            a_flat, log_p_flat, rnn_last_hidden = self.get_action(obs_td, rnn_last_hidden)
            actions = torch.cat([actions, a_flat.unsqueeze(1)], dim=1)
            log_probs = torch.cat([log_probs, log_p_flat.unsqueeze(1)], dim=1)

            # unflatten action and take a step in the environment
            action = torch.stack([a_flat // self.n_nodes, a_flat % self.n_nodes], dim=1)  # (batch_size, 2)
            obs_td.update({"action": action})
            obs_td = self.env.step(obs_td)

            assert obs_td.get("reward") is not None
            reward = cast(torch.FloatTensor, obs_td["reward"])  # (batch_size, 1)
            rewards = torch.cat([rewards, reward], dim=1)  # TODO: we may save intermediate rewards

            if n_step > self.env.n_custs * 2:
                raise RuntimeError("Too many steps, maybe stuck in an infinite loop? Check your Env.")

        return obs_td, actions, log_probs, rewards

    def get_node_x(self, obs_td: TensorDict) -> torch.Tensor:
        loc = cast(
            torch.FloatTensor, obs_td["loc"]
        )  # (batch_size, n_nodes, dimension) ... n_nodes = n_custs + n_agents
        demand = cast(torch.FloatTensor, obs_td["demand"])  # (batch_size, n_custs)
        # Augment demand with dummy demand for agents
        aug_demand = torch.cat(
            [-torch.ones((demand.shape[0], self.env.n_agents), device=obs_td.device), demand], dim=1
        )  # (batch_size, n_nodes)
        node_x = torch.cat([loc, aug_demand.unsqueeze(-1)], dim=-1)  # (batch_size, n_nodes, dimension + 1)
        node_x = node_x.transpose(1, 2)  # (batch_size, dimension + 1, n_nodes)

        assert node_x.shape[1] == self.node_encoder.input_size
        return node_x

    def get_agent_x(self, obs_td: TensorDict) -> torch.Tensor:
        agent_loc = cast(torch.FloatTensor, obs_td["agent_loc"])  # MATRIX: departure embedding
        # (batch_size, n_agents, dimension)
        depot_loc = cast(torch.FloatTensor, obs_td["loc"][:, : self.env.n_agents, :])  # MATRIX: arrive embedding
        # (batch_size, n_agents, dimension)
        capacity = cast(torch.FloatTensor, obs_td["remaining_capacity"].unsqueeze(-1))
        # (batch_size, n_agents, 1)
        agent_x = torch.cat([agent_loc, depot_loc, capacity], dim=-1)
        # (batch_size, n_agents, 2 * dimension + 1)
        agent_x = agent_x.transpose(1, 2)  # (batch_size, 2 * dimension + 1, n_agents)

        assert agent_x.shape[1] == self.agent_encoder.input_size
        return agent_x

    def get_distance_x(self, obs_td: TensorDict) -> tuple[torch.Tensor, torch.Tensor]:
        dist_u = cast(torch.FloatTensor, obs_td["dist_u"])  # (batch_size, n_agents + n_nodes, n_agents + n_nodes)
        dist_v = cast(torch.FloatTensor, obs_td["dist_v"])  # (batch_size, n_agents + n_nodes, n_agents + n_nodes)
        agent_loc_idx = cast(torch.LongTensor, obs_td["agent_loc_idx"])  # (batch_size, n_agents)

        # Note that we use inverse distance, as we want to minimize the distance
        depart_x = torch.gather(dist_u, dim=1, index=agent_loc_idx.unsqueeze(-1).expand(-1, -1, dist_u.shape[-1]))
        depart_x = depart_x[:, :, : self.svd_q]
        # (batch_size, n_agents, svd_q)
        arrive_x = dist_v[:, :, : self.svd_q]
        # (batch_size, n_nodes, svd_q)

        depart_x = depart_x.transpose(1, 2)  # (batch_size, svd_q, n_agents)
        arrive_x = arrive_x.transpose(1, 2)  # (batch_size, svd_q, n_nodes)

        return depart_x, arrive_x

    def get_action(
        self, obs_td: TensorDict, rnn_last_hidden: torch.FloatTensor | None
    ) -> tuple[torch.Tensor, torch.FloatTensor, torch.FloatTensor]:
        """
        Get action by decoding with the pointer network.
        Unlike the original paper, we assume that all the input changes each step.

        Args:
            obs_td: Observation TensorDict
            rnn_input_x: input to the pointer network rnn
            rnn_last_hidden: hidden state of the pointer network rnn

        Returns:
            Tuple of action, log probability of the action, and hidden state of the pointer network
        """

        # Node embeddings
        node_x = self.get_node_x(obs_td)  # (batch_size, dimension + 1, n_nodes)
        node_z = self.node_encoder(node_x)  # (batch_size, hidden_size, n_nodes) // embeddings for nodes to arrive at

        # Agent embeddings
        agent_x = self.get_agent_x(obs_td)  # (batch_size, 2 * dimension + 1, n_agents)
        agent_z = self.agent_encoder(agent_x)
        # (batch_size, hidden_size, n_agents) // embeddings for agents to depart from

        # MATRIX: distance embeddings
        depart_x, arrive_x = self.get_distance_x(obs_td)
        depart_z = self.depart_encoder(depart_x)  # (batch_size, hidden_size, n_agents)
        arrive_z = self.arrive_encoder(arrive_x)  # (batch_size, hidden_size, n_nodes)

        # Aggregate embeddings
        node_w_dist_z = self.arrive_map(torch.cat([node_z, arrive_z], dim=1).transpose(1, 2))
        # (batch_size, n_nodes, hidden_size)

        agent_w_dist_z = self.depart_map(torch.cat([agent_z, depart_z], dim=1).transpose(1, 2))
        # (batch_size, n_agents, hidden_size)

        ### Pointer Network
        agent_dist_mat = cast(torch.FloatTensor, obs_td["agent_dist_mat"])
        depot_dist_mat = cast(torch.FloatTensor, obs_td["dist_mat"][:, : self.env.n_agents, :])
        # (batch_size, n_agents, n_nodes) each

        # Use inverse of exponential distance as bias because we want to minimize the distance
        inv_agent_dist_mat = torch.exp(-agent_dist_mat)
        inv_depot_dist_mat = torch.exp(-depot_dist_mat)
        dist_bias = self.bias_map(torch.stack([inv_agent_dist_mat, inv_depot_dist_mat], dim=-1))

        action_logit, rnn_hidden = self.ptrnet(
            agent_w_dist_z.unsqueeze(-2),
            rnn_last_hidden,
            att_key=node_w_dist_z.transpose(1, 2),
            bias=dist_bias,
        )
        # (batch_size, n_agents * n_nodes), (batch_size, hidden_size)

        # Sample actions with action mask
        action_mask = cast(torch.BoolTensor, obs_td["action_mask"])  # (batch_size, n_agents, n_nodes)
        action_mask = action_mask.flatten(start_dim=1)  # (batch_size, n_agents * n_nodes)
        action_logit_masked = action_logit + action_mask.log()  # (batch_size, n_agents * n_nodes)
        action_probs = F.softmax(action_logit_masked, dim=-1)  # (batch_size, n_agents * n_nodes)

        if self.phase == "train":
            action_dist = torch.distributions.Categorical(action_probs)
            a_flat = action_dist.sample()  # (batch_size,)
            log_p_flat = action_dist.log_prob(a_flat)  # (batch_size,)
        else:  # self.phase == "eval"
            p_flat, a_flat = torch.max(action_probs, dim=-1)  # (batch_size,)
            log_p_flat = p_flat.log()  # (batch_size,)

        return a_flat, log_p_flat, rnn_hidden
