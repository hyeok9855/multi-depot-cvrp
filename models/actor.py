from typing import Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from tensordict import TensorDict

from envs import MDCVRPEnv
from models.base import Encoder
from models.pointer import PtrNet


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
        agent_encoder_params: dict[str, Any],
        rnn_input_encoder_params: dict[str, Any],
        ptrnet_params: dict[str, Any],
    ):
        super().__init__()
        self.env = env
        self.loc_encoder = Encoder(**loc_encoder_params)
        self.agent_encoder = Encoder(**agent_encoder_params)
        self.rnn_input_encoder = Encoder(**rnn_input_encoder_params)
        self.ptrnet = PtrNet(**ptrnet_params)
        self.phase = "train"

        self.n_actions = self.env.n_agents + self.env.n_custs

    def forward(self, obs_td: TensorDict) -> tuple[TensorDict, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the actor
        """
        batch_size = obs_td["loc"].shape[0]

        n_step = 0
        rnn_input_x = torch.zeros((batch_size, self.rnn_input_encoder.input_size, 1), device=obs_td.device)
        rnn_last_hidden = None
        actions = log_probs = rewards = torch.empty((batch_size, 0), device=obs_td.device)
        while "done" not in obs_td.keys() or not torch.all(cast(torch.BoolTensor, obs_td["done"])):
            a_flat, log_p_flat, rnn_last_hidden = self.get_action(obs_td, rnn_input_x, rnn_last_hidden)
            actions = torch.cat([actions, a_flat.unsqueeze(1)], dim=1)
            log_probs = torch.cat([log_probs, log_p_flat.unsqueeze(1)], dim=1)

            # unflatten action and take a step in the environment
            action = torch.stack([a_flat // self.n_actions, a_flat % self.n_actions], dim=1)  # (batch_size, 2)
            obs_td.update({"action": action})
            obs_td = self.env.step(obs_td)

            rnn_input_loc = torch.gather(
                input=obs_td["agent_loc"], index=action[:, 0].unsqueeze(-1).unsqueeze(-1).expand((-1, -1, 2)), dim=1
            )  # (batch_size, 1, 2)
            rnn_input_depot = torch.gather(
                input=obs_td["loc"], index=action[:, 0].unsqueeze(-1).unsqueeze(-1).expand((-1, -1, 2)), dim=1
            )  # (batch_size, 1, 2)
            rnn_input_cap = torch.gather(
                input=obs_td["remaining_capacity"], index=action[:, 0].unsqueeze(-1), dim=1
            )  # (batch_size, 1)
            rnn_input_x = torch.cat(
                [rnn_input_loc, rnn_input_depot, rnn_input_cap.unsqueeze(-1)], dim=-1
            )  # (batch_size, 1, 3)
            # if action is to depot, then set the input as 0
            rnn_input_x = rnn_input_x.transpose(1, 2)  # (batch_size, 5, 1)

            assert obs_td.get("reward") is not None
            reward = cast(torch.FloatTensor, obs_td["reward"])  # (batch_size, 1)
            rewards = torch.cat([rewards, reward], dim=1)  # TODO: we may save intermediate rewards

            if n_step > self.env.n_custs * 2:
                raise RuntimeError("Too many steps, maybe stuck in an infinite loop? Check your Env.")

        return obs_td, actions, log_probs, rewards

    def get_action(
        self, obs_td: TensorDict, rnn_input_x: torch.Tensor, rnn_last_hidden: torch.FloatTensor | None
    ) -> tuple[torch.Tensor, torch.FloatTensor, torch.FloatTensor]:
        """
        Get action by decoding with the pointer network.
        Unlike the original paper, we assume that all the input changes each step.

        Args:
            obs_td: Observation TensorDict
            rnn_input_x: input to the pointer network
            rnn_last_hidden: hidden state of the pointer network

        Returns:
            Tuple of action, log probability of the action, and hidden state of the pointer network
        """

        ### Location Encoding
        loc = cast(torch.FloatTensor, obs_td["loc"])  # (batch_size, n_agents + n_custs, 2)
        demand = cast(torch.FloatTensor, obs_td["demand"])  # (batch_size, n_custs)
        # Augment demand with dummy demand for agents
        aug_demand = torch.cat(
            [-torch.ones((demand.shape[0], self.env.n_agents), device=obs_td.device), demand], dim=1
        )  # (batch_size, n_custs + n_agents)
        loc_x = torch.cat([loc, aug_demand.unsqueeze(-1)], dim=-1)  # (batch_size, n_agents + n_custs, 3)
        loc_x = loc_x.transpose(1, 2)  # (batch_size, 3, n_agents + n_custs)
        loc_z = self.loc_encoder(loc_x)  # (batch_size, hidden_size, n_agents + n_custs)

        ### Agent Encoding
        agent_loc = cast(torch.FloatTensor, obs_td["agent_loc"])  # (batch_size, n_agents, 2)
        remaining_capacity = cast(torch.FloatTensor, obs_td["remaining_capacity"])  # (batch_size, n_agents)
        # TODO: add depot location to agent_x
        agent_x = torch.cat([agent_loc, remaining_capacity.unsqueeze(-1)], dim=-1)  # (batch_size, n_agents, 3)
        agent_x = agent_x.transpose(1, 2)  # (batch_size, 3, n_agents)
        agent_z = self.agent_encoder(agent_x)  # (batch_size, hidden_size, n_agents)

        ### Pointer Network
        rnn_input_z = self.rnn_input_encoder(rnn_input_x).transpose(1, 2)  # (batch_size, 1, hidden_size)

        loc_z_expand = loc_z.unsqueeze(-2).expand(-1, -1, self.env.n_agents, -1)
        agent_z_expand = agent_z.unsqueeze(-1).expand(-1, -1, -1, self.env.n_custs + self.env.n_agents)
        att_key_z = torch.cat([loc_z_expand, agent_z_expand], dim=1).flatten(start_dim=2)
        # att_key_z: (batch_size, 2 * hidden_size, n_agents * (n_agents + n_custs))

        action_logit, rnn_hidden = self.ptrnet(rnn_input_z, rnn_last_hidden, att_key_z)
        # action_logit: (batch_size, n_agents * (n_agents + n_custs))
        # rnn_hidden: (batch_size, hidden_size)

        # Sample actions with action mask
        action_mask = cast(torch.BoolTensor, obs_td["action_mask"])  # (batch_size, n_agents, n_agents + n_custs)
        action_mask = action_mask.flatten(start_dim=1)  # (batch_size, n_agents * (n_agents + n_custs))
        action_logit_masked = action_logit + action_mask.log()  # (batch_size, n_agents * (n_agents + n_custs))
        action_probs = F.softmax(action_logit_masked, dim=-1)  # (batch_size, n_agents * (n_agents + n_custs))

        if self.phase == "train":
            action_dist = torch.distributions.Categorical(action_probs)
            a_flat = action_dist.sample()  # (batch_size,)
            log_p_flat = action_dist.log_prob(a_flat)  # (batch_size,)
        else:  # self.phase == "eval"
            p_flat, a_flat = torch.max(action_probs, dim=-1)  # (batch_size,)
            log_p_flat = p_flat.log()  # (batch_size,)

        # unflatten action
        return a_flat, log_p_flat, rnn_hidden
