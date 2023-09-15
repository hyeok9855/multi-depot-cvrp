from typing import Any, cast

from tensordict import TensorDict
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base import Encoder
from envs import MDCVRPEnv


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Critic(nn.Module):
    """Estimates the problem complexity.

    This is a basic module that just looks at the log-probabilities predicted by
    the encoder + decoder, and returns an estimate of complexity
    """

    def __init__(self, env: MDCVRPEnv, loc_encoder_params: dict[str, Any], hidden_size: int):
        super().__init__()

        self.env = env
        self.loc_encoder = Encoder(**loc_encoder_params)

        self.conv1 = nn.Conv1d(hidden_size, 20, kernel_size=1)
        self.conv2 = nn.Conv1d(20, 20, kernel_size=1)
        self.conv3 = nn.Conv1d(20, 1, kernel_size=1)

        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, obs_td: TensorDict):
        # Use the same input as the actor's loc_encoder
        loc = cast(torch.FloatTensor, obs_td["loc"])  # (batch_size, n_agents + n_custs, 2)
        demand = cast(torch.FloatTensor, obs_td["demand"])  # (batch_size, n_custs)
        # Augment demand with dummy demand for agents
        aug_demand = torch.cat(
            [-torch.ones(demand.shape[0], self.env.n_agents, device=obs_td.device), demand], dim=1
        )  # (batch_size, n_custs + n_agents)
        loc_x = torch.cat([loc, aug_demand.unsqueeze(-1)], dim=-1)  # (batch_size, n_agents + n_custs, 3)
        loc_x = loc_x.transpose(1, 2)  # (batch_size, 3, n_agents + n_custs)

        output = F.relu(self.loc_encoder(loc_x))  # (batch_size, hidden_size, n_agents + n_custs)
        output = F.relu(self.conv1(output))
        output = F.relu(self.conv2(output))
        output = self.conv3(output).sum(dim=2)
        return output  # (batch_size, 1)
