from typing import Any, cast

from tensordict import TensorDict
import torch
import torch.nn as nn
import torch.nn.functional as F

from distmat_models.base import Encoder
from distmat_envs import MDCVRPEnv, MDCVRPFromFileEnv, MDCVRPUserInputEnv


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Critic(nn.Module):
    """Estimates the problem complexity.

    This is a basic module that just looks at the log-probabilities predicted by
    the encoder + decoder, and returns an estimate of complexity
    """

    def __init__(
        self,
        env: MDCVRPEnv | MDCVRPFromFileEnv | MDCVRPUserInputEnv,
        dist_encoder_params: dict[str, Any],
    ):
        super().__init__()

        self.env = env

        self.svd_q = dist_encoder_params["input_size"]
        self.depart_encoder = Encoder(**dist_encoder_params)
        self.arrive_encoder = Encoder(**dist_encoder_params)
        self.dist_map = nn.Sequential(
            nn.Linear(dist_encoder_params["hidden_size"] * 2, dist_encoder_params["hidden_size"]),
            nn.ReLU(),
            nn.Linear(dist_encoder_params["hidden_size"], dist_encoder_params["hidden_size"]),
        )

        self.conv1 = nn.Conv1d(dist_encoder_params["hidden_size"], 20, kernel_size=1)
        self.conv2 = nn.Conv1d(20, 20, kernel_size=1)
        self.conv3 = nn.Conv1d(20, 1, kernel_size=1)

        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, obs_td: TensorDict):
        ### Distance encoding
        dist_u = cast(torch.FloatTensor, obs_td["dist_u"])  # (batch_size, n_agents + n_nodes, n_agents + n_nodes)
        dist_v = cast(torch.FloatTensor, obs_td["dist_v"])  # (batch_size, n_agents + n_nodes, n_agents + n_nodes)

        # Note that we use inverse distance, as we want to minimize the distance
        depart_x, arrive_x = dist_u[:, :, : self.svd_q], dist_v[:, :, : self.svd_q]
        # (batch_size, n_nodes, svd_q) each

        # (batch_size, n_agents + n_custs, svd_q), both
        depart_z = self.depart_encoder(depart_x.transpose(1, 2))  # (batch_size, hidden_size, n_agents + n_custs)
        arrive_z = self.arrive_encoder(arrive_x.transpose(1, 2))  # (batch_size, hidden_size, n_agents + n_custs)
        dist_z = self.dist_map(
            torch.cat([depart_z, arrive_z], dim=1).transpose(1, 2)
        )  # (batch_size, n_agents + n_custs, hidden_size)

        output = F.relu(dist_z.transpose(1, 2))
        output = F.relu(self.conv1(output))
        output = F.relu(self.conv2(output))
        output = self.conv3(output).sum(dim=2)
        return output  # (batch_size, 1)
