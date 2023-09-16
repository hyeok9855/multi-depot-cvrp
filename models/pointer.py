import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base import Attention


class PtrNet(nn.Module):
    """Attention based pointer network, which decodes an action with given input embeddings"""

    def __init__(self, hidden_size: int, num_layers: int = 1, dropout: float = 0.1, glimpse: bool = True) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.glimpse = glimpse

        self.rnn = nn.GRU(
            hidden_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        self.context_attention = Attention(
            hidden_size=hidden_size, query_hidden_size=hidden_size, key_hidden_size=hidden_size
        )
        self.output_attention = Attention(
            hidden_size=hidden_size, query_hidden_size=hidden_size, key_hidden_size=hidden_size
        )

        self.drop_rnn = nn.Dropout(p=dropout)
        self.drop_hh = nn.Dropout(p=dropout)

    def forward(
        self,
        rnn_input_z: torch.Tensor,
        rnn_last_hidden: torch.Tensor | None,
        att_key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Args:
            rnn_input_z: (batch_size, n_agents, 1, hidden_size)
            rnn_last_hidden: (num_layers, batch_size, hidden_size)
            att_key: (batch_size, hidden_size, n_nodes) ... node info
        """

        n_agents = rnn_input_z.shape[1]
        n_nodes = att_key.shape[-1]

        # flatten rnn_input_z
        rnn_input_z = rnn_input_z.flatten(start_dim=0, end_dim=1)
        # expand rnn_last_hidden to match the batch size of rnn_input_z
        if rnn_last_hidden is not None:
            rnn_last_hidden = rnn_last_hidden.unsqueeze(2).expand(-1, -1, n_agents, -1)
            rnn_last_hidden = rnn_last_hidden.flatten(start_dim=1, end_dim=2)
            # (num_layers, batch_size * n_agents, hidden_size)

        rnn_out, rnn_hidden = self.rnn(rnn_input_z, rnn_last_hidden)
        # (batch_size * n_agents, 1, hidden_size), (num_layers, batch_size * n_agents, hidden_size)

        # apply dropout on the RNN output
        if self.dropout > 0:
            rnn_out = self.drop_rnn(rnn_out)
            if self.num_layers == 1:
                # If > 1 layer dropout is already applied
                rnn_hidden = self.drop_hh(rnn_hidden)

        # unflatten rnn_out
        rnn_out = rnn_out.squeeze(1)  # (batch_size * n_agents, hidden_size)
        rnn_out = rnn_out.view(-1, n_agents, self.hidden_size)  # (batch_size, n_agents, hidden_size)
        att_query = rnn_out.transpose(1, 2)  # (batch_size, hidden_size, n_agents) ... agent info

        # Given a summary of the current trajectory, obtain an input context
        if self.glimpse:
            context_w = self.context_attention(att_query, att_key, return_prob=True)
            # (batch_size, n_agents * n_nodes)
            context_w = context_w.view(-1, n_agents, n_nodes)  # (batch_size, n_agents, n_nodes)
            context = torch.bmm(context_w, att_key.transpose(1, 2))  # (batch_size, n_agents, hidden_size)
            context = context.transpose(1, 2)  # (batch_size, hidden_size, n_agents)
        else:
            context = att_query

        output_logit = self.output_attention(context, att_key, return_prob=False)  # (batch_size, n_agents * n_nodes)

        # unflatten and apply max pooling to agent axis
        rnn_hidden = rnn_hidden.view(self.num_layers, -1, n_agents, self.hidden_size)
        # (num_layers, batch_size, n_agents, hidden_size)
        rnn_hidden = rnn_hidden.mean(dim=2)  # (num_layers, batch_size, hidden_size)

        return output_logit, rnn_hidden
