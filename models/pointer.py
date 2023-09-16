import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base import Encoder, Attention


class PtrNet(nn.Module):
    """Attention based pointer network, which decodes an action with given input embeddings"""

    def __init__(self, hidden_size, num_layers=1, dropout=0.1) -> None:
        super().__init__()

        self.rnn = nn.GRU(
            hidden_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        self.context_attention = Attention(
            hidden_size=hidden_size, query_hidden_size=hidden_size, key_hidden_size=2 * hidden_size
        )
        self.output_attention = Attention(
            hidden_size=hidden_size, query_hidden_size=2 * hidden_size, key_hidden_size=2 * hidden_size
        )

        self.num_layers = num_layers
        self.drop_rnn = nn.Dropout(p=dropout)
        self.drop_hh = nn.Dropout(p=dropout)

    def forward(
        self,
        rnn_input_z: torch.Tensor,
        rnn_last_hidden: torch.Tensor | None,
        att_key_z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Args:
            rnn_input_z: (batch_size, 1, hidden_size)
            rnn_last_hidden: (batch_size, num_layers, hidden_size)
            att_key_z: (batch_size, 2 * hidden_size, num_keys)
        """
        rnn_out, rnn_hidden = self.rnn(rnn_input_z, rnn_last_hidden)
        # rnn_out: (batch_size, 1, hidden_size)
        # rnn_hidden: (batch_size, num_layers, hidden_size)

        # Always apply dropout on the RNN output
        rnn_out = self.drop_rnn(rnn_out)
        if self.num_layers == 1:
            # If > 1 layer dropout is already applied
            rnn_hidden = self.drop_hh(rnn_hidden)

        # Given a summary of the current trajectory, obtain an input context
        context_w = self.context_attention(rnn_out, att_key_z)  # (batch_size, 1, num_keys)
        context = context_w.bmm(att_key_z.permute(0, 2, 1))  # (batch_size, 1, 2 * hidden_size)

        output_prob = self.output_attention(context, att_key_z)  # (batch_size, 1, num_keys)
        output_prob = output_prob.squeeze(1)  # (batch_size, num_keys)

        return output_prob, rnn_hidden
