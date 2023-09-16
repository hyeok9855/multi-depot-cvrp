import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """Simple 1d Convolution Encoder"""

    def __init__(self, input_size: int, hidden_size: int) -> None:
        # TODO: deeper encoder
        super().__init__()
        self.input_size = input_size
        self.conv = nn.Conv1d(input_size, hidden_size, kernel_size=1)

    def forward(self, input) -> torch.Tensor:
        output = self.conv(input)
        return output  # (batch, hidden_size, seq_len)


class Attention(nn.Module):
    """Calculates attention over the input nodes given the current state."""

    def __init__(self, hidden_size, query_hidden_size, key_hidden_size) -> None:
        super(Attention, self).__init__()

        self.hidden_size = hidden_size

        # W processes features from static decoder elements
        self.v = nn.Parameter(torch.randn((1, 1, hidden_size), requires_grad=True))
        self.W = nn.Parameter(torch.randn((1, hidden_size, query_hidden_size + key_hidden_size), requires_grad=True))

    def forward(self, att_query: torch.Tensor, att_key: torch.Tensor, return_prob=False) -> torch.Tensor:
        """
        Args:
            att_query: (batch_size, query_hidden_size, num_queries)
            att_key: (batch_size, key_hidden_size, num_keys)
        """
        batch_size = att_query.shape[0]
        num_queries, num_keys = att_query.shape[-1], att_key.shape[-1]

        att_query = att_query.unsqueeze(3).expand(-1, -1, -1, num_keys)
        att_key = att_key.unsqueeze(2).expand(-1, -1, num_queries, -1)
        # both shape is (batch_size, query/key_hidden_size, num_queries, num_keys)

        # Concatenate the query and key features
        att_hidden = torch.cat((att_query, att_key), 1).flatten(start_dim=2)
        # (batch_size, query_hidden_size + key_hidden_size, num_queries * num_keys)

        # Broadcast some dimensions so we can do batch-matrix-multiply
        v = self.v.expand(batch_size, 1, self.hidden_size)
        W = self.W.expand(batch_size, self.hidden_size, -1)

        attns = torch.bmm(v, torch.tanh(torch.bmm(W, att_hidden)))  # (batch_size, 1, num_queries * num_keys)
        if return_prob:
            attns = F.softmax(attns, dim=-1).squeeze(1)

        return attns.squeeze(1)  # (batch_size, num_queries * num_keys)
