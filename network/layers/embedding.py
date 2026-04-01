import math
from typing import List, Optional

import torch
import torch.nn as nn

from torch import Tensor
from utilities.weight_init import weight_init


class Embedding(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super(Embedding, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(1, hidden_dim * 2),
                    nn.LayerNorm(hidden_dim * 2),
                    nn.ReLU(inplace=True),
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for _ in range(input_dim)
            ]
        )
        self.to_out = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.apply(weight_init)

    def forward(
        self, continuous_inputs: Optional[Tensor] = None, categorical_embs: Optional[List[Tensor]] = None
    ) -> Tensor:
        if continuous_inputs is None:
            if categorical_embs is not None:
                x = torch.stack(categorical_embs).sum(dim=0)
            else:
                raise ValueError("Both continuous_inputs and categorical_embs are None")
        else:
            assert continuous_inputs.shape[-1] == self.input_dim, "Embedding dimensions is incorrect"
            x = continuous_inputs.unsqueeze(-1)
            continuous_embs: List[Optional[Tensor]] = [None] * self.input_dim
            for i in range(self.input_dim):
                continuous_embs[i] = self.mlps[i](x[:, i])
            x = torch.stack(continuous_embs).sum(dim=0)
            if categorical_embs is not None:
                x = x + torch.stack(categorical_embs).sum(dim=0)
        return self.to_out(x)
