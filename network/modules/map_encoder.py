import torch
import torch.nn as nn

from typing import Dict

from network.layers.embedding import Embedding
from utilities.weight_init import weight_init


class MapEncoder(nn.Module):
    def __init__(self, hidden_dim: int):
        super(MapEncoder, self).__init__()
        self.hidden_dim = hidden_dim

        input_dim_x_pt = 1

        self.x_pt_emb = Embedding(input_dim=input_dim_x_pt, hidden_dim=hidden_dim)
        self.apply(weight_init)

    def forward(self, data: Dict) -> Dict:
        vector_pt = data["lanes_end_locs"] - data["lanes_start_locs"]
        x_pt = torch.norm(vector_pt, p=2, dim=-1).unsqueeze(-1)
        x_pt = self.x_pt_emb(continuous_inputs=x_pt, categorical_embs=None)

        return {"x_pt": x_pt}
