import torch
import torch.nn as nn

from typing import Dict
from torch import Tensor

from network.layers.embedding import Embedding
from utilities.transformation import angle_between_2d_vectors
from utilities.weight_init import weight_init


class AgentEncoder(nn.Module):
    def __init__(self, hidden_dim: int, rel_dis_norm_factor: float, vel_norm_factor: float, max_num_input_frames: int):
        super(AgentEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.rel_dis_norm_factor = rel_dis_norm_factor
        self.vel_norm_factor = vel_norm_factor
        self.max_num_input_frames = max_num_input_frames

        input_dim_x_a = 6
        type_dim_x_a = 1

        self.x_a_emb = Embedding(input_dim=input_dim_x_a, hidden_dim=hidden_dim)
        self.type_a_emb = Embedding(input_dim=type_dim_x_a, hidden_dim=hidden_dim)

        self.apply(weight_init)

    def forward(self, current_frame: int, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        # set the max num input frames
        if (self.max_num_input_frames is not None) and (current_frame + 1 > self.max_num_input_frames):
            max_num_input_frames = self.max_num_input_frames
        else:
            max_num_input_frames = current_frame + 1

        # igonre t=-2 as it is not helpful motion vector
        head_a = data["agents_heading_rad"][:, current_frame + 1 - max_num_input_frames : current_frame + 1]  # (A, 3)
        head_vector_a = torch.stack([head_a.cos(), head_a.sin()], dim=-1)
        pos_a = data["agents_position"][:, current_frame + 1 - max_num_input_frames : current_frame + 1]  # (A, 3, 2)
        motion_vector_a = pos_a[:, 1:] - pos_a[:, :-1]  # (A, 2, 2)
        motion_vector_a = torch.cat([torch.zeros_like(motion_vector_a[:, -1:]), motion_vector_a], dim=1)  # (A, 3, 2)
        vel_a = data["agents_vel"][:, current_frame + 1 - max_num_input_frames : current_frame + 1]  # (A, 3, 2)
        categorical_embs = [
            self.type_a_emb(data["agents_type"].float().unsqueeze(1)).repeat_interleave(
                repeats=max_num_input_frames, dim=0
            )  # [(A*3, 64)]
        ]
        a_angle_vel_and_head = angle_between_2d_vectors(ctr_vector=head_vector_a, nbr_vector=vel_a[:, :, :2])
        a_angle_motion_and_head = angle_between_2d_vectors(
            ctr_vector=head_vector_a, nbr_vector=motion_vector_a[:, :, :2]
        )
        x_a = torch.stack(
            [
                torch.norm(vel_a[:, :, :2], p=2, dim=-1) / self.vel_norm_factor,
                a_angle_vel_and_head.cos(),
                a_angle_vel_and_head.sin(),
                torch.norm(motion_vector_a[:, :, :2], p=2, dim=-1) / self.rel_dis_norm_factor,
                a_angle_motion_and_head.cos(),
                a_angle_motion_and_head.sin(),
            ],
            dim=-1,
        )
        x_a = self.x_a_emb(continuous_inputs=x_a.view(-1, x_a.size(-1)), categorical_embs=categorical_embs)
        x_a = x_a.view(-1, max_num_input_frames, self.hidden_dim)

        return {"x_a": x_a}
