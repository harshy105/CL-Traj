import torch

from torch import Tensor
from typing import Optional

from utilities.utils import bipartite_dense_to_sparse
from utilities.transformation import wrap_angle_rad, angle_between_2d_vectors

def purge_far_away_edges(edge_a2b: Tensor, dis_threshold: float,
                         pos_a: Tensor, pos_b: Tensor) -> Tensor:
    rel_pos_a2b = pos_a[edge_a2b[0]] - pos_b[edge_a2b[1]]
    edge_a2b_purged = edge_a2b[:, rel_pos_a2b.norm(dim=1) < dis_threshold]
    return edge_a2b_purged

def compute_rel_pos_time_emb(edge_a2b: Tensor,
            pos_a: Tensor, pos_b: Tensor, 
            head_a: Tensor, head_b: Tensor,
            time_a: Optional[Tensor] = None, 
            time_b: Optional[Tensor] = None, 
            rel_dis_norm_factor: Optional[float] = 1.0,
            rel_time_norm_factor: Optional[float] = 1.0,
            ) -> Tensor: 
    rel_pos_a2b = pos_a[edge_a2b[0]] - pos_b[edge_a2b[1]]
    rel_head_a2b = wrap_angle_rad(head_a[edge_a2b[0]] - head_b[edge_a2b[1]])
    head_vector_b = torch.stack([head_b.cos(), head_b.sin()], dim=-1)
    rel_direction_a2b = angle_between_2d_vectors(ctr_vector=head_vector_b[edge_a2b[1]], nbr_vector=rel_pos_a2b[:, :2])
    rel_a2b = torch.stack([torch.norm(rel_pos_a2b[:, :2], p=2, dim=-1)/rel_dis_norm_factor,
                            rel_direction_a2b.cos(),
                            rel_direction_a2b.sin(),
                            rel_head_a2b.cos(),
                            rel_head_a2b.sin()], dim=-1)
    if time_a is not None and time_b is not None:
        rel_time_a2b = (time_a[edge_a2b[0]] - time_b[edge_a2b[1]])/rel_time_norm_factor
        rel_a2b = torch.concat([rel_a2b, rel_time_a2b[:, None]], dim=-1)
    return rel_a2b
    
def get_prev2mode_edges(mask_at: Tensor, mask_am: Tensor) -> Tensor:
    "Create connections from input time steps to all the modes for each agents"
    edge_at2am_dense = mask_at.unsqueeze(2) & mask_am.unsqueeze(1)
    edge_at2am = bipartite_dense_to_sparse(edge_at2am_dense)
    return edge_at2am

def get_surr2mode_edges(num_modes: int, mask_am: Tensor, mask_ac: Tensor, agents_batch_idx: Tensor) -> Tensor:
    "Create connections from surrounding agents last time step to all the modes for each agents"
    # set the modes mask to nan for the agents which are not predicted
    mask_am_nan = torch.ones_like(mask_am) * torch.nan 
    mask_am_nan[mask_am] = True  
    mask_ma_nan = mask_am_nan.transpose(0, 1) * agents_batch_idx
    # set the agents mask to nan if they are not observable at the current time step
    mask_ac_nan = torch.ones_like(mask_ac) * torch.nan
    mask_ac_nan[mask_ac] = True
    mask_ac_nan = mask_ac_nan * agents_batch_idx
    mask_mac_nan = mask_ac_nan.unsqueeze(0).repeat(num_modes, 1)
    # make the edge connection agents to modes for every mode individually
    edge_ma2ma_dense = (mask_mac_nan.unsqueeze(2) - mask_ma_nan.unsqueeze(1)) == 0 
    edge_ma2ma = bipartite_dense_to_sparse(edge_ma2ma_dense)
    # remove mode connection its own agents
    edge_ma2ma = edge_ma2ma[:, edge_ma2ma[0] != edge_ma2ma[1]]
    return edge_ma2ma
    
def get_map2mode_edges(num_modes: int, mask_am: Tensor, agents_batch_idx: Tensor, lanes_batch_idx: Tensor) -> Tensor:
    "Create connections from map to all the modes for each agents"
    # set the modes mask to nan for the agents which are not predicted
    mask_am_nan = torch.ones_like(mask_am) * torch.nan 
    mask_am_nan[mask_am] = True
    mask_ma_nan = mask_am_nan.transpose(0, 1) * agents_batch_idx
    # make the edge connection from map to mode every mode individually
    no_mask_mpt = (lanes_batch_idx.unsqueeze(0).repeat(num_modes, 1))  # all the maps nodes are visible to all modes
    edge_mpt2ma_dense = (no_mask_mpt.unsqueeze(2) - mask_ma_nan.unsqueeze(1)) == 0  # make edge to agent connection with same batch and mode
    edge_mpt2ma = bipartite_dense_to_sparse(edge_mpt2ma_dense)
    return edge_mpt2ma

def get_goal2mode_edges(mask_ag: Tensor, mask_am: Tensor) -> Tensor:
    "Create connections from goal time steps to all the modes for each agents"
    edge_ag2am_dense = mask_ag.unsqueeze(2) & mask_am.unsqueeze(1)
    edge_ag2am = bipartite_dense_to_sparse(edge_ag2am_dense)
    return edge_ag2am