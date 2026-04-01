import numpy as np
import math
import torch

from torch import Tensor
from shapely.affinity import affine_transform
from shapely.geometry import Polygon, Point
from typing import Optional, Tuple, Dict

from config.nuscenes_config import CityCenterlinesGraph, CityNodesFeatures
from torch_geometric.utils import subgraph


def calculate_yaw_angle(quaternion):
    qw, qx, qy, qz = quaternion
    return np.rad2deg((math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))))


def wrap_angle_deg(angle: Tensor, min_val: float = -180.0, max_val: float = 180.0) -> Tensor:
    return min_val + (angle + max_val) % (max_val - min_val)


def wrap_angle_rad(angle: Tensor, min_val: float = -np.pi, max_val: float = np.pi) -> Tensor:
    return min_val + (angle + max_val) % (max_val - min_val)


def create_patch(patch_size: Tuple, angle_rad: float, agent_pos: Tuple, offset: Tuple) -> Tuple[Polygon, np.vectorize]:
    """
    Theta = 0 along the x axis of the agent
    """
    patch_h, patch_w = patch_size
    patch = Polygon(((patch_h / 2, patch_w / 2), (-patch_h / 2, patch_w / 2), (-patch_h / 2, -patch_w / 2), (patch_h / 2, -patch_w / 2)))
    rot_mat = [np.cos(angle_rad), -np.sin(angle_rad), np.sin(angle_rad), np.cos(angle_rad), 0, 0]
    patch_rot = affine_transform(patch, rot_mat)  # patch at the target_loc at the angle
    offset_rotated = affine_transform(Point(offset), rot_mat)
    off_w, off_h = offset_rotated.x, offset_rotated.y
    trans_matrix_vcs = [1, 0, 0, 1, agent_pos[0] - off_w, agent_pos[1] - off_h]
    patch_rot_shift = affine_transform(patch_rot, trans_matrix_vcs)
    patch_container = np.vectorize(lambda p: patch_rot_shift.contains(Point(p)), signature="(n)->()", otypes=[np.bool_])
    return patch_rot_shift, patch_container


def sample_subgraph_in_patch(patch_container: np.vectorize, vectorized_map: CityCenterlinesGraph) -> CityCenterlinesGraph:
    nodes_id = vectorized_map.nodes_features.nodes_id
    start_locs = vectorized_map.nodes_features.start_locs
    end_locs = vectorized_map.nodes_features.end_locs
    edges_indices = vectorized_map.edges_indices
    sampled_nodes = patch_container(np.array(start_locs))
    # sampled features
    sampled_nodes_features = CityNodesFeatures(nodes_id=nodes_id[sampled_nodes], start_locs=start_locs[sampled_nodes], end_locs=end_locs[sampled_nodes])
    sampled_edges_indices = subgraph(torch.tensor(sampled_nodes), edges_indices.to(torch.int64), relabel_nodes=True)[0]
    vectorized_map_patch = CityCenterlinesGraph(nodes_features=sampled_nodes_features, edges_indices=sampled_edges_indices)
    return vectorized_map_patch


# transform trajectory from global coordinates to vehicle coordinates system
def gcs_to_vcs_tensor(
    target_gcs_pos: Tensor,
    target_gcs_head_deg: torch.FloatTensor,
    traj_gcs_pos: Tensor,
    traj_gcs_head_deg: Optional[Tensor] = None,
) -> Tuple[Tensor, Optional[Tensor]]:
    """
    INPUT:
    target_gcs_pos: target vehicle pos at T=0 in global coordinates, shape=(2)
    target_gcs_head_deg: target vehicle heading at T=0 in global coordinates, Float
    traj_gcs_pos: all vehicles pos in global coordinates, shape=(Na, T, 2)
    traj_gcs_head_deg: all vehicles heading in global coordinates, shape=(Na, T)

    OUTPUT:
    traj_vcs_pos: all vehicles pos in target vehicle coordinates at T=0, shape=(Na, T, 2)
    traj_vcs_head_rad: all vehicles heading in target vehicle coordinates at T=0, shape=(Na, T)
    
    #################### Global coordinate System ###########################
            ^ Y (along the direction of vehicle)
            |
       X<---| (LHS)
    heading (degrees): zero along the Y axis, +ve towards the X axis (RHS)
    
    #################### Vehicle coordinate System ###########################
            ^ X (along the direction of vehicle)
            |
       Y<---| (RHS)
    heading (radians): zero along the X axis, +ve towards the Y axis (RHS)
    origin: target agent
    
    """
    d_type = traj_gcs_pos.dtype
    target_gcs_pos = target_gcs_pos.to(dtype=d_type)
    target_gcs_head_deg = target_gcs_head_deg.to(dtype=d_type)

    # swap the axis
    target_gcs_rhs_pos = target_gcs_pos[..., [1, 0]]
    traj_gcs_rhs_pos = traj_gcs_pos[..., [1, 0]]
    target_gcs_rhs_head_deg = target_gcs_head_deg # alredy in RHS
    if traj_gcs_head_deg is not None:
        traj_gcs_rhs_head_deg = traj_gcs_head_deg # already in RHS
    
    # compute the trasnformation matrix
    target_gcs_rhs_head_rad = torch.deg2rad(target_gcs_rhs_head_deg)
    vcs_to_gcs_rhs_trans_matrix = torch.stack(
        (
            torch.stack((torch.cos(target_gcs_rhs_head_rad), -torch.sin(target_gcs_rhs_head_rad), target_gcs_rhs_pos[..., 0]), dim=-1),
            torch.stack((torch.sin(target_gcs_rhs_head_rad), torch.cos(target_gcs_rhs_head_rad), target_gcs_rhs_pos[..., 1]), dim=-1),
            torch.stack((torch.zeros_like(target_gcs_rhs_head_rad), torch.zeros_like(target_gcs_rhs_head_rad), torch.ones_like(target_gcs_rhs_head_rad)), dim=-1),
        ),
        dim=-2,
    )  # shape (3, 3)
    gcs_rhs_to_vcs_trans_matrix = torch.linalg.inv(vcs_to_gcs_rhs_trans_matrix)
    
    # transform the trajectories in rhs
    traj_gcs_rhs_pos = torch.concat([traj_gcs_rhs_pos, torch.ones_like(traj_gcs_rhs_pos[:, :, :1])], dim=-1)
    traj_vcs_pos = torch.matmul(gcs_rhs_to_vcs_trans_matrix, traj_gcs_rhs_pos.transpose(-2, -1)).transpose(-2, -1)
    traj_vcs_pos = traj_vcs_pos[:, :, :2]

    # compute heading of traj in vcs
    if traj_gcs_head_deg is not None:
        traj_vcs_head_deg = traj_gcs_rhs_head_deg - target_gcs_rhs_head_deg
        traj_vcs_head_rad = torch.deg2rad(traj_vcs_head_deg)
        traj_vcs_head_rad = wrap_angle_rad(traj_vcs_head_rad)
    else:
        traj_vcs_head_rad = None

    return traj_vcs_pos, traj_vcs_head_rad


def vcs_to_gcs_tensor(
    target_gcs_pos: Tensor,
    target_gcs_head_deg: Tensor,
    traj_vcs_pos: Tensor,
    traj_vcs_head_rad: Optional[Tensor] = None,
) -> Tuple[Tensor, Optional[Tensor]]:
    """
    INPUT:
    INPUT:
    target_gcs_pos: target vehicle pos at T=0 in global coordinates, shape=(2)
    target_gcs_head_deg: target vehicle heading at T=0 in global coordinates, Float
    traj_vcs_pos: all vehicles pos in target vehicle coordinates at T=0, shape=(Na, M, T, 2)
    traj_vcs_head_rad: all vehicles heading in target vehicle coordinates at T=0, shape=(Na, M, T)

    OUTPUT:
    traj_gcs_pos: all vehicles pos in global coordinates, shape=(Na, M, T, 2)
    traj_gcs_head_deg: all vehicles heading in global coordinates, shape=(Na, M, T)
    """
    target_gcs_rhs_pos = target_gcs_pos[..., [1, 0]]
    target_gcs_rhs_head_deg = target_gcs_head_deg # alredy in RHS
    
    # get the transformation matrix
    target_gcs_rhs_head_rad = torch.deg2rad(target_gcs_rhs_head_deg)
    vcs_to_gcs_rhs_trans_matrix = torch.stack( 
        (
            torch.stack((torch.cos(target_gcs_rhs_head_rad), -torch.sin(target_gcs_rhs_head_rad), target_gcs_rhs_pos[..., 0]), dim=-1),
            torch.stack((torch.sin(target_gcs_rhs_head_rad), torch.cos(target_gcs_rhs_head_rad), target_gcs_rhs_pos[..., 1]), dim=-1),
            torch.stack((torch.zeros_like(target_gcs_rhs_head_rad), torch.zeros_like(target_gcs_rhs_head_rad), torch.ones_like(target_gcs_rhs_head_rad)), dim=-1),
        ),
        dim=-2,
    ) # shape (3, 3)
    
    # transform the trajectory
    traj_vcs_pos = torch.concat([traj_vcs_pos, torch.ones_like(traj_vcs_pos[..., :1])], dim=-1)
    traj_gcs_rhs_pos = torch.matmul(vcs_to_gcs_rhs_trans_matrix, traj_vcs_pos.transpose(-2, -1)).transpose(-2, -1)
    traj_gcs_rhs_pos = traj_gcs_rhs_pos[..., :2]
    
    if traj_vcs_head_rad is not None:
        traj_vcs_head_deg = torch.rad2deg(traj_vcs_head_rad)
        traj_gcs_rhs_head_deg = wrap_angle_deg(traj_vcs_head_deg + target_gcs_rhs_head_deg)  
    else:
        traj_gcs_rhs_head_deg = None
    
    # swap the axis
    traj_gcs_pos = traj_gcs_rhs_pos[..., [1, 0]]
    if traj_vcs_head_rad is not None:
        traj_gcs_head_deg = traj_gcs_rhs_head_deg # already in RHS
    else: 
        traj_gcs_head_deg = None

    return traj_gcs_pos, traj_gcs_head_deg


def local_to_vcs_tensor(
    local_vcs_pos: Tensor,
    local_vcs_head_rad: Tensor,
    traj_local_pos: Tensor,
    traj_local_head_rad: Optional[Tensor] = None,
    scale_local_pos: Optional[torch.Tensor] = None,
) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
    """    
    INPUT:
    local_vcs_pos: all vehicles pos at T=0 in target vehicle coordinates at T=0, shape=(Na, 2)
    local_vcs_head_rad: all vehicles heading at T=0 in target vehicle coordinates at T=0, shape=(Na,)
    traj_local_pos: all vehicles pos in their respective vehicle coordinates at T=0, shape=(Na, M, T, 2)
    traj_local_head_rad: all vehicles heading in their respective vehicle coordinates at T=0, shape=(Na, M, T)
    scale_local_pos: all vehicles pos_scales in their respective vehicle coordinates at T=0, shape=(Na, M, T, 2, 2)
    OUTPUT:
    traj_vcs_pos: all vehicles pos in target vehicle coordinates at T=0, shape=(Na, M, T, 2)
    traj_vcs_head_rad: all vehicles heading in target vehicle coordinates at T=0, shape=(Na, M, T)
    scale_vcs_pos: all vehicles pos_scales in target vehicle coordinates at T=0, shape=(Na, M, T, 2, 2)
    """
    M = traj_local_pos.shape[1]
    # compute the transformation matrix
    local_to_vcs_trans_matrix = torch.stack(  # (Na, 3, 3)
        (
            torch.stack((torch.cos(local_vcs_head_rad), -torch.sin(local_vcs_head_rad), local_vcs_pos[:, 0]), dim=-1),
            torch.stack((torch.sin(local_vcs_head_rad), torch.cos(local_vcs_head_rad), local_vcs_pos[:, 1]), dim=-1),
            torch.stack((torch.zeros_like(local_vcs_head_rad), torch.zeros_like(local_vcs_head_rad), torch.ones_like(local_vcs_head_rad)), dim=-1),
        ),
        dim=-2,
    )
    local_to_vcs_trans_matrix = local_to_vcs_trans_matrix.unsqueeze(1).expand(-1, M, -1, -1) # (Na, M, 3, 3)
    
    # transform the trajectories
    traj_local_pos = torch.concat((traj_local_pos[..., :2], torch.ones_like(traj_local_pos[..., 0:1])), dim=-1)
    traj_vcs_pos = torch.matmul(local_to_vcs_trans_matrix, traj_local_pos.transpose(-2, -1)).transpose(-2, -1)
    traj_vcs_pos = traj_vcs_pos[..., :2]

    if traj_local_head_rad is not None:
        traj_vcs_head_rad = traj_local_head_rad + local_vcs_head_rad.unsqueeze(1).unsqueeze(2).expand(*traj_local_head_rad.shape)
        traj_vcs_head_rad = wrap_angle_rad(traj_vcs_head_rad)
    else:
        traj_vcs_head_rad = None

    if scale_local_pos is not None:
        T = traj_local_pos.shape[2]
        assert scale_local_pos.shape[-2:] == (2, 2), 'Transformation of scale is not written for Euler angles'
        local_to_vcs_rot_matrix = local_to_vcs_trans_matrix[:, :, None, :2, :2].repeat(1, 1, T, 1, 1) # (Na, M, T, 2, 2)
        scale_vcs_pos = local_to_vcs_rot_matrix @ scale_local_pos @ local_to_vcs_rot_matrix.transpose(-2, -1) # (Na, M, T, 2, 2)
    else:
        scale_vcs_pos = None
    
    return traj_vcs_pos, traj_vcs_head_rad, scale_vcs_pos


def vcs_to_local_tensor(
    local_vcs_pos: Tensor,
    local_vcs_head_rad: Tensor,
    traj_vcs_pos: Tensor,
    traj_vcs_head_rad: Optional[Tensor] = None,
    scale_vcs_pos: Optional[torch.Tensor] = None,
) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
    """
    INPUT:
    local_vcs_pos: all vehicles pos at T=0 in target vehicle coordinates at T=0, shape=(Na, 2)
    local_vcs_head_rad: all vehicles heading at T=0 in target vehicle coordinates at T=0, shape=(Na,)
    traj_vcs_pos: all vehicles pos in target vehicle coordinates at T=0, shape=(Na, T, 2)
    traj_vcs_head_rad: all vehicles heading in target vehicle coordinates at T=0, shape=(Na, T)
    scale_vcs_pos: all vehicles pos_scales in target vehicle coordinates at T=0, shape=(Na, T, 2, 2)
    OUTPUT:
    traj_local_pos: all vehicles pos in their respective vehicle coordinates at T=0, shape=(Na, T, 2)
    traj_local_head_rad: all vehicles heading in their respective vehicle coordinates at T=0, shape=(Na, T)
    scale_local_pos: all vehicles pos_scales in their respective vehicle coordinates at T=0, shape=(Na, T, 2, 2)
    """
    # compute the transformation matrix
    local_to_vcs_trans_matrix = torch.stack(  # (Na, 3, 3)
        (
            torch.stack((torch.cos(local_vcs_head_rad), -torch.sin(local_vcs_head_rad), local_vcs_pos[:, 0]), dim=-1),
            torch.stack((torch.sin(local_vcs_head_rad), torch.cos(local_vcs_head_rad), local_vcs_pos[:, 1]), dim=-1),
            torch.stack((torch.zeros_like(local_vcs_head_rad), torch.zeros_like(local_vcs_head_rad), torch.ones_like(local_vcs_head_rad)), dim=-1),
        ),
        dim=-2,
    )
    vcs_to_local_trans_matrix = torch.linalg.inv(local_to_vcs_trans_matrix)
    
    # transform the trajectories
    traj_vcs_pos = torch.concat((traj_vcs_pos[..., :2], torch.ones_like(traj_vcs_pos[..., 0:1])), dim=-1)
    traj_local_pos = torch.matmul(vcs_to_local_trans_matrix, traj_vcs_pos.transpose(-2, -1)).transpose(-2, -1)
    traj_local_pos = traj_local_pos[..., :2]

    if traj_vcs_head_rad is not None:
        traj_local_head_rad = traj_vcs_head_rad - local_vcs_head_rad.unsqueeze(1).expand(*traj_vcs_head_rad.shape)
        traj_local_head_rad = wrap_angle_rad(traj_local_head_rad)
    else:
        traj_local_head_rad = None

    if scale_vcs_pos is not None:
        T = traj_vcs_pos.shape[1]
        vcs_to_local_rot_matrix = vcs_to_local_trans_matrix[:, None, :2, :2].repeat(1, T, 1, 1) # (Na, T, 2, 2)
        scale_local_pos = vcs_to_local_rot_matrix @ scale_vcs_pos @ vcs_to_local_rot_matrix.transpose(-2, -1) # (Na, T, 2, 2)
    else:
        scale_local_pos = None
    
    return traj_local_pos, traj_local_head_rad, scale_local_pos


def angle_between_2d_vectors(ctr_vector: Tensor, nbr_vector: Tensor) -> Tensor:
    return safe_atan2(
        ctr_vector[..., 0] * nbr_vector[..., 1] - ctr_vector[..., 1] * nbr_vector[..., 0], 
        (ctr_vector[..., :2] * nbr_vector[..., :2]).sum(dim=-1)
    )


def recurr_to_vcs_to_local_tensor(
    recurr_vcs_pos: Tensor,
    recurr_vcs_head_rad: Tensor,
    local_vcs_pos: Tensor,
    local_vcs_head_rad: Tensor,
    traj_recurr_pos: Tensor,
    scale_recurr_pos: Optional[Tensor] = None,
    sample_frequency: Optional[int] = None,
) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
    '''
    Input:
    recurr_vcs_pos: shape=(Na, 2)
    recurr_vcs_head_rad: shape=(Na,)
    local_vcs_pos: shape=(Na, 2)
    local_vcs_head_rad: shape=(Na,)
    traj_recurr_pos: shape=(Na, T, 2)
    scale_recurr_pos: shape=(Na, T, 2, 2)
    
    Output:
    traj_vcs_pos: shape=(Na, T, 2)
    traj_vcs_head_rad: shape=(Na, T)
    traj_vcs_vel: shape=(Na, T, 2)
    traj_local_pos: shape=(Na, T, 2)
    scale_local_pos: shape=(Na, T, 2, 2)
    
    Coordinate system:
    recurr: current agent position
    local: T=0 agent position
    vcs: T=0 target vehicle position
    '''
    # concatanate T=0
    traj_recurr_pos_ = torch.concat([torch.zeros_like(traj_recurr_pos[:, :1, :]), traj_recurr_pos], dim=-2) # (Na, T+1, 2)
    if scale_recurr_pos is not None:
        scale_recurr_pos_ = torch.concat([torch.zeros_like(scale_recurr_pos[:, :1, :, :]), scale_recurr_pos], dim=-3) # (Na, T+1, 2, 2)
    else:
        scale_recurr_pos_ = torch.zeros(*(*traj_recurr_pos_.shape, 2), device=traj_recurr_pos.device)
    
    # transform from recurr coordinate system to vcs 
    traj_vcs_pos_, _, scale_vcs_pos_ = local_to_vcs_tensor(recurr_vcs_pos, recurr_vcs_head_rad, traj_recurr_pos_[:, None, :, :], 
                                        scale_local_pos=scale_recurr_pos_[:, None, :, :, :]) # (Na, 1, T+1, 2), (Na, 1, T+1, 2, 2)
    traj_vcs_pos_ = traj_vcs_pos_.squeeze(1) # (Na, T+1, 2)
    scale_vcs_pos_ = scale_vcs_pos_.squeeze(1) # (Na, T+1, 2, 2)
    
    # calculate heading and velocity in the vcs
    traj_vcs_head_rad = safe_atan2(
        traj_vcs_pos_[:, 1:, 1] - traj_vcs_pos_[:, :-1, 1], 
        traj_vcs_pos_[:, 1:, 0] - traj_vcs_pos_[:, :-1, 0]  
    ) # (Na, T)
    if sample_frequency is not None:
        traj_vcs_vel = sample_frequency * torch.stack(
            [traj_vcs_pos_[:, 1:, 0] - traj_vcs_pos_[:, :-1, 0], 
            traj_vcs_pos_[:, 1:, 1] - traj_vcs_pos_[:, :-1, 1]],  
            dim=-1,
        ) # (Na, T, 2)
    else: 
        traj_vcs_vel = None
    traj_vcs_pos = traj_vcs_pos_[:, 1:, :] # (Na, T, 2)
    scale_vcs_pos = scale_vcs_pos_[:, 1:, :, :] # (Na, T, 2, 2)
    
    # transform from vcs to local coordinate system
    traj_local_pos, _, scale_local_pos = vcs_to_local_tensor(local_vcs_pos, local_vcs_head_rad, traj_vcs_pos,
                                         scale_vcs_pos=scale_vcs_pos) # (Na, T, 2), (Na, T, 2, 2)
    
    return traj_vcs_pos, traj_vcs_head_rad, traj_vcs_vel, traj_local_pos, scale_local_pos
    
    
def simulate_single_trajectory_per_agent_(
    data: Dict, sim_mask: Tensor, current_frame: int, num_sim_steps: int, 
    trajectories_vcs_pos: Tensor, trajectories_vcs_head_rad: Tensor, trajectories_vcs_vel: Tensor,
    diff_simulator: bool = False,
):
    '''
    trajectories_vcs_pos: (A, T, 2)
    trajectories_vcs_head_rad: (A, T)
    trajectories_vcs_vel: (A, T)
    sim_mask: (A,)
    '''
    # 1. Assertions for non-differentiable mode
    if not diff_simulator:
        assert not trajectories_vcs_pos.requires_grad, "Simulated trajectories must be detached"
        assert not trajectories_vcs_head_rad.requires_grad, "Simulated trajectories must be detached"
        assert not trajectories_vcs_vel.requires_grad, "Simulated trajectories must be detached"
    else: 
        assert trajectories_vcs_pos.requires_grad, "Simulated trajectories must have gradients"
        assert trajectories_vcs_head_rad.requires_grad, "Simulated trajectories must have gradients"
        assert trajectories_vcs_vel.requires_grad, "Simulated trajectories must have gradients"

    # 2. Select Targets: Clone if diff_simulator, else use Reference
    pos = data["agents_position"].clone() if diff_simulator else data["agents_position"]
    head = data["agents_heading_rad"].clone() if diff_simulator else data["agents_heading_rad"]
    vel = data["agents_vel"].clone() if diff_simulator else data["agents_vel"]
    reg_mask = data["agents_input_reg_mask"].clone() if diff_simulator else data["agents_input_reg_mask"]

    # 3. Common Assignment Logic
    t_slice = slice(current_frame + 1, current_frame + 1 + num_sim_steps)
    sim_slice = slice(0, num_sim_steps)

    pos[sim_mask, t_slice, :] = trajectories_vcs_pos[sim_mask, sim_slice, :]
    head[sim_mask, t_slice] = trajectories_vcs_head_rad[sim_mask, sim_slice]
    vel[sim_mask, t_slice, :] = trajectories_vcs_vel[sim_mask, sim_slice, :]
    reg_mask[:, t_slice] = data["agents_log_reg_mask"][:, t_slice]

    # 4. Update dictionary (only necessary if we created clones)
    if diff_simulator:
        data["agents_position"] = pos
        data["agents_heading_rad"] = head
        data["agents_vel"] = vel
        data["agents_input_reg_mask"] = reg_mask


def get_bb_corners(centers: Tensor, rotation_angle: Tensor, sizes: Tensor) -> Tensor:
    """
    INPUTS:
    centers: (N, T, 2)
    rotation_angle: (N, T)
    bb_sizes: (N, T, 2) where each row is (width, length), where length is synonym to height

    OUTPUTS:
    (N, T, 4, 2) where each row is (x, y) coordinates of the corners in clockwise order
    
    :                  +------------+ X
      :                |            |
      :                |          length
      :                |            |
      :              Y +-- width --(xy)
      
    """
    rot_mat = torch.stack(  # (N, T, 3, 3) Not a transformation matrix, but a rotation matrix, x, y being zeros
        (
            torch.stack((torch.cos(rotation_angle), -torch.sin(rotation_angle), torch.zeros_like(rotation_angle)), dim=-1),
            torch.stack((torch.sin(rotation_angle), torch.cos(rotation_angle), torch.zeros_like(rotation_angle)), dim=-1),
            torch.stack((torch.zeros_like(rotation_angle), torch.zeros_like(rotation_angle), torch.ones_like(rotation_angle)), dim=-1),
        ),
        dim=-2,
    )

    offsets = torch.stack(  # (N, T, 4, 2)
        [
            torch.stack([-sizes[..., 1] / 2, -sizes[..., 0] / 2], dim=-1),
            torch.stack([sizes[..., 1] / 2, -sizes[..., 0] / 2], dim=-1),
            torch.stack([sizes[..., 1] / 2, sizes[..., 0] / 2], dim=-1),
            torch.stack([-sizes[..., 1] / 2, sizes[..., 0] / 2], dim=-1),
        ],
        dim=-2,
    )
    offsets = torch.concat([offsets, torch.ones_like(offsets[..., -1:])], dim=-1)  # (N, T, 4, 3)

    rot_offsets = torch.matmul(offsets, rot_mat.transpose(-2, -1))  # (N, T, 4, 3)
    rot_offsets = rot_offsets[..., :2]  # (N, T, 4, 2)
    corners = centers.unsqueeze(-2) + rot_offsets  # (N, T, 4, 2)
    return corners


def safe_atan2(y, x, eps=1e-6):
    # Determine which agents are stationary
    is_stationary = (torch.abs(x) < eps) & (torch.abs(y) < eps)
    
    # create safe_x and safe_y leading to 0 heading for stationary agents 
    safe_x = torch.where(is_stationary, torch.full_like(x, eps), x)
    safe_y = torch.where(is_stationary, torch.zeros_like(y), y)
    
    return torch.atan2(safe_y, safe_x)