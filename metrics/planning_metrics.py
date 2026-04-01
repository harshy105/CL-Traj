import torch

from torch import Tensor
from typing import Dict
from utilities.transformation import get_bb_corners


class PlanningMetric:
    """
    OBB-only collision metric with per-agent per-timestep width/length support.
    Supports multi-modal predictions.
    """

    def __init__(self, n_timesteps: int = 6):
        super().__init__()
        self.n_timesteps = n_timesteps
        self.L2_list = []  # List of (B, M, n_timesteps) per batch
        self.obj_box_col_list = []  # List of (B, M, n_timesteps) per batch
        self.mode_probs_list = []  # List of (B, M) per batch

    @staticmethod
    def _obb_overlap_pairs(c1: Tensor, c2: Tensor, eps: float = 1e-8) -> Tensor:
        """
        Vectorized SAT for many rectangle pairs.
        c1: (M,4,2)
        c2: (M,4,2)
        returns: (M,) bool whether overlap
        """
        e1_a = c1[:, 1, :] - c1[:, 0, :]  # (M,2)
        e1_b = c1[:, 2, :] - c1[:, 1, :]
        e2_a = c2[:, 1, :] - c2[:, 0, :]
        e2_b = c2[:, 2, :] - c2[:, 1, :]

        axes = torch.stack([e1_a, e1_b, e2_a, e2_b], dim=1)  # (M,4,2)
        axis_norm = torch.norm(axes, dim=-1, keepdim=True).clamp_min(eps)
        axes = axes / axis_norm  # (M,4,2) normalized

        proj1 = torch.einsum("mcj,mkj->mkc", c1, axes)  # (M,4_axes,4_corners)
        proj2 = torch.einsum("mcj,mkj->mkc", c2, axes)

        min1, _ = proj1.min(dim=-1)  # (M,4_axes)
        max1, _ = proj1.max(dim=-1)
        min2, _ = proj2.min(dim=-1)
        max2, _ = proj2.max(dim=-1)

        overlap = (max1 >= min2) & (max2 >= min1)  # (M,4_axes)
        return overlap.all(dim=-1)  # (M,)

    def _pairwise_ego_vs_others(
        self, ego_centers: Tensor, ego_heads: Tensor, 
        ego_sizes: Tensor, others_centers: Tensor, others_heads: Tensor, 
        others_sizes: Tensor
    ) -> Tensor:
        """
        ego_centers: (B, M, n_timesteps, 2)
        ego_heads: (B, M, n_timesteps)
        ego_sizes: (B, n_timesteps, 2)
        others_centers: (B, n_agents, n_timesteps, 2)
        others_heads: (B, n_agents, n_timesteps)
        others_sizes: (B, n_agents, n_timesteps, 2)
        returns: (B, n_agents, n_timesteps) bool collisions between ego box and each other box
        """
        B, n_agents, n_timesteps, _ = others_centers.shape
        M = ego_centers.shape[1]

        c1 = get_bb_corners(
            ego_centers.reshape(B, M * n_timesteps, 2),
            ego_heads.reshape(B, M * n_timesteps),
            ego_sizes.unsqueeze(1).repeat(1, M, 1, 1).reshape(B, M * n_timesteps, 2),
        )  # (B, M*n_timesteps, 4, 2)
        c1 = (
            c1.reshape(B, M, n_timesteps, 4, 2)
            .unsqueeze(2)
            .repeat(1, 1, n_agents, 1, 1, 1)
            .reshape(B * M * n_agents * n_timesteps, 4, 2)
        )

        c2 = get_bb_corners(
            others_centers.reshape(B, n_agents * n_timesteps, 2),
            others_heads.reshape(B, n_agents * n_timesteps),
            others_sizes.reshape(B, n_agents * n_timesteps, 2),
        )  # (B, n_agents*n_timesteps, 4, 2)
        c2 = c2.unsqueeze(1).repeat(1, M, 1, 1, 1).reshape(B * M * n_agents * n_timesteps, 4, 2)

        overlap_flat = self._obb_overlap_pairs(c1, c2)
        return overlap_flat.reshape(B, M, n_agents, n_timesteps)

    def compute_L2(
        self, trajs: Tensor, ego_gt_trajs: Tensor, 
        ego_gt_trajs_mask: Tensor
    ) -> Tensor:
        """
        trajs: (B, M, n_timesteps, 3)
        ego_gt_trajs: (B, n_timesteps, 3)
        ego_gt_trajs_mask: (B, n_timesteps)
        """
        diff = trajs[:, :, :, :2] - ego_gt_trajs[:, None, :, :2]  # (B, M, n_timesteps, 2)
        mask = ego_gt_trajs_mask[:, None, :, None]  # (B, 1, n_timesteps, 1)
        l2 = torch.sqrt((diff**2 * mask).sum(dim=-1))  # (B, M, n_timesteps)
        return l2

    def evaluate_coll(
        self, ego_trajs: Tensor, ego_gt_trajs: Tensor, ego_wh_trajs: Tensor,
        ego_gt_trajs_mask: Tensor, others_gt_trajs: Tensor,
        others_wh_trajs: Tensor, others_gt_trajs_mask: Tensor,
    ) -> Tensor:
        """
        ego_trajs: (B, M, n_timesteps, 3)
        ego_gt_trajs: (B, n_timesteps, 3)
        ego_wh_trajs: (B, n_timesteps, 2)
        ego_gt_trajs_mask: (B, n_timesteps)
        others_gt_trajs: (B, n_agents, n_timesteps, 3)
        others_wh_trajs: (B, n_agents, n_timesteps, 2)
        others_gt_trajs_mask: (B, n_agents, n_timesteps)
        """
        B, M, n_timesteps, _ = ego_trajs.shape
        n_agents = others_gt_trajs.shape[1] if (others_gt_trajs.numel() > 0) else 0
        DEVICE = ego_trajs.device
        DTYPE = ego_trajs.dtype

        if n_agents == 0:
            return torch.zeros(B, M, n_timesteps, dtype=DTYPE, device=DEVICE)

        others_gt_trajs_mask = others_gt_trajs_mask.to(dtype=torch.bool, device=DEVICE)
        ego_gt_trajs_mask = ego_gt_trajs_mask.to(dtype=torch.bool, device=DEVICE)

        others_gt_centers = others_gt_trajs[:, :, :, :2]  # (B, n_agents, n_timesteps, 2)
        others_gt_heads = others_gt_trajs[:, :, :, 2]  # (B, n_agents, n_timesteps)

        ego_pred_centers = ego_trajs[:, :, :, :2]  # (B, M, n_timesteps, 2)
        ego_pred_heads = ego_trajs[:, :, :, 2]  # (B, M, n_timesteps)

        ego_pred_overlap = self._pairwise_ego_vs_others(
            ego_pred_centers, ego_pred_heads, ego_wh_trajs,
            others_gt_centers, others_gt_heads, others_wh_trajs,
        )  # (B, M, n_agents, n_timesteps)
        ego_pred_overlap = (
            ego_pred_overlap & ego_gt_trajs_mask[:, None, None, :] & others_gt_trajs_mask[:, None, :, :]
        )  # (B, M, n_agents, n_timesteps)

        ego_gt_centers = ego_gt_trajs[:, :, :2]  # (B, n_timesteps, 2)
        ego_gt_heads = ego_gt_trajs[:, :, 2]  # (B, n_timesteps)
        ego_gt_overlap = self._pairwise_ego_vs_others(
            ego_gt_centers[:, None, :, :], ego_gt_heads[:, None, :], ego_wh_trajs,
            others_gt_centers, others_gt_heads, others_wh_trajs,
        )  # (B, 1, n_agents, n_timesteps)
        ego_gt_overlap = (
            ego_gt_overlap & ego_gt_trajs_mask[:, None, None, :] & others_gt_trajs_mask[:, None, :, :]
        )  # (B, 1, n_agents, n_timesteps)

        overlap = ego_pred_overlap & (~ego_gt_overlap)  # (B, M, n_agents, n_timesteps)
        box_collision_any = overlap.any(dim=2).to(dtype=DTYPE, device=DEVICE)  # (B, M, n_timesteps)

        return box_collision_any  # (B, M, n_timesteps)

    def update(
        self, ego_trajs: Tensor, ego_mode_probs: Tensor, ego_gt_trajs: Tensor,
        ego_wh_trajs: Tensor, ego_gt_trajs_mask: Tensor, others_gt_trajs: Tensor,
        others_wh_trajs: Tensor, others_gt_trajs_mask: Tensor,
    ) -> None:
        """
        ego_trajs: (B, M, n_timesteps, 3)
        ego_mode_probs: (B, M)
        ego_gt_trajs: (B, n_timesteps, 3)
        ego_wh_trajs: (B, n_timesteps, 2)
        ego_gt_trajs_mask: (B, n_timesteps)
        others_gt_trajs: (B, n_agents, n_timesteps, 3)
        others_wh_trajs: (B, n_agents, n_timesteps, 2)
        others_gt_trajs_mask: (B, n_agents, n_timesteps)
        """
        L2 = self.compute_L2(ego_trajs, ego_gt_trajs, ego_gt_trajs_mask)  # (B, M, n_timesteps)
        obj_box_coll = self.evaluate_coll(
            ego_trajs, ego_gt_trajs, ego_wh_trajs, ego_gt_trajs_mask,
            others_gt_trajs, others_wh_trajs, others_gt_trajs_mask,
        )  # (B, M, n_timesteps)

        self.L2_list.append(L2)
        self.obj_box_col_list.append(obj_box_coll)
        self.mode_probs_list.append(ego_mode_probs)

    def compute(self, n: int = 1) -> Dict:
        """
        For each batch, select top-n modes by probability, and compute min L2 and min box_col_percent over those modes.
        Returns per-timestep averages over all batches.
        n: number of top modes to consider (1 <= n <= M)
        """
        if not self.L2_list:
            return {"box_col_percent": [0.0] * self.n_timesteps, "L2": [0.0] * self.n_timesteps}

        L2_all = torch.cat(self.L2_list, dim=0)  # (N, M, n_timesteps)
        obj_box_col_all = torch.cat(self.obj_box_col_list, dim=0)  # (N, M, n_timesteps)
        mode_probs_all = torch.cat(self.mode_probs_list, dim=0)  # (N, M)
        N, M, T = L2_all.shape

        # For each batch, get indices of top-n modes
        topn_idx = torch.topk(mode_probs_all, n, dim=1, largest=True, sorted=True).indices  # (N, n)

        # Gather metrics for top-n modes
        batch_indices = torch.arange(N).unsqueeze(1).expand(N, n)  # (N, n)
        topn_L2 = L2_all[batch_indices, topn_idx]  # (N, n, n_timesteps)
        topn_obj_box_col = obj_box_col_all[batch_indices, topn_idx]  # (N, n, n_timesteps)

        # Min over modes for each batch
        min_L2 = topn_L2.min(dim=1).values  # (N, n_timesteps)
        min_obj_box_col = topn_obj_box_col.min(dim=1).values  # (N, n_timesteps)

        # Average over batches
        avg_L2 = min_L2.mean(dim=0)
        avg_obj_box_col = min_obj_box_col.mean(dim=0)

        return {
            "box_col_percent": torch.round(avg_obj_box_col * 100, decimals=3),  # (n_timesteps)
            "L2": torch.round(avg_L2, decimals=3),  # (n_timesteps)
        }

    def reset(self) -> None:
        del self.L2_list, self.obj_box_col_list, self.mode_probs_list
        self.L2_list = []
        self.obj_box_col_list = []
        self.mode_probs_list = []
