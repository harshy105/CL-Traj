import torch

from torch import Tensor
from typing import Dict, Optional, Tuple
from utilities.transformation import get_bb_corners


class PlanningMetric:
    """
    OBB-only collision + TTC-violation metric with per-agent per-timestep width/length support.
    Supports multi-modal predictions.

    TTC-violation follows the navsim-style single-shot constant-velocity/heading projection:
    at each timestep, ego and other agents are projected forward by `ttc_horizon` seconds using
    their current (finite-differenced) velocity, and a violation is flagged if the projected
    boxes overlap. This is cheaper than nuPlan's 0.1s-substep earliest-intersection search, at
    the cost of only checking the single projected instant rather than the full swept path.
    """

    def __init__(
        self,
        n_timesteps: int = 6,
        dt: float = 0.5,
        ttc_horizon: float = 1.0,
        ttc_box_shrink: float = 0.2,
        ttc_lateral_threshold: float = 2.0,
        ttc_rear_margin: float = 1.0,
        at_fault_lateral_threshold: float = 2.0,
        at_fault_rear_margin: float = 5.0,
    ):
        """
        dt: seconds per timestep, used to finite-difference velocity and to interpret ttc_horizon.
        ttc_horizon: forward projection window in seconds (navsim default: 1.0s).
        ttc_box_shrink: meters subtracted from each box dimension before the TTC overlap check
            (does not affect the main collision metric). Tune this if TTC-violation looks too
            trigger-happy or too lenient.
        ttc_lateral_threshold: meters; an agent within this lateral offset of ego (in ego's local
            frame) is considered "adjacent" regardless of whether it's ahead or behind.
        ttc_rear_margin: meters; an "adjacent" agent must not be more than this far behind ego
            (prevents flagging agents that are clearly behind and to the side as relevant).
        These two thresholds are a geometric proxy for nuPlan's map-based front/cross-traffic/
        lateral filter, since this schema has no lane/map info. Treat as tunable, not literature-sourced.
        """
        super().__init__()
        self.n_timesteps = n_timesteps
        self.dt = dt
        self.ttc_horizon = ttc_horizon
        self.ttc_box_shrink = ttc_box_shrink
        self.ttc_lateral_threshold = ttc_lateral_threshold
        self.ttc_rear_margin = ttc_rear_margin
        self.at_fault_lateral_threshold = at_fault_lateral_threshold
        self.at_fault_rear_margin = at_fault_rear_margin

        self.L2_list = []  # List of (B, M, n_timesteps) per batch
        self.obj_box_col_list = []  # List of (B, M, n_timesteps) per batch
        self.ttc_violation_list = []  # List of (B, M, n_timesteps) per batch
        self.mode_probs_list = []  # List of (B, M) per batch
        self.at_fault_col_list = []

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
        others_centers: (B, n_agents, M, n_timesteps, 2)
            alongside that plan.
        others_heads: (B, n_agents, M, n_timesteps)
        others_sizes: (B, n_agents, n_timesteps, 2)
        returns: (B, M, n_agents, n_timesteps) bool collisions between ego box and each other box
        """
        B, n_agents, M, n_timesteps, _ = others_centers.shape

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

        oc = others_centers.permute(0, 2, 1, 3, 4)  # (B, M, n_agents, T, 2)
        oh = others_heads.permute(0, 2, 1, 3)  # (B, M, n_agents, T)
        osz = others_sizes.unsqueeze(1).repeat(1, M, 1, 1, 1)  # (B, M, n_agents, T, 2) -- broadcast

        c2 = get_bb_corners(
            oc.reshape(B, M * n_agents * n_timesteps, 2),
            oh.reshape(B, M * n_agents * n_timesteps),
            osz.reshape(B, M * n_agents * n_timesteps, 2),
        )  # (B, M*n_agents*n_timesteps, 4, 2)
        c2 = c2.reshape(B * M * n_agents * n_timesteps, 4, 2)

        overlap_flat = self._obb_overlap_pairs(c1, c2)
        return overlap_flat.reshape(B, M, n_agents, n_timesteps)

    def _finite_diff_velocity(self, centers: Tensor, dt: float) -> Tensor:
        """
        centers: (..., n_timesteps, 2)
        returns velocity of same shape via forward difference; the last timestep repeats the
        second-to-last velocity (no t+1 available to difference against there).
        """
        if centers.shape[-2] < 2:
            return torch.zeros_like(centers)
        vel = torch.zeros_like(centers)
        vel[..., :-1, :] = (centers[..., 1:, :] - centers[..., :-1, :]) / dt
        vel[..., -1, :] = vel[..., -2, :]
        return vel

    def _relevant_mask(
        self, ego_centers: Tensor, ego_heads: Tensor, other_centers: Tensor,
        lateral_threshold: Optional[float] = None, rear_margin: Optional[float] = None,
    ) -> Tensor:
        """
        ego_centers: (B, M, n_timesteps, 2)
        ego_heads: (B, M, n_timesteps)
        other_centers: (B, n_agents, M, n_timesteps, 2)
        returns: (B, M, n_agents, n_timesteps) bool - True if the other agent is in front of ego
        (in ego's own heading frame) or laterally adjacent to it. Geometric proxy for nuPlan's
        map-based front/cross-traffic/lateral filter.
        lateral_threshold / rear_margin: override self.ttc_lateral_threshold / self.ttc_rear_margin.
            Lets this geometry be reused for a different purpose (at-fault attribution) with its
            own tuned thresholds. Defaults to the TTC thresholds when not given.
        """
        lateral_threshold = self.ttc_lateral_threshold if lateral_threshold is None else lateral_threshold
        rear_margin = self.ttc_rear_margin if rear_margin is None else rear_margin

        other_centers_m = other_centers.permute(0, 2, 1, 3, 4)  # (B, M, n_agents, T, 2)
        delta = other_centers_m - ego_centers.unsqueeze(2)  # (B, M, n_agents, T, 2)
        cos_h = torch.cos(ego_heads).unsqueeze(2)  # (B, M, 1, T)
        sin_h = torch.sin(ego_heads).unsqueeze(2)  # (B, M, 1, T)

        local_fwd = delta[..., 0] * cos_h + delta[..., 1] * sin_h  # (B, M, n_agents, T)
        local_lat = -delta[..., 0] * sin_h + delta[..., 1] * cos_h  # (B, M, n_agents, T)

        in_front = local_fwd > 0
        adjacent = (local_lat.abs() <= lateral_threshold) & (local_fwd > -rear_margin)
        return in_front | adjacent

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
        others_trajs: Optional[Tensor] = None,  # ADDED
    ) -> Tuple[Tensor]:
        """
        ego_trajs: (B, M, n_timesteps, 3)
        ego_gt_trajs: (B, n_timesteps, 3)
        ego_wh_trajs: (B, n_timesteps, 2)
        ego_gt_trajs_mask: (B, n_timesteps)
        others_gt_trajs: (B, n_agents, n_timesteps, 3)
        others_wh_trajs: (B, n_agents, n_timesteps, 2)
        others_gt_trajs_mask: (B, n_agents, n_timesteps)
        others_trajs: (B, n_agents, M, n_timesteps, 3), OPTIONAL
        """
        B, M, n_timesteps, _ = ego_trajs.shape
        n_agents = others_gt_trajs.shape[1] if (others_gt_trajs.numel() > 0) else 0
        DEVICE = ego_trajs.device
        DTYPE = ego_trajs.dtype

        if n_agents == 0:
            zero_overlap = torch.zeros(B, M, n_timesteps, dtype=DTYPE, device=DEVICE)
            return (zero_overlap, zero_overlap)

        others_gt_trajs_mask = others_gt_trajs_mask.to(dtype=torch.bool, device=DEVICE)
        ego_gt_trajs_mask = ego_gt_trajs_mask.to(dtype=torch.bool, device=DEVICE)

        others_gt_centers = others_gt_trajs[:, :, :, :2]  # (B, n_agents, n_timesteps, 2)
        others_gt_heads = others_gt_trajs[:, :, :, 2]  # (B, n_agents, n_timesteps)

        if others_trajs is None:
            # Log-replay fallback: no per-mode simulation happened for surrounding agents,
            # so every mode sees an identical (GT) trajectory for them 
            others_pred_centers = others_gt_centers.unsqueeze(2).expand(-1, -1, M, -1, -1)  # (B, n_agents, M, T, 2)
            others_pred_heads = others_gt_heads.unsqueeze(2).expand(-1, -1, M, -1)  # (B, n_agents, M, T)
        else:
            others_pred_centers = others_trajs[:, :, :, :, :2]  # (B, n_agents, M, n_timesteps, 2)
            others_pred_heads = others_trajs[:, :, :, :, 2]  # (B, n_agents, M, n_timesteps)

        ego_pred_centers = ego_trajs[:, :, :, :2]  # (B, M, n_timesteps, 2)
        ego_pred_heads = ego_trajs[:, :, :, 2]  # (B, M, n_timesteps)

        ego_pred_overlap = self._pairwise_ego_vs_others(
            ego_pred_centers, ego_pred_heads, ego_wh_trajs,
            others_pred_centers, others_pred_heads, others_wh_trajs,
        )  # (B, M, n_agents, n_timesteps)
        ego_pred_overlap = (
            ego_pred_overlap & ego_gt_trajs_mask[:, None, None, :] & others_gt_trajs_mask[:, None, :, :]
        )  # (B, M, n_agents, n_timesteps)

        ego_gt_centers = ego_gt_trajs[:, :, :2]  # (B, n_timesteps, 2)
        ego_gt_heads = ego_gt_trajs[:, :, 2]  # (B, n_timesteps)
        ego_gt_overlap = self._pairwise_ego_vs_others(
            ego_gt_centers[:, None, :, :], ego_gt_heads[:, None, :], ego_wh_trajs,
            others_gt_centers.unsqueeze(2), others_gt_heads.unsqueeze(2), others_wh_trajs,
        )  # (B, 1, n_agents, n_timesteps) -- GT side is single-"mode" (M=1) on both sides
        ego_gt_overlap = (
            ego_gt_overlap & ego_gt_trajs_mask[:, None, None, :] & others_gt_trajs_mask[:, None, :, :]
        )  # (B, 1, n_agents, n_timesteps)

        overlap = ego_pred_overlap & (~ego_gt_overlap)  # (B, M, n_agents, n_timesteps)
        box_collision_any = overlap.any(dim=2).to(dtype=DTYPE, device=DEVICE)  # (B, M, n_timesteps)

        # --- at-fault attribution ---
        relevant_for_fault = self._relevant_mask(
            ego_pred_centers, ego_pred_heads, others_pred_centers,
            lateral_threshold=self.at_fault_lateral_threshold,
            rear_margin=self.at_fault_rear_margin,
        )  # (B, M, n_agents, n_timesteps)
        at_fault_overlap = overlap & relevant_for_fault
        at_fault_collision_any = at_fault_overlap.any(dim=2).to(dtype=DTYPE, device=DEVICE) # (B, M, n_timesteps)

        return box_collision_any, at_fault_collision_any

    def evaluate_ttc(
        self, ego_trajs: Tensor, ego_gt_trajs: Tensor, ego_wh_trajs: Tensor,
        ego_gt_trajs_mask: Tensor, others_gt_trajs: Tensor,
        others_wh_trajs: Tensor, others_gt_trajs_mask: Tensor,
        others_trajs: Optional[Tensor] = None, 
    ) -> Tensor:
        """
        Same signature/shapes as evaluate_coll (plus the same `others_trajs`
        argument). Returns (B, M, n_timesteps) bool: True where the predicted ego
        trajectory has a TTC-violation (projected constant-velocity/heading
        overlap with a front/adjacent other agent within `self.ttc_horizon`) that
        is NOT already present when projecting the GT ego trajectory forward the
        same way -- mirrors evaluate_coll's GT-subtraction convention.

        """
        B, M, n_timesteps, _ = ego_trajs.shape
        n_agents = others_gt_trajs.shape[1] if (others_gt_trajs.numel() > 0) else 0
        DEVICE = ego_trajs.device
        DTYPE = ego_trajs.dtype

        if n_agents == 0:
            return torch.zeros(B, M, n_timesteps, dtype=DTYPE, device=DEVICE)

        others_gt_trajs_mask = others_gt_trajs_mask.to(dtype=torch.bool, device=DEVICE)
        ego_gt_trajs_mask = ego_gt_trajs_mask.to(dtype=torch.bool, device=DEVICE)

        others_gt_centers = others_gt_trajs[:, :, :, :2]  # (B, n_agents, T, 2)
        others_gt_heads = others_gt_trajs[:, :, :, 2]  # (B, n_agents, T)
        others_gt_vel = self._finite_diff_velocity(others_gt_centers, self.dt)
        others_gt_proj_centers = others_gt_centers + others_gt_vel * self.ttc_horizon
        others_shrunk_wh = (others_wh_trajs - self.ttc_box_shrink).clamp_min(0.1)  # (B, n_agents, T, 2)

        if others_trajs is None:
            others_pred_centers = others_gt_centers.unsqueeze(2).expand(-1, -1, M, -1, -1)  # (B, n_agents, M, T, 2)
            others_pred_heads = others_gt_heads.unsqueeze(2).expand(-1, -1, M, -1)  # (B, n_agents, M, T)
        else:
            others_pred_centers = others_trajs[:, :, :, :, :2]  # (B, n_agents, M, T, 2)
            others_pred_heads = others_trajs[:, :, :, :, 2]  # (B, n_agents, M, T)
        others_pred_vel = self._finite_diff_velocity(others_pred_centers, self.dt)  # (B, n_agents, M, T, 2)
        others_pred_proj_centers = others_pred_centers + others_pred_vel * self.ttc_horizon  # (B, n_agents, M, T, 2)

        # --- prediction side ---
        ego_pred_centers = ego_trajs[:, :, :, :2]  # (B, M, T, 2)
        ego_pred_heads = ego_trajs[:, :, :, 2]  # (B, M, T)
        ego_pred_vel = self._finite_diff_velocity(ego_pred_centers, self.dt)
        ego_pred_proj_centers = ego_pred_centers + ego_pred_vel * self.ttc_horizon
        ego_shrunk_wh = (ego_wh_trajs - self.ttc_box_shrink).clamp_min(0.1)  # (B, T, 2)

        pred_overlap = self._pairwise_ego_vs_others(
            ego_pred_proj_centers, ego_pred_heads, ego_shrunk_wh,
            others_pred_proj_centers, others_pred_heads, others_shrunk_wh,
        )  # (B, M, n_agents, T)
        pred_relevant = self._relevant_mask(ego_pred_centers, ego_pred_heads, others_pred_centers)  # (B, M, n_agents, T)

        # Current (unprojected, unshrunk) overlap -- same boxes/positions evaluate_coll uses.
        # Gates out agents ego is already colliding with right now, so TTC only fires for
        # agents that are currently clear but would be hit after the ttc_horizon projection.
        pred_current_overlap = self._pairwise_ego_vs_others(
            ego_pred_centers, ego_pred_heads, ego_wh_trajs,
            others_pred_centers, others_pred_heads, others_wh_trajs,
        )  # (B, M, n_agents, T)

        pred_violation = (
            pred_overlap
            & pred_relevant
            & (~pred_current_overlap)
            & ego_gt_trajs_mask[:, None, None, :]
            & others_gt_trajs_mask[:, None, :, :]
        )  # (B, M, n_agents, T)

        # --- GT side (to exclude TTC violations already inherent to the scene) ---
        ego_gt_centers = ego_gt_trajs[:, :, :2]  # (B, T, 2)
        ego_gt_heads = ego_gt_trajs[:, :, 2]  # (B, T)
        ego_gt_vel = self._finite_diff_velocity(ego_gt_centers, self.dt)
        ego_gt_proj_centers = (ego_gt_centers + ego_gt_vel * self.ttc_horizon)  # (B, T, 2)

        gt_overlap = self._pairwise_ego_vs_others(
            ego_gt_proj_centers[:, None, :, :], ego_gt_heads[:, None, :], ego_shrunk_wh,
            others_gt_proj_centers.unsqueeze(2), others_gt_heads.unsqueeze(2), others_shrunk_wh,
        )  # (B, 1, n_agents, T) -- GT side is single-"mode" (M=1) on both sides
        gt_relevant = self._relevant_mask(
            ego_gt_centers[:, None, :, :], ego_gt_heads[:, None, :], others_gt_centers.unsqueeze(2)
        )  # (B, 1, n_agents, T)

        # Same current-collision gate, applied to the GT-side projection.
        gt_current_overlap = self._pairwise_ego_vs_others(
            ego_gt_centers[:, None, :, :], ego_gt_heads[:, None, :], ego_wh_trajs,
            others_gt_centers.unsqueeze(2), others_gt_heads.unsqueeze(2), others_wh_trajs,
        )  # (B, 1, n_agents, T)

        gt_violation = (
            gt_overlap
            & gt_relevant
            & (~gt_current_overlap)
            & ego_gt_trajs_mask[:, None, None, :]
            & others_gt_trajs_mask[:, None, :, :]
        )  # (B, 1, n_agents, T)

        violation = pred_violation & (~gt_violation)  # (B, M, n_agents, T), broadcasts over M
        ttc_violation_any = violation.any(dim=2).to(dtype=DTYPE, device=DEVICE)  # (B, M, n_timesteps)

        return ttc_violation_any

    def update(
        self, ego_trajs: Tensor, ego_mode_probs: Tensor, ego_gt_trajs: Tensor,
        ego_wh_trajs: Tensor, ego_gt_trajs_mask: Tensor, others_gt_trajs: Tensor,
        others_wh_trajs: Tensor, others_gt_trajs_mask: Tensor,
        others_trajs: Optional[Tensor] = None,  # ADDED
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
        others_trajs: (B, n_agents, M, n_timesteps, 3), OPTIONAL 
        """
        L2 = self.compute_L2(ego_trajs, ego_gt_trajs, ego_gt_trajs_mask)  # (B, M, n_timesteps)
        obj_box_coll, at_fault_coll = self.evaluate_coll(
            ego_trajs, ego_gt_trajs, ego_wh_trajs, ego_gt_trajs_mask,
            others_gt_trajs, others_wh_trajs, others_gt_trajs_mask,
            others_trajs=others_trajs,
        )  # (B, M, n_timesteps), (B, M, n_timesteps)
        ttc_violation = self.evaluate_ttc(
            ego_trajs, ego_gt_trajs, ego_wh_trajs, ego_gt_trajs_mask,
            others_gt_trajs, others_wh_trajs, others_gt_trajs_mask,
            others_trajs=others_trajs,  # ADDED
        )  # (B, M, n_timesteps)

        self.L2_list.append(L2)
        self.obj_box_col_list.append(obj_box_coll)
        self.ttc_violation_list.append(ttc_violation)
        self.mode_probs_list.append(ego_mode_probs)
        self.at_fault_col_list.append(at_fault_coll)

    def compute(self, n: int = 1) -> Dict:
        """
        For each batch, select top-n modes by probability.
        L2 uses min-over-modes (minADE-style, credits the best proposal).
        Collision and TTC-violation use mean-over-modes (safety metrics: an unweighted average
        across the top-n considered modes, rather than crediting only the single safest one).
        Returns per-timestep averages over all batches.
        n: number of top modes to consider (1 <= n <= M)
        """
        if not self.L2_list:
            return {
                "mean_box_col_percent": [0.0] * self.n_timesteps,
                "mean_ttc_violation_percent": [0.0] * self.n_timesteps,
                "min_L2": [0.0] * self.n_timesteps,
                "mean_at_fault_col_percent": [0.0] * self.n_timesteps,
                "mean_at_fault_share": [0.0] * self.n_timesteps,
            }

        L2_all = torch.cat(self.L2_list, dim=0)  # (N, M, n_timesteps)
        obj_box_col_all = torch.cat(self.obj_box_col_list, dim=0)  # (N, M, n_timesteps)
        ttc_violation_all = torch.cat(self.ttc_violation_list, dim=0)  # (N, M, n_timesteps)
        mode_probs_all = torch.cat(self.mode_probs_list, dim=0)  # (N, M)
        at_fault_col_all = torch.cat(self.at_fault_col_list, dim=0)  # (N, M, n_timesteps)
        N, M, T = L2_all.shape

        # For each batch, get indices of top-n modes
        topn_idx = torch.topk(mode_probs_all, n, dim=1, largest=True, sorted=True).indices  # (N, n)

        # Gather metrics for top-n modes
        batch_indices = torch.arange(N).unsqueeze(1).expand(N, n)  # (N, n)
        topn_L2 = L2_all[batch_indices, topn_idx]  # (N, n, n_timesteps)
        topn_obj_box_col = obj_box_col_all[batch_indices, topn_idx]  # (N, n, n_timesteps)
        topn_ttc_violation = ttc_violation_all[batch_indices, topn_idx]  # (N, n, n_timesteps)
        topn_at_fault_col = at_fault_col_all[batch_indices, topn_idx] # (N, n, n_timesteps)

        min_L2 = topn_L2.min(dim=1).values  # (N, n_timesteps)
        mean_obj_box_col = topn_obj_box_col.mean(dim=1)  # (N, n_timesteps)
        mean_ttc_violation = topn_ttc_violation.mean(dim=1)  # (N, n_timesteps)
        mean_at_fault_col = topn_at_fault_col.mean(dim=1) # (N, n_timesteps)

        # Average over batches
        avg_L2 = min_L2.mean(dim=0)
        avg_obj_box_col = mean_obj_box_col.mean(dim=0)
        avg_ttc_violation = mean_ttc_violation.mean(dim=0)
        avg_at_fault_col = mean_at_fault_col.mean(dim=0)

        return {
            "mean_box_col_percent": torch.round(avg_obj_box_col * 100, decimals=3),  # (n_timesteps)
            "mean_ttc_violation_percent": torch.round(avg_ttc_violation * 100, decimals=3),  # (n_timesteps)
            "min_L2": torch.round(avg_L2, decimals=3),  # (n_timesteps)
            "mean_at_fault_col_percent": torch.round(avg_at_fault_col * 100, decimals=3),
            "mean_at_fault_share": torch.round(
                (avg_at_fault_col / avg_obj_box_col.clamp_min(1e-6)) * 100, decimals=3
            ),  # % of total collisions that were at-fault
        }

    def reset(self) -> None:
        del self.L2_list, self.obj_box_col_list, self.ttc_violation_list, self.mode_probs_list, self.at_fault_col_list
        self.L2_list = []
        self.obj_box_col_list = []
        self.ttc_violation_list = []
        self.mode_probs_list = []
        self.at_fault_col_list = []