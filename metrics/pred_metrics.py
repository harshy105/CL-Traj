import torch

from torch import Tensor
from typing import Tuple

class PredMetric:
    def __init__(self) -> None:
        self.ade_list = []
        self.fde_list = []
        self.miss_dist_list = []
        self.mode_prob_list = []
    
    def update(self, preds: Tensor, gt: Tensor, reg_mask: Tensor,
                pi: Tensor) -> None:
        de = torch.norm(preds[..., :2] - gt.unsqueeze(1), dim=-1) # (A, M, T)
        ade_masked = (de * reg_mask[:, None, :]).sum(dim=-1) / reg_mask.sum(dim=-1).clamp(min=1)[:, None]  # (A, M)
        fde_masked = de[..., -1] * reg_mask[:, None, -1]  # (A, M)

        # Miss-distance per (agent, mode): max pointwise L2 over valid timesteps
        # (nuScenes convention). Invalid timesteps are masked to -inf so they
        # never dominate the max.
        mask_bool = reg_mask[:, None, :].bool()  # (A, 1, T), broadcasts against (A, M, T)
        de_for_max = de.masked_fill(~mask_bool, float('-inf'))
        miss_dist_masked = de_for_max.max(dim=-1).values  # (A, M)

        self.ade_list.append(ade_masked)
        self.fde_list.append(fde_masked)
        self.miss_dist_list.append(miss_dist_masked)
        self.mode_prob_list.append(pi)
            
    def compute(self, n: int = 1, tau: float = 2.0) -> Tuple[Tensor, Tensor, Tensor]:
        ade_all = torch.cat(self.ade_list, dim=0) # (N, M)
        fde_all = torch.cat(self.fde_list, dim=0) # (N, M)
        miss_dist_all = torch.cat(self.miss_dist_list, dim=0) # (N, M)
        mode_prob_all = torch.cat(self.mode_prob_list, dim=0) # (N)
        
        # For each batch, get indices of top-n modes
        topn_idx = torch.topk(mode_prob_all, n, dim=1, largest=True, sorted=True).indices  # (N, n)
        
        # Gather metrics for top-n modes
        N = mode_prob_all.shape[0]
        batch_indices = torch.arange(N).unsqueeze(1).expand(N, n)  # (N, n)
        topn_ade_all = ade_all[batch_indices, topn_idx]  # (N, n)
        topn_fde_all = fde_all[batch_indices, topn_idx]  # (N, n)
        topn_miss_dist_all = miss_dist_all[batch_indices, topn_idx]  # (N, n)
        
        # Min over modes for each batch
        min_ade_all = topn_ade_all.min(dim=1).values  # (N)
        min_fde_all = topn_fde_all.min(dim=1).values  # (N)
        min_miss_dist_all = topn_miss_dist_all.min(dim=1).values  # (N)

        # A scene is a miss if even the best top-n mode exceeds the threshold
        mr_all = (min_miss_dist_all > tau).float()  # (N)
        
        # Average over batches
        avg_min_ade_all = min_ade_all.mean(dim=0)
        avg_min_fde_all = min_fde_all.mean(dim=0)
        avg_mr_all = mr_all.mean(dim=0)
        
        return avg_min_ade_all, avg_min_fde_all, avg_mr_all

    def reset(self) -> None:
        del self.ade_list, self.fde_list, self.miss_dist_list, self.mode_prob_list
        self.ade_list = []
        self.fde_list = []
        self.miss_dist_list = []
        self.mode_prob_list = []