import torch

from torch import Tensor
from typing import Tuple

class PredMetric:
    def __init__(self) -> None:
        self.ade_list = []
        self.fde_list = []
        self.mode_prob_list = []
    
    def update(self, preds: Tensor, gt: Tensor, reg_mask: Tensor,
                pi: Tensor) -> None:
        de = torch.norm(preds[..., :2] - gt.unsqueeze(1), dim=-1) # (A, M, T)
        ade_masked = (de * reg_mask[:, None, :]).sum(dim=-1) / reg_mask.sum(dim=-1).clamp(min=1)[:, None]  # (A, M)
        fde_masked = de[..., -1] * reg_mask[:, None, -1]  # (A, M)
        
        self.ade_list.append(ade_masked)
        self.fde_list.append(fde_masked)
        self.mode_prob_list.append(pi)
            
    def compute(self, n: int = 1) -> Tuple[Tensor, Tensor]:
        ade_all = torch.cat(self.ade_list, dim=0) # (N, M)
        fde_all = torch.cat(self.fde_list, dim=0) # (N, M)
        mode_prob_all = torch.cat(self.mode_prob_list, dim=0) # (N)
        
        # For each batch, get indices of top-n modes
        topn_idx = torch.topk(mode_prob_all, n, dim=1, largest=True, sorted=True).indices  # (N, n)
        
        # Gather metrics for top-n modes
        N = mode_prob_all.shape[0]
        batch_indices = torch.arange(N).unsqueeze(1).expand(N, n)  # (N, n)
        topn_ade_all = ade_all[batch_indices, topn_idx]  # (N, n)
        topn_fde_all = fde_all[batch_indices, topn_idx]  # (N, n)
        
        # Min over modes for each batch
        min_ade_all = topn_ade_all.min(dim=1).values  # (N)
        min_fde_all = topn_fde_all.min(dim=1).values  # (N)
        
        # Average over batches
        avg_min_ade_all = min_ade_all.mean(dim=0)
        avg_min_fde_all = min_fde_all.mean(dim=0)
        
        return avg_min_ade_all, avg_min_fde_all

    def reset(self) -> None:
        del self.ade_list, self.fde_list, self.mode_prob_list
        self.ade_list = []
        self.fde_list = []
        self.mode_prob_list = []