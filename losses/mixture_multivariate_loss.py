import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

from losses.multivariate_loss import MultiVariateLoss

class MixtureMultivariateLoss(nn.Module):
    def __init__(self,
        eps: float = 1e-6, 
        distance_coeff: float = 1.0,
        distance_decay: str = 'l1',       
    ) -> None:
        super().__init__()
        self.nll_loss = MultiVariateLoss(
            eps=eps,
            distance_coeff=distance_coeff,
            distance_decay=distance_decay,
        )
        
    def forward(self, pred: Tensor, scale: Tensor, target: Tensor,
                prob: Tensor, reg_mask: Tensor) -> Tensor:
        '''
        Input:
        pred: shape(A, M, T, 2)
        scale: shape(A, M, T, 2, 2)
        target: shape(A, T, 2)
        prob: shape(A, M)
        reg_mask: shape(A, T)
        Output:
        loss: shape(A,)
        '''
        assert pred.requires_grad == False, 'Classification Loss should not improve trajectory profile'
        assert scale.requires_grad == False, 'Classification Loss should not improve trajectory profile'
        M = pred.shape[1]
        nll = self.nll_loss(
            pred=pred,
            scale=scale,
            target=target.unsqueeze(1).repeat(1, M, 1, 1),
        ) * reg_mask.unsqueeze(1).repeat(1, M, 1) # (A, M, T)
        nll = nll.sum(dim=-1) # (A, M)
        
        log_pi = F.log_softmax(prob, dim=-1) # (A, M)
        loss = -torch.logsumexp(log_pi - nll, dim=-1) # (A,)
        return loss