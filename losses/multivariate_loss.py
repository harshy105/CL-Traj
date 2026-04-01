import torch
import torch.nn as nn

from torch import Tensor

class MultiVariateLoss(nn.Module):
    def __init__(self, 
            eps: float = 1e-6, 
            distance_coeff: float = 1.0,
            distance_decay: str = 'l1'
            ) -> None:
        super().__init__()
        self.eps = eps
        self.distance_coeff = distance_coeff
        self.distance_decay = distance_decay

    def forward(self, pred: Tensor, scale: Tensor, target: Tensor) -> Tensor:
        '''
        Input:
        pred: shape(..., 2)
        scale: shape(..., 2, 2)
        target: shape(..., 2)
        Output:
        nll: shape(...)
        '''
        D = 2
        assert pred.shape[-1] == D, 'Not tested for D=3'
        assert scale.shape[-2:] == (D, D)
        assert pred.shape[-1] == D

        scale_stable = scale + self.eps * torch.eye(D, device=scale.device)
        lower_scale_stable = torch.linalg.cholesky(scale_stable)

        log_det_scale_stable = lower_scale_stable.diagonal(dim1=-2, dim2=-1).log().sum(-1) # (...)

        diff_unsqueezed = (target - pred).unsqueeze(-1) # (..., 2, 1)
        precision_times_diff = torch.cholesky_solve(diff_unsqueezed, lower_scale_stable) # (..., 2, 1)
        quadratic_form = diff_unsqueezed.transpose(-2, -1) @ precision_times_diff # (..., 1, 1)
        if self.distance_decay == 'l1': # similar to Laplace
            distance = torch.sqrt(quadratic_form.squeeze(-1).squeeze(-1) + self.eps) # (...)
        elif self.distance_decay == 'l2': # similar to gaussian
            distance = quadratic_form.squeeze(-1).squeeze(-1)
        else:
            raise ValueError('Distance Decay is not defined correctly')

        nll = log_det_scale_stable + self.distance_coeff*distance # (...)

        return nll