import torch

from torch import Tensor
from typing import Dict


def bipartite_dense_to_sparse(adj: Tensor) -> Tensor:
    index = adj.nonzero(as_tuple=True)
    if len(index) == 3:
        batch_src = index[0] * adj.size(1)
        batch_dst = index[0] * adj.size(2)
        index = (batch_src + index[1], batch_dst + index[2])
    return torch.stack(index, dim=0)


def append_one_for_single_batch(data: Dict) -> Dict:
    for k, v in data.copy().items():
        data[k] = v.unsqueeze(0)

    return data


def combine_batch_elements_dim(data: Dict) -> Dict:
    # combine batch and the elements dim
    for k, v in data.copy().items():
        batch = torch.arange(v.shape[0], device=v.device, dtype=torch.int32).unsqueeze(-1).expand(-1, v.shape[1])
        v = v.reshape(-1, *v.shape[2:])
        batch = batch.reshape(-1)
        mask = ~torch.isinf(v).view(v.shape[0], -1).any(dim=1)
        data[k] = v[mask]
        data[k.split("_")[0] + "_batch"] = batch[mask]

    return data


def unbatch_and_pad(data: Dict) -> Dict:
    output_dict = {}
    elements_type = []
    DEVICE = data[next(iter(data))].device
    for k in data.keys():
        if "_batch" in k:
            elements_type.append(k.split("_")[0])
    for et in elements_type:
        batch_indices = data[et + "_batch"]
        num_elements = next((v for k, v in data.items() if et in k)).shape[0]

        num_batches = batch_indices.max().item() + 1
        elements_per_batch = torch.bincount(batch_indices)
        max_num_elements = elements_per_batch.max().item()

        arange_total = torch.arange(num_elements, device=DEVICE)
        batch_starts = torch.cat(
            (torch.tensor([0], dtype=torch.long, device=DEVICE), elements_per_batch.cumsum(0)[:-1])
        )
        element_indices = arange_total - torch.gather(batch_starts, 0, batch_indices.long())

        for k, v in data.copy().items():
            if et in k:
                feature_dims = v.shape[1:]
                output_shape = (num_batches, max_num_elements) + feature_dims
                padded_tensor = torch.zeros(
                    output_shape, dtype=v.dtype, device=DEVICE
                )  # keep zeros, also works for masks
                padded_tensor[batch_indices, element_indices] = v
                output_dict[k] = padded_tensor

        output_dict[et + "_pad_mask"] = torch.arange(max_num_elements, device=DEVICE) < elements_per_batch.unsqueeze(1)

    return output_dict


def change_device_of_dict(data: Dict, device: torch.device) -> Dict:
    for k in data.keys():
        data[k] = data[k].to(device)
    return data
