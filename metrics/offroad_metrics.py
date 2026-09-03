import torch
from torch import Tensor


def point_to_segment_distance(points: Tensor, seg_start: Tensor, seg_end: Tensor) -> Tensor:
    """
    points:    (P, 2)
    seg_start: (L, 2)
    seg_end:   (L, 2)
    returns:   (P, L) distance from each point to each lane node segment
    """
    seg_vec = seg_end - seg_start                                     # (L, 2)
    seg_len_sq = seg_vec.pow(2).sum(-1).clamp(min=1e-12)               # (L,)

    diff = points.unsqueeze(1) - seg_start.unsqueeze(0)                # (P, L, 2)
    t = (diff * seg_vec.unsqueeze(0)).sum(-1) / seg_len_sq             # (P, L)
    t = t.clamp(0.0, 1.0)

    closest = seg_start.unsqueeze(0) + t.unsqueeze(-1) * seg_vec.unsqueeze(0)  # (P, L, 2)
    return torch.norm(points.unsqueeze(1) - closest, dim=-1)           # (P, L)


class OffroadMetric:
    """
    Offroad rate with top-n mode selection by probability, mirroring
    PredMetric's minADE/minFDE convention: at update() time we compute a
    per-(agent, mode) offroad indicator and store mode probabilities. At
    compute(n) time we pick the n most likely modes per agent, then take the
    min offroad indicator over that top-n (0 = at least one likely mode
    stayed on-road, 1 = all top-n likely modes went offroad), and average
    over the dataset.
    """

    def __init__(self, half_width: float = 2.0) -> None:
        self.half_width = half_width
        self.offroad_list = []     # per-call (a, M) offroad indicator
        self.mode_prob_list = []   # per-call (a, M) mode probabilities

    @torch.no_grad()
    def update(
        self,
        preds: Tensor,             # (A, M, T, 2+)
        reg_mask: Tensor,          # (A, T)
        pi: Tensor,                # (A, M)
        lanes_start_locs: Tensor,  # (L, 2) all lane nodes across every scene in this batch
        lanes_end_locs: Tensor,    # (L, 2)
        agent_scene_idx: Tensor,   # (A,) scene id per agent
        lane_scene_idx: Tensor,    # (L,) scene id per lane node
    ) -> None:
        offroad_list_this_call = []
        prob_list_this_call = []

        for scene_id in torch.unique(agent_scene_idx):
            agent_sel = agent_scene_idx == scene_id
            lane_sel = lane_scene_idx == scene_id

            a_preds = preds[agent_sel]           # (a, M, T, 2+)
            a_mask = reg_mask[agent_sel]         # (a, T)
            a_pi = pi[agent_sel]                 # (a, M)
            a, M, T, _ = a_preds.shape

            seg_start = lanes_start_locs[lane_sel]  # (l, 2)
            seg_end = lanes_end_locs[lane_sel]      # (l, 2)

            if seg_start.shape[0] == 0:
                continue  # no lane graph for this scene -- skip rather than poison the metric

            points = a_preds[..., :2].reshape(a * M * T, 2)
            dist = point_to_segment_distance(points, seg_start, seg_end)  # (a*M*T, l)
            min_dist = dist.min(dim=-1).values.view(a, M, T)              # (a, M, T)

            point_offroad = (min_dist > self.half_width) & a_mask.bool().unsqueeze(1)  # (a, M, T)
            mode_offroad = point_offroad.any(dim=-1).float()               # (a, M)

            offroad_list_this_call.append(mode_offroad)
            prob_list_this_call.append(a_pi)

        if offroad_list_this_call:
            self.offroad_list.append(torch.cat(offroad_list_this_call, dim=0))
            self.mode_prob_list.append(torch.cat(prob_list_this_call, dim=0))

    def compute(self, n: int = 1) -> Tensor:
        offroad_all = torch.cat(self.offroad_list, dim=0)        # (N, M)
        mode_prob_all = torch.cat(self.mode_prob_list, dim=0)    # (N, M)

        topn_idx = torch.topk(mode_prob_all, n, dim=1, largest=True, sorted=True).indices  # (N, n)

        N = mode_prob_all.shape[0]
        batch_indices = torch.arange(N).unsqueeze(1).expand(N, n)  # (N, n)
        topn_offroad = offroad_all[batch_indices, topn_idx]         # (N, n)

        # aggregrate the offroad for the top-n modes
        avg_topn_offroad = topn_offroad.mean()

        return avg_topn_offroad

    def reset(self) -> None:
        del self.offroad_list, self.mode_prob_list
        self.offroad_list = []
        self.mode_prob_list = []