import torch
from torch import Tensor
from typing import Dict, Optional

from utilities.transformation import safe_atan2


# Per-signal magnitude thresholds (|x| <= bound) unless asymmetric.
# nuPlan PDM-Closed comfort constants, reused by NAVSIM Comfort/History Comfort:
# nuplan-devkit/nuplan/planning/simulation/planner/pdm_planner/scoring/pdm_comfort_metrics.py
DEFAULT_THRESHOLDS = {
    "lon_accel_min": -4.05,       # m/s^2
    "lon_accel_max": 2.40,        # m/s^2
    "lat_accel_abs_max": 4.89,    # m/s^2
    "lon_jerk_abs_max": 4.13,     # m/s^3
    "lat_jerk_abs_max": 8.37,     # m/s^3, reuses nuPlan's total-jerk bound (no per-axis lateral bound exists)
    "jerk_mag_abs_max": 8.37,     # m/s^3, sqrt(lon^2 + lat^2)
    "yaw_rate_abs_max": 0.95,     # rad/s
    "yaw_accel_abs_max": 1.93,    # rad/s^2
    "speed_abs_max": 25.0,        # m/s, not in nuPlan/NAVSIM, placeholder urban bound
}

# hard = physical/tire/steering envelope (Waymax-style |accel| infeasibility + speed sanity bound)
# soft = achievable but uncomfortable (nuPlan/NAVSIM Comfort subscore signals)
HARD_SIGNALS_DEFAULT = ("lon_accel", "lat_accel", "speed")
SOFT_SIGNALS_DEFAULT = ("lon_jerk", "lat_jerk", "jerk_mag", "yaw_rate", "yaw_accel")


class ComfortMetric:
    def __init__(
        self,
        dt: float = 0.5,
        thresholds: Optional[Dict[str, float]] = None,
        hard_signals: tuple = HARD_SIGNALS_DEFAULT,
        soft_signals: tuple = SOFT_SIGNALS_DEFAULT,
        soft_weights: Optional[Dict[str, float]] = None,
        sanity_check_local_frame: bool = True,
        sanity_check_max_speed: float = 60.0,  # m/s, generous bound to catch wrong-frame inputs
        compute_lon_accel: bool = True,
        compute_lat_accel: bool = True,
        compute_lon_jerk: bool = True,
        compute_lat_jerk: bool = True,
        compute_jerk_mag: bool = True,
        compute_yaw_rate: bool = True,
        compute_yaw_accel: bool = True,
        compute_speed_bound: bool = True,
    ):
        self.dt = float(dt)
        self.thresholds = dict(DEFAULT_THRESHOLDS)
        if thresholds is not None:
            self.thresholds.update(thresholds)

        self.sanity_check_local_frame = bool(sanity_check_local_frame)
        self.sanity_check_max_speed = float(sanity_check_max_speed)

        self.enabled = {
            "lon_accel": compute_lon_accel,
            "lat_accel": compute_lat_accel,
            "lon_jerk": compute_lon_jerk,
            "lat_jerk": compute_lat_jerk,
            "jerk_mag": compute_jerk_mag,
            "yaw_rate": compute_yaw_rate,
            "yaw_accel": compute_yaw_accel,
            "speed": compute_speed_bound,
        }

        self.hard_signals = tuple(s for s in hard_signals if self.enabled.get(s, False))
        self.soft_signals = tuple(s for s in soft_signals if self.enabled.get(s, False))

        if soft_weights is None:
            self.soft_weights = {s: 1.0 for s in self.soft_signals}
        else:
            self.soft_weights = {s: soft_weights.get(s, 1.0) for s in self.soft_signals}

        self.mode_prob_list = []           # list of (a, M)
        self.hard_goodness_list = []       # list of (a, M) in {0,1}
        self.soft_goodness_list = []       # list of (a, M) in [0,1]
        self.final_goodness_list = []      # list of (a, M) in [0,1]
        self.per_signal_rate_list = {s: [] for s in self.enabled if self.enabled[s]}
        self.per_signal_any_list = {s: [] for s in self.enabled if self.enabled[s]}

    def _backward_diff(self, x: Tensor, dt: float) -> Tensor:
        # x: (..., T', C) -> (..., T'-1, C); output[t] = (x[t+1] - x[t]) / dt
        return (x[..., 1:, :] - x[..., :-1, :]) / dt

    def _sanity_check_local_frame(self, preds_xy: Tensor, reg_mask: Tensor) -> None:
        # checks implied speed from origin to first valid pred; catches wrong-frame (global vs local) inputs
        A, M, T = reg_mask.shape[0], preds_xy.shape[1], preds_xy.shape[2]
        reg_mask_b = reg_mask.bool().unsqueeze(1).expand(A, M, T)  # (A, M, T)

        idx_range = torch.arange(T, device=preds_xy.device).view(1, 1, T)
        first_valid_idx = torch.where(
            reg_mask_b.any(dim=-1),
            torch.where(reg_mask_b, idx_range, T).min(dim=-1).values,
            torch.full((A, M), -1, device=preds_xy.device),
        )  # (A, M)
        has_valid = first_valid_idx >= 0
        if not has_valid.any():
            return

        safe_idx = first_valid_idx.clamp(min=0)  # (A, M)
        first_pos = torch.gather(
            preds_xy, dim=2, index=safe_idx.view(A, M, 1, 1).expand(A, M, 1, 2)
        ).squeeze(2)  # (A, M, 2)

        elapsed = (safe_idx.float() + 1.0) * self.dt  # (A, M)
        implied_speed = first_pos.norm(dim=-1) / elapsed.clamp(min=self.dt)  # (A, M)

        offending = has_valid & (implied_speed > self.sanity_check_max_speed)
        if offending.any():
            a_idx, m_idx = torch.nonzero(offending, as_tuple=True)
            worst = implied_speed[offending].max().item()
            raise AssertionError(
                f"ComfortMetric: implied speed from origin to the first valid "
                f"predicted timestep exceeds sanity_check_max_speed "
                f"({self.sanity_check_max_speed} m/s) for "
                f"{offending.sum().item()} (agent, mode) pair(s) (worst = "
                f"{worst:.1f} m/s, e.g. agent={a_idx[0].item()}, "
                f"mode={m_idx[0].item()}). preds is probably not in the "
                f"per-agent local frame (check vcs_to_local_tensor output "
                f"was passed, not raw/global vcs coords). Raise "
                f"sanity_check_max_speed or set sanity_check_local_frame=False "
                f"if this is a false positive."
            )

    @torch.no_grad()
    def _compute_kinematics(self, preds_xy: Tensor) -> Dict[str, Tensor]:
        # preds_xy: (A, M, T, 2); returns per-(agent, mode, t) signals (A, M, T) + "<name>_valid" masks
        A, M, T, _ = preds_xy.shape
        dt = self.dt
        device = preds_xy.device

        p0 = torch.zeros(A, 2, device=device, dtype=preds_xy.dtype)  # local-frame origin, per-agent

        p0_bmt = p0.unsqueeze(1).unsqueeze(2).expand(A, M, 1, 2)  # (A, M, 1, 2)
        pos_aug = torch.cat([p0_bmt, preds_xy], dim=-2)  # (A, M, T+1, 2)

        vel = self._backward_diff(pos_aug, dt)  # (A, M, T, 2)

        speed = vel.norm(dim=-1)  # (A, M, T)
        heading = safe_atan2(vel[..., 1], vel[..., 0])  # (A, M, T)

        accel_world = self._backward_diff(vel, dt)  # (A, M, T-1, 2)

        heading_for_accel = heading[..., 1:]  # (A, M, T-1)
        cos_h = heading_for_accel.cos()
        sin_h = heading_for_accel.sin()
        ax = accel_world[..., 0]
        ay = accel_world[..., 1]
        lon_accel_valid = ax * cos_h + ay * sin_h  # (A, M, T-1), body-frame rotation R(-heading)
        lat_accel_valid = -ax * sin_h + ay * cos_h  # (A, M, T-1)

        jerk_world = self._backward_diff(accel_world, dt)  # (A, M, T-2, 2)
        heading_for_jerk = heading[..., 2:]  # (A, M, T-2)
        jx = jerk_world[..., 0]
        jy = jerk_world[..., 1]
        cos_hj = heading_for_jerk.cos()
        sin_hj = heading_for_jerk.sin()
        lon_jerk_valid = jx * cos_hj + jy * sin_hj  # (A, M, T-2)
        lat_jerk_valid = -jx * sin_hj + jy * cos_hj  # (A, M, T-2)
        jerk_mag_valid = (lon_jerk_valid.square() + lat_jerk_valid.square()).sqrt()

        dh = heading[..., 1:] - heading[..., :-1]  # (A, M, T-1)
        dh = torch.atan2(dh.sin(), dh.cos())  # wrap to (-pi, pi]
        yaw_rate_valid = dh / dt  # (A, M, T-1)

        yaw_accel_valid = self._backward_diff(
            yaw_rate_valid.unsqueeze(-1), dt
        ).squeeze(-1)  # (A, M, T-2)

        def _pad_lead(x, k):
            # prepend k zero timesteps to bring shape back to T; never read due to validity mask
            pad_shape = list(x.shape)
            pad_shape[-1] = k
            pad = torch.zeros(pad_shape, dtype=x.dtype, device=device)
            return torch.cat([pad, x], dim=-1)

        def _mask_lead(shape_M_T, k):
            m = torch.zeros(shape_M_T, dtype=torch.bool, device=device)
            m[..., k:] = True
            return m

        out: Dict[str, Tensor] = {}

        out["speed"] = speed
        out["speed_valid"] = torch.ones(A, M, T, dtype=torch.bool, device=device)

        out["lon_accel"] = _pad_lead(lon_accel_valid, 1)
        out["lat_accel"] = _pad_lead(lat_accel_valid, 1)
        out["yaw_rate"] = _pad_lead(yaw_rate_valid, 1)
        base_2nd_valid = _mask_lead((A, M, T), 1)
        out["lon_accel_valid"] = base_2nd_valid.clone()
        out["lat_accel_valid"] = base_2nd_valid.clone()
        out["yaw_rate_valid"] = base_2nd_valid.clone()

        out["lon_jerk"] = _pad_lead(lon_jerk_valid, 2)
        out["lat_jerk"] = _pad_lead(lat_jerk_valid, 2)
        out["jerk_mag"] = _pad_lead(jerk_mag_valid, 2)
        out["yaw_accel"] = _pad_lead(yaw_accel_valid, 2)
        base_3rd_valid = _mask_lead((A, M, T), 2)
        out["lon_jerk_valid"] = base_3rd_valid.clone()
        out["lat_jerk_valid"] = base_3rd_valid.clone()
        out["jerk_mag_valid"] = base_3rd_valid.clone()
        out["yaw_accel_valid"] = base_3rd_valid.clone()

        return out

    def _violation_flag(self, name: str, values: Tensor) -> Tensor:
        th = self.thresholds
        if name == "lon_accel":
            return (values < th["lon_accel_min"]) | (values > th["lon_accel_max"])
        elif name == "lat_accel":
            return values.abs() > th["lat_accel_abs_max"]
        elif name == "lon_jerk":
            return values.abs() > th["lon_jerk_abs_max"]
        elif name == "lat_jerk":
            return values.abs() > th["lat_jerk_abs_max"]
        elif name == "jerk_mag":
            return values.abs() > th["jerk_mag_abs_max"]
        elif name == "yaw_rate":
            return values.abs() > th["yaw_rate_abs_max"]
        elif name == "yaw_accel":
            return values.abs() > th["yaw_accel_abs_max"]
        elif name == "speed":
            return values.abs() > th["speed_abs_max"]
        else:
            raise KeyError(f"Unknown signal: {name}")

    @torch.no_grad()
    def update(self, preds: Tensor, reg_mask: Tensor, pi: Tensor) -> None:
        # preds: (A, M, T, 2+) local-frame trajectory; reg_mask: (A, T); pi: (A, M) mode probs
        A, M, T, _ = preds.shape
        xy = preds[..., :2]

        if self.sanity_check_local_frame:
            self._sanity_check_local_frame(xy, reg_mask)

        sig = self._compute_kinematics(xy)

        pred_valid = reg_mask.bool().unsqueeze(1).expand(A, M, T)  # (A, T) -> (A, M, T)

        per_signal_rate = {}  # (A, M) float in [0, 1]
        per_signal_any = {}   # (A, M) float in {0, 1}

        for name, enabled in self.enabled.items():
            if not enabled:
                continue
            values = sig[name]  # (A, M, T)
            valid = sig[name + "_valid"] & pred_valid  # (A, M, T)
            violated = self._violation_flag(name, values) & valid  # (A, M, T)

            valid_count = valid.sum(dim=-1).clamp(min=1)  # (A, M)
            viol_count = violated.sum(dim=-1)  # (A, M)
            rate = viol_count.float() / valid_count.float()  # (A, M)
            any_viol = (viol_count > 0).float()  # (A, M)

            has_any_valid = (valid.sum(dim=-1) > 0)  # (A, M), no valid steps -> pass (rate=0, any=0)
            rate = torch.where(has_any_valid, rate, torch.zeros_like(rate))
            any_viol = torch.where(has_any_valid, any_viol, torch.zeros_like(any_viol))

            per_signal_rate[name] = rate
            per_signal_any[name] = any_viol

            self.per_signal_rate_list[name].append(rate)
            self.per_signal_any_list[name].append(any_viol)

        # hard tier: AND across hard signals via product of (1 - any_viol)
        if len(self.hard_signals) > 0:
            hard_ok = torch.ones(A, M, device=preds.device)
            for name in self.hard_signals:
                hard_ok = hard_ok * (1.0 - per_signal_any[name])
        else:
            hard_ok = torch.ones(A, M, device=preds.device)

        # soft tier: weighted mean of (1 - rate) across soft signals
        if len(self.soft_signals) > 0:
            total_w = sum(self.soft_weights[s] for s in self.soft_signals)
            if total_w <= 0:
                soft_ok = torch.ones(A, M, device=preds.device)
            else:
                soft_ok = torch.zeros(A, M, device=preds.device)
                for name in self.soft_signals:
                    w = self.soft_weights[name] / total_w
                    soft_ok = soft_ok + w * (1.0 - per_signal_rate[name])
        else:
            soft_ok = torch.ones(A, M, device=preds.device)

        final_ok = hard_ok * soft_ok  # (A, M), multiplicative gate

        self.mode_prob_list.append(pi)
        self.hard_goodness_list.append(hard_ok)
        self.soft_goodness_list.append(soft_ok)
        self.final_goodness_list.append(final_ok)

    def compute(self, n: int = 1) -> Dict[str, object]:
        mode_prob_all = torch.cat(self.mode_prob_list, dim=0)  # (N, M)
        hard_all = torch.cat(self.hard_goodness_list, dim=0)   # (N, M)
        soft_all = torch.cat(self.soft_goodness_list, dim=0)   # (N, M)
        final_all = torch.cat(self.final_goodness_list, dim=0) # (N, M)

        N, M = mode_prob_all.shape
        topn_idx = torch.topk(
            mode_prob_all, n, dim=1, largest=True, sorted=True
        ).indices  # (N, n)
        batch_indices = torch.arange(N, device=mode_prob_all.device).unsqueeze(1).expand(N, n)

        def _topn_mean_over_A(x_NM: Tensor) -> Tensor:
            topn = x_NM[batch_indices, topn_idx]  # (N, n)
            return topn.mean(dim=-1).mean()  # mean over modes, then agents

        result: Dict[str, object] = {}
        result["comfort_final"] = _topn_mean_over_A(final_all)
        result["comfort_hard"] = _topn_mean_over_A(hard_all)
        result["comfort_soft"] = _topn_mean_over_A(soft_all)

        rate_stats = {}
        any_stats = {}
        for name in self.per_signal_rate_list:
            if len(self.per_signal_rate_list[name]) == 0:
                continue
            r_all = torch.cat(self.per_signal_rate_list[name], dim=0)  # (N, M)
            a_all = torch.cat(self.per_signal_any_list[name], dim=0)   # (N, M)
            rate_stats[name] = _topn_mean_over_A(r_all)
            any_stats[name] = _topn_mean_over_A(a_all)

        result["per_signal_violation_rate"] = rate_stats
        result["per_signal_any_violation"] = any_stats
        return result

    def reset(self) -> None:
        del self.mode_prob_list
        del self.hard_goodness_list
        del self.soft_goodness_list
        del self.final_goodness_list
        del self.per_signal_rate_list
        del self.per_signal_any_list
        self.mode_prob_list = []
        self.hard_goodness_list = []
        self.soft_goodness_list = []
        self.final_goodness_list = []
        self.per_signal_rate_list = {s: [] for s in self.enabled if self.enabled[s]}
        self.per_signal_any_list = {s: [] for s in self.enabled if self.enabled[s]}