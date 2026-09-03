"""
Metrics recording + loading for multi-checkpoint / multi-Tsim eval sweeps.

One JSON record per (checkpoint, Tsim, data_split) eval run, appended as a
line to a `<model_name>/eval_metrics.jsonl` file. Each record has a `summary`
block (flat, dataframe-friendly) alongside a `full` block (raw per-timestep
arrays), for cases where summary reductions aren't enough.
"""
import json
from pathlib import Path
from typing import Literal, Optional

import pandas as pd

TrainCondition = Literal["reactive", "non_reactive", "mixed"]


def _reduce(per_timestep_pct) -> dict:
    """Both reductions you actually end up needing: terminal value (collision, L2)
    and horizon-mean (TTC-violation, which is instantaneous per timestep, not cumulative).
    Store both so you never have to re-derive one from `full` later."""
    vals = list(per_timestep_pct)
    return {"final": vals[-1], "mean": sum(vals) / len(vals)}


def build_metrics_record(
    *,
    checkpoint: str,
    model_name: str,
    data_split: str,
    tsim: float,
    train_condition: Optional[TrainCondition] = None,
    eval_reactive: Optional[bool] = None,
    planning_metrics_holder=None,
    target_pred_metrics_holder=None,
    target_offroad_metrics_holder=None,
    target_comfort_metrics_holder=None,
    scene_pred_metrics_holder=None,
    topk=(1, 5),
) -> dict:
    """Builds one JSON-serializable record for a single (checkpoint, Tsim) eval run.
    Pass None for any holder that wasn't used (mirrors use_target_net / use_scene_net) --
    that block is simply omitted from the record."""

    record = {
        "checkpoint": checkpoint,
        "model_name": model_name,
        "data_split": data_split,
        "tsim": tsim,
        "train_condition": train_condition,  # "reactive" | "non_reactive" | "mixed" | None
        "eval_condition": (
            "reactive" if eval_reactive else "non_reactive" if eval_reactive is not None else None
        ),
        "scene_prediction": None,
        "topk": {},
    }

    if scene_pred_metrics_holder is not None:
        min_ade, min_fde, mr = scene_pred_metrics_holder.compute(n=1)
        record["scene_prediction"] = {
            "min_ade": min_ade.item(),
            "min_fde": min_fde.item(),
            "miss_rate": mr.item(),
        }

    if planning_metrics_holder is not None:
        for n in topk:
            plan = planning_metrics_holder.compute(n=n)
            min_ade, min_fde, mr = target_pred_metrics_holder.compute(n=n)
            off_road = target_offroad_metrics_holder.compute(n=n)
            comfort = target_comfort_metrics_holder.compute(n=n)

            col = plan["mean_box_col_percent"].cpu().numpy().tolist()
            ttc_violation = plan["mean_ttc_violation_percent"].cpu().numpy().tolist()
            l2 = plan["min_L2"].cpu().numpy().tolist()

            summary = {
                "collision_pct": _reduce(col),
                "ttc_violation_pct": _reduce(ttc_violation),
                "l2": _reduce(l2),
                "min_ade": min_ade.item(),
                "min_fde": min_fde.item(),
                "miss_rate": mr.item(),
                "off_road_pct": off_road.item(),
                "comfort_final": comfort["comfort_final"].item(),
                "comfort_hard": comfort["comfort_hard"].item(),
                "comfort_soft": comfort["comfort_soft"].item(),
            }
            full = {
                "collision_pct": col,
                "ttc_violation_pct": ttc_violation,
                "l2": l2,
                "comfort_per_signal_violation_rate": {
                    k: v.item() for k, v in comfort["per_signal_violation_rate"].items()
                },
                "comfort_per_signal_any_violation": {
                    k: v.item() for k, v in comfort["per_signal_any_violation"].items()
                },
            }

            # at-fault fields are optional: only present once PlanningMetric.compute()
            # returns mean_at_fault_col_percent / mean_at_fault_share (see the at-fault patch).
            # Older checkpoints/holders without that patch still produce a valid record.
            if "mean_at_fault_col_percent" in plan:
                at_fault_col = plan["mean_at_fault_col_percent"].cpu().numpy().tolist()
                at_fault_share = plan["mean_at_fault_share"].cpu().numpy().tolist()
                summary["at_fault_collision_pct"] = _reduce(at_fault_col)
                summary["at_fault_share_pct"] = _reduce(at_fault_share)
                full["at_fault_collision_pct"] = at_fault_col
                full["at_fault_share_pct"] = at_fault_share

            record["topk"][str(n)] = {"summary": summary, "full": full}

    return record


class MetricsWriter:
    """Appends one record per line -- except when a record already exists for the
    same evaluation (matched on checkpoint, tsim, data_split, eval_condition),
    in which case that record is overwritten in place rather than duplicated.
    Safe to re-run a sweep (or one (checkpoint, Tsim) cell of it) without the
    file accumulating stale duplicate rows for the same evaluation."""

    # Fields that jointly identify "the same evaluation" -- a new record whose
    # values match all of these replaces the old one instead of appending.
    KEY_FIELDS = ("checkpoint", "tsim", "data_split", "eval_condition")

    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _same_run(self, a: dict, b: dict) -> bool:
        return all(a.get(f) == b.get(f) for f in self.KEY_FIELDS)

    def write(self, record: dict) -> None:
        existing = []
        if self.path.exists():
            with open(self.path) as f:
                existing = [json.loads(line) for line in f if line.strip()]

        # Drop any prior record for this same evaluation, then append the new one.
        existing = [r for r in existing if not self._same_run(r, record)]
        existing.append(record)

        with open(self.path, "w") as f:
            for r in existing:
                f.write(json.dumps(r) + "\n")


# --------------------------------------------------------------------------
# Loading / analysis side
# --------------------------------------------------------------------------

def load_records(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def find_record(
    path: str,
    checkpoint: Optional[str] = None,
    tsim: Optional[float] = None,
    data_split: Optional[str] = None,
    eval_condition: Optional[str] = None,
) -> dict:
    """Returns the first record matching the given filters. Raises if none match --
    this is the lookup the plotting script uses in place of building a per-(ckpt,tsim) filename."""
    for r in load_records(path):
        if checkpoint is not None and r["checkpoint"] != checkpoint:
            continue
        if tsim is not None and r["tsim"] != tsim:
            continue
        if data_split is not None and r["data_split"] != data_split:
            continue
        if eval_condition is not None and r["eval_condition"] != eval_condition:
            continue
        return r
    raise KeyError(
        f"No record matching checkpoint={checkpoint!r}, tsim={tsim!r}, "
        f"data_split={data_split!r}, eval_condition={eval_condition!r} in {path}",
        
    )


def summary_dataframe(path: str, k: int = 1) -> pd.DataFrame:
    """Flattens the `summary` block for a given top-k into one row per
    (checkpoint, tsim, train_condition, eval_condition) run."""
    records = load_records(path)
    rows = []
    for r in records:
        topk = r["topk"].get(str(k))
        if topk is None:
            continue
        row = {
            "checkpoint": r["checkpoint"],
            "model_name": r["model_name"],
            "data_split": r["data_split"],
            "tsim": r["tsim"],
            "train_condition": r["train_condition"],
            "eval_condition": r["eval_condition"],
        }
        for metric, val in topk["summary"].items():
            if isinstance(val, dict):
                row[f"{metric}_final"] = val["final"]
                row[f"{metric}_mean"] = val["mean"]
            else:
                row[metric] = val
        if r["scene_prediction"] is not None:
            row["scene_min_ade"] = r["scene_prediction"]["min_ade"]
            row["scene_min_fde"] = r["scene_prediction"]["min_fde"]
            row["scene_miss_rate"] = r["scene_prediction"]["miss_rate"]
        rows.append(row)
    return pd.DataFrame(rows)


def full_arrays(path: str, checkpoint: str, tsim: float, k: int = 1) -> dict:
    """Pulls the per-timestep arrays back out for one specific run, for deep dives."""
    return find_record(path, checkpoint=checkpoint, tsim=tsim)["topk"][str(k)]["full"]