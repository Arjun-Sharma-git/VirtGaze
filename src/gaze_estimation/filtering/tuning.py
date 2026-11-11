"""Fit the UKF + One Euro filter parameters to a recorded eye-tracking session.

The session pairs the raw (unfiltered) gaze predictions with where a moving
target actually was.  Each candidate parameter set is scored by replaying the
raw predictions through the real :class:`FilterStage` (UKF, One Euro and the
fixation-driven noise scaling, exactly as live) and measuring how far the
filtered gaze is from the target:

* while the target **holds still**, error is mostly jitter, which more
  smoothing reduces;
* while it **moves** (at varying speed), error is mostly lag, which less
  smoothing reduces.

Minimising the combined error picks the trade-off that suits this person,
camera and lighting.  The constant calibration offset is the same for every
candidate, so it does not bias the choice.
"""

from __future__ import annotations

import copy
import json
import math
import os
import queue
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from gaze_estimation.config.config import Config
from gaze_estimation.pipeline.schemas import PredictionPacket

# Searched parameters and their (log-uniform) ranges
PARAM_RANGES: Dict[str, tuple] = {
    "ukf.process_noise": (0.01, 5000.0),
    "ukf.measurement_noise": (0.1, 500.0),
    "one_euro.min_cutoff": (0.05, 10.0),
    "one_euro.beta": (1e-5, 1.0),
}


@dataclass
class TuningSession:
    """Raw predictions and the target positions at the same instants."""

    t: np.ndarray  # (N,) frame timestamps, seconds
    raw: np.ndarray  # (N, 2) unfiltered predictions, pixels
    target: np.ndarray  # (N, 2) target position at t, pixels
    moving: np.ndarray  # (N,) bool: target in motion (pursuit) vs holding
    weight: np.ndarray  # (N,) 0 for samples to ignore (e.g. right after a jump)
    ears: np.ndarray  # (N, 2) left/right eye aspect ratio (fixation detector)


@dataclass
class TuningResult:
    params: Dict[str, float]
    error_px: float
    hold_error_px: float
    moving_error_px: float
    jitter_px: float = 0.0
    baseline: Dict[str, float] = field(default_factory=dict)


def _replay(session: TuningSession, config: Config) -> np.ndarray:
    """Run *session* through a fresh FilterStage built from *config*."""
    from gaze_estimation.pipeline.pipeline import FilterStage

    out: queue.Queue = queue.Queue()
    stage = FilterStage(queue.Queue(), out, threading.Event(), config=config)
    filtered = np.empty_like(session.raw)
    for i in range(len(session.t)):
        stage.process(
            PredictionPacket(
                timestamp=float(session.t[i]),
                screen_x=float(session.raw[i, 0]),
                screen_y=float(session.raw[i, 1]),
                confidence=1.0,
                raw_features={"left_ear": float(session.ears[i, 0]),
                              "right_ear": float(session.ears[i, 1])},
                source="ridge",
            )
        )
        est = out.get_nowait()
        filtered[i] = (est.screen_x, est.screen_y)
    return filtered


def with_params(base: Config, params: Dict[str, float]) -> Config:
    cfg = Config(copy.deepcopy(base.raw))
    for key, value in params.items():
        cfg.set(f"filtering.{key}", float(value))
    return cfg


def jitter(session: TuningSession, config: Config) -> float:
    """Mean frame-to-frame movement of the filtered gaze while the target holds still."""
    filtered = _replay(session, config)
    steps = np.linalg.norm(np.diff(filtered, axis=0), axis=1)
    hold = (session.weight[1:] > 0) & ~session.moving[1:] & ~session.moving[:-1]
    return float(steps[hold].mean()) if hold.any() else 0.0


def score(session: TuningSession, config: Config) -> tuple:
    """Return (overall, hold, moving) mean error in pixels."""
    filtered = _replay(session, config)
    err = np.linalg.norm(filtered - session.target, axis=1)
    w = session.weight
    hold = w * ~session.moving
    move = w * session.moving

    def _mean(weights: np.ndarray) -> float:
        total = float(weights.sum())
        return float((err * weights).sum() / total) if total > 0 else float("nan")

    return _mean(w), _mean(hold), _mean(move)


def current_params(config: Config) -> Dict[str, float]:
    return {k: float(config.get(f"filtering.{k}")) for k in PARAM_RANGES}


def tune(
    session: TuningSession,
    base: Config,
    n_random: int = 120,
    n_refine: int = 80,
    seed: int = 0,
    progress=None,
    steadiness: float = 3.0,
) -> TuningResult:
    """Random search over :data:`PARAM_RANGES`, then a local log-space refinement.

    Minimises ``error + steadiness * jitter`` (both pixels).  Error alone
    favours no smoothing whenever the model is noisy, which leaves the dot
    shaky; *steadiness* sets how much a steady dot is worth versus lag.
    """
    rng = np.random.default_rng(seed)
    keys = list(PARAM_RANGES)
    lo = np.log10([PARAM_RANGES[k][0] for k in keys])
    hi = np.log10([PARAM_RANGES[k][1] for k in keys])

    def evaluate(logv: np.ndarray) -> float:
        params = dict(zip(keys, (10.0 ** logv).tolist()))
        cfg = with_params(base, params)
        return score(session, cfg)[0] + steadiness * jitter(session, cfg)

    start = np.log10([current_params(base)[k] for k in keys])
    best_logv, best = start, evaluate(start)
    baseline_err = best
    total = n_random + n_refine
    for i in range(n_random):
        cand = rng.uniform(lo, hi)
        e = evaluate(cand)
        if e < best:
            best, best_logv = e, cand
        if progress:
            progress((i + 1) / total)
    step = 0.5
    for i in range(n_refine):
        cand = np.clip(best_logv + rng.normal(0, step, len(keys)), lo, hi)
        e = evaluate(cand)
        if e < best:
            best, best_logv = e, cand
        else:
            step = max(0.05, step * 0.97)
        if progress:
            progress((n_random + i + 1) / total)

    params = dict(zip(keys, (10.0 ** best_logv).tolist()))
    overall, hold, moving = score(session, with_params(base, params))
    b_overall, b_hold, b_moving = score(session, base)
    return TuningResult(
        params=params,
        error_px=overall,
        hold_error_px=hold,
        moving_error_px=moving,
        jitter_px=jitter(session, with_params(base, params)),
        baseline={"error_px": b_overall if math.isfinite(b_overall) else baseline_err,
                  "hold_error_px": b_hold, "moving_error_px": b_moving,
                  "jitter_px": jitter(session, base)},
    )


def save_filter_params(path: str, result: TuningResult) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    data = {
        "filtering": _nest(result.params),
        "stats": {
            "error_px": result.error_px,
            "hold_error_px": result.hold_error_px,
            "moving_error_px": result.moving_error_px,
            "jitter_px": result.jitter_px,
            "default_error_px": result.baseline.get("error_px"),
        },
    }
    with open(path, "w") as fh:
        json.dump(data, fh, indent=2)


def apply_filter_params(config: Config, path: str) -> Optional[Dict[str, float]]:
    """Override ``filtering.*`` in *config* with a saved tuning, if present."""
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        data = json.load(fh)
    applied: Dict[str, float] = {}
    for section, values in data.get("filtering", {}).items():
        for key, value in values.items():
            dotted = f"{section}.{key}"
            if dotted in PARAM_RANGES:
                config.set(f"filtering.{dotted}", float(value))
                applied[dotted] = float(value)
    return applied


def _nest(params: Dict[str, float]) -> Dict[str, Dict[str, float]]:
    nested: Dict[str, Dict[str, float]] = {}
    for dotted, value in params.items():
        section, key = dotted.split(".", 1)
        nested.setdefault(section, {})[key] = float(value)
    return nested


def interpolate_targets(
    frame_t: Sequence[float], target_t: Sequence[float], target_xy: np.ndarray
) -> np.ndarray:
    """Target position at each frame time (linear between recorded UI frames)."""
    tt = np.asarray(target_t)
    xy = np.asarray(target_xy, dtype=np.float64)
    ft = np.asarray(frame_t)
    return np.column_stack([np.interp(ft, tt, xy[:, 0]), np.interp(ft, tt, xy[:, 1])])


def segments_to_flags(
    frame_t: np.ndarray, segments: List[tuple], settle_sec: float = 0.4
) -> tuple:
    """Per-frame (moving, weight) from ``(start, end, kind)`` segments.

    Frames within *settle_sec* of a segment start get weight 0: the eyes need
    a moment to reach a jumped-to target or lock on to a moving one.
    """
    moving = np.zeros(len(frame_t), dtype=bool)
    weight = np.zeros(len(frame_t), dtype=np.float64)
    for start, end, kind in segments:
        inside = (frame_t >= start) & (frame_t < end)
        moving[inside] = kind == "move"
        weight[inside & (frame_t >= start + settle_sec)] = 1.0
    return moving, weight
