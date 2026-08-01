"""Render NAVSIM scenes (BEV + 8 cameras) for a set of tokens.

Uses the (restored) ``navsim.visualization`` package. Does NOT need the metric
cache or a trained model: it loads scenes straight from the OpenScene logs and
renders

  * ``plot_cameras_frame_with_annotations`` -> 8 surround cameras + BEV with GT boxes
  * ``plot_bev_with_agent``                 -> BEV map with GT (human) trajectory
                                               and a ConstantVelocity baseline

Env required (already set in this workspace):
  OPENSCENE_DATA_ROOT, NUPLAN_MAPS_ROOT, NUPLAN_MAP_VERSION

Run (from the navsim venv):
  cd /workspace/lead/3rd_party/navsim_workspace/navsimv1.1
  ./venv/bin/python /workspace/lead/scripts_clapad/vis_navsim_scenes.py
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from navsim.agents.abstract_agent import AbstractAgent
from navsim.agents.constant_velocity_agent import ConstantVelocityAgent
from navsim.agents.carla_transfuser_agent import CarlaTransfuserAgent
from navsim.agents.transfuser.transfuser_config import TransfuserConfig
from navsim.common.dataclasses import SceneFilter, SensorConfig
from navsim.common.dataloader import SceneLoader
from navsim.visualization.plots import (
    plot_cameras_frame_with_annotations,
    configure_bev_ax,
    configure_ax,
)
from navsim.visualization.bev import (
    add_configured_bev_on_ax,
    add_trajectory_to_bev_ax,
)
from navsim.visualization.config import TRAJECTORY_CONFIG

V4_CKPT = (
    "/data/datasets/AD_datasets/Bench2Drive-ckpts/thomas/clapad/navsim/"
    "TestByeaghajanzadeh_20260506_155115_v4/model_0060.pth"
)

V5_CKPT = (
    "/data/datasets/AD_datasets/Bench2Drive-ckpts/thomas/clapad/navsim/"
    "TestByeaghajanzadeh_20260722_084623/model_0060.pth"
)

DEFAULT_CKPT = V5_CKPT

# token -> label. Mix of failure buckets (from the navtest CSV) and clean scenes.
DEFAULT_TOKENS = {
    "ca8b5c5ed914507c": "dac_fail",
    "13310ca9a7715154": "dac_fail",
    "d987180f18155985": "dac_fail",
    "53f754a75b955383": "collision",
    "c2ce362b9cf95fee": "collision",
    "d9b79e606e595700": "collision",
    "96a252e16d1553e1": "perfect",
    "c672f1584cb75697": "perfect",
}


def _apply_legacy_clapad_defaults(checkpoint: str) -> None:
    """Apply compatibility defaults only for missing checkpoint config keys.

    Some older checkpoints do not persist all training config fields in
    ``config.json``. We only backfill missing fields:
      * ``path_head_hidden_dim_multipliers`` -> ``[2, 1]``
      * ``speed_bins`` -> 7-bin NAVSIM paper bins.

    If a key exists in ``config.json``, its explicit value is kept.
    """
    import json
    import torch
    from lead.training.config_training import TrainingConfig as CarlaTrainingConfig

    ckpt_dir = checkpoint if os.path.isdir(checkpoint) else os.path.dirname(checkpoint)
    cfg_path = os.path.join(ckpt_dir, "config.json")
    cfg = {}
    if os.path.exists(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

    if "path_head_hidden_dim_multipliers" not in cfg:
        CarlaTrainingConfig.path_head_hidden_dim_multipliers = [2, 1]

    if "speed_bins" not in cfg:
        CarlaTrainingConfig.speed_bins = torch.tensor(
            [0.0, 2.5, 5.0, 7.0, 8.5, 10.0, 12.0]
        )


def build_agent(kind: str, checkpoint: str) -> AbstractAgent:
    """Build the requested agent. 'cv' = constant velocity, 'clapad' = CLAP-AD."""
    if kind == "cv":
        agent: AbstractAgent = ConstantVelocityAgent()
    elif kind == "clapad":
        if os.path.isdir(checkpoint):
            checkpoint = os.path.join(checkpoint, "model_0060.pth")
        _apply_legacy_clapad_defaults(checkpoint)
        agent = CarlaTransfuserAgent(
            config=TransfuserConfig(), checkpoint_path=checkpoint
        )
        agent.initialize()
        # torch 2.0.1 autocast rejects bf16 on this device; weights are fp32, so
        # just disable mixed precision for inference.
        agent._carla_open_loop_inference.config_training.use_mixed_precision_training = False
        return agent
    else:
        raise ValueError(f"Unknown agent kind: {kind}")
    agent.initialize()
    return agent


# BEV overlay colors.
HUMAN_COLOR = "#1b5e20"      # dark green -> human (GT) future trajectory
TRAJ_COLOR = "#8e44ad"       # purple     -> CLAP-AD resolved trajectory
PRED_PATH_COLOR = "#5dade2"  # light blue -> CLAP-AD raw predicted path (sdc_traj)
GT_PATH_COLOR = "#f4c20d"    # yellow     -> extracted GT path (ego future route)
OLD_GT_COLOR = "#e67e22"     # orange     -> previous (windowed) GT path, --compare-gt


def zero_score_tokens(csv_path: str) -> dict:
    """Map every score==0 token to the PDMS gate it failed (priority order)."""
    import pandas as pd

    df = pd.read_csv(csv_path)
    df = df[df["score"] == 0]
    labels: dict = {}
    for _, row in df.iterrows():
        if row["drivable_area_compliance"] == 0:
            label = "dac"
        elif row["no_at_fault_collisions"] == 0:
            label = "nc"
        elif row["time_to_collision_within_bound"] == 0:
            label = "ttc"
        elif row["driving_direction_compliance"] == 0:
            label = "ddc"
        else:
            label = "other"
        labels[row["token"]] = label
    return labels


def clapad_prediction(agent: AbstractAgent, scene, threshold: float = 0.3):
    """Run CLAP-AD once and return decision-relevant outputs.

    Returns ``(current_speed, target_speed, scores, speed_bins)`` where
    ``scores`` is the per-criterion sigmoid score array of shape ``[M, S]``
    (M safety criteria x S speed bins). ``target_speed`` reproduces
    ``traj_from_clapad``: median (rounded up) bin whose collision score
    (criterion 0) exceeds ``threshold``.
    """
    import cv2
    import numpy as np
    import torch

    agent.eval()
    features: dict = {}
    for builder in agent.get_feature_builders():
        features.update(builder.compute_features(scene.get_agent_input()))
    features = {k: v.unsqueeze(0) for k, v in features.items()}

    rgb = features["camera_feature"].numpy()
    rgb = cv2.imdecode(np.frombuffer(rgb, np.uint8), cv2.IMREAD_COLOR)
    rgb = np.transpose(rgb, (2, 0, 1))
    rgb = torch.tensor(rgb).unsqueeze(0).float()
    with torch.no_grad():
        output = agent._carla_open_loop_inference.forward(
            {
                "rgb": rgb,
                "command": features["status_feature"][:, :4].reshape(-1, 4),
                "speed": torch.linalg.norm(
                    features["status_feature"][:, 4:6]
                ).reshape(-1, 1),
                "acceleration": torch.linalg.norm(
                    features["status_feature"][:, 6:8]
                ).reshape(-1, 1),
            }
        )
    logits = output.pred_clapad["speed_logits"][0, 0]  # (M, S)
    scores = torch.sigmoid(logits.float()).cpu().numpy()  # (M, S)
    current_speed = float(torch.linalg.norm(features["status_feature"][:, 4:6]))
    speed_bins = np.asarray(agent._carla_model_config.speed_bins, dtype=float)

    coll = scores[0]
    valid = np.where(coll > threshold)[0]
    if len(valid):
        n = len(valid)
        median_idx = int(valid[(n - 1) // 2] if n % 2 == 1 else valid[n // 2])
        target_speed = float(speed_bins[median_idx])
    else:
        target_speed = float(speed_bins[0])

    pred_path = output.pred_clapad["sdc_traj"][0].detach().cpu().numpy()  # (T, 2)
    gt_scores = _clapad_gt_scores(
        agent, scene, output.pred_clapad, current_speed, scores.shape
    )
    return current_speed, target_speed, scores, gt_scores, speed_bins, pred_path


# Safety criteria actually consumed by CLAP-AD's NAVSIM controller. Each entry is
# (display name, index into the [M, S] score matrix). CLAP-AD currently selects
# speed from the collision/TTC criterion (index 0) only; append entries here as
# more criteria are wired into the controller.
USED_CRITERIA = [
    ("collision/TTC", 0),
]


def _clapad_gt_scores(agent, scene, pred_clapad, current_speed, pred_shape):
    """Ground-truth per-bin scores via the model's own SpeedScoringLoss (debug).

    Reproduces exactly what the model is distilled against: build the scene score
    info from the navsim scene and score the model's predicted path against the
    ground-truth actors. Returns an ``[M, S]`` array aligned with the predicted
    score order.
    """
    import numpy as np
    import torch
    from navsim.agents.transfuser.transfuser_features import (
        get_scene_score_info_from_scene,
    )
    from lead.data_loader.navsim_dataset import preprocess_scene_info

    model = agent._carla_open_loop_inference.nets[0]
    loss_speed = model.planning_decoder.loss_speed
    score_keys = list(loss_speed.score_cfg.keys())
    num_bins = pred_shape[1]
    try:
        ssi = [preprocess_scene_info(get_scene_score_info_from_scene(scene))]
        path_pred = pred_clapad["sdc_traj"]  # [1, T, 2], model frame
        device = path_pred.device
        dummy_pred = torch.zeros(1, 1, len(score_keys), num_bins, device=device)
        curr = torch.tensor([current_speed], device=device)
        losses = loss_speed(
            curr, path_pred, dummy_pred, ssi,
            path_pred_augmented=None, debug=True,
        )
        raw = losses["raw_scores"]
        gt = np.stack(
            [raw[k][0, 0].detach().cpu().numpy() for k in score_keys]
        )  # [M, S]
    except Exception as exc:  # visualization is best-effort
        print(f"  [warn] GT score computation failed: {exc}")
        gt = np.full(pred_shape, np.nan)
    return gt


def draw_score_row(ax, score_matrix, speed_bins, target_speed, ylabel, threshold=0.3):
    """Draw grouped bars of the *used* per-bin scores (one group per criterion)."""
    import numpy as np

    x = np.arange(len(speed_bins))
    n_used = len(USED_CRITERIA)
    width = 0.8 / n_used
    for j, (name, idx) in enumerate(USED_CRITERIA):
        if idx < score_matrix.shape[0]:
            ax.bar(x + j * width - 0.4 + width / 2, score_matrix[idx],
                   width=width, label=name)
    ax.axhline(threshold, color="k", ls="--", lw=1, label=f"thresh {threshold}")
    target_idx = int(np.argmin(np.abs(speed_bins - target_speed)))
    ax.axvline(target_idx, color="red", ls=":", lw=1.5, label="selected speed")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b:.1f}" for b in speed_bins])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=7, ncol=n_used + 2, loc="lower left")


def draw_bev_on_ax(ax, scene, agent) -> None:
    """Replicate ``plot_bev_with_agent`` onto a provided axes."""
    frame_idx = scene.scene_metadata.num_history_frames - 1
    human_trajectory = scene.get_future_trajectory()
    agent_trajectory = agent.compute_trajectory(scene.get_agent_input())
    add_configured_bev_on_ax(ax, scene.map_api, scene.frames[frame_idx])
    human_cfg = {**TRAJECTORY_CONFIG["human"],
                 "fill_color": HUMAN_COLOR, "line_color": HUMAN_COLOR}
    agent_cfg = {**TRAJECTORY_CONFIG["agent"],
                 "fill_color": TRAJ_COLOR, "line_color": TRAJ_COLOR}
    add_trajectory_to_bev_ax(ax, human_trajectory, human_cfg)
    add_trajectory_to_bev_ax(ax, agent_trajectory, agent_cfg)
    configure_bev_ax(ax)
    configure_ax(ax)


def extract_gt_route(agent, scene, future_local=None):
    """Reproduce the path-head GT (``data['route']``) for plotting.

    Mirrors the navsim dataloader/cache builder: run ``preprocess_path`` at a
    fixed 1 m spacing from the ego origin, with ``num_route_points_prediction``
    points. NAVSIM always uses this fixed 1 m spacing -- NOT
    ``config.target_first_distance`` (that field is CARLA-leaderboard-scoped,
    e.g. 2.5 m, and unrelated to the NAVSIM route head; using it here would
    silently mismatch the actual cached training GT). ``preprocess_path``
    returns the CARLA frame (y flipped); we negate y back to the navsim frame
    so it overlays consistently with the other trajectories.

    ``future_local`` optionally supplies a longer future ego path (N, 2) pulled
    straight from the raw log (see ``build_full_future_paths``); without it we
    fall back to the truncated scene window via ``get_future_path``.
    """
    from lead.data_loader.navsim_dataset import preprocess_path

    cfg = agent._carla_model_config
    if future_local is None:
        raw = agent.get_target_builders()[0].get_future_path(scene).poses  # (N, 3)
    else:
        raw = future_local  # (N, 2) full remaining log future
    route, mask = preprocess_path(
        raw,
        sampling_distance=1.0,
        max_points=cfg.num_route_points_prediction,
    )
    route = route[mask].copy()          # (M, 2), CARLA frame
    route[:, 1] = -route[:, 1]           # CARLA -> navsim frame
    return route


def extract_gt_route_windowed(agent, scene):
    """OLD (pre-fix) GT route: 1 m path from the truncated scene window only.

    Reproduces the previous cache behaviour (``num_future_frames`` future frames)
    for side-by-side comparison against the full raw-log route.
    """
    from lead.data_loader.navsim_dataset import preprocess_path
    from nuplan.common.actor_state.state_representation import StateSE2
    from navsim.agents.transfuser.transfuser_features import (
        convert_absolute_to_relative_se2_array,
    )

    cfg = agent._carla_model_config
    cur = scene.scene_metadata.num_history_frames - 1
    start = scene.frames[cur].ego_status.ego_pose
    fut = np.array(
        [scene.frames[i].ego_status.ego_pose
         for i in range(cur + 1, len(scene.frames))],
        dtype=np.float64,
    )
    local = convert_absolute_to_relative_se2_array(StateSE2(*start), fut)
    route, mask = preprocess_path(
        local,
        sampling_distance=1.0,
        max_points=cfg.num_route_points_prediction,
    )
    route = route[mask].copy()
    route[:, 1] = -route[:, 1]
    return route


def build_full_future_paths(log_dir, tokens):
    """Scan the raw logs once; return ``{token: future ego path (N, 2)}``.

    The scene window only loads ``num_future_frames`` future frames (~20 m),
    which is often shorter than CLAP-AD's 48 m predicted path. The raw logs hold
    the full remaining drive (hundreds of metres), so we extract every future ego
    pose after the token and express it in the current-ego frame (x forward, y
    left) -- the exact frame ``get_future_path`` uses (verified to <4 mm).
    """
    import math
    import pickle

    from nuplan.common.actor_state.state_representation import StateSE2
    from navsim.agents.transfuser.transfuser_features import (
        convert_absolute_to_relative_se2_array,
    )

    def yaw_from_quat(q):
        w, x, y, z = q
        return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    wanted = set(tokens)
    out: dict = {}
    for log_file in sorted(Path(log_dir).iterdir()):
        if not (wanted - set(out)):
            break
        try:
            frames = pickle.load(open(log_file, "rb"))
        except Exception:
            continue
        toks = [f["token"] for f in frames]
        for t in (wanted & set(toks)) - set(out):
            i = toks.index(t)
            seq = frames[i:]
            g = np.array([
                [f["ego2global_translation"][0],
                 f["ego2global_translation"][1],
                 yaw_from_quat(f["ego2global_rotation"])]
                for f in seq
            ])
            if len(g) < 2:
                continue
            out[t] = convert_absolute_to_relative_se2_array(
                StateSE2(*g[0]), g[1:]
            )[:, :2]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--agent",
        default="clapad",
        choices=["cv", "clapad"],
        help="Agent whose trajectory is drawn on the BEV.",
    )
    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CKPT,
        help="CLAP-AD checkpoint .pth (or its directory) for --agent clapad.",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="OpenScene split under $OPENSCENE_DATA_ROOT (navsim_logs/<split>).",
    )
    parser.add_argument(
        "--tokens",
        nargs="*",
        default=None,
        help="Scene tokens to render (default: built-in failure/success mix).",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Results CSV; render every token with score==0, labeled by which "
        "PDMS gate failed (dac/nc/ttc/ddc).",
    )
    parser.add_argument(
        "--no-cameras",
        action="store_true",
        help="Skip the 8-camera grid render (faster for bulk runs).",
    )
    parser.add_argument(
        "--compare-gt",
        action="store_true",
        help="Also overlay the previous windowed GT route (orange) for old-vs-new "
        "comparison, with valid-point counts in the legend.",
    )
    parser.add_argument(
        "--gt-only",
        action="store_true",
        help="Render only a plain 2D plot of the GT route points (no BEV, no "
        "cameras, no score bars). Honors --compare-gt.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output dir (default: /workspace/temp_vis/navsim_viz/<timestamp>).",
    )
    args = parser.parse_args()

    if args.csv:
        token_labels = zero_score_tokens(args.csv)
        print(f"Selected {len(token_labels)} zero-PDMS tokens from {args.csv}")
    elif args.tokens:
        token_labels = {t: "custom" for t in args.tokens}
    else:
        token_labels = dict(DEFAULT_TOKENS)
    tokens = list(token_labels.keys())

    data_root = Path(os.environ["OPENSCENE_DATA_ROOT"])
    data_path = data_root / "navsim_logs" / args.split
    sensor_blobs_path = data_root / "sensor_blobs" / args.split

    out_dir = Path(
        args.out
        or f"/workspace/temp_vis/navsim_viz/{datetime.now():%Y.%m.%d.%H.%M.%S}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir}")

    # Load only the requested tokens; sensors only at the current frame (idx 3).
    scene_filter = SceneFilter(
        num_history_frames=4,
        num_future_frames=10,
        frame_interval=1,
        has_route=True,
        max_scenes=len(tokens),
        log_names=None,
        tokens=tokens,
    )
    print(f"Scanning '{args.split}' logs for {len(tokens)} tokens...")
    scene_loader = SceneLoader(
        data_path=data_path,
        sensor_blobs_path=sensor_blobs_path,
        scene_filter=scene_filter,
        sensor_config=SensorConfig.build_all_sensors(include=[3]),
    )

    agent = build_agent(args.agent, args.checkpoint)
    print(f"Agent: {args.agent}")

    found = set(scene_loader.tokens)
    missing = [t for t in tokens if t not in found]
    if missing:
        print(f"WARNING: {len(missing)} token(s) not found in split: {missing}")

    # Full future ego paths (raw logs) so the GT route can fill the whole horizon.
    full_future = build_full_future_paths(data_path, scene_loader.tokens)
    print(f"Full-future paths extracted for {len(full_future)} tokens")

    for token in scene_loader.tokens:
        label = token_labels.get(token, "scene")
        scene = scene_loader.get_scene_from_token(token)
        frame_idx = scene.scene_metadata.num_history_frames - 1

        if args.gt_only:
            n_pts = agent._carla_model_config.num_route_points_prediction
            gt_route = extract_gt_route(agent, scene, full_future.get(token))
            fig, ax = plt.subplots(figsize=(6, 9))
            r = np.concatenate([[[0.0, 0.0]], gt_route])
            ds = 1.0  # NAVSIM GT route is always sampled at a fixed 1 m spacing
            ax.plot(
                r[:, 1],
                r[:, 0],
                color=GT_PATH_COLOR,
                ls="-",
                marker="x",
                markersize=5,
                label=f"GT new ({len(gt_route)}/{n_pts}, ds={ds:g}m)",
            )
            if args.compare_gt:
                old_route = extract_gt_route_windowed(agent, scene)
                o = np.concatenate([[[0.0, 0.0]], old_route])
                ax.plot(
                    o[:, 1],
                    o[:, 0],
                    color=OLD_GT_COLOR,
                    ls=":",
                    marker="+",
                    markersize=7,
                    label=f"GT old ({len(old_route)}/{n_pts}, ds={ds:g}m)",
                )
            ax.plot(0, 0, marker="s", color="black", markersize=8, label="ego")
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.3)
            ax.invert_xaxis()  # left of ego on the left, matching BEV
            ax.set_xlabel("lateral y (m)")
            ax.set_ylabel("forward x (m)")
            ax.set_title(f"[{label}] {token}", fontsize=9)
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(out_dir / f"{label}_{token}_gtroute.png", dpi=120)
            plt.close(fig)
            print(f"  rendered [{label}] {token} (gt-only)")
            continue

        if not args.no_cameras:
            fig, _ = plot_cameras_frame_with_annotations(scene, frame_idx)
            fig.savefig(out_dir / f"{label}_{token}_cameras.png", dpi=120)
            plt.close(fig)

        is_clapad = args.agent == "clapad"
        if is_clapad:
            fig = plt.figure(figsize=(8, 12))
            gs = fig.add_gridspec(3, 1, height_ratios=[6, 1, 1])
            ax = fig.add_subplot(gs[0])
            ax_pred = fig.add_subplot(gs[1])
            ax_gt = fig.add_subplot(gs[2])
        else:
            fig, ax = plt.subplots(figsize=(8, 8))
            ax_pred = ax_gt = None

        draw_bev_on_ax(ax, scene, agent)
        agent_label = "CLAP-AD trajectory" if is_clapad else "Const-vel"
        legend_handles = [
            Line2D([0], [0], color=HUMAN_COLOR, lw=2, marker="o",
                   markeredgecolor="black", label="Human (GT)"),
            Line2D([0], [0], color=TRAJ_COLOR, lw=2, marker="o",
                   markeredgecolor="black", label=agent_label),
        ]
        if is_clapad:
            current_speed, target_speed, pred_scores, gt_scores, speed_bins, pred_path = (
                clapad_prediction(agent, scene)
            )
            # Extracted GT path = the path-head target (data["route"]) sampled at
            # a fixed 1 m spacing from the ego origin, sourced from the full
            # raw-log future so it fills the whole horizon.
            gt_route = extract_gt_route(agent, scene, full_future.get(token))
            gt_path = np.concatenate([[[0.0, 0.0]], gt_route])
            ax.plot(gt_path[:, 1], gt_path[:, 0], color=GT_PATH_COLOR, lw=1.5,
                    ls="-", marker="x", markersize=5, zorder=2.4)
            if args.compare_gt:
                old_route = extract_gt_route_windowed(agent, scene)
                old_path = np.concatenate([[[0.0, 0.0]], old_route])
                ax.plot(old_path[:, 1], old_path[:, 0], color=OLD_GT_COLOR,
                        lw=1.5, ls=":", marker="+", markersize=7, zorder=2.5)
            # CLAP-AD raw predicted path (sdc_traj, before speed -> trajectory). Same
            # frame as the trajectory: negate y and anchor at the ego origin.
            path_xy = pred_path[:, :2].copy()
            path_xy[:, 1] = -path_xy[:, 1]
            path_xy = np.concatenate([[[0.0, 0.0]], path_xy])
            ax.plot(path_xy[:, 1], path_xy[:, 0], color=PRED_PATH_COLOR, lw=1.5,
                    ls="--", marker="x", markersize=6, zorder=2.4)
            legend_handles.append(
                Line2D([0], [0], color=GT_PATH_COLOR, lw=1.5, marker="x",
                       label=f"GT path new ({len(gt_route)}/"
                             f"{agent._carla_model_config.num_route_points_prediction}, "
                             f"ds=1m)"),
            )
            if args.compare_gt:
                legend_handles.append(
                    Line2D([0], [0], color=OLD_GT_COLOR, lw=1.5, ls=":",
                           marker="+",
                           label=f"GT path old ({len(old_route)}/"
                                 f"{agent._carla_model_config.num_route_points_prediction}, "
                                 f"ds=1m)"),
                )
            legend_handles.append(
                Line2D([0], [0], color=PRED_PATH_COLOR, lw=1.5, marker="x",
                       label="CLAP-AD path")
            )
            ax.text(
                0.02, 0.98,
                f"current speed: {current_speed:4.1f} m/s\n"
                f"target speed:  {target_speed:4.1f} m/s",
                transform=ax.transAxes, va="top", ha="left", fontsize=10,
                family="monospace",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )
            ax.set_title(f"[{label}] {token}", fontsize=9)
            draw_score_row(ax_pred, pred_scores, speed_bins, target_speed,
                           ylabel="predicted")
            ax_pred.set_title("Predicted speed scores", fontsize=9)
            draw_score_row(ax_gt, gt_scores, speed_bins, target_speed,
                           ylabel="ground truth")
            ax_gt.set_title("Ground-truth speed scores", fontsize=9)
            ax_gt.set_xlabel("speed bin (m/s)")
        ax.legend(handles=legend_handles, loc="upper right", fontsize=8,
                  framealpha=0.8)
        fig.tight_layout()
        fig.savefig(out_dir / f"{label}_{token}_bev.png", dpi=120)
        plt.close(fig)

        print(f"  rendered [{label}] {token}")

    print(f"Done. {len(scene_loader.tokens)} scene(s) -> {out_dir}")


if __name__ == "__main__":
    main()
