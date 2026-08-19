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

from shapely import affinity

from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.agents.abstract_agent import AbstractAgent
from navsim.agents.constant_velocity_agent import ConstantVelocityAgent
from navsim.agents.carla_transfuser_agent import CarlaTransfuserAgent
from navsim.agents.transfuser.transfuser_config import TransfuserConfig
from navsim.common.dataclasses import SceneFilter, SensorConfig
from navsim.common.dataloader import MetricCacheLoader, SceneLoader
from navsim.evaluate.pdm_score import pdm_score
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.visualization.plots import (
    plot_cameras_frame_with_annotations,
    configure_bev_ax,
    configure_ax,
)
from navsim.visualization.bev import (
    add_configured_bev_on_ax,
    add_trajectory_to_bev_ax,
    add_polygon_to_bev_ax,
    add_oriented_box_to_bev_ax,
)
from navsim.visualization.config import TRAJECTORY_CONFIG

# Shared proposal sampling for PDM simulation/scoring: 40 poses @ 0.1s = 4s,
# matching navsim/planning/script/config/pdm_scoring/default_scoring_parameters.yaml
# (used both for the --detailed LQR rollout overlay and for real PDMS scoring).
PROPOSAL_SAMPLING = TrajectorySampling(num_poses=40, interval_length=0.1)

V4_CKPT = (
    "/data/datasets/AD_datasets/Bench2Drive-ckpts/thomas/clapad/navsim/"
    "TestByeaghajanzadeh_20260506_155115_v4/model_0060.pth"
)

V5_CKPT = (
    "/data/datasets/AD_datasets/Bench2Drive-ckpts/thomas/clapad/navsim/"
    "TestByeaghajanzadeh_20260722_084623/model_0060.pth"
)

V6_CKPT = (
    "/data/datasets/AD_datasets/Bench2Drive-ckpts/thomas/clapad/navsim/"
    "TestByeaghajanzadeh_20260722_084623_v3/model_0060.pth"
)

V7_CKPT = (
    "/data/datasets/AD_datasets/Bench2Drive-ckpts/thomas/clapad/navsim/"
    "TestByeaghajanzadeh_20260722_084623_v7/model_0037.pth"
)

V8_CKPT = (
    "/data/datasets/AD_datasets/Bench2Drive-ckpts/thomas/clapad/navsim/"
    "TestByeaghajanzadeh_20260722_084623_v8/model_0060.pth"
)

OVERFIT_CKPT = (
    "/workspace/lead/outputs/local_training/navsim_overfit_eaghajanzadeh_20260804_192203/model_0113.pth"
)

V9_CKPT = (
    "/data/datasets/AD_datasets/Bench2Drive-ckpts/thomas/clapad/navsim/"
    "TestByeaghajanzadeh_20260805_094518/model_0060.pth"
)

V12_CKPT = (
    "/workspace/lead/outputs/local_training/TestByeaghajanzadeh_20260810_164007/model_0059.pth"
)

# Latent TFv6 (--agent ltf): predicts a plain route (no CLAP-AD speed-scoring
# head), so it does NOT need the CLAP-AD legacy-config backfill and is
# plotted without the CLAP-AD-specific overlays/score bars.
LTF_CKPT = "/workspace/lead/outputs/checkpoints/ltfv6-navsim/model_0060.pth"

DEFAULT_CKPT = V12_CKPT

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
        # CarlaTrainingConfig.speed_bins = torch.tensor([
        #     0.0,
        #     2.77777778,
        #     5.55555556,
        #     6.94444444,
        #     8.33333333,
        #     9.72222222,
        #     11.11111111,
        #     12.5,
        #     13.88888889,
        #     15.27777778,
        #     16.66666667,
        #     18.05555556,
        #     19.44444444
        # ])


def build_agent(kind: str, checkpoint: str) -> AbstractAgent:
    """Build the requested agent. 'cv' = constant velocity, 'clapad' = CLAP-AD,
    'ltf' = latent TFv6 (plain route head, no CLAP-AD speed-scoring head)."""
    if kind == "cv":
        agent: AbstractAgent = ConstantVelocityAgent()
    elif kind in ("clapad", "ltf"):
        if os.path.isdir(checkpoint):
            checkpoint = os.path.join(checkpoint, "model_0060.pth")
        if kind == "clapad":
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


# Default BEV framing: keeps the full CLAP-AD path (~48 m) and GT route in
# view, unlike the navsim default (+/-32 m square), which sometimes clips the
# route ahead of the ego. Forward/backward are in the ego +x direction (drawn
# on the ax y-axis); lateral is +/-y (drawn on the ax x-axis).
BEV_FORWARD_M = 55.0
BEV_BACKWARD_M = 25.0
BEV_LATERAL_M = 40.0

# Drivable-area boundary overlay (--detailed). Not one of navsim's default BEV
# layers (``add_map_to_bev_ax`` only draws LANE/WALKWAYS/CARPARK_AREA/
# INTERSECTION/STOP_LINE/CROSSWALK) but matches what the PDM scorer actually
# checks ego against for drivable-area compliance.
DRIVABLE_AREA_COLOR = "#e6194b"
DRIVABLE_AREA_CONFIG = {
    "fill_color": DRIVABLE_AREA_COLOR,
    "fill_color_alpha": 0.0,
    "line_color": DRIVABLE_AREA_COLOR,
    "line_color_alpha": 1.0,
    "line_width": 1.5,
    "line_style": "--",
    "zorder": 1.5,
}



# Layers that make up "drivable area" for object-based map queries.
# NOTE: ``SemanticMapLayer.DRIVABLE_AREA`` itself has no object representation
# in NuPlanMap.get_proximal_map_objects (raises AssertionError -- it only
# backs a raster/vector layer, not queryable MapObjects). navsim's own PDM
# scorer hits this exact limitation (see
# ``PDMDrivableMap.from_simulation``, which has a
# "TODO: Fix SemanticMapLayer.DRIVABLE_AREA problems" comment) and instead
# reconstructs the drivable area from these four object layers -- mirrored
# here for the visualization overlay.
DRIVABLE_AREA_LAYERS = [
    SemanticMapLayer.ROADBLOCK,
    SemanticMapLayer.ROADBLOCK_CONNECTOR,
    SemanticMapLayer.INTERSECTION,
    SemanticMapLayer.CARPARK_AREA,
]


def add_drivable_area_to_bev_ax(ax, map_api, origin: StateSE2, radius: float) -> None:
    """Overlay the drivable-area polygon boundary on a BEV axes.

    Reproduces the local-frame transform used internally by
    ``navsim.visualization.bev.add_map_to_bev_ax`` (which does not expose a
    drivable-area layer) over the ``DRIVABLE_AREA_LAYERS`` union.
    """
    map_object_dict = map_api.get_proximal_map_objects(
        point=origin.point, radius=radius, layers=DRIVABLE_AREA_LAYERS
    )

    a, b = np.cos(origin.heading), np.sin(origin.heading)
    d, e = -np.sin(origin.heading), np.cos(origin.heading)
    xoff, yoff = -origin.x, -origin.y

    for layer in DRIVABLE_AREA_LAYERS:
        for map_object in map_object_dict[layer]:
            translated = affinity.affine_transform(map_object.polygon, [1, 0, 0, 1, xoff, yoff])
            rotated = affinity.affine_transform(translated, [a, b, d, e, 0, 0])
            add_polygon_to_bev_ax(ax, rotated, DRIVABLE_AREA_CONFIG)


# LQR rollout overlay (--detailed). navsim's PDM scorer never scores the raw
# predicted trajectory directly: ``navsim/evaluate/pdm_score.py::pdm_score``
# re-simulates it through ``PDMSimulator`` (``BatchLQRTracker`` +
# ``BatchKinematicBicycleModel``, ``proposal_sampling`` = 40 poses @ 0.1s = 4s,
# see ``navsim/planning/script/config/pdm_scoring/default_scoring_parameters.yaml``)
# before scoring it. This overlay reproduces that exact closed-loop rollout so
# the 40 future ego boxes shown match what the PDM scorer actually evaluates.
LQR_BOX_COLOR = "#3d85c6"
LQR_BOX_CONFIG = {
    "fill_color": LQR_BOX_COLOR,
    "fill_color_alpha": 0.08,
    "line_color": LQR_BOX_COLOR,
    "line_color_alpha": 0.5,
    "line_width": 1.0,
    "line_style": "-",
    "zorder": 1.8,
}
HUMAN_LQR_BOX_COLOR = HUMAN_COLOR
HUMAN_LQR_BOX_CONFIG = {
    "fill_color": HUMAN_LQR_BOX_COLOR,
    "fill_color_alpha": 0.08,
    "line_color": HUMAN_LQR_BOX_COLOR,
    "line_color_alpha": 0.5,
    "line_width": 1.0,
    "line_style": "-",
    "zorder": 1.8,
}


def _build_initial_ego_state(scene, frame_idx: int):
    """Build nuPlan's ``EgoState`` for a scene frame (mirrors
    ``NavSimScenario.get_ego_state_at_iteration``, which navsim itself uses to
    feed the PDM simulator/scorer)."""
    from nuplan.common.actor_state.ego_state import EgoState
    from nuplan.common.actor_state.state_representation import StateVector2D, TimePoint

    frame = scene.frames[frame_idx]
    return EgoState.build_from_rear_axle(
        StateSE2(*frame.ego_status.ego_pose),
        tire_steering_angle=0.0,
        vehicle_parameters=get_pacifica_parameters(),
        time_point=TimePoint(int(frame.timestamp)),
        rear_axle_velocity_2d=StateVector2D(*frame.ego_status.ego_velocity),
        rear_axle_acceleration_2d=StateVector2D(*frame.ego_status.ego_acceleration),
    )


def add_lqr_rollout_to_bev_ax(ax, scene, trajectory, box_config=LQR_BOX_CONFIG) -> None:
    """Overlay the 40 future ego bounding boxes from the LQR-tracked rollout.

    Reproduces ``navsim/evaluate/pdm_score.py::pdm_score``: converts a raw
    predicted/GT ``Trajectory`` (ego frame) to an absolute-frame reference
    trajectory, then re-simulates it through ``PDMSimulator`` (LQR tracker +
    kinematic bicycle model) exactly as the PDM scorer does before computing
    DAC/collision/TTC/etc, rather than plotting the raw (untracked) path.
    Works for both the agent's predicted trajectory and the human (GT)
    trajectory, since both are navsim ``Trajectory`` dataclasses.
    """
    from navsim.agents.transfuser.transfuser_features import (
        convert_absolute_to_relative_se2_array,
    )
    from navsim.evaluate.pdm_score import get_trajectory_as_array, transform_trajectory
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import StateIndex

    frame_idx = scene.scene_metadata.num_history_frames - 1
    initial_ego_state = _build_initial_ego_state(scene, frame_idx)
    origin = StateSE2(*scene.frames[frame_idx].ego_status.ego_pose)

    proposal_sampling = PROPOSAL_SAMPLING
    pred_trajectory = transform_trajectory(trajectory, initial_ego_state)
    pred_states = get_trajectory_as_array(pred_trajectory, proposal_sampling, initial_ego_state.time_point)

    simulator = PDMSimulator(proposal_sampling)
    simulated_states = simulator.simulate_proposals(pred_states[None, ...], initial_ego_state)[0]

    global_se2 = simulated_states[1:, StateIndex.STATE_SE2]  # (40, 3); skip t=0 (== current ego)
    local_se2 = convert_absolute_to_relative_se2_array(origin, global_se2)

    vehicle = get_pacifica_parameters()
    for x, y, heading in local_se2:
        box = OrientedBox(StateSE2(x, y, heading), vehicle.length, vehicle.width, vehicle.height)
        add_oriented_box_to_bev_ax(ax, box, box_config, add_heading=False)


# Ordered (label, PDMResults field) pairs for rendering PDMS subscores.
PDMS_FIELDS = [
    ("NC", "no_at_fault_collisions"),
    ("DAC", "drivable_area_compliance"),
    ("EP", "ego_progress"),
    ("TTC", "time_to_collision_within_bound"),
    ("Comf", "comfort"),
    ("DDC", "driving_direction_compliance"),
]


def evaluate_pdms(metric_cache_loader: MetricCacheLoader, token: str, agent_trajectory):
    """Run the actual PDM-Score evaluation for one token, if it's cached.

    Reproduces ``navsim/planning/script/run_pdm_score.py``'s per-token scoring
    step: loads the pre-built ``MetricCache`` (GT observation, centerline,
    route, drivable-area map -- see
    ``navsim/planning/metric_caching/metric_cache_processor.py``) and calls
    ``navsim.evaluate.pdm_score.pdm_score`` on the agent's predicted
    trajectory. Returns ``None`` (with a warning) if ``token`` has no cached
    metric cache entry, rather than building one on the fly (that requires
    the full nuPlan scenario/map pipeline and is what the offline caching
    step -- ``scripts_navsim/build_navsim_cache.sh`` et al. -- is for).
    :return: ``navsim.common.dataclasses.PDMResults`` or ``None``.
    """
    if token not in metric_cache_loader.metric_cache_paths:
        print(f"  [warn] token {token} not found in metric cache; skipping PDMS")
        return None

    metric_cache = metric_cache_loader.get_from_token(token)
    simulator = PDMSimulator(PROPOSAL_SAMPLING)
    scorer = PDMScorer(PROPOSAL_SAMPLING)
    return pdm_score(
        metric_cache=metric_cache,
        model_trajectory=agent_trajectory,
        future_sampling=PROPOSAL_SAMPLING,
        simulator=simulator,
        scorer=scorer,
    )


def draw_pdms_text_on_ax(ax, pdms_result) -> None:
    """Render the PDMS subscores + final score as a text box (bottom-left)."""
    lines = [f"PDMS: {pdms_result.score:.3f}"]
    lines += [f"  {label}: {getattr(pdms_result, field):.2f}" for label, field in PDMS_FIELDS]
    ax.text(
        0.02, 0.02, "\n".join(lines),
        transform=ax.transAxes, va="bottom", ha="left", fontsize=9,
        family="monospace", zorder=5,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )


def draw_bev_on_ax(
    ax,
    scene,
    agent,
    forward_m: float = BEV_FORWARD_M,
    backward_m: float = BEV_BACKWARD_M,
    lateral_m: float = BEV_LATERAL_M,
    detailed: bool = False,
):
    """Replicate ``plot_bev_with_agent`` onto a provided axes.

    Overrides the default (symmetric +/-32 m) BEV framing so that
    ``forward_m`` metres ahead of the ego are always visible, since CLAP-AD's
    predicted/GT paths can extend ~50 m ahead and the default view clips them.
    ``detailed=True`` additionally overlays the drivable-area boundary.
    :return: ``(human_trajectory, agent_trajectory)`` navsim ``Trajectory``
        dataclasses, so callers (e.g. PDMS scoring) can reuse them without an
        extra agent forward pass.
    """
    frame_idx = scene.scene_metadata.num_history_frames - 1
    human_trajectory = scene.get_future_trajectory()
    agent_trajectory = agent.compute_trajectory(scene.get_agent_input())
    add_configured_bev_on_ax(ax, scene.map_api, scene.frames[frame_idx])
    if detailed:
        origin = StateSE2(*scene.frames[frame_idx].ego_status.ego_pose)
        radius = max(forward_m, backward_m, lateral_m)
        add_drivable_area_to_bev_ax(ax, scene.map_api, origin, radius)
        add_lqr_rollout_to_bev_ax(ax, scene, agent_trajectory, LQR_BOX_CONFIG)
        add_lqr_rollout_to_bev_ax(ax, scene, human_trajectory, HUMAN_LQR_BOX_CONFIG)
    human_cfg = {**TRAJECTORY_CONFIG["human"],
                 "fill_color": HUMAN_COLOR, "line_color": HUMAN_COLOR}
    agent_cfg = {**TRAJECTORY_CONFIG["agent"],
                 "fill_color": TRAJ_COLOR, "line_color": TRAJ_COLOR}
    add_trajectory_to_bev_ax(ax, human_trajectory, human_cfg)
    add_trajectory_to_bev_ax(ax, agent_trajectory, agent_cfg)
    configure_bev_ax(ax)
    configure_ax(ax)
    # x forward, y sideways; ax.set_ylim controls the forward/backward extent.
    ax.set_ylim(-backward_m, forward_m)
    ax.set_xlim(lateral_m, -lateral_m)  # inverted: left is +y
    return human_trajectory, agent_trajectory


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
        choices=["cv", "clapad", "ltf"],
        help="Agent whose trajectory is drawn on the BEV. 'ltf' = latent TFv6 "
        "(plain 4s route, no CLAP-AD speed-scoring head/overlays).",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint .pth (or its directory) for --agent clapad/ltf "
        f"(default: {DEFAULT_CKPT!r} for clapad, {LTF_CKPT!r} for ltf).",
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
        "--detailed",
        action="store_true",
        help="Additionally overlay the drivable-area boundary (the region "
        "the PDM scorer's drivable-area-compliance metric checks ego against) "
        "on the BEV plot.",
    )
    parser.add_argument(
        "--metric-cache-path",
        default=None,
        help="Directory of a pre-built PDM metric cache (see "
        "scripts_navsim/build_navsim_cache.sh), used to actually run PDM-Score "
        "evaluation for each rendered token (default: "
        "$NAVSIM_EXP_ROOT/metric_cache_navtest).",
    )
    parser.add_argument(
        "--no-pdms",
        action="store_true",
        help="Skip running PDM-Score evaluation (no PDMS text on the BEV plot).",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output dir (default: /workspace/lead/outputs/navsim_viz/<timestamp>).",
    )
    args = parser.parse_args()
    if args.checkpoint is None:
        args.checkpoint = LTF_CKPT if args.agent == "ltf" else DEFAULT_CKPT

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
        or f"/workspace/lead/outputs/navsim_viz/{datetime.now():%Y.%m.%d.%H.%M.%S}"
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

    metric_cache_loader = None
    if not args.no_pdms:
        metric_cache_path = args.metric_cache_path or os.path.join(
            os.environ["NAVSIM_EXP_ROOT"], "metric_cache_navtest"
        )
        try:
            metric_cache_loader = MetricCacheLoader(Path(metric_cache_path))
            print(f"PDMS evaluation: ON ({len(metric_cache_loader)} cached tokens in {metric_cache_path})")
        except Exception as exc:
            print(f"  [warn] could not load metric cache at {metric_cache_path} ({exc}); skipping PDMS")

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

        _, agent_trajectory = draw_bev_on_ax(ax, scene, agent, detailed=args.detailed)
        pdms_result = None
        if metric_cache_loader is not None:
            pdms_result = evaluate_pdms(metric_cache_loader, token, agent_trajectory)
            if pdms_result is not None:
                draw_pdms_text_on_ax(ax, pdms_result)
                print(f"  PDMS[{token}] = {pdms_result.score:.3f} "
                      + " ".join(f"{label}={getattr(pdms_result, field):.2f}"
                                 for label, field in PDMS_FIELDS))
        agent_label = {"clapad": "CLAP-AD trajectory", "ltf": "LTF trajectory"}.get(
            args.agent, "Const-vel"
        )
        legend_handles = [
            Line2D([0], [0], color=HUMAN_COLOR, lw=2, marker="o",
                   markeredgecolor="black", label="Human (GT)"),
            Line2D([0], [0], color=TRAJ_COLOR, lw=2, marker="o",
                   markeredgecolor="black", label=agent_label),
        ]
        if args.detailed:
            legend_handles.append(
                Line2D([0], [0], color=DRIVABLE_AREA_COLOR, lw=1.5, ls="--",
                       label="Drivable area")
            )
            legend_handles.append(
                Line2D([0], [0], color=LQR_BOX_COLOR, lw=1.0, marker="s",
                       markersize=6, linestyle="none",
                       label="Agent LQR rollout (40 poses)")
            )
            legend_handles.append(
                Line2D([0], [0], color=HUMAN_LQR_BOX_COLOR, lw=1.0, marker="s",
                       markersize=6, linestyle="none",
                       label="Human LQR rollout (40 poses)")
            )
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
    PROJECT_DIR = "/workspace/lead"
    NAVSIM_ROOT = f"{PROJECT_DIR}/3rd_party/navsim_workspace"
    DEVKIT = f"{NAVSIM_ROOT}/navsimv1.1"

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["HYDRA_FULL_ERROR"] = "1"
    os.environ["LEAD_PROJECT_ROOT"] = PROJECT_DIR
    os.environ["NUPLAN_MAP_VERSION"] = "nuplan-maps-v1.0"
    os.environ["NUPLAN_MAPS_ROOT"] = f"{NAVSIM_ROOT}/dataset/maps"
    os.environ["NAVSIM_EXP_ROOT"] = f"{NAVSIM_ROOT}/exp"
    os.environ["NAVSIM_DEVKIT_ROOT"] = DEVKIT
    os.environ["OPENSCENE_DATA_ROOT"] = f"{NAVSIM_ROOT}/dataset"

    pythonpath = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = (
        f"{PROJECT_DIR}:{pythonpath}" if pythonpath else PROJECT_DIR
    )
    
    main()
