from typing import Any, Dict, List, Union, Tuple
from pathlib import Path
from dataclasses import asdict
from datetime import datetime
import traceback
import logging
import lzma
import pickle
import os
import uuid

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import pandas as pd

from nuplan.planning.script.builders.logging_builder import build_logger
from nuplan.planning.utils.multithreading.worker_pool import Task

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataloader import SceneLoader, SceneFilter, MetricCacheLoader
from navsim.common.dataclasses import SensorConfig
from navsim.evaluate.pdm_score import pdm_score
from navsim.planning.script.builders.worker_pool_builder import build_worker
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from navsim.planning.metric_caching.metric_cache import MetricCache
import numpy as np
import cv2

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/pdm_scoring"
CONFIG_NAME = "default_run_pdm_score"


def _balance_data_points_by_token_count(
    data_points: List[Dict[str, Any]], num_chunks: int
) -> List[List[Dict[str, Any]]]:
    """
    Groups per-log data points (each holding all scenario tokens of one log file) into
    `num_chunks` chunks, balanced by total token (scenario) count rather than by number of logs.

    `scene_loader.get_tokens_list_per_log()` yields one entry per log file, but the number of
    scenario tokens per log varies drastically (some logs yield only a handful of scenes after
    filtering, others yield hundreds). Naively splitting the list of per-log entries into
    equal-sized chunks (e.g. via `np.array_split`, as `worker_map`/`chunk_list` do) balances the
    number of *logs* per worker/rank, not the number of *scenarios* -- causing large workload
    imbalance across workers/ranks (e.g. one rank getting 1400 scenarios, another 2700).

    This uses a greedy longest-processing-time-first (LPT) bin-packing heuristic: logs are
    sorted by token count (descending) and each is assigned to the chunk with the currently
    smallest total token count.
    :param data_points: list of per-log dicts, each with a "tokens" list.
    :param num_chunks: number of chunks (i.e. worker.number_of_threads) to balance across.
    :return: list of chunks (each a list of per-log dicts), empty chunks omitted.
    """
    num_chunks = max(num_chunks, 1)
    sorted_data_points = sorted(data_points, key=lambda data_point: len(data_point["tokens"]), reverse=True)

    chunks: List[List[Dict[str, Any]]] = [[] for _ in range(num_chunks)]
    chunk_token_counts = [0] * num_chunks
    for data_point in sorted_data_points:
        target_chunk_idx = min(range(num_chunks), key=lambda i: chunk_token_counts[i])
        chunks[target_chunk_idx].append(data_point)
        chunk_token_counts[target_chunk_idx] += len(data_point["tokens"])

    return [chunk for chunk in chunks if len(chunk) > 0]


def run_pdm_score(args: List[Dict[str, Union[List[str], DictConfig]]]) -> List[Dict[str, Any]]:
    """
    Helper function to run PDMS evaluation in.
    :param args: input arguments
    """
    node_id = int(os.environ.get("NODE_RANK", 0))
    thread_id = str(uuid.uuid4())
    logger.info(f"Starting worker in thread_id={thread_id}, node_id={node_id}")

    log_names = [a["log_file"] for a in args]
    tokens = [t for a in args for t in a["tokens"]]
    cfg: DictConfig = args[0]["cfg"]

    simulator: PDMSimulator = instantiate(cfg.simulator)
    scorer: PDMScorer = instantiate(cfg.scorer)
    assert (
        simulator.proposal_sampling == scorer.proposal_sampling
    ), "Simulator and scorer proposal sampling has to be identical"
    agent: AbstractAgent = instantiate(cfg.agent)
    agent.initialize()

    metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))
    scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    scene_filter.log_names = log_names
    scene_filter.tokens = tokens
    scene_loader = SceneLoader(
        sensor_blobs_path=Path(cfg.sensor_blobs_path),
        data_path=Path(cfg.navsim_log_path),
        scene_filter=scene_filter,
        sensor_config=agent.get_sensor_config(),
    )

    tokens_to_evaluate = list(set(scene_loader.tokens) & set(metric_cache_loader.tokens))
    # tokens_to_evaluate = ['a4f4a240e54e56b2', 'f203a01d3a8055f1', '8b0d50e645eb5ac7', 'c35e290a671a5084', '555af87f71415a3d', '927aa2c8ef7d5b5c', '6fed9368351f54d5', 'c195a8cb7ccb59fd', '8580d0bf545b5261', '8013f74ddcf055db', '4bda94d100d65f0d', 'b8cf57333fc65efe', '7b7d556d563f5ac4', '932938d86cfc5d6f', '42dc421e37c154e5', '52094c2441105f4d', '73ce0b2145845c71', 'eb1315c4860c5189', '0bab0d24762a596d', '63ec67c4a58a5e5b']

    pdm_results: List[Dict[str, Any]] = []
    base_path_for_vis = "/workspace/temp_vis/navsim_traj"
    timestamp = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
    base_path_for_vis = os.path.join(base_path_for_vis, timestamp)
    os.makedirs(base_path_for_vis, exist_ok=True)
    for idx, (token) in enumerate(tokens_to_evaluate):
        logger.info(
            f"Processing scenario {idx + 1} / {len(tokens_to_evaluate)} in thread_id={thread_id}, node_id={node_id}"
        )
        score_row: Dict[str, Any] = {"token": token, "valid": True}
        try:
            metric_cache_path = metric_cache_loader.metric_cache_paths[token]
            with lzma.open(metric_cache_path, "rb") as f:
                metric_cache: MetricCache = pickle.load(f)

            agent_input = scene_loader.get_agent_input_from_token(token)
            if agent.requires_scene:
                scene = scene_loader.get_scene_from_token(token)
                trajectory = agent.compute_trajectory(agent_input, scene)
            else:
                trajectory = agent.compute_trajectory(agent_input)

            pdm_result = pdm_score(
                metric_cache=metric_cache,
                model_trajectory=trajectory,
                future_sampling=simulator.proposal_sampling,
                simulator=simulator,
                scorer=scorer,
            )
            debug = False
            if debug:
                scene = scene_loader.get_scene_from_token(token)
                vis(idx, agent, base_path_for_vis, scene, pdm_result)
            score_row.update(asdict(pdm_result))
        except Exception as e:
            logger.warning(f"----------- Agent failed for token {token}:")
            traceback.print_exc()
            score_row["valid"] = False

        pdm_results.append(score_row)
    return pdm_results

def vis(idx, agent, base_vis_path, scene, pdm_result):
    """Visualize the camera features and trajectory."""
    import matplotlib.pyplot as plt
    from navsim.visualization.plots import plot_bev_with_agent
    from navsim.agents.constant_velocity_agent import ConstantVelocityAgent
    import torch
    # bev:
    frame_idx = scene.scene_metadata.num_history_frames - 1 # current frame   
    fig, ax = plot_bev_with_agent(scene, agent)
    plt.show()
    fig_name = f"traj_vis_{idx}.png"
    plt.savefig(os.path.join(base_vis_path, fig_name))
    # cameras:
    from navsim.visualization.plots import plot_cameras_frame
    fig, ax = plot_cameras_frame(scene, frame_idx)
    plt.show()
    fig_name = f"cameras_vis_{idx}.png"
    plt.savefig(os.path.join(base_vis_path, fig_name))
    # traj on front img:
    pred_path, pred_score = agent.compute_clap(scene.get_agent_input())
    if pred_path is None:
        pixels_clap = None
    else:
        pred_path = pred_path.cpu().numpy()  # (T, 2)
        pred_score = pred_score.cpu().numpy()  # (S,)
        pixels_clap = traj_to_pixel(pred_path, scene.frames[frame_idx].cameras.cam_f0)
    pixels_clipped = get_pixels(scene, agent)
    ground_truth_clipped = get_pixels(scene, agent,cam='front',human=True)
    frame_idx = scene.scene_metadata.num_history_frames - 1
    cam = 'front'
    if cam == 'front':
        camera = scene.frames[frame_idx].cameras.cam_f0
    elif cam == 'left':
        camera = scene.frames[frame_idx].cameras.cam_l0
    else:
        camera = scene.frames[frame_idx].cameras.cam_r0
    image = camera.image
    plt.figure(figsize=(10, 10))
    gs = plt.GridSpec(3, 1, height_ratios=[3, 1, 0.5])

    # Top: Camera image with projected waypoints
    ax_img = plt.subplot(gs[0])
    ax_img.imshow(image, alpha=0.6)
    ax_img.scatter(pixels_clipped[:, 0], pixels_clipped[:, 1], c='red', s=40, marker='o', label='Pred Traj')
    ax_img.plot(pixels_clipped[:, 0], pixels_clipped[:, 1], color='red', linewidth=2, marker='o', markersize=5)
    if pixels_clap is not None:
        ax_img.scatter(pixels_clap[:, 0], pixels_clap[:, 1], c='blue', s=40, marker='x', label='CLAP Traj')
    if ground_truth_clipped is not None:
        ax_img.scatter(ground_truth_clipped[:, 0], ground_truth_clipped[:, 1], c='green', s=40, marker='o', label='GT Traj')
        ax_img.plot(ground_truth_clipped[:, 0], ground_truth_clipped[:, 1], color='green', linewidth=2, marker='o', markersize=5)
    ax_img.set_title("Projected Waypoints onto Camera Image")
    ax_img.axis("off")
    ax_img.legend()

    # Middle: Bar plot of predicted scores per speed bin
    if pred_score is not None:
        ax_score = plt.subplot(gs[1])
        speed_bins = torch.tensor([0, 2.5, 5, 7, 8.5, 10, 12], dtype=torch.float32)
        x = speed_bins.numpy()
        ax_score.bar(x, pred_score, width=0.8, align='center', color='skyblue', edgecolor='k')
        ax_score.set_xticks(x)
        ax_score.set_xlabel("Speed Bin Edge (m/s)")
        ax_score.set_ylabel("Score")
        ax_score.set_title("Predicted Scores per Speed Bin")
        ax_score.grid(True, axis='y')

    # Bottom: Subscores as text
    ax_text = plt.subplot(gs[2])
    ax_text.axis('off')
    # Format subscores as text
    subscore_text = (
        f"no_at_fault_collisions: {pdm_result.no_at_fault_collisions:.2f}\n"
        f"drivable_area_compliance: {pdm_result.drivable_area_compliance:.2f}\n"
        f"ego_progress: {pdm_result.ego_progress:.2f}\n"
        f"time_to_collision_within_bound: {pdm_result.time_to_collision_within_bound:.2f}\n"
        f"comfort: {pdm_result.comfort:.2f}\n"
        f"driving_direction_compliance: {pdm_result.driving_direction_compliance:.2f}\n"
        f"score: {pdm_result.score:.2f}"
    )
    ax_text.text(0.01, 0.5, subscore_text, fontsize=12, va='center', ha='left', family='monospace')

    plt.tight_layout()
    plt.savefig(os.path.join(base_vis_path, f"traj_and_scores_{idx}.png"))
    plt.show()
    plt.close("all")

# cast to the camera 
def get_pixels(scene, agent, cam='front', human=False):
    frame_idx = scene.scene_metadata.num_history_frames - 1
    if cam == 'front':
        camera = scene.frames[frame_idx].cameras.cam_f0
    elif cam == 'left':
        camera = scene.frames[frame_idx].cameras.cam_l0
    else:
        camera = scene.frames[frame_idx].cameras.cam_r0
    if human:
        agent_trajectory = scene.get_future_trajectory()
    else:
        agent_trajectory = agent.compute_trajectory(scene.get_agent_input())
    trajectory = agent_trajectory.poses
    pixels_clipped = traj_to_pixel(trajectory, camera)
    return pixels_clipped


def traj_to_pixel(trajectory, camera):
    # transformation matrices 
    R_c2l = camera.sensor2lidar_rotation        # camera-to-lidar (3x3)
    T_c2l = camera.sensor2lidar_translation     # camera-to-lidar (3,)
    intrinsics = camera.intrinsics
    # add z 
    trajectory_xyz = np.hstack([trajectory[:, :2], np.zeros((trajectory.shape[0], 1))])  # (N, 3)
    
    # lidar_to_camera = inverse of camera_to_lidar
    R_l2c = R_c2l.T                              # inverse of rotation
    T_l2c = -R_l2c @ T_c2l                       # inverse of translation
    
    # apply to points
    trajectory_cam = (R_l2c @ trajectory_xyz.T).T + T_l2c  # shape: (N, 3)
    in_front = trajectory_cam[:, 2] > 0
    points_cam = trajectory_cam[in_front]
    
    # project to image plane using intrinsics
    K = intrinsics  # 3x3
    projected = (K @ points_cam.T).T  # shape: (N, 3)
    
    # pixel coordinates
    pixels = projected[:, :2] / projected[:, 2:3]  # divide x and y by z
    # print(pixels)
    image = camera.image  # shape (H, W, 3), 
    img_h, img_w = image.shape[:2]
    valid = (
        (pixels[:, 0] >= 0) & (pixels[:, 0] < img_w) &
        (pixels[:, 1] >= 0) & (pixels[:, 1] < img_h)
    )
    pixels_clipped = pixels[valid]
    return pixels_clipped
    

@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entrypoint for running PDMS evaluation.
    :param cfg: omegaconf dictionary
    """

    build_logger(cfg)
    worker = build_worker(cfg)

    # Extract scenes based on scene-loader to know which tokens to distribute across workers
    # TODO: infer the tokens per log from metadata, to not have to load metric cache and scenes here
    scene_loader = SceneLoader(
        sensor_blobs_path=None,
        data_path=Path(cfg.navsim_log_path),
        scene_filter=instantiate(cfg.train_test_split.scene_filter),
        sensor_config=SensorConfig.build_no_sensors(),
    )
    metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))

    tokens_to_evaluate = list(set(scene_loader.tokens) & set(metric_cache_loader.tokens))
    num_missing_metric_cache_tokens = len(set(scene_loader.tokens) - set(metric_cache_loader.tokens))
    num_unused_metric_cache_tokens = len(set(metric_cache_loader.tokens) - set(scene_loader.tokens))
    if num_missing_metric_cache_tokens > 0:
        logger.warning(f"Missing metric cache for {num_missing_metric_cache_tokens} tokens. Skipping these tokens.")
    if num_unused_metric_cache_tokens > 0:
        logger.warning(f"Unused metric cache for {num_unused_metric_cache_tokens} tokens. Skipping these tokens.")
    logger.info("Starting pdm scoring of %s scenarios...", str(len(tokens_to_evaluate)))
    data_points = [
        {
            "cfg": cfg,
            "log_file": log_file,
            "tokens": tokens_list,
        }
        for log_file, tokens_list in scene_loader.get_tokens_list_per_log().items()
    ]
    data_point_chunks = _balance_data_points_by_token_count(data_points, worker.number_of_threads)
    scattered_score_rows = worker.map(Task(fn=run_pdm_score), data_point_chunks)
    score_rows: List[Tuple[Dict[str, Any], int, int]] = [
        score_row for score_rows in scattered_score_rows for score_row in score_rows
    ]

    pdm_score_df = pd.DataFrame(score_rows)
    num_sucessful_scenarios = pdm_score_df["valid"].sum()
    num_failed_scenarios = len(pdm_score_df) - num_sucessful_scenarios
    average_row = pdm_score_df.drop(columns=["token", "valid"]).mean(skipna=True)
    average_row["token"] = "average"
    average_row["valid"] = pdm_score_df["valid"].all()
    pdm_score_df.loc[len(pdm_score_df)] = average_row

    save_path = Path(cfg.output_dir)
    timestamp = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
    pdm_score_df.to_csv(save_path / f"{timestamp}.csv")

    logger.info(
        f"""
        Finished running evaluation.
            Number of successful scenarios: {num_sucessful_scenarios}.
            Number of failed scenarios: {num_failed_scenarios}.
            Final average score of valid results: {pdm_score_df['score'].mean()}.
            Results are stored in: {save_path / f"{timestamp}.csv"}.
        """
    )


if __name__ == "__main__":
    main()
