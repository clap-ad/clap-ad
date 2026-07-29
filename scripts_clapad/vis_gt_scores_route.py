"""Offline visualization of the ground-truth speed scores for a whole CARLA route.

This mirrors what the training/inference visualizer produces on WandB, but runs
completely offline against a recorded dataset route. For every frame it loads the
RGB image, the BEV, the labels and the ground-truth speed-bin scores (TTC, traffic
light, stop sign, overtake) and renders them into a single frame, exactly the way
the training visualizer does.

Nothing is reimplemented here: the frame is produced by
``Visualizer.visualize_training_labels`` and the GT speed-score bars are produced
by ``Visualizer._gt_speed_logits_clap`` (which internally builds the training
``SpeedScoringLoss`` and draws with ``draw_speed_scores_append_bottom``). The only
thing this script "dummies" is the model prediction: the GT scores need a planned
path, so we feed the ego's recorded route (``data["route"]``) as the path and zero
speed logits -- precisely the model-free trick used by
``scripts_clapad/debug_gt_scores_pedestrian.py``.

A route directory is the folder that directly contains ``rgb/``, ``bboxes/``,
``metas/``, ``lidar/`` ... i.e. ``<dataset>/<ScenarioType>/<route>/``.

Run (from /workspace/lead, with the training PYTHONPATH set):

    python scripts_clapad/vis_gt_scores_route.py \
        --route /path/to/dataset/<ScenarioType>/<route> \
        --out   /workspace/temp_vis/<route>_gtscores
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import tempfile
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import cv2
import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate

from lead.common import common_utils
from lead.data_loader.carla_dataset import CARLAData
from lead.training.config_training import TrainingConfig
from lead.visualization.visualizer import Visualizer

# Number of GT score types produced by SpeedScoringLoss when the overtake score is
# enabled (TTC, traffic light, stop sign, overtake). This matches the config used
# inside Visualizer._gt_speed_logits_clap.
NUM_SCORE_TYPES = 4

# CARLA leaderboard simulation rate in Hz (leaderboard_evaluator.frame_rate). The
# dataset stores every `data_save_freq`-th tick, so the recorded frames are at
# CARLA_SIM_HZ / data_save_freq Hz (e.g. 20 / 5 = 4 Hz). Playing the video back at
# that rate reproduces real time.
CARLA_SIM_HZ = 20.0
FALLBACK_FPS = 4


def infer_fps(route_dir: str) -> int:
    """Derive the real playback fps from the route's recorded data_save_freq.

    Frames are saved every `data_save_freq` simulation ticks at CARLA_SIM_HZ, so the
    real-time frame rate is CARLA_SIM_HZ / data_save_freq. Falls back to FALLBACK_FPS
    if the info is unavailable.
    """
    metas = sorted(glob.glob(os.path.join(route_dir, "metas", "*.pkl")))
    if not metas:
        return FALLBACK_FPS
    try:
        meta = common_utils.read_pickle(metas[0])
        save_freq = meta.get("dataset_information", {}).get("data_save_freq")
        if save_freq:
            return max(1, round(CARLA_SIM_HZ / float(save_freq)))
    except Exception:
        pass
    return FALLBACK_FPS


def build_config(config_json: str | None) -> TrainingConfig:
    """Build a TrainingConfig suitable for offline dataset visualization."""
    config_dict = {}
    if config_json is not None and os.path.exists(config_json):
        with open(config_json, "r") as f:
            config_dict = json.load(f)
    config = TrainingConfig(loaded_config=config_dict, raise_error_on_missing_key=False)
    # Disable augmentation / route randomization and skip bucket-cache writing so we
    # deterministically render the recorded frames.
    config.visualize_dataset = True
    # Rebuild the bucket collection from the (single-route) root we pass in, instead
    # of loading the pre-built full-dataset cache. Combined with visualize_dataset
    # (which disables cache saving) this leaves the real cache untouched.
    config.force_rebuild_bucket = True
    return config


def make_single_route_dataset(route_dir: str, config: TrainingConfig):
    """Build a CARLAData that contains only the frames of ``route_dir``.

    The bucket collection expects ``root/<ScenarioType>/<route>/...``. To isolate a
    single route we create a throw-away scenario directory containing one symlink to
    the requested route and point the dataset at it.
    """
    route_dir = os.path.abspath(route_dir)
    if not os.path.isdir(os.path.join(route_dir, "bboxes")):
        raise SystemExit(
            f"'{route_dir}' does not look like a route directory "
            "(expected a 'bboxes/' subfolder).",
        )
    scenario_name = os.path.basename(os.path.dirname(route_dir))
    route_name = os.path.basename(route_dir)

    tmp_root = tempfile.mkdtemp(prefix="gt_scores_route_")
    scenario_dir = os.path.join(tmp_root, scenario_name or "scenario")
    os.makedirs(scenario_dir, exist_ok=True)
    os.symlink(route_dir, os.path.join(scenario_dir, route_name))

    dataset = CARLAData(root=[scenario_dir], config=config, random=False)
    return dataset, tmp_root


def ordered_indices(dataset) -> list[int]:
    """Return dataset indices sorted by frame number (chronological route order)."""
    order = []
    for i, meta in enumerate(dataset.metas):
        path = bytes(meta)
        stem = os.path.basename(path).split(b".")[0]
        try:
            frame_no = int(stem)
        except ValueError:
            frame_no = i
        order.append((frame_no, i))
    order.sort()
    return [i for _, i in order]


def crop_bottom_whitespace(image: np.ndarray, white_thresh: int = 250, margin: int = 12) -> np.ndarray:
    """Trim the trailing (bottom) near-white rows.

    The visualizer appends a fixed-size white meta panel, which leaves a large band
    of empty space at the bottom of the frame. Cropping the trailing white rows
    removes that band. The number of populated rows is constant across a route
    (same meta fields every frame), so the cropped height stays consistent.
    """
    non_white_rows = np.any(image < white_thresh, axis=(1, 2))
    if not non_white_rows.any():
        return image
    last = int(np.flatnonzero(non_white_rows)[-1])
    bottom = min(image.shape[0], last + 1 + margin)
    return np.ascontiguousarray(image[:bottom])


def reencode_h264(video_path: str, fps: int) -> None:
    """Re-encode a video to H.264, matching the closed-loop visualization pipeline.

    OpenCV writes ``mp4v`` (MPEG-4 Part 2), which many players/browsers refuse to
    load. The closed-loop VideoRecorder.compress_video re-encodes to libx264; we do
    the same here (with an even-dimension pad + yuv420p so it plays everywhere).
    The output framerate is pinned to ``fps`` (10 Hz for CARLA training data).
    """
    if not shutil.which("ffmpeg"):
        print("ffmpeg not found; leaving the raw mp4v video (may not play everywhere).")
        return
    temp_path = os.path.splitext(video_path)[0] + "_temp.mp4"
    command = (
        f"ffmpeg -i {video_path} -c:v libx264 -crf 23 -preset medium -r {fps} "
        f'-vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -pix_fmt yuv420p '
        f"-movflags +faststart -an {temp_path} -y"
    )
    if os.system(command) == 0 and os.path.exists(temp_path):
        os.replace(temp_path, video_path)
        print(f"Re-encoded video to H.264: {video_path}")
    else:
        print("ffmpeg re-encode failed; keeping the raw mp4v video.")


def render_frame(config: TrainingConfig, raw: dict) -> np.ndarray:
    """Render one frame (RGB + BEV + labels + GT speed scores) as an RGB image."""
    # scene_score_info is a plain dict of variable-length lists; keep it out of the
    # default collate (which would try to batch it) and re-attach it as-is so the
    # visualizer's [scene_score_info] wrapping yields the expected list-of-one.
    scene_score_info = raw.pop("scene_score_info", None)
    # The per-actor future trajectory keys drive an unrelated (and, for this data,
    # buggy) future-bounding-box overlay in Visualizer._bounding_boxes. They are not
    # needed for GT speed-score visualization, so we dummy them out.
    raw.pop("vehicles_future_waypoints", None)
    raw.pop("vehicles_future_yaws", None)
    batched: dict = default_collate([raw])
    if scene_score_info is not None:
        batched["scene_score_info"] = scene_score_info

    visualizer = Visualizer(
        config=config,
        data=batched,
        prediction=None,
        training=False,
    )

    can_score = (
        scene_score_info is not None
        and config.use_planning_decoder
        and config.use_clapad_head
        and batched.get("route") is not None
    )
    if can_score:
        speed_bins = visualizer._get_speed_bins()
        # Dummy prediction: use the recorded ego route as the planned path and zero
        # speed logits, so _gt_speed_logits_clap computes the GT scores model-free.
        gt_path = batched["route"].float()  # [1, T, 2]
        dummy_prediction = SimpleNamespace(
            pred_clapad={
                "sdc_traj": gt_path,
                "speed_logits": torch.zeros(1, 1, NUM_SCORE_TYPES, len(speed_bins)),
            },
        )
        visualizer.predictions = dummy_prediction  # type: ignore[assignment]
        # Sets visualizer.barplot_gt, which visualize_training_labels() appends.
        visualizer._gt_speed_logits_clap()
        # Restore the GT-only path (visualize_training_labels / _meta expect no
        # prediction); the computed bars live on visualizer.barplot_gt.
        visualizer.predictions = None

    return crop_bottom_whitespace(visualizer.visualize_training_labels())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--route",
        default="/data/datasets/AD_datasets/lead_carla/data/carla_leaderboard2/data/ConstructionObstacleTwoWays/Town12_Rep0_2259_0_route0_01_11_00_26_30/",
        help="Path to the route directory (contains rgb/, bboxes/, metas/, ...).",
    )
    parser.add_argument(
        "--out",
        default="/workspace/temp_vis/debug_gtscores",
        help="Output directory for the per-frame PNGs and the mp4 video.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional path to a training config.json to load the TrainingConfig from.",
    )
    # Playback fps defaults to the route's real recording rate (CARLA_SIM_HZ /
    # data_save_freq, e.g. 4 Hz). Pass --fps to override.
    parser.add_argument(
        "--fps",
        type=int,
        default=None,
        help="Video frame rate. Defaults to the route's recorded rate (real time).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional cap on the number of frames to render.",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Only write per-frame PNGs, skip the mp4.",
    )
    args = parser.parse_args()

    route_dir = os.path.abspath(args.route.rstrip("/"))
    route_name = os.path.basename(route_dir)
    out_dir = args.out or os.path.join("/workspace/temp_vis", f"{route_name}_gtscores")
    frames_dir = os.path.join(out_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    fps = args.fps if args.fps is not None else infer_fps(route_dir)
    print(f"Using {fps} fps (real-time recording rate).")

    config = build_config(args.config)
    dataset, tmp_root = make_single_route_dataset(route_dir, config)

    try:
        indices = ordered_indices(dataset)
        # Stop at the second-to-last frame: the very last frame of a route tends to
        # have incomplete/degenerate labels which break the GT scoring/visualization.
        if len(indices) > 1:
            indices = indices[:-1]
        if args.max_frames is not None:
            indices = indices[: args.max_frames]
        if not indices:
            raise SystemExit("No frames found for this route.")
        print(f"Rendering {len(indices)} frames for route '{route_name}'...")

        video_writer = None
        video_size = None
        video_path = None
        for count, idx in enumerate(indices):
            raw = dataset[idx]
            frame_no = raw.get("frame_number", str(count))
            image_rgb = render_frame(config, raw)
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

            png_path = os.path.join(frames_dir, f"{str(frame_no).zfill(5)}.png")
            cv2.imwrite(png_path, image_bgr)

            if not args.no_video:
                if video_writer is None:
                    video_size = (image_bgr.shape[1], image_bgr.shape[0])
                    video_path = os.path.join(out_dir, f"{route_name}_gtscores.mp4")
                    video_writer = cv2.VideoWriter(
                        video_path,
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        fps,
                        video_size,
                    )
                if (image_bgr.shape[1], image_bgr.shape[0]) != video_size:
                    image_bgr = cv2.resize(image_bgr, video_size)
                video_writer.write(image_bgr)

            print(f"  [{count + 1}/{len(indices)}] frame {frame_no}", flush=True)

        if video_writer is not None:
            video_writer.release()
            reencode_h264(video_path, fps)
            print(f"Saved video to {video_path}")
        print(f"Saved {len(indices)} frames to {frames_dir}")
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    main()
