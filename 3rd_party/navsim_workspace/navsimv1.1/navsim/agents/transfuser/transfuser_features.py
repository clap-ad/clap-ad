from enum import IntEnum
from typing import Any, Dict, List, Tuple
import glob
import math
import os
import pickle
from collections import OrderedDict
import cv2
import numpy as np
import numpy.typing as npt

import torch
from torchvision import transforms

from shapely import affinity
from shapely.geometry import Polygon, LineString
from scipy.interpolate import CubicSpline

from nuplan.common.maps.abstract_map import AbstractMap, SemanticMapLayer, MapObject
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType

from navsim.agents.transfuser.transfuser_config import TransfuserConfig
from navsim.common.dataclasses import AgentInput, Scene, Annotations, Trajectory
from navsim.common.enums import BoundingBoxIndex, LidarIndex
from navsim.planning.scenario_builder.navsim_scenario_utils import tracked_object_types
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import (
    convert_absolute_to_relative_se2_array,
)
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from py123d.geometry.transform.transform_se2 import convert_relative_to_absolute_se2_array


# ---------------------------------------------------------------------------
# Full-future GT path support.
# The scene window only exposes ``num_future_frames`` future frames, which
# truncates the path-head GT (``get_future_path``) to ~20 m and forces the path
# loss to mask most of the 48-point prediction. We instead pull the FULL
# remaining future straight from the raw OpenScene log (adjacent frames beyond
# the window) so ``preprocess_path`` can fill the whole horizon without masking.
# Logs are cached per name to avoid repeated disk reads during caching.
# ---------------------------------------------------------------------------
_RAW_LOG_FRAMES_CACHE: "OrderedDict[str, Any]" = OrderedDict()
_RAW_LOG_CACHE_MAXSIZE = 8  # bound memory during full-split caching
_FUTURE_MODE = "raw_log"  # "const_vel"  # "raw_log" or "const_vel" for scene_info future positions


def _load_raw_log_frames(log_name: str):
    """Load (and LRU-cache) the raw frame list for ``log_name`` from OpenScene logs."""
    if log_name in _RAW_LOG_FRAMES_CACHE:
        _RAW_LOG_FRAMES_CACHE.move_to_end(log_name)
        return _RAW_LOG_FRAMES_CACHE[log_name]
    frames = None
    root = os.environ.get("OPENSCENE_DATA_ROOT")
    if root:
        matches = glob.glob(os.path.join(root, "navsim_logs", "*", f"{log_name}.pkl"))
        if matches:
            with open(matches[0], "rb") as handle:
                frames = pickle.load(handle)
    _RAW_LOG_FRAMES_CACHE[log_name] = frames
    if len(_RAW_LOG_FRAMES_CACHE) > _RAW_LOG_CACHE_MAXSIZE:
        _RAW_LOG_FRAMES_CACHE.popitem(last=False)
    return frames


def _yaw_from_quaternion(quat) -> float:
    """Yaw (rad) from a [w, x, y, z] quaternion."""
    w, x, y, z = quat
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _raw_future_annotations(scene, count: int):
    """Future (boxes, names, track_tokens, ego_pose) tuples per frame, read straight from
    the raw OpenScene log rather than the truncated scene window. This lets agent future
    positions be looked up directly instead of falling back to a constant-velocity guess
    when the requested horizon exceeds ``scene.frames``. Returns ``None`` if the raw log
    is unavailable, in which case callers should fall back to the windowed scene frames.
    """
    frames = _load_raw_log_frames(scene.scene_metadata.log_name)
    if not frames:
        return None
    cur_token = scene.scene_metadata.initial_token
    idx = next((k for k, f in enumerate(frames) if f["token"] == cur_token), None)
    if idx is None:
        return None

    future = []
    for f in frames[idx + 1: idx + 1 + count]:
        anns = f["anns"]
        ego_pose = np.array(
            [
                f["ego2global_translation"][0],
                f["ego2global_translation"][1],
                _yaw_from_quaternion(f["ego2global_rotation"]),
            ],
            dtype=np.float64,
        )
        # timestamp (microseconds) so callers can compute the REAL elapsed time
        # of each future sample instead of assuming a fixed cadence -- frames
        # are ~0.5s apart but not exactly, and the raw log can also have gaps
        # (dropped frames), see ``get_scene_score_info_from_scene``.
        future.append((anns["gt_boxes"], anns["gt_names"], anns["track_tokens"], ego_pose, f["timestamp"]))
    return future


def _full_future_global_poses(scene):
    """Global SE2 poses (N, 3) for every frame after the current one, taken from
    the raw log. Returns ``None`` if the log or current token cannot be resolved.
    """
    frames = _load_raw_log_frames(scene.scene_metadata.log_name)
    if not frames:
        return None
    cur_token = scene.scene_metadata.initial_token
    idx = next((k for k, f in enumerate(frames) if f["token"] == cur_token), None)
    if idx is None or idx + 1 >= len(frames):
        return None
    future = frames[idx + 1:]
    return np.array(
        [
            [
                f["ego2global_translation"][0],
                f["ego2global_translation"][1],
                _yaw_from_quaternion(f["ego2global_rotation"]),
            ]
            for f in future
        ],
        dtype=np.float64,
    )


class TransfuserFeatureBuilder(AbstractFeatureBuilder):
    """Input feature builder for TransFuser."""

    def __init__(self, config: TransfuserConfig):
        """
        Initializes feature builder.
        :param config: global config dataclass of TransFuser
        """
        self._config = config

    def get_unique_name(self) -> str:
        """Inherited, see superclass."""
        return "transfuser_feature"

    def compute_features(self, agent_input: AgentInput) -> Dict[str, torch.Tensor]:
        """Inherited, see superclass."""
        features = {}

        features["camera_feature"] = self._get_camera_feature(agent_input)
        features["lidar_feature"] = self._get_lidar_feature(agent_input)
        features["status_feature"] = torch.concatenate(
            [
                torch.tensor(agent_input.ego_statuses[-1].driving_command, dtype=torch.float32),
                torch.tensor(agent_input.ego_statuses[-1].ego_velocity, dtype=torch.float32),
                torch.tensor(agent_input.ego_statuses[-1].ego_acceleration, dtype=torch.float32),
            ],
        )

        return features

    def _get_camera_feature(self, agent_input: AgentInput) -> torch.Tensor:
        """
        Extract stitched camera from AgentInput
        :param agent_input: input dataclass
        :return: stitched front view image as torch tensor
        """

        cameras = agent_input.cameras[-1]
        
        l0 = cameras.cam_l0.image
        f0 = cameras.cam_f0.image
        r0 = cameras.cam_r0.image
        b0 = cameras.cam_b0.image
        
        stitched_image = np.concatenate([l0, f0, r0, b0], axis=1)
        resized_image = cv2.resize(stitched_image, (stitched_image.shape[1] // 4, stitched_image.shape[0] // 4))
        # Compression with JPEG quality 30
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 30]
        _, compressed_image = cv2.imencode('.jpg', resized_image, encode_param)
        return torch.from_numpy(compressed_image)

    def _get_lidar_feature(self, agent_input: AgentInput) -> torch.Tensor:
        """
        Compute LiDAR feature as 2D histogram, according to Transfuser
        :param agent_input: input dataclass
        :return: LiDAR histogram as torch tensors
        """

        # only consider (x,y,z) & swap axes for (N,3) numpy array
        lidar_pc = agent_input.lidars[-1].lidar_pc[LidarIndex.POSITION].T

        # NOTE: Code from
        # https://github.com/autonomousvision/carla_garage/blob/main/team_code/data.py#L873
        def splat_points(point_cloud):
            # 256 x 256 grid
            xbins = np.linspace(
                self._config.lidar_min_x,
                self._config.lidar_max_x,
                (self._config.lidar_max_x - self._config.lidar_min_x) * int(self._config.pixels_per_meter) + 1,
            )
            ybins = np.linspace(
                self._config.lidar_min_y,
                self._config.lidar_max_y,
                (self._config.lidar_max_y - self._config.lidar_min_y) * int(self._config.pixels_per_meter) + 1,
            )
            hist = np.histogramdd(point_cloud[:, :2], bins=(xbins, ybins))[0]
            hist[hist > self._config.hist_max_per_pixel] = self._config.hist_max_per_pixel
            overhead_splat = hist / self._config.hist_max_per_pixel
            return overhead_splat

        # Remove points above the vehicle
        lidar_pc = lidar_pc[lidar_pc[..., 2] < self._config.max_height_lidar]
        below = lidar_pc[lidar_pc[..., 2] <= self._config.lidar_split_height]
        above = lidar_pc[lidar_pc[..., 2] > self._config.lidar_split_height]
        above_features = splat_points(above)
        if self._config.use_ground_plane:
            below_features = splat_points(below)
            features = np.stack([below_features, above_features], axis=-1)
        else:
            features = np.stack([above_features], axis=-1)
        features = np.transpose(features, (2, 0, 1)).astype(np.float32)

        return torch.tensor(features)


class TransfuserTargetBuilder(AbstractTargetBuilder):
    """Output target builder for TransFuser."""

    def __init__(self, config: TransfuserConfig):
        """
        Initializes target builder.
        :param config: global config dataclass of TransFuser
        """
        self._config = config

    def get_unique_name(self) -> str:
        """Inherited, see superclass."""
        return "transfuser_target"

    def compute_targets(self, scene: Scene) -> Dict[str, torch.Tensor]:
        """Inherited, see superclass."""

        trajectory = torch.tensor(
            scene.get_future_trajectory(num_trajectory_frames=self._config.trajectory_sampling.num_poses).poses
        )
        frame_idx = scene.scene_metadata.num_history_frames - 1
        annotations = scene.frames[frame_idx].annotations
        ego_pose = StateSE2(*scene.frames[frame_idx].ego_status.ego_pose)

        agent_states, agent_labels = self._compute_agent_targets(annotations)
        bev_semantic_map = self._compute_bev_semantic_map(annotations, scene.map_api, ego_pose)
        path = torch.tensor(self.get_future_path(scene).poses)
        # Precompute the 1 m-spaced GT route so training can use it directly,
        # without re-fitting a spline on every __getitem__. NAVSIM always uses a
        # fixed 1 m spacing, starting from the ego origin -- NOT
        # ``config.target_first_distance`` (that field is CARLA-leaderboard-
        # scoped and unrelated to the NAVSIM route head).
        from lead.data_loader.navsim_dataset import preprocess_path
        route, route_mask = preprocess_path(
            np.asarray(path),
            sampling_distance=1.0,
            max_points=self._config.num_route_points_prediction,
        )
        scene_info = get_scene_score_info_from_scene(scene, future_mode=_FUTURE_MODE)

        return {
            "trajectory": trajectory,
            "agent_states": agent_states,
            "agent_labels": agent_labels,
            "bev_semantic_map": bev_semantic_map,
            "scene_info": scene_info,
            "path": path,
            "route": torch.tensor(route),
            "route_mask": torch.tensor(route_mask),
            # Marker for backward/forward compatibility: caches built before
            # 2026-08-10 stored agent "bbox" as [w * 2, l * 2] (accidentally
            # doubled full width/length). Caches built with this flag present
            # store the correct un-doubled [w, l]. Consumers (e.g.
            # ``lead.data_loader.navsim_dataset.preprocess_scene_info``) check
            # for this key's absence to auto-halve bbox values from old caches.
            "bbox_scale_fixed": True,
        }

    def _compute_agent_targets(self, annotations: Annotations) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extracts 2D agent bounding boxes in ego coordinates
        :param annotations: annotation dataclass
        :return: tuple of bounding box values and labels (binary)
        """

        max_agents = self._config.num_bounding_boxes
        agent_states_list: List[npt.NDArray[np.float32]] = []

        def _xy_in_lidar(x: float, y: float, config: TransfuserConfig) -> bool:
            return (config.lidar_min_x <= x <= config.lidar_max_x) and (config.lidar_min_y <= y <= config.lidar_max_y)

        for box, name in zip(annotations.boxes, annotations.names):
            box_x, box_y, box_heading, box_length, box_width = (
                box[BoundingBoxIndex.X],
                box[BoundingBoxIndex.Y],
                box[BoundingBoxIndex.HEADING],
                box[BoundingBoxIndex.LENGTH],
                box[BoundingBoxIndex.WIDTH],
            )

            if name == "vehicle" and _xy_in_lidar(box_x, box_y, self._config):
                agent_states_list.append(np.array([box_x, box_y, box_heading, box_length, box_width], dtype=np.float32))

        agents_states_arr = np.array(agent_states_list)

        # filter num_instances nearest
        agent_states = np.zeros((max_agents, BoundingBox2DIndex.size()), dtype=np.float32)
        agent_labels = np.zeros(max_agents, dtype=bool)

        if len(agents_states_arr) > 0:
            distances = np.linalg.norm(agents_states_arr[..., BoundingBox2DIndex.POINT], axis=-1)
            argsort = np.argsort(distances)[:max_agents]

            # filter detections
            agents_states_arr = agents_states_arr[argsort]
            agent_states[: len(agents_states_arr)] = agents_states_arr
            agent_labels[: len(agents_states_arr)] = True

        return torch.tensor(agent_states), torch.tensor(agent_labels)

    def _compute_bev_semantic_map(
        self, annotations: Annotations, map_api: AbstractMap, ego_pose: StateSE2
    ) -> torch.Tensor:
        """
        Creates sematic map in BEV
        :param annotations: annotation dataclass
        :param map_api: map interface of nuPlan
        :param ego_pose: ego pose in global frame
        :return: 2D torch tensor of semantic labels
        """

        bev_semantic_map = np.zeros(self._config.bev_semantic_frame, dtype=np.int64)
        for label, (entity_type, layers) in self._config.bev_semantic_classes.items():
            if entity_type == "polygon":
                entity_mask = self._compute_map_polygon_mask(map_api, ego_pose, layers)
            elif entity_type == "linestring":
                entity_mask = self._compute_map_linestring_mask(map_api, ego_pose, layers)
            else:
                entity_mask = self._compute_box_mask(annotations, layers)
            bev_semantic_map[entity_mask] = label

        return torch.Tensor(bev_semantic_map)

    def _compute_map_polygon_mask(
        self, map_api: AbstractMap, ego_pose: StateSE2, layers: List[SemanticMapLayer]
    ) -> npt.NDArray[np.bool_]:
        """
        Compute binary mask given a map layer class
        :param map_api: map interface of nuPlan
        :param ego_pose: ego pose in global frame
        :param layers: map layers
        :return: binary mask as numpy array
        """

        map_object_dict = map_api.get_proximal_map_objects(
            point=ego_pose.point, radius=self._config.bev_radius, layers=layers
        )
        map_polygon_mask = np.zeros(self._config.bev_semantic_frame[::-1], dtype=np.uint8)
        for layer in layers:
            for map_object in map_object_dict[layer]:
                polygon: Polygon = self._geometry_local_coords(map_object.polygon, ego_pose)
                exterior = np.array(polygon.exterior.coords).reshape((-1, 1, 2))
                exterior = self._coords_to_pixel(exterior)
                cv2.fillPoly(map_polygon_mask, [exterior], color=255)
        # OpenCV has origin on top-left corner
        map_polygon_mask = np.rot90(map_polygon_mask)[::-1]
        return map_polygon_mask > 0

    def _compute_map_linestring_mask(
        self, map_api: AbstractMap, ego_pose: StateSE2, layers: List[SemanticMapLayer]
    ) -> npt.NDArray[np.bool_]:
        """
        Compute binary of linestring given a map layer class
        :param map_api: map interface of nuPlan
        :param ego_pose: ego pose in global frame
        :param layers: map layers
        :return: binary mask as numpy array
        """
        map_object_dict = map_api.get_proximal_map_objects(
            point=ego_pose.point, radius=self._config.bev_radius, layers=layers
        )
        map_linestring_mask = np.zeros(self._config.bev_semantic_frame[::-1], dtype=np.uint8)
        for layer in layers:
            for map_object in map_object_dict[layer]:
                linestring: LineString = self._geometry_local_coords(map_object.baseline_path.linestring, ego_pose)
                points = np.array(linestring.coords).reshape((-1, 1, 2))
                points = self._coords_to_pixel(points)
                cv2.polylines(map_linestring_mask, [points], isClosed=False, color=255, thickness=2)
        # OpenCV has origin on top-left corner
        map_linestring_mask = np.rot90(map_linestring_mask)[::-1]
        return map_linestring_mask > 0

    def _compute_box_mask(self, annotations: Annotations, layers: TrackedObjectType) -> npt.NDArray[np.bool_]:
        """
        Compute binary of bounding boxes in BEV space
        :param annotations: annotation dataclass
        :param layers: bounding box labels to include
        :return: binary mask as numpy array
        """
        box_polygon_mask = np.zeros(self._config.bev_semantic_frame[::-1], dtype=np.uint8)
        for name_value, box_value in zip(annotations.names, annotations.boxes):
            agent_type = tracked_object_types[name_value]
            if agent_type in layers:
                # box_value = (x, y, z, length, width, height, yaw) TODO: add intenum
                x, y, heading = box_value[0], box_value[1], box_value[-1]
                box_length, box_width, box_height = box_value[3], box_value[4], box_value[5]
                agent_box = OrientedBox(StateSE2(x, y, heading), box_length, box_width, box_height)
                exterior = np.array(agent_box.geometry.exterior.coords).reshape((-1, 1, 2))
                exterior = self._coords_to_pixel(exterior)
                cv2.fillPoly(box_polygon_mask, [exterior], color=255)
        # OpenCV has origin on top-left corner
        box_polygon_mask = np.rot90(box_polygon_mask)[::-1]
        return box_polygon_mask > 0

    @staticmethod
    def _query_map_objects(
        self, map_api: AbstractMap, ego_pose: StateSE2, layers: List[SemanticMapLayer]
    ) -> List[MapObject]:
        """
        Queries map objects
        :param map_api: map interface of nuPlan
        :param ego_pose: ego pose in global frame
        :param layers: map layers
        :return: list of map objects
        """

        # query map api with interesting layers
        map_object_dict = map_api.get_proximal_map_objects(point=ego_pose.point, radius=self, layers=layers)
        map_objects: List[MapObject] = []
        for layer in layers:
            map_objects += map_object_dict[layer]
        return map_objects

    @staticmethod
    def _geometry_local_coords(geometry: Any, origin: StateSE2) -> Any:
        """
        Transform shapely geometry in local coordinates of origin.
        :param geometry: shapely geometry
        :param origin: pose dataclass
        :return: shapely geometry
        """

        a = np.cos(origin.heading)
        b = np.sin(origin.heading)
        d = -np.sin(origin.heading)
        e = np.cos(origin.heading)
        xoff = -origin.x
        yoff = -origin.y

        translated_geometry = affinity.affine_transform(geometry, [1, 0, 0, 1, xoff, yoff])
        rotated_geometry = affinity.affine_transform(translated_geometry, [a, b, d, e, 0, 0])

        return rotated_geometry

    def _coords_to_pixel(self, coords):
        """
        Transform local coordinates in pixel indices of BEV map
        :param coords: _description_
        :return: _description_
        """

        # NOTE: remove half in backward direction
        pixel_center = np.array([[self._config.bev_pixel_height / 2.0, self._config.bev_pixel_width / 2.0]])
        coords_idcs = (coords / self._config.bev_pixel_size) + pixel_center

        return coords_idcs.astype(np.int32)

    def get_future_path(self, scene):
        """
        Extracts the future path of the human operator until the end of the scenario.

        Captures all future ego poses after the current frame. When the raw log is
        available this uses the FULL remaining future (adjacent frames beyond the
        loaded scene window) so the path-head GT can fill the whole prediction
        horizon without masking; otherwise it falls back to the windowed scene
        frames.
        """
        start_frame_idx = scene.scene_metadata.num_history_frames - 1
        start_ego_pose = scene.frames[start_frame_idx].ego_status.ego_pose

        # Full remaining future from the raw log (preferred), else the window.
        future_poses = _full_future_global_poses(scene)
        if future_poses is None:
            future_poses = np.array(
                [
                    scene.frames[frame_idx].ego_status.ego_pose
                    for frame_idx in range(start_frame_idx + 1, len(scene.frames))
                ],
                dtype=np.float64,
            )

        local_poses = convert_absolute_to_relative_se2_array(
            StateSE2(*start_ego_pose), future_poses
        )

        return Trajectory(
            local_poses,
            TrajectorySampling(
                num_poses=len(local_poses),
                interval_length=1.0,  # interval_length is arbitrary here
            ),
        )
        
def get_scene_score_info_from_scene(scene, future_mode: str = "const_vel") -> list[dict]:
    """
    Extracts agent bounding boxes and their future waypoints for scene scoring.

    :param future_mode: how per-agent future positions are obtained:
        - "auto" (default): prefer the raw open-loop log (real recorded motion),
          falling back to the windowed ``scene.frames`` if the log is unavailable.
          Matches the original/backward-compatible behavior.
        - "raw_log": always use the raw log (raises if it cannot be resolved).
        - "window": always use the windowed ``scene.frames``, never the raw log.
        - "const_vel": ignore all future frames and always extrapolate every
          agent's future position with constant velocity. Useful to reproduce/
          inspect the legacy assumption this feature replaces.
    Returns a list of dicts, each containing:
        - type: str
        - p_o_x, p_o_y: current position
        - p_o_x_fut, p_o_y_fut: future position
        - vx, vy: velocity
        - yaw: heading
        - bbox: [width, height]
        - fut_wp: upsampled future waypoints (N, 2)
        - fut_wp_invalid: bool
        - corners: (4, 2) array of bbox corners
    """
    assert future_mode in ("auto", "raw_log", "window", "const_vel"), f"Unknown future_mode: {future_mode}"
    num_poses = 8
    
    frame_idx = scene.scene_metadata.num_history_frames - 1
    annotations = scene.frames[frame_idx].annotations
    boxes = annotations.boxes
    names = annotations.names
    track_tokens = annotations.track_tokens
    velocities = annotations.velocity_3d
    headings = [box[BoundingBoxIndex.HEADING] for box in boxes]
    lengths = [box[BoundingBoxIndex.LENGTH] for box in boxes]
    widths = [box[BoundingBoxIndex.WIDTH] for box in boxes]
    ttc_time_horizon = 4 # seconds

    # Prefer the FULL open-loop log for future agent poses (real recorded motion,
    # not a constant-velocity assumption) since NAVSIM predicts one path for the
    # whole 4s horizon rather than re-planning every 0.05s like closed-loop. Falls
    # back to the windowed scene frames if the raw log isn't available.
    if future_mode == "const_vel":
        future_frames = scene.frames[frame_idx + 1 : frame_idx + 1 + num_poses]
        using_raw_log = False
    else:
        raw_future = _raw_future_annotations(scene, num_poses) if future_mode != "window" else None
        if raw_future is None and future_mode == "raw_log":
            raise RuntimeError("future_mode='raw_log' requested but the raw log could not be resolved")
        if raw_future is not None:
            future_frames = raw_future
            using_raw_log = True
        else:
            future_frames = scene.frames[frame_idx + 1 : frame_idx + 1 + num_poses]
            using_raw_log = False
    start_ego_pos = scene.frames[frame_idx].ego_status.ego_pose[:3] #x,y,angle
    # Current-frame timestamp (microseconds), used to compute each future
    # sample's REAL elapsed time below (frames are only ~0.5s apart on
    # average, not exactly, and can have gaps/dropped detections).
    cur_timestamp = scene.frames[frame_idx].timestamp

    scene_score_info = []
    for idx, (box, name, track_token, velocity, yaw, l, w) in enumerate(zip(boxes, names, track_tokens, velocities, headings, lengths, widths)):
        if name != "vehicle":
            continue

        # Current position
        x, y = box[BoundingBoxIndex.X], box[BoundingBoxIndex.Y]
        vx, vy = velocity[0], velocity[1]

        # Gather future waypoints for this agent, paired with each sample's
        # REAL elapsed time (seconds since the current frame) so the spline
        # below is anchored correctly regardless of frame cadence/gaps. In
        # "const_vel" mode we deliberately skip reading ANY future frame data
        # (raw log or window) so the legacy constant-velocity extrapolation
        # below is always exercised, matching this mode's docstring ("ignore
        # all future frames and always extrapolate ..."). Previously this loop
        # still ran over `future_frames` in const_vel mode (identical to
        # "window"), silently making const_vel behave like window/raw_log
        # whenever the scene had enough future frames -- fixed here.
        fut_wp = []
        fut_dt = []
        if future_mode != "const_vel":
            for future_frame in future_frames:
                if using_raw_log:
                    f_boxes, f_names, f_tokens, ego_fut_pos, f_timestamp = future_frame
                else:
                    future_annotations = future_frame.annotations
                    f_boxes, f_names, f_tokens = (
                        future_annotations.boxes,
                        future_annotations.names,
                        future_annotations.track_tokens,
                    )
                    ego_fut_pos = future_frame.ego_status.ego_pose[:3]  # x,y,angle
                    f_timestamp = future_frame.timestamp
                for f_box, f_name, f_token in zip(f_boxes, f_names, f_tokens):
                    if f_name == "vehicle" and f_token == track_token:
                        # --- Step 1: agent pose in FUTURE EGO frame ---
                        agent_fut_relative = f_box[:3]

                        # --- Step 2: convert to GLOBAL frame ---
                        agent_global = convert_relative_to_absolute_se2_array(
                            ego_fut_pos,
                            agent_fut_relative
                        )

                        # --- Step 3: convert GLOBAL → CURRENT EGO frame ---
                        agent_relative_now = convert_absolute_to_relative_se2_array(
                            StateSE2(*start_ego_pos),
                            agent_global
                        )[0]

                        fut_wp.append(agent_relative_now[:2]) # x,y,angle
                        fut_dt.append((f_timestamp - cur_timestamp) / 1e6)  # microseconds -> seconds
                        break  # Only one match per frame

        fut_wp = np.array(fut_wp)
        fut_dt = np.array(fut_dt)
        # Future position (constant velocity fallback if no waypoints)
        if len(fut_wp) > 0:
            p_fut = fut_wp[-1]
        else:
            v_ego_x = np.cos(yaw) * vx - np.sin(yaw) * vy
            v_ego_y = np.sin(yaw) * vx + np.cos(yaw) * vy
            p_fut = np.array([x + v_ego_x * ttc_time_horizon, y + v_ego_y * ttc_time_horizon])

        # Upsample waypoints if possible. Anchor the spline at the agent's
        # CURRENT position (t=0), not just its future samples, and use each
        # sample's REAL elapsed time (``fut_dt``) rather than assuming a fixed
        # cadence spanning exactly [0, ttc_time_horizon]. Previously
        # ``t_old = np.linspace(0, ttc_time_horizon, len(fut_wp))`` silently
        # assumed the first collected future sample was at t=0 and the last
        # was at exactly t=4.0s, which is wrong: the first sample is really
        # ~0.5s out (frame cadence) and the last sample can land at whatever
        # time the raw log / window actually provides. That squeezed all
        # future motion earlier than it really happens, making e.g. the first
        # upsampled waypoint (nominally "t=0.1s") actually show where the
        # agent will be around real t~=0.5s -- a multi-meter error for fast
        # agents, which is exactly the "first point is further ahead" pattern
        # seen when visualizing.
        new_hz = 10
        old_hz = 2
        if len(fut_wp) < int(old_hz * ttc_time_horizon):
            fut_wp_ups = np.zeros((int(new_hz * ttc_time_horizon), 2))
            fut_wp_invalid = True
        else:
            t_old = np.concatenate(([0.0], fut_dt))
            wp_old = np.concatenate(([[x, y]], fut_wp), axis=0)
            t_new = np.linspace(0, ttc_time_horizon, int(new_hz * ttc_time_horizon) + 1)[1:]
            cs_x = CubicSpline(t_old, wp_old[:, 0])
            cs_y = CubicSpline(t_old, wp_old[:, 1])
            x_new = cs_x(t_new)
            y_new = cs_y(t_new)
            fut_wp_ups = np.stack([x_new, y_new], axis=1)
            fut_wp_invalid = False

        # Compute corners
        c, s = np.cos(yaw), np.sin(yaw)
        rot = np.array([[c, -s], [s, c]])
        half_w, half_l = w / 2, l / 2
        corners = np.array([
            [-half_l, -half_w],
            [ half_l, -half_w],
            [ half_l,  half_w],
            [-half_l,  half_w]
        ])
        corners = (corners @ rot.T) + np.array([x, y])

        obj_dict = {
            "type": name.lower(),
            "p_o_x": x,
            "p_o_y": y,
            "p_o_x_fut": p_fut[0],
            "p_o_y_fut": p_fut[1],
            "vx": vx,
            "vy": vy,
            "yaw": yaw,
            # NOTE: w, l are already FULL width/length (see ``half_w, half_l = w / 2, l / 2``
            # above), NOT half-extents -- do not double them here (that was a bug, fixed
            # 2026-08-10; see ``bbox_scale_fixed`` marker in ``TransfuserTargetBuilder.compute_targets``).
            "bbox": [w, l],
            "fut_wp": fut_wp_ups,
            "fut_wp_invalid": fut_wp_invalid,
            "corners": corners
        }
        scene_score_info.append(obj_dict)
    return scene_score_info

def get_traffic_rules(scene, frame_idx):
    """
    Extracts traffic rules from map API for a given frame in the scene.
    Returns a list of dicts, each containing:
    - type: str
    - long_dist: distance along the lane to the traffic rule
    - lat_dist: lateral distance from the lane center to the traffic rule
    - dist_to_ego: Euclidean distance from the ego to the traffic rule
    """
    map_api = scene.map_api
    ego_pose = StateSE2(*scene.frames[frame_idx].ego_status.ego_pose)
    layers_of_interest = [SemanticMapLayer.STOP_SIGN, SemanticMapLayer.TRAFFIC_LIGHT]
    raise NotImplementedError("Traffic rule extraction not fully implemented yet.")
    traffic_rules = []
    for layer in layers_of_interest:
        map_objects = map_api.get_proximal_map_objects(point=ego_pose.point, radius=100, layers=[layer])[layer]
        for obj in map_objects:
            if isinstance(obj.polygon, Polygon):
                geometry = TransfuserTargetBuilder._geometry_local_coords(obj.polygon, ego_pose)
            elif isinstance(obj.baseline_path.linestring, LineString):
                geometry = TransfuserTargetBuilder._geometry_local_coords(obj.baseline_path.linestring, ego_pose)
            else:
                continue
    
    return traffic_rules

def vis(scene, tag="", future_mode: str = "auto"):
    """Render BEV/camera/scene-score-info debug plots for ``scene`` into a
    timestamped folder. ``tag`` is an optional suffix (e.g. "raw_log" vs
    "const_vel") to tell apart multiple runs of the same scene. ``future_mode``
    is forwarded to ``get_scene_score_info_from_scene`` to control how agent
    future positions are obtained (see its docstring); defaults preserve
    backward-compatible behavior.
    """
    folder_path = "/workspace/temp_vis/navsim"
    # check exist or create it
    import os
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    vis_folder = os.path.join(folder_path, timestamp)
    os.makedirs(vis_folder, exist_ok=True)
    print(f"Visualization directory: {os.path.abspath(vis_folder)}")
    os.chdir(vis_folder)
    import matplotlib.pyplot as plt
    suffix = f"_{tag}" if tag else ""
    frame_idx = scene.scene_metadata.num_history_frames - 1 # current frame
    
    
    from navsim.visualization.plots import plot_bev_with_agent
    from navsim.agents.constant_velocity_agent import ConstantVelocityAgent
    agent = ConstantVelocityAgent()
    fig, ax = plot_bev_with_agent(scene, agent)
    plt.show()
    plt.savefig(f"bev_vis_agent{suffix}.png")
    from navsim.visualization.plots import plot_cameras_frame
    fig, ax = plot_cameras_frame(scene, frame_idx)
    plt.show()
    plt.savefig(f"cameras_vis{suffix}.png")
    
    # Scene_score_info
    scene_score_info = get_scene_score_info_from_scene(scene, future_mode=future_mode)

    figsize=(10, 10)
    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.get_cmap('tab10', len(scene_score_info))

    for idx, obj in enumerate(scene_score_info):
        x, y = obj["p_o_x"], obj["p_o_y"]
        corners = obj["corners"]
        fut_wp = obj["fut_wp"]
        fut_wp_invalid = obj["fut_wp_invalid"]
        label = obj["type"]

        # Plot agent center
        ax.plot(x, y, 'o', color=colors(idx), label=f'{label} center' if idx == 0 else None)
        ax.plot(obj["p_o_x_fut"], obj["p_o_y_fut"], 'o', color=colors(idx), label=f'{label} center' if idx == 0 else None)

        # Plot bounding box corners
        corners_closed = np.vstack([corners, corners[0]])  # close the box
        ax.plot(corners_closed[:, 0], corners_closed[:, 1], '-', color=colors(idx), lw=2, label=f'{label} bbox' if idx == 0 else None)

        # Plot upsampled future waypoints
        if not fut_wp_invalid and fut_wp.shape[0] > 0:
            ax.plot(fut_wp[:, 0], fut_wp[:, 1], 'x', color=colors(idx), label=f'{label} fut_wp' if idx == 0 else None)
        else:
            ax.plot([], [], ' ', label=f'{label} fut_wp INVALID')

    ax.set_aspect('equal')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'Scene Score Info Visualization{suffix}')
    ax.legend()
    plt.show()
    plt.savefig(f"Scene score info{suffix}.png")
    return vis_folder


def main():
    """Load one or more scenes and render ``vis()`` twice per scene: once using
    real future agent positions from the raw open-loop log (``future_mode=
    "raw_log"``), once using the legacy constant-velocity extrapolation
    (``future_mode="const_vel"``), so the two can be compared side by side to
    sanity-check correctness.
    """
    import argparse
    from pathlib import Path

    from navsim.common.dataclasses import SceneFilter, SensorConfig
    from navsim.common.dataloader import SceneLoader

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="test", help="OpenScene split folder name")
    parser.add_argument(
        "--tokens", nargs="+", default=None, help="Scene tokens to visualize (space-separated)"
    )
    parser.add_argument(
        "--num-scenes", type=int, default=1, help="Number of scenes to sample when --tokens is not given"
    )
    args = parser.parse_args()

    data_root = Path(os.environ["OPENSCENE_DATA_ROOT"])
    data_path = data_root / "navsim_logs" / args.split
    sensor_blobs_path = data_root / "sensor_blobs" / args.split

    scene_filter = SceneFilter(
        num_history_frames=4,
        num_future_frames=10,
        frame_interval=1,
        has_route=True,
        max_scenes=args.num_scenes if args.tokens is None else None,
        log_names=None,
        tokens=args.tokens,
    )
    scene_loader = SceneLoader(
        data_path=data_path,
        sensor_blobs_path=sensor_blobs_path,
        scene_filter=scene_filter,
        sensor_config=SensorConfig.build_all_sensors(include=[3]),
    )
    tokens = args.tokens or list(scene_loader.tokens)
    print(f"Visualizing {len(tokens)} token(s): {tokens}")

    for token in tokens:
        print(f"Visualizing token={token}")
        scene = scene_loader.get_scene_from_token(token)

        # Case 1: real future agent positions read straight from the raw log.
        vis(scene, tag=f"{token}_raw_log", future_mode="raw_log")

        # Case 2: legacy constant-velocity extrapolation from current velocity.
        vis(scene, tag=f"{token}_const_vel", future_mode="const_vel")


if __name__ == "__main__":
    main()



class BoundingBox2DIndex(IntEnum):
    """Intenum for bounding boxes in TransFuser."""

    _X = 0
    _Y = 1
    _HEADING = 2
    _LENGTH = 3
    _WIDTH = 4

    @classmethod
    def size(cls):
        valid_attributes = [
            attribute
            for attribute in dir(cls)
            if attribute.startswith("_") and not attribute.startswith("__") and not callable(getattr(cls, attribute))
        ]
        return len(valid_attributes)

    @classmethod
    @property
    def X(cls):
        return cls._X

    @classmethod
    @property
    def Y(cls):
        return cls._Y

    @classmethod
    @property
    def HEADING(cls):
        return cls._HEADING

    @classmethod
    @property
    def LENGTH(cls):
        return cls._LENGTH

    @classmethod
    @property
    def WIDTH(cls):
        return cls._WIDTH

    @classmethod
    @property
    def POINT(cls):
        # assumes X, Y have subsequent indices
        return slice(cls._X, cls._Y + 1)

    @classmethod
    @property
    def STATE_SE2(cls):
        # assumes X, Y, HEADING have subsequent indices
        return slice(cls._X, cls._HEADING + 1)
