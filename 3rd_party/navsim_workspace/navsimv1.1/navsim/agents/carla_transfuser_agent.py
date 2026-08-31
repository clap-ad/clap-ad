import json
import logging
import os
import random
from tabnanny import check
from typing import List, Dict, Union

import cv2
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
import pytorch_lightning as pl

from navsim.agents.abstract_agent import AbstractAgent
from navsim.agents.transfuser.transfuser_config import TransfuserConfig
from navsim.agents.transfuser.transfuser_features import TransfuserFeatureBuilder, TransfuserTargetBuilder
from navsim.common.dataclasses import SensorConfig
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder

from lead.inference.open_loop_inference import OpenLoopInference as CarlaOpenLoopInference
from lead.inference.open_loop_inference import OpenLoopPrediction as CarlaOpenLoopPrediction
from lead.inference.config_open_loop import OpenLoopConfig as CarlaOpenLoopConfig
from lead.training.config_training import TrainingConfig as CarlaTrainingConfig
import numpy as np

logger =logging.getLogger(__name__)

class CarlaTransfuserAgent(AbstractAgent):
    """Agent interface for TransFuser baseline."""
    def __init__(
        self,
        config: TransfuserConfig,
        checkpoint_path: str
    ):
        """
        Initializes TransFuser agent.
        :param config: global config of TransFuser agent.
        :param checkpoint_path: optional path string to checkpoint, defaults to None.
        """
        super().__init__()

        self._config = config

        if checkpoint_path is not None:
            self._checkpoint_path = checkpoint_path.rsplit("/", 1)[0]
            self._model_filename = checkpoint_path.rsplit("/", 1)[1]

    def name(self) -> str:
        """Inherited, see superclass."""
        return self.__class__.__name__

    def initialize(self) -> None:
        """Inherited, see superclass."""
        with open(os.path.join(self._checkpoint_path, "config.json"), "r", encoding="utf-8") as f:
            json_config = json.load(f)
        self._carla_model_config = CarlaTrainingConfig(json_config)
        self._carla_config_open_loop = CarlaOpenLoopConfig()
        gpu_count = torch.cuda.device_count()
        if gpu_count > 1:
            idx = random.randint(0, gpu_count - 1)
            device = torch.device(f"cuda:{idx}")
        elif gpu_count == 1:
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
        if device.type == "cuda":
            # worker=single_machine_thread_pool runs each agent's initialize()
            # in its own host thread, but never calls torch.cuda.set_device()
            # for that thread. cuDNN/cuBLAS handles (used for the conv
            # backbone) are cached per-thread and keyed off the thread's
            # *current* CUDA device, not off the device of the tensors being
            # operated on. Without this call every new thread implicitly
            # defaults to "current device" cuda:0, so if this agent's model
            # actually lives on a different GPU (idx above), cuDNN can bind
            # its handle/workspace to the wrong device. This doesn't fail
            # immediately -- it silently corrupts device-bound state and
            # surfaces later as an asynchronous
            # "CUDA error: an illegal memory access was encountered", often
            # only after many scenarios have been processed. Pinning the
            # thread's current device here makes all implicit-current-device
            # CUDA state (cuDNN handles, RNG, etc.) consistent with `device`.
            torch.cuda.set_device(device)
        # Mixed precision training uses bfloat16; some GPUs (e.g. older
        # architectures) don't support it, which raises
        # "Current CUDA Device does not support bfloat16" from
        # torch.amp.autocast at inference time. Weights are fp32, so it's safe
        # to just disable mixed precision here rather than fail eval/vis runs.
        if (
            self._carla_model_config.use_mixed_precision_training
            and device.type == "cuda"
            and not torch.cuda.is_bf16_supported()
        ):
            self._carla_model_config.use_mixed_precision_training = False
        self._carla_open_loop_inference = CarlaOpenLoopInference(
            config_training=self._carla_model_config,
            config_open_loop=self._carla_config_open_loop,
            model_path=self._checkpoint_path,
            device=device,
            prefix=self._model_filename
        )

    def get_sensor_config(self) -> SensorConfig:
        """Inherited, see superclass."""
        return SensorConfig.build_all_sensors(include=[3])

    def get_target_builders(self) -> List[AbstractTargetBuilder]:
        """Inherited, see superclass."""
        return [TransfuserTargetBuilder(config=self._config)]

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        """Inherited, see superclass."""
        return [TransfuserFeatureBuilder(config=self._config)]

    def _get_rasterized_lidar(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Build the "rasterized_lidar" tensor expected by TransfuserBackbone.forward
        when config.LTF=False, from the raw "lidar_feature" produced by
        TransfuserFeatureBuilder._get_lidar_feature (a (C, X, Y) float BEV
        histogram, uncompressed -- unlike camera_feature this is not
        JPEG-encoded, so no cv2.imdecode is needed here).

        Mirrors the fix applied to lead.data_loader.navsim_dataset.NavsimData
        for the training-time cache: NAVSIM's splat_points() does NOT
        transpose its histogram (unlike CARLA's rasterize_lidar, which
        explicitly returns `overhead_splat.T`), so the array needs its last
        two axes swapped ((..., X, Y) -> (..., H, W)), followed by the same
        90-degree rotation used elsewhere ("Align NavSim BEV with CARLA BEV")
        so this matches the orientation/frame the model was trained on.

        NOTE: unlike a raw compute_features() output, `features` here has
        already gone through AbstractAgent.compute_trajectory()'s
        `{k: v.unsqueeze(0) for k, v in features.items()}`, so
        "lidar_feature" arrives as (1, C, X, Y), not (C, X, Y).
        """
        lidar_feature: torch.Tensor = features["lidar_feature"]
        lidar_feature = lidar_feature.numpy()
        if lidar_feature.ndim == 3:
            lidar_feature = lidar_feature[:, None]  # add channel dim -> (B, 1, X, Y)
        rasterized_lidar = np.swapaxes(lidar_feature, -2, -1)  # (B, C, X, Y) -> (B, C, H, W)
        rasterized_lidar = np.rot90(rasterized_lidar, 1, axes=(-2, -1)).copy()
        return torch.tensor(rasterized_lidar).float()  # (B, C, H, W)

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Inherited, see superclass."""
        
        rgb : torch.Tensor = features["camera_feature"]
        rgb = rgb.numpy()
        rgb = cv2.imdecode(np.frombuffer(rgb, np.uint8), cv2.IMREAD_COLOR)
        rgb = np.transpose(rgb, (2, 0, 1))  # HWC to CHW
        rgb = torch.tensor(rgb).unsqueeze(0).float()  # CHW to NCHW
        
        forward_data = {
            "rgb": rgb,
            "command": features["status_feature"][:, :4].reshape(-1, 4),
            "speed": torch.linalg.norm(features["status_feature"][:, 4:6]).reshape(-1, 1),
            "acceleration": torch.linalg.norm(features["status_feature"][:, 6:8]).reshape(-1, 1),
        }
        if not self._carla_model_config.LTF:
            forward_data["rasterized_lidar"] = self._get_rasterized_lidar(features)
        output: CarlaOpenLoopPrediction = self._carla_open_loop_inference.forward(forward_data)
        pred_clapad = output.pred_clapad
        if pred_clapad is None:
            future_waypoints = output.pred_future_waypoints.clone()
            # Model trained in mixed dataset, need to flip y axis
            future_waypoints[:, :, 1] = -future_waypoints[:, :, 1]
            future_headings = -output.pred_future_headings.clone()
        else:
            current_speed = features["status_feature"][:, 4:6].norm(dim=-1).item()
            pred_traj = traj_from_clapad(pred_clapad, current_speed=current_speed, speed_bins=self._carla_model_config.speed_bins)
            future_waypoints = pred_traj.unsqueeze(0)  # (1, 8, 2)
            # add (0,0) for current position to compute headings
            future_waypoints_pad = torch.cat([torch.zeros(1, 1, 2).to(future_waypoints.device), future_waypoints], dim=1)  # (1, 9, 2)
            # heading at waypoint t is the direction of the 2-segment fragment
            # spanning (t-1) -> (t+1), i.e. the average of the incoming and
            # outgoing segment directions (computed as a single vector
            # t-1 -> t+1 rather than averaging two angles, to avoid
            # wraparound issues near +-pi). The last waypoint has no outgoing
            # segment, so its incoming segment (t-1 -> t) is reused instead.
            
            # ######### Incoming ##########
            # # add (0,0) for current position to compute headings
            # future_waypoints_pad = torch.cat([torch.zeros(1, 1, 2).to(future_waypoints.device), future_waypoints], dim=1)  # (1, 9, 2)
            # # heading at waypoint t is the direction of the incoming segment
            # # spanning (t-1) -> t
            # future_headings = torch.atan2(
            #     future_waypoints_pad[:, 1:, 1] - future_waypoints_pad[:, :-1, 1],
            #     future_waypoints_pad[:, 1:, 0] - future_waypoints_pad[:, :-1, 0]
            # )
            
            
            ######### Average of incoming and outgoing ##########
            future_headings = torch.atan2(
                future_waypoints_pad[:, 2:, 1] - future_waypoints_pad[:, :-2, 1],
                future_waypoints_pad[:, 2:, 0] - future_waypoints_pad[:, :-2, 0]
            )
            last_heading = torch.atan2(
                future_waypoints_pad[:, -1, 1] - future_waypoints_pad[:, -2, 1],
                future_waypoints_pad[:, -1, 0] - future_waypoints_pad[:, -2, 0]
            ).unsqueeze(1)
            future_headings = torch.cat([future_headings, last_heading], dim=1)
            
            # manually tuned values on 4 tokens for the LQR rollout
            future_headings = future_headings * torch.tensor([3.0, 2.0, 0.8, 1.0, 0.8, 1.0, 1.0, 1.0], device=future_headings.device)


        return {
            "trajectory": torch.concatenate([future_waypoints, future_headings.unsqueeze(-1)], dim=-1).cpu()
        }

    def compute_loss(
        self,
        features: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        predictions: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        raise NotImplementedError("CARLA TransFuser supports only inference.")

    def get_optimizers(self) -> Union[Optimizer, Dict[str, Union[Optimizer, LRScheduler]]]:
        """Inherited, see superclass."""
        raise NotImplementedError("CARLA TransFuser supports only inference.")

    def get_training_callbacks(self) -> List[pl.Callback]:
        """Inherited, see superclass."""
        raise NotImplementedError("CARLA TransFuser supports only inference.")
    
    def get_pred_clap(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Get score preds for speed bins"""
        rgb : torch.Tensor = features["camera_feature"]
        rgb = rgb.numpy()
        rgb = cv2.imdecode(np.frombuffer(rgb, np.uint8), cv2.IMREAD_COLOR)
        rgb = np.transpose(rgb, (2, 0, 1))  # HWC to CHW
        rgb = torch.tensor(rgb).unsqueeze(0).float()  # CHW to NCHW
        
        forward_data = {
            "rgb": rgb,
            "command": features["status_feature"][:, :4].reshape(-1, 4),
            "speed": torch.linalg.norm(features["status_feature"][:, 4:6]).reshape(-1, 1),
            "acceleration": torch.linalg.norm(features["status_feature"][:, 6:8]).reshape(-1, 1),
        }
        if not self._carla_model_config.LTF:
            forward_data["rasterized_lidar"] = self._get_rasterized_lidar(features)
        output: CarlaOpenLoopPrediction = self._carla_open_loop_inference.forward(forward_data)
        pred_clapad = output.pred_clapad
        if pred_clapad is None:
            return None, None
        else:
            path = pred_clapad["sdc_traj"][0].clone()          # (T, 2), 1m spacing
            path[:, 1] = -path[:, 1]
            speed_logits = pred_clapad["speed_logits"][0,0]  # (M, S)
            coll_score = speed_logits[0]  # (S,)
            # Median speed bin with collision score > 0.3
            coll_score_sigmoid = torch.sigmoid(coll_score.float())
            return path, coll_score_sigmoid.cpu() # (S,)
    
    def compute_clap(self, agent_input):
        """
        Computes the ego vehicle trajectory.
        :param current_input: Dataclass with agent inputs.
        :return: Trajectory dataclass with predicted trajectory.
        """
        self.eval()
        features: Dict[str, torch.Tensor] = {}
        # build features
        for builder in self.get_feature_builders():
            features.update(builder.compute_features(agent_input))

        # add batch dimension
        features = {k: v.unsqueeze(0) for k, v in features.items()}

        # forward pass
        with torch.no_grad():
            path, speed_score = self.get_pred_clap(features)

        # extract trajectory
        return path, speed_score
            


def traj_from_clapad(pred_clapad: dict, current_speed: float, speed_bins):
    """
    Convert CLAPAD predictions to trajectory format.
    pred_clapad is a dict containing:
        - sdc_traj: (B, T, 2) tensor of predicted path in the SDC frame
        - speed_logits (B, NUM_PATHS, M, S) where
            - B is batch size
            - NUM_PATHS is the number of augmented paths (path[0] corresponds to sdc_traj)
            - M is safety criteria, S is the number of speed bins.
    The function implements the logic to look at the speed logits and pick appropriate for speed profile
    to travel along the predicted path. The output is a tensor representing temporal trajectory
    """

    path = pred_clapad["sdc_traj"][0].clone()          # (T, 2), 1m spacing
    logits = pred_clapad["speed_logits"][0,0] # (M, S)
    
    # flip y axis of path:
    path[:, 1] = -path[:, 1]
    # collision criterion assumed at index 0
    scores = torch.sigmoid(logits[0])

    # Option to discard some upper classes (e.g., highest speed bins)
    # Set `num_upper_classes_to_discard` to the number of upper classes to ignore
    num_upper_classes_to_discard = 0  # Change this value as needed
    if num_upper_classes_to_discard > 0:
        scores = scores[:-num_upper_classes_to_discard]
        speed_bins_used = speed_bins[:-num_upper_classes_to_discard]
    else:
        speed_bins_used = speed_bins

    valid_idxs = (scores > 0.8).nonzero(as_tuple=True)[0]
    if valid_idxs.numel() > 0:
        n = valid_idxs.numel()
        # If even, round up median to upper index
        median_idx = valid_idxs[(n - 1) // 2] if n % 2 == 1 else valid_idxs[n // 2]
        # Use max valid speed instead of median
        max_idx = valid_idxs[-1]  # Last index is the maximum
        percentile_80_idx = valid_idxs[int(0.8 * (n - 1))]  # 80th percentile index
        percentile_90_idx = valid_idxs[int(0.9 * (n - 1))]  # 90th percentile index
        second_to_last_idx = valid_idxs[-2] if n > 2 else valid_idxs[-1]  # Second to last index
        speed = speed_bins_used[percentile_80_idx]  # Use 80th percentile speed
    else:
        speed = speed_bins_used[0]  # default to lowest speed if no valid path

    # cumulative arc length (only consider x axis for forward movement)
    origin = torch.tensor([[0.0, 0.0]], device=path.device)
    if torch.norm(path[0]) > 1e-3:
        path = torch.cat([origin, path], dim=0)
    ds = torch.norm(path[1:] - path[:-1], dim=-1)
    s_path = torch.cat([torch.zeros(1, device=path.device), ds.cumsum(0)])

    # 4 sec @ 2Hz -> 8 future points
    # t = torch.arange(1, 9, device=path.device) * 0.5
    # speed_profile = current_speed + (speed - current_speed) * (t / t[-1])  # linear speed profile
    dt = 0.5
    max_accel = 2.0   # m/s^2
    max_decel = 4.0   # m/s^2
    num_steps = 8
    target_speed = speed.item()

    def simulate_speed_profile(start_speed: float, ramp_target: float) -> List[float]:
        profile = [start_speed]
        for _ in range(num_steps):
            current_v = profile[-1]
            delta = ramp_target - current_v
            # accelerate
            if delta > 0:
                next_v = current_v + min(delta, max_accel * dt)
            # brake
            else:
                next_v = current_v - min(-delta, max_decel * dt)
            next_v = max(next_v, 0.0)
            profile.append(next_v)
        return profile

    def profile_distance(profile: List[float]) -> float:
        return sum(0.5 * (profile[i] + profile[i + 1]) * dt for i in range(len(profile) - 1))

    # Ramping current_speed -> target_speed and then holding at target_speed
    # overshoots the distance a constant target_speed would cover over the
    # trajectory horizon (e.g. 4 m/s over 4s implies at most 16m) whenever
    # current_speed > target_speed, since speed stays above target_speed for
    # the whole braking ramp. To avoid overshooting, find the largest
    # "effective" ramp target (<= target_speed) whose resulting profile still
    # respects the distance budget, so the vehicle keeps braking below
    # target_speed by the end if that's what's needed to not overshoot.
    total_time = num_steps * dt
    distance_budget = target_speed * total_time
    speed_profile = simulate_speed_profile(current_speed, target_speed)
    if profile_distance(speed_profile) > distance_budget:
        lo, hi = 0.0, target_speed
        for _ in range(30):
            mid = 0.5 * (lo + hi)
            if profile_distance(simulate_speed_profile(current_speed, mid)) > distance_budget:
                hi = mid
            else:
                lo = mid
        speed_profile = simulate_speed_profile(current_speed, lo)

    speed_profile = torch.tensor(
        speed_profile,
        device=path.device,
        dtype=torch.float32
    )

    # trapezoidal integration
    s_query = torch.zeros_like(speed_profile)

    s_query[1:] = torch.cumsum(
        0.5 * (speed_profile[:-1] + speed_profile[1:]) * dt,
        dim=0
    )

    # remove t=0 state -> keep future trajectory only
    s_query = s_query[1:]

    # interpolate along path
    idx = torch.searchsorted(s_path, s_query).clamp(1, len(s_path)-1)

    s0, s1 = s_path[idx-1], s_path[idx]
    p0, p1 = path[idx-1], path[idx]

    alpha = ((s_query - s0) / (s1 - s0 + 1e-6)).unsqueeze(-1)

    traj = p0 + alpha * (p1 - p0)

    return traj   # (8, 2)