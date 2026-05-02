from dataclasses import dataclass

import jaxtyping as jt
import numpy as np
import torch
from beartype import beartype

from lead.common.pid_controller import LateralPIDController, PIDController, get_throttle
from lead.expert.config_expert import ExpertConfig
from lead.inference.config_closed_loop import ClosedLoopConfig
from lead.inference.open_loop_inference import OpenLoopInference, OpenLoopPrediction
from lead.tfv6.tfv6 import Prediction
from lead.training.config_training import TrainingConfig

np.set_printoptions(suppress=True)


@dataclass
@beartype
class ClosedLoopPrediction(OpenLoopPrediction):
    steer: float
    throttle: float
    brake: float
    waypoints_steer: float
    waypoints_throttle: float
    waypoints_brake: float
    route_steer: float
    target_speed_throttle: float
    target_speed_brake: float


class ClosedLoopInference(OpenLoopInference):
    @beartype
    def __init__(
        self,
        config_training: TrainingConfig,
        config_closed_loop: ClosedLoopConfig,
        config_expert: ExpertConfig,
        model_path: str,
        device: torch.device,
        prefix: str,
    ):
        super().__init__(
            config_training=config_training,
            config_open_loop=config_closed_loop,
            model_path=model_path,
            device=device,
            prefix=prefix,
        )
        self.config_expert = config_expert
        self.config_training = config_training
        self.config_closed_loop = config_closed_loop

        self.lateral_waypoint_controller = PIDController(
            k_p=self.config_closed_loop.turn_kp,
            k_i=self.config_closed_loop.turn_ki,
            k_d=self.config_closed_loop.turn_kd,
            n=self.config_closed_loop.turn_n,
        )
        self.longitudinal_waypoint_controller = PIDController(
            k_p=self.config_closed_loop.speed_kp,
            k_i=self.config_closed_loop.speed_ki,
            k_d=self.config_closed_loop.speed_kd,
            n=self.config_closed_loop.speed_n,
        )
        self.lateral_route_controller = LateralPIDController(self.config_closed_loop)
        self.longitudinal_target_speed_controller = PIDController(
            k_p=self.config_closed_loop.speed_kp,
            k_i=self.config_closed_loop.speed_ki,
            k_d=self.config_closed_loop.speed_kd,
            n=self.config_closed_loop.speed_n,
        )

        self.step = 4  # Constant so produced images start with 5, not really important
        self.speed_bins=[0, 2.5, 5, 7, 8.5, 10, 12] # speed bins for the speed head
        self.stop_completed = False
        self.stop_wait_count = 0

    @jt.jaxtyped(typechecker=beartype)
    def execute_route_and_target_speed(
        self,
        pred_checkpoints: jt.Float[torch.Tensor, "1 num_checkpoints 2"],
        pred_target_speed: jt.Float[torch.Tensor, "1 1"],
        speed: jt.Float[torch.Tensor, "1 1"],
        ego_vehicle_location: float = 0.0,
        ego_vehicle_rotation: float = 0.0,
    ) -> tuple[float, float, float]:
        """Predicts vehicle control with a PID controller. Use route and target speed predictions.

        Args:
            pred_checkpoints: Predicted route checkpoints in ego-vehicle coordinates.
            pred_target_speed: Predicted target speed in m/s.
            speed: Current speed of the vehicle in m/s.
            ego_vehicle_location: Current lateral location of the ego vehicle.
            ego_vehicle_rotation: Current rotation of the ego vehicle.
        Returns:
            steer: Steering command in [-1, 1]
            throttle: Throttle command in [0, 1]
            brake: Brake command in [0, 1]
        """
        pred_checkpoints = pred_checkpoints[0].data.cpu().float().numpy()
        speed = float(speed[0].data.cpu().float().numpy())
        pred_target_speed = float(pred_target_speed[0].data.cpu().float().numpy())

        brake = bool(
            pred_target_speed < 0.01
            or (speed / pred_target_speed) > self.config_closed_loop.brake_ratio,
        )
        steer = self.lateral_route_controller.step(
            pred_checkpoints,
            speed,
            ego_vehicle_location,
            ego_vehicle_rotation,
            sensor_agent_steer_correction=self.config_closed_loop.sensor_agent_steer_correction,
        )
        throttle, brake = get_throttle(
            brake,
            pred_target_speed,
            speed,
            self.config_expert,
        )

        return steer, throttle, float(brake)

    @jt.jaxtyped(typechecker=beartype)
    def execute_waypoints(
        self,
        waypoints: jt.Float[torch.Tensor, "1 num_waypoints 2"],
        velocity: jt.Float[torch.Tensor, "1 1"],
    ) -> tuple[float, float, float]:
        """Predicts vehicle control with a PID controller. Use waypoint predictions.

        Args:
            waypoints: Predicted future waypoints in ego-vehicle coordinates.
            velocity: Current speed of the vehicle in m/s.
        Returns:
            steer: Steering command in [-1, 1]
            throttle: Throttle command in [0, 1]
            brake: Brake command in [0, 1]
        """
        waypoints = waypoints[0].data.cpu().float().numpy()
        speed = velocity[0].data.cpu().float().numpy()

        one_second = int(
            self.config_training.carla_fps // self.config_training.data_save_freq,
        )
        half_second = one_second // 2

        desired_speed = (
            np.linalg.norm(waypoints[half_second - 1] - waypoints[one_second - 1]) * 2.0
        )
        delta_speed = np.clip(
            desired_speed - speed,
            0.0,
            self.config_closed_loop.wp_delta_clip,
        )

        brake = (desired_speed < self.config_closed_loop.brake_speed) or (
            (speed / desired_speed) > self.config_closed_loop.brake_ratio
        )
        throttle = self.longitudinal_waypoint_controller.step(delta_speed)
        throttle = throttle if not brake else 0.0

        if (
            self.config_closed_loop.tuned_aim_distance
        ):  # In LB2, we go faster, so we need to choose waypoints farther ahead
            # range [2.4, 10.5] same as in the disentangled rep.
            aim_distance = np.clip(0.975532 * speed + 1.915288, 24, 105) / 10
        else:
            # To replicate the slow TransFuser behaviour we have a different distance
            # inside and outside of intersections (detected by desired_speed)
            if desired_speed < self.config_closed_loop.aim_distance_threshold:
                aim_distance = self.config_closed_loop.aim_distance_slow
            else:
                aim_distance = self.config_closed_loop.aim_distance_fast

        # We follow the waypoint that is at least a certain distance away
        aim_index = waypoints.shape[0] - 1
        for index, predicted_waypoint in enumerate(waypoints):
            if np.linalg.norm(predicted_waypoint) >= aim_distance:
                aim_index = index
                break

        aim = waypoints[aim_index]
        angle = np.degrees(np.arctan2(aim[1], aim[0])) / 90.0
        if speed < 0.01:
            # When we don't move we don't want the angle error to accumulate in the integral
            angle = 0.0
        if brake:
            angle = 0.0

        steer = self.lateral_waypoint_controller.step(angle)
        steer = np.clip(steer, -1.0, 1.0)  # Valid steering values are in [-1,1]
        return float(steer), float(throttle), float(brake)


    @jt.jaxtyped(typechecker=beartype)
    def execute_route_and_best_speed(
        self,
        pred_checkpoints: jt.Float[torch.Tensor, "1 num_checkpoints 2"],
        pred_speed_logits: jt.Float[torch.Tensor, "B N M S"],
        speed: jt.Float[torch.Tensor, "1 1"],
        ego_vehicle_location: float = 0.0,
        ego_vehicle_rotation: float = 0.0,
    ) -> tuple[float, float, float]:
        """Predicts vehicle control with a PID controller. Use route and speed scores.

        Args:
            pred_checkpoints: Predicted route checkpoints in ego-vehicle coordinates. [B, T, 2]
            pred_speed_logits: Predicted speed scores for pre-defined speed bins. [B, NUM_PATH, NUM_MODE, NUM_SPEED_BINS]
            speed: Current speed of the vehicle in m/s.
            ego_vehicle_location: Current lateral location of the ego vehicle.
            ego_vehicle_rotation: Current rotation of the ego vehicle.
        Returns:
            steer: Steering command in [-1, 1]
            throttle: Throttle command in [0, 1]
            brake: Brake command in [0, 1]
        """
        def get_desired_speed_from_scores(pred_speed_scores, threshold=0.3, accel_awareness=False, curr_speed=None):
            # Find bins where all criteria have non-zero scores
            valid_bins = np.all(pred_speed_scores > threshold, axis=0)  # shape: [S], True if all criteria > 0
            # Filter valid bins based on acceleration awareness
            if accel_awareness:
                assert curr_speed is not None, "Current speed must be provided for acceleration awareness"

                curr_speed_idx = np.searchsorted(self.speed_bins, curr_speed, side='right') - 1
                curr_speed_idx = np.clip(curr_speed_idx, 0, len(self.speed_bins) - 1)
                # For bins higher than the current speed, check that all bins between it and curr_speed_idx are also valid
                connected_bins = np.zeros_like(valid_bins)
                for i in range(len(valid_bins)):
                    if i > curr_speed_idx:
                        # accelerating checks for feasability
                        if np.all(valid_bins[curr_speed_idx + 1:i + 1]):
                            connected_bins[i] = True
                    else:
                        # braking is faster and safety critical
                        connected_bins[i] = valid_bins[i]
                valid_bins = connected_bins
            
            # Choose bin index
            if np.any(valid_bins):
                valid_candidates = np.where(valid_bins)[0]
                # best_speed_bin_idx = int(np.median(valid_candidates))
                # best_speed_bin_idx = int(np.max(valid_candidates))
                best_speed_bin_idx = int(np.percentile(valid_candidates, 50, interpolation='higher')) # like median but rounds up
            else:
                # Fallback: 0 m/s
                best_speed_bin_idx = 0
            target_speed = self.speed_bins[best_speed_bin_idx]
            return target_speed
        
        def stop_sign_controller():
            """Looks at pred scores to check whether a stop sign has been detected.
            It handles the temporal aspect of stop signs and overwrites predictions if necessary."""
            stop_scores = pred_speed_scores[2:3]

            # speed bin 2.5m/s is too noisy, therefore we ignore it in the stop sign controller
            # stop_scores[:, 1] = 0.0

            stop_target = get_desired_speed_from_scores(stop_scores, self.config_closed_loop.score_threshold)
            
            if stop_target > 0 and speed > 2.0:
                # If stop sign is not detected, reset stop completed status
                self.stop_completed = False
                self.stop_wait_count = 0
            if self.stop_completed:
                # overwrite scores when stop is completed
                pred_speed_scores[2:3] = np.ones_like(stop_scores)
                return pred_speed_scores
            
            if stop_target == 0 and speed < 0.01:
                # If stop sign is detected and we are stopped, wait 1 second (20 frames)
                self.stop_wait_count += 1
                if self.stop_wait_count >= self.config_closed_loop.stop_sign_wait_frames:
                    self.stop_completed = True
            else:
                self.stop_wait_count = 0

            return pred_speed_scores
        
        pred_checkpoints = pred_checkpoints[0].data.cpu().float().numpy()
        speed = float(speed[0].data.cpu().float().numpy())
        pred_speed_logits = pred_speed_logits.cpu().float().numpy()
        pred_speed_logits = pred_speed_logits[0][0] # .float()  # shape: [M, S], M=criteria, S=speed bins
        pred_speed_scores = 1 / (1 + np.exp(-pred_speed_logits))
        stop_sign_controller()
        pred_target_speed = get_desired_speed_from_scores(
            pred_speed_scores,
            threshold=self.config_closed_loop.score_threshold,
            accel_awareness=self.config_closed_loop.acceleration_aware_speed_selection,
            curr_speed=speed
        )


        brake = bool(
            pred_target_speed < 0.01
            or (speed / pred_target_speed) > self.config_closed_loop.brake_ratio
        )
        steer = self.lateral_route_controller.step(
            pred_checkpoints,
            speed,
            ego_vehicle_location,
            ego_vehicle_rotation,
            sensor_agent_steer_correction=self.config_closed_loop.sensor_agent_steer_correction,
        )
        pred_target_speed = float(pred_target_speed)
        throttle, brake = get_throttle(
            brake, pred_target_speed, speed, self.config_expert
        )

        return steer, throttle, float(brake)  

    @beartype
    def ensemble(
        self,
        data: dict[str, torch.Tensor],
        predictions: list[Prediction],
    ) -> ClosedLoopPrediction:
        """
        Args:
            data: Dictionary containing the input data
            predictions: List of dictionaries containing the predictions of each model
        Returns:
            ClosedLoopPrediction: Dataclass containing all predictions
        """
        ego_speed = data["speed"].to(self.device, dtype=torch.float32).unsqueeze(1)
        open_loop_prediction: OpenLoopPrediction = super().ensemble(data, predictions)

        # Convert high-level commands to vehicle controls
        steer = throttle = brake = waypoint_steer = waypoint_throttle = (
            waypoint_brake
        ) = route_steer = target_speed_throttle = target_speed_brake = None

        if open_loop_prediction.pred_route is not None:
            route_steer, target_speed_throttle, target_speed_brake = (
                self.execute_route_and_target_speed(
                    open_loop_prediction.pred_route,
                    open_loop_prediction.pred_target_speed_scalar,
                    ego_speed,
                )
            )
        if open_loop_prediction.pred_future_waypoints is not None:
            waypoints_steer, waypoints_throttle, waypoints_brake = (
                self.execute_waypoints(
                    open_loop_prediction.pred_future_waypoints,
                    ego_speed,
                )
            )
        
        if open_loop_prediction.pred_clapad is not None: # CLAP-AD control logic
            pred_route = open_loop_prediction.pred_clapad["sdc_traj"]
            speed_scores =  open_loop_prediction.pred_clapad["speed_logits"]
            route_steer, target_speed_throttle, target_speed_brake = (
                self.execute_route_and_best_speed(
                    pred_route,
                    speed_scores,
                    ego_speed,
                )
            )
            steer = route_steer
            throttle = target_speed_throttle
            brake = target_speed_brake
            open_loop_prediction.pred_route = pred_route # for later closed-loop post-processings

        else: # Normal TFV6 control logic
            # Select which high-level modality we want to use for each control
            if self.config_closed_loop.steer_modality == "route":
                steer = route_steer
            elif self.config_closed_loop.steer_modality == "waypoint":
                steer = waypoints_steer
            else:
                raise ValueError(
                    f"Invalid steer_modality: {self.config_closed_loop.steer_modality}"
                )

            if self.config_closed_loop.throttle_modality == "target_speed":
                throttle = target_speed_throttle
            elif self.config_closed_loop.throttle_modality == "waypoint":
                throttle = waypoints_throttle
            else:
                raise ValueError(
                    f"Invalid throttle_modality: {self.config_closed_loop.throttle_modality}"
                )

            if self.config_closed_loop.brake_modality == "target_speed":
                brake = target_speed_brake
            elif self.config_closed_loop.brake_modality == "waypoint":
                brake = waypoints_brake
            else:
                raise ValueError(
                    f"Invalid brake_modality: {self.config_closed_loop.brake_modality}"
                )

        # Turn off throttle if we brake
        if brake > 0.0:
            throttle = 0.0
            if (
                ego_speed < 0.01
            ):  # When we don't move we don't want the angle error to accumulate in the integral
                steer = 0.0

        # Return all predictions
        return ClosedLoopPrediction(
            **vars(open_loop_prediction),
            steer=steer,
            throttle=throttle,
            brake=brake,
            waypoints_steer=waypoint_steer,
            waypoints_throttle=waypoint_throttle,
            waypoints_brake=waypoint_brake,
            route_steer=route_steer,
            target_speed_throttle=target_speed_throttle,
            target_speed_brake=target_speed_brake,
        )
