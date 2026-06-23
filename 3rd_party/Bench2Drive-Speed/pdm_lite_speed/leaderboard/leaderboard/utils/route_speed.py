#!/usr/bin/env python
# Added for giving target speed

# import math
# import xml.etree.ElementTree as ET

import carla
# from agents.navigation.local_planner import RoadOption
# from srunner.scenarioconfigs.route_scenario_configuration import RouteSpeedScenarioConfiguration
# from srunner.scenarioconfigs.scenario_configuration import ScenarioConfiguration, ActorConfigurationData
from dataclasses import dataclass

@dataclass
class SpeedPlanNode:
    location: carla.Location
    target_speed: float
    index: int

def compute_keypoint_distances(keypoints):
    distances = [0.0]
    for i in range(1, len(keypoints)):
        distances.append(
            distances[-1] + keypoints[i].distance(keypoints[i - 1])
        )
    return distances

def build_speed_plan_from_xml(keypoints, speed_config):
    """
    Build speed plan strictly based on XML keypoints and relative speed config.
    """
    keypoint_distances = compute_keypoint_distances(keypoints)
    total_length = keypoint_distances[-1]

    # Default: None means "not assigned yet"
    target_speeds = [None] * len(keypoints)

    for seg in speed_config:
        start_d = seg["start"] * total_length
        end_d   = seg["end"]   * total_length
        v       = seg["target_speed"]

        for i, d in enumerate(keypoint_distances):
            if start_d <= d < end_d:
                target_speeds[i] = v

    # Fill uncovered points (inherit previous, or first valid)
    last_speed = None
    for i in range(len(target_speeds)):
        if target_speeds[i] is None:
            target_speeds[i] = last_speed
        else:
            last_speed = target_speeds[i]

    # Safety: if still None (e.g. no speed_config at all)
    if target_speeds[0] is None:
        raise RuntimeError("Speed config does not cover any keypoint.")

    return target_speeds

def generate_speed_plan(route_config):
    keypoints = route_config.keypoints
    speed_cfg = route_config.speed_config

    target_speeds = build_speed_plan_from_xml(keypoints, speed_cfg)

    speed_plan = []
    for i, (loc, v) in enumerate(zip(keypoints, target_speeds)):
        speed_plan.append(
            SpeedPlanNode(
                location=loc,
                target_speed=v,
                index=i
            )
        )

    return speed_plan
