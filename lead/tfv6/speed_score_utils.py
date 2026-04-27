
import math
import numpy as np
import torch


def _decel_to_score(required_decel, decel_min, decel_max):
    """Map required deceleration (m/s^2) to a [0,1] score.

    Score = 1 when required_decel <= decel_min
    Score = 0 when required_decel >= decel_max
    Linear in between.
    """
    if decel_max <= decel_min:
        raise ValueError(f"decel_max must be > decel_min (got {decel_min}..{decel_max})")
    score = (decel_max - required_decel) / (decel_max - decel_min)
    return float(np.clip(score, 0.0, 1.0))


def _required_decel_to_stop(speed, dist_to_trig, stop_distance_from_trigger):
    """Required constant deceleration magnitude to stop by stop_distance_from_trigger.

    dist_to_trig: distance (m) from ego to trigger volume.
    stop_distance_from_trigger: desired distance (m) from trigger volume at which speed should be 0.
    """
    dist_to_stop = float(dist_to_trig) - float(stop_distance_from_trigger)
    if dist_to_stop <= 0.0:
        return float("inf")
    return (float(speed) * float(speed)) / (2.0 * dist_to_stop)

def _get_bbox_distance(ego_x, ego_y, ego_w, ego_l, obj_x, obj_y, obj_w, obj_l):
    """
    Calculates the shortest distance between two axis-aligned bounding boxes.
    Assumes x,y is the center of the box.

    Because in Lead, x is forward and y is right, we treat ego_w as the width along the y-axis and ego_l as the length along the x-axis.
    """
    # Calculate the distance between edges on the X axis
    # dx = max(0, abs(ego_x - obj_x) - (ego_w + obj_w) / 2)
    dx = max(0, abs(ego_x - obj_x) - (ego_l + obj_l) / 2)
    
    # Calculate the distance between edges on the Y axis
    # dy = max(0, abs(ego_y - obj_y) - (ego_l + obj_l) / 2)
    dy = max(0, abs(ego_y - obj_y) - (ego_w + obj_w) / 2)
    
    # If both dx and dy are 0, the boxes are colliding
    return math.sqrt(dx**2 + dy**2)

def _compute_swept_bbox_distance(p_e_now, p_e_future, p_o_now, p_o_future, ego_wl, obj_wl, steps=5):
    """
    Interpolates between current and future positions to find the minimum 
    distance reached during the time interval, preventing "teleportation" 
    at high speeds.
    """
    min_dist = float('inf')
    # Linear interpolation: pos = start + alpha * (end - start)
    # We use a loop to check the 'swept' area
    for i in range(steps + 1):
        alpha = i / steps
        
        # Interpolate positions
        curr_e = p_e_now + alpha * (p_e_future - p_e_now)
        curr_o = p_o_now + alpha * (p_o_future - p_o_now)
        
        # Calculate distance at this specific sub-step
        d = _get_bbox_distance(
            curr_e[0], curr_e[1], ego_wl[0], ego_wl[1],
            curr_o[0], curr_o[1], obj_wl[0], obj_wl[1]
        )
        
        if d < min_dist:
            min_dist = d
            
        # Optimization: If we hit something, distance is 0, no need to keep checking
        if min_dist <= 0:
            return 0.0

    return min_dist

def _compute_ttc_for_bin(
        speed, path_pred, scene_score_info,
        time_horizon, safe_distance,
        coll_radius, ego_wl
    ):
    """
    Batched version: path_pred [B, T, 2], scene_score_info is list of B dicts.
    Returns: torch.Tensor of shape [B]
    """
    B, T, _ = path_pred.shape
    device = path_pred.device
    results = []
    for b in range(B):
        # Ego position and velocity (2D)
        p_e = np.array([0.0, 0.0])  # Assume ego starts at origin

        # Predict ego position after TIME_HORIZON on the given path
        target_dist = speed * time_horizon
        cum_dist = np.arange(T)
        p_e_future = np.array([
            np.interp(target_dist, cum_dist, path_pred[b, :, 0].cpu().numpy()),
            np.interp(target_dist, cum_dist, path_pred[b, :, 1].cpu().numpy())
        ])

        threat_actor = None
        min_distance = safe_distance
        for actor in scene_score_info[b]["actors"]:
            if not actor["ttc_cal"]:
                continue
            if actor["p_o_x_fut"] is None:
                continue
            p_o = np.array([actor["p_o_x"], actor["p_o_y"]])
            p_o_future = np.array([actor["p_o_x_fut"], actor["p_o_y_fut"]])
            # Convert bbox to list of two numpy floats if needed
            obj_wl = [float(x) for x in actor["bbox"]]
            dist_future = _compute_swept_bbox_distance(
                p_e_now=p_e,
                p_e_future=p_e_future,
                p_o_now=p_o,
                p_o_future=p_o_future,
                ego_wl=ego_wl,
                obj_wl=obj_wl,
                steps=20
            )
            if dist_future < min_distance:
                threat_actor = actor
                min_distance = dist_future
        if min_distance >= safe_distance:
            results.append(1.0)
        elif min_distance <= coll_radius:
            results.append(0.0)
        else:
            ttc = (min_distance - coll_radius) / (safe_distance - coll_radius)
            results.append(ttc)
    return torch.tensor(results, device=device)

def _compute_ttc_for_bin_dep(
        speed, path_pred, scene_score_info,
        time_horizon, safe_distance,
        coll_radius, ego_wl
    ):
    """
    It calculates TTC for a given speed and path. 
    Basically, it predicts where the ego vehicle would be after TIME_HORIZON
    """
    # Ego position and velocity (2D)
    # p_e = np.array([scene_score_info["p_e_x"], scene_score_info["p_e_y"]])
    p_e = np.array([0.0, 0.0])  # Assume ego starts at origin

    # Predict ego position after TIME_HORIZON on the given path
    # Here we simply move along the path by speed * time_horizon
    target_dist = speed * time_horizon
    cum_dist = np.arange(len(path_pred))
    p_e_future = np.array([
        np.interp(target_dist, cum_dist, path_pred[:, 0]),
        np.interp(target_dist, cum_dist, path_pred[:, 1])
    ])

    threat_actor = None
    min_distance = safe_distance
    for actor in scene_score_info["actors"]:
        if not actor["ttc_cal"]:
            continue
        if actor["p_o_x_fut"] is None: # has disappeared
            continue
        # To calculate TTC, we can either use constant velocity model or look at future position
        p_o = np.array([actor["p_o_x"], actor["p_o_y"]])
        p_o_future = np.array([actor["p_o_x_fut"], actor["p_o_y_fut"]]) # In 0.5s
        # Compute distance at future time
        # dist_future = np.linalg.norm(p_e_future - p_o_future)
        # dist_future = _get_bbox_distance(
        #     p_e_future[0], p_e_future[1], self.ego_wl[0], self.ego_wl[1],
        #     p_o_future[0], p_o_future[1], actor["bbox"][0], actor["bbox"][1]
        # )
        dist_future = _compute_swept_bbox_distance(
            p_e_now=p_e,
            p_e_future=p_e_future,
            p_o_now=p_o,
            p_o_future=p_o_future,
            ego_wl= ego_wl,
            obj_wl=actor["bbox"]
        )
        if dist_future < min_distance:
            threat_actor = actor
            min_distance = dist_future
    if min_distance >= safe_distance:
        return torch.tensor(1.0)  # Safe
    elif min_distance <= coll_radius:
        # print("Collision imminent: {} - {}".format(threat_actor, min_distance))
        return torch.tensor(0.0)  # Collision
    else:
        # Linear scaling between coll_radius and safe_distance
        # print("Collision warning: {} - {}".format(threat_actor, min_distance))
        ttc = (min_distance - coll_radius) / (safe_distance - coll_radius)
        return torch.tensor(ttc)
    
def compute_ttc(path_pred, scene_score_info, speed_bins,
                time_horizon, safe_distance, coll_radius,
                ego_wl
    ):
    """
    Compute time-to-collision (TTC) for self.speed_bins based on scene information.
    
    Args:
        scene_info: Tensor of shape [B] - scene information for the current step
    """
    # Compute TTC for each speed bin based on scene information
    ttc = []
    for speed in speed_bins:
        # Compute TTC for this bin edge
        speed_score = _compute_ttc_for_bin(
            speed=speed,
            path_pred=path_pred,
            scene_score_info=scene_score_info,
            time_horizon=time_horizon,
            safe_distance=safe_distance,
            coll_radius=coll_radius,
            ego_wl=ego_wl,
        )
        ttc.append(speed_score)
    # return torch.stack(ttc).unsqueeze(0)  # Shape: [B, num_classes]
    return torch.stack(ttc, dim=0).T  # Shape: [B, num_bins]

def compute_comfort(speed, speed_bins):
    """
    Compute comfort scores based on speed.
    Here we use a simple quadratic penalty for high speeds.
    
    Args:
        speed: Tensor of shape [B] - current speed in m/s
    Returns:
        comfort_scores = torch.clamp(1.0 - (speed / self.speed_bins[-1])**2, min=0.0, max=1.0)
    """
    # [num_bins]
    bins = torch.tensor(
        speed_bins,
        dtype=speed.dtype
    )
    # [B, num_bins]
    diff = speed.view(1,1)- bins.unsqueeze(0)
    # bandwidth controls sharpness (¼ of bin range is a good default)
    sigma = 0.25 * bins[-1]
    comfort_scores = torch.exp(-(diff ** 2) / (2 * sigma ** 2))
    # normalize → proper distribution
    comfort_scores = comfort_scores / comfort_scores.sum(dim=1, keepdim=True)
    return comfort_scores

def compute_feasibility(path, speed, speed_bins):
    """
    Compute feasibility scores based on path curvature and current speed.
    """
    feasibility_scores = []
    for bin_speed in speed_bins:
        # Simple heuristic: higher speeds get lower feasibility if path is curvy
        curvature = torch.mean(torch.abs(path[:, 1:] - path[:, :-1]))  # Simplified curvature measure
        score = torch.clamp(1.0 - (curvature * bin_speed / 10.0), min=0.0, max=1.0)
        feasibility_scores.append(score)
    return torch.stack(feasibility_scores, dim=0).unsqueeze(0)

def _stop_sign_score(speed, distance, speed_bins, proximity_threshold, speed_threshold):
    """
    This is called when already know there is a stop sign ahead at certain distance.
    The logic is:
        - If Stop sign affects Ego (within proximity threshold)
        - If Ego is affected check:
            - If current speed > speed_threshold:
                - Check if stop_completed:
                    - If yes (Ego has already stopped before): All get score 1.0
                    - Else (Ego is still approaching):
                        - All get score 0 except speed bin = 0.0: score = 1.0
            - Else (Ego is already stopped or nearly stopped):
                - All get score 1.0
    TODO: The logic the broken with usage of stop_completed flag
    """
    if distance > proximity_threshold:
        # Stop sign too far, no effect
        return [1.0 for _ in speed_bins]
    scores = []
    if speed > speed_threshold:
        # Ego is still moving
        if stop_completed:
            # Already stopped before, now can go
            scores = [1.0 for _ in speed_bins]
        else:
            # Still approaching stop sign, only 0 speed bin gets score 1.0
            for bin_speed in speed_bins:
                if bin_speed < 1e-2:
                    scores.append(1.0)
                else:
                    # Incentivize smaller speeds to stop gradually
                    score = max(0.0, 1.0 - bin_speed / max(speed_bins))
                    scores.append(score)
    else:
        # Ego is already stopped or nearly stopped
        scores = [1.0 for _ in speed_bins]
        stop_completed = True  # Mark that stop has been completed
    return scores

def compute_traffic_light_adherence(
    speed_bins,
    scene_score_info,
    stop_distance_from_trigger,
    decel_min,
    decel_max,
):
    """
    Batched version: scene_score_info is a list of B dicts.
    Returns: torch.Tensor of shape [B, num_bins]
    """
    B = len(scene_score_info)
    num_bins = len(speed_bins)
    all_scores = []
    for b in range(B):
        rules_score = [1.0 for _ in speed_bins]
        traffic_rules_info = scene_score_info[b].get("traffic_rules", {})
        for tf_rule in traffic_rules_info:
            if isinstance(tf_rule["type"], list):
                tf_rule["type"] = tf_rule["type"][0]  # Handle case where type is a list
            if tf_rule["type"] == "traffic_light":
                light_distance = tf_rule["dist_to_trig"]  # in meters
                for idx, bin_speed in enumerate(speed_bins):
                    if bin_speed < 1e-2:
                        rules_score[idx] = 1.0
                        continue
                    required_decel = _required_decel_to_stop(
                        speed=bin_speed,
                        dist_to_trig=light_distance,
                        stop_distance_from_trigger=stop_distance_from_trigger,
                    )
                    if math.isinf(required_decel):
                        rules_score[idx] = 0.0
                    else:
                        rules_score[idx] = _decel_to_score(required_decel, decel_min, decel_max)
        all_scores.append(rules_score)
    return torch.tensor(all_scores, dtype=torch.float32)  # Shape: [B, num_bins]



def compute_stop_sign_adherence(
    speed_bins,
    scene_score_info,
    stop_distance_from_trigger,
    decel_min,
    decel_max,
):
    """
    Batched version: scene_score_info is a list of B dicts.
    Returns: torch.Tensor of shape [B, num_bins]
    """
    B = len(scene_score_info)
    num_bins = len(speed_bins)
    all_scores = []
    for b in range(B):
        rules_score = [1.0 for _ in speed_bins]
        traffic_rules_info = scene_score_info[b].get("traffic_rules", {})
        for tf_rule in traffic_rules_info:
            if isinstance(tf_rule["type"], list):
                tf_rule["type"] = tf_rule["type"][0]  # Handle case where type is a list
            if tf_rule["type"] == "stop_sign":
                distance = tf_rule["dist_to_trig"]  # in meters
                for idx, bin_speed in enumerate(speed_bins):
                    if bin_speed < 1e-2:
                        rules_score[idx] = 1.0
                        continue
                    required_decel = _required_decel_to_stop(
                        speed=bin_speed,
                        dist_to_trig=distance,
                        stop_distance_from_trigger=stop_distance_from_trigger,
                    )
                    if math.isinf(required_decel):
                        rules_score[idx] = 0.0
                    else:
                        rules_score[idx] = _decel_to_score(required_decel, decel_min, decel_max)
        all_scores.append(rules_score)
    return torch.tensor(all_scores, dtype=torch.float32)  # Shape: [B, num_bins]

def lateral_shift_paths(path):
    """
    path: [B, T, 2]  (1m spaced)
    returns: [B, K, T, 2]
    """
    B, T, _ = path.shape
    device = path.device

    # ---- Tangents & normals ----
    tangents = path[:, 1:] - path[:, :-1]
    tangents = torch.cat([tangents, tangents[:, -1:]], dim=1)
    tangents = tangents / (torch.norm(tangents, dim=-1, keepdim=True) + 1e-6)

    normals = torch.stack([-tangents[..., 1], tangents[..., 0]], dim=-1)

    aug_paths = []

    # 0) Original
    aug_paths.append(path)

    # ---- 1) Constant lateral shifts ----
    lateral_shifts = torch.linspace(-0.6, 0.6, steps=2, device=device)
    for s in lateral_shifts:
        aug_paths.append(path + s * normals)

    # ---- 2) Progressive lateral drift (cut-in / drift) ----
    t = torch.linspace(0, 1, T, device=device).view(1, T, 1)

    for s in [-0.8, 0.8]:
        drift = s * t
        aug_paths.append(path + drift * normals)

    # ---- 3) Mild curvature bias (bend path) ----
    # rotate tangent slightly over time
    angles = torch.linspace(-0.15, 0.15, T, device=device)  # radians
    cos_a = torch.cos(angles)
    sin_a = torch.sin(angles)

    rot = torch.stack([
        torch.stack([cos_a, -sin_a], dim=-1),
        torch.stack([sin_a,  cos_a], dim=-1)
    ], dim=-2)  # [T, 2, 2]

    rotated_tangents = torch.einsum("tij,btj->bti", rot, tangents)

    curved_path = torch.zeros_like(path)
    curved_path[:, 0] = path[:, 0]
    curved_path[:, 1:] = curved_path[:, 0:1] + torch.cumsum(rotated_tangents[:, :-1], dim=1)

    aug_paths.append(curved_path)

    return torch.stack(aug_paths, dim=1)  # [B, K+1, T, 2]