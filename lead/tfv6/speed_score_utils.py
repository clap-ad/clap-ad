
import math
import numpy as np
import torch
import datetime
import time

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

def get_stable_yaws(path, initial_yaw, threshold=0.05):
    """
    Since our collision is dependent on precise bounding boxes, jitter
    in the yaw derived from small movements can cause large changes in the bounding box corners,
    which can lead to unstable distance calculations.
    """
    yaws = np.zeros(len(path))
    yaws[0] = initial_yaw
    
    # Calculate differences
    diffs = np.diff(path, axis=0)
    
    for i in range(len(diffs)):
        dist = np.linalg.norm(diffs[i])
        
        if dist > threshold:
            # Only update if the movement is significant
            yaws[i+1] = np.arctan2(diffs[i, 1], diffs[i, 0])
        else:
            # Keep the previous yaw if jitter is too small
            yaws[i+1] = yaws[i]
            
    # Final pass: wrap and unwrap to keep it smooth
    return np.unwrap(yaws)

def is_overlapping(pts1, pts2):
    """ SAT Check for Overlap (The "Precise" way to return 0.0)"""
    for poly in [pts1, pts2]:
        for i in range(4):
            # Get the edge vector and its normal (perpendicular)
            edge = poly[(i+1)%4] - poly[i]
            normal = np.array([-edge[1], edge[0]])
            
            # Project corners onto the normal
            p1 = pts1 @ normal
            p2 = pts2 @ normal
            
            # Check for a gap
            if np.max(p1) < np.min(p2) or np.max(p2) < np.min(p1):
                return False # Found a gap!
    return True # No gaps found on any axis

def point_to_segment_dist(p, a, b):
    """Distance from point p to line segment ab."""
    ab = b - a
    ap = p - a
    # Project p onto ab, normalized by segment length squared
    t = np.dot(ap, ab) / (np.dot(ab, ab) + 1e-6)
    # Clamp t to the range [0, 1] to stay on the segment
    t = np.clip(t, 0, 1)
    closest_point = a + t * ab
    return np.linalg.norm(p - closest_point)

def _get_bbox_distance(ego_x, ego_y, ego_w, ego_l, obj_x, obj_y, obj_w, obj_l, ego_yaw=None, obj_yaw=None):
    """
    Calculates the shortest distance between two axis-aligned bounding boxes.
    Assumes x,y is the center of the box.

    Because in Lead, x is forward and y is right, we treat ego_w as the width along the y-axis and ego_l as the length along the x-axis.
    """
    if ego_yaw is None:
        # Calculate the distance between edges on the X axis
        # dx = max(0, abs(ego_x - obj_x) - (ego_w + obj_w) / 2)
        dx = max(0, abs(ego_x - obj_x) - (ego_l + obj_l) / 2)
        
        # Calculate the distance between edges on the Y axis
        # dy = max(0, abs(ego_y - obj_y) - (ego_l + obj_l) / 2)
        dy = max(0, abs(ego_y - obj_y) - (ego_w + obj_w) / 2)
        
        # If both dx and dy are 0, the boxes are colliding
        return math.sqrt(dx**2 + dy**2)
    else:
        # Get corners of the object box
        c, s = math.cos(obj_yaw), math.sin(obj_yaw)
        rot = np.array([[c, -s], [s, c]])
        half_w, half_l = obj_w / 2, obj_l / 2
        corners = np.array([
            [-half_l, -half_w],
            [ half_l, -half_w],
            [ half_l,  half_w],
            [-half_l,  half_w]
        ]) @ rot.T + np.array([obj_x, obj_y])
        ec, es = math.cos(ego_yaw), math.sin(ego_yaw)
        ego_rot = np.array([[ec, -es], [es, ec]])
        ego_corners = np.array([
            [-ego_l/2, -ego_w/2],
            [ ego_l/2, -ego_w/2],
            [ ego_l/2,  ego_w/2],
            [-ego_l/2,  ego_w/2]
        ]) @ ego_rot.T + np.array([ego_x, ego_y])
        min_dist = float('inf')

        if is_overlapping(ego_corners, corners):
            return 0.0, ego_corners, corners
        # Check every corner of Object against every edge of Ego
        # And every corner of Ego against every edge of Object 
        for i in range(4):
            ego_edge_a = ego_corners[i]
            ego_edge_b = ego_corners[(i + 1) % 4]

            obj_edge_a = corners[i]
            obj_edge_b = corners[(i + 1) % 4]
            if (ego_edge_a[0] < obj_edge_b[0] and ego_edge_b[0] > obj_edge_a[0] and
                ego_edge_a[1] < obj_edge_b[1] and ego_edge_b[1] > obj_edge_a[1]):
                return 0.0, ego_corners, corners

            for j in range(4):
                # Distance from Obj corner to Ego edge
                d1 = point_to_segment_dist(corners[j], ego_edge_a, ego_edge_b)
                # Distance from Ego corner to Obj edge
                d2 = point_to_segment_dist(ego_corners[j], obj_edge_a, obj_edge_b)
                min_dist = min(min_dist, d1, d2)
        return min_dist, ego_corners, corners

def _compute_swept_bbox_distance(p_e_now, p_e_future, p_o_now, p_o_future, ego_wl, obj_wl, steps=5):
    """
    Interpolates between current and future positions to find the minimum 
    distance reached during the time interval, preventing "teleportation" 
    at high speeds.
    Used by p_o_fut approach where constant velocity is assumed and we only have one future position.
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

def compute_min_distance_traj(ego_traj, obj_traj, ego_wl, obj_wl, ego_yaws=None, obj_yaws=None):
    """
    ego_traj: [T, 2]
    obj_traj: [T, 2]
    Computes the minimum distance between two trajectories of bounding boxes over time.
    Used by fut_wp approach where we have a full future trajectory and can check distance at multiple time steps.
    If yaw info is provided, more accurate edge by edge distance will be calculated.
    """
    min_dist = float("inf")
    if ego_yaws is None:
        for t in range(len(ego_traj)):
            p_e = ego_traj[t]
            p_o = obj_traj[t]

            dist = _get_bbox_distance(
                p_e[0], p_e[1], ego_wl[0], ego_wl[1],
                p_o[0], p_o[1], obj_wl[0], obj_wl[1]
            )

            if dist < min_dist:
                min_dist = dist
        debug = False
        if debug:
            # Visualization: plot ego and agent bounding boxes at each time step using corners
            import matplotlib.pyplot as plt
            import matplotlib.cm as cm
            fig, ax = plt.subplots(figsize=(8, 8))
            cmap = cm.get_cmap('viridis', len(ego_traj))
            for t in range(len(ego_traj)):
                color = cmap(t)
                ego_pos = ego_traj[t]
                obj_pos = obj_traj[t]
                ax.plot(ego_pos[0], ego_pos[1], 'o', color=color, label='Ego' if t == 0 else None)
                ax.plot(obj_pos[0], obj_pos[1], 'x', color=color, label='Agent' if t == 0 else None)
                # Plot ego bounding box (axis-aligned)
                ego_w, ego_l = ego_wl
                ego_corners = np.array([
                    [ego_pos[0] - ego_l/2, ego_pos[1] - ego_w/2],
                    [ego_pos[0] + ego_l/2, ego_pos[1] - ego_w/2],
                    [ego_pos[0] + ego_l/2, ego_pos[1] + ego_w/2],
                    [ego_pos[0] - ego_l/2, ego_pos[1] + ego_w/2]
                ])
                ego_poly = plt.Polygon(ego_corners, closed=True, edgecolor=color, facecolor='none', lw=2, alpha=0.5)
                ax.add_patch(ego_poly)
                # Plot agent bounding box (axis-aligned)
                obj_w, obj_l = obj_wl
                obj_corners = np.array([
                    [obj_pos[0] - obj_l/2, obj_pos[1] - obj_w/2],
                    [obj_pos[0] + obj_l/2, obj_pos[1] - obj_w/2],
                    [obj_pos[0] + obj_l/2, obj_pos[1] + obj_w/2],
                    [obj_pos[0] - obj_l/2, obj_pos[1] + obj_w/2]
                ])
                obj_poly = plt.Polygon(obj_corners, closed=True, edgecolor=color, facecolor='none', lw=2, linestyle='--', alpha=0.5)
                ax.add_patch(obj_poly)
            ax.set_aspect('equal')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            plt.legend()
            ax.set_title('Ego and Agent Trajectories Over Time (Color = Time)')
            plt.savefig("debug_traj_distance.png")
            plt.show()
    else:
        for t in range(len(ego_traj)):
            p_e = ego_traj[t]
            p_o = obj_traj[t]
            # Todo: logic for discarding calculation based on w/l <10m
            
            dist, ego_corners, corners = _get_bbox_distance(
                p_e[0], p_e[1], ego_wl[0], ego_wl[1],
                p_o[0], p_o[1], obj_wl[0], obj_wl[1],
                ego_yaw=ego_yaws[t], obj_yaw=obj_yaws[t]
            )
            if dist < min_dist:
                min_dist = dist
        debug = False
        if debug:
            # Visualization: plot ego and agent bounding boxes at each time step using corners
            import matplotlib.pyplot as plt
            import matplotlib.cm as cm

            fig, ax = plt.subplots(figsize=(8, 8))
            cmap = cm.get_cmap('viridis', len(ego_traj))
            distances = []
            for t in range(len(ego_traj)):
                color = cmap(t)
                # Use ego_corners and corners from _get_bbox_distance
                dist, ego_corners, corners = _get_bbox_distance(
                    ego_traj[t][0], ego_traj[t][1], ego_wl[0], ego_wl[1],
                    obj_traj[t][0], obj_traj[t][1], obj_wl[0], obj_wl[1],
                    ego_yaw=ego_yaws[t], obj_yaw=obj_yaws[t]
                )
                distances.append(dist)
                # Ego box
                ego_poly = plt.Polygon(ego_corners, closed=True, edgecolor=color, facecolor='none', lw=2, label='Ego' if t == 0 else None)
                ax.add_patch(ego_poly)
                # Agent box
                agent_poly = plt.Polygon(corners, closed=True, edgecolor=color, facecolor='none', lw=2, linestyle='--', label='Agent' if t == 0 else None)
                ax.add_patch(agent_poly)
                # Plot centers
                ax.plot(ego_traj[t][0], ego_traj[t][1], 'o', color=color)
                ax.plot(obj_traj[t][0], obj_traj[t][1], 'x', color=color)

            ax.set_aspect('equal')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_title('Ego and Agent Bounding Boxes Over Time (Color = Time)')
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys())
            # Write distances as text below the figure
            for t, dist in enumerate(distances):
                ax.text(0.5, -0.2 - t*0.2, f'Time {t}: Dist={dist:.2f}', transform=ax.transAxes, fontsize=8, color=cmap(t))
            now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            plt.savefig(f"/workspace/temp_vis/bboxes/debug_swept_bbox_distance_{now_str}.png")
            plt.show()
            plt.close(fig)

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
        path_pred_b = np.concatenate([p_e.reshape(1, -1), path_pred[b].cpu().numpy()], axis=0)  # [T+1, 2], prepend current position
        cum_dist = np.linspace(0, T, T + 1)

        # Get ego position after TIME_HORIZON on the given path
        target_dist = speed * time_horizon
        p_e_future = np.array([
            np.interp(target_dist, cum_dist, path_pred_b[:, 0]),
            np.interp(target_dist, cum_dist, path_pred_b[:, 1])
        ])

        # Get ego trajectory up to TIME_HORIZON for fut_wp approach
        dt = time_horizon / 20 
        future_timestamps = np.linspace(dt, time_horizon, 20)
        future_target_dists = speed * future_timestamps

        # 3. Interpolate
        p_e_future_points = np.stack([
            np.interp(future_target_dists, cum_dist, path_pred_b[:, 0]),
            np.interp(future_target_dists, cum_dist, path_pred_b[:, 1])
        ], axis=1)

        threat_actor = None
        min_distance = safe_distance
        ego_full_path = np.vstack([p_e.reshape(1, 2), p_e_future_points])
        ego_yaws = get_stable_yaws(ego_full_path, 0.0, threshold=0.05)
        min_dist_agent_for_debugging = float('inf')
        p_o_x_min = None
        for actor in scene_score_info[b]["actors"]:
            if not actor["ttc_cal"]:
                continue
            fut_wp = actor.get("fut_wp", None)
            fut_wp_invalid = actor.get("fut_wp_invalid", True)
            if (fut_wp is None) or (fut_wp is not None and fut_wp_invalid): # If fut_wp is not provided, we use constant velocity to predict future position
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
                    steps=10 if time_horizon <=1.0 else 20
                )
                if dist_future < min_distance:
                    threat_actor = actor
                    min_distance = dist_future
            else:
                # fut_wp has shape [T, 2], we can directly use it as future trajectory and check swept distance at multiple time steps
                obj_wl = [float(x) for x in actor["bbox"]]
                fut_wp = fut_wp[0] if fut_wp.ndim == 3 else fut_wp  # Handle case where fut_wp has extra batch dimension
                edge_by_edge = True  # Whether to check distance edge by edge or just point to point
                if edge_by_edge:
                    curr_yaw = actor["yaw"]
                    # calculate yaw for each time step based on trajectory, then calculate corners for each time step, and check distance between corners and ego box
                    p_o = np.array([actor["p_o_x"], actor["p_o_y"]])
                    agent_full_path = np.vstack([p_o.reshape(1, 2), fut_wp])
                    agent_yaws = get_stable_yaws(agent_full_path, curr_yaw, threshold=0.05)

                dist_future = compute_min_distance_traj(
                    ego_traj=p_e_future_points,
                    obj_traj=fut_wp,
                    ego_wl=ego_wl,
                    obj_wl=obj_wl,
                    ego_yaws=ego_yaws if edge_by_edge else None,
                    obj_yaws=agent_yaws if edge_by_edge else None
                )
                if dist_future < min_dist_agent_for_debugging:
                    min_dist_agent_for_debugging = dist_future
                    p_o_x_min = actor["p_o_x"]
                if dist_future < min_distance:
                    threat_actor = actor
                    min_distance = dist_future
        # print(f"Batch {b}, p_o_x_min: {p_o_x_min}, min_dist_agent: {min_dist_agent_for_debugging}")
        if min_distance >= safe_distance:
            results.append(1.0)
        elif min_distance <= coll_radius:
            results.append(0.0)
        else:
            ttc = (min_distance - coll_radius) / (safe_distance - coll_radius)
            results.append(ttc)
    return torch.tensor(results, device=device)

############################################
#### Utils for batched TTC computations ####
############################################

def get_stable_yaws_multipath(path, initial_yaw, threshold=0.05):
    """
    Stabilizes yaws same as get_stable_yaws but for a batch of paths.
    path: [B, K, S, N+1, 2] where B=batch size, K=number of paths, S=number of scenes, N+1=time steps (including current position)
    Return:
    ego_yaws: [B, K, S, N+1] stable yaws for each path and time step
    """
    # differences
    diffs = path[..., 1:, :] - path[..., :-1, :]  # [B,K,S,N,2]

    dx = diffs[..., 0]
    dy = diffs[..., 1]

    dist = torch.sqrt(dx**2 + dy**2)  # [B,K,S,N]

    # candidate yaws from motion
    cand_yaw = torch.atan2(dy, dx)  # [B,K,S,N]

    # mask where movement is significant
    valid = dist > threshold  # [B,K,S,N]

    # initialize yaw tensor
    B, K, S, N = cand_yaw.shape
    ego_yaws = torch.zeros((B, K, S, N + 1), device=path.device)

    ego_yaws[..., 0] = initial_yaw

    # propagate yaw (sequential over time, vectorized over B,K,S)
    for t in range(N):
        prev = ego_yaws[..., t]
        curr = cand_yaw[..., t]
        ego_yaws[..., t + 1] = torch.where(valid[..., t], curr, prev)

    # unwrap (vectorized)
    delta = ego_yaws[..., 1:] - ego_yaws[..., :-1]
    delta = (delta + torch.pi) % (2 * torch.pi) - torch.pi
    ego_yaws[..., 1:] = ego_yaws[..., 0:1] + torch.cumsum(delta, dim=-1)

    return ego_yaws

def get_corners(xy, w, l, yaw, device=None, dtype=torch.float32):
    """
    xy: (N,2)
    w,l,yaw: (N,)
    """
    x, y = xy[:, 0], xy[:, 1]

    c = torch.cos(yaw)
    s = torch.sin(yaw)

    rot = torch.stack([
        torch.stack([c, -s], dim=-1),
        torch.stack([s,  c], dim=-1)
    ], dim=-2)  # (N,2,2)

    base = torch.tensor([
        [-0.5, -0.5],
        [ 0.5, -0.5],
        [ 0.5,  0.5],
        [-0.5,  0.5]
    ], device=device, dtype=dtype)  # (4,2)

    scales = torch.stack([l, w], dim=-1).unsqueeze(1).to(device, dtype=dtype)  # (N,1,2)
    corners = base.unsqueeze(0) * scales              # (N,4,2)

    return torch.bmm(corners, rot.transpose(1, 2)) + torch.stack([x, y], dim=-1).unsqueeze(1)

def get_edges(corners):
    # (N,4,2,2)
    return torch.stack([corners, torch.roll(corners, shifts=-1, dims=1)], dim=2)

def aabb_filter(ego_xy, obj_xy, ego_l, obj_l, ego_w, obj_w):
    dx = torch.abs(ego_xy[:, 0] - obj_xy[:, 0])
    dy = torch.abs(ego_xy[:, 1] - obj_xy[:, 1])

    return (dx < (ego_l + obj_l) * 0.5) & (dy < (ego_w + obj_w) * 0.5)

def is_overlapping_eff(ego_c, obj_c):
    def get_axes(c):
        edges = torch.roll(c, -1, dims=1) - c  # (N,4,2)
        normals = torch.stack([-edges[...,1], edges[...,0]], dim=-1)
        normals = normals / torch.clamp(torch.norm(normals, dim=-1, keepdim=True), 1e-8)
        return normals

    axes = torch.cat([get_axes(ego_c), get_axes(obj_c)], dim=1)  # (N,8,2)

    def project(c, axes):
        # (N,4,2) @ (N,8,2)
        return torch.einsum("nkd,nad->nak", c, axes)

    p1 = project(ego_c, axes)
    p2 = project(obj_c, axes)

    return torch.all(
        (p1.max(dim=-1).values >= p2.min(dim=-1).values) &
        (p2.max(dim=-1).values >= p1.min(dim=-1).values),
        dim=1
    )

def point_to_seg(P, A, B):
    AB = B - A
    AP = P - A

    denom = torch.clamp((AB * AB).sum(dim=-1), min=1e-8)

    t = (AP * AB).sum(dim=-1) / denom
    t = torch.clamp(t, 0.0, 1.0).unsqueeze(-1)

    proj = A + t * AB
    return ((P - proj) ** 2).sum(dim=-1)  # squared distance (no sqrt)


def _compute_min_distance_traj_batched_multipath(ego_traj, obj_traj, ego_wl, obj_wl, cuda_device, ego_yaws, obj_yaws):
    """
    Fully vectorized batched version (no for-loop over batches).
    ego_traj: [B, K, S, T, 2]
    obj_traj: [B, A, T, 2]
    ego_wl: [2]
    obj_wl: [B, A, 2]
    ego_yaws: [B, K, S, T+1]
    obj_yaws: [B, A, T+1]
    Returns: [B, K, S, A] minimum distance for each batch/actor.
    """
    ego_yaws = ego_yaws[..., 1:]
    obj_yaws = obj_yaws[..., 1:]
    # obj_traj: [B, A, T, 2]
    # ego_traj: [B, K, S, T, 2]
    # ego_yaws: [B, K, S, T]
    # obj_yaws: [B, A, T]

    B, K, S, T, _ = ego_traj.shape
    A = obj_traj.shape[1]

    # Expand ego_traj and ego_yaws to [B, K, S, A, T, ...]
    ego_traj_exp = np.expand_dims(ego_traj, axis=3)         # [B, K, S, 1, T, 2]
    ego_traj_exp = np.broadcast_to(ego_traj_exp, (B, K, S, A, T, 2))
    ego_yaws_exp = np.expand_dims(ego_yaws, axis=3)         # [B, K, S, 1, T]
    ego_yaws_exp = np.broadcast_to(ego_yaws_exp, (B, K, S, A, T))

    # Expand obj_traj and obj_yaws to [B, K, S, A, T, ...]
    obj_traj_exp = np.expand_dims(np.expand_dims(obj_traj, axis=1), axis=1)  # [B, 1, 1, A, T, 2]
    obj_traj_exp = np.broadcast_to(obj_traj_exp, (B, K, S, A, T, 2))
    obj_yaws_exp = np.expand_dims(np.expand_dims(obj_yaws, axis=1), axis=1)  # [B, 1, 1, A, T]
    obj_yaws_exp = np.broadcast_to(obj_yaws_exp, (B, K, S, A, T))

    # Now flatten everything to [B*K*S*A*T, ...]
    N = B * K * S * A * T
    ego_xy = ego_traj_exp.reshape(N, 2)
    obj_xy = obj_traj_exp.reshape(N, 2)
    ego_yaw = ego_yaws_exp.reshape(N)
    obj_yaw = obj_yaws_exp.reshape(N)

    # Broadcast widths/lengths
    ew = np.full((N, 2), ego_wl)  # [N, 2]
    ow = np.broadcast_to(obj_wl[:, None, None, :, None, :], (B, K, S, A, T, 2)).reshape(N, 2)  # [N, 2]

    ego_w = ew[:, 0]
    ego_l = ew[:, 1]
    obj_w = ow[:, 0]
    obj_l = ow[:, 1]
    # Eff below
    device = cuda_device
    dtype = torch.float16
    ego_xy = torch.from_numpy(ego_xy).to(device, dtype=dtype)
    obj_xy = torch.from_numpy(obj_xy).to(device, dtype=dtype)
    ego_w = torch.from_numpy(ego_w).to(device, dtype=dtype)
    ego_l = torch.from_numpy(ego_l).to(device, dtype=dtype)
    ego_yaw = torch.from_numpy(ego_yaw).to(device, dtype=dtype)
    obj_w = torch.from_numpy(obj_w).to(device, dtype=dtype)
    obj_l = torch.from_numpy(obj_l).to(device, dtype=dtype)
    obj_yaw = torch.from_numpy(obj_yaw).to(device, dtype=dtype)
    N = ego_xy.shape[0]
    # ---------- AABB filter ----------
    ego_c = get_corners(ego_xy, ego_w, ego_l, ego_yaw, device=device, dtype=dtype)
    obj_c = get_corners(obj_xy, obj_w, obj_l, obj_yaw, device=device, dtype=dtype)

    overlap = is_overlapping_eff(ego_c, obj_c)
    min_dist = torch.full((N,), float("inf"), device=device, dtype=dtype)
    # 1. overlapping → 0
    min_dist[overlap] = 0.0
    # 2. compute only for non-overlapping
    non_overlap = ~overlap
    idx = non_overlap.nonzero(as_tuple=True)[0]
    if idx.numel() > 0:
        ego_c_f = ego_c[idx]
        obj_c_f = obj_c[idx]
        ego_e_f = get_edges(ego_c_f)
        obj_e_f = get_edges(obj_c_f)
        d1 = point_to_seg(
            obj_c_f[:, :, None, :],
            ego_e_f[:, None, :, 0, :],
            ego_e_f[:, None, :, 1, :]
        )

        d2 = point_to_seg(
            ego_c_f[:, :, None, :],
            obj_e_f[:, None, :, 0, :],
            obj_e_f[:, None, :, 1, :]
        )

        dist_f = torch.minimum(
            torch.amin(d1, dim=(1, 2)),
            torch.amin(d2, dim=(1, 2))
        ).to(dtype=dtype)

        min_dist[idx] = dist_f
    min_dist = torch.sqrt(min_dist)  # shape (N,)
    min_dist = min_dist.reshape(B, K, S, A, T)
    # minimum over time in torch
    min_dists = torch.min(min_dist, dim=4).values  # shape (B, K, S, A)
    # Add a visualization here to plot bounding boxes over time and their corresponding distances calculated above
    debug = False
    if debug:
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        # Plot for arbitrary K and S (number of paths and speed bins)
        # ego_traj: [B, K, S, T, 2], obj_traj: [B, A, T, 2]
        # ego_c, obj_c: [N, 4, 2], where N = B*K*S*A*T
        # min_dist: [B, K, S, A, T]
        max_batches = min(2, B)  # Limit number of batches to plot
        max_k = min(2, K)
        max_s = min(2, S)
        for b in range(max_batches):
            for k in range(max_k):
                for s in range(max_s):
                    fig, ax = plt.subplots(figsize=(8, 8))
                    cmap = cm.get_cmap('tab10', A)
                    for a in range(A):
                        # Only plot for agents with initial position less than 30 meters to ego
                        ego_init = ego_traj[b, k, s, 0]  # [2]
                        obj_init = obj_traj[b, a, 0]     # [2]
                        for t in range(T):
                            color = cmap(a)
                            # Compute flat index for ego_c/obj_c
                            idx = (
                                b * K * S * A * T +
                                k * S * A * T +
                                s * A * T +
                                a * T +
                                t
                            )
                            ego_box = ego_c[idx].cpu().numpy()
                            obj_box = obj_c[idx].cpu().numpy()
                            # Plot ego bounding box (full polygon)
                            ax.plot(
                                np.append(ego_box[:, 0], ego_box[0, 0]),
                                np.append(ego_box[:, 1], ego_box[0, 1]),
                                '-', color=color, lw=2, label=f'Ego (A{a})' if t == 0 else None
                            )
                            # Plot object bounding box (full polygon)
                            ax.plot(
                                np.append(obj_box[:, 0], obj_box[0, 0]),
                                np.append(obj_box[:, 1], obj_box[0, 1]),
                                '--', color=color, lw=2, label=f'Obj (A{a})' if t == 0 else None
                            )
                            # Plot centers
                            ax.plot(ego_box[:, 0].mean(), ego_box[:, 1].mean(), 'o', color=color)
                            ax.plot(obj_box[:, 0].mean(), obj_box[:, 1].mean(), 'x', color=color)
                            # Annotate distance
                            ax.text(
                                obj_box[:, 0].mean(),
                                obj_box[:, 1].mean(),
                                f"{float(min_dist[b, k, s, a, t]):.2f}",
                                color=color,
                                fontsize=8,
                                ha='center',
                                va='bottom'
                            )
                    ax.set_aspect('equal')
                    ax.set_xlabel('X')
                    ax.set_ylabel('Y')
                    ax.set_title(f'Batch {b}, Path {k}, SpeedBin {s}: Ego and Objects Bounding Boxes Over Time')
                    handles, labels = ax.get_legend_handles_labels()
                    by_label = dict(zip(labels, handles))
                    ax.legend(by_label.values(), by_label.keys())
                    plt.savefig(f"/workspace/temp_vis/bboxes/debug_bbox_batch{b}_k{k}_s{s}.png")
                    plt.show()
                    plt.close(fig)
    return min_dists

def _compute_min_distance_traj_batched(ego_traj, obj_traj, ego_wl, obj_wl, cuda_device, ego_yaws=None, obj_yaws=None):
    """
    Fully vectorized batched version (no for-loop over batches).
    ego_traj: [B, T, 2]
    obj_traj: [B, A, T, 2]
    ego_wl: [2] or [B, 2]
    obj_wl: [B, A, 2]
    Returns: [B, A] minimum distance for each batch/actor.
    """
    if ego_yaws is None:
        B, T, _ = ego_traj.shape
        # Broadcast widths/lengths
        ew = ego_wl if ego_wl.ndim == 2 else np.broadcast_to(ego_wl, (B, 2))
        ow = obj_wl
        # ego_traj: [B, T, 2], obj_traj: [B, T, 2]
        dx = np.abs(ego_traj[..., 0] - obj_traj[..., 0]) - (ew[:, None, 1] + ow[:, None, 1]) / 2
        dy = np.abs(ego_traj[..., 1] - obj_traj[..., 1]) - (ew[:, None, 0] + ow[:, None, 0]) / 2
        dx = np.clip(dx, 0, None)
        dy = np.clip(dy, 0, None)
        dists = np.sqrt(dx ** 2 + dy ** 2)  # [B, T]
        min_dists = np.min(dists, axis=1)   # [B]
        # If any collision at any time, set min_dist to 0
        min_dists = np.where(np.any(dists == 0, axis=1), 0.0, min_dists)
        return min_dists
    else:
        ego_yaws = ego_yaws[:, 1:]
        obj_yaws = obj_yaws[:, :, 1:]
        B, A, T, _ = obj_traj.shape
        # Expand ego_traj and ego_yaws to [B, A, T, ...]
        ego_traj_exp = np.expand_dims(ego_traj, axis=1)         # [B, 1, T, 2]
        ego_traj_exp = np.broadcast_to(ego_traj_exp, (B, A, T, 2))
        ego_yaws_exp = np.expand_dims(ego_yaws, axis=1)         # [B, 1, T]
        ego_yaws_exp = np.broadcast_to(ego_yaws_exp, (B, A, T))
        # Now flatten everything to [B*A*T, ...]
        N = B * A * T
        ego_xy = ego_traj_exp.reshape(N, 2)
        obj_xy = obj_traj.reshape(N, 2)
        ego_yaw = ego_yaws_exp.reshape(N)
        obj_yaw = obj_yaws.reshape(N)
        # Broadcast widths/lengths
        ew = np.full((N, 2), ego_wl)  # [N, 2]
        ow = np.repeat(obj_wl, T, axis=1).reshape(N, 2)  # [N, 2]

        ego_w = ew[:, 0]
        ego_l = ew[:, 1]
        obj_w = ow[:, 0]
        obj_l = ow[:, 1]
        # Compute corners for all
        efficient = True
        if efficient:
            device = cuda_device
            dtype = torch.float16
            ego_xy = torch.from_numpy(ego_xy).to(device, dtype=dtype)
            obj_xy = torch.from_numpy(obj_xy).to(device, dtype=dtype)
            ego_w = torch.from_numpy(ego_w).to(device, dtype=dtype)
            ego_l = torch.from_numpy(ego_l).to(device, dtype=dtype)
            ego_yaw = torch.from_numpy(ego_yaw).to(device, dtype=dtype)
            obj_w = torch.from_numpy(obj_w).to(device, dtype=dtype)
            obj_l = torch.from_numpy(obj_l).to(device, dtype=dtype)
            obj_yaw = torch.from_numpy(obj_yaw).to(device, dtype=dtype)
            N = ego_xy.shape[0]
            # ---------- AABB filter ----------
            ego_c = get_corners(ego_xy, ego_w, ego_l, ego_yaw, device=device, dtype=dtype)
            obj_c = get_corners(obj_xy, obj_w, obj_l, obj_yaw, device=device, dtype=dtype)

            overlap = is_overlapping_eff(ego_c, obj_c)
            min_dist = torch.full((N,), float("inf"), device=device, dtype=dtype)
            # 1. overlapping → 0
            min_dist[overlap] = 0.0
            # 2. compute only for non-overlapping
            non_overlap = ~overlap
            idx = non_overlap.nonzero(as_tuple=True)[0]
            if idx.numel() > 0:
                ego_c_f = ego_c[idx]
                obj_c_f = obj_c[idx]
                ego_e_f = get_edges(ego_c_f)
                obj_e_f = get_edges(obj_c_f)
                d1 = point_to_seg(
                    obj_c_f[:, :, None, :],
                    ego_e_f[:, None, :, 0, :],
                    ego_e_f[:, None, :, 1, :]
                )

                d2 = point_to_seg(
                    ego_c_f[:, :, None, :],
                    obj_e_f[:, None, :, 0, :],
                    obj_e_f[:, None, :, 1, :]
                )

                dist_f = torch.minimum(
                    torch.amin(d1, dim=(1, 2)),
                    torch.amin(d2, dim=(1, 2))
                ).to(dtype=dtype)

                min_dist[idx] = dist_f
            min_dist = min_dist.reshape(B, A, T)
            # minimum over time in torch
            min_dists = torch.min(min_dist, dim=2).values  # shape (B, A)
        else:
            # Now, for each pair, call the SAT/edge-corner logic (still needs a loop)
            dists = np.zeros(N)
            all_ego_corners = np.zeros((N, 4, 2))
            all_corners = np.zeros((N, 4, 2))
            for i in range(N):
                # You can optimize by writing a vectorized SAT/edge-corner check, but here we call the original logic
                dist, ego_corners, corners = _get_bbox_distance(
                    ego_xy[i, 0], ego_xy[i, 1], ego_w[i], ego_l[i],
                    obj_xy[i, 0], obj_xy[i, 1], obj_w[i], obj_l[i],
                    ego_yaw=ego_yaw[i], obj_yaw=obj_yaw[i]
                )
                dists[i] = dist
                all_ego_corners[i] = ego_corners
                all_corners[i] = corners
            dists = dists.reshape(B, A, T)
            min_dists = np.min(dists, axis=2)  # minimum over time
            collision_mask = np.any(dists == 0, axis=2)  # shape (B, A)
            min_dists = np.where(collision_mask, 0.0, min_dists)
        # Add a visualization here to plot bounding boxes over time and their corresponding distances calculated above
        debug = False
        if debug:
            import matplotlib.pyplot as plt
            import matplotlib.cm as cm
            for b in range(B):
                fig, ax = plt.subplots(figsize=(8, 8))
                cmap = cm.get_cmap('tab10', A)
                for a in range(A):
                    # Only plot for agents with initial position less than 30 meters to ego
                    ego_init = ego_traj[b, 0]
                    obj_init = obj_traj[b, a, 0]
                    dist_init = np.linalg.norm(ego_init - obj_init)
                    if dist_init >= 30.0:
                        continue
                    for t in range(T):
                        color = cmap(a)
                        # Get corners for ego and object at this timestep
                        idx = b * A * T + a * T + t
                        ego_box = all_ego_corners[idx]
                        obj_box = all_corners[idx]
                        # Plot ego bounding box
                        # ego_poly = plt.Polygon(ego_box, closed=True, edgecolor=color, facecolor='none', lw=2, label=f'Ego (A{a})' if t == 0 else None)
                        # ax.add_patch(ego_poly)
                        # # Plot object bounding box
                        # obj_poly = plt.Polygon(obj_box, closed=True, edgecolor=color, facecolor='none', lw=2, linestyle='--', label=f'Obj (A{a})' if t == 0 else None)
                        # ax.add_patch(obj_poly)
                        # Plot centers
                        ax.plot(ego_box[:, 0].mean(), ego_box[:, 1].mean(), 'o', color=color)
                        ax.plot(obj_box[:, 0].mean(), obj_box[:, 1].mean(), 'x', color=color)
                        # Annotate distance
                        ax.text(
                            obj_box[:, 0].mean(),
                            obj_box[:, 1].mean(),
                            f"{float(dists[b,a,t]):.2f}",
                            color=color,
                            fontsize=8,
                            ha='center',
                            va='bottom'
                        )
                ax.set_aspect('equal')
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_title(f'Batch {b}: Ego and Objects Bounding Boxes Over Time')
                handles, labels = ax.get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                ax.legend(by_label.values(), by_label.keys())
                plt.savefig(f"/workspace/temp_vis/bboxes/debug_bbox_batch{b}.png")
                plt.show()
                plt.close(fig)
        return min_dists
        

def _compute_swept_bbox_batched(
    p_e_now, p_e_future, p_o_now, p_o_future, ego_wl, obj_wl, steps=5
):
    """
    Vectorized batched version for axis-aligned bounding boxes.
    p_e_now, p_e_future: [B, 2]
    p_o_now, p_o_future: [B, A, 2]
    ego_wl: [2]
    obj_wl: [B, A, 2] or [1, A, 2]
    Returns: np.ndarray of shape [B, A] with minimum distances for each batch/actor.
    """
    B, A, _ = p_o_now.shape
    alphas = np.linspace(0, 1, steps + 1).reshape(-1, 1, 1, 1)  # [steps+1, 1, 1, 1]

    # Expand ego to [B, A, 2]
    ego_wl_arr = np.broadcast_to(np.array(ego_wl).reshape(1, 1, 2), (B, A, 2))

    # Interpolate ego and obj positions: [steps+1, B, A, 2]
    curr_e = (
        p_e_now[None, :, None, :] + alphas * (p_e_future[None, :, None, :] - p_e_now[None, :, None, :])
    )
    curr_o = (
        p_o_now[None, :, :, :] + alphas * (p_o_future[None, :, :, :] - p_o_now[None, :, :, :])
    )

    # Get lengths and widths
    ego_l = ego_wl_arr[..., 1]
    ego_w = ego_wl_arr[..., 0]
    obj_l = obj_wl[..., 1]
    obj_w = obj_wl[..., 0]

    # [steps+1, B, A]
    dx = np.abs(curr_e[..., 0] - curr_o[..., 0]) - (ego_l + obj_l) / 2
    dy = np.abs(curr_e[..., 1] - curr_o[..., 1]) - (ego_w + obj_w) / 2
    dx = np.clip(dx, 0, None)
    dy = np.clip(dy, 0, None)
    dists = np.sqrt(dx ** 2 + dy ** 2)
    # Min over steps
    min_dists = np.min(dists, axis=0)
    # Clamp to zero if any step is colliding
    min_dists = np.where(np.any(dists == 0, axis=0), 0.0, min_dists)
    return min_dists

def _build_padded_actor_info(scene_score_info, step, edge_by_edge=False, yaw_threshold=0.05):
    """
    Extracts and pads actor info for batched computation.
    Returns:
        p_o:        [B, A_max, 2]
        p_o_future: [B, A_max, 2]
        obj_wl:     [B, A_max, 2]
        fut_wp:     [B, A_max, T, 2] (padded with zeros if fut_wp is missing/short)
        obj_mask:   [B, A_max] (True if valid)
        agent_yaws: [B, A_max, T+1] (if edge_by_edge, else None)
    """
    B = len(scene_score_info)
    A_max = max(len(scene["actors"]) for scene in scene_score_info)
    p_o        = np.zeros((B, A_max, 2), dtype=np.float32)
    p_o_future = np.zeros((B, A_max, 2), dtype=np.float32)
    obj_wl     = np.zeros((B, A_max, 2), dtype=np.float32)
    fut_wp     = np.zeros((B, A_max, step, 2), dtype=np.float32)
    obj_mask   = np.zeros((B, A_max), dtype=bool)
    agent_yaws = np.zeros((B, A_max, step+1), dtype=np.float32) if edge_by_edge else None

    for b, scene in enumerate(scene_score_info):
        for a, actor in enumerate(scene["actors"]):
            if not actor.get("ttc_cal", True):
                continue
            p_o[b, a]        = [float(actor["p_o_x"]), float(actor["p_o_y"])]
            p_o_future[b, a] = [float(actor.get("p_o_x_fut", 0.0)), float(actor.get("p_o_y_fut", 0.0))]
            obj_wl[b, a]     = [float(x) for x in actor["bbox"]]
            fut_wp_arr = actor.get("fut_wp", None)
            fut_wp_invalid = actor.get("fut_wp_invalid", False)
            t_yaw = 0
            if fut_wp_arr is None or fut_wp_invalid:
                # If fut_wp is missing or invalid, we can optionally fill in a constant velocity prediction based on current and future position
                # Force scalar conversion to ensure a clean (2,) shape
                curr_pos = np.array([float(actor["p_o_x"]), float(actor["p_o_y"])])
                fut_pos  = np.array([float(actor["p_o_x_fut"]), float(actor["p_o_y_fut"])])
                for t in range(step):
                    alpha = (t + 1) / step
                    fut_wp[b, a, t] = curr_pos + alpha * (fut_pos - curr_pos)
                t_yaw = step
            else:
                arr = np.array(fut_wp_arr)
                if arr.ndim == 3: arr = arr[0]
                t_len = min(step, arr.shape[0])
                fut_wp[b, a, :t_len] = arr[:t_len]
                t_yaw = t_len
            obj_mask[b, a] = True
            if edge_by_edge:
                # Only compute agent_yaws if fut_wp is available and t_len > 0
                if fut_wp_arr is not None and t_yaw > 0 and "yaw" in actor:
                    curr_yaw = actor["yaw"]
                    agent_full_path = np.vstack([p_o[b, a].reshape(1, 2), fut_wp[b, a, :t_yaw]])
                    yaws = get_stable_yaws(agent_full_path, curr_yaw, threshold=yaw_threshold)
                    agent_yaws[b, a, :t_yaw+1] = yaws[:t_yaw+1]
                else:
                    agent_yaws[b, a, :] = 0.0  # or np.nan if you prefer
    return p_o, p_o_future, obj_wl, fut_wp, obj_mask, agent_yaws

def _build_ego_traj(path_pred, speed, time_horizon, steps=20):
    """
        Build ego trajectory from path_pred for efficient TTC computation.
        path_pred: [B, T, 2] predicted waypoints relative to current position (0,0)
        speed: scalar value for ego speed
    """
    B, T, _ = path_pred.shape
    device = path_pred.device
    p0 = torch.zeros((B, 1, 2), device=device)
    path = torch.cat([p0, path_pred], dim=1)  # [B, T+1, 2]
    cum_dist = torch.linspace(0, T, T + 1, device=device)
    target_dist = speed * time_horizon
    # 1. Ego future point at time_horizon [B, 2]
    path_np = path.cpu().numpy()  # [B, T+1, 2]
    if isinstance(target_dist, torch.Tensor):
        target_dist_np = target_dist.cpu().numpy()
    else:
        target_dist_np = target_dist
    p_e_future = np.stack([
        [np.interp(target_dist_np, cum_dist, path_np[b, :, 0]),
        np.interp(target_dist_np, cum_dist, path_np[b, :, 1])]
        for b in range(B)
    ], axis=0)  # [B, 2]

    # 2. ego trajectory points [B, 20, 2] for fut_wp approach
    dt = time_horizon / steps
    future_timestamps = torch.linspace(dt, time_horizon, steps, device=device)
    future_target_dists = speed * future_timestamps
    p_e_future_points = np.stack([
        np.stack([
            np.interp(future_target_dists, cum_dist, path_np[b, :, 0]),
            np.interp(future_target_dists, cum_dist, path_np[b, :, 1])
        ], axis=1)  # [N, 2]
        for b in range(B)
    ], axis=0)  # [B, N, 2]
    ego_full_path = np.concatenate([np.zeros((B, 1, 2)), p_e_future_points], axis=1)
    ego_yaws = np.stack([
        get_stable_yaws(ego_full_path[b], 0.0, threshold=0.05)
        for b in range(B)
    ], axis=0)
    return ego_full_path, p_e_future, p_e_future_points, ego_yaws

def _build_ego_traj_multipath(path_pred, speed, time_horizon, steps=20):
    """
        Build ego traj for each of multiple path_pred and multiple speed bins provided.
        path_pred: [B, K, T, 2], B=batch size, K=number of path predictions per batch, T=number of waypoints, 2 for x,y
        speed: [S] S is number of speed bins
        Note: we want to apply each of the speed bins to every path prediction, so we will end up with B x K x S trajectories.
        It is in vectorized form, so we will not actually loop over B, K, S, but we will compute everything in a batched manner.
    """
    B, K, T, _ = path_pred.shape
    S = speed.shape[0]
    device = path_pred.device

    # prepend origin
    p0 = torch.zeros((B, K, 1, 2), device=device)
    path = torch.cat([p0, path_pred], dim=2)  # [B, K, T+1, 2]

    # cumulative "distance" parameter (same as before)
    cum_dist = torch.linspace(0, T, T + 1, device=device)  # [T+1]

    # time setup
    dt = time_horizon / steps
    future_timestamps = torch.linspace(dt, time_horizon, steps, device=device)  # [N]

    # distances for each speed bin
    future_target_dists = speed[:, None] * future_timestamps[None, :]  # [S, N]

    # expand for vectorized interpolation
    path_x = path[..., 0]  # [B, K, T+1]
    path_y = path[..., 1]  # [B, K, T+1]

    # reshape for broadcasting
    cum_dist = cum_dist.view(1, 1, 1, T + 1)
    cum_dist_exp = cum_dist.view(1, 1, 1, -1).repeat(B, K, S, 1)  # already contiguous
    target = future_target_dists.view(1, 1, S, steps)        # [1,1,S,N]
    target = target.repeat(B, K, 1, 1)  # already contiguous
    # find segment indices
    idx = torch.searchsorted(cum_dist_exp, target, right=True) - 1
    idx = idx.clamp(min=0, max=T-1)

    # gather segment endpoints
    idx0 = idx
    idx1 = idx + 1

    def gather(tensor, idx):
        return torch.gather(
            tensor.unsqueeze(2).expand(-1, -1, S, -1),
            dim=3,
            index=idx
        )

    x0 = gather(path_x, idx0)
    x1 = gather(path_x, idx1)
    y0 = gather(path_y, idx0)
    y1 = gather(path_y, idx1)

    d0 = cum_dist[..., :-1].expand(B, K, S, -1).gather(3, idx0)
    d1 = cum_dist[..., 1:].expand(B, K, S, -1).gather(3, idx0)

    # linear interpolation
    w = (target - d0) / (d1 - d0 + 1e-8)

    interp_x = x0 + w * (x1 - x0)
    interp_y = y0 + w * (y1 - y0)

    p_e_future_points = torch.stack([interp_x, interp_y], dim=-1)  # [B,K,S,N,2]

    # prepend origin
    zeros = torch.zeros((B, K, S, 1, 2), device=device)
    ego_full_path = torch.cat([zeros, p_e_future_points], dim=3)  # [B,K,S,N+1,2]
    ego_yaws = get_stable_yaws_multipath(ego_full_path, initial_yaw=0.0, threshold=0.05)
    # yaw computation (still loop per batch element)
    # ego_yaws = torch.stack([
    #     torch.stack([
    #         torch.stack([
    #             torch.tensor(get_stable_yaws(ego_full_path[b, k, s].cpu().numpy(), 0.0, threshold=0.05))
    #             for s in range(S)
    #         ])
    #         for k in range(K)
    #     ])
    #     for b in range(B)
    # ])

    return ego_full_path, None, p_e_future_points, ego_yaws


def _compute_ttc_for_bin_efficient_multipath(
        speed, path_pred, scene_score_info,
        time_horizon, safe_distance,
        coll_radius, ego_wl, cuda_device
    ):
    """Efficient multipath version: path_pred [B, K, T, 2], scene_score_info is list of B dicts.
    speed: [S] speed bins
    Return: torch.Tensor of shape [B, K, S] with TTC values for each batch and path prediction, over different speed bins.
    """
    B, K, T, _ = path_pred.shape
    device = cuda_device
    calculation_type = "fut_wp_edge"  # or "fut_wp" or "fut_wp_edge"
    edge_by_edge = True if calculation_type == "fut_wp_edge" else False
    # Build ego trajs for all path predictions and speed bins
    ego_traj, p_e_future, wp_e_fut, ego_yaws = _build_ego_traj_multipath(path_pred, speed, time_horizon, steps=20)
    # Build padded actor info
    p_o_now, p_o_future, obj_wl, wp_o_fut, obj_mask, agent_yaws = _build_padded_actor_info(
        scene_score_info, 20, edge_by_edge=edge_by_edge, yaw_threshold=0.05
    )
    # Compute distances for all path predictions and speed bins
    # for now only has been implemented for fut_wp_edge, but can be implemented for other calculation types as well
    if calculation_type == "fut_wp_edge":
        distances = _compute_min_distance_traj_batched_multipath(
            ego_traj=wp_e_fut, # [B, K, S, N, 2]
            obj_traj=wp_o_fut, # [B, A, N, 2]
            ego_wl=ego_wl, # [2]
            obj_wl=obj_wl, # [B, A, 2]
            ego_yaws=ego_yaws, # [B, K, S, N+1]
            obj_yaws=agent_yaws, # [B, A, N+1]
            cuda_device=cuda_device
        )
    else:
        raise ValueError(f"Unknown calculation type: {calculation_type}")
    # Mask out invalid actors
    # distances: [B, K, S, A], obj_mask: [B, A] -> need to expand obj_mask to [B, 1, 1, A] for broadcasting
    obj_mask_exp = torch.from_numpy(obj_mask[:, None, None, :]).to(distances.device)
    distances = torch.where(obj_mask_exp, distances, torch.tensor(float('inf'), device=distances.device, dtype=distances.dtype))
    min_dists, _ = torch.min(distances, dim=3)  # [B, K, S]
    # Now compute TTC based on min_dists
    ttc = torch.where(
        min_dists >= safe_distance,
        torch.tensor(1.0, device=device),
        torch.where(
            min_dists <= coll_radius,
            torch.tensor(0.0, device=device),
            (min_dists - coll_radius) / (safe_distance - coll_radius)
        )
    )
    # move to cpu:
    return ttc.cpu()

def _compute_ttc_for_bin_efficient(
        speed, path_pred, scene_score_info,
        time_horizon, safe_distance,
        coll_radius, ego_wl, cuda_device
    ):
    """
    Batched version: path_pred [B, T, 2], scene_score_info is list of B dicts.
    This is the efficient version of _compute_ttc that instead of for loop, it 
    vectorrizes the computation.
    Returns: torch.Tensor of shape [B]
    """
    B, T, _ = path_pred.shape
    device = path_pred.device
    calculation_type = "fut_wp_edge"  # or "fut_wp" or "fut_wp_edge"
    edge_by_edge = True if calculation_type == "fut_wp_edge" else False
    # Build ego trajs
    ego_traj, p_e_future, wp_e_fut, ego_yaws = _build_ego_traj(path_pred, speed, time_horizon)
    # Build padded actor info
    p_o_now, p_o_future, obj_wl, wp_o_fut, obj_mask, agent_yaws = _build_padded_actor_info(
        scene_score_info, 20, edge_by_edge=edge_by_edge, yaw_threshold=0.05
    )
    # Match ego traj length to T
    p_e_now = np.zeros((B, 2), dtype=np.float32)

    # Compute distances
    if calculation_type == "p_o_fut":
        distances = _compute_swept_bbox_batched(
            p_e_now=p_e_now,
            p_e_future=p_e_future,
            p_o_now=p_o_now,
            p_o_future=p_o_future,
            ego_wl=ego_wl,
            obj_wl=obj_wl,
            steps=10 if time_horizon <= 1.0 else 20
        )
        # print("Batched distances:", distances)
    elif calculation_type == "fut_wp":
        distances = _compute_min_distance_traj_batched(
            ego_traj=wp_e_fut,
            obj_traj=wp_o_fut,
            ego_wl=ego_wl,
            obj_wl=obj_wl,
            ego_yaws=None,
            obj_yaws=None
        )
    elif calculation_type == "fut_wp_edge":
        distances = _compute_min_distance_traj_batched(
            ego_traj=wp_e_fut,
            obj_traj=wp_o_fut,
            ego_wl=ego_wl,
            obj_wl=obj_wl,
            ego_yaws=ego_yaws,
            obj_yaws=agent_yaws,
            cuda_device=cuda_device
        )
    else:
        raise ValueError(f"Unknown calculation type: {calculation_type}")
    results = []
    # distances = np.where(obj_mask, distances, np.inf)
    # min_dists = np.min(distances, axis=1)  # [B]
    distances = torch.where(torch.from_numpy(obj_mask).to(distances.device), distances, torch.tensor(float('inf'), device=distances.device, dtype=distances.dtype))
    min_dists, _ = torch.min(distances, dim=1)  # [B]
    for b in range(B):
        min_distance = min_dists[b]
        # print(f"Batch {b}, min_dist: {min_distance}")
        if min_distance >= safe_distance:
            results.append(1.0)
        elif min_distance <= coll_radius:
            results.append(0.0)
        else:
            ttc = (min_distance - coll_radius) / (safe_distance - coll_radius)
            results.append(ttc)
    return torch.tensor(results, device=device)

def compute_ttc(path_pred, scene_score_info, speed_bins,
                time_horizon, safe_distance, coll_radius,
                ego_wl, cuda_device
    ):
    """
    Compute time-to-collision (TTC) for self.speed_bins based on scene information.
    path_pred: Tensor of shape [B, K, T, 2] - predicted future path of the ego vehicle, B is batch size, K is number of path predictions, T is time steps
    
    Args:
        scene_info: Tensor of shape [B] - scene information for the current step
    """
    # Compute TTC for each speed bin based on scene information
    ttc = _compute_ttc_for_bin_efficient_multipath(
        speed=torch.tensor(speed_bins, device=path_pred.device, dtype=path_pred.dtype),
        path_pred=path_pred,
        scene_score_info=scene_score_info,
        time_horizon=time_horizon,
        safe_distance=safe_distance,
        coll_radius=coll_radius,
        ego_wl=ego_wl,
        cuda_device=cuda_device
    ) # Shape: [B, K, num_bins]
    return ttc
    ttc = []
    for speed in speed_bins:
        # Compute TTC for this bin edge
        # speed_score = _compute_ttc_for_bin(
        #     speed=speed,
        #     path_pred=path_pred,
        #     scene_score_info=scene_score_info,
        #     time_horizon=time_horizon,
        #     safe_distance=safe_distance,
        #     coll_radius=coll_radius,
        #     ego_wl=ego_wl,
        # )
        # print("org speed_score:", speed_score)
        speed_score = _compute_ttc_for_bin_efficient(
            speed=speed,
            path_pred=path_pred,
            scene_score_info=scene_score_info,
            time_horizon=time_horizon,
            safe_distance=safe_distance,
            coll_radius=coll_radius,
            ego_wl=ego_wl,
            cuda_device=cuda_device
        )
        # print("efficient speed_score:", speed_score)
        ttc.append(speed_score)
    # return torch.stack(ttc).unsqueeze(0)  # Shape: [B, num_classes]
    return torch.stack(ttc, dim=0).T  # Shape: [B, num_bins]


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
    lateral_shifts = torch.linspace(-0.4, 0.4, steps=2, device=device)
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

def compute_overtake_adherence(
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
        overtake_info = scene_score_info[b].get("overtake_info", None)
        if overtake_info is not None:
            dist_to_trig = overtake_info["dist_to_trig"]
            expert_throttle = overtake_info["expert_throttle"]
            scen_type = overtake_info["type"][0]
            dist_trig_bound = [19, 21]
            if scen_type == "accidenttwoways":
                dist_trig_bound = [17, 21]
            elif scen_type == "constructionobstacletwoways":
                dist_trig_bound = [19, 21]
            # We can specify the type-based condition here
            if dist_trig_bound[0] <= dist_to_trig <= dist_trig_bound[1]:
                if expert_throttle == 1:
                    for idx, bin_speed in enumerate(speed_bins):
                        rules_score[idx] = 1.0 if idx != 0 else 0.0
                else:
                    for idx, bin_speed in enumerate(speed_bins):
                        rules_score[idx] = 0.0 if idx != 0 else 1.0
            elif dist_to_trig > dist_trig_bound[1]: # decrease speed gradually
                for idx, bin_speed in enumerate(speed_bins):
                    if bin_speed < 1e-2:
                        rules_score[idx] = 1.0
                        continue
                    required_decel = _required_decel_to_stop(
                        speed=bin_speed,
                        dist_to_trig=dist_to_trig,
                        stop_distance_from_trigger=stop_distance_from_trigger,
                    )
                    if math.isinf(required_decel):
                        rules_score[idx] = 0.0
                    else:
                        rules_score[idx] = _decel_to_score(required_decel, decel_min, decel_max)
            else:
                rules_score = [1.0 for _ in speed_bins]
        all_scores.append(rules_score)
    return torch.tensor(all_scores, dtype=torch.float32)  # Shape: [B, num_bins]