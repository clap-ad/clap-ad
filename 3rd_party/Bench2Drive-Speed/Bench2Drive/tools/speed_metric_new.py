import os
import re
import json
import math
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict

# ================== Hyper-parameters ==================
FPS = 10.0
ALPHA = 1.0
FOLLOW_TOLERANCE = 0.75
# ======================================================


# ======================================================
# ---------------- XML Parsing --------------------------
# ======================================================

def formalize_scenario_name(scene_name):
    if "+" in scene_name:
        scene_name = scene_name.split("+")[1]
    if "Road" in scene_name:
        scene_name = scene_name.split("_")
        scene_name = scene_name[:2] + scene_name[3:]
        scene_name = "_".join(scene_name)
        scene_name = scene_name.replace("-", "_")
    return scene_name

def infer_scenario_mode(route):
    scenarios = route.find("scenarios")
    if scenarios is None:
        return "Follow"

    for sc in scenarios.findall("scenario"):
        sc_type = sc.attrib.get("type", "")
        if "Overtake" in sc_type:
            behavior = sc.find("behavior")
            if behavior is not None and behavior.attrib.get("value") == "overtake":
                return "Overtake"
    return "Follow"


def parse_follow_constraint(route, wps, cum_dist):
    scenarios = route.find("scenarios")
    if scenarios is None:
        return None

    for sc in scenarios.findall("scenario"):
        sc_type = sc.attrib.get("type", "")
        behavior = sc.find("behavior")

        if "Overtake" in sc_type and behavior is not None and behavior.attrib.get("value") == "follow":
            tp = sc.find("trigger_point")
            sp = sc.find("speed")
            if tp is None or sp is None:
                continue

            tp_xy = np.array([float(tp.attrib["x"]), float(tp.attrib["y"])])
            idx = np.argmin(np.linalg.norm(wps - tp_xy, axis=1))
            trigger_s = cum_dist[idx]
            lead_speed = float(sp.attrib["value"]) / 3.6

            return {
                "trigger_s": float(trigger_s),
                "lead_speed": float(lead_speed)
            }
    return None


def load_routes(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    routes = {}

    for route in root.findall("route"):
        rid = route.attrib["id"]
        town = route.attrib["town"]

        scenario_mode = infer_scenario_mode(route)

        wps = []
        for p in route.find("waypoints").findall("position"):
            wps.append([float(p.attrib["x"]), float(p.attrib["y"])])
        wps = np.array(wps)

        cum_dist = [0.0]
        for i in range(1, len(wps)):
            cum_dist.append(cum_dist[-1] + np.linalg.norm(wps[i] - wps[i - 1]))
        cum_dist = np.array(cum_dist)

        speed_segments = []
        sc = route.find("speed_config")
        if sc is not None:
            for seg in sc.findall("segment"):
                speed_segments.append({
                    "start": float(seg.attrib["start"]),
                    "end": float(seg.attrib["end"]),
                    "speed": float(seg.attrib["target_speed"]) / 3.6
                })

        follow_constraint = parse_follow_constraint(route, wps, cum_dist)

        key = (town, rid, scenario_mode)
        routes[key] = {
            "wps": wps,
            "cum_dist": cum_dist,
            "length": float(cum_dist[-1]),
            "speed_segments": speed_segments,
            "follow_constraint": follow_constraint
        }

    return routes


def query_target_speed(p, segments):
    for seg in segments:
        if seg["start"] <= p < seg["end"]:
            return seg["speed"]
    return segments[-1]["speed"]


# ======================================================
# ---------------- Scene Loading ------------------------
# ======================================================

def load_anno(scene_dir):
    path = os.path.join(scene_dir, "metric_info.json")
    if not os.path.exists(path):
        return []

    with open(path, "r") as f:
        data = json.load(f)

    annos = []
    for frame_str, d in data.items():
        frame = int(frame_str)

        vx, vy, vz = d["velocity"]
        speed_norm = math.sqrt(vx * vx + vy * vy + vz * vz)

        x, y = d["location"][:2]

        annos.append({
            "t": frame / FPS,
            "x": x,
            "y": y,
            "vx": vx,
            "vy": vy,
            "vz": vz,
            "speed_norm": speed_norm
        })

    annos.sort(key=lambda a: a["t"])
    return annos


def project_to_route(annos, route):
    wps = route["wps"]
    cum = route["cum_dist"]

    s_list = []
    for a in annos:
        pos = np.array([a["x"], a["y"]])
        idx = np.argmin(np.linalg.norm(wps - pos, axis=1))
        s_list.append(cum[idx])
    return np.array(s_list)


# ======================================================
# ---------------- Speed Metric -------------------------
# ======================================================

def compute_speed_metric_v2(annos, s_list, route):
    speed_segments = route["speed_segments"]
    follow_constraint = route.get("follow_constraint", None)
    total_len = route["length"]

    weighted_score_sum = 0.0
    weight_sum = 0.0
    details = []

    prev_pos = None

    for i, a in enumerate(annos):
        pos = np.array([a["x"], a["y"]])

        if prev_pos is None:
            prev_pos = pos
            continue

        w = np.linalg.norm(pos - prev_pos)
        prev_pos = pos

        if w < 1e-4:
            continue

        v_actual = a["speed_norm"]

        s = s_list[i]
        p = s / max(total_len, 1e-6)
        v_target = query_target_speed(p, speed_segments)

        err = abs(v_actual - v_target) / max(v_target, 1e-3)
        score = math.exp(-ALPHA * err)

        if follow_constraint is not None:
            trigger_s = follow_constraint["trigger_s"]
            v_lead = follow_constraint["lead_speed"]

            if s >= trigger_s and v_lead <= v_actual < v_target:
                score = 1.0 - (1.0 - score) * (1.0 - FOLLOW_TOLERANCE)

        weighted_score_sum += w * score
        weight_sum += w

        details.append({
            "s": float(s),
            "v_actual": float(v_actual),
            "v_target": float(v_target),
            "weight": float(w),
            "score": float(score)
        })

    final_score = weighted_score_sum / weight_sum if weight_sum > 0 else 0.0
    return float(final_score), details


# ======================================================
# ---------------- Visualization ------------------------
# ======================================================

def compute_tangent_speed(annos, route):
    wps = route["wps"]
    v_tangent = []

    for a in annos:
        pos = np.array([a["x"], a["y"]])
        idx = np.argmin(np.linalg.norm(wps - pos, axis=1))

        if idx < len(wps) - 1:
            tangent = wps[idx + 1] - wps[idx]
        else:
            tangent = wps[idx] - wps[idx - 1]

        norm = np.linalg.norm(tangent)
        if norm < 1e-6:
            v_tangent.append(0.0)
            continue

        tangent = tangent / norm
        v_vec = np.array([a["vx"], a["vy"]])

        v_tangent.append(float(np.dot(v_vec, tangent)))

    return np.array(v_tangent)


def plot_speed_time(scene_dir, scene_name, annos, v_tangent):
    t = [a["t"] for a in annos]
    v_norm = [a["speed_norm"] for a in annos]

    plt.figure(figsize=(10, 4))
    plt.plot(t, v_norm, label="Sensor speed |v|", linewidth=1.5)
    plt.plot(t, v_tangent, label="Tangent speed", linestyle="--")
    plt.xlabel("Time (s)")
    plt.ylabel("Speed (m/s)")
    plt.title(scene_name)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(scene_dir, "speed_time.png"))
    plt.close()


def plot_speed_profile_v2(scene_dir, scene_name, details):
    if not details:
        return

    s = [d["s"] for d in details]
    v_actual = [d["v_actual"] for d in details]
    v_target = [d["v_target"] for d in details]

    plt.figure(figsize=(10, 4))
    plt.plot(s, v_actual, label="Actual speed", linewidth=1.5)
    plt.plot(s, v_target, label="Target speed", linestyle="--")
    plt.xlabel("Route position s (m)")
    plt.ylabel("Speed (m/s)")
    plt.title(scene_name)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(scene_dir, "speed_profile.png"))
    plt.close()


# ======================================================
# ---------------- Scene Meta ---------------------------
# ======================================================

def parse_scene_key(scene_name):
    m = re.search(r"_Town([^_]+)_Route([^_]+)_", scene_name)
    if not m:
        return None

    town = "Town" + m.group(1)
    rid = m.group(2)
    scenario_mode = "Overtake" if ("Overtake" in scene_name and "Follow" not in scene_name) else "Follow"
    return town, rid, scenario_mode


def parse_scene_meta(scene_name):
    parts = scene_name.split("_")
    timestamp = tuple(map(int, parts[-5:]))
    route_key = "_".join(parts[:-6])

    difficulty = None
    for d in ["easy", "medium", "hard"]:
        if d in scene_name:
            difficulty = d
            break

    return route_key, difficulty, timestamp


# ======================================================
# ---------------- Main Pipeline ------------------------
# ======================================================

def main(output_dir, xml_path):
    routes = load_routes(xml_path)

    scene_records = []

    scenes = [s for s in os.listdir(output_dir)
              if os.path.isdir(os.path.join(output_dir, s))]

    for scene in tqdm(scenes, desc="Processing scenes"):
        scene_dir = os.path.join(output_dir, scene)

        scene = formalize_scenario_name(scene)

        key = parse_scene_key(scene)
        if key is None or key not in routes:
            continue

        annos = load_anno(scene_dir)
        if not annos:
            continue

        s_list = project_to_route(annos, routes[key])

        v_tangent = compute_tangent_speed(annos, routes[key])
        plot_speed_time(scene_dir, scene, annos, v_tangent)

        score, details = compute_speed_metric_v2(annos, s_list, routes[key])

        with open(os.path.join(scene_dir, "speed_result.json"), "w") as f:
            json.dump({
                "scene": scene,
                "town": key[0],
                "route_id": key[1],
                "scenario_mode": key[2],
                "score": score,
                "frames": details
            }, f, indent=2)

        plot_speed_profile_v2(scene_dir, scene, details)

        route_key, difficulty, timestamp = parse_scene_meta(scene)
        completion = max(s_list) / max(routes[key]["length"], 1e-6)
        completion_bin = int(completion * 10)

        scene_records.append({
            "scene": scene,
            "route_key": route_key,
            "difficulty": difficulty,
            "timestamp": timestamp,
            "completion": completion,
            "completion_bin": completion_bin,
            "score": score
        })


    # ---------------- Eliminate repetitions: according to route_key ----------------
    best_by_route = {}
    for r in scene_records:
        k = r["route_key"]
        if k not in best_by_route or r["timestamp"] > best_by_route[k]["timestamp"]:
            best_by_route[k] = r
    # best_by_route = {}
    # for r in scene_records:
    #     k = r["route_key"]
    #     if k not in best_by_route:
    #         best_by_route[k] = r
    #     else:
    #         old = best_by_route[k]
    #         if r["completion_bin"] > old["completion_bin"] or (
    #             r["completion_bin"] == old["completion_bin"]
    #             and r["timestamp"] > old["timestamp"]
    #         ):
    #             best_by_route[k] = r

    final_records = list(best_by_route.values())
    
    # ---------------- Statistics ----------------
    all_scores = [r["score"] for r in final_records]

    overall = {
        "mean": float(np.mean(all_scores)) if all_scores else 0.0,
        "std": float(np.std(all_scores)) if all_scores else 0.0,
        "num_scenes": len(all_scores)
    }

    by_difficulty = defaultdict(list)
    for r in final_records:
        if r["difficulty"]:
            by_difficulty[r["difficulty"]].append(r["score"])

    difficulty_stats = {
        d: {
            "mean": float(np.mean(v)),
            "std": float(np.std(v)),
            "num": len(v)
        }
        for d, v in by_difficulty.items()
    }

    # -------- scenes are bucketified by difficulty --------
    scenes_by_difficulty = defaultdict(dict)
    for r in final_records:
        d = r["difficulty"] if r["difficulty"] is not None else "unknown"
        scenes_by_difficulty[d][r["scene"]] = {
            "score": r["score"],
            "completion": r["completion"]
        }

    summary = {
        "overall": overall,
        "by_difficulty": difficulty_stats,
        "scenes": dict(scenes_by_difficulty)
    }

    out_path = os.path.join(output_dir, "speed_summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    import argparse

    DEFAULT_XML = "/workspace/lead/3rd_party/Bench2Drive-Speed/Bench2Drive/leaderboard/data/b2dspd_eval48.xml"

    parser = argparse.ArgumentParser(
        description="Compute the desired-speed metric for a Bench2Drive-Speed evaluation run."
    )
    parser.add_argument(
        "output_dir",
        help="Path to the evaluation results directory (one subfolder per route).",
    )
    parser.add_argument(
        "-x",
        "--xml",
        default=DEFAULT_XML,
        help="Path to the routes XML file (default: %(default)s).",
    )
    args = parser.parse_args()

    main(args.output_dir, args.xml)
