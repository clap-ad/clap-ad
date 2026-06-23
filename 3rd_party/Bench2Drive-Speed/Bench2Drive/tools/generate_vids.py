import os
import re
import json
import math
import xml.etree.ElementTree as ET
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import cv2

IMG_FPS = 20.0
METRIC_FPS = 20.0
MAX_FRAMES = 2000

# ---------------- XML Parsing ----------------
def infer_scenario_mode_from_name(scene_name):
    name = scene_name.lower()
    if "overtake" in name and "follow" not in name:
        return "Overtake"
    return "Follow"


def load_routes(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    routes = {}

    for route in root.findall("route"):
        rid = route.attrib["id"]
        town = route.attrib["town"]

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

        routes[(town, rid)] = {
            "wps": wps,
            "cum_dist": cum_dist,
            "speed_segments": speed_segments,
            "length": float(cum_dist[-1]) if len(cum_dist) > 0 else 1.0
        }

    return routes


def query_target_speed(p, segments):
    for seg in segments:
        if seg["start"] <= p < seg["end"]:
            return seg["speed"]

    return segments[-1]["speed"] if segments else 0.0


# ---------------- Annotation ----------------
def load_metric(scene_dir):
    path = os.path.join(scene_dir, "metric_info.json")

    if not os.path.exists(path):
        return []

    with open(path, "r") as f:
        data = json.load(f)

    annos = []

    for frame_str, d in data.items():
        frame = int(frame_str)

        vx, vy, vz = d["velocity"]
        speed_norm = math.sqrt(vx**2 + vy**2 + vz**2)

        x, y = d["location"][:2]

        annos.append({
            "frame": frame,
            "t": frame / METRIC_FPS,
            "x": x,
            "y": y,
            "vx": vx,
            "vy": vy,
            "vz": vz,
            "speed_norm": speed_norm
        })

    annos.sort(key=lambda a: a["frame"])

    return annos


def project_to_route(annos, route):
    if len(route.get("wps", [])) == 0:
        return np.zeros(len(annos))

    wps = route["wps"]
    cum = route["cum_dist"]

    s_list = []

    for a in annos:
        pos = np.array([a["x"], a["y"]])
        idx = np.argmin(np.linalg.norm(wps - pos, axis=1))
        s_list.append(cum[idx])

    return np.array(s_list)


# ---------------- Video Generation ----------------
def create_video(scene_dir, route_info, output_dir):

    scene_name = os.path.basename(scene_dir)

    # metric: 2000 frames (for 1000 images)
    annos = load_metric(scene_dir)[:MAX_FRAMES * 2]

    if len(annos) == 0:
        return

    s_list = project_to_route(annos, route_info)

    scenario_mode = infer_scenario_mode_from_name(scene_name)

    rgb_folder = os.path.join(scene_dir, "rgb_front")

    if not os.path.exists(rgb_folder):
        return

    frames = sorted(
        [f for f in os.listdir(rgb_folder) if f.endswith(".png")]
    )[:MAX_FRAMES]

    if len(frames) == 0:
        return

    os.makedirs(output_dir, exist_ok=True)

    video_path = os.path.join(output_dir, f"{scene_name}.mp4")

    first_frame = cv2.imread(os.path.join(rgb_folder, frames[0]))

    if first_frame is None:
        return

    height, width = first_frame.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    out = cv2.VideoWriter(video_path, fourcc, IMG_FPS, (width, height))

    for i, frame_file in enumerate(frames):

        img = cv2.imread(os.path.join(rgb_folder, frame_file))

        if img is None:
            continue

        # time sync
        img_time = i / IMG_FPS
        metric_idx = int(img_time * METRIC_FPS)

        if metric_idx >= len(annos):
            metric_idx = len(annos) - 1

        v_actual = annos[metric_idx]["speed_norm"]

        p = s_list[metric_idx]

        v_target = query_target_speed(
            p / max(route_info.get("length", 1.0), 1e-6),
            route_info.get("speed_segments", [])
        )

        text = f"Command: {scenario_mode} | Target: {v_target:.2f} m/s | Actual: {v_actual:.2f} m/s"

        # black edge
        cv2.putText(img, text, (10, 56),
                    cv2.FONT_HERSHEY_DUPLEX, 1.4, (0, 0, 0), 5)

        # yellow text
        cv2.putText(img, text, (10, 56),
                    cv2.FONT_HERSHEY_DUPLEX, 1.4, (0, 255, 255), 2)

        out.write(img)

    out.release()


# ---------------- Main ----------------
def main(parent_dir, xml_path, max_workers=4):

    routes = load_routes(xml_path)

    output_videos_dir = os.path.join(parent_dir, "videos")

    os.makedirs(output_videos_dir, exist_ok=True)

    scene_dirs = [
        os.path.join(parent_dir, d)
        for d in os.listdir(parent_dir)
        if os.path.isdir(os.path.join(parent_dir, d))
    ]

    def process_scene(scene_dir):

        route_info = {}

        scene_name = os.path.basename(scene_dir)

        town_match = re.search(r"_Town([^_]+)", scene_name)
        route_match = re.search(r"_Route([^_]+)", scene_name)

        if town_match and route_match:

            town = "Town" + town_match.group(1)
            rid = route_match.group(1)

            route_info = routes.get(
                (town, rid),
                {
                    "wps": np.array([]),
                    "cum_dist": np.array([]),
                    "speed_segments": [],
                    "length": 1.0
                }
            )

        create_video(scene_dir, route_info, output_videos_dir)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:

        list(
            tqdm(
                executor.map(process_scene, scene_dirs),
                total=len(scene_dirs),
                desc="All scenes"
            )
        )


if __name__ == "__main__":

    main(
        "/path/to/evaluation/results",
        "/path/to/leaderboard/data/b2dspd_eval48.xml",
        max_workers=4
    )