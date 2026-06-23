import os
import json
import gzip
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import math
from collections import defaultdict
import random

# ======================
# Parameters
# ======================
FPS = 10
FUTURE_FRAMES = 40
NUM_WORKERS = max(1, cpu_count() - 1)
TENDENCY_BIN = 5.0
PRINT_EVERY_SCENE = 5  # print stats once every PRINT_EVERY_SCENE scenes
MAX_EXTEND = 10.0       # virtual_target_speed max extend m/s # long virtual speed
# MAX_EXTEND = 3.0       # virtual_target_speed max extend m/s # short virtual speed
TIME_MIN = 0.5
TIME_MAX = 3.        # virtual_target_speed max extend m/s # long virtual speed
# TIME_MAX = 1.5 # virtual_target_speed max extend m/s # short virtual speed
random.seed(42)

# ======================
# tool function
# ======================
def compute_tendency_speed(curr_speed, fut_speed):
    """calculate tendency_speed"""
    if not fut_speed:
        return curr_speed

    v1 = fut_speed[0]
    if v1 == curr_speed:
        return curr_speed

    if v1 > curr_speed:  # accelerate trend
        max_v = v1
        prev = v1
        for v in fut_speed[1:]:
            if v >= prev:
                max_v = v
                prev = v
            else:
                break
        return max_v
    else:  #decelerate trend
        min_v = v1
        prev = v1
        for v in fut_speed[1:]:
            if v <= prev:
                min_v = v
                prev = v
            else:
                break
        return min_v

def compute_virtual_target_speed(curr_tendency, prev_tendency):
    """Continue this trend, calculate virtual_target_speed"""
    delta = (curr_tendency - prev_tendency) * FPS * random.uniform(TIME_MIN, TIME_MAX)
    delta = max(min(delta, MAX_EXTEND), -MAX_EXTEND)
    v_target = curr_tendency + delta
    return max(v_target, 0.0)

def speed_to_bin(v: float) -> int:
    return int(math.floor(v / TENDENCY_BIN))

def print_stats(count, sum_v, min_v, max_v, hist, name="Speed"):
    if count == 0:
        print(f"No {name} data yet.")
        return
    mean_v = sum_v / count
    print(f"\n====== {name} Stats (So Far) ======")
    print(f"Frames: {count};  Min: {min_v:.4f} m/s;  Max: {max_v:.4f} m/s;  Mean: {mean_v:.4f} m/s")
    print("Distribution:")
    dist_str = ""
    for b in sorted(hist.keys()):
        l = b * TENDENCY_BIN
        r = (b + 1) * TENDENCY_BIN
        dist_str += f"[{l:.1f}, {r:.1f}) m/s: {hist[b]};  "
    print(dist_str)
    print("====================================\n")

# ======================
# single scene processing
# ======================
def process_scene(scene_dir: str):
    anno_dir = os.path.join(scene_dir, "anno")
    if not os.path.isdir(anno_dir):
        return None

    out_dir = os.path.join(scene_dir, "appended_anno")
    os.makedirs(out_dir, exist_ok=True)

    frame_files = sorted(f for f in os.listdir(anno_dir) if f.endswith(".json.gz"))
    if not frame_files:
        return None

    annos = []
    speeds = []

    for fname in frame_files:
        fpath = os.path.join(anno_dir, fname)
        try:
            with gzip.open(fpath, "rt", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        speeds.append(float(data.get("speed", 0.0)))
        annos.append(data)

    local_stats_tendency = {
        "count": 0, "sum": 0.0, "min": float("inf"), "max": -float("inf"), "hist": defaultdict(int)
    }
    local_stats_virtual = {
        "count": 0, "sum": 0.0, "min": float("inf"), "max": -float("inf"), "hist": defaultdict(int)
    }
    local_stats_delta = {
        "count": 0, "sum": 0.0, "min": float("inf"), "max": -float("inf"), "hist": defaultdict(int)
    }

    prev_tendency = annos[0]["speed"] if annos else 0.0

    for i, fname in enumerate(frame_files):
        fut = speeds[i + 1 : i + 1 + FUTURE_FRAMES]
        curr_speed = speeds[i]

        t_speed = compute_tendency_speed(curr_speed, fut)
        v_target = compute_virtual_target_speed(t_speed, prev_tendency)

        annos[i]["fut_speed"] = fut
        annos[i]["tendency_speed"] = t_speed
        annos[i]["virtual_target_speed"] = v_target

        prev_tendency = t_speed

        # update tendency stats
        local_stats_tendency["count"] += 1
        local_stats_tendency["sum"] += t_speed
        local_stats_tendency["min"] = min(local_stats_tendency["min"], t_speed)
        local_stats_tendency["max"] = max(local_stats_tendency["max"], t_speed)
        local_stats_tendency["hist"][speed_to_bin(t_speed)] += 1

        # update virtual stats
        local_stats_virtual["count"] += 1
        local_stats_virtual["sum"] += v_target
        local_stats_virtual["min"] = min(local_stats_virtual["min"], v_target)
        local_stats_virtual["max"] = max(local_stats_virtual["max"], v_target)
        local_stats_virtual["hist"][speed_to_bin(v_target)] += 1
        
        local_stats_delta["count"] += 1
        local_stats_delta["sum"] += (v_target - curr_speed) 
        local_stats_delta["min"] = min(local_stats_delta["min"], v_target - curr_speed)
        local_stats_delta["max"] = max(local_stats_delta["max"], v_target - curr_speed)
        local_stats_delta["hist"][speed_to_bin(v_target - curr_speed)] += 1

        out_path = os.path.join(out_dir, fname)
        with gzip.open(out_path, "wt", encoding="utf-8") as f:
            json.dump(annos[i], f, indent=4)

    return local_stats_tendency, local_stats_virtual, local_stats_delta

# ======================
# scene collection
# ======================
def collect_scenes(root_dirs):
    scenes = []
    for root in root_dirs:
        for name in os.listdir(root):
            if name.startswith("invalid_"):
                continue
            path = os.path.join(root, name)
            if os.path.isdir(path):
                scenes.append(path)
    return scenes

# ======================
# main process
# ======================
def main(root_dirs):
    scenes = collect_scenes(root_dirs)
    print(f"Found {len(scenes)} scenes")

    total_tendency = {"count":0,"sum":0.0,"min":float("inf"),"max":-float("inf"),"hist":defaultdict(int)}
    total_virtual = {"count":0,"sum":0.0,"min":float("inf"),"max":-float("inf"),"hist":defaultdict(int)}
    total_delta = {"count":0,"sum":0.0,"min":float("inf"),"max":-float("inf"),"hist":defaultdict(int)}

    processed = 0

    with Pool(NUM_WORKERS) as pool:
        for res in tqdm(pool.imap_unordered(process_scene, scenes), total=len(scenes),
                        desc="Appending speeds"):
            if res is None:
                continue
            tendency_res, virtual_res, delta_res = res

            # global stats
            for key in ["count","sum","min","max"]:
                total_tendency[key] += tendency_res[key] if key=="count" or key=="sum" else 0
                total_virtual[key] += virtual_res[key] if key=="count" or key=="sum" else 0
                total_delta[key] += delta_res[key] if key=="count" or key=="sum" else 0

            total_tendency["min"] = min(total_tendency["min"], tendency_res["min"])
            total_tendency["max"] = max(total_tendency["max"], tendency_res["max"])
            total_virtual["min"] = min(total_virtual["min"], virtual_res["min"])
            total_virtual["max"] = max(total_virtual["max"], virtual_res["max"])
            total_delta["min"] = min(total_delta["min"], delta_res["min"])
            total_delta["max"] = max(total_delta["max"], delta_res["max"])

            for k,v in tendency_res["hist"].items():
                total_tendency["hist"][k] += v
            for k,v in virtual_res["hist"].items():
                total_virtual["hist"][k] += v
            for k,v in delta_res["hist"].items():
                total_delta["hist"][k] += v

            processed += 1
            if processed % PRINT_EVERY_SCENE == 0:
                print_stats(total_tendency["count"], total_tendency["sum"],
                            total_tendency["min"], total_tendency["max"],
                            total_tendency["hist"], name="Tendency Speed")

                print_stats(total_virtual["count"], total_virtual["sum"],
                            total_virtual["min"], total_virtual["max"],
                            total_virtual["hist"], name="Virtual Target Speed")
                
                print_stats(total_delta["count"], total_delta["sum"],
                            total_delta["min"], total_delta["max"],
                            total_delta["hist"], name="Delta Speed")

    print("\n========== Final Statistics ==========")
    print_stats(total_tendency["count"], total_tendency["sum"],
                total_tendency["min"], total_tendency["max"],
                total_tendency["hist"], name="Tendency Speed")

    print_stats(total_virtual["count"], total_virtual["sum"],
                total_virtual["min"], total_virtual["max"],
                total_virtual["hist"], name="Virtual Target Speed")
    
    print_stats(total_delta["count"], total_delta["sum"],
                total_delta["min"], total_delta["max"],
                total_delta["hist"], name="Delta Speed")

# ======================
# entry
# ======================
if __name__ == "__main__":
    main(
        ["/path/to/dataset"]
    )