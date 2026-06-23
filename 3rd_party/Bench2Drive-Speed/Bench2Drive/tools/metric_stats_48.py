import json
import copy
import os
from collections import defaultdict
import numpy as np

# ------------------ basic tools ------------------

def extract_difficulty(save_name):
    if save_name is None:
        return "unknown"
    if "easy" in save_name:
        return "easy"
    if "medium" in save_name:
        return "medium"
    if "hard" in save_name:
        return "hard"
    return "unknown"

def is_success(record):
    """
    transplanted from merge_route_json
    """
    if record.get("status") not in ("Completed", "Perfect"):
        return False

    infractions = record.get("infractions", {})
    for k, v in infractions.items():
        if k == "min_speed_infractions":
            continue
        if v:
            return False
    return True

def collect_scores(records):
    sr, sp, sc = [], [], []
    for r in records:
        s = r["scores"]
        sr.append(s["score_route"])
        sp.append(s["score_penalty"])
        sc.append(s["score_composed"])
    return sr, sp, sc

def compute_block_stats(records):
    """
    calculate a stats block (global or certain difficulty)
    """
    eval_num = len(records)

    if eval_num == 0:
        return {
            "eval_num": 0,
            "driving_score": 0.0,
            "success_rate": 0.0,
            "score_route": {"mean": 0.0, "std": 0.0},
            "score_penalty": {"mean": 0.0, "std": 0.0},
            "score_composed": {"mean": 0.0, "std": 0.0},
        }

    sr, sp, sc = collect_scores(records)
    success_num = sum(is_success(r) for r in records)

    return {
        "eval_num": eval_num,
        "driving_score": float(np.mean(sc)),
        "success_rate": success_num / eval_num,
        "score_route": {
            "mean": float(np.mean(sr)),
            "std": float(np.std(sr)),
        },
        "score_penalty": {
            "mean": float(np.mean(sp)),
            "std": float(np.std(sp)),
        },
        "score_composed": {
            "mean": float(np.mean(sc)),
            "std": float(np.std(sc)),
        },
    }

def compute_full_stats(records):
    """
    global + by_difficulty
    """
    result = {
        "global": compute_block_stats(records),
        "by_difficulty": {}
    }

    by_diff = defaultdict(list)
    for r in records:
        d = extract_difficulty(r.get("save_name"))
        by_diff[d].append(r)

    for d, recs in by_diff.items():
        result["by_difficulty"][d] = compute_block_stats(recs)

    return result

# ------------------ eliminate repetition ------------------

def parse_timestamp_from_save_name(save_name):
    """
    Extract MM_DD_hh_mm_ss timestamp from save_name
    Returning tuple(int, int, int, int, int) for comparison
    """
    if save_name is None:
        return (0, 0, 0, 0, 0)
    parts = save_name.split("_")
    if len(parts) < 5:
        return (0, 0, 0, 0, 0)
    try:
        month = int(parts[-5])
        day = int(parts[-4])
        hour = int(parts[-3])
        minute = int(parts[-2])
        second = int(parts[-1])
        return (month, day, hour, minute, second)
    except:
        return (0, 0, 0, 0, 0)

def load_records(json_paths):
    """
    load JSON, eliminate repetition, reserve newest save_name
    """
    route_dict = {}  # key: route_id_without_rep, value: record

    for p in json_paths:
        with open(p, "r") as f:
            data = json.load(f)
            for r in data["_checkpoint"]["records"]:
                r = copy.deepcopy(r)
                r.pop("index", None)

                base_route_id = "_".join(r["route_id"].split("_")[:-1])
                
                ts = parse_timestamp_from_save_name(r.get("save_name"))

                # reserve the newest record
                if base_route_id in route_dict:
                    print(f"[debug] detected multiple entries of {base_route_id}, save_name = '{r.get('save_name')}'")
                    existing_ts = parse_timestamp_from_save_name(route_dict[base_route_id].get("save_name"))
                    print(f"[debug] old timestamp = {existing_ts}, new timestamp = {ts}")
                    if ts > existing_ts:
                        print(f"[debug] updated to {r.get('save_name')}")
                        route_dict[base_route_id] = r
                else:
                    route_dict[base_route_id] = r

    return list(route_dict.values())

# ------------------ handling overtake ------------------

def recompute_without_overtake(records, overtake_penalty=0.7):
    new_records = []
    for r in records:
        r_new = copy.deepcopy(r)
        infra = r_new.get("infractions", {})
        scores = r_new["scores"]

        has_overtake = "overtake" in infra and infra["overtake"]
        if has_overtake:
            scores["score_penalty"] = scores["score_penalty"] / overtake_penalty

        scores["score_composed"] = scores["score_route"] * scores["score_penalty"]
        new_records.append(r_new)

    return new_records

# ------------------ main ------------------

def main():
    eval_json_paths = [
        "/path/to/json/result/1",
        "/path/to/json/result/2",
        "..."
    ]

    output_dir = "/the/folder/you/want/to/put/result/in"
    output_name = "output_file_name"
    os.makedirs(output_dir, exist_ok=True)

    # ====== merge + eliminate repetition ======
    records_with_overtake = load_records(eval_json_paths)
    records_traditional = recompute_without_overtake(records_with_overtake)

    # ====== statistics ======
    stats = {
        "traditional": compute_full_stats(records_traditional),
        "with_overtake": compute_full_stats(records_with_overtake)
    }

    # ====== write json ======
    with open(os.path.join(output_dir, f"{output_name}_with_overtake.json"), "w") as f:
        json.dump({"_checkpoint": {"records": records_with_overtake}}, f, indent=2)

    with open(os.path.join(output_dir, f"{output_name}_traditional.json"), "w") as f:
        json.dump({"_checkpoint": {"records": records_traditional}}, f, indent=2)

    with open(os.path.join(output_dir, f"{output_name}_statistic.json"), "w") as f:
        json.dump(stats, f, indent=2)

if __name__ == "__main__":
    main()