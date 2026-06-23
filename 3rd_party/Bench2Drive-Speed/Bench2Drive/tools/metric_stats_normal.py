import json
import copy
import os
import numpy as np

# ------------------ success judgement ------------------

def is_success(record):
    """
    consistent with merge_route_json
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


# ------------------ merge ------------------

def load_and_merge_records(json_paths):
    all_records = []

    for p in json_paths:
        with open(p, "r") as f:
            data = json.load(f)
            for r in data["_checkpoint"]["records"]:
                r = copy.deepcopy(r)
                r.pop("index", None)
                all_records.append(r)

    all_records = sorted(all_records, key=lambda d: d.get("route_id", ""), reverse=True)
    return all_records


# ------------------ overall statistics ------------------

def compute_overall_stats(records):
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

    sr = [r["scores"]["score_route"] for r in records]
    sp = [r["scores"]["score_penalty"] for r in records]
    sc = [r["scores"]["score_composed"] for r in records]

    success_num = sum(is_success(r) for r in records)

    return {
        "eval num": eval_num,
        "driving score": float(np.mean(sc)),
        "success rate": success_num / eval_num,
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


# ------------------ main ------------------

def main():
    eval_json_paths = [
        "eval_result_1.json",
        "eval_result_2.json",
        "..."
    ]

    output_dir = "/the/folder/to/put/merged/results"
    output_name = "merged_results.json"
    os.makedirs(output_dir, exist_ok=True)

    # ===== merge =====
    records = load_and_merge_records(eval_json_paths)

    # ===== overall stats =====
    overall_stats = compute_overall_stats(records)

    # ===== write =====
    out_data = {
        "_checkpoint": {
            "records": records
        },
        "overall": overall_stats
    }
    for key in overall_stats:
        out_data[key] = overall_stats[key]

    out_path = os.path.join(output_dir, output_name)
    with open(out_path, "w") as f:
        json.dump(out_data, f, indent=2)

    print(f"[Done] eval_num={overall_stats['eval num']} -> {out_path}")


if __name__ == "__main__":
    main()
