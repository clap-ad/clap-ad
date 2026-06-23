import json
import xml.etree.ElementTree as ET
import numpy as np
from collections import defaultdict
import os


# ==============================
# Utility
# ==============================

def extract_difficulty(route_id):
    if "easy" in route_id:
        return "easy"
    if "medium" in route_id:
        return "medium"
    if "hard" in route_id:
        return "hard"
    return "unknown"


def parse_routes_xml(xml_path):
    """
    Returns:
    {
        route_id: {
            "waypoints": [(x,y,z)...],
            "overtake_triggers": [(x,y,z)...]
        }
    }
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    routes = {}

    for route in root.findall("route"):
        route_id = route.get("id")

        # waypoints
        waypoints = []
        for pos in route.find("waypoints").findall("position"):
            waypoints.append((
                float(pos.get("x")),
                float(pos.get("y")),
                float(pos.get("z"))
            ))

        # overtake triggers
        triggers = []
        scenarios = route.find("scenarios")
        if scenarios is not None:
            for s in scenarios.findall("scenario"):
                if s.get("type") == "OvertakeRoute":
                    tp = s.find("trigger_point")
                    triggers.append((
                        float(tp.get("x")),
                        float(tp.get("y")),
                        float(tp.get("z"))
                    ))

        routes[route_id] = {
            "waypoints": waypoints,
            "overtake_triggers": triggers
        }

    return routes

def compute_progress_ratio(waypoints, trigger):
    cum_dist = [0.0]
    for i in range(1, len(waypoints)):
        prev = np.array(waypoints[i - 1])
        cur = np.array(waypoints[i])
        d = np.linalg.norm(cur - prev)
        cum_dist.append(cum_dist[-1] + d)

    total_dist = cum_dist[-1]

    dists = []
    for i, wp in enumerate(waypoints):
        dist = np.linalg.norm(np.array(wp) - np.array(trigger))
        dists.append((dist, i))

    _, idx = min(dists)

    return cum_dist[idx] / total_dist


def compute_route_overtake_score(route_meta, record):
    """
    single route overtaking score
    """
    waypoints = route_meta["waypoints"]
    triggers = route_meta["overtake_triggers"]
    total = len(triggers)

    if total == 0:
        return None  # no overtaking scene

    score_route = record["scores"]["score_route"]

    infra = record.get("infractions", {})
    has_fail = bool(infra.get("overtake"))

    score_sum = 0

    for trigger in triggers:
        progress_ratio = compute_progress_ratio(waypoints, trigger)

        if score_route >= progress_ratio * 100:
            # triggered
            if has_fail:
                score_sum += 0
            else:
                score_sum += 100
        else:
            score_sum += 0

    return score_sum / total


# ==============================
# Statistics
# ==============================

def compute_stats(scores_dict):
    values = [v for v in scores_dict.values() if v is not None]

    if not values:
        return {
            "eval_num": 0,
            "mean": 100.0,
            "std": 0.0
        }

    return {
        "eval_num": len(values),
        "mean": float(np.mean(values)),
        "std": float(np.std(values))
    }


# ==============================
# Main
# ==============================

def main():
    route_xml = "/path/to/leaderboard/data/b2dspd_eval48.xml"
    score_json = "/result/with/overtake.json" # ({output_name}_with_overtake.json) in readme
    output_path = "overtake_results.json"

    routes_meta = parse_routes_xml(route_xml)

    with open(score_json, "r") as f:
        data = json.load(f)

    records = data["_checkpoint"]["records"]

    route_scores = {}
    by_difficulty = defaultdict(dict)

    for r in records:
        route_id_full = r["route_id"]

        # extract route_id from xml
        print(route_id_full)
        route_id = route_id_full.split("_")[1]

        if route_id not in routes_meta:
            continue

        score = compute_route_overtake_score(routes_meta[route_id], r)

        route_scores[route_id] = score

        diff = extract_difficulty(route_id)
        by_difficulty[diff][route_id] = score

    # ===== stat =====
    result = {
        "global": compute_stats(route_scores),
        "by_difficulty": {},
        "by_route": route_scores
    }

    for diff, scores in by_difficulty.items():
        result["by_difficulty"][diff] = compute_stats(scores)

    # ===== output =====
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print("Overtake score statistics saved to", output_path)


if __name__ == "__main__":
    main()
