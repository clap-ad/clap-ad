import argparse
import xml.etree.ElementTree as ET
import yaml
import os
from collections import defaultdict


def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    input_xml = config.get("input_xml")
    output_xml = config.get("output_xml", "output.xml")

    scenario_cfg = config.get("scenario", {})
    road_cfg = config.get("road", {})

    return {
        "input_xml": input_xml,
        "output_xml": output_xml,
        "scenario_mode": scenario_cfg.get("mode"),
        "scenario_list": scenario_cfg.get("list", []),
        "road_mode": road_cfg.get("mode"),
        "road_list": road_cfg.get("list", []),
    }


def print_config_summary(cfg):
    print("===================================")
    print(" Loaded Configuration Summary")
    print("===================================")
    print(f"Input XML : {cfg['input_xml']}")
    print(f"Output XML: {cfg['output_xml']}")
    print("-----------------------------------")

    print("Scenario filter:")
    if cfg["scenario_mode"]:
        print(f"  Mode: {cfg['scenario_mode']}")
        print("  List:")
        for s in cfg["scenario_list"]:
            print(f"    - {s}")
    else:
        print("  (No scenario filter)")

    print("\nRoad filter:")
    if cfg["road_mode"]:
        print(f"  Mode: {cfg['road_mode']}")
        print("  List:")
        for r in cfg["road_list"]:
            print(f"    - {r.get('town')} / Road {r.get('road_id')}")
    else:
        print("  (No road filter)")
    print("===================================")

all_scenarios = []

def should_keep_route(route, cfg):
    town = route.get("town")
    road_id = route.get("road_id")
    # if not road_id:
    #     return False

    # Road filtering
    if cfg["road_mode"] and cfg["road_list"]:
        road_match = any(
            (r.get("town") == town and str(r.get("road_id")) == str(road_id))
            for r in cfg["road_list"]
        )
        if cfg["road_mode"] == "include" and not road_match:
            return False
        if cfg["road_mode"] == "exclude" and road_match:
            return False

    # Scenario filtering
    if cfg["scenario_mode"] and cfg["scenario_list"]:
        scenario_types = [s.get("type") for s in route.findall("scenarios/scenario")]
        for s in scenario_types:
            if s not in all_scenarios:
                print(f"[debug] doing {s}, len(all_scenarios) = {len(all_scenarios)}")
                all_scenarios.append(s)
        if cfg["scenario_mode"] == "include":
            if not any(s in cfg["scenario_list"] for s in scenario_types):
                return False
        elif cfg["scenario_mode"] == "exclude":
            if any(s in cfg["scenario_list"] for s in scenario_types):
                return False

    return True


def process_xml(cfg):
    input_xml = cfg["input_xml"]
    output_xml = cfg["output_xml"]

    if not os.path.exists(input_xml):
        print(f"❌ Input XML file not found: {input_xml}")
        return None

    tree = ET.parse(input_xml)
    root = tree.getroot()
    new_root = ET.Element("routes")

    kept_routes = []
    for route in root.findall("route"):
        if should_keep_route(route, cfg):
            new_root.append(route)
            kept_routes.append(route)

    ET.ElementTree(new_root).write(output_xml, encoding="utf-8", xml_declaration=True)
    print(f"\n✅ Filtered XML saved to: {output_xml}")
    print(f"   Routes kept: {len(kept_routes)}/{len(root.findall('route'))}")
    return kept_routes


def balance_scenarios(routes, output_xml):
    print("\n⚖️  Balancing scenario route counts (per-route removal by road)...")

    def build_scenario_map(routes):
        m = defaultdict(list)
        for r in routes:
            for s in r.findall("scenarios/scenario"):
                m[s.get("type")].append(r)
        return m

    def get_route_key(route):
        return (route.get("town"), route.get("road_id"), route.get("id"))

    scenario_to_routes = build_scenario_map(routes)
    if not scenario_to_routes:
        print("⚠️  No scenarios found, skip balancing.")
        return

    scenario_counts = {k: len(v) for k, v in scenario_to_routes.items()}
    min_count = min(scenario_counts.values())
    total_roads = len({(r.get("town"), r.get("road_id")) for r in routes})

    print(f"Initial scenario counts: {scenario_counts}")
    print(f"Total distinct roads: {total_roads}")
    print(f"Default lower bound = max(min_count={min_count}, total_roads={total_roads})")

    # 🆕 Ask user for lower bound
    user_input = input("Enter a lower bound for balancing (press Enter to use default): ").strip()
    if user_input.isdigit():
        lower_bound = int(user_input)
        print(f"✅ Using custom lower bound: {lower_bound}")
    else:
        lower_bound = max(min_count, total_roads)
        print(f"✅ Using default lower bound: {lower_bound}")

    target_count = max(lower_bound, min_count, total_roads)
    print(f"→ Target per-scenario route count: {target_count}")
    print("-----------------------------------")

    removed_routes = set()

    changed = True
    while changed:
        changed = False
        scenario_to_routes = build_scenario_map(routes)
        scenario_counts = {k: len(v) for k, v in scenario_to_routes.items()}

        for scenario, sc_routes in scenario_to_routes.items():
            if len(sc_routes) <= target_count:
                continue  # Already balanced

            changed = True
            road_to_routes = defaultdict(list)
            for r in sc_routes:
                road_to_routes[(r.get("town"), r.get("road_id"))].append(r)

            most_loaded_road = max(road_to_routes.items(), key=lambda x: len(x[1]))
            road_key, candidate_routes = most_loaded_road

            route_score = {r: len(r.findall("scenarios/scenario")) for r in candidate_routes}
            worst_route = max(route_score, key=route_score.get)

            routes.remove(worst_route)
            removed_routes.add(get_route_key(worst_route))

        scenario_to_routes = build_scenario_map(routes)
        if all(len(v) <= target_count for v in scenario_to_routes.values()):
            break

    print("-----------------------------------")
    print(f"Removed {len(removed_routes)} routes.")
    print(f"Remaining {len(routes)} routes.")

    new_root = ET.Element("routes")
    for r in routes:
        new_root.append(r)
    balanced_path = f"{output_xml}"
    ET.ElementTree(new_root).write(balanced_path, encoding="utf-8", xml_declaration=True)

    final_counts = {k: len(v) for k, v in build_scenario_map(routes).items()}
    print(f"✅ Balanced XML saved as: {balanced_path}")
    print(f"Final scenario counts: {final_counts}")
    print("===================================")



def main():
    parser = argparse.ArgumentParser(description="CARLA Route XML Filter Tool (YAML-driven + Balance)")
    parser.add_argument("--config", "-f", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print("❌ The specified YAML config file does not exist.")
        return

    print("===================================")
    print("   CARLA Route XML Filter Tool")
    print("===================================")

    cfg = load_config(args.config)
    print_config_summary(cfg)

    confirm = input("Proceed with these settings? (Y/N): ").strip().lower()
    if confirm != "y":
        print("❌ Operation cancelled by user.")
        return

    kept_routes = process_xml(cfg)
    if not kept_routes:
        print("❌ No routes were kept after filtering.")
        return

    balance = input("Would you like to balance scenario route counts? (Y/N): ").strip().lower()
    if balance == "y":
        balance_scenarios(kept_routes, cfg["output_xml"])
    else:
        print("Skipping balancing step.")


if __name__ == "__main__":
    main()
