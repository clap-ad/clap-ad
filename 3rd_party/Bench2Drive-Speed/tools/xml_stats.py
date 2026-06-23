import argparse
import xml.etree.ElementTree as ET
from collections import defaultdict
import os
import yaml

def parse_routes(xml_path):
    """
    Parse CARLA routes XML file.
    Extract route info (id, road_id, town, scenarios).
    Returns:
        roads: dict[(town, road_id) -> list of route info]
        scenario_count: dict[scenario_type -> total count across all routes]
        total_routes: int
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    roads = defaultdict(list)
    scenario_count = defaultdict(int)
    total_routes = 0

    for route in root.findall("route"):
        route_id = route.get("id")
        road_id = route.get("road_id")
        town = route.get("town")

        if not road_id:
            continue  # Skip routes without road_id

        total_routes += 1
        route_info = {
            "id": route_id,
            "road_id": road_id,
            "town": town,
            "scenarios": []
        }

        scenarios = route.find("scenarios")
        if scenarios is not None:
            for sc in scenarios.findall("scenario"):
                scenario_type = sc.get("type")
                scenario_count[scenario_type] += 1
                route_info["scenarios"].append(scenario_type)

        roads[(town, road_id)].append(route_info)

    return roads, scenario_count, total_routes

def generate_markdown(roads, scenario_count, total_routes):
    """
    Generate Markdown summary grouped by (town, road_id):
      - Town / Road
      - ScenarioCount (unique scenario types)
      - RouteCount (total routes)
      - ScenarioNames (list of scenario types)
      - ExcludedScenarios (scenario types not present on this road)
      - Each scenario column lists route IDs
    """
    md = []
    md.append("# CARLA Route Statistics\n")

    md.append("## 1. Summary\n")
    md.append(f"- Total valid routes (with road_id): **{total_routes}**\n")
    md.append(f"- Total unique roads: **{len(roads)}**\n")
    md.append("- Scenario type counts (global):\n")
    for t, c in scenario_count.items():
        md.append(f"  - **{t}**: {c}\n")

    md.append("\n## 2. Detailed Table (grouped by Town + Road, sorted by ScenarioCount)\n")

    all_scenarios = sorted(scenario_count.keys())
    header = "| Town / Road | ScenarioCount | RouteCount | ScenarioNames | ExcludedScenarios | " + " | ".join(all_scenarios) + " |"
    md.append(header)
    md.append("|" + "|".join(["---"] * (len(all_scenarios) + 5)) + "|")

    merged_data = []
    for (town, road_id), routes in roads.items():
        scenario_to_routes = defaultdict(list)
        all_scenarios_in_road = set()
        for route in routes:
            for sc in route["scenarios"]:
                scenario_to_routes[sc].append(route["id"])
                all_scenarios_in_road.add(sc)

        scenario_count_unique = len(all_scenarios_in_road)
        route_count = len(routes)
        merged_data.append((town, road_id, scenario_to_routes, scenario_count_unique, route_count, all_scenarios_in_road))

    merged_data.sort(key=lambda x: (x[3], x[4]), reverse=True)

    for town, road_id, scenario_map, scenario_count_unique, route_count, scenario_names_set in merged_data:
        row_label = f"{town} / Road {road_id}"
        scenario_names = ", ".join(sorted(scenario_names_set)) if scenario_names_set else ""
        excluded_scenarios = ", ".join(sorted(set(all_scenarios) - scenario_names_set)) if scenario_names_set else ", ".join(all_scenarios)
        row = [row_label, str(scenario_count_unique), str(route_count), scenario_names, excluded_scenarios]
        for sc_type in all_scenarios:
            if sc_type in scenario_map:
                route_list = ", ".join(scenario_map[sc_type])
                row.append(route_list)
            else:
                row.append("")
        md.append("| " + " | ".join(row) + " |")

    return "\n".join(md)

def save_files(markdown_text, output_md, output_html):
    with open(output_md, "w", encoding="utf-8") as f:
        f.write(markdown_text)

    try:
        import markdown
        html_content = markdown.markdown(
            markdown_text,
            extensions=["tables", "fenced_code"]
        )
        styled_html = f"""
        <html>
        <head>
            <meta charset="utf-8">
            <title>CARLA Route Statistics</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 40px;
                    background-color: #fafafa;
                }}
                table {{
                    border-collapse: collapse;
                    width: 100%;
                    margin-top: 20px;
                    display: block;
                    overflow-x: auto;
                    white-space: nowrap;
                }}
                th, td {{
                    border: 1px solid #ccc;
                    padding: 6px 10px;
                    text-align: left;
                }}
                th {{
                    background-color: #eee;
                }}
                h1, h2 {{
                    color: #333;
                }}
            </style>
        </head>
        <body>
        {html_content}
        </body>
        </html>
        """
        with open(output_html, "w", encoding="utf-8") as f:
            f.write(styled_html)
        print(f"HTML version generated: {output_html}")
    except ImportError:
        print("⚠️ HTML export skipped (module 'markdown' not found).")
        print("   You can install it with:  pip install markdown")

def greedy_subset_config(roads, scenario_count, total_limit, limit_mode="route"):
    """
    Generate subset_config.yml using greedy algorithm:
    1. Start with all scenarios count = 0.
    2. Iteratively pick the scenario with fewest routes, select road covering it with most scenarios.
    3. Stop when total routes >= total_limit.
    """
    all_scenarios = sorted(scenario_count.keys())
    road_info = {}
    scenario_to_roads = defaultdict(set)
    for (town, road_id), routes in roads.items():
        road_scenarios = set()
        for route in routes:
            road_scenarios.update(route["scenarios"])
        road_info[(town, road_id)] = {"route_count": len(routes), "scenarios": road_scenarios}
        for sc in road_scenarios:
            scenario_to_roads[sc].add((town, road_id))

    scenario_counter = {s:0 for s in all_scenarios}
    selected_roads = set()
    total_selected_routes = 0

    qualified = False
    while not qualified:
        # find scenario with fewest selected routes
        s_min = min(scenario_counter.items(), key=lambda x: x[1])[0]
        # candidate roads containing s_min and not yet selected
        candidates = [r for r in scenario_to_roads[s_min] if r not in selected_roads]
        if not candidates:
            break
        # pick road with most scenarios (greedy)
        road_to_add = max(candidates, key=lambda r: len(road_info[r]["scenarios"]))
        selected_roads.add(road_to_add)
        total_selected_routes += road_info[road_to_add]["route_count"]
        # update scenario_counter
        for sc in road_info[road_to_add]["scenarios"]:
            scenario_counter[sc] += road_info[road_to_add]["route_count"]
        qualified = (total_selected_routes >= total_limit and limit_mode=="route") or \
                    len(selected_roads) >= total_limit and limit_mode=="road"

    # build subset_config.yml
    subset_config = {
        "scenario": {"mode": "include", "list": sorted([sc for sc, cnt in scenario_counter.items() if cnt>0])},
        "road": {"mode": "include", "list": [{"town": t, "road_id": r} for t,r in sorted(selected_roads)]}
    }
    return subset_config, total_selected_routes

def main():
    parser = argparse.ArgumentParser(description="CARLA Route Statistics & Subset Tool")
    parser.add_argument("--xml", "-f", required=True, help="Path to CARLA route XML file")
    args = parser.parse_args()

    if not os.path.exists(args.xml):
        print("The specified XML file does not exist. Please check the path.")
        return

    print("===================================")
    print("   CARLA Route Statistics Tool")
    print("===================================")
    print("Type 's' to start statistics, 'n' to generate subset config with route number limit, 'r' to generate subset config with road number limit, or any other key to exit.")
    cmd = input(">> ").strip().lower()

    roads, scenario_count, total_routes = parse_routes(args.xml)

    if cmd == "s":
        markdown_text = generate_markdown(roads, scenario_count, total_routes)
        output_md = "stats.md"
        output_html = "stats.html"
        save_files(markdown_text, output_md, output_html)
        print(f"✅ Markdown version generated: {output_md}")
        print("All done!")
    elif cmd == "n":
        limit_input = input("Enter total route limit (integer): ").strip()
        try:
            total_limit = int(limit_input)
        except ValueError:
            print("Invalid input. Exiting.")
            return
        subset_config, selected_routes = greedy_subset_config(roads, scenario_count, total_limit, "route")
        output_yaml = "subset_config.yml"
        with open(output_yaml, "w", encoding="utf-8") as f:
            yaml.dump(subset_config, f, sort_keys=False)
        print(f"✅ Subset config generated: {output_yaml}")
        print(f"Total routes selected: {selected_routes}")
        print("You can now use this YAML as filter for XML processing.")
    elif cmd == "r":
        limit_input = input("Enter total road limit (integer): ").strip()
        try:
            total_limit = int(limit_input)
        except ValueError:
            print("Invalid input. Exiting.")
            return
        subset_config, selected_routes = greedy_subset_config(roads, scenario_count, total_limit, "road")
        output_yaml = "subset_config.yml"
        with open(output_yaml, "w", encoding="utf-8") as f:
            yaml.dump(subset_config, f, sort_keys=False)
        print(f"✅ Subset config generated: {output_yaml}")
        print(f"Total routes selected: {selected_routes}")
        print("You can now use this YAML as filter for XML processing.")
    else:
        print("Exiting without action.")

if __name__ == "__main__":
    main()
