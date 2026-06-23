import argparse
import xml.etree.ElementTree as ET
import os
from collections import defaultdict
import copy


def up_balance_xml(xml_path):
    """
    Up-balance CARLA routes XML:
    For each scenario type, replicate existing routes until all scenarios have the same number of routes as the max one.
    New route_id = k * 100000 + original route_id
    """
    if not os.path.exists(xml_path):
        print(f"❌ Input XML file not found: {xml_path}")
        return

    tree = ET.parse(xml_path)
    root = tree.getroot()
    routes = root.findall("route")

    if not routes:
        print("⚠️ No <route> elements found in XML file.")
        return

    # 1️⃣ Map scenario -> list of routes
    scenario_to_routes = defaultdict(list)
    for route in routes:
        scenarios = route.find("scenarios")
        if scenarios is not None:
            for sc in scenarios.findall("scenario"):
                sc_type = sc.get("type")
                if sc_type:
                    scenario_to_routes[sc_type].append(route)

    # 2️⃣ Find maximum count
    scenario_counts = {k: len(v) for k, v in scenario_to_routes.items()}
    max_count = max(scenario_counts.values())
    print("===================================")
    print("   Scenario Route Balancing")
    print("===================================")
    for sc, count in scenario_counts.items():
        print(f"{sc}: {count} routes")
    print(f"\n👉 Max scenario count: {max_count}")

    # 3️⃣ Replicate missing routes
    new_routes = []
    for sc, route_list in scenario_to_routes.items():
        current_count = len(route_list)
        if current_count == 0:
            continue
        if current_count >= max_count:
            continue

        print(f"\n🔄 Balancing scenario: {sc}")
        k = 1
        while len(route_list) + len([r for r in new_routes if any(s.get('type') == sc for s in r.findall('scenarios/scenario'))]) < max_count:
            for r in route_list:
                if len(route_list) + len([r for r in new_routes if any(s.get('type') == sc for s in r.findall('scenarios/scenario'))]) >= max_count:
                    break
                new_r = copy.deepcopy(r)
                old_id = int(r.get("id"))
                new_id = k * 100000 + old_id
                new_r.set("id", str(new_id))
                new_routes.append(new_r)
            k += 1

    # 4️⃣ Append new routes
    for r in new_routes:
        root.append(r)

    print(f"\n✅ Added {len(new_routes)} new routes. Total now: {len(root.findall('route'))}")

    # 5️⃣ Save new XML
    xml_dir = os.path.dirname(xml_path)
    xml_name = os.path.splitext(os.path.basename(xml_path))[0]
    out_path = os.path.join(xml_dir, f"{xml_name}_supl.xml")
    tree_out = ET.ElementTree(root)
    tree_out.write(out_path, encoding="utf-8", xml_declaration=True)

    print(f"✅ Balanced XML saved to: {out_path}")
    print("===================================")


def main():
    parser = argparse.ArgumentParser(description="CARLA Route XML Up-Balancer")
    parser.add_argument("--xml", "-f", required=True, help="Path to CARLA route XML file")
    args = parser.parse_args()

    up_balance_xml(args.xml)


if __name__ == "__main__":
    main()
