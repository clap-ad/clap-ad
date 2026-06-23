import argparse
import xml.etree.ElementTree as ET
import os

def split_xml(xml_path, n_parts):
    """
    Split a CARLA routes XML file into n parts (by order, modulo n).
    Each output keeps the same XML structure but only contains a subset of <route> entries.
    """
    if not os.path.exists(xml_path):
        print(f"❌ Input file not found: {xml_path}")
        return

    # Parse XML
    tree = ET.parse(xml_path)
    root = tree.getroot()
    routes = root.findall("route")

    if not routes:
        print("⚠️ No <route> elements found in XML file.")
        return

    # Prepare n empty roots
    new_roots = [ET.Element("routes") for _ in range(n_parts)]

    # Distribute routes
    for idx, route in enumerate(routes):
        new_roots[idx % n_parts].append(route)

    # Get base name and output directory
    xml_dir = os.path.dirname(xml_path)
    xml_name = os.path.splitext(os.path.basename(xml_path))[0]

    # Write split files
    for i, new_root in enumerate(new_roots):
        out_path = os.path.join(xml_dir, f"{xml_name}_{i}.xml")
        new_tree = ET.ElementTree(new_root)
        new_tree.write(out_path, encoding="utf-8", xml_declaration=True)
        print(f"✅ Saved: {out_path}  ({len(new_root.findall('route'))} routes)")

    print("===================================")
    print(f"Split complete! Total routes: {len(routes)} → {n_parts} parts.")
    print("===================================")


def main():
    parser = argparse.ArgumentParser(description="CARLA Route XML Splitter")
    parser.add_argument("--xml", "-f", required=True, help="Path to CARLA route XML file")
    parser.add_argument("--number", "-n", required=True, type=int, help="Number of parts to split into")
    args = parser.parse_args()

    if args.number <= 0:
        print("❌ Number of parts must be greater than 0.")
        return

    split_xml(args.xml, args.number)


if __name__ == "__main__":
    main()
