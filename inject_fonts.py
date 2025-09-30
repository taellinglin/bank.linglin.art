#!/usr/bin/env python3
import os
import re
from lxml import etree
from tqdm import tqdm

# ------------------------------
# Utility functions
# ------------------------------
def extract_number(filename: str):
    """Extract first integer in filename for sorting."""
    match = re.search(r'(\d+)', filename)
    return int(match.group(1)) if match else float('inf')

def collect_svgs(root_dirs):
    """Recursively collect SVG files from root_dirs, grouped by subdirectory name."""
    groups = {}
    for root_dir in root_dirs:
        for current_dir, _, files in os.walk(root_dir):
            svgs = sorted(
                [os.path.join(current_dir, f) for f in files if f.lower().endswith(".svg")],
                key=extract_number
            )
            if svgs:
                group_name = os.path.relpath(current_dir, root_dir) or os.path.basename(root_dir)
                groups[group_name] = svgs
    return groups

def process_svg(file_path, output_dir="./output"):
    """Dummy SVG processing (can be expanded)."""
    try:
        tree = etree.parse(file_path)
        root = tree.getroot()
        # Example: normalize width/height attributes
        if "width" in root.attrib:
            root.attrib["width"] = "1000"
        if "height" in root.attrib:
            root.attrib["height"] = "1000"

        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, os.path.basename(file_path))
        tree.write(out_path, pretty_print=True, xml_declaration=True, encoding="utf-8")
    except Exception as e:
        print(f"[!] Failed to process {file_path}: {e}")

# ------------------------------
# Main script
# ------------------------------
if __name__ == "__main__":
    root_dirs = ["./", "./images"]
    groups = collect_svgs(root_dirs)

    total_files = sum(len(files) for files in groups.values())
    print(f"Found {total_files} SVG files across {len(groups)} groups.")

    with tqdm(total=total_files, desc="Processing SVGs", unit="file") as pbar:
        for group, files in groups.items():
            for f in files:
                process_svg(f)
                pbar.update(1)
