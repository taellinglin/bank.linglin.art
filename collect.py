#!/usr/bin/env python3
import os
import gzip
from lxml import etree
from fontTools.ttLib import TTFont
from fontTools.pens.svgPathPen import SVGPathPen
from tqdm import tqdm

SVG_NS = "http://www.w3.org/2000/svg"

# --------------------------
# Font Handling
# --------------------------
def load_fonts(font_dir="./fonts"):
    fonts = []
    for f in os.listdir(font_dir):
        if f.lower().endswith((".ttf", ".otf")):
            fonts.append(os.path.join(font_dir, f))
    if not fonts:
        raise FileNotFoundError(f"No font files found in {font_dir}")
    return fonts

def text_to_path(char, fontfile, font_size=12):
    font = TTFont(fontfile)
    glyph_set = font.getGlyphSet()
    cmap = font.getBestCmap()
    glyph_name = cmap.get(ord(char))
    if not glyph_name:
        return ""
    pen = SVGPathPen(glyph_set)
    glyph_set[glyph_name].draw(pen)
    path_d = pen.getCommands()
    scale = font_size / font['head'].unitsPerEm
    # Round coordinates to 2 decimals for compression
    path_d = ",".join([f"{float(v):.2f}" if v.replace(".","",1).isdigit() else v for v in path_d.replace("M","M ").replace("L","L ").replace("C","C ").replace("Z","Z ").split()])
    return f'<path d="{path_d}" transform="scale({scale},-{scale})"/>'

# --------------------------
# SVG Conversion
# --------------------------
def convert_text_elements(svg_path, fontfiles):
    tree = etree.parse(svg_path)
    root = tree.getroot()
    text_elements = root.findall(f".//{{{SVG_NS}}}text")
    for text_elem in tqdm(text_elements, desc=f"Converting text to paths in {os.path.basename(svg_path)}"):
        x = float(text_elem.get("x", "0"))
        y = float(text_elem.get("y", "0"))
        font_size = float(text_elem.get("font-size", "12"))
        fill = text_elem.get("fill", "#000")
        stroke = text_elem.get("stroke")
        stroke_width = text_elem.get("stroke-width")
        opacity = text_elem.get("opacity", "1")
        content = text_elem.text or ""
        # Pick a font (first one)
        fontfile = fontfiles[0]
        paths = [text_to_path(c, fontfile, font_size) for c in content]
        group = etree.Element(f"{{{SVG_NS}}}g", transform=f"translate({x},{y})")
        for p in paths:
            if p:
                path_elem = etree.fromstring(p)
                path_elem.set("fill", fill)
                if stroke: path_elem.set("stroke", stroke)
                if stroke_width: path_elem.set("stroke-width", stroke_width)
                path_elem.set("opacity", opacity)
                group.append(path_elem)
        parent = text_elem.getparent()
        parent.replace(text_elem, group)
    return root

# --------------------------
# Utility
# --------------------------
def get_svg_size(svg_elem):
    w = float(svg_elem.get("width", "100").replace("px", ""))
    h = float(svg_elem.get("height", "100").replace("px", ""))
    return w, h

def collect_svgs(root="./images"):
    data = {}
    for name in tqdm(os.listdir(root), desc="Collecting SVGs"):
        name_path = os.path.join(root, name)
        if not os.path.isdir(name_path):
            continue
        covers, backs = [], []
        # Backs: ./images/<name>/*.svg
        for f in os.listdir(name_path):
            if f.lower().endswith(".svg"):
                backs.append(os.path.join(name_path, f))
        # Covers: ./images/<name>/<denomination>/*.svg
        for denom in os.listdir(name_path):
            denom_path = os.path.join(name_path, denom)
            if os.path.isdir(denom_path):
                for f in os.listdir(denom_path):
                    if f.lower().endswith(".svg"):
                        covers.append(os.path.join(denom_path, f))
        if covers or backs:
            data[name] = {"covers": covers, "backs": backs}
    return data

# --------------------------
# Collage and Compression
# --------------------------
def combine_svgs_two_columns(name, cover_svgs, back_svgs, fontfiles, output_dir="./output"):
    parsed_covers, parsed_backs = [], []

    # Convert fronts
    for f in tqdm(cover_svgs, desc=f"Converting front covers for {name}"):
        parsed_covers.append(convert_text_elements(f, fontfiles))
    # Convert backs
    for f in tqdm(back_svgs, desc=f"Converting backs for {name}"):
        parsed_backs.append(convert_text_elements(f, fontfiles))

    max_cover_width = max((get_svg_size(svg)[0] for svg in parsed_covers), default=100)
    max_back_width = max((get_svg_size(svg)[0] for svg in parsed_backs), default=100)
    total_width = max_cover_width + max_back_width
    total_height = max(
        sum(get_svg_size(svg)[1] for svg in parsed_covers),
        sum(get_svg_size(svg)[1] for svg in parsed_backs)
    )

    root_svg = etree.Element("svg", xmlns=SVG_NS, width=str(total_width), height=str(total_height), version="1.1")

    # Place front covers (left column)
    y_offset = 0
    for svg in tqdm(parsed_covers, desc=f"Placing front covers for {name}"):
        w, h = get_svg_size(svg)
        g = etree.SubElement(root_svg, "g", transform=f"translate(0,{y_offset})")
        for child in svg:
            g.append(child)
        y_offset += h

    # Place backs (right column)
    y_offset = 0
    for svg in tqdm(parsed_backs, desc=f"Placing backs for {name}"):
        w, h = get_svg_size(svg)
        g = etree.SubElement(root_svg, "g", transform=f"translate({max_cover_width},{y_offset})")
        for child in svg:
            g.append(child)
        y_offset += h

    # Optimize: remove metadata, comments
    for elem in root_svg.xpath('//comment() | //desc | //metadata'):
        parent = elem.getparent()
        if parent is not None:
            parent.remove(elem)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{name}_combined.svgz")
    # Save as compressed SVGZ
    with gzip.open(out_path, "wb") as f:
        f.write(etree.tostring(root_svg, pretty_print=True, xml_declaration=True, encoding="UTF-8"))
    print(f"[+] Saved compressed {out_path}")

# --------------------------
# Main
# --------------------------
def main():
    fontfiles = load_fonts("./fonts")
    all_data = collect_svgs("./images")
    for name, content in tqdm(all_data.items(), desc="Processing all names"):
        combine_svgs_two_columns(name, content["covers"], content["backs"], fontfiles)

if __name__ == "__main__":
    main()
