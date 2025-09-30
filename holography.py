#!/usr/bin/env python3
"""
holography_batch.py

Creates multiple holographic security patterns in batch with embedded information.
Generates hologram_1 to hologram_100 with varied shapes and textures.

Usage:
    python holography_batch.py --datetime "2025-09-06T12:34:56" --serial "SER123" -d friendly --out holo_batch --res 1024
"""

from __future__ import annotations
import argparse
import os
import math
import random
import hashlib
import json
from dataclasses import dataclass, field
from typing import Tuple, List, Dict, Any, Optional
import numpy as np
from shapely.geometry import Point, Polygon, MultiPolygon, LineString
from shapely.ops import unary_union
from shapely.affinity import translate as shp_translate, rotate as shp_rotate, scale as shp_scale
from PIL import Image, ImageDraw, ImageFont
import shutil
from tqdm import tqdm

# -------------------------
# Utilities & deterministic hashing
# -------------------------
def deterministic_seed(*parts: str) -> int:
    h = hashlib.sha256("||".join([str(p) for p in parts]).encode("utf8")).hexdigest()
    return int(h[:16], 16)

def hsv_to_rgb(h, s, v):
    i = int(h*6)
    f = (h*6) - i
    p = v*(1-s)
    q = v*(1-f*s)
    t = v*(1-(1-f)*s)
    i = i % 6
    if i == 0: r,g,b = v,t,p
    elif i == 1: r,g,b = q,v,p
    elif i == 2: r,g,b = p,v,t
    elif i == 3: r,g,b = p,q,v
    elif i == 4: r,g,b = t,p,v
    else: r,g,b = v,p,q
    return (r,g,b)

def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))

# -------------------------
# Primitive & Layer dataclasses
# -------------------------
@dataclass
class Primitive:
    name: str
    layer: int
    geom: Polygon
    color: Tuple[float,float,float]  # rgb 0..1
    alpha: float                    # base alpha (0..1)
    dim: int = 3
    extra_coords: Tuple[float,...] = field(default_factory=tuple)
    moves: List[str] = field(default_factory=list)
    alive: bool = True

    mesh_vertices: List[Tuple[float,float,float]] = field(default_factory=list)
    mesh_faces: List[Tuple[int,int,int]] = field(default_factory=list)

@dataclass
class Layer:
    index: int
    primitives: List[Primitive] = field(default_factory=list)
    visible: bool = True

# -------------------------
# Primitive factories
# -------------------------
def make_circle(center, r, segments=64):
    return Point(center).buffer(r, resolution=segments)

def make_regular_polygon(center, radius, sides, rotation=0.0):
    angles = [rotation + 2*math.pi*i/sides for i in range(sides)]
    pts = [(center[0] + radius*math.cos(a), center[1] + radius*math.sin(a)) for a in angles]
    return Polygon(pts)

def make_star(center, outer_r, inner_r, points=5, rotation=0.0):
    pts = []
    for i in range(points*2):
        r = outer_r if i%2==0 else inner_r
        a = rotation + i * math.pi / points
        pts.append((center[0] + r*math.cos(a), center[1] + r*math.sin(a)))
    return Polygon(pts)

def make_ring(center, r_outer, r_inner, res_outer=128, res_inner=64):
    outer = Point(center).buffer(r_outer, resolution=res_outer)
    inner = Point(center).buffer(r_inner, resolution=res_inner)
    return outer.difference(inner)

def make_cross(center, width, height, thickness):
    half_w = width / 2
    half_h = height / 2
    half_t = thickness / 2
    
    # Vertical rectangle
    vert = Polygon([
        (center[0] - half_t, center[1] - half_h),
        (center[0] + half_t, center[1] - half_h),
        (center[0] + half_t, center[1] + half_h),
        (center[0] - half_t, center[1] + half_h)
    ])
    
    # Horizontal rectangle
    horz = Polygon([
        (center[0] - half_w, center[1] - half_t),
        (center[0] + half_w, center[1] - half_t),
        (center[0] + half_w, center[1] + half_t),
        (center[0] - half_w, center[1] + half_t)
    ])
    
    return vert.union(horz)

def make_gear(center, outer_radius, inner_radius, teeth, tooth_depth, rotation=0.0):
    points = []
    angle_step = 2 * math.pi / (teeth * 2)
    
    for i in range(teeth * 2):
        angle = rotation + i * angle_step
        radius = outer_radius if i % 2 == 0 else outer_radius - tooth_depth
        points.append((center[0] + radius * math.cos(angle), 
                       center[1] + radius * math.sin(angle)))
    
    gear_outer = Polygon(points)
    inner_circle = Point(center).buffer(inner_radius, resolution=64)
    
    return gear_outer.difference(inner_circle)

def make_text_shape(text, font_path, center, size, rotation=0.0):
    try:
        font = ImageFont.truetype(font_path, int(size * 100))
    except:
        # Fallback to default font
        font = ImageFont.load_default()
    
    # Create a temporary image to measure text
    temp_img = Image.new('L', (1, 1), 0)
    draw = ImageDraw.Draw(temp_img)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # Create a proper image for the text
    img = Image.new('L', (text_width + 10, text_height + 10), 0)
    draw = ImageDraw.Draw(img)
    draw.text((5, 5), text, font=font, fill=255)
    
    # Convert to polygon
    pixels = img.load()
    polygons = []
    for y in range(img.height):
        for x in range(img.width):
            if pixels[x, y] > 128:
                polygons.append(Polygon([
                    (x/size, y/size),
                    ((x+1)/size, y/size),
                    ((x+1)/size, (y+1)/size),
                    (x/size, (y+1)/size)
                ]))
    
    if not polygons:
        return Polygon()
    
    text_shape = unary_union(polygons)
    text_shape = shp_scale(text_shape, size/100, size/100, origin=(0, 0))
    text_shape = shp_translate(text_shape, center[0] - text_width/(2*size), center[1] - text_height/(2*size))
    text_shape = shp_rotate(text_shape, rotation, origin=center)
    
    return text_shape

# -------------------------
# HoloStructure
# -------------------------
class HoloStructure:
    def __init__(self, seed_parts: List[str], canvas_size=(1.0,1.0), layers=5, info_text=""):
        seed_int = deterministic_seed(*seed_parts)
        random.seed(seed_int)
        np.random.seed(seed_int & 0xffffffff)
        self.canvas_size = canvas_size
        self.layers_count = layers
        self.layers: List[Layer] = [Layer(i) for i in range(layers)]
        self.primitives_by_name: Dict[str, Primitive] = {}
        self.info_text = info_text
        self._build_initial_structure()

    def _norm(self, x, y):
        return (x * self.canvas_size[0], y * self.canvas_size[1])

    def _random_color(self):
        h = random.random()
        s = 0.45 + random.random()*0.35
        v = 0.8 + random.random()*0.2
        return hsv_to_rgb(h,s,v)

    def _choose_prim(self, iteration):
        # Vary shape types based on iteration for more diversity
        r = (random.random() + iteration/100) % 1.0
        if r < 0.20: return "circle"
        if r < 0.40: return "polygon"
        if r < 0.60: return "star"
        if r < 0.75: return "ring"
        if r < 0.85: return "cross"
        if r < 0.95: return "gear"
        return "text"

    def _build_initial_structure(self):
        w,h = self.canvas_size
        base_density = 6 + int(random.random()*8)
        
        for li in range(self.layers_count):
            for i in range(base_density):
                cx = random.random()*0.8 + 0.1
                cy = random.random()*0.8 + 0.1
                r = 0.03 + random.random()*0.08
                typ = self._choose_prim(i)
                name = f"L{li}_{typ}_{i}"
                color = self._random_color()
                alpha = 0.35 + random.random()*0.45
                
                if typ == "circle":
                    geom = make_circle(self._norm(cx,cy), r)
                elif typ == "polygon":
                    sides = random.randint(3,8)
                    geom = make_regular_polygon(self._norm(cx,cy), r*(0.8+random.random()*0.6), 
                                               sides, rotation=random.random()*2*math.pi)
                elif typ == "star":
                    geom = make_star(self._norm(cx,cy), r*1.1, r*0.45, 
                                    points=random.randint(5,7), rotation=random.random()*2*math.pi)
                elif typ == "ring":
                    geom = make_ring(self._norm(cx,cy), r*1.2, r*0.6)
                elif typ == "cross":
                    geom = make_cross(self._norm(cx,cy), r*2, r*2, r*0.5)
                elif typ == "gear":
                    geom = make_gear(self._norm(cx,cy), r, r*0.4, 
                                    random.randint(6,12), r*0.3, rotation=random.random()*2*math.pi)
                else:  # text
                    # Use part of info text for shape
                    text_len = min(3, len(self.info_text))
                    text_part = self.info_text[i % text_len:(i % text_len) + 1] or "X"
                    geom = make_text_shape(text_part, "./fonts/DejaVuSans.ttf", 
                                          self._norm(cx,cy), r*0.8, rotation=random.random()*2*math.pi)
                
                prim = Primitive(
                    name=name,
                    layer=li,
                    geom=geom,
                    color=color,
                    alpha=alpha,
                    dim=3,
                    extra_coords=((li+1)/self.layers_count , random.random())
                )
                self.layers[li].primitives.append(prim)
                self.primitives_by_name[name] = prim
                
        self._apply_boolean_patterns()

    def _apply_boolean_patterns(self):
        for li in range(self.layers_count-1):
            a = self.layers[li]
            b = self.layers[li+1]
            if not a.primitives or not b.primitives: continue
            
            for _ in range(2):
                pa = random.choice(a.primitives)
                pb = random.choice(b.primitives)
                op = random.choice(["difference","intersection","union","xor"])
                
                try:
                    if op == "difference":
                        g = pa.geom.difference(pb.geom)
                    elif op == "intersection":
                        g = pa.geom.intersection(pb.geom)
                    elif op == "union":
                        g = pa.geom.union(pb.geom)
                    else:
                        g = pa.geom.symmetric_difference(pb.geom)
                    
                    if g.is_empty: continue
                    
                    name = f"bool_{pa.name}_{pb.name}_{op}"
                    color = self._random_color()
                    alpha = (pa.alpha + pb.alpha)/2.0
                    derived = Primitive(name=name, layer=li+1, geom=g, color=color, alpha=alpha)
                    b.primitives.append(derived)
                    self.primitives_by_name[name] = derived
                except Exception:
                    continue

# -------------------------
# Camera
# -------------------------
@dataclass
class Camera:
    eye: Tuple[float,float,float]
    target: Tuple[float,float,float]
    up: Tuple[float,float,float]
    fov_deg: float = 45.0
    near: float = 0.01
    far: float = 10.0

    def view_matrix(self):
        ex,ey,ez = self.eye
        tx,ty,tz = self.target
        ux,uy,uz = self.up
        fx,fy,fz = (tx-ex, ty-ey, tz-ez)
        flen = math.sqrt(fx*fx+fy*fy+fz*fz)
        fx,fy,fz = fx/flen, fy/flen, fz/flen
        rx = uy*fz - uz*fy
        ry = uz*fx - ux*fz
        rz = ux*fy - uy*fx
        rlen = math.sqrt(rx*rx+ry*ry+rz*rz)
        rx,ry,rz = rx/rlen, ry/rlen, rz/rlen
        ux2 = fy*rz - fz*ry
        uy2 = fz*rx - fx*rz
        uz2 = fx*ry - fy*rx
        M = np.array([
            [rx, ry, rz, -(rx*ex + ry*ey + rz*ez)],
            [ux2, uy2, uz2, -(ux2*ex + uy2*ey + uz2*ez)],
            [-fx, -fy, -fz, (fx*ex + fy*ey + fz*ez)],
            [0,0,0,1.0]
        ], dtype=float)
        return M

# -------------------------
# Mesh extrusion
# -------------------------
def extrude_polygon_to_mesh(poly, height: float=0.01, z_base: float=0.0):
    """
    Extrude Polygon or MultiPolygon to mesh (verts, faces).
    Handles MultiPolygons by concatenating all sub-polygons.
    """
    verts, faces = [], []

    if poly.is_empty:
        return verts, faces

    # Helper to extrude a single Polygon
    def _extrude_single(p: Polygon, vert_offset: int):
        local_verts, local_faces = [], []
        exterior = list(p.exterior.coords)
        if len(exterior) > 1 and exterior[0] == exterior[-1]:
            exterior = exterior[:-1]
        if len(exterior) < 3:
            return [], []

        top_index_start = vert_offset
        for x, y in exterior:
            local_verts.append((x, y, z_base + height))
        bottom_index_start = vert_offset + len(exterior)
        for x, y in exterior:
            local_verts.append((x, y, z_base))

        n = len(exterior)
        # Top face
        for i in range(1, n-1):
            local_faces.append((top_index_start + 0, top_index_start + i, top_index_start + i + 1))
        # Bottom face
        for i in range(1, n-1):
            local_faces.append((bottom_index_start + 0, bottom_index_start + i + 1, bottom_index_start + i))
        # Side faces
        for i in range(n):
            i_next = (i + 1) % n
            t0 = top_index_start + i
            t1 = top_index_start + i_next
            b0 = bottom_index_start + i
            b1 = bottom_index_start + i_next
            local_faces.append((t0, b0, b1))
            local_faces.append((t0, b1, t1))

        return local_verts, local_faces

    if isinstance(poly, Polygon):
        v, f = _extrude_single(poly, vert_offset=len(verts))
        verts.extend(v)
        faces.extend(f)
    elif isinstance(poly, MultiPolygon):
        for p in poly.geoms:
            v, f = _extrude_single(p, vert_offset=len(verts))
            verts.extend(v)
            faces.extend(f)

    return verts, faces

# -------------------------
# Geometry helper
# -------------------------
def _extract_polygons(geom):
    if geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, MultiPolygon):
        return list(geom.geoms)
    polys = []
    try:
        for g in geom.geoms:
            polys.extend(_extract_polygons(g))
    except Exception:
        pass
    return polys

# -------------------------
# Rasterization & blending
# -------------------------
def rasterize_and_blend(struct, cam, out_res=1024):
    W = out_res
    H = out_res
    img = np.zeros((H, W, 3), dtype=np.float32)
    prim_entries = []
    V = cam.view_matrix()

    # Collect primitives & depth
    for layer in struct.layers:
        for prim in layer.primitives:
            if not prim.alive:
                continue
            polys = _extract_polygons(prim.geom)
            for poly in polys:
                if poly.is_empty:
                    continue
                zx = prim.extra_coords[0] if prim.extra_coords else (layer.index+1)/struct.layers_count
                z_world = -0.2 - zx * 0.6
                centroid_world = np.array([
                    (poly.centroid.x - struct.canvas_size[0]/2.0),
                    (poly.centroid.y - struct.canvas_size[1]/2.0),
                    z_world, 1.0
                ])
                centroid_view = V.dot(centroid_world)
                depth = centroid_view[2]
                prim_entries.append({
                    "prim": prim,
                    "poly2": poly,
                    "depth": float(depth)
                })

    pixel_prims = []
    for ent in prim_entries:
        poly, prim = ent["poly2"], ent["prim"]
        minx, miny, maxx, maxy = poly.bounds
        minx = clamp(minx, 0.0, struct.canvas_size[0])
        maxx = clamp(maxx, 0.0, struct.canvas_size[0])
        miny = clamp(miny, 0.0, struct.canvas_size[1])
        maxy = clamp(maxy, 0.0, struct.canvas_size[1])
        px0 = int(minx / struct.canvas_size[0] * (W-1))
        px1 = int(maxx / struct.canvas_size[0] * (W-1))
        py0 = int((1.0 - maxy / struct.canvas_size[1]) * (H-1))
        py1 = int((1.0 - miny / struct.canvas_size[1]) * (H-1))
        px0 = clamp(px0, 0, W-1)
        px1 = clamp(px1, 0, W-1)
        py0 = clamp(py0, 0, H-1)
        py1 = clamp(py1, 0, H-1)
        ent["px0"], ent["px1"], ent["py0"], ent["py1"] = px0, px1, py0, py1
        pixel_prims.append(ent)

    # Rasterize per row with tqdm
    pixel_layers = [[[] for _ in range(W)] for _ in range(H)]
    for py in tqdm(range(H), desc="Rasterizing rows"):
        y = 1.0 - (py / (H-1)) * struct.canvas_size[1]
        for px in range(W):
            x = (px / (W-1)) * struct.canvas_size[0]
            for ent in pixel_prims:
                poly = ent["poly2"]
                prim = ent["prim"]
                px0, px1, py0, py1 = ent["px0"], ent["px1"], ent["py0"], ent["py1"]
                if not (px0 <= px <= px1 and py0 <= py <= py1):
                    continue
                if poly.bounds[0] <= x <= poly.bounds[2] and poly.bounds[1] <= y <= poly.bounds[3]:
                    pt = Point(x, y)
                    if poly.contains(pt) or poly.touches(pt):
                        pixel_layers[py][px].append((ent["depth"], prim))

    prim_total_contribution = {p["prim"].name: 0.0 for p in pixel_prims}
    prim_pixel_counts = {p["prim"].name: 0 for p in pixel_prims}

    # Combine colors per pixel with tqdm
    for py in tqdm(range(H), desc="Blending pixels"):
        for px in range(W):
            hits = pixel_layers[py][px]
            if not hits:
                continue
            hits_sorted = sorted(hits, key=lambda x: x[0])
            base_alphas = [h[1].alpha for h in hits_sorted]
            s = sum(base_alphas)
            if s <= 0:
                continue
            scale = 1.0 / s
            pix_rgb = np.zeros(3, dtype=float)
            for (d, prim) in hits_sorted:
                contrib_alpha = prim.alpha * scale
                c = np.array(prim.color, dtype=float)
                pix_rgb += c * contrib_alpha
                prim_total_contribution[prim.name] += contrib_alpha
                prim_pixel_counts[prim.name] += 1
            img[py, px, :] = np.clip(pix_rgb, 0.0, 1.0)

    prim_avg_contrib = {}
    for name, total in prim_total_contribution.items():
        cnt = prim_pixel_counts.get(name, 0)
        prim_avg_contrib[name] = (total / cnt) if cnt > 0 else 0.0

    return img, prim_avg_contrib

# -------------------------
# OBJ export
# -------------------------
def export_obj(struct: HoloStructure, prim_avg_contrib: Dict[str,float], out_dir: str, obj_name="hologram"):
    os.makedirs(out_dir, exist_ok=True)
    obj_path = os.path.join(out_dir, f"{obj_name}.obj")
    mtl_path = os.path.join(out_dir, f"{obj_name}.mtl")
    obj_lines, mtl_lines = [], []
    obj_lines.append(f"mtllib {os.path.basename(mtl_path)}\n")
    vertex_offset = 1
    
    for li, layer in enumerate(struct.layers):
        for prim in layer.primitives:
            if prim.geom.is_empty or not prim.alive:
                continue
            height = 0.004 + (prim.extra_coords[0] if prim.extra_coords else 0.0) * 0.02
            z_base = li * 0.002
            verts, faces = extrude_polygon_to_mesh(prim.geom, height=height, z_base=z_base)
            if not verts or not faces:
                continue
            avg = prim_avg_contrib.get(prim.name, 0.0)
            weight = clamp(avg, 0.05, 1.0)
            mr = clamp(prim.color[0] * (0.6 + 0.8 * weight), 0.0, 1.0)
            mg = clamp(prim.color[1] * (0.6 + 0.8 * weight), 0.0, 1.0)
            mb = clamp(prim.color[2] * (0.6 + 0.8 * weight), 0.0, 1.0)
            mat_name = f"mat_{prim.name}"
            mtl_lines.append(f"newmtl {mat_name}\n")
            mtl_lines.append(f"Kd {mr:.4f} {mg:.4f} {mb:.4f}\n")  # diffuse color
            mtl_lines.append("Ka 0.0 0.0 0.0\n")                   # ambient
            mtl_lines.append("Ks 0.0 0.0 0.0\n")                   # specular
            mtl_lines.append("d 1.0\n")                            # opacity
            mtl_lines.append("illum 1\n\n")

            obj_lines.append(f"usemtl {mat_name}\n")

            # Write vertices
            for v in verts:
                obj_lines.append(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

            # Write faces (OBJ is 1-indexed)
            for f in faces:
                v1, v2, v3 = f
                obj_lines.append(f"f {v1+vertex_offset} {v2+vertex_offset} {v3+vertex_offset}\n")

            vertex_offset += len(verts)

    # Write files
    with open(obj_path, "w") as fobj:
        fobj.writelines(obj_lines)
    with open(mtl_path, "w") as fmtl:
        fmtl.writelines(mtl_lines)

    return obj_path, mtl_path

# -------------------------
# Text embedding functions
# -------------------------
def embed_text_in_image(img, text, position=(10, 10), font_size=20, color=(255, 255, 255)):
    """Embed text into the image"""
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("fonts/daemon.otf", font_size)
    except:
        font = ImageFont.load_default()
    
    draw.text(position, text, font=font, fill=color)
    return img

def shape_gradient_color(prim, cam_up, info_text, index):
    """Create gradient colors based on shape and embedded information"""
    cx, cy = prim.geom.centroid.x, prim.geom.centroid.y
    
    # Use text hash to influence color
    text_hash = hashlib.md5(info_text.encode()).hexdigest()
    hue_seed = int(text_hash[:8], 16) / 0xFFFFFFFF
    
    # Use index to vary colors across holograms
    hue = (hue_seed + index/100 + (cx*cam_up[0] + cy*cam_up[1])) % 1.0
    saturation = 0.6 + 0.3 * (int(text_hash[8:12], 16) / 0xFFFF)
    value = 0.7 + 0.2 * (int(text_hash[12:16], 16) / 0xFFFF)
    
    return hsv_to_rgb(hue, saturation, value)

# -------------------------
# Main - Batch generation
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Generate batch of holographic security patterns")
    parser.add_argument("--datetime", type=str, default=None, help="Timestamp string (ISO8601)")
    parser.add_argument("--serial", type=str, default=None, help="Serial ID string")
    parser.add_argument("-d", "--denomination", type=str, choices=["friendly","numeral","exponent"], default="friendly")
    parser.add_argument("--out", type=str, default="./holograms_batch", help="Output directory")
    parser.add_argument("--res", type=int, default=1024, help="Image resolution")
    parser.add_argument("--count", type=int, default=100, help="Number of holograms to generate")
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.out, exist_ok=True)
    
    # Create info text from arguments
    info_text = f"{args.datetime or ''} {args.serial or ''} {args.denomination or ''}".strip()
    
    # Generate batch of holograms
    for i in tqdm(range(1, args.count + 1), desc="Generating holograms"):
        # Seed parts for deterministic but varied structure
        seed_parts = [
            args.datetime or "",
            args.serial or "", 
            args.denomination,
            str(i)  # Add iteration to vary each hologram
        ]
        
        # Create structure with embedded info text
        struct = HoloStructure(seed_parts, canvas_size=(1.0,1.0), layers=5, info_text=info_text)
        
        # Camera setup - vary camera angle for each hologram
        angle = (i / args.count) * 2 * math.pi
        radius = 1.2 + 0.3 * math.sin(i * 0.1)  # Slightly vary radius
        eye = (math.cos(angle)*radius, math.sin(angle)*radius, 0.6 + 0.2 * math.cos(i * 0.05))
        cam = Camera(eye=eye, target=(0.0,0.0,0.0), up=(0,0,1))
        
        # Apply gradient colors based on embedded info
        for layer in struct.layers:
            for prim in layer.primitives:
                prim.color = shape_gradient_color(prim, cam.up, info_text, i)
        
        # Render hologram
        img_float, prim_avg = rasterize_and_blend(struct, cam, out_res=args.res)
        img_uint8 = (np.clip(img_float,0,1)*255).astype(np.uint8)
        img = Image.fromarray(img_uint8, mode="RGB")
        
        # Embed info text in image
        img = embed_text_in_image(img, info_text, position=(20, 20), font_size=24, color=(255, 255, 255))
        img = embed_text_in_image(img, f"Hologram #{i}:", position=(20, 50), font_size=18, color=(200, 200, 200))
        
        # Save image
        img_path = os.path.join(args.out, f"hologram_{i:03d}.png")
        img.save(img_path)
        
        # Export OBJ
        obj_name = f"hologram_{i:03d}"
        obj_path, mtl_path = export_obj(struct, prim_avg, args.out, obj_name=obj_name)
        
        print(f"[+] Generated hologram {i:03d} → {img_path}, {obj_path}")

    print(f"[+] Batch generation complete. Created {args.count} holograms in '{args.out}' directory.")

if __name__ == "__main__":
    main()