#!/usr/bin/env python3
"""
make_grid.py — nav2 occupancy grids (PNG + YAML per storey) from the
<name>_full.json written by clean_ifc.py.

    python3 make_grid.py --full-json data/cleaned_ifc/IDAC_Model_full.json --resolution 0.05

Per obstacle layer (storey, or the whole model when it has no storeys):
    free     (254) = space footprints + door footprints (buffered 5 cm)
    occupied (0)   = floor-level obstacles; overrides free
    unknown  (128) = everything else
nav2's map_server (trinary, negate 0) reads a pixel p as occupancy
(255 - p) / 255: 254 -> 0.004 <= free_thresh 0.25 -> free; 0 -> 1.0 >=
occupied_thresh 0.65 -> occupied; 128 -> 0.50 -> unknown. (205, the usual
"grey", gives 0.196 and would load as FREE.)

Coordinates are the project engineering frame of the JSON (metres; map
conversion not applied). Only shapely, numpy and the standard library.
"""

import argparse
import json
import math
import re
import struct
import sys
import zlib
from pathlib import Path

import numpy as np
import shapely

FREE, OCCUPIED, UNKNOWN = 254, 0, 128
DOOR_BUFFER_M = 0.05
MARGIN_M = 0.5
ROW_BLOCK = 256           # rasterise in row blocks to bound memory


def to_shape(footprint):
    """[polygon][ring][point] -> shapely MultiPolygon (or None)."""
    if not footprint:
        return None
    geom = shapely.MultiPolygon([shapely.Polygon(poly[0], poly[1:]) for poly in footprint])
    return geom if geom.is_valid else shapely.make_valid(geom)   # JSON from older extractor versions


def write_png_gray(path, img):
    """8-bit grayscale PNG with the standard library only (no Pillow)."""
    h, w = img.shape
    raw = b"".join(b"\x00" + img[row].tobytes() for row in range(h))

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


def plan_layer(full, layer, resolution):
    """Shapes and raster size for one obstacle layer."""
    sid = layer["storey"]            # None on storey-less models: matches every record
    spaces = [to_shape(s["geometry"]["footprint"]) for s in full["spaces"]
              if s["storey"] == sid and s.get("geometry") and s["geometry"].get("footprint")]
    doors = [to_shape(e["geometry"]["footprint"]) for e in full["elements"]
             if "door" in e and e["storey"] == sid and e.get("geometry") and e["geometry"].get("footprint")]
    blocked = [(it, to_shape(it["footprint"])) for it in layer["items"] if it["height_band"] == "floor-level"]
    free = shapely.union_all([s for s in spaces if s] + [d.buffer(DOOR_BUFFER_M) for d in doors if d])
    occupied = shapely.union_all([g for _, g in blocked if g]) if blocked else shapely.Polygon()
    extent = shapely.union_all([free, occupied])
    if extent.is_empty:
        return None
    minx, miny, maxx, maxy = extent.buffer(MARGIN_M).bounds
    w = int(math.ceil((maxx - minx) / resolution))
    h = int(math.ceil((maxy - miny) / resolution))
    return {"free": free, "occupied": occupied, "spaces_extent": shapely.union_all(spaces) if spaces else None,
            "blocked": blocked, "bounds": (minx, miny, maxx, maxy), "w": w, "h": h}


def far_items(plan, limit=5, distance=50.0):
    """Obstacles lying far outside the rooms: the usual cause of a huge grid
    (e.g. a Revit survey-point proxy kilometres away)."""
    if plan["spaces_extent"] is None:
        return []
    zone = plan["spaces_extent"].envelope.buffer(distance)
    out = [it for it, g in plan["blocked"] if g is not None and not zone.contains(g)]
    return out[:limit]


def rasterise(plan, resolution):
    minx, _, _, maxy = plan["bounds"]
    w, h = plan["w"], plan["h"]
    img = np.full((h, w), UNKNOWN, dtype=np.uint8)
    shapely.prepare(plan["free"])
    shapely.prepare(plan["occupied"])
    xs = minx + (np.arange(w) + 0.5) * resolution
    for r0 in range(0, h, ROW_BLOCK):
        rows = np.arange(r0, min(r0 + ROW_BLOCK, h))
        ys = maxy - (rows + 0.5) * resolution         # row 0 = top of image
        gx, gy = np.meshgrid(xs, ys)
        block = img[r0:r0 + len(rows)]
        block[shapely.contains_xy(plan["free"], gx, gy)] = FREE
        block[shapely.contains_xy(plan["occupied"], gx, gy)] = OCCUPIED
    return img


def main():
    parser = argparse.ArgumentParser(description="Write nav2 occupancy grids from a clean_ifc.py _full.json.")
    parser.add_argument("--full-json", required=True, help="<name>_full.json from clean_ifc.py")
    parser.add_argument("--resolution", type=float, default=0.05, help="Metres per cell (default 0.05)")
    parser.add_argument("--output-dir", help="Default: the directory of --full-json")
    parser.add_argument("--max-cells", type=int, default=50_000_000,
                        help="Refuse grids larger than this many cells per storey (default 50M)")
    args = parser.parse_args()
    if args.resolution <= 0:
        parser.error("--resolution must be positive")

    src = Path(args.full_json)
    full = json.loads(src.read_text())
    out_dir = Path(args.output_dir) if args.output_dir else src.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = src.name[:-len("_full.json")] if src.name.endswith("_full.json") else src.stem

    plans, errors = [], []
    for idx, layer in enumerate(full["obstacles"]):
        plan = plan_layer(full, layer, args.resolution)
        if plan is None:
            print(f"Skipped {layer['name']}: no spaces, doors or floor-level obstacles")
            continue
        cells = plan["w"] * plan["h"]
        if cells > args.max_cells:
            minx, miny, maxx, maxy = plan["bounds"]
            msg = (f"{layer['name']}: extent {maxx - minx:,.1f} x {maxy - miny:,.1f} m -> "
                   f"{plan['w']:,} x {plan['h']:,} = {cells:,} cells > --max-cells {args.max_cells:,}.")
            far = far_items(plan)
            if far:
                msg += "\n    Obstacles far outside the rooms (likely cause):" + "".join(
                    f"\n      {it['guid']} {it['ifc_class']} {it['name']!r}" for it in far)
            else:
                msg += "\n    Check for outlying geometry or a wrong unit; or raise --resolution / --max-cells."
            errors.append(msg)
        else:
            for it in far_items(plan):
                print(f"WARNING {layer['name']}: obstacle far outside the rooms widens the grid: "
                      f"{it['guid']} {it['ifc_class']} {it['name']!r}")
        plans.append((idx, layer, plan))
    if errors:
        print("ERROR: grid too large, nothing written.\n  " + "\n  ".join(errors), file=sys.stderr)
        sys.exit(2)

    for idx, layer, plan in plans:
        img = rasterise(plan, args.resolution)
        slug = re.sub(r"[^A-Za-z0-9]+", "_", layer["name"] or f"storey{idx}").strip("_")
        png = out_dir / f"{stem}_grid_{idx:02d}_{slug}.png"
        write_png_gray(png, img)
        minx, miny = plan["bounds"][:2]
        png.with_suffix(".yaml").write_text(
            f"image: {png.name}\nmode: trinary\nresolution: {args.resolution}\n"
            f"origin: [{minx:.4f}, {miny:.4f}, 0.0]\nnegate: 0\n"
            f"occupied_thresh: 0.65\nfree_thresh: 0.25\n"
            f"# storey: {layer['storey']} ({layer['name']}), floor z {layer['floor_z_m']} m, "
            f"project engineering frame\n")
        counts = {name: int((img == v).sum()) for name, v in (("free", FREE), ("occupied", OCCUPIED), ("unknown", UNKNOWN))}
        print(f"{png} ({plan['w']} x {plan['h']}) {counts}")


if __name__ == "__main__":
    main()
