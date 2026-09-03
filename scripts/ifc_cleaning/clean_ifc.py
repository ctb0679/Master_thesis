#!/usr/bin/env python3
"""
clean_ifc.py — standalone IFC-to-JSON cleaner for the object-search pipeline.

Run manually, per file, when needed:
    python3 clean_ifc.py --input path/to/building.ifc --output path/to/building_cleaned.json

Design contract (do not violate when editing):
- No imports from legacy_graph_pipeline or vlm_reasoning. Zero project-internal
  dependencies. Only ifcopenshell + stdlib.
- Input: one IFC file path. Output: one JSON file. Nothing before or after
  this script is assumed to run.
- "Clean" means: keep everything needed for spatial/functional reasoning and
  navigation; drop only what's irrelevant to that (materials, structural/
  thermal/energy analysis data, Revit administrative metadata, raw geometry
  meshes). When in doubt, KEEP — this script is deliberately conservative
  about what it drops, and prints a summary so drops are auditable, not silent.

What's kept, and why:
- Spatial hierarchy (Project/Site/Building/Storey) — needed for multi-storey
  buildings later.
- Every IfcSpace: GlobalId, Name, LongName, ObjectType, floor area (exact
  field match on NetFloorArea/GrossFloorArea only — substring "area" matching
  pulls in unrelated Energy Analysis constants, per prior pipeline lesson),
  and a 2D centroid — required for the "which room is nearer" proximity/path
  task, not just room identity.
- Every piece of equipment (IfcFurniture + IfcBuildingElementProxy): GlobalId,
  Name, IFC class, PredefinedType if set, containing room, and a 2D centroid
  — needed for eventual navigation targets, not just "this room has a sink".
- Every IfcDoor: GlobalId, Name, and the room(s) it connects — this is the
  adjacency graph the path-planning task needs (room A is next to room B),
  not just "there is a door".
- Any property-set entries NOT matched by DROP_PSET_KEYWORDS below — i.e.
  custom/unknown properties are kept by default, since a real building's
  export may use classification fields this script's author didn't
  anticipate. Only known-irrelevant categories are dropped explicitly.

What's dropped, and why:
- Property sets whose name matches DROP_PSET_KEYWORDS (materials, structural,
  thermal, energy analysis, cost, phase/workset/revit-admin metadata).
- Raw geometry (meshes, breps, extrusion parameters) beyond the single
  centroid point per entity — an LLM doesn't need vertex-level geometry to
  reason about room function or adjacency.
- IFC entities unrelated to spatial/functional/navigation reasoning
  (IfcAnnotation, dimension/grid lines, structural elements, MEP ductwork,
  etc.) — anything not in KEEP_CLASSES is not walked at all.
"""

import argparse
import json
import sys
from pathlib import Path

import ifcopenshell
import ifcopenshell.geom
import ifcopenshell.util.element as ifc_element_util

KEEP_CLASSES_EQUIPMENT = ("IfcFurniture", "IfcBuildingElementProxy")

# Property SET names are matched by substring, case-insensitive. Anything
# matching gets dropped wholesale. Everything else is kept.
DROP_PSET_KEYWORDS = (
    "material", "thermal", "structural", "energy", "cost", "schedule",
    "phase", "phasing", "workset", "coordination", "reinforcement", "concrete",
    "manufacturer", "acoustic", "fire rating", "identitydata",
)

# Exact field names only for area (substring "area" matching pulls in
# unrelated Energy Analysis constants — confirmed issue in prior extraction).
AREA_FIELDS = ("NetFloorArea", "GrossFloorArea")


def get_psets_filtered(entity):
    """Return dict of {pset_name: {prop: value}} with irrelevant psets dropped."""
    try:
        all_psets = ifc_element_util.get_psets(entity)
    except Exception:
        return {}
    kept = {}
    for pset_name, props in all_psets.items():
        lname = pset_name.lower()
        if any(kw in lname for kw in DROP_PSET_KEYWORDS):
            continue
        # never keep the boilerplate 'id' field ifcopenshell injects
        props = {k: v for k, v in props.items() if k != "id"}
        if props:
            kept[pset_name] = props
    return kept


def get_area(entity):
    psets = get_psets_filtered(entity)
    for pset_name, props in psets.items():
        for field in AREA_FIELDS:
            if field in props and isinstance(props[field], (int, float)):
                return round(float(props[field]), 3)
    return None


def get_centroid(entity, geom_settings):
    """World-space 2D centroid via the actual geometry engine — not manual
    profile parsing. Earlier version only handled IfcArbitraryClosedProfileDef
    with a polyline curve, and assumed profile points were already world
    coordinates. Both were wrong: real Revit exports commonly use
    IfcRectangleProfileDef (no OuterCurve at all) for simple rooms, and every
    profile is defined in a local coordinate system that must be resolved
    through the entity's placement chain. Using ifcopenshell.geom with
    USE_WORLD_COORDS handles any profile type and any placement nesting
    correctly. Returns None only if the entity genuinely has no representation
    ifcopenshell can process (rare) — this is a real gap to check manually,
    not a parsing shortcoming.

    Note: this is the mean of all mesh vertices, not a bounding-box center.
    For strongly non-convex (e.g. L-shaped) rooms this can land slightly off
    the true visual center; it will still be inside or very near the room for
    any layout seen so far. Flag if a specific room's centroid looks wrong.
    """
    try:
        shape = ifcopenshell.geom.create_shape(geom_settings, entity)
        verts = shape.geometry.verts
        if not verts:
            return None
        xs = verts[0::3]
        ys = verts[1::3]
        return [round(sum(xs) / len(xs), 3), round(sum(ys) / len(ys), 3)]
    except Exception:
        return None


def clean_ifc(input_path: str) -> dict:
    model = ifcopenshell.open(input_path)
    geom_settings = ifcopenshell.geom.settings()
    geom_settings.set(geom_settings.USE_WORLD_COORDS, True)

    result = {
        "source_file": Path(input_path).name,
        "schema": model.schema,
        "project": None,
        "storeys": [],
        "spaces": [],
        "equipment": [],
        "doors": [],
    }

    proj = model.by_type("IfcProject")
    if proj:
        result["project"] = proj[0].Name

    for storey in model.by_type("IfcBuildingStorey"):
        result["storeys"].append({
            "guid": storey.GlobalId,
            "name": storey.Name,
            "elevation": getattr(storey, "Elevation", None),
        })

    space_guid_to_name = {}
    for space in model.by_type("IfcSpace"):
        name = space.Name
        space_guid_to_name[space.GlobalId] = name
        result["spaces"].append({
            "guid": space.GlobalId,
            "name": name,
            "long_name": space.LongName,
            "object_type": space.ObjectType,
            "area_m2": get_area(space),
            "centroid_xy": get_centroid(space, geom_settings),
            "properties": get_psets_filtered(space),
        })

    for cls in KEEP_CLASSES_EQUIPMENT:
        for ent in model.by_type(cls):
            container = None
            try:
                rels = ent.ContainedInStructure
                if rels:
                    container = rels[0].RelatingStructure.Name
            except Exception:
                pass
            result["equipment"].append({
                "guid": ent.GlobalId,
                "name": ent.Name,
                "ifc_class": cls,
                "predefined_type": getattr(ent, "PredefinedType", None),
                "room": container,
                "centroid_xy": get_centroid(ent, geom_settings),
                "properties": get_psets_filtered(ent),
            })

    # Single pass over space boundaries -> element GUID -> [room names],
    # instead of re-scanning all boundaries per door (scales to large buildings).
    element_to_rooms = {}
    for boundary in model.by_type("IfcRelSpaceBoundary"):
        elem = boundary.RelatedBuildingElement
        if elem is None:
            continue
        element_to_rooms.setdefault(elem.GlobalId, []).append(boundary.RelatingSpace.Name)

    for door in model.by_type("IfcDoor"):
        connected_rooms = element_to_rooms.get(door.GlobalId, [])
        result["doors"].append({
            "guid": door.GlobalId,
            "name": door.Name,
            "connects_rooms": connected_rooms,
            "overall_width": getattr(door, "OverallWidth", None),
            "overall_height": getattr(door, "OverallHeight", None),
        })

    return result


def main():
    parser = argparse.ArgumentParser(description="Clean an IFC file into a lightweight JSON summary.")
    parser.add_argument("--input", required=True, help="Path to source .ifc file")
    parser.add_argument("--output", required=True, help="Path to write cleaned .json")
    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"ERROR: input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    original_bytes = Path(args.input).stat().st_size
    result = clean_ifc(args.input)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    cleaned_bytes = out_path.stat().st_size

    # Audit summary — so drops are visible, not silent.
    print(f"Input:  {args.input} ({original_bytes:,} bytes)")
    print(f"Output: {args.output} ({cleaned_bytes:,} bytes, "
          f"{100 * cleaned_bytes / original_bytes:.1f}% of original)")
    print(f"Spaces:    {len(result['spaces'])}")
    print(f"Equipment: {len(result['equipment'])}")
    print(f"Doors:     {len(result['doors'])}")
    missing_area = sum(1 for s in result["spaces"] if s["area_m2"] is None)
    missing_centroid = sum(1 for s in result["spaces"] if s["centroid_xy"] is None)
    if missing_area:
        print(f"WARNING: {missing_area} space(s) have no NetFloorArea/GrossFloorArea found.")
    if missing_centroid:
        print(f"WARNING: {missing_centroid} space(s) have no extractable centroid "
              f"(non-extruded or unusual geometry — check manually).")


if __name__ == "__main__":
    main()
