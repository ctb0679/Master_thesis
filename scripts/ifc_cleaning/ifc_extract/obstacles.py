"""Obstacle layers: per storey, the footprint and z-range of every physical
element except doors, openings, coverings, slabs and spaces, plus slab
openings as fall hazards. Every element that cannot be placed on a layer is
counted in audit.obstacles_skipped by reason and class."""

from .common import is_any, rnd
from .geometry import footprint_json, voids_slab

# Physical elements that are not obstacles. Slabs are the floor itself; slab
# openings are added back separately.
OBSTACLE_EXCLUDE = (
    "IfcDoor", "IfcFeatureElement", "IfcCovering", "IfcSlab", "IfcVirtualElement",
)

OVERHEAD_CLEARANCE_M = 2.0     # obstacle bottom this far above floor -> "overhead"
BELOW_FLOOR_TOL_M = 0.02       # top at or below floor + this -> not an obstacle


def _kind(e, rels):
    if e.is_a("IfcElement") and not is_any(e, OBSTACLE_EXCLUDE):
        return "element"
    if e.is_a("IfcOpeningElement") and getattr(e, "PredefinedType", None) != "RECESS" and voids_slab(e, rels):
        return "slab_opening"
    return None


def build_obstacles(storeys, elements, rels, geom, floor_z, space_index, audit):
    """Returns [{storey, name, floor_z_m, items: [...]}]. A model without
    IfcBuildingStorey (e.g. IDAC_Model: spaces directly under IfcSite) gets one
    whole-model layer whose floor is the lowest space floor."""
    if storeys:
        layers = {s.id(): (s.GlobalId, s.Name, floor_z[s.id()]) for s in storeys}
    else:
        audit.no_storeys = True
        floors = [zmin for _, _, zmin, _ in space_index if zmin is not None]
        layers = {None: (None, "whole model (no IfcBuildingStorey)", min(floors) if floors else 0.0)}
    items = {key: [] for key in layers}

    def layer_key(e, zmin):
        if not storeys:
            return None, None
        s = rels.storey_of(e)
        if s is not None:
            return (s.id(), None) if s.id() in layers else (None, "unmatched_storey")
        below = [st for st in storeys if floor_z[st.id()] <= zmin + 0.05]
        if not below:
            return None, "below_lowest_storey"
        return max(below, key=lambda st: floor_z[st.id()]).id(), None

    for e in elements:
        kind = _kind(e, rels)
        if kind is None:
            continue
        shape = geom.shapes.get(e.id())
        if shape is None:
            continue           # already counted in audit.geometry_failures
        if shape["footprint"] is None:
            audit.obstacles_skipped[f"no_footprint | {e.is_a()}"] += 1
            continue
        zmin, zmax = float(shape["bbox_min"][2]), float(shape["bbox_max"][2])
        key, reason = layer_key(e, zmin)
        if reason:
            audit.obstacles_skipped[f"{reason} | {e.is_a()}"] += 1
            continue
        floor = layers[key][2]
        if kind == "element" and zmax <= floor + BELOW_FLOOR_TOL_M:
            audit.obstacles_skipped[f"below_floor | {e.is_a()}"] += 1
            continue
        band = "overhead" if kind == "element" and zmin - floor >= OVERHEAD_CLEARANCE_M else "floor-level"
        items[key].append({
            "guid": e.GlobalId, "ifc_class": e.is_a(), "name": e.Name, "kind": kind,
            "z_min": rnd(zmin), "z_max": rnd(zmax), "height_band": band,
            "footprint": footprint_json(shape["footprint"]),
        })
    return [{"storey": gid, "name": name, "floor_z_m": rnd(floor), "items": items[key]}
            for key, (gid, name, floor) in layers.items()]
