"""Rooms, doors and adjacency: room function and area, door details, the
room graph, and equipment-to-room assignment.

Relations are always tried first. Geometric fallbacks (door probes,
point-in-polygon) fill gaps and every use of them is recorded per record
(room_link_source / room_source) and in the audit."""

from collections import defaultdict

import numpy as np
import shapely

import ifcopenshell.util.element as ifc_element

from .common import OUTSIDE, guid_of, is_any, rnd

# Exact field names for floor area (substring "area" matching pulled in
# unrelated Energy Analysis constants in v1).
AREA_FIELDS = ("NetFloorArea", "GrossFloorArea")

# Exact pset field names accepted as a room function, in priority order.
FUNCTION_FIELDS = ("OccupancyType", "SpaceFunction", "Function", "Usage")

# Objects the robot may be sent to find. Pipe/duct/cable segments and fittings
# are distribution elements too, but they are network pieces, not targets.
EQUIPMENT_CLASSES = (
    "IfcFurnishingElement", "IfcBuildingElementProxy",
    "IfcDistributionElement", "IfcTransportElement",
)
EQUIPMENT_EXCLUDE = ("IfcFlowSegment", "IfcFlowFitting", "IfcDistributionChamberElement")

# Door probes sit half the door (or host wall) depth plus this margin either
# side of the door centre. A fixed 0.5 m landed inside walls thicker than ~1 m.
PROBE_MARGIN_M = 0.3


def is_equipment(entity):
    return is_any(entity, EQUIPMENT_CLASSES) and not is_any(entity, EQUIPMENT_EXCLUDE)


class RoomsAndDoors:
    def __init__(self, rels, geom, records, audit):
        self.rels, self.geom, self.records, self.audit = rels, geom, records, audit
        self.space_index = []      # (space entity, prepared footprint, zmin, zmax)

    # --- rooms ------------------------------------------------------------

    def enrich_spaces(self, spaces):
        for sp in spaces:
            rec = self.records[sp.id()]
            shape = self.geom.shapes.get(sp.id())
            fp = shape["footprint"] if shape else None
            zmin = float(shape["bbox_min"][2]) if shape else None
            zmax = float(shape["bbox_max"][2]) if shape else None
            if fp is not None:
                shapely.prepare(fp)
                self.space_index.append((sp, fp, zmin, zmax))
            area, source = self._area(rec)
            if area is None and fp is not None:
                area, source = fp.area, "footprint"
                self.audit.rooms_area_from_footprint.append(sp.GlobalId)
            rec.update({
                "area_m2": rnd(area, 2), "area_source": source,
                "floor_elevation_m": rnd(zmin),
                "clear_height_m": rnd(zmax - zmin) if shape else None,
                "doors": [], "virtual_openings": [],
            })
            rec["function"], rec["function_source"] = self.room_function(sp, rec)
            if rec["function_source"] in (None, "Name"):
                self.audit.rooms_without_function.append({"guid": sp.GlobalId, "name": sp.Name})
            storey = self.rels.storey_of(sp)
            if storey is not None and storey.id() in self.records:
                self.records[storey.id()]["spaces"].append(sp.GlobalId)

    @staticmethod
    def _area(rec):
        for field in AREA_FIELDS:
            for group in (rec["qtos"], rec["psets"]):
                for props in group.values():
                    v = props.get(field) if isinstance(props, dict) else None
                    if isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0:
                        return v, field
        return None, None

    @staticmethod
    def room_function(space, rec):
        """LongName > ObjectType > classification > function/usage pset field > Name."""
        if space.LongName:
            return space.LongName, "LongName"
        if space.ObjectType:
            return space.ObjectType, "ObjectType"
        for c in rec["classifications"]:
            text = c["description"] or c["name"] or c["code"]
            if text:
                return text, "classification"
        for field in FUNCTION_FIELDS:
            for pset, props in rec["psets"].items():
                v = props.get(field) if isinstance(props, dict) else None
                if isinstance(v, str) and v.strip():
                    return v, f"pset:{pset}.{field}"
        if space.Name:
            return space.Name, "Name"
        return None, None

    def locate_room(self, x, y, z, tol=0.3):
        """Smallest space whose footprint contains (x, y) and whose height
        range contains z. Used by both geometric fallbacks."""
        best = None
        for sp, fp, zmin, zmax in self.space_index:
            if z is not None and zmin is not None and not (zmin - tol <= z <= zmax + tol):
                continue
            if shapely.contains_xy(fp, x, y) and (best is None or fp.area < best[1].area):
                best = (sp, fp)
        return best[0] if best else None

    # --- doors ------------------------------------------------------------

    def door_probe(self, door, host_thickness):
        """Rooms on the -normal and +normal side of the door (local Y axis,
        the IFC opening direction), or None if the door has no geometry."""
        shape = self.geom.shapes.get(door.id())
        m = self.geom.placement(door)
        if shape is None or m is None:
            return None
        n = np.array(m[:2, 1])
        if np.linalg.norm(n) < 1e-9:
            return None
        n /= np.linalg.norm(n)
        fp = shape["footprint"]
        lo, hi = shape["bbox_min"], shape["bbox_max"]
        if fp is not None:
            pts, c = shapely.get_coordinates(fp), np.array([fp.centroid.x, fp.centroid.y])
        else:
            pts = np.array([[lo[0], lo[1]], [lo[0], hi[1]], [hi[0], lo[1]], [hi[0], hi[1]]])
            c = (lo[:2] + hi[:2]) / 2
        proj = pts @ n
        depth = max(float(proj.max() - proj.min()), host_thickness or 0.0)
        offset = depth / 2 + PROBE_MARGIN_M
        z = float(lo[2]) + 0.3
        return {"plus": self.locate_room(*(c + offset * n), z),
                "minus": self.locate_room(*(c - offset * n), z),
                "offset_m": rnd(offset), "normal": [rnd(n[0], 4), rnd(n[1], 4)],
                "mirrored": bool(np.linalg.det(m[:3, :3]) < 0)}

    def door_details(self, door, rec):
        type_obj = ifc_element.get_type(door)
        common = rec["psets"].get("Pset_DoorCommon", {})
        attrs = rec["attributes"]
        overall_w = attrs.get("OverallWidth") or (type_obj and getattr(type_obj, "OverallWidth", None))
        # operation type: IFC4 occurrence attribute, else IfcDoorType / IfcDoorStyle (IFC2X3)
        op, op_source = attrs.get("OperationType"), "occurrence"
        if op in (None, "NOTDEFINED") and type_obj is not None:
            op, op_source = getattr(type_obj, "OperationType", None), type_obj.is_a()
        if op is None:
            op_source = None
        # clear width: explicit property > overall - 2 * lining > overall width
        clear_w, clear_src = None, None
        for group in (rec["psets"], rec["qtos"]):
            for pset, props in group.items():
                for key in ("ClearWidth", "ClearOpeningWidth"):
                    if clear_w is None and isinstance(props.get(key), (int, float)):
                        clear_w, clear_src = props[key], f"{pset}.{key}"
        lining = rec["psets"].get("IfcDoorLiningProperties", {}).get("LiningThickness")
        if clear_w is None and overall_w and isinstance(lining, (int, float)):
            clear_w, clear_src = overall_w - 2 * lining, "OverallWidth-2*LiningThickness"
        if clear_w is None and overall_w:
            clear_w, clear_src = overall_w, "OverallWidth"
        # host wall: door -> IfcRelFillsElement -> opening -> IfcRelVoidsElement -> wall
        opening = self.rels.filled_opening.get(door.id())
        host = self.rels.voided_host.get(opening.id()) if opening is not None else None
        handed = None
        if op:
            handed = "both" if op.startswith("DOUBLE_DOOR") else \
                "left" if op.endswith("LEFT") else "right" if op.endswith("RIGHT") else None
        return {
            "overall_width_m": rnd(overall_w), "overall_height_m": rnd(attrs.get("OverallHeight")),
            "clear_width_m": rnd(clear_w), "clear_width_source": clear_src,
            "operation_type": op, "operation_type_source": op_source,
            "swing": {"kind": (None if not op else "swing" if "SWING" in op else op.lower()),
                      "handedness": handed},
            "host_wall": guid_of(host), "host_wall_class": host.is_a() if host is not None else None,
            "flags": {k: v for k, v in common.items() if isinstance(v, bool)},
            "is_external": common.get("IsExternal") if isinstance(common.get("IsExternal"), bool) else None,
        }

    def _host_thickness(self, door):
        opening = self.rels.filled_opening.get(door.id())
        host = self.rels.voided_host.get(opening.id()) if opening is not None else None
        mats = self.records[host.id()]["materials"] if host is not None and host.id() in self.records else None
        return mats.get("total_thickness_m") if isinstance(mats, dict) else None

    def _door_rooms(self, door, info):
        """(rooms, source) from boundaries, completed or tie-broken by the probe."""
        bnds = self.rels.boundaries_by_element.get(door.id(), [])
        rooms = list(dict.fromkeys(b["space"] for b in bnds if b["space"]))
        external = info.get("is_external") or any(b["internal_or_external"] == "EXTERNAL" for b in bnds)
        source = "relation" if rooms else None
        probe = None
        with self.audit.guard("door_probe", door):
            probe = self.door_probe(door, self._host_thickness(door))
        if probe is not None:
            sides = [guid_of(probe["minus"]), guid_of(probe["plus"])]
            if len(rooms) > 2 and sides[0] != sides[1] and all(s in rooms for s in sides):
                # boundaries of a third space touch the door (seen in Revit
                # exports); keep the two rooms the door actually separates
                rooms, source = sides, "relation+geometric"
            elif len(rooms) < 2 and not (rooms and external):
                extra = [g for g in dict.fromkeys(sides) if g and g not in rooms]
                if extra:
                    rooms += extra
                    source = "relation+geometric" if source else "geometric"
            swing_into = probe["minus"] if probe["mirrored"] else probe["plus"]
            info.setdefault("swing", {}).update(
                {"opens_towards": probe["normal"], "opens_into_room": guid_of(swing_into)})
            info["probe_offset_m"] = probe["offset_m"]
        if external and OUTSIDE not in rooms and len(rooms) == 1:
            rooms.append(OUTSIDE)
        return rooms, source

    # --- adjacency --------------------------------------------------------

    def link_doors(self, spaces, doors):
        """Fill door/space records and return the adjacency graph."""
        edges = {}

        def add_edge(a, b, via, via_class, source):
            if a == b or a is None or b is None:
                return
            key = (min(a, b), max(a, b), via)
            edges.setdefault(key, {"a": key[0], "b": key[1], "via": via,
                                   "via_class": via_class, "source": source})

        space_recs = {sp.GlobalId: self.records[sp.id()] for sp in spaces}
        for door in doors:
            rec = self.records[door.id()]
            info = {}
            with self.audit.guard("door", door):
                info = self.door_details(door, rec)
            rooms, source = self._door_rooms(door, info)
            info.update({"rooms": rooms, "room_link_source": source})
            rec["door"] = info
            if not rooms:
                self.audit.doors_without_room.append({"guid": door.GlobalId, "name": door.Name})
            elif source == "geometric":
                self.audit.doors_geometric_only.append({"guid": door.GlobalId, "name": door.Name})
            for r in rooms:
                if r in space_recs:
                    space_recs[r]["doors"].append(door.GlobalId)
            for i, a in enumerate(rooms):
                for b in rooms[i + 1:]:
                    add_edge(a, b, door.GlobalId, "IfcDoor", source)

        for via, group in self._virtual_groups().items():
            for s in group:
                if s in space_recs:
                    space_recs[s]["virtual_openings"].append(via)
            if len(group) < 2:
                self.audit.virtual_boundaries_without_edge.append({"via": via, "spaces": group})
            for i, a in enumerate(group):
                for b in group[i + 1:]:
                    add_edge(a, b, via, "virtual", "relation")

        for rec in space_recs.values():
            rec["doors"] = list(dict.fromkeys(rec["doors"]))
            rec["virtual_openings"] = list(dict.fromkeys(rec["virtual_openings"]))
        nodes = list(space_recs)
        if any(OUTSIDE in (e["a"], e["b"]) for e in edges.values()):
            nodes.append(OUTSIDE)
        return {"nodes": nodes, "edges": list(edges.values())}

    def _virtual_groups(self):
        """Virtual boundaries grouped into openings: by shared virtual element,
        or, without an element, by the CorrespondingBoundary pair. The pair is
        keyed by the smaller boundary GUID so both sides yield one opening."""
        groups = defaultdict(list)
        partner = {}           # symmetric, even if only one side sets the link
        for b in self.rels.space_boundaries:
            if b["corresponding_boundary"]:
                partner[b["guid"]] = b["corresponding_boundary"]
                partner.setdefault(b["corresponding_boundary"], b["guid"])
        for b in self.rels.space_boundaries:
            if b["physical_or_virtual"] != "VIRTUAL" or not b["space"]:
                continue
            if b["element"]:
                groups[b["element"]].append(b["space"])
                continue
            groups[min(b["guid"], partner.get(b["guid"], b["guid"]))].append(b["space"])
        return {via: list(dict.fromkeys(spaces)) for via, spaces in groups.items()}

    # --- equipment --------------------------------------------------------

    def place_equipment(self, elements):
        for e in elements:
            if not is_equipment(e):
                continue
            rec = self.records[e.id()]
            room, source = self.rels.space_of(e), "relation"
            if room is None:
                source = None
                c = (rec["geometry"] or {}).get("centroid")
                if c:
                    room = self.locate_room(c[0], c[1], c[2])
                    if room is not None:
                        source = "geometric"
                        self.audit.equipment_geometric.append({"guid": e.GlobalId, "name": e.Name})
            if room is None:
                self.audit.equipment_without_room.append({"guid": e.GlobalId, "name": e.Name})
            rec["room"], rec["room_source"] = guid_of(room), source
