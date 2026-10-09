"""IfcExtractor: routes every rooted entity into a bucket, then runs the
passes (geometry, records, rooms/doors, equipment, groups, obstacles) and
assembles the full and LLM JSON views."""

import datetime
import multiprocessing
from pathlib import Path

import shapely

import ifcopenshell

from . import EXTRACTOR_VERSION
from .audit import AuditReport
from .common import OUTSIDE, guid_of, is_any, rnd
from .exporter import DROP_PSET_EXACT, EntityExporter
from .geometry import GeometryExtractor, describe_frame
from .obstacles import OVERHEAD_CLEARANCE_M, build_obstacles
from .relations import RelationshipResolver
from .rooms_doors import PROBE_MARGIN_M, RoomsAndDoors
from .units import UnitResolver

# Rooted classes (and subclasses) never exported.
SKIP_CLASSES = (
    "IfcAnnotation", "IfcGrid",
    "IfcStructuralItem", "IfcStructuralActivity", "IfcStructuralAnalysisModel",
)

# Checked in this order. IfcSpatialZone is a spatial element but behaves like
# a group, so groups come before spatial. IfcProject is not an IfcContext in
# IFC2X3, hence listed explicitly.
GROUP_CLASSES = ("IfcGroup", "IfcSpatialZone")
SPATIAL_CLASSES = ("IfcProject", "IfcContext", "IfcSpatialElement", "IfcSpatialStructureElement")


class IfcExtractor:
    def __init__(self, input_path, drop_psets=(), threads=None):
        self.input_path = Path(input_path)
        self.model = ifcopenshell.open(str(input_path))
        self.audit = AuditReport()
        self.units = UnitResolver(self.model, self.audit)
        self.rels = RelationshipResolver(self.model, self.audit)
        self.geom = GeometryExtractor(self.model, self.units, self.audit,
                                      threads or multiprocessing.cpu_count())
        self.drop_psets = set(DROP_PSET_EXACT) | set(drop_psets)
        self.exporter = EntityExporter(self.units, self.rels, self.geom, self.audit, self.drop_psets)
        self.buckets = {k: [] for k in ("spatial", "storeys", "spaces", "elements", "types", "groups")}
        self.records = {}          # entity id -> record (GUIDs may be duplicated)
        self.floor_z = {}
        self.rooms = None
        self.adjacency = None
        self.obstacles = None

    def _route(self):
        seen_guids = set()
        for e in self.model.by_type("IfcRoot"):
            cls = e.is_a()
            if is_any(e, SKIP_CLASSES):
                self.audit.skipped[cls] += 1
                continue
            if e.is_a("IfcRelationship") or e.is_a("IfcPropertyDefinition") or not e.is_a("IfcObjectDefinition"):
                self.audit.folded[cls] += 1      # folded onto the objects they relate
                continue
            if e.GlobalId in seen_guids:
                self.audit.duplicate_guids.append(e.GlobalId)
            seen_guids.add(e.GlobalId)
            if e.is_a("IfcTypeObject"):
                bucket = "types"
            elif e.is_a("IfcBuildingStorey"):
                bucket = "storeys"
            elif e.is_a("IfcSpace"):
                bucket = "spaces"
            elif is_any(e, GROUP_CLASSES):
                bucket = "groups"
            elif is_any(e, SPATIAL_CLASSES):
                bucket = "spatial"
            else:
                bucket = "elements"
            self.buckets[bucket].append(e)
            self.audit.exported[cls] += 1

    def _all_objects(self):
        for bucket in self.buckets.values():
            yield from bucket

    def run(self):
        self._route()
        self.geom.run(self.geom.select(list(self._all_objects()), self.rels))
        for e in self._all_objects():
            self.records[e.id()] = self.exporter.export(e)
        self._storeys()
        self.rooms = RoomsAndDoors(self.rels, self.geom, self.records, self.audit)
        self.rooms.enrich_spaces(self.buckets["spaces"])
        doors = [e for e in self.buckets["elements"] if e.is_a("IfcDoor")]
        self.adjacency = self.rooms.link_doors(self.buckets["spaces"], doors)
        self.rooms.place_equipment(self.buckets["elements"])
        self._groups()
        self.obstacles = build_obstacles(self.buckets["storeys"], self.buckets["elements"], self.rels,
                                         self.geom, self.floor_z, self.rooms.space_index, self.audit)

    def _storeys(self):
        for s in self.buckets["storeys"]:
            rec = self.records[s.id()]
            elev = rec["attributes"].get("Elevation")
            m = self.geom.placement(s)
            floor = float(m[2, 3]) if m is not None else elev
            rec.update({"elevation_m": rnd(elev), "floor_z_m": rnd(floor), "spaces": []})
            self.floor_z[s.id()] = floor if floor is not None else 0.0

    def _groups(self):
        for g in self.buckets["groups"]:
            members = list(self.rels.group_members.get(g.id(), ()))
            members += self.rels.referenced.get(g.id(), [])
            members += self.rels.contained.get(g.id(), [])
            self.records[g.id()]["members"] = list(dict.fromkeys(guid_of(m) for m in members))

    # --- output views -----------------------------------------------------

    def full_json(self):
        recs = {k: [self.records[e.id()] for e in v] for k, v in self.buckets.items()}
        return {
            "meta": {
                "source_file": self.input_path.name,
                "schema": self.model.schema_identifier,
                "extractor": f"ifc_extract {EXTRACTOR_VERSION}",
                "ifcopenshell": ifcopenshell.version,
                "shapely": shapely.__version__,
                "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                "project": next((p.Name for p in self.model.by_type("IfcProject")), None),
                "drop_psets": sorted(self.drop_psets),
                "settings": {"door_probe_margin_m": PROBE_MARGIN_M,
                             "overhead_clearance_m": OVERHEAD_CLEARANCE_M},
            },
            "frame": describe_frame(self.model, self.units, self.geom, self.audit),
            "units": self.units.describe(),
            "spatial": recs["spatial"],
            "storeys": recs["storeys"],
            "spaces": recs["spaces"],
            "elements": recs["elements"],
            "types": recs["types"],
            "groups": recs["groups"],
            "space_boundaries": self.rels.space_boundaries,
            "adjacency": self.adjacency,
            "obstacles": self.obstacles,
            "audit": self.audit.to_dict(self.model, self.units),
        }

    @staticmethod
    def llm_json(full):
        storey_names = {s["guid"]: s["name"] for s in full["storeys"]}
        rooms = [{
            "guid": s["guid"], "name": s["name"],
            "function": s["function"], "function_source": s["function_source"],
            "storey": storey_names.get(s["storey"]),
            "area_m2": s["area_m2"],
            "centroid": s["geometry"]["centroid"][:2] if s["geometry"] and s["geometry"]["centroid"] else None,
        } for s in full["spaces"]]
        equipment = [{
            "guid": e["guid"], "name": e["name"], "ifc_class": e["ifc_class"],
            "type": (e["type_object"] or {}).get("name") or e["resolved_type"],
            "room": e["room"], "room_source": e["room_source"],
        } for e in full["elements"] if "room_source" in e]
        return {
            "meta": {"source_file": full["meta"]["source_file"], "schema": full["meta"]["schema"],
                     "units": "metres", "outside_node": OUTSIDE},
            "rooms": rooms,
            "equipment": equipment,
            "adjacency": [{k: e[k] for k in ("a", "b", "via", "via_class")} for e in full["adjacency"]["edges"]],
        }
