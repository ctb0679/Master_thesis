"""Geometry: one multi-threaded ifcopenshell.geom.iterator pass, each mesh
reduced to centroid, 3D bbox, 2D footprint and yaw; plus the coordinate frame
(map conversion, site placement, true north).

World coordinates in metres; opening subtraction stays on so walls have gaps
at doors."""

import math
from collections import defaultdict

import numpy as np
import shapely
from shapely.geometry import mapping
from shapely.validation import make_valid

import ifcopenshell.geom
import ifcopenshell.util.geolocation as ifc_geolocation
import ifcopenshell.util.placement as ifc_placement
import ifcopenshell.util.unit as ifc_unit

from .common import COORD_DECIMALS, by_type, rnd


class GeometryExtractor:
    def __init__(self, model, units, audit, threads):
        self.model, self.units, self.audit, self.threads = model, units, audit, threads
        self.shapes = {}       # entity id -> {"bbox_min", "bbox_max", "footprint"(shapely)}

    def select(self, objects, rels):
        """Products worth tessellating. IfcSite (terrain) needs no footprint,
        and an opening only matters when it voids a slab (a fall hazard). Both
        still get placement-based fields; the skips are counted in the audit."""
        out = []
        for e in objects:
            if not e.is_a("IfcProduct") or getattr(e, "Representation", None) is None:
                continue
            if e.is_a("IfcSite") or (e.is_a("IfcOpeningElement") and not voids_slab(e, rels)):
                self.audit.tessellation_skipped[e.is_a()] += 1
                continue
            out.append(e)
        return out

    def run(self, products):
        if not products:
            return
        settings = ifcopenshell.geom.settings()
        settings.set("use-world-coords", True)
        it = ifcopenshell.geom.iterator(settings, self.model, self.threads, include=products)
        meshes = defaultdict(list)
        if it.initialize():
            while True:
                shape = it.get()
                with self.audit.guard("geometry_mesh", self.model.by_id(shape.id)):
                    verts = np.asarray(shape.geometry.verts, dtype=float).reshape(-1, 3)
                    faces = np.asarray(shape.geometry.faces, dtype=np.int64).reshape(-1, 3)
                    if len(verts):
                        meshes[shape.id].append((verts, faces))
                if not it.next():
                    break
        for product in products:
            parts = meshes.get(product.id())
            if not parts:
                self.audit.geometry_failures[product.is_a()] += 1
                continue
            with self.audit.guard("geometry_footprint", product):
                self.shapes[product.id()] = self._summarise(parts)

    @staticmethod
    def _summarise(parts):
        verts = np.vstack([v for v, _ in parts])
        tri_parts = [v[f] for v, f in parts if len(f)]
        footprint = None
        if tri_parts:
            xy = np.vstack(tri_parts)[:, :, :2]
            area2 = np.abs((xy[:, 1, 0] - xy[:, 0, 0]) * (xy[:, 2, 1] - xy[:, 0, 1])
                           - (xy[:, 2, 0] - xy[:, 0, 0]) * (xy[:, 1, 1] - xy[:, 0, 1]))
            xy = xy[area2 > 2e-6]          # drop vertical faces (zero projected area)
            if len(xy):
                # snapped to the 1 mm output grid, so rounding in footprint_json
                # cannot make the polygon invalid (make_grid.py reads it back)
                merged = shapely.union_all(shapely.polygons(xy), grid_size=10.0 ** -COORD_DECIMALS)
                footprint = _polygonal(make_valid(merged)).simplify(0.002)
                if footprint.is_empty:
                    footprint = None
        return {"bbox_min": verts.min(axis=0), "bbox_max": verts.max(axis=0), "footprint": footprint}

    def placement(self, entity):
        """Absolute 4x4 placement with translation in metres, or None."""
        op = getattr(entity, "ObjectPlacement", None)
        if op is None:
            return None
        with self.audit.guard("placement", entity):
            m = np.array(ifc_placement.get_local_placement(op), dtype=float)
            m[:3, 3] *= self.units.length
            return m
        return None

    def fields(self, entity):
        """The four per-product geometry fields, JSON-ready."""
        shape = self.shapes.get(entity.id())
        m = self.placement(entity)
        yaw = rnd(math.degrees(math.atan2(m[1, 0], m[0, 0])), 2) if m is not None else None
        if shape is None:
            return None if m is None else {
                "centroid": None, "bbox": None, "footprint": None, "yaw_deg": yaw,
                "placement_origin": [rnd(v) for v in m[:3, 3]],
            }
        lo, hi = shape["bbox_min"], shape["bbox_max"]
        fp = shape["footprint"]
        cx, cy = (fp.centroid.x, fp.centroid.y) if fp is not None else ((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2)
        return {
            "centroid": [rnd(cx), rnd(cy), rnd((lo[2] + hi[2]) / 2)],
            "bbox": {"min": [rnd(v) for v in lo], "max": [rnd(v) for v in hi]},
            "footprint": footprint_json(fp),
            "yaw_deg": yaw,
        }


def voids_slab(opening, rels):
    host = rels.voided_host.get(opening.id())
    return host is not None and host.is_a("IfcSlab")


def _polygonal(geom):
    """Keep only the polygonal part of a make_valid() result."""
    if geom.geom_type in ("Polygon", "MultiPolygon"):
        return geom
    polys = [g for g in getattr(geom, "geoms", ()) if g.geom_type in ("Polygon", "MultiPolygon")]
    return shapely.union_all(polys) if polys else shapely.Polygon()


def footprint_json(geom):
    """MultiPolygon coordinates: [polygon][ring][point] = [x, y]."""
    if geom is None or geom.is_empty:
        return None
    coords = mapping(geom)["coordinates"]
    polys = [coords] if geom.geom_type == "Polygon" else coords
    return [[[[rnd(x), rnd(y)] for x, y, *_ in ring] for ring in poly] for poly in polys]


def _dms(value):
    """IfcCompoundPlaneAngleMeasure (deg, min, sec, millionths) -> decimal degrees."""
    if not value:
        return None
    parts = list(value) + [0] * (4 - len(value))
    d, m, s, us = parts[:4]
    sign = -1 if any(p < 0 for p in parts[:4]) else 1
    return round(sign * (abs(d) + abs(m) / 60 + abs(s) / 3600 + abs(us) / 3.6e9), 8)


def describe_frame(model, units, geom, audit):
    """Top-level "frame": how project coordinates relate to the world."""
    out = {
        "coordinates": "project engineering coordinates in metres; "
                       "map conversion is recorded here, not applied",
        "map_conversion": None, "site": None, "true_north_deg": None,
    }
    with audit.guard("frame", "IfcMapConversion"):
        mcs = by_type(model, "IfcMapConversion")
        if mcs:
            mc, crs = mcs[0], mcs[0].TargetCRS
            ax, ay = mc.XAxisAbscissa, mc.XAxisOrdinate
            map_unit = getattr(crs, "MapUnit", None) if crs is not None else None
            out["map_conversion"] = {
                "eastings": mc.Eastings, "northings": mc.Northings,
                "orthogonal_height": mc.OrthogonalHeight,
                "x_axis_abscissa": ax, "x_axis_ordinate": ay,
                "rotation_deg": rnd(math.degrees(math.atan2(ay, ax)), 6) if ax is not None and ay is not None else None,
                "scale": mc.Scale,
                "target_crs": None if crs is None else {
                    "name": crs.Name, "description": crs.Description,
                    "geodetic_datum": getattr(crs, "GeodeticDatum", None),
                    "vertical_datum": getattr(crs, "VerticalDatum", None),
                    "map_projection": getattr(crs, "MapProjection", None),
                    "map_zone": getattr(crs, "MapZone", None),
                    "map_unit": ifc_unit.get_full_unit_name(map_unit) if map_unit is not None else None,
                },
            }
    with audit.guard("frame", "IfcSite"):
        sites = model.by_type("IfcSite")
        if sites:
            site = sites[0]
            m = geom.placement(site)
            out["site"] = {
                "guid": site.GlobalId,
                "origin": [rnd(v) for v in m[:3, 3]] if m is not None else None,
                "yaw_deg": rnd(math.degrees(math.atan2(m[1, 0], m[0, 0])), 4) if m is not None else None,
                "ref_latitude_deg": _dms(site.RefLatitude),
                "ref_longitude_deg": _dms(site.RefLongitude),
                "ref_elevation_m": rnd(units.convert(site.RefElevation, "IfcLengthMeasure")),
            }
    with audit.guard("frame", "TrueNorth"):
        out["true_north_deg"] = rnd(ifc_geolocation.get_true_north(model), 4)
        out["true_north_convention"] = "angle from project +Y to true north, anticlockwise positive"
    return out
