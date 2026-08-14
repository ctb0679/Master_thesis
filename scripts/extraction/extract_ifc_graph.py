"""
Stage 1 (extended): deterministic IFC extraction with spatial relationships,
structured for graph import (Neo4j).

Usage:
    python3 extract_ifc_graph.py <input.ifc> <output.json>
"""

import sys
import json
import ifcopenshell
import ifcopenshell.util.element as elutil


def extract_building(f):
    data = {"storeys": [], "spaces": [], "doors": [], "windows": [], "equipment": []}

    storey_entities = f.by_type("IfcBuildingStorey")
    if storey_entities:
        for storey in storey_entities:
            data["storeys"].append({"guid": storey.GlobalId, "name": storey.Name})
    else:
        # Some models have no IfcBuildingStorey at all -- spaces/elements are
        # aggregated directly under IfcSite instead (seen on IDAC_Model.ifc:
        # a genuinely single-floor building authored without an explicit
        # storey entity). Synthesize one storey from the site so ON_STOREY
        # still works, and the schema stays unchanged for real multi-storey
        # buildings used in the "unseen buildings" evaluation.
        for site in f.by_type("IfcSite"):
            data["storeys"].append({"guid": site.GlobalId, "name": site.Name})

    rel_contained = f.by_type("IfcRelContainedInSpatialStructure")
    rel_aggregates = f.by_type("IfcRelAggregates")
    boundaries = f.by_type("IfcRelSpaceBoundary")

    # storey lookup: any element -> storey name, via IfcRelContainedInSpatialStructure
    elem_to_storey = {}
    for rel in rel_contained:
        for elem in rel.RelatedElements:
            elem_to_storey[elem.id()] = rel.RelatingStructure.Name

    # Spaces aren't always related to their storey via
    # IfcRelContainedInSpatialStructure (the loop above) -- some models
    # aggregate a space directly under its IfcBuildingStorey, or (this
    # model's case, since there is no storey) directly under IfcSite, via
    # IfcRelAggregates instead. Without this, every space's "storey" field
    # would silently stay None even though the fallback storey above exists.
    for rel in rel_aggregates:
        parent = rel.RelatingObject
        if parent.is_a("IfcBuildingStorey") or parent.is_a("IfcSite"):
            for child in rel.RelatedObjects:
                if child.is_a("IfcSpace"):
                    elem_to_storey.setdefault(child.id(), parent.Name)

    # full containing entity (space, storey, or site) per element -- needed
    # for equipment, since furniture/equipment is usually contained directly
    # in an IfcSpace via this same relationship, not just a storey.
    # Doors/windows/spaces keep using elem_to_storey above unchanged, since
    # that behavior is already validated against ac20_office.ifc ground truth.
    elem_to_container = {}
    for rel in rel_contained:
        for elem in rel.RelatedElements:
            elem_to_container[elem.id()] = rel.RelatingStructure

    # element -> list of space LongNames it bounds, via IfcRelSpaceBoundary
    elem_to_spaces = {}
    for b in boundaries:
        elem = b.RelatedBuildingElement
        if elem:
            elem_to_spaces.setdefault(elem.id(), []).append(b.RelatingSpace.GlobalId)

    for s in f.by_type("IfcSpace"):
        psets = elutil.get_psets(s)
        # Area field names are standardized by IFC even though the *pset* name
        # that contains them is not (Revit: "Qto_SpaceBaseQuantities",
        # ArchiCAD: "BaseQuantities"). Search by exact field name across all
        # psets, preferring NetFloorArea then GrossFloorArea. A loose
        # "area" in k.lower() substring match was tried first and rejected:
        # it grabbed unrelated fields like "Specified Power Load per area"
        # (an Energy Analysis design constant, identical across all rooms).
        area = None
        for field_name in ("NetFloorArea", "GrossFloorArea"):
            for pdata in psets.values():
                if field_name in pdata and isinstance(pdata[field_name], (int, float)):
                    area = pdata[field_name]
                    break
            if area is not None:
                break
        data["spaces"].append({
            "guid": s.GlobalId,
            "number": s.Name,
            "function": s.LongName,
            "function_category": None,   # filled in by Stage 2 classifier
            "storey": elem_to_storey.get(s.id()),
            "area_m2": area,
        })

    for d in f.by_type("IfcDoor"):
        data["doors"].append({
            "guid": d.GlobalId,
            "name": d.Name,
            "storey": elem_to_storey.get(d.id()),
            "bounds_spaces": elem_to_spaces.get(d.id(), []),  # list of space GUIDs
        })

    for w in f.by_type("IfcWindow"):
        data["windows"].append({
            "guid": w.GlobalId,
            "name": w.Name,
            "storey": elem_to_storey.get(w.id()),
            "bounds_spaces": elem_to_spaces.get(w.id(), []),
        })

    # Office items come from TWO different IFC classes in this model, and
    # both are needed for complete coverage:
    #   - IfcFurniture: desks, chairs, closets, shelving, tables, a printer.
    #     A proper IFC furniture class with real semantics.
    #   - IfcBuildingElementProxy: the generic catch-all class used when the
    #     authoring tool's object wasn't mapped to a proper IFC type -- here,
    #     monitors and a projection screen. Carries no structured properties
    #     beyond Name/ObjectType, unlike IfcFurniture.
    # Both are treated as one "office item" category in the graph (label
    # Equipment) since they're both findable physical objects, not
    # architecture -- but the source IFC class is kept as `ifc_type` in case
    # that distinction ever matters.
    for ifc_type in ("IfcFurniture", "IfcBuildingElementProxy"):
        for e in f.by_type(ifc_type):
            container = elem_to_container.get(e.id())
            located_in_space = (
                container.GlobalId if (container and container.is_a("IfcSpace")) else None
            )
            if container is None:
                storey_name = None
            elif container.is_a("IfcSpace"):
                # the item's own storey isn't in elem_to_storey (that dict is
                # keyed by the item's id, not the space's id) -- look it up
                # via the containing space's id instead.
                storey_name = elem_to_storey.get(container.id())
            else:
                # IfcBuildingStorey, or IfcSite when there's no storey level
                storey_name = container.Name

            raw_name = e.Name
            # Revit item names are typically "Family:Type:ElementId"
            # (e.g. "Computer monitor:Computer monitor:2485087"). Keep the
            # raw string but surface a clean label = text before the first ':'.
            clean_name = raw_name.split(":")[0].strip() if raw_name else raw_name

            data["equipment"].append({
                "guid": e.GlobalId,
                "name": clean_name,
                "raw_name": raw_name,
                "ifc_type": ifc_type,
                "object_type": e.ObjectType,
                "storey": storey_name,
                "located_in_space": located_in_space,  # space GUID, or None
                                                         # if only storey-level
                                                         # containment is known
            })

    return data


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 extract_ifc_graph.py <input.ifc> <output.json>")
        sys.exit(1)

    ifc_path, out_path = sys.argv[1], sys.argv[2]
    f = ifcopenshell.open(ifc_path)
    building = extract_building(f)

    with open(out_path, "w") as out:
        json.dump(building, out, indent=2, ensure_ascii=False)

    n_bounded_doors = sum(1 for d in building["doors"] if d["bounds_spaces"])
    n_located_equipment = sum(1 for e in building["equipment"] if e["located_in_space"])
    n_furniture = sum(1 for e in building["equipment"] if e["ifc_type"] == "IfcFurniture")
    n_proxy = sum(1 for e in building["equipment"] if e["ifc_type"] == "IfcBuildingElementProxy")
    print(f"Extracted {len(building['storeys'])} storeys, "
          f"{len(building['spaces'])} spaces, "
          f"{len(building['doors'])} doors ({n_bounded_doors} with space links), "
          f"{len(building['windows'])} windows, "
          f"{len(building['equipment'])} equipment items "
          f"({n_furniture} furniture, {n_proxy} proxy, "
          f"{n_located_equipment} with room-level location) -> {out_path}")
