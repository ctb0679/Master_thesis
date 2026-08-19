"""
Stage 1: deterministic IFC extraction (no LLM).

Usage:
    python3 extract_ifc.py <input.ifc> <output.json>

Example:
    python3 scripts/extraction/extract_ifc.py \\
        data/raw_ifc/ac20_office.ifc \\
        data/extracted/ac20_extracted.json
"""

import sys
import json
import ifcopenshell
import ifcopenshell.util.element as elutil


def extract_building(f):
    data = {"spaces": [], "doors": [], "windows": [], "storeys": []}

    for storey in f.by_type("IfcBuildingStorey"):
        data["storeys"].append({"id": storey.GlobalId, "name": storey.Name})

    rel_contained = f.by_type("IfcRelContainedInSpatialStructure")

    for s in f.by_type("IfcSpace"):
        storey_name = None
        for rel in rel_contained:
            if s in rel.RelatedElements:
                storey_name = rel.RelatingStructure.Name

        psets = elutil.get_psets(s)
        area = None
        for pname, pdata in psets.items():
            for k, v in pdata.items():
                if "area" in k.lower() and isinstance(v, (int, float)):
                    area = v

        data["spaces"].append({
            "guid": s.GlobalId,
            "number": s.Name,
            "function": s.LongName,       # raw label, not yet categorized
            "function_category": None,     # filled in by Stage 2 classifier
            "storey": storey_name,
            "area_m2": area,
        })

    for d in f.by_type("IfcDoor"):
        data["doors"].append({"guid": d.GlobalId, "name": d.Name})

    for w in f.by_type("IfcWindow"):
        data["windows"].append({"guid": w.GlobalId, "name": w.Name})

    return data


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 extract_ifc.py <input.ifc> <output.json>")
        sys.exit(1)

    ifc_path, out_path = sys.argv[1], sys.argv[2]
    f = ifcopenshell.open(ifc_path)
    building = extract_building(f)

    with open(out_path, "w") as out:
        json.dump(building, out, indent=2, ensure_ascii=False)

    print(f"Extracted {len(building['spaces'])} spaces, "
          f"{len(building['doors'])} doors, "
          f"{len(building['windows'])} windows -> {out_path}")
