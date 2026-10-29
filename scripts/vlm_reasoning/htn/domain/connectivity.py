"""Connectivity report for the HTN load step: how the building's rooms hang
together through doors, as the cleaned JSON describes them. Nothing is
inferred: no door, edge or stair link is added. Printed once when the
building is loaded and written as connectivity.json into every run folder."""

import json
from pathlib import Path

from .building import OUTSIDE
from .methods import reachability


def unlinked_stairs(full_json, space_guids):
    """IfcStair (and IfcStairFlight without a parent stair) that the cleaned
    JSON does not relate to any space. A stair counts as linked only when it
    is contained in a space or is the `via` of an adjacency edge; one that
    is only contained in a storey is not linked."""
    full = json.loads(Path(full_json).read_text())
    storeys = {st["guid"]: st.get("name") for st in full.get("storeys", [])}
    vias = {e.get("via") for e in (full.get("adjacency") or {}).get("edges", [])}
    stair_guids = {e["guid"] for e in full.get("elements", []) if e.get("ifc_class") == "IfcStair"}
    out = []
    for e in full.get("elements", []):
        rel = e.get("relations") or {}
        if e.get("ifc_class") == "IfcStairFlight" and rel.get("aggregate_parent") in stair_guids:
            continue                                   # part of a stair listed on its own
        if e.get("ifc_class") not in ("IfcStair", "IfcStairFlight"):
            continue
        if rel.get("container") in space_guids or e["guid"] in vias:
            continue
        out.append({"guid": e["guid"], "name": e.get("name"), "storey": storeys.get(e.get("storey"))})
    return out


def connectivity_report(state, stairs):
    """-> dict: groups of rooms connected through interior doors (outside
    is not a connector), the rooms this robot can reach from its current
    space, `stairs` (from unlinked_stairs) and the rooms with no door."""
    spaces = state.spaces

    def room(g):
        s = spaces[g]
        return {"space": g, "name": s["name"], "function": s.get("function", ""), "storey": s.get("storey")}

    neighbours = {g: set() for g in spaces}
    exterior = {g: 0 for g in spaces}
    for d in state.doors.values():
        a, b = d["rooms"]
        if OUTSIDE in (a, b):
            exterior[b if a == OUTSIDE else a] += 1
        else:
            neighbours[a].add(b)
            neighbours[b].add(a)
    groups, seen = [], set()
    for g in spaces:
        if g in seen:
            continue
        stack, members = [g], []
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x)
            members.append(x)
            stack.extend(neighbours[x] - seen)
        groups.append(members)
    groups.sort(key=lambda m: (-len(m), spaces[m[0]]["name"]))

    reachable, _ = reachability(state)
    return {
        "start_space": room(state.robot["space"]),
        "storeys": sorted({s.get("storey") for s in spaces.values()}, key=lambda x: (x is None, str(x))),
        "groups": [{"rooms": [room(g) for g in m], "storeys": sorted({str(spaces[g].get("storey")) for g in m}),
                    "exterior_doors": sum(exterior[g] for g in m)} for m in groups],
        "reachable_from_start": [room(g) for g in reachable],
        "unlinked_stairs": stairs,
        "rooms_without_door": [room(g) for g in spaces if not neighbours[g] and not exterior[g]],
    }


def format_connectivity(report, label):
    """The report as printable lines; `label(guid)` names a room."""
    names = lambda rooms: ", ".join(label(r["space"]) for r in rooms) or "-"
    groups, stairs = report["groups"], report["unlinked_stairs"]
    n_rooms = sum(len(g["rooms"]) for g in groups)
    lines = [f"Connectivity: {len(groups)} group(s) of rooms connected through interior doors; "
             f"{len(report['reachable_from_start'])} of {n_rooms} rooms reachable from "
             f"{label(report['start_space']['space'])} for this robot; "
             f"{len(stairs)} stair(s) not linked to any space; "
             f"{len(report['rooms_without_door'])} room(s) with no door."]
    for i, g in enumerate(groups, 1):
        lines.append(f"  group {i} ({len(g['rooms'])} room(s), storey {'/'.join(g['storeys'])}, "
                     f"{g['exterior_doors']} exterior door(s)): {names(g['rooms'])}")
    lines.append(f"  reachable from {label(report['start_space']['space'])}: {names(report['reachable_from_start'])}")
    if stairs:
        per = {}
        for s in stairs:
            per[str(s["storey"])] = per.get(str(s["storey"]), 0) + 1
        lines.append("  stairs not linked to any space: "
                     + ", ".join(f"{k}: {v}" for k, v in sorted(per.items())))
    if report["rooms_without_door"]:
        lines.append(f"  rooms with no door: {names(report['rooms_without_door'])}")
    return lines
