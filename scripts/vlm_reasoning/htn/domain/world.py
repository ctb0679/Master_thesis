"""load_world(): the initial WorldState from the building JSON files, the
robot profile and the (assumed) initial door states.

The adjacency list is the _llm.json adjacency joined with the door blocks of
_full.json on the door GUID. Edges whose door, width, position or rooms are
missing are left out and reported, never guessed. Storeys come from the
cleaned JSON only; a room without one is reported, not placed by its z."""

import json
from pathlib import Path

import shapely

from htn.core import WorldState
from rank_candidates import room_keys

from .building import OUTSIDE, Door, Space, edge

DEFAULT_SETTINGS = {"prob_mass": 0.9, "min_prob": 0.01, "max_visits_per_room": 1, "distance": "door_graph"}


class WorldError(RuntimeError):
    """The building data cannot support HTN planning; the message says why."""


def default_door_states_path(llm_json, robot_dir):
    """data/robot/door_states_<stem>.json, else door_states_<first token of
    stem>.json (IDAC_Model_llm.json -> door_states_IDAC.json), else None."""
    stem = Path(llm_json).name.removesuffix("_llm.json").removesuffix(".json")
    for name in (stem, stem.split("_")[0]):
        path = Path(robot_dir) / f"door_states_{name}.json"
        if path.exists():
            return path
    return None


def read_door_states(path, door_guids):
    """{door guid: "open" | "closed"} from {"default": ..., "overrides": {...}};
    every door open when path is None."""
    if path is None:
        return {g: "open" for g in door_guids}, "assumed: all open (no door-states file)"
    data = json.loads(Path(path).read_text())
    default = data.get("default", "open")
    overrides = data.get("overrides", {})
    unknown = sorted(set(overrides) - set(door_guids))
    if unknown:
        raise WorldError(f"{path}: overrides for doors not in the building: {unknown}")
    bad = sorted({v for v in [default, *overrides.values()] if v not in ("open", "closed")})
    if bad:
        raise WorldError(f"{path}: door states must be 'open' or 'closed', got {bad}")
    return {g: overrides.get(g, default) for g in door_guids}, str(path)


def _approach(door_rec, rooms, footprints):
    """Door centroid +/- probe_offset_m along swing.opens_towards, each point
    labelled with the room whose footprint contains it ("outside" for the
    leftover point of an exterior door)."""
    d = door_rec["door"]
    towards, offset = (d.get("swing") or {}).get("opens_towards"), d.get("probe_offset_m")
    if not towards or not offset:
        return {}, f"door {door_rec['guid']}: no swing direction or probe offset, no approach points"
    x, y = door_rec["geometry"]["centroid"][:2]
    points = [(round(x + s * offset * towards[0], 3), round(y + s * offset * towards[1], 3)) for s in (1, -1)]
    out, unlabelled = {}, []
    for p in points:
        owner = next((r for r in rooms if r in footprints and footprints[r].contains(shapely.Point(p))), None)
        if owner and owner not in out:
            out[owner] = p
        else:
            unlabelled.append(p)
    leftover = [r for r in rooms if r not in out]
    if len(unlabelled) == 1 and leftover == [OUTSIDE]:
        out[OUTSIDE] = unlabelled[0]
    elif unlabelled:
        return out, f"door {door_rec['guid']}: approach point(s) {unlabelled} lie in neither room {list(rooms)}"
    return out, None


def _footprint(space):
    fp = (space.get("geometry") or {}).get("footprint")
    if not fp:
        return None
    return shapely.make_valid(shapely.MultiPolygon([shapely.Polygon(p[0], p[1:]) for p in fp]))


def load_world(llm_json, full_json, robot, door_states=None, start_space=None, settings=None):
    """-> (WorldState, report). `robot` is a RobotProfile, `door_states` a
    path or None, `start_space` a space GUID (default: the first room).
    `report` lists everything left out or unusual."""
    llm = json.loads(Path(llm_json).read_text())
    full = json.loads(Path(full_json).read_text())
    report = []

    length_unit = full.get("units", {}).get("output_units", {}).get("length")
    if length_unit != "m":
        raise WorldError(f"{full_json}: coordinates are in '{length_unit}', expected metres")

    keys = room_keys(llm["rooms"])
    spaces = {r["guid"]: Space(r["guid"], keys[r["guid"]], r.get("function") or "", tuple(r["centroid"][:2]),
                               r.get("storey"))
              for r in llm["rooms"] if r.get("centroid")}
    storey_names = {st["guid"]: st.get("name") for st in full.get("storeys", [])}
    no_storey = [s.name for s in spaces.values() if s.storey is None]
    if storey_names and no_storey:
        report.append(f"{len(no_storey)} room(s) have no storey in {Path(llm_json).name} although the IFC has "
                      f"{len(storey_names)} IfcBuildingStorey ({', '.join(map(str, storey_names.values()))}); "
                      f"they count as one storey of their own, not placed by z: {', '.join(no_storey)}")
    report += [f"room {keys[r['guid']]}: no centroid, left out" for r in llm["rooms"] if not r.get("centroid")]
    footprints = {s["guid"]: f for s in full.get("spaces", []) if (f := _footprint(s)) is not None}
    door_recs = {e["guid"]: e for e in full.get("elements", []) if e.get("door")}

    doors, adjacency = {}, {g: [] for g in spaces}
    for a in llm.get("adjacency", []):
        ends, via = (a["a"], a["b"]), a.get("via")
        rec = door_recs.get(via)
        where = f"edge {keys.get(ends[0], ends[0])} - {keys.get(ends[1], ends[1])} via {via}"
        if rec is None:
            report.append(f"{where}: no door record in {Path(full_json).name}, edge left out")
            continue
        d = rec["door"]
        problems = []
        if d.get("clear_width_m") is None:
            problems.append("no clear width")
        if not (rec.get("geometry") or {}).get("centroid"):
            problems.append("no position")
        if set(d.get("rooms") or []) != set(ends):
            problems.append(f"door rooms {d.get('rooms')} differ from the edge's")
        if any(e != OUTSIDE and e not in spaces for e in ends):
            problems.append("edge names an unknown room")
        if problems:
            report.append(f"{where}: {', '.join(problems)}; edge left out")
            continue
        approach, issue = _approach(rec, ends, footprints)
        if issue:
            report.append(issue)
        doors[via] = Door(via, ends, d["clear_width_m"], tuple(rec["geometry"]["centroid"][:2]), approach,
                          is_exterior=OUTSIDE in ends, name=rec.get("name") or "",
                          storey=storey_names.get(rec.get("storey")))
        if d.get("is_external") != (OUTSIDE in ends):
            report.append(f"door {via}: is_external={d.get('is_external')} but the edge "
                          f"{'touches' if OUTSIDE in ends else 'does not touch'} 'outside'; the edge decides")

    states, source = read_door_states(door_states, list(doors))
    for g, door in doors.items():
        door.state = states[g]
        a, b = door.rooms
        adjacency.setdefault(a, []).append(edge(g, b, door.state))
        adjacency.setdefault(b, []).append(edge(g, a, door.state))

    if not spaces:
        raise WorldError(f"{llm_json}: no rooms with a centroid")
    if start_space is not None and start_space not in spaces:
        raise WorldError(f"start space {start_space} is not a room of this building")
    state = WorldState(
        robot=robot.to_state(start_space or next(iter(spaces))),
        spaces={g: s.to_state() for g, s in spaces.items()},
        doors={g: d.to_state() for g, d in doors.items()},
        adjacency=adjacency,
        task={},
        belief={"prior": {}, "posterior": {}},
        search={"visits": {}, "stage": {}, "searched": [], "found": [], "observations": []},
        settings={**DEFAULT_SETTINGS, **(settings or {}), "door_states_source": source},
    )
    return state, report


def set_task(state, name, target="", mode="single", restricted=None, recorded=(), text=""):
    """Put the interpreted task into the state (rooms as GUIDs)."""
    state.task = {"name": name, "target": target, "mode": mode, "restricted": restricted,
                  "recorded": list(recorded), "text": text}
    state.search = {"visits": {}, "stage": {}, "searched": [], "found": [], "observations": []}
    state.belief = {"prior": {}, "posterior": {}}
    return state
