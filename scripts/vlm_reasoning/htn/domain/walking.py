"""Walking distances from navbim for the visit order (settings distance = "navbim").

The table is made by scripts/navbim/navbim_distance_table.py from navbim's
pre-planned map (README section 6). Default file:
data/distances/<building>_preplanned_centre_walking_distances.json
(option B, room centre to room centre), rooms keyed by IfcSpace GUID, the
same GUIDs as in <building>_llm.json.

navbim knows walls but not door states or the robot's door rules. So a
navbim distance is only used when every door on navbim's route is passable
for this robot (door_problem() is None). Otherwise, e.g. a closed door the
robot cannot open, the door-graph route length is used for that pair; such
pairs are collected in distance.fallbacks."""

import functools
import json
from pathlib import Path

from .building import DoorGraph, door_problem

ROOT = Path(__file__).resolve().parents[4]
DISTANCES_DIR = ROOT / "data" / "distances"


def default_table_path(llm_json):
    """data/distances/<stem>_preplanned_centre_walking_distances.json for
    <stem>_llm.json."""
    stem = Path(llm_json).name.removesuffix("_llm.json").removesuffix(".json")
    return DISTANCES_DIR / f"{stem}_preplanned_centre_walking_distances.json"


@functools.lru_cache(maxsize=8)
def load_table(path):
    """The table as {"method", "index" {guid: row}, "matrix", "routes"}."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {"method": data.get("method"),
            "index": {r["guid"]: i for i, r in enumerate(data["rooms"])},
            "matrix": data["matrix_m"],
            "routes": data.get("routes", {})}


def table_problem(path, space_guids):
    """Why the table cannot be used for these rooms (a message), or None."""
    if path is None or not Path(path).exists():
        return f"no walking-distance table at {path}"
    try:
        table = load_table(str(path))
    except (OSError, ValueError, KeyError) as e:
        return f"{path} cannot be read ({e})"
    missing = sorted(set(space_guids) - set(table["index"]))
    if missing:
        return (f"{path} lacks {len(missing)} room(s) of this building (e.g. {missing[:3]}); "
                "was it made from the same IFC file?")
    if any("door_guids" not in r for r in table["routes"].values()):
        return f"{path} was made by an older navbim_distance_table.py (no door GUIDs); build it again"
    return None


def walking_distance(state, graph=None):
    """distance(a, b) -> metres for the visit order: navbim's walking
    distance when its route is passable for state.robot, else the door-graph
    route length (None when b cannot be reached at all). distance.fallbacks
    holds the (a, b) pairs that used the door graph."""
    graph = graph or DoorGraph.from_state(state)
    table = load_table(str(state.settings["distance_table"]))
    robot = state.robot

    def distance(a, b):
        if a == b:
            return 0.0
        i, j = table["index"].get(a), table["index"].get(b)
        route = table["routes"].get(f"{a}->{b}")
        if (i is not None and j is not None and route is not None
                and table["matrix"][i][j] is not None and graph.same_storey(a, b)):
            doors = route.get("door_guids") or []
            if all(d in graph.doors and door_problem(graph.doors[d], robot) is None for d in doors):
                return table["matrix"][i][j]
        distance.fallbacks.add((a, b))
        return graph.route_length(a, b, robot)

    distance.fallbacks = set()
    return distance
