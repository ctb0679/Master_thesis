"""Spaces, doors and the door graph of a building (from _llm.json + _full.json).

Paper (Sec. II): "a door-based adjacency list where each edge stores the door
id, the door action required to pass it, and the capabilities needed to
traverse it". Deviation (docs/htn_design.md, 8): routes are the shortest
paths by length through door positions (Dijkstra), not the fewest-doors path
(BFS), so route() and route_length() always describe the same path.

Single-storey planning: a route never enters a space on another storey and
no distance between storeys is computed (no stair or lift traversal)."""

import heapq
import math
from dataclasses import dataclass, field

from htn.core import PlanFailure, Problem

OUTSIDE = "outside"


@dataclass
class Space:
    """A room (IfcSpace). `name` is the room key the user and the ranking
    model see (rank_candidates.room_keys)."""
    guid: str
    name: str
    function: str = ""
    centroid: tuple = None
    storey: str = None              # storey name from the cleaned JSON; None if the IFC has none

    @property
    def label(self):
        return f"{self.name} ({self.function})" if self.function and self.function != self.name else self.name

    def to_state(self):
        return {"name": self.name, "function": self.function, "centroid": list(self.centroid),
                "storey": self.storey}


@dataclass
class Door:
    """A door between two spaces (or a space and "outside"). `approach` maps
    each side's space to a point probe_offset_m from the door centroid, for a
    later navigation exporter."""
    guid: str
    rooms: tuple
    clear_width_m: float
    position: tuple
    approach: dict = field(default_factory=dict)
    is_exterior: bool = False
    state: str = "open"
    name: str = ""
    storey: str = None

    def other(self, space):
        a, b = self.rooms
        return b if space == a else a

    def to_state(self):
        return {"name": self.name, "rooms": list(self.rooms), "clear_width_m": self.clear_width_m,
                "position": list(self.position), "approach": {k: list(v) for k, v in self.approach.items()},
                "is_exterior": self.is_exterior, "state": self.state, "storey": self.storey}

    @classmethod
    def from_state(cls, guid, d):
        return cls(guid, tuple(d["rooms"]), d["clear_width_m"], tuple(d["position"]),
                   {k: tuple(v) for k, v in d.get("approach", {}).items()},
                   d.get("is_exterior", False), d.get("state", "open"), d.get("name", ""), d.get("storey"))


def edge(door_guid, to_space, door_state):
    """One adjacency-list edge, as in the paper."""
    closed = door_state != "open"
    return {"to": to_space, "door": door_guid,
            "door_action": "open" if closed else "none",
            "capabilities": ["manipulate_door", "navigate"] if closed else ["navigate"]}


def door_problem(door, robot):
    """Why the robot cannot pass this door (a Problem), or None. The
    traversal rule from the specification."""
    if door.is_exterior and not robot.get("allow_exterior", False):
        return Problem("exterior_not_allowed",
                       f"door {door.guid} leads outside and the robot profile has allow_exterior = false")
    need = robot["width_m"] + robot["safety_margin_m"]
    if door.clear_width_m is None or door.clear_width_m < need:
        return Problem("door_too_narrow",
                       f"door {door.guid} is {door.clear_width_m} m wide; robot needs {need:.3f} m "
                       f"(width {robot['width_m']} + margin {robot['safety_margin_m']})")
    if door.state != "open" and "manipulate_door" not in robot.get("capabilities", ()):
        return Problem("missing_capability",
                       f"door {door.guid} is {door.state} and robot '{robot.get('id')}' cannot open doors "
                       f"(no 'manipulate_door' capability)")
    return None


class DoorGraph:
    """The building as a graph: spaces are nodes, doors are edges."""

    def __init__(self, spaces, doors, adjacency):
        self.spaces = spaces            # guid -> Space
        self.doors = doors              # guid -> Door
        self.adjacency = adjacency      # space guid -> [edge dict]

    @classmethod
    def from_state(cls, state):
        spaces = {g: Space(g, s["name"], s.get("function", ""), tuple(s["centroid"]), s.get("storey"))
                  for g, s in state.spaces.items()}
        doors = {g: Door.from_state(g, d) for g, d in state.doors.items()}
        return cls(spaces, doors, state.adjacency)

    def label(self, space):
        return self.spaces[space].label if space in self.spaces else space

    def storey(self, space):
        return self.spaces[space].storey if space in self.spaces else None

    def same_storey(self, a, b):
        """"outside" belongs to every storey; None (no storeys in the IFC) is
        a storey like any other, so a building without storeys is one storey."""
        return OUTSIDE in (a, b) or self.storey(a) == self.storey(b)

    def neighbours(self, space):
        """[(door guid, other space)] in adjacency-list order."""
        return [(e["door"], e["to"]) for e in self.adjacency.get(space, [])]

    def _shortest(self, a, b, robot):
        """Dijkstra from a's centroid to b's centroid, moving door to door.
        Only doors the robot can pass count when `robot` is given; all doors
        when it is None. Returns (length_m, [(door, next space), ...]) or None."""
        if a == b:
            return 0.0, []
        if not self.same_storey(a, b):
            return None
        start, goal = self.spaces[a].centroid, self.spaces[b].centroid
        heap = [(0.0, 0, a, start, ())]           # cost, tie-break, space, position, path
        seen, counter = set(), 0
        while heap:
            cost, _, space, pos, path = heapq.heappop(heap)
            if space is None:                     # popped the goal
                return cost, list(path)
            key = path[-1] if path else ("start",)
            if key in seen:
                continue
            seen.add(key)
            if space == b:
                counter += 1
                heapq.heappush(heap, (cost + math.dist(pos, goal), counter, None, goal, path))
                continue
            for door_guid, other in self.neighbours(space):
                door = self.doors[door_guid]
                if robot is not None and door_problem(door, robot):
                    continue
                if (door_guid, other) in seen or not self.same_storey(a, other):
                    continue
                counter += 1
                heapq.heappush(heap, (cost + math.dist(pos, door.position), counter, other,
                                      door.position, path + ((door_guid, other),)))
        return None

    def route(self, a, b, robot):
        """Shortest passable route as [(next space, door guid), ...] (empty
        when a == b), or a PlanFailure that says which door blocks the way."""
        found = self._shortest(a, b, robot)
        if found is not None:
            return [(space, door) for door, space in found[1]]
        task = f"route({self.label(a)}, {self.label(b)})"
        if not self.same_storey(a, b):
            return PlanFailure(task, "other_storey",
                               f"{self.label(b)} is on storey {self.storey(b)}, the robot is on storey "
                               f"{self.storey(a)}; planning is single-storey (no stair or lift traversal)")
        anyway = self._shortest(a, b, None)
        if anyway is None:
            return PlanFailure(task, "unreachable_space",
                               f"no sequence of doors connects {self.label(a)} to {self.label(b)} in the building model")
        blocked = [(door, door_problem(self.doors[door], robot)) for door, _ in anyway[1]]
        blocked = [(door, p) for door, p in blocked if p]
        first = blocked[0][1]
        return PlanFailure(task, first.code,
                           f"{self.label(b)} cannot be reached from {self.label(a)}: {first.message}",
                           {"blocked_doors": [{"door": d, "code": p.code, "message": p.message} for d, p in blocked]})

    def route_length(self, a, b, robot):
        """Length in metres of the same path route() returns, or None."""
        found = self._shortest(a, b, robot)
        return None if found is None else found[0]
