"""The world state the planner reasons over."""

import copy
from dataclasses import asdict, dataclass, field


@dataclass
class WorldState:
    """HTN concept: world state (paper: robot position, capabilities, door
    states and a door-based adjacency list). Plain JSON-ready data only: no
    model clients, no file handles, so copy() is a cheap deep copy and
    to_dict()/from_dict() give exact logging and replay.

    robot      id, space (current), capabilities, width_m, safety_margin_m, allow_exterior
    spaces     guid -> {name, function, centroid}
    doors      guid -> {rooms, clear_width_m, position, approach, is_exterior, state}
    adjacency  space -> [{to, door, door_action, capabilities}]
    task       what was asked (domain-defined)
    belief     prior / posterior per space
    search     progress of the current task (visits, stages, results)
    settings   planner parameters (domain-defined)"""
    robot: dict = field(default_factory=dict)
    spaces: dict = field(default_factory=dict)
    doors: dict = field(default_factory=dict)
    adjacency: dict = field(default_factory=dict)
    task: dict = field(default_factory=dict)
    belief: dict = field(default_factory=dict)
    search: dict = field(default_factory=dict)
    settings: dict = field(default_factory=dict)

    def copy(self):
        return copy.deepcopy(self)

    def to_dict(self):
        return copy.deepcopy(asdict(self))

    @classmethod
    def from_dict(cls, data):
        return cls(**copy.deepcopy(data))
