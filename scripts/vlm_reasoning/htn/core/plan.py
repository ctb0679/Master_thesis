"""The planner's output: an ordered list of primitive actions."""

import json
from dataclasses import dataclass, field


@dataclass
class PrimitiveAction:
    """One robot-executable step, {"action", "robot", "args"} as in the paper,
    so each can later be mapped to one behaviour-tree subtree."""
    action: str
    robot: str
    args: dict

    def to_dict(self):
        return {"action": self.action, "robot": self.robot, "args": _plain(self.args)}

    def matches(self, action, **args):
        return self.action == action and all(self.args.get(k) == v for k, v in args.items())


@dataclass
class Plan:
    """A totally ordered plan plus the initial robot and door states, so a run
    can be reproduced from the JSON alone."""
    actions: list
    initial_robot: dict = field(default_factory=dict)
    initial_doors: dict = field(default_factory=dict)

    def to_dict(self):
        return {"initial_robot": _plain(self.initial_robot),
                "initial_doors": _plain(self.initial_doors),
                "actions": [a.to_dict() for a in self.actions]}

    def to_json(self, indent=2):
        return json.dumps(self.to_dict(), indent=indent)

    def index_of(self, action, **args):
        """Index of the first action matching the name and args, or None."""
        return next((i for i, a in enumerate(self.actions) if a.matches(action, **args)), None)

    def is_before(self, first, second):
        """Ordering check: does the first action matching `first` come before
        the first one matching `second`? Each is (action, {arg: value})."""
        i, j = self.index_of(first[0], **first[1]), self.index_of(second[0], **second[1])
        return i is not None and j is not None and i < j


def _plain(value):
    """Tuples -> lists, recursively, so the JSON is stable."""
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value
