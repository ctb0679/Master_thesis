"""Object-search domain: conditions, effects and the six primitive operators
(navigate, manipulate_door, capture_rgb, detect, update_belief, record_result).

Per-space search progress lives in state.search["stage"][space]:
None -> "image" (capture_rgb) -> "observed" (detect) -> "counted"
(update_belief) -> None again (record_result)."""

from htn.core import Condition, Effect, Operator

from .building import DoorGraph, edge


def _label(state, space):
    s = state.spaces.get(space)
    return f"{s['name']} ({s['function']})" if s and s.get("function") else (s["name"] if s else space)


# --- conditions ---------------------------------------------------------------

class RobotAt(Condition):
    def check(self, state):
        return state.robot.get("space") == self.args[0]

    def explain(self, state):
        return (f"robot is in {_label(state, state.robot.get('space'))}, "
                f"not in {_label(state, self.args[0])}")


class DoorConnects(Condition):
    def check(self, state):
        door, a, b = self.args
        return door in state.doors and set(state.doors[door]["rooms"]) == {a, b}


class DoorOpen(Condition):
    def check(self, state):
        return state.doors[self.args[0]]["state"] == "open"


class DoorClosed(Condition):
    def check(self, state):
        return state.doors[self.args[0]]["state"] != "open"


class DoorFits(Condition):
    reason_code = "door_too_narrow"

    def check(self, state):
        d = state.doors[self.args[0]]
        return d["clear_width_m"] is not None and \
            d["clear_width_m"] >= state.robot["width_m"] + state.robot["safety_margin_m"]

    def explain(self, state):
        d, r = state.doors[self.args[0]], state.robot
        return (f"door {self.args[0]} is {d['clear_width_m']} m wide; robot needs "
                f"{r['width_m'] + r['safety_margin_m']:.3f} m")


class ExteriorAllowed(Condition):
    reason_code = "exterior_not_allowed"

    def check(self, state):
        return not state.doors[self.args[0]]["is_exterior"] or state.robot.get("allow_exterior", False)


class RobotBesideDoor(Condition):
    def check(self, state):
        return state.robot.get("space") in state.doors[self.args[0]]["rooms"]


class HasCapability(Condition):
    reason_code = "missing_capability"

    def check(self, state):
        return self.args[0] in state.robot.get("capabilities", ())

    def explain(self, state):
        return f"robot '{state.robot.get('id')}' lacks capability '{self.args[0]}'"


class SpaceReachable(Condition):
    """A passable route from the robot's space exists. On failure the code
    is the route's own (unreachable_space, door_too_narrow, ...)."""

    def _route(self, state):
        return DoorGraph.from_state(state).route(state.robot["space"], self.args[0], state.robot)

    def check(self, state):
        return isinstance(self._route(state), list)

    def explain(self, state):
        return self._route(state).message

    def code(self, state):
        return self._route(state).reason_code

    def details(self, state):
        return self._route(state).details


class _StageIs(Condition):
    stage = None

    def check(self, state):
        return state.search["stage"].get(self.args[0]) == self.stage


class ImageCaptured(_StageIs):
    stage = "image"


class ObservationPending(_StageIs):
    stage = "observed"


class VisitCounted(_StageIs):
    stage = "counted"


# --- effects ------------------------------------------------------------------

class RobotMovesTo(Effect):
    def apply(self, state):
        state.robot["space"] = self.args[0]


class SetDoorState(Effect):
    """Changes the door and its two adjacency-list edges (door_action and
    capabilities follow the door state)."""

    def apply(self, state):
        door, value = self.args
        state.doors[door]["state"] = value
        for space, edges in state.adjacency.items():
            for i, e in enumerate(edges):
                if e["door"] == door:
                    edges[i] = edge(door, e["to"], value)


class SetStage(Effect):
    def apply(self, state):
        space, stage = self.args
        state.search["stage"][space] = stage


class CountVisit(Effect):
    def apply(self, state):
        space = self.args[0]
        state.search["visits"][space] = state.search["visits"].get(space, 0) + 1


class RecordResult(Effect):
    def apply(self, state):
        state.search["stage"][self.args[0]] = None
        state.search["searched"].append(self.args[0])


# --- operators ----------------------------------------------------------------

def operators():
    """The six primitive operators, in table order."""
    return [
        Operator(
            "navigate",
            ("from_space", "to_space", "door", "door_position", "approach_from", "approach_to", "to_centroid"),
            [RobotAt("?from_space"), DoorConnects("?door", "?from_space", "?to_space"), DoorOpen("?door"),
             DoorFits("?door"), ExteriorAllowed("?door")],
            [RobotMovesTo("?to_space")],
            {"navigate"},
            description="drive from one space through a door into the next space"),
        Operator(
            "manipulate_door",
            ("door", "action", "door_position", "approach_point"),
            [RobotBesideDoor("?door"), DoorClosed("?door")],
            [SetDoorState("?door", "open")],
            {"manipulate_door"},
            description="open a closed door from the side the robot is on"),
        Operator(
            "capture_rgb", ("space",),
            [RobotAt("?space")],
            [SetStage("?space", "image")],
            {"rgb"},
            description="take an RGB image in the current space"),
        Operator(
            "detect", ("space", "target"),
            [ImageCaptured("?space")],
            [SetStage("?space", "observed")],
            deterministic=False,
            description="ask the detector whether the target is in the image; "
                        "outcome known only at run time, planned as 'not found'"),
        Operator(
            "update_belief", ("space",),
            [ObservationPending("?space")],
            [CountVisit("?space"), SetStage("?space", "counted")],
            description="count the visit; the executor applies the Bayesian update with the real observation"),
        Operator(
            "record_result", ("space", "target"),
            [VisitCounted("?space")],
            [RecordResult("?space")],
            description="store the observation for this space"),
    ]
