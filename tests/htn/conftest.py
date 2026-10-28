"""Shared fixtures: the IDAC building, the A300 profile, and a builder for
small hand-made buildings. No Ollama: rankers and detectors are stubbed."""

import json
from pathlib import Path

import pytest

from htn.core import HTNPlanner
from htn.domain import RobotProfile, build_domain, load_world, set_task

ROOT = Path(__file__).resolve().parents[2]
IDAC_LLM = ROOT / "data" / "cleaned_ifc" / "IDAC_Model_llm.json"
IDAC_FULL = ROOT / "data" / "cleaned_ifc" / "IDAC_Model_full.json"


def a300(**changes):
    """The A300 as derived from its URDF (data/robot/profiles/a300.json)."""
    fields = dict(id="a300", width_m=0.665, capabilities=["navigate", "rgb", "depth"])
    fields.update(changes)
    return RobotProfile(**fields)


@pytest.fixture
def planner():
    return HTNPlanner(*build_domain())


@pytest.fixture
def idac():
    """-> (state, {room key: guid}) with the A300 in room 3 (Workspace)."""
    state, _ = load_world(IDAC_LLM, IDAC_FULL, a300())
    keys = {s["name"]: g for g, s in state.spaces.items()}
    state.robot["space"] = keys["3"]
    return state, keys


def idac_with(robot=None, door_states=None, start="3", tmp_path=None, closed=()):
    """IDAC with a different robot and/or closed doors."""
    path = None
    if closed:
        path = tmp_path / "doors.json"
        path.write_text(json.dumps({"default": "open", "overrides": {d: "closed" for d in closed}}))
    state, _ = load_world(IDAC_LLM, IDAC_FULL, robot or a300(), path)
    keys = {s["name"]: g for g, s in state.spaces.items()}
    state.robot["space"] = keys[start]
    return state, keys


def write_building(tmp_path, rooms, doors):
    """A tiny building as _llm.json + _full.json.
    rooms: {name: (cx, cy)}; the GUID is the name.
    doors: [(guid, room_a, room_b, (x, y), width)]; room_b may be "outside"."""
    llm = {"rooms": [{"guid": n, "name": n, "function": "", "centroid": list(c)} for n, c in rooms.items()],
           "adjacency": [{"a": a, "b": b, "via": g, "via_class": "IfcDoor"} for g, a, b, _, _ in doors]}
    full = {"units": {"output_units": {"length": "m"}},
            "spaces": [{"guid": n, "geometry": {}} for n in rooms],
            "elements": [{"guid": g, "ifc_class": "IfcDoor", "geometry": {"centroid": [x, y, 1.0]},
                          "door": {"clear_width_m": w, "rooms": [a, b], "is_external": b == "outside"}}
                         for g, a, b, (x, y), w in doors]}
    (tmp_path / "toy_llm.json").write_text(json.dumps(llm))
    (tmp_path / "toy_full.json").write_text(json.dumps(full))
    return tmp_path / "toy_llm.json", tmp_path / "toy_full.json"


def toy_world(tmp_path, rooms, doors, start, robot=None, settings=None):
    llm, full = write_building(tmp_path, rooms, doors)
    state, _ = load_world(llm, full, robot or a300(), None, start, settings)
    return state


def search_task(state, restricted=None, recorded=(), mode="single", target="first aid kit"):
    set_task(state, "find_object", target, mode, restricted, recorded, f"find the {target}")
    return state
