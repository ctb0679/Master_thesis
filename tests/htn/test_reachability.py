"""Unreachable probability, single-storey planning and the connectivity
report. Nothing is inferred: rooms without a passable route keep their
probability and are reported with the reason."""

import pytest

from htn.core import CompoundTask, HTNPlanner
from htn.domain import (DoorGraph, build_domain, connectivity_report, format_connectivity, load_world,
                        unlinked_stairs)
from htn.domain.methods import reachability
from htn.execution import BeliefModel, NullLogger, PlanExecutor, SimulatedNavigation, StubBackend

from conftest import ROOT, a300, search_task, toy_world, write_building

FANTASY = ROOT / "data" / "cleaned_ifc" / "fantasy_office_building_1"

# A start; B, C behind ordinary doors; D only through an exterior door;
# F has no door; G is behind a door too narrow for the A300.
ROOMS = {"A": (0, 0), "B": (5, 0), "C": (10, 0), "D": (0, 5), "F": (20, 20), "G": (5, 5)}
DOORS = [("dAB", "A", "B", (2.5, 0), 0.9), ("dBC", "B", "C", (7.5, 0), 0.9),
         ("dAo", "A", "outside", (-2, 0), 0.9), ("dDo", "D", "outside", (0, 7), 0.9),
         ("dBG", "B", "G", (5, 2.5), 0.5)]


def search(state, prior):
    """find_object with a stub ranker returning `prior` and a detector that never finds."""
    search_task(state)
    ranker = lambda text: [{"space": g, "probability": p, "likelihood": p, "reasoning": ""} for g, p in prior.items()]
    ex = PlanExecutor(HTNPlanner(*build_domain()), state, BeliefModel(), StubBackend(), SimulatedNavigation(),
                      ranker, NullLogger(), miss_rate=0.2, log=lambda *a: None)
    return ex.run(CompoundTask("find_object", ("first aid kit", "single")))


def test_reachability_codes(tmp_path):
    state = toy_world(tmp_path, ROOMS, DOORS, "A")
    reachable, unreachable = reachability(state)
    assert sorted(reachable) == ["A", "B", "C"]
    assert {g: u["code"] for g, u in unreachable.items()} == {
        "D": "exterior_not_allowed", "F": "unreachable_space", "G": "door_too_narrow"}


def test_unreachable_mass_above_residual_gives_reachable_exhausted(tmp_path):
    state = toy_world(tmp_path, ROOMS, DOORS, "A")
    prior = {"A": 0.20, "B": 0.20, "C": 0.14, "D": 0.20, "F": 0.16, "G": 0.10}     # 0.46 unreachable
    summary = search(state, prior)
    assert summary.outcome == "not_found_reachable_exhausted"
    assert sorted(summary.visit_order) == ["A", "B", "C"]
    assert summary.probability["unreachable"]["prior"] == pytest.approx(0.46)
    assert summary.probability["reachable_searched"]["prior"] == pytest.approx(0.54)
    assert summary.probability["reachable_not_searched"]["prior"] == 0
    listed = [(r["space"], r["code"], r["prior"]) for r in summary.unreachable]
    assert listed == [("D", "exterior_not_allowed", 0.2), ("F", "unreachable_space", 0.16),
                      ("G", "door_too_narrow", 0.1)]
    assert "unreachable rooms hold" in summary.reason


def test_small_unreachable_mass_gives_not_found(tmp_path):
    """The rule uses the posterior: misses in A, B, C raise D's share
    (prior 0.05 -> 0.05 / (0.95 * 0.2 + 0.05) = 0.21 > residual_mass), so
    D's prior must be lower for "not_found" (0.01 -> 0.048)."""
    state = toy_world(tmp_path, ROOMS, DOORS, "A")
    summary = search(state, {"A": 0.4, "B": 0.3, "C": 0.29, "D": 0.01, "F": 0.0, "G": 0.0})
    assert summary.outcome == "not_found"
    assert summary.probability["unreachable"]["posterior"] < 0.1
    state = toy_world(tmp_path, ROOMS, DOORS, "A")
    summary = search(state, {"A": 0.4, "B": 0.3, "C": 0.25, "D": 0.05, "F": 0.0, "G": 0.0})
    assert summary.outcome == "not_found_reachable_exhausted"
    assert summary.probability["unreachable"]["posterior"] == pytest.approx(0.05 / (0.95 * 0.2 + 0.05), abs=1e-3)
    assert [r["space"] for r in summary.unreachable] == ["D", "F", "G"]          # still listed


def test_infeasible_outcome_names_the_code(tmp_path, planner):
    state = toy_world(tmp_path, ROOMS, DOORS, "A")
    from htn.domain import set_task
    set_task(state, "go_to", target="D")
    ex = PlanExecutor(planner, state, BeliefModel(), StubBackend(), SimulatedNavigation(), None, NullLogger(),
                      log=lambda *a: None)
    assert ex.run(CompoundTask("go_to", ("D",))).outcome == "infeasible: exterior_not_allowed"


# --- storeys ---------------------------------------------------------------------

STOREYS = {"A": "EG", "B": "EG", "C": "OG1"}
# C lies exactly above A; dBC is a (wrong) door record linking B on EG to C on OG1.
STOREY_ROOMS = {"A": (0, 0), "B": (5, 0), "C": (0, 0)}
STOREY_DOORS = [("dAB", "A", "B", (2.5, 0), 0.9), ("dBC", "B", "C", (2.5, 0), 0.9)]


def test_rooms_on_another_storey_are_unreachable_and_never_measured(tmp_path):
    state = toy_world(tmp_path, STOREY_ROOMS, STOREY_DOORS, "A", storeys=STOREYS, ifc_storeys=("EG", "OG1"))
    assert state.spaces["C"]["storey"] == "OG1" and state.doors["dAB"]["storey"] == "EG"
    graph = DoorGraph.from_state(state)
    assert graph.route("A", "C", state.robot).reason_code == "other_storey"
    assert graph.route_length("A", "C", state.robot) is None
    assert graph.route_length("A", "C", None) is None                 # not even ignoring the robot
    _, unreachable = reachability(state)
    assert unreachable["C"]["code"] == "other_storey"
    summary = search(state, {"A": 0.3, "B": 0.2, "C": 0.5})
    assert summary.outcome == "not_found_reachable_exhausted" and "C" not in summary.visit_order
    assert summary.unreachable[0]["storey"] == "OG1"


def test_missing_storeys_are_reported_not_guessed(tmp_path):
    llm, full = write_building(tmp_path, {"A": (0, 0), "B": (5, 0)}, [("dAB", "A", "B", (2.5, 0), 0.9)],
                               ifc_storeys=("EG", "OG1"))
    state, report = load_world(llm, full, a300())
    assert any("no storey" in line and "IfcBuildingStorey" in line for line in report)
    assert {s["storey"] for s in state.spaces.values()} == {None}


def test_building_without_storeys_is_one_storey(idac):
    state, _ = idac                                  # IDAC has no IfcBuildingStorey
    reachable, unreachable = reachability(state)
    assert len(reachable) == 15 and not unreachable


# --- connectivity report ----------------------------------------------------------

def test_connectivity_report(tmp_path):
    rooms = {"A": (0, 0), "B": (5, 0), "C": (10, 0), "D": (20, 0)}
    doors = [("dAB", "A", "B", (2.5, 0), 0.9), ("dCo", "C", "outside", (12, 0), 0.9)]
    llm, full = write_building(tmp_path, rooms, doors, stairs=[("stair1", "EG")], ifc_storeys=("EG",))
    state, _ = load_world(llm, full, a300(), None, "A")
    report = connectivity_report(state, unlinked_stairs(full, set(state.spaces)))
    assert [[r["space"] for r in g["rooms"]] for g in report["groups"]] == [["A", "B"], ["C"], ["D"]]
    assert [g["exterior_doors"] for g in report["groups"]] == [0, 1, 0]
    assert [r["space"] for r in report["reachable_from_start"]] == ["A", "B"]
    assert [s["guid"] for s in report["unlinked_stairs"]] == ["stair1"]
    assert [r["space"] for r in report["rooms_without_door"]] == ["D"]
    assert format_connectivity(report, str)[0].startswith("Connectivity: 3 group(s)")


def test_fantasy_office_building_1():
    """The German Revit export: no circulation spaces, 14 stairs not linked
    to any space; from room 1 the A300 reaches only rooms 1-4."""
    state, _ = load_world(f"{FANTASY}_llm.json", f"{FANTASY}_full.json", a300())
    keys = {s["name"]: g for g, s in state.spaces.items()}
    state.robot["space"] = keys["1"]
    report = connectivity_report(state, unlinked_stairs(f"{FANTASY}_full.json", set(state.spaces)))
    assert sorted(state.spaces[r["space"]]["name"] for r in report["reachable_from_start"]) == ["1", "2", "3", "4"]
    assert len(report["unlinked_stairs"]) == 14
    _, unreachable = reachability(state)
    assert unreachable[keys["8"]]["code"] == "other_storey"            # Meeting Room on OG1
    assert unreachable[keys["24"]]["code"] == "exterior_not_allowed"   # Lobby on EG, only via outside
