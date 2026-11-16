"""settings distance = "navbim": the visit order uses navbim's walking-distance
table (htn/domain/walking.py), with the door graph as fallback for routes
through doors the robot cannot pass."""

import json

from conftest import IDAC_LLM, idac_with
from htn.domain import DoorGraph
from htn.domain.methods import distance_function, select_rooms, visit_order
from htn.domain.walking import ROOT, default_table_path, load_table, table_problem

TABLE = ROOT / "data" / "distances" / "IDAC_Model_preplanned_centre_walking_distances.json"


def navbim_state(**kw):
    state, keys = idac_with(**kw)
    state.settings.update(distance="navbim", distance_table=str(TABLE))
    return state, keys


def table_metres(a, b):
    t = load_table(str(TABLE))
    return t["matrix"][t["index"][a]][t["index"][b]]


def test_default_table_path_is_the_centre_table():
    assert default_table_path(IDAC_LLM) == TABLE
    assert TABLE.exists()


def test_table_fits_idac():
    state, _ = idac_with()
    assert table_problem(TABLE, state.spaces) is None


def test_table_problem_names_missing_file_and_rooms(tmp_path):
    state, _ = idac_with()
    assert "no walking-distance table" in table_problem(tmp_path / "none.json", state.spaces)
    data = json.loads(TABLE.read_text(encoding="utf-8"))
    data["rooms"] = data["rooms"][:3]
    small = tmp_path / "small.json"
    small.write_text(json.dumps(data))
    assert "lacks 12 room(s)" in table_problem(small, state.spaces)


def test_navbim_distance_is_the_table_value_when_all_doors_open():
    state, keys = navbim_state()
    distance = distance_function(state)
    for a in state.spaces:
        for b in state.spaces:
            assert distance(a, b) == (0.0 if a == b else table_metres(a, b))
    assert not distance.fallbacks


def test_closed_door_on_navbim_route_falls_back_to_door_graph(tmp_path):
    state, _ = navbim_state()
    rooms = list(state.spaces)
    a, b = rooms[0], rooms[-1]
    route = load_table(str(TABLE))["routes"][f"{a}->{b}"]
    closed_state, _ = navbim_state(tmp_path=tmp_path, closed=[route["door_guids"][0]])
    distance = distance_function(closed_state)
    expected = DoorGraph.from_state(closed_state).route_length(a, b, closed_state.robot)
    assert distance(a, b) == expected                 # door graph (None = unreachable)
    assert (a, b) in distance.fallbacks


def test_visit_order_minimises_navbim_walking():
    """Held-Karp with the navbim setting gives the order with the shortest
    total walk in the table (checked against all orders)."""
    from itertools import permutations
    state, keys = navbim_state(start="3")
    rooms = [g for g in state.spaces if g != state.robot["space"]][:5]
    order = [c["room"] for c in visit_order(state, [{"room": r} for r in rooms])]

    def walk(seq):
        seq = [state.robot["space"], *seq]
        return sum(table_metres(x, y) for x, y in zip(seq, seq[1:]))
    assert abs(walk(order) - min(walk(p) for p in permutations(rooms))) < 1e-9


def test_select_rooms_works_with_navbim(planner):
    state, keys = navbim_state()
    state.belief["posterior"] = {g: 1 / len(state.spaces) for g in state.spaces}
    state.settings.update(prob_mass=0.5, min_prob=0.0)
    chosen = select_rooms(state)
    assert chosen and len(set(chosen)) == len(chosen)


def test_executor_reports_navbim_walking_metres():
    """A full simulated search (no target anywhere): the summary's walking
    metres are the table distances along the rooms actually searched, and
    every plan line says which distance was used."""
    from htn.core import CompoundTask, HTNPlanner
    from htn.domain import build_domain
    from htn.execution import BeliefModel, NullLogger, PlanExecutor, SimulatedNavigation, StubBackend
    from conftest import search_task

    state, keys = navbim_state()
    search_task(state)
    start = state.robot["space"]
    prior = {g: 1 / len(keys) for g in keys.values()}
    lines = []
    ex = PlanExecutor(HTNPlanner(*build_domain()), state, BeliefModel(), StubBackend(), SimulatedNavigation(),
                      lambda text: [{"space": g, "probability": p, "likelihood": p, "reasoning": ""}
                                    for g, p in prior.items()],
                      NullLogger(), miss_rate=0.2, log=lines.append)
    summary = ex.run(CompoundTask("find_object", ("first aid kit", "single")))
    seq = [start, *summary.visit_order]
    assert summary.distance == "navbim"
    # searching the room the robot is already in is no walk, so it has no leg
    assert [leg["to"] for leg in summary.walking_legs] == [b for a, b in zip(seq, seq[1:]) if a != b]
    assert abs(summary.walking_m - sum(table_metres(a, b) for a, b in zip(seq, seq[1:]))) < 0.05
    assert any("navbim distances" in line for line in lines)
