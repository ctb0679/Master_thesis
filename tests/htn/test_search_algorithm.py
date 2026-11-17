"""HTN path with settings search_algorithm (= --search_algorithm): only the
ordering of the ranked rooms changes; methods, candidate selection, belief
updates, replanning, detection and the recorded/restricted/go_to cases stay."""

import argparse
import types

import pytest

from conftest import IDAC_FULL, IDAC_LLM, idac_with, search_task, write_building
from htn.core import CompoundTask, HTNPlanner
from htn.domain import build_domain
from htn.domain.methods import distance_function, eligible_spaces, select_rooms, visit_order
from htn.execution import BeliefModel, NullLogger, PlanExecutor, SimulatedNavigation, StubBackend
from rank_candidates import load_building, select_candidates
from search_planners.blr import blr_visit_order
from search_planners.held_karp import optimal_visit_order

FIND = CompoundTask("find_object", ("first aid kit", "single"))


def prior(state, keys, high=("2", "4", "14", "7", "12")):
    p = {g: 0.02 for g in state.spaces}
    for k, v in zip(high, (0.35, 0.2, 0.15, 0.1, 0.08)):
        p[keys[k]] = v
    total = sum(p.values())
    return {g: v / total for g, v in p.items()}


def with_belief(state, keys, algorithm):
    state.settings["search_algorithm"] = algorithm
    state.belief["posterior"] = prior(state, keys)
    return state


def candidates(state):
    """The candidate list exactly as select_rooms builds it."""
    eligible, _ = eligible_spaces(state)
    post = state.belief["posterior"]
    mass = sum(post[g] for g in eligible)
    ranked = sorted(({"room": g, "probability": post[g] / mass} for g in eligible),
                    key=lambda r: (-r["probability"], r["room"]))
    return select_candidates(ranked, state.settings["prob_mass"], state.settings["min_prob"])


def test_areas_are_in_the_state():
    state, _ = idac_with()
    assert all(s["area_m2"] > 0 for s in state.spaces.values())


def test_default_setting_is_held_karp_and_unchanged():
    state, keys = idac_with()
    assert state.settings["search_algorithm"] == "held_karp"
    state.belief["posterior"] = prior(state, keys)
    cands = candidates(state)
    centroids = {g: s["centroid"] for g, s in state.spaces.items()}
    hk = optimal_visit_order(cands, centroids, state.robot["space"], distance=distance_function(state))
    assert select_rooms(state) == [c["room"] for c in hk]


@pytest.mark.parametrize("distance", ["door_graph", "euclidean"])
def test_blr_orders_the_same_candidates_with_the_same_distance(distance):
    state, keys = idac_with()
    state.settings["distance"] = distance
    with_belief(state, keys, "blr")
    cands = candidates(state)
    centroids = {g: s["centroid"] for g, s in state.spaces.items()}
    areas = {g: s["area_m2"] for g, s in state.spaces.items()}
    exp = blr_visit_order(cands, centroids, state.robot["space"], distance_function(state), areas)
    blr_rooms = select_rooms(state)
    assert blr_rooms == [c["room"] for c in exp]
    state.settings["search_algorithm"] = "held_karp"
    assert sorted(select_rooms(state)) == sorted(blr_rooms) == sorted(c["room"] for c in cands)


def test_recorded_rooms_keep_held_karp_under_blr(planner):
    state, keys = idac_with()
    recorded = [keys[k] for k in ("10", "2", "14")]
    search_task(state, recorded=recorded)
    with_belief(state, keys, "blr")
    result = planner.plan(state, [FIND])
    assert result.tree.children[0].method_name == "m_search_recorded"
    searched = [a.args["space"] for a in result.plan.actions if a.action == "capture_rgb"]
    hk = [c["room"] for c in visit_order(state, [{"room": r} for r in recorded], algorithm="held_karp")]
    assert searched == hk


def test_restricted_and_go_to_are_unchanged(planner):
    plans = {}
    for alg in ("held_karp", "blr"):
        state, keys = idac_with()
        search_task(state, restricted=keys["10"])
        with_belief(state, keys, alg)
        r = planner.plan(state, [FIND])
        state2, keys2 = idac_with()
        state2.settings["search_algorithm"] = alg
        state2.task = {"name": "go_to", "target": keys2["10"]}
        g = planner.plan(state2, [CompoundTask("go_to", (keys2["10"],))])
        plans[alg] = ([a.to_dict() for a in r.plan.actions], [a.to_dict() for a in g.plan.actions])
    assert plans["held_karp"] == plans["blr"]


def run(algorithm, found_in=()):
    state, keys = idac_with()
    search_task(state)
    state.settings["search_algorithm"] = algorithm
    p = prior(state, keys)
    calls, perception = [], StubBackend(found_in=[keys[k] for k in found_in])
    ex = PlanExecutor(HTNPlanner(*build_domain()), state, BeliefModel(), perception, SimulatedNavigation(),
                      lambda text: calls.append(text) or [{"space": g, "probability": v, "likelihood": v,
                                                           "reasoning": ""} for g, v in p.items()],
                      NullLogger(), miss_rate=0.2, log=lambda *a: None)
    return ex.run(FIND), ex, calls, p, keys


@pytest.mark.parametrize("alg", ["held_karp", "blr"])
def test_belief_updates_and_replanning_are_the_executors(alg):
    summary, ex, calls, p, _ = run(alg)
    assert len(calls) == 1                                           # ranking model asked once
    belief = BeliefModel()
    belief.set_prior(p)
    for room in summary.visit_order:                                 # the executor's own miss update
        belief.update_on_miss(room, 0.2)
    assert ex.belief.posterior() == pytest.approx(belief.posterior())
    assert summary.iterations == len(summary.visit_order) + 2        # prior fetch + one plan per room + stop
    assert summary.outcome in ("not_found", "not_found_reachable_exhausted")


def test_detection_is_unchanged():
    for alg in ("held_karp", "blr"):
        summary, *_ = run(alg, found_in=["2"])
        assert summary.outcome == "found" and len(summary.found) == 1


def test_blr_and_held_karp_differ_only_in_the_order():
    """Same rooms searched (target nowhere), same distance; different order.
    IDAC, robot in room 3: Held-Karp starts with nearby room 7, BLR with the
    most likely room 2 (p 0.35 vs 0.02 for room 7)."""
    hk, _, _, _, keys = run("held_karp")
    blr, *_ = run("blr")
    assert set(hk.visit_order) == set(blr.visit_order)
    assert hk.visit_order != blr.visit_order
    assert hk.visit_order[0] == keys["7"] and blr.visit_order[0] == keys["2"]
    assert hk.distance == blr.distance
    assert hk.walking_m < blr.walking_m             # Held-Karp minimises the walk


# --- CLI glue ---------------------------------------------------------------------

def htn_args(algorithm, llm, full):
    from htn.cli import add_htn_args
    p = argparse.ArgumentParser()
    add_htn_args(p)
    a = p.parse_args(["--miss_rate", "0.2", "--distance", "door_graph"])
    a.prob_mass, a.min_prob, a.search_algorithm = 0.9, 0.01, algorithm
    a.cleaned_json, a.full_json = str(llm), str(full)
    return a


def fake_session(llm):
    b = load_building(llm)
    return types.SimpleNamespace(building=b, start_room=b["spaces"][0]["name"], detector=None)


def test_cli_puts_the_algorithm_in_the_settings():
    from htn.cli import HTNSearch
    for alg in ("held_karp", "blr"):
        h = HTNSearch.from_args(htn_args(alg, IDAC_LLM, IDAC_FULL), fake_session(IDAC_LLM))
        assert h.world.settings["search_algorithm"] == alg


def test_cli_blr_without_areas_is_refused_held_karp_is_not(tmp_path):
    from htn.cli import HTNSearch
    llm, full = write_building(tmp_path, {"A": (0, 0), "B": (4, 0)}, [("d1", "A", "B", (2, 0), 1.0)])
    HTNSearch.from_args(htn_args("held_karp", llm, full), fake_session(llm))
    with pytest.raises(RuntimeError, match="needs a positive floor area"):
        HTNSearch.from_args(htn_args("blr", llm, full), fake_session(llm))
