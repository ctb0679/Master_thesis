"""Belief update, the replanning executor, and the distance change in
rank_candidates.optimal_visit_order."""

import math
import random

import pytest

from htn.core import CompoundTask
from htn.execution import BeliefModel, NullLogger, PlanExecutor, SimulatedNavigation, StubBackend
from rank_candidates import euclidean, optimal_visit_order

from conftest import search_task

FIND = CompoundTask("find_object", ("first aid kit", "single"))


# --- belief update -----------------------------------------------------------------

def test_miss_lowers_the_room_by_the_miss_rate_and_sums_to_one():
    b = BeliefModel()
    b.set_prior({"a": 0.5, "b": 0.3, "c": 0.2})
    b.update_on_miss("a", 0.2)
    post = b.posterior()
    assert math.isclose(sum(post.values()), 1.0)
    # unnormalised: a 0.1, b 0.3, c 0.2 -> / 0.6
    assert post["a"] == pytest.approx(0.1 / 0.6)
    assert post["b"] == pytest.approx(0.5) and post["c"] == pytest.approx(0.2 / 0.6)
    assert post["a"] / post["b"] == pytest.approx(0.2 * 0.5 / 0.3)   # ratio to others drops by the miss rate
    assert post["a"] > 0                                            # never set to 0


def test_stop_rule():
    b = BeliefModel()
    b.set_prior({"a": 0.95, "b": 0.05})
    assert b.should_stop([], 0.1)[0]
    assert b.should_stop(["b"], 0.1) == (True, "remaining probability 0.050 < residual_mass 0.100")
    assert b.should_stop(["a", "b"], 0.1) == (False, "")


# --- executor --------------------------------------------------------------------------

def prior_for(keys, high=("2", "4", "14")):
    p = {g: 0.02 for g in keys.values()}
    for k, v in zip(high, (0.5, 0.2, 0.15)):
        p[keys[k]] = v
    return p


def run(state, keys, found_in=(), miss_rate=0.2, max_visits=1, residual=None, skip=(), **task):
    state.settings["max_visits_per_room"] = max_visits
    search_task(state, **task)
    calls = []

    def ranker(text):
        calls.append(text)
        return [{"space": g, "probability": p, "likelihood": p, "reasoning": f"stub reason for {g}"}
                for g, p in prior_for(keys).items()]

    perception = StubBackend(found_in=found_in, skip=skip)
    from htn.core import HTNPlanner
    from htn.domain import build_domain
    ex = PlanExecutor(HTNPlanner(*build_domain()), state, BeliefModel(), perception, SimulatedNavigation(),
                      ranker, NullLogger(), miss_rate=miss_rate, residual_mass=residual, log=lambda *a: None)
    return ex.run(FIND), calls, ex


def test_finds_the_target_and_stops(idac):
    state, keys = idac
    summary, calls, _ = run(state, keys, found_in=[keys["14"]])
    assert summary.outcome == "found" and summary.found == [keys["14"]]
    assert summary.visit_order[-1] == keys["14"]
    assert len(calls) == 1                                     # ranking model asked once per task


def test_rankings_keep_the_model_reasoning(idac):
    state, keys = idac
    summary, _, _ = run(state, keys, found_in=[keys["14"]])
    assert summary.rankings[0]["space"] == keys["2"]           # sorted by probability
    assert all(r["reasoning"] == f"stub reason for {r['space']}" for r in summary.rankings)
    assert len(summary.rankings) == len(keys)


def test_restricted_search_never_calls_the_ranker(idac):
    state, keys = idac
    summary, calls, _ = run(state, keys, restricted=keys["2"])
    assert summary.outcome == "not_found" and summary.visit_order == [keys["2"]]
    assert calls == []


def test_recorded_rooms_then_semantic_with_one_ranker_call(idac):
    state, keys = idac
    summary, calls, ex = run(state, keys, found_in=[keys["4"]], recorded=[keys["15"]])
    assert summary.visit_order[0] == keys["15"] and summary.outcome == "found"
    assert len(calls) == 1
    assert keys["15"] not in summary.visit_order[1:]
    assert ex.belief.posterior()[keys["15"]] < ex.belief.prior[keys["15"]]   # earlier miss applied


def test_skipped_image_leaves_the_belief_unchanged(idac):
    state, keys = idac
    summary, _, ex = run(state, keys, skip=list(keys.values()), residual=0.0)
    assert ex.belief.posterior() == pytest.approx(ex.belief.prior)
    assert all(ex.state.search["visits"][s] == 1 for s in summary.visit_order)


def test_miss_rate_only_changes_the_stop_rule_when_each_room_is_visited_once(idac):
    """max_visits_per_room = 1: rooms already searched are no longer
    eligible and selection uses the renormalised posterior of the eligible
    rooms, so the miss update cannot change the visit order."""
    import copy
    state, keys = idac
    without, _, _ = run(copy.deepcopy(state), keys, miss_rate=1.0, residual=0.0)     # 1.0 = no update
    with_update, _, _ = run(copy.deepcopy(state), keys, miss_rate=0.2, residual=0.0)
    assert without.visit_order == with_update.visit_order
    stop_early, _, _ = run(copy.deepcopy(state), keys, miss_rate=0.2)                # default residual
    assert stop_early.visit_order == with_update.visit_order[:len(stop_early.visit_order)]


def test_miss_rate_changes_the_visit_order_when_rooms_can_be_revisited(idac):
    import copy
    state, keys = idac
    without, _, _ = run(copy.deepcopy(state), keys, miss_rate=1.0, max_visits=2, residual=0.0)
    with_update, _, _ = run(copy.deepcopy(state), keys, miss_rate=0.2, max_visits=2, residual=0.0)
    assert without.visit_order != with_update.visit_order


def test_find_without_miss_rate_is_refused(idac):
    state, keys = idac
    with pytest.raises(RuntimeError, match="miss_rate"):
        run(state, keys, miss_rate=None)


def test_go_to(planner, idac):
    state, keys = idac
    from htn.domain import set_task
    set_task(state, "go_to", target=keys["11"])
    ex = PlanExecutor(planner, state, BeliefModel(), StubBackend(), SimulatedNavigation(), None, NullLogger(),
                      log=lambda *a: None)
    summary = ex.run(CompoundTask("go_to", (keys["11"],)))
    assert summary.outcome == "reached" and ex.state.robot["space"] == keys["11"]


# --- optimal_visit_order: distance parameter ---------------------------------------------

def old_optimal_visit_order(candidates, centroid_lookup, start_room, max_exact=18):
    """rank_candidates.optimal_visit_order as of commit 11b9dde, copied verbatim
    (the greedy fallback is not needed for <= 18 rooms)."""
    n = len(candidates)
    if n == 0:
        return []
    start_xy = centroid_lookup.get(start_room)
    xy = [centroid_lookup[c["room"]] for c in candidates]
    dist_from_start = [euclidean(start_xy, p) for p in xy]
    dist = [[euclidean(xy[i], xy[j]) for j in range(n)] for i in range(n)]
    FULL = (1 << n) - 1
    INF = float("inf")
    dp = [[INF] * n for _ in range(1 << n)]
    parent = [[-1] * n for _ in range(1 << n)]
    for i in range(n):
        dp[1 << i][i] = dist_from_start[i]
    for mask in range(1 << n):
        for i in range(n):
            if dp[mask][i] == INF or not (mask & (1 << i)):
                continue
            for j in range(n):
                if mask & (1 << j):
                    continue
                new_mask = mask | (1 << j)
                cost = dp[mask][i] + dist[i][j]
                if cost < dp[new_mask][j]:
                    dp[new_mask][j] = cost
                    parent[new_mask][j] = i
    best_end = min(range(n), key=lambda i: dp[FULL][i])
    order = []
    mask, i = FULL, best_end
    while i != -1:
        order.append(i)
        prev = parent[mask][i]
        mask ^= (1 << i)
        i = prev
    order.reverse()
    ordered = []
    prev_xy = start_xy
    for idx in order:
        c = candidates[idx]
        d = euclidean(prev_xy, xy[idx])
        ordered.append(dict(c, distance_from_previous_m=round(d, 2)))
        prev_xy = xy[idx]
    return ordered


def test_euclidean_default_reproduces_the_old_function_exactly(idac):
    state, _ = idac
    lookup = {g: s["centroid"] for g, s in state.spaces.items()}
    rooms = sorted(lookup)
    rng = random.Random(0)
    for _ in range(50):
        chosen = rng.sample(rooms, rng.randint(1, 9))
        cands = [{"room": r, "probability": rng.random()} for r in chosen]
        start = rng.choice(rooms)
        expected = old_optimal_visit_order(cands, lookup, start)
        assert optimal_visit_order(cands, lookup, start) == expected
        assert optimal_visit_order(cands, lookup, start,
                                   distance=lambda a, b: euclidean(lookup[a], lookup[b])) == expected
