"""--no-htn path (interactive_search.SearchSession) with --search_algorithm.
No Ollama: the ranking call, the interpreter and the image prompt are stubbed."""

from pathlib import Path

import pytest

import interactive_search
from interactive_search import SearchSession
from rank_candidates import select_candidates, softmax_probabilities, validate_rankings
from search_planners import order_rooms
from search_planners.held_karp import optimal_visit_order

ROOT = Path(__file__).resolve().parents[2]
IDAC_LLM = ROOT / "data" / "cleaned_ifc" / "IDAC_Model_llm.json"


def session(algorithm=None):
    kw = {} if algorithm is None else {"search_algorithm": algorithm}
    return SearchSession(str(IDAC_LLM), None, 0.9, 0.01, 0.1, "stub-model", "http://stub", {}, **kw)


@pytest.fixture
def fake_ranking(monkeypatch):
    """The model's scores: a few rooms likely, the rest unlikely."""
    def call_ollama(host, model, task, summaries, note=None, options=None):
        s = session()
        rooms = [sp["name"] for sp in s.building["spaces"]]
        high = {rooms[2]: 0.9, rooms[7]: 0.85, rooms[11]: 0.8, rooms[5]: 0.75, rooms[13]: 0.72}
        return [{"room": r, "likelihood": high.get(r, 0.1), "reasoning": f"stub {r}"} for r in rooms]
    monkeypatch.setattr(interactive_search, "call_ollama", call_ollama)
    return call_ollama


def expected(s, call_ollama, algorithm):
    rankings = softmax_probabilities(validate_rankings(call_ollama(0, 0, "", None), s.building), s.softmax_temp)
    cands = select_candidates(rankings, s.prob_mass, s.min_prob)
    return rankings, cands, order_rooms(cands, s.centroid_lookup, s.start_room, algorithm, areas=s.areas)


def test_default_is_held_karp_and_unchanged(fake_ranking):
    s = session()
    assert s.search_algorithm == "held_karp"
    rankings, cands, _ = expected(s, fake_ranking, "held_karp")
    got_rankings, plan = s.rank_and_plan("find the printer")
    assert got_rankings == rankings
    assert plan == optimal_visit_order(cands, s.centroid_lookup, s.start_room)   # the pre-refactor call
    assert session("held_karp").rank_and_plan("find the printer")[1] == plan


def test_blr_orders_the_same_candidates(fake_ranking):
    hk = session("held_karp").rank_and_plan("find the printer")[1]
    s = session("blr")
    _, cands, exp = expected(s, fake_ranking, "blr")
    rankings, plan = s.rank_and_plan("find the printer")
    assert plan == exp
    assert sorted(c["room"] for c in plan) == sorted(c["room"] for c in hk) == sorted(c["room"] for c in cands)
    assert all(c["blr_score"] is None or c["blr_score"] > 0 for c in plan)


def test_areas_come_from_the_building(fake_ranking):
    s = session("blr")
    assert len(s.areas) == 15 and all(a > 0 for a in s.areas.values())


def test_known_rooms_follow_the_algorithm_with_equal_probabilities():
    """Rooms recorded in the IFC: Held-Karp exactly as before; BLR with every
    room equally likely (probability 1.0 each), i.e. by 1 / (d * sqrt(A))."""
    from search_planners.blr import blr_visit_order
    for alg in ("held_karp", "blr"):
        s = session(alg)
        rooms = [sp["name"] for sp in s.building["spaces"]][3:7]
        entries = [{"room": r, "likelihood": 1.0, "probability": 1.0, "reasoning": "recorded"} for r in rooms]
        expected = (optimal_visit_order(entries, s.centroid_lookup, s.start_room) if alg == "held_karp"
                    else blr_visit_order(entries, s.centroid_lookup, s.start_room, None, s.areas))
        assert s.plan_known_rooms(rooms, "recorded") == expected
        one = s.plan_known_rooms([rooms[0]], "named room / go_to")              # a single room: no choice
        assert [p["room"] for p in one] == [rooms[0]]


def test_no_htn_navbim_distance_uses_the_table(fake_ranking):
    from htn.domain.walking import default_table_path, load_table, table_distance
    from search_planners.common import euclidean
    s = session("held_karp")
    table = default_table_path(IDAC_LLM)
    guid_of = {sp["name"]: sp["guid"] for sp in s.building["spaces"]}
    s.distance = table_distance(table, guid_of, lambda a, b: euclidean(s.centroid_lookup[a], s.centroid_lookup[b]))
    t = load_table(str(table))
    _, plan = s.rank_and_plan("find the printer")
    prev = s.start_room
    for stop in plan:
        metres = t["matrix"][t["index"][guid_of[prev]]][t["index"][guid_of[stop["room"]]]]
        assert stop["distance_from_previous_m"] == round(metres, 2)
        prev = stop["room"]
    assert not s.distance.fallbacks
    _, cands, _ = expected(s, fake_ranking, "held_karp")
    assert plan == optimal_visit_order(cands, s.centroid_lookup, s.start_room, distance=s.distance)


def test_distance_defaults_per_path():
    from htn.cli import add_htn_args, check_htn_args
    import argparse

    def parse(*argv):
        p = argparse.ArgumentParser()
        add_htn_args(p)
        a = p.parse_args(list(argv))
        check_htn_args(p, a)
        return a.distance
    assert parse("--no-htn") == "euclidean"                       # unchanged baseline
    assert parse("--no-htn", "--distance", "navbim") == "navbim"
    assert parse("--miss_rate", "0.2") == "navbim"                # HTN default as before
    with pytest.raises(SystemExit):
        parse("--no-htn", "--distance", "door_graph")


def run_task_with(monkeypatch, s, interp, images=()):
    monkeypatch.setattr(s, "interpret", lambda text: interp)
    answers = iter(images)
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers, "skip"))
    return s.run_task("x")


def interp(kind, **kw):
    base = {"task_type": kind, "target": "", "answer": "", "reason": "", "resolved_room": None,
            "rooms_with_target": [], "quantity": "single"}
    return {**base, **kw}


@pytest.mark.parametrize("alg", ["held_karp", "blr"])
def test_recorded_rooms_are_visited_first(monkeypatch, fake_ranking, capsys, alg):
    s = session(alg)
    rooms = [sp["name"] for sp in s.building["spaces"]]
    run_task_with(monkeypatch, s, interp("find_object", target="printer", rooms_with_target=[rooms[4]]))
    out = capsys.readouterr().out
    assert out.index("Visit plan (from building model)") < out.index("Querying VLM for room probabilities")


@pytest.mark.parametrize("alg", ["held_karp", "blr"])
def test_other_task_kinds_are_unchanged(monkeypatch, capsys, alg):
    s = session(alg)
    room = s.building["spaces"][3]["name"]
    assert run_task_with(monkeypatch, s, interp("none", answer="hello")) is None
    assert run_task_with(monkeypatch, s, interp("building_query", answer="there are 15 rooms")) is None
    run_task_with(monkeypatch, s, interp("go_to_room", target=room, resolved_room=room))
    run_task_with(monkeypatch, s, interp("find_object", target="cup", resolved_room=room))
    out = capsys.readouterr().out
    assert "hello" in out and "there are 15 rooms" in out
    assert "Navigation plan" in out and "Visit plan (room specified by user)" in out


def test_cli_flag_default_and_choices():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--search_algorithm", choices=interactive_search.ALGORITHMS,
                        default=interactive_search.DEFAULT_ALGORITHM)
    assert parser.parse_args([]).search_algorithm == "held_karp"
    src = Path(interactive_search.__file__).read_text(encoding="utf-8")
    assert '"--search_algorithm", choices=ALGORITHMS, default=DEFAULT_ALGORITHM' in src
