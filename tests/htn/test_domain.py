"""The object-search domain on IDAC and on hand-made buildings, mapped to the
paper's validation criteria (Tandon et al., Sec. III): completeness,
correctness, ordering, groundedness, infeasibility; plus determinism."""

import math

from htn.core import CompoundTask
from htn.domain import DoorGraph
from htn.domain.methods import select_rooms

from conftest import a300, idac_with, search_task, toy_world

FIND = CompoundTask("find_object", ("first aid kit", "single"))
PRIOR_ROOMS = {"2": 0.4, "4": 0.2, "14": 0.15, "15": 0.1}      # rest share 0.15


def with_prior(state, keys):
    rest = (1 - sum(PRIOR_ROOMS.values())) / (len(keys) - len(PRIOR_ROOMS))
    post = {g: PRIOR_ROOMS.get(k, rest) for k, g in keys.items()}
    state.belief = {"prior": dict(post), "posterior": post}
    return state


def semantic_plan(planner, idac):
    state, keys = idac
    return planner.plan(with_prior(search_task(state), keys), [FIND]), state, keys


# --- determinism ----------------------------------------------------------------

def test_same_plan_json_in_10_runs(planner, idac):
    state, keys = idac
    with_prior(search_task(state), keys)
    plans = {planner.plan(state.copy(), [FIND]).plan.to_json() for _ in range(10)}
    assert len(plans) == 1


# --- completeness -----------------------------------------------------------------

def test_every_compound_task_decomposes_to_primitives(planner, idac):
    result, state, _ = semantic_plan(planner, idac)
    assert result.success and result.plan.actions
    primitives = []

    def check(task_node):
        """A task node is primitive, or has exactly one used method whose
        subtasks are all (recursively) decomposed."""
        if task_node.method_name is None:
            assert task_node.status == "primitive", task_node.task
            primitives.append(task_node.task)
            return
        used = [c for c in task_node.children if c.status == "expanded"]
        assert task_node.status == "expanded" and len(used) == 1, task_node.task
        for sub in used[0].children:
            check(sub)

    for top in result.tree.children:
        check(top)
    assert len(primitives) == len(result.plan.actions)


# --- correctness --------------------------------------------------------------------

def test_simulated_execution_meets_every_precondition_and_the_goal(planner, idac):
    result, state, _ = semantic_plan(planner, idac)
    selected = select_rooms(state)
    sim = state.copy()
    for action in result.plan.actions:
        op = planner.operators.get(action.action)
        args = tuple(action.args[p] for p in op.parameters)
        ok, reasons = op.applicable(sim, args)
        assert ok, (action.action, reasons)
        sim = op.apply(sim, args)
    assert sim.search["searched"] == selected                 # every selected room searched, in order
    assert all(sim.search["visits"][r] == 1 for r in selected)
    assert sim.robot["space"] == selected[-1]


# --- ordering ------------------------------------------------------------------------

def test_closed_door_is_opened_before_it_is_traversed(planner, tmp_path):
    door = "0xust1cwr8XOVmfb2C$iKT"                            # Workspace (3) <-> Printer room (1)
    state, keys = idac_with(a300(capabilities=["navigate", "rgb", "manipulate_door"]),
                            tmp_path=tmp_path, closed=[door])
    result = planner.plan(search_task(state, restricted=keys["1"]), [FIND])
    assert result.success
    assert result.plan.initial_doors[door] == "closed"
    assert result.plan.is_before(("manipulate_door", {"door": door}), ("navigate", {"door": door}))


def test_route_order_is_preserved(planner, idac):
    result, state, _ = semantic_plan(planner, idac)
    here = state.robot["space"]
    for a in result.plan.actions:
        if a.action == "navigate":
            assert a.args["from_space"] == here
            here = a.args["to_space"]
        elif a.action == "capture_rgb":
            assert a.args["space"] == here


# --- groundedness ---------------------------------------------------------------------

def test_everything_in_the_plan_comes_from_the_building_robot_or_state(planner, idac):
    result, state, _ = semantic_plan(planner, idac)
    caps = set(state.robot["capabilities"])
    for a in result.plan.actions:
        assert a.robot == "a300"
        op = planner.operators.get(a.action)
        assert op.required_capabilities <= caps
        for key in ("space", "from_space", "to_space"):
            if key in a.args:
                assert a.args[key] in state.spaces
        if "door" in a.args:
            door = state.doors[a.args["door"]]
            assert list(a.args["door_position"]) == door["position"]
            assert list(a.args["approach_from"]) == door["approach"][a.args["from_space"]]
            assert list(a.args["approach_to"]) == door["approach"][a.args["to_space"]]
            assert list(a.args["to_centroid"]) == state.spaces[a.args["to_space"]]["centroid"]
        if "target" in a.args:
            assert a.args["target"] == state.task["target"]


# --- infeasibility ------------------------------------------------------------------------

def test_room_behind_closed_door_without_manipulate_door(planner, tmp_path):
    door = "0xust1cwr8XOVmfb2C$iKT"                            # the only door of room 1
    state, keys = idac_with(tmp_path=tmp_path, closed=[door])
    result = planner.plan(search_task(state, restricted=keys["1"]), [FIND])
    assert not result.success and result.plan is None
    assert result.failure.reason_code == "missing_capability"
    assert result.failure.details["blocked_doors"][0]["door"] == door


def test_unreachable_space(planner, tmp_path):
    state = toy_world(tmp_path, {"A": (0, 0), "B": (4, 0), "C": (9, 9)}, [("dAB", "A", "B", (2, 0), 1.0)], "A")
    result = planner.plan(search_task(state, restricted="C"), [FIND])
    assert not result.success and result.failure.reason_code == "unreachable_space"


def test_door_narrower_than_robot(planner):
    state, keys = idac_with(a300(width_m=0.85))                # Lab's door is 0.9 m: 0.85 + 0.1 > 0.9
    result = planner.plan(search_task(state, restricted=keys["15"]), [FIND])
    assert not result.success and result.failure.reason_code == "door_too_narrow"


def test_exterior_door_is_never_used(idac):
    state, _ = idac
    outside = [g for g, d in state.doors.items() if d["is_exterior"]]
    assert len(outside) == 1
    graph = DoorGraph.from_state(state)
    for a in state.spaces:
        for b in state.spaces:
            route = graph.route(a, b, state.robot)
            assert outside[0] not in [door for _, door in route]


def test_missing_rgb_capability(planner):
    state, keys = idac_with(a300(capabilities=["navigate"]))
    result = planner.plan(search_task(state, restricted=keys["2"]), [FIND])
    assert not result.success and result.failure.reason_code == "missing_capability"
    assert "rgb" in result.failure.message


def test_without_prior_the_planner_asks_for_one(planner, idac):
    state, _ = idac
    result = planner.plan(search_task(state), [FIND])
    assert not result.success and result.failure.reason_code == "prior_needed"


def test_recorded_rooms_first_then_semantic_by_backtracking(planner, idac):
    state, keys = idac
    search_task(state, recorded=[keys["15"]])
    first = planner.plan(state, [FIND])
    assert first.tree.children[0].method_name == "m_search_recorded"
    state.search["visits"][keys["15"]] = 1                     # Lab searched, nothing found
    with_prior(state, keys)
    second = planner.plan(state, [FIND])
    attempts = {c.method_name: c for c in second.tree.children[0].children}
    assert attempts["m_search_recorded"].status == "rejected"
    assert "has been searched" in attempts["m_search_recorded"].reasons[-1]
    assert second.tree.children[0].method_name == "m_search_semantic"
    assert keys["15"] not in [a.args.get("space") for a in second.plan.actions]


# --- distances ------------------------------------------------------------------------------

def test_door_graph_route_office4_to_office3_goes_through_5_and_14(idac):
    state, keys = idac
    route = DoorGraph.from_state(state).route(keys["10"], keys["11"], state.robot)
    assert [state.spaces[s]["name"] for s, _ in route] == ["5", "14", "11"]


def test_route_and_route_length_are_the_same_shortest_path(tmp_path):
    """A cycle where the fewest-doors path (A-D direct, 1 door) is longer than
    the path A-B-C-D (3 doors). BFS would pick A-D; Dijkstra picks A-B-C-D, and
    route_length measures exactly that path."""
    rooms = {"A": (0, 0), "B": (1, 0), "C": (1, 2), "D": (0, 2)}
    doors = [("dAD", "A", "D", (10, 1), 1.0), ("dAB", "A", "B", (0.5, 0), 1.0),
             ("dBC", "B", "C", (1, 1), 1.0), ("dCD", "C", "D", (0.5, 2), 1.0)]
    state = toy_world(tmp_path, rooms, doors, "A")
    graph = DoorGraph.from_state(state)
    route = graph.route("A", "D", state.robot)
    assert [door for _, door in route] == ["dAB", "dBC", "dCD"]
    points = [(0, 0), (0.5, 0), (1, 1), (0.5, 2), (0, 2)]
    expected = sum(math.dist(p, q) for p, q in zip(points, points[1:]))
    assert math.isclose(graph.route_length("A", "D", state.robot), expected)
    direct = 2 * math.dist((0, 0), (10, 1))
    assert expected < direct


def test_euclidean_setting_orders_by_straight_lines(planner, idac):
    state, keys = idac
    with_prior(search_task(state), keys)
    state.settings["distance"] = "euclidean"
    a = select_rooms(state)
    state.settings["distance"] = "door_graph"
    b = select_rooms(state)
    assert sorted(a) == sorted(b)                              # same rooms, possibly another order
