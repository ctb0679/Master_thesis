"""The HTN framework on its own, on a toy domain (two rooms, one door) that
uses nothing from htn.domain."""

from htn.core import (CompoundTask, Condition, Effect, HTNPlanner, Method, MethodLibrary, Operator,
                      OperatorLibrary, PrimitiveTask, WorldState)
import pytest


class At(Condition):
    def check(self, state):
        return state.robot["space"] == self.args[0]


class IsOpen(Condition):
    def check(self, state):
        return state.doors[self.args[0]]["state"] == "open"

    def explain(self, state):
        return f"door {self.args[0]} is {state.doors[self.args[0]]['state']}"


class IsClosed(Condition):
    def check(self, state):
        return state.doors[self.args[0]]["state"] == "closed"


class MoveTo(Effect):
    def apply(self, state):
        state.robot["space"] = self.args[0]


class Open(Effect):
    def apply(self, state):
        state.doors[self.args[0]]["state"] = "open"


def toy_domain(walk_first=True):
    ops = OperatorLibrary()
    ops.register(Operator("move", ("from", "to", "door"), [At("?from"), IsOpen("?door")], [MoveTo("?to")], {"drive"}))
    ops.register(Operator("open", ("door",), [IsClosed("?door")], [Open("?door")], {"arm"}))
    ms = MethodLibrary()
    walk = Method("m_walk_in", "enter", ("from", "to", "door"), [IsOpen("?door")],
                  lambda s, a, b, d: [PrimitiveTask("move", (a, b, d))])
    open_walk = Method("m_open_then_walk", "enter", ("from", "to", "door"), [],
                       lambda s, a, b, d: [PrimitiveTask("open", (d,)), PrimitiveTask("move", (a, b, d))])
    for m in ([walk, open_walk] if walk_first else [open_walk, walk]):
        ms.register(m)
    return ops, ms


def toy_state(door="closed", caps=("drive", "arm")):
    return WorldState(robot={"id": "toy", "space": "A", "capabilities": list(caps)},
                      doors={"d1": {"rooms": ["A", "B"], "state": door}})


ENTER = CompoundTask("enter", ("A", "B", "d1"))


def test_backtracking_picks_second_method_and_keeps_rejected_one_in_tree():
    result = HTNPlanner(*toy_domain()).plan(toy_state("closed"), [ENTER])
    assert result.success
    assert [a.action for a in result.plan.actions] == ["open", "move"]
    enter = result.tree.children[0]
    assert enter.method_name == "m_open_then_walk"
    first, second = enter.children
    assert (first.method_name, first.status) == ("m_walk_in", "rejected")
    assert first.reasons == ["precondition_failed: door d1 is closed"]
    assert (second.method_name, second.status) == ("m_open_then_walk", "expanded")


class Close(Effect):
    def apply(self, state):
        state.doors[self.args[0]]["state"] = "closed"


def test_backtracks_when_a_later_task_fails():
    """'enter' succeeds with its first method, which leaves the door open;
    the NEXT task needs it closed and fails, so the planner goes back into
    'enter' and uses its second method (which closes the door behind it)."""
    ops = OperatorLibrary()
    ops.register(Operator("move", ("from", "to", "door"), [At("?from"), IsOpen("?door")], [MoveTo("?to")]))
    ops.register(Operator("open", ("door",), [IsClosed("?door")], [Open("?door")]))
    ops.register(Operator("close", ("door",), [IsOpen("?door")], [Close("?door")]))
    ops.register(Operator("check_closed", ("door",), [IsClosed("?door")], []))
    ms = MethodLibrary()
    ms.register(Method("m_open_walk", "enter", ("from", "to", "door"), [],
                       lambda s, a, b, d: [PrimitiveTask("open", (d,)), PrimitiveTask("move", (a, b, d))]))
    ms.register(Method("m_open_walk_close", "enter", ("from", "to", "door"), [],
                       lambda s, a, b, d: [PrimitiveTask("open", (d,)), PrimitiveTask("move", (a, b, d)),
                                           PrimitiveTask("close", (d,))]))
    result = HTNPlanner(ops, ms).plan(toy_state("closed"), [ENTER, PrimitiveTask("check_closed", ("d1",))])
    assert result.success
    enter = result.tree.children[0]
    assert [(c.method_name, c.status) for c in enter.children] == \
        [("m_open_walk", "failed"), ("m_open_walk_close", "expanded")]
    assert [a.action for a in result.plan.actions] == ["open", "move", "close", "check_closed"]


def test_operator_apply_never_mutates_its_input():
    ops, _ = toy_domain()
    state = toy_state("closed")
    before = state.to_dict()
    new = ops.get("open").apply(state, ("d1",))
    assert state.to_dict() == before
    assert new.doors["d1"]["state"] == "open"


def test_same_input_same_plan():
    planner = HTNPlanner(*toy_domain())
    plans = {planner.plan(toy_state("closed"), [ENTER]).plan.to_json() for _ in range(10)}
    assert len(plans) == 1


def test_missing_capability_gives_no_plan_and_a_reason():
    result = HTNPlanner(*toy_domain()).plan(toy_state("closed", caps=("drive",)), [ENTER])
    assert not result.success and result.plan is None
    assert result.failure.reason_code == "missing_capability"


def test_libraries_reject_duplicates_and_unknown_names():
    ops, ms = toy_domain()
    with pytest.raises(ValueError):
        ops.register(Operator("move", (), [], []))
    with pytest.raises(KeyError):
        ops.get("fly")
    assert [m["name"] for m in ms.as_table()] == ["m_walk_in", "m_open_then_walk"]
    assert ops.as_table()[0]["preconditions"] == ["At(?from)", "IsOpen(?door)"]


def test_task_is_abstract_and_printable():
    with pytest.raises(TypeError):
        from htn.core import Task
        Task("x")
    assert str(ENTER) == "enter(A, B, d1)"
    assert hash(ENTER) == hash(CompoundTask("enter", ("A", "B", "d1")))
