"""The HTN planner: depth-first, totally ordered decomposition with backtracking.

Paper (Sec. II): "a custom Python HTN planner ... recursively decomposes
high-level tasks ... Operator effects are simulated on a copy of the world
state, and the planner backtracks to the next method if a branch fails."
Rule-based and deterministic: no model calls, no randomness, no I/O."""

from dataclasses import dataclass, field

from .plan import Plan, PrimitiveAction
from .task import CompoundTask, PrimitiveTask

GENERIC = "precondition_failed"


@dataclass
class DecompositionNode:
    """HTN concept: one node of the decomposition tree. A compound task node
    has one child per method tried (task repeated, `method_name` set):
    "rejected" (preconditions failed, `reasons` say why), "failed"
    (expanded, but a later step failed and the planner backtracked) or
    "expanded" (used; its children are the subtasks). Primitive nodes have
    status "primitive" or "failed"."""
    task: str
    method_name: str = None
    status: str = "failed"
    reasons: list = field(default_factory=list)
    children: list = field(default_factory=list)

    def to_dict(self):
        out = {"task": self.task, "method": self.method_name, "status": self.status}
        if self.reasons:
            out["reasons"] = list(self.reasons)
        if self.children:
            out["children"] = [c.to_dict() for c in self.children]
        return out


@dataclass
class PlanFailure:
    """Explicit infeasibility: which task failed, a machine-readable code
    (e.g. unreachable_space, missing_capability, door_too_narrow,
    exterior_not_allowed, no_eligible_rooms, no_applicable_method), a message
    and details."""
    task: str
    reason_code: str
    message: str
    details: dict = field(default_factory=dict)

    def to_dict(self):
        return {"task": self.task, "reason_code": self.reason_code,
                "message": self.message, "details": self.details}


@dataclass
class PlanResult:
    success: bool
    plan: Plan
    tree: DecompositionNode
    failure: PlanFailure
    initial_state: dict

    def to_dict(self):
        return {"success": self.success,
                "plan": self.plan.to_dict() if self.plan else None,
                "failure": self.failure.to_dict() if self.failure else None,
                "tree": self.tree.to_dict()}


class HTNPlanner:
    """HTN concept: planner. plan(state, tasks) decomposes the task list
    left to right. Primitive tasks are checked and simulated with their
    operator; compound tasks try their methods in priority order. When a
    branch fails anywhere later in the list, the planner backtracks to the
    most recent method choice and tries the next method."""

    def __init__(self, operators, methods, max_depth=50):
        self.operators = operators
        self.methods = methods
        self.max_depth = max_depth

    def plan(self, state, tasks):
        root = DecompositionNode(task="goal(" + ", ".join(str(t) for t in tasks) + ")", status="expanded")
        agenda = [(t, root, 0) for t in tasks]
        steps, failure = self._seek(state, agenda)
        if failure is not None:
            root.status = "failed"
            root.reasons = [f"{failure.reason_code}: {failure.message}"]
            return PlanResult(False, None, root, failure, state.to_dict())
        robot_id = state.robot.get("id", "robot")
        actions = [PrimitiveAction(task.name, robot_id, dict(zip(op.parameters, task.args)))
                   for task, op in steps]
        plan = Plan(actions,
                    initial_robot={k: state.robot.get(k) for k in ("id", "space", "capabilities")},
                    initial_doors={g: d.get("state") for g, d in state.doors.items()})
        return PlanResult(True, plan, root, None, state.to_dict())

    # -- recursion -------------------------------------------------------------
    # agenda: list of (task, parent node, depth). Returns ([(task, operator)], None)
    # on success or (None, PlanFailure).

    def _seek(self, state, agenda):
        if not agenda:
            return [], None
        (task, parent, depth), rest = agenda[0], agenda[1:]
        if depth > self.max_depth:
            return None, PlanFailure(str(task), "max_depth", f"decomposition deeper than {self.max_depth}")
        if isinstance(task, PrimitiveTask):
            return self._primitive(state, task, parent, depth, rest)
        if isinstance(task, CompoundTask):
            return self._compound(state, task, parent, depth, rest)
        raise TypeError(f"{task!r} is neither a PrimitiveTask nor a CompoundTask")

    def _primitive(self, state, task, parent, depth, rest):
        op = self.operators.get(task.name)
        node = DecompositionNode(str(task), status="primitive")
        parent.children.append(node)
        problems = op.check(state, task.args)
        if problems:
            node.status = "failed"
            node.reasons = [f"{p.code}: {p.message}" for p in problems]
            return None, PlanFailure(str(task), problems[0].code, problems[0].message,
                                     {"problems": [p.__dict__ for p in problems]})
        steps, failure = self._seek(op.apply(state, task.args), rest)
        if failure is not None:
            return None, failure
        return [(task, op)] + steps, None

    def _compound(self, state, task, parent, depth, rest):
        node = DecompositionNode(str(task), status="failed")
        parent.children.append(node)
        methods = self.methods.methods_for(task.name)
        failures = []
        for method in methods:
            attempt = DecompositionNode(str(task), method.name, "rejected")
            node.children.append(attempt)
            problems = method.check(state, task.args)
            if problems:
                attempt.reasons = [f"{p.code}: {p.message}" for p in problems]
                p = problems[0]
                failures.append((method.name, PlanFailure(str(task), p.code, p.message, p.details)))
                continue
            attempt.status = "expanded"
            subtasks = method.decompose(state, task.args)
            steps, failure = self._seek(state, [(t, attempt, depth + 1) for t in subtasks] + rest)
            if failure is None:
                node.method_name, node.status = method.name, "expanded"
                return steps, None
            attempt.status = "failed"            # backtrack: try the next method
            attempt.reasons = [f"{failure.reason_code}: {failure.message}"]
            failures.append((method.name, failure))
        return None, self._combine(task, failures)

    @staticmethod
    def _combine(task, failures):
        """One failure for a task whose methods all failed. A specific code
        (anything but the generic 'precondition_failed') is passed up when
        all methods that gave one agree on it, e.g. go_to fails with
        door_too_narrow, not with "no method applies"; otherwise the result
        is no_applicable_method with every method's reason in the details."""
        details = {"methods": {name: f"{f.reason_code}: {f.message}" for name, f in failures}}
        specific = [f for _, f in failures if f.reason_code != GENERIC]
        if specific and len({f.reason_code for f in specific}) == 1:
            first = specific[0]
            return PlanFailure(first.task, first.reason_code, first.message, {**first.details, **details})
        if not failures:
            return PlanFailure(str(task), "no_applicable_method", f"no method registered for '{task.name}'")
        return PlanFailure(str(task), "no_applicable_method", f"no method of '{task.name}' applies", details)
