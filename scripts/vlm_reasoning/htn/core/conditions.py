"""Preconditions and effects as small named objects.

They are declared as data on operators and methods (not hidden in function
bodies), so they can be listed, printed and tested. An argument written as
"?name" is a parameter placeholder: Operator/Method bind it to the task's
actual argument before checking or applying (as in PDDL), e.g.
RobotAt("?to_space") becomes RobotAt("<space guid>")."""

import copy


def _bind_args(args, bindings):
    return tuple(bindings[a[1:]] if isinstance(a, str) and a.startswith("?") else a for a in args)


class _Named:
    def __init__(self, *args):
        self.args = tuple(args)

    @property
    def name(self):
        return type(self).__name__

    def bind(self, bindings):
        """A copy with every "?param" replaced by bindings[param]."""
        bound = copy.copy(self)
        bound.args = _bind_args(self.args, bindings)
        return bound

    def __str__(self):
        return f"{self.name}({', '.join(str(a) for a in self.args)})"

    __repr__ = __str__


class Condition(_Named):
    """HTN concept: precondition. Subclasses implement check(); explain()
    gives the failure message used in the decomposition tree and in
    PlanFailure, and `reason_code` the machine-readable failure code."""

    reason_code = "precondition_failed"     # generic: "this method/operator does not apply"

    def check(self, state):
        raise NotImplementedError

    def explain(self, state):
        return f"{self} is false"

    def code(self, state):
        return self.reason_code

    def details(self, state):
        return {}


class Effect(_Named):
    """HTN concept: effect. apply() changes the state in place; the planner
    only ever applies effects to a copy of the world state."""

    def apply(self, state):
        raise NotImplementedError
