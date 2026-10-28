"""Operators (primitive actions) and the operator library.

Paper (Sec. II): an OperatorLibrary of primitive actions, each with explicit
preconditions and effects, simulated on a copy of the world state."""

from dataclasses import dataclass, field


@dataclass
class Problem:
    """One failed precondition: machine-readable code, human message, details."""
    code: str
    message: str
    details: dict = field(default_factory=dict)


def bind_parameters(owner, parameters, args):
    if len(args) != len(parameters):
        raise ValueError(f"{owner} takes {len(parameters)} argument(s) {parameters}, got {len(args)}: {args}")
    return dict(zip(parameters, args))


def failed_conditions(conditions, state, bindings):
    problems = []
    for cond in conditions:
        bound = cond.bind(bindings)
        if not bound.check(state):
            problems.append(Problem(bound.code(state), bound.explain(state), bound.details(state)))
    return problems


@dataclass
class Operator:
    """HTN concept: operator, the executable definition of one primitive task.
    `deterministic=False` marks an action whose outcome is only known at run
    time (here: detect); the planner simulates its declared effects and so
    plans as if the outcome were the default one."""
    name: str
    parameters: tuple
    preconditions: list
    effects: list
    required_capabilities: set = field(default_factory=set)
    deterministic: bool = True
    description: str = ""

    def check(self, state, args):
        """All problems that stop this operator from applying (empty = applicable)."""
        bindings = bind_parameters(self.name, self.parameters, args)
        have = set(state.robot.get("capabilities", ()))
        problems = [Problem("missing_capability", f"robot '{state.robot.get('id')}' lacks capability '{cap}'")
                    for cap in sorted(self.required_capabilities - have)]
        return problems + failed_conditions(self.preconditions, state, bindings)

    def applicable(self, state, args):
        """(True, []) or (False, [reason, ...])."""
        problems = self.check(state, args)
        return not problems, [p.message for p in problems]

    def apply(self, state, args):
        """A NEW state with the effects applied; the input is never changed."""
        bindings = bind_parameters(self.name, self.parameters, args)
        new = state.copy()
        for effect in self.effects:
            effect.bind(bindings).apply(new)
        return new

    def describe(self):
        return {
            "name": self.name,
            "parameters": list(self.parameters),
            "preconditions": [str(c) for c in self.preconditions],
            "effects": [str(e) for e in self.effects],
            "capabilities": sorted(self.required_capabilities),
            "deterministic": self.deterministic,
            "description": self.description,
        }


class OperatorLibrary:
    """HTN concept: operator library (paper: OperatorLibrary)."""

    def __init__(self):
        self._ops = {}

    def register(self, operator):
        if operator.name in self._ops:
            raise ValueError(f"operator '{operator.name}' is already registered")
        self._ops[operator.name] = operator
        return operator

    def get(self, name):
        if name not in self._ops:
            raise KeyError(f"no operator named '{name}' (known: {sorted(self._ops)})")
        return self._ops[name]

    def all(self):
        return list(self._ops.values())

    def as_table(self):
        return [op.describe() for op in self._ops.values()]
