"""Methods (ways to decompose a compound task) and the method library.

Paper (Sec. II): a MethodLibrary where each compound task has an ordered list
of candidate methods; the planner tries them in that order."""

from dataclasses import dataclass

from .operator import bind_parameters, failed_conditions


@dataclass
class Method:
    """HTN concept: method. If all `preconditions` hold, `body(state, *args)`
    returns the ordered list of subtasks. Bodies only read the state: no
    model calls, no I/O, no randomness."""
    name: str
    task_name: str
    parameters: tuple
    preconditions: list
    body: object                      # callable(state, *args) -> list[Task]
    description: str = ""

    def check(self, state, args):
        """Failed preconditions as Problems (empty = the method applies)."""
        return failed_conditions(self.preconditions, state,
                                 bind_parameters(self.name, self.parameters, args))

    def decompose(self, state, args):
        return list(self.body(state, *args))

    def describe(self):
        return {
            "name": self.name,
            "task": self.task_name,
            "parameters": list(self.parameters),
            "preconditions": [str(c) for c in self.preconditions],
            "description": self.description,
        }


class MethodLibrary:
    """HTN concept: method library. Registration order = priority order."""

    def __init__(self):
        self._methods = {}

    def register(self, method):
        existing = self._methods.setdefault(method.task_name, [])
        if any(m.name == method.name for m in existing):
            raise ValueError(f"method '{method.name}' is already registered for '{method.task_name}'")
        existing.append(method)
        return method

    def methods_for(self, task_name):
        return list(self._methods.get(task_name, []))

    def all(self):
        return [m for ms in self._methods.values() for m in ms]

    def as_table(self):
        return [m.describe() for m in self.all()]
