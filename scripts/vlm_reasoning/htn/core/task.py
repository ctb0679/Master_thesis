"""HTN tasks: the units of work the planner decomposes.

A primitive task is executed by exactly one operator of the same name; a
compound task is decomposed by one of the methods registered for its name
(Tandon et al., ECPPM 2026, Sec. II: "high-level tasks ... decomposed into
subtasks until only primitive, robot-executable actions remain")."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Task:
    """HTN concept: task. Abstract: use PrimitiveTask or CompoundTask.
    Immutable and hashable, so the args must be hashable too (strings,
    numbers, tuples)."""
    name: str
    args: tuple = ()

    def __post_init__(self):
        if type(self) is Task:
            raise TypeError("Task is abstract: use PrimitiveTask or CompoundTask")

    def __str__(self):
        return f"{self.name}({', '.join(str(a) for a in self.args)})"


@dataclass(frozen=True)
class PrimitiveTask(Task):
    """HTN concept: primitive task. Executed by the operator with the same name."""


@dataclass(frozen=True)
class CompoundTask(Task):
    """HTN concept: compound task. Decomposed by a method registered for its name."""
