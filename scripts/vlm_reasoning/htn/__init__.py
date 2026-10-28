"""HTN planning layer for object search (docs/htn_design.md).

core/       domain-independent HTN framework (tasks, operators, methods, planner)
domain/     the object-search domain: building, robot, operators, methods
execution/  plan -> execute -> observe -> replan, with belief update and logs
cli.py      flags and wiring for interactive_search.py"""

from .core import HTNPlanner
from .domain import build_domain, load_world
from .execution import PlanExecutor

__all__ = ["HTNPlanner", "PlanExecutor", "build_domain", "load_world"]
