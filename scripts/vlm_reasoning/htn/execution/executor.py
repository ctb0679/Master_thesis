"""PlanExecutor: plan -> execute -> observe -> update belief -> replan.

The only place with model calls: the injected `ranker` (room probabilities,
at most once per task) and the perception backend (one detection per
searched room). The planner gets their results through the world state."""

import math
from dataclasses import asdict, dataclass, field

from htn.domain.building import DoorGraph
from htn.domain.methods import eligible_spaces, reachability


class ExecutionError(RuntimeError):
    pass


@dataclass
class RunSummary:
    # found | not_found | not_found_reachable_exhausted | reached | infeasible: <code> | aborted
    outcome: str
    reason: str = ""
    found: list = field(default_factory=list)
    probability: dict = None          # find_object: prior/posterior mass per group (see _account)
    unreachable: list = field(default_factory=list)   # rooms this robot cannot reach, with the reason
    rankings: list = field(default_factory=list)   # ranking model output, with its reasoning per room
    visit_order: list = field(default_factory=list)
    travelled_m: float = 0.0
    iterations: int = 0
    model_calls: dict = field(default_factory=dict)
    failure: dict = None
    log_dir: str = None

    def to_dict(self):
        return asdict(self)


class PlanExecutor:
    """Owns the replanning loop for one task (find_object or go_to).

    find_object: plan the full sweep assuming "not found" everywhere,
    execute up to and including the first detect and its update_belief /
    record_result, observe, update the belief, replan. go_to: plan once,
    execute everything."""

    def __init__(self, planner, state, belief, perception, navigation, ranker, logger,
                 miss_rate=None, detect_threshold=0.0, residual_mass=None, log=print):
        self.planner, self.state, self.belief = planner, state, belief
        self.perception, self.navigation, self.ranker, self.logger = perception, navigation, ranker, logger
        self.miss_rate, self.detect_threshold = miss_rate, detect_threshold
        self.residual_mass = residual_mass if residual_mass is not None else 1 - state.settings["prob_mass"]
        self.log = log
        self.ranker_calls = 0
        self.rankings = []

    def label(self, space):
        return DoorGraph.from_state(self.state).label(space)

    # -- main loop ---------------------------------------------------------------

    def run(self, task):
        find = task.name == "find_object"
        if find and self.miss_rate is None:
            raise ExecutionError("find_object needs the detector miss rate (--miss_rate); it has no default")
        summary = RunSummary("not_found")
        pos = tuple(self.state.spaces[self.state.robot["space"]]["centroid"])
        max_iterations = len(self.state.spaces) * self.state.settings["max_visits_per_room"] + 3

        for n in range(1, max_iterations + 1):
            summary.iterations = n
            record = {"iteration": n, "task": {"name": task.name, "args": list(task.args)},
                      "state": self.state.to_dict(), "prior": dict(self.belief.prior),
                      "posterior": self.belief.posterior()}

            if find and self.belief.has_prior:
                eligible, unreachable = eligible_spaces(self.state)
                stop, why = self.belief.should_stop(eligible, self.residual_mass, list(unreachable))
                if stop:
                    summary.outcome, summary.reason = ("found" if summary.found else "not_found"), why
                    record["stopped"] = why
                    self.logger.log_iteration(n, record)
                    break

            result = self.planner.plan(self.state, [task])
            top = result.tree.children[0]
            record.update(chosen_method=top.method_name, tree=result.tree.to_dict(),
                          plan=result.plan.to_dict() if result.plan else None,
                          failure=result.failure.to_dict() if result.failure else None)

            if not result.success:
                code = result.failure.reason_code
                if code == "prior_needed" and self.ranker_calls == 0:
                    self._fetch_prior()
                    record["note"] = "no prior yet: asked the ranking model, replanning"
                    record["rankings"] = self.rankings
                    self.logger.log_iteration(n, record)
                    continue
                if code == "no_eligible_rooms":
                    summary.outcome = "found" if summary.found else "not_found"
                    summary.reason = result.failure.message
                else:
                    summary.outcome, summary.reason = f"infeasible: {code}", result.failure.message
                    summary.failure = result.failure.to_dict()
                    self.log(f"No plan: {code}: {result.failure.message}")
                self.logger.log_iteration(n, record)
                break

            if not result.plan.actions:
                if find:
                    summary.outcome = "found" if summary.found else "not_found"
                    summary.reason = "nothing left to search"
                else:
                    summary.outcome, summary.reason = "reached", "already there"
                self.logger.log_iteration(n, record)
                break

            self._print_plan(n, top.method_name, result.plan)
            executed, observation, pos, travelled = self._execute(result.plan, pos)
            summary.travelled_m = round(summary.travelled_m + travelled, 2)
            record.update(executed=executed, observation=observation.to_dict() if observation else None,
                          posterior_after=self.belief.posterior())
            self.logger.log_iteration(n, record)

            if observation is not None:
                summary.visit_order.append(observation.space)
                if observation.aborted:
                    summary.outcome, summary.reason = "aborted", "user ended the session"
                    break
                if self._is_find(observation):
                    summary.found.append(observation.space)
                    if self.state.task.get("mode") == "single":
                        summary.outcome = "found"
                        summary.reason = f"detection confidence {observation.confidence}"
                        break
            if not find:
                summary.outcome, summary.reason = "reached", f"reached {self.label(task.args[0])}"
                break
        else:
            summary.reason = f"stopped after {max_iterations} iterations"

        if find:
            self._account(summary)
        summary.rankings = self.rankings
        summary.model_calls = {"ranker": self.ranker_calls,
                               "detector": getattr(self.perception, "model_calls", 0)}
        summary.log_dir = str(self.logger.dir) if self.logger.dir else None
        return summary

    # -- steps ---------------------------------------------------------------------

    def _fetch_prior(self):
        """Ask the ranking model once; misses observed before (in recorded
        rooms) are then applied to the new prior. The ranker returns
        [{space, probability, likelihood, reasoning}, ...]."""
        self.log("\nQuerying the model for room probabilities...")
        self.rankings = sorted(self.ranker(self.state.task.get("text") or self.state.task.get("target")),
                               key=lambda r: -r["probability"])
        self.ranker_calls += 1
        self.belief.set_prior({r["space"]: r["probability"] for r in self.rankings})
        for obs in self.state.search["observations"]:
            if not obs["skipped"] and not obs["found"]:
                self.belief.update_on_miss(obs["space"], self.miss_rate)
        self.state.belief = self.belief.to_state()
        for r in self.rankings:
            score = f"score={r['likelihood']:.2f}  " if r.get("likelihood") is not None else ""
            self.log(f"  p={self.belief.prior[r['space']]:.3f}  {score}"
                     f"{self.label(r['space']):28s} {r.get('reasoning', '')}")

    def _account(self, summary):
        """Where the probability is at the end: reachable rooms searched,
        reachable rooms not searched, unreachable rooms (prior and posterior
        mass; None without a prior), and each unreachable room with its
        reason. "not_found" becomes "not_found_reachable_exhausted" when the
        unreachable rooms hold more than residual_mass."""
        reachable, unreachable = reachability(self.state)
        visits = self.state.search["visits"]
        groups = {"reachable_searched": [g for g in reachable if visits.get(g, 0) > 0],
                  "reachable_not_searched": [g for g in reachable if visits.get(g, 0) == 0],
                  "unreachable": list(unreachable)}
        prior, post = self.belief.prior, self.belief.posterior()
        if self.belief.has_prior:
            summary.probability = {k: {"prior": round(sum(prior.get(g, 0.0) for g in v), 4),
                                       "posterior": round(sum(post.get(g, 0.0) for g in v), 4),
                                       "rooms": len(v)} for k, v in groups.items()}
        rooms = self.state.spaces
        summary.unreachable = sorted(
            ({"space": g, "name": rooms[g]["name"], "function": rooms[g].get("function", ""),
              "storey": rooms[g].get("storey"),
              "prior": round(prior[g], 4) if g in prior else None,
              "posterior": round(post[g], 4) if g in post else None, **why}
             for g, why in unreachable.items()),
            key=lambda r: (-(r["prior"] or 0.0), r["name"]))
        if (summary.outcome == "not_found" and self.belief.has_prior
                and self.belief.mass(groups["unreachable"]) > self.residual_mass):
            summary.outcome = "not_found_reachable_exhausted"

    def _is_find(self, obs):
        return (not obs.skipped and obs.found
                and (obs.confidence if obs.confidence is not None else 0.0) >= self.detect_threshold)

    def _execute(self, plan, pos):
        """Run actions up to the end of the first searched space (record_result),
        or all of them for go_to. Returns (executed, observation, position, metres)."""
        state, ops = self.state, self.planner.operators
        executed, observation, travelled = [], None, 0.0
        for action in plan.actions:
            op = ops.get(action.action)
            args = tuple(action.args[p] for p in op.parameters)
            ok, reasons = op.applicable(state, args)
            if not ok:
                raise ExecutionError(f"{action.action} not applicable at run time: {reasons}")
            if not self.navigation.execute(action):
                raise ExecutionError(f"navigation backend failed on {action.action}")
            if action.action == "navigate":
                travelled += math.dist(pos, action.args["door_position"])
                pos = tuple(action.args["door_position"])
                self.log(f"  navigate -> {self.label(action.args['to_space'])} "
                         f"(door {action.args['door']}, {state.doors[action.args['door']]['clear_width_m']} m)")
            elif action.action == "manipulate_door":
                self.log(f"  open door {action.args['door']}")
            elif action.action == "capture_rgb":
                centroid = tuple(state.spaces[action.args["space"]]["centroid"])
                travelled += math.dist(pos, centroid)
                pos = centroid
            elif action.action == "detect":
                space = action.args["space"]
                observation = self.perception.observe(space, self.label(space), state.task.get("text") or "")
                if observation.aborted:
                    self.state = state
                    executed.append(action.to_dict())
                    return executed, observation, pos, travelled
            state = op.apply(state, args)
            if action.action == "update_belief":
                self._update_belief(state, observation)
            executed.append(action.to_dict())
            if action.action == "record_result":
                break
        if executed and executed[-1]["action"] == "navigate":         # go_to ends at the room centroid
            centroid = tuple(state.spaces[state.robot["space"]]["centroid"])
            travelled += math.dist(pos, centroid)
            pos = centroid
        self.state = state
        return executed, observation, pos, travelled

    def _update_belief(self, state, obs):
        """Skipped image: belief unchanged. Find: recorded. Miss: Bayesian update."""
        found = self._is_find(obs)
        state.search["observations"].append({**obs.to_dict(), "found": found})
        if obs.skipped:
            self.log(f"  {self.label(obs.space)}: no image, belief unchanged")
        elif found:
            state.search["found"].append(obs.space)
            self.log(f"  {self.label(obs.space)}: FOUND (confidence {obs.confidence}) {obs.description}")
        else:
            self.log(f"  {self.label(obs.space)}: not found (confidence {obs.confidence}) {obs.description}")
            if self.belief.has_prior:
                self.belief.update_on_miss(obs.space, self.miss_rate)
        state.belief = self.belief.to_state()

    def _print_plan(self, n, method, plan):
        rooms = [a.args["space"] for a in plan.actions if a.action == "capture_rgb"]
        target = [a.args["to_space"] for a in plan.actions if a.action == "navigate"][-1:] if not rooms else []
        stops = " -> ".join(self.label(s) for s in (rooms or target)) or "(none)"
        self.log(f"\n[plan {n}] {method}: {stops}  ({len(plan.actions)} actions)")
