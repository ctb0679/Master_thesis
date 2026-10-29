"""Flags for the HTN path and the glue between interactive_search.py's REPL
and the PlanExecutor (HTNSearch)."""

import sys
from pathlib import Path

from rank_candidates import call_ollama, softmax_probabilities, validate_rankings

from .core import CompoundTask, HTNPlanner
from .domain import (build_domain, connectivity_report, format_connectivity, load_world, select_robot,
                     set_task, unlinked_stairs)
from .domain.world import default_door_states_path
from .execution import BeliefModel, ManualImageBackend, PlanExecutor, RunLogger, SimulatedNavigation

ROOT = Path(__file__).resolve().parents[3]
ROBOT_DIR = ROOT / "data" / "robot"
RESULTS_DIR = ROOT / "results" / "htn_runs"


def note(message):
    print(message, file=sys.stderr)


def add_htn_args(parser):
    g = parser.add_argument_group("HTN planning (default; --no-htn for the old SearchSession baseline)")
    g.add_argument("--no-htn", dest="no_htn", action="store_true",
                   help="Use the old SearchSession search (baseline for the evaluation).")
    g.add_argument("--robot", help="URDF stem in data/robot/urdf/ (needed when there are several).")
    g.add_argument("--door_states", help="Door-states JSON (default data/robot/door_states_<building>.json; "
                                         "all doors open if there is none).")
    g.add_argument("--miss_rate", type=float,
                   help="Detector miss rate P(not detected | present) for the belief update. "
                        "Required with the HTN; no default.")
    g.add_argument("--detect_threshold", type=float, default=0.0,
                   help="A detection counts when found is true and confidence >= this (default 0.0).")
    g.add_argument("--max_visits_per_room", type=int, default=1, help="Visits per room per task (default 1).")
    g.add_argument("--residual_mass", type=float,
                   help="Stop when the unsearched rooms hold less probability than this (default 1 - prob_mass).")
    g.add_argument("--distance", choices=("door_graph", "euclidean"), default="door_graph",
                   help="Distance for the visit order: route length through doors (default) or straight lines.")


def check_htn_args(parser, args):
    if args.no_htn:
        return
    if args.miss_rate is None:
        parser.error("--miss_rate is required for HTN find_object runs (or use --no-htn)")
    if not 0.0 <= args.miss_rate <= 1.0:
        parser.error("--miss_rate must be in [0, 1]")
    if args.max_visits_per_room < 1:
        parser.error("--max_visits_per_room must be >= 1")


class HTNSearch:
    """Runs find_object / go_to_room tasks from the REPL through the HTN.
    The world (robot position, door states) carries over between tasks."""

    def __init__(self, args, session, llm_json, full_json):
        self.args = args
        self.robot = select_robot(ROBOT_DIR / "urdf", ROBOT_DIR / "profiles", args.robot, note)
        self.key_to_guid = {s["name"]: s["guid"] for s in session.building["spaces"]}
        door_states = args.door_states or default_door_states_path(llm_json, ROBOT_DIR)
        settings = {"prob_mass": args.prob_mass, "min_prob": args.min_prob,
                    "max_visits_per_room": args.max_visits_per_room, "distance": args.distance}
        self.world, report = load_world(llm_json, full_json, self.robot, door_states,
                                        self.key_to_guid[session.start_room], settings)
        for line in report:
            note(f"NOTE: {line}")
        self.planner = HTNPlanner(*build_domain())
        self.perception = ManualImageBackend(session.detector)
        note(f"NOTE: HTN planning with robot '{self.robot.id}' (width {self.robot.width_m} m, "
            f"capabilities {self.robot.capabilities}); door states: {self.world.settings['door_states_source']}.")
        self.stairs = unlinked_stairs(full_json, set(self.world.spaces))
        for line in format_connectivity(connectivity_report(self.world, self.stairs), self._label):
            note(line)

    def _label(self, space):
        s = self.world.spaces[space]
        return f"{s['name']} ({s['function']})" if s.get("function") and s["function"] != s["name"] else s["name"]

    @classmethod
    def from_args(cls, args, session):
        llm_json = Path(args.cleaned_json)
        full_json = Path(args.full_json) if args.full_json else None
        if not llm_json.name.endswith("_llm.json") or full_json is None or not full_json.exists():
            raise RuntimeError("the HTN needs a <name>_llm.json and its <name>_full.json (doors and positions); "
                               f"{llm_json.name} has none. Use --no-htn for this building.")
        return cls(args, session, llm_json, full_json)

    def make_ranker(self, session):
        """call_ollama -> validate_rankings -> softmax, as a list of
        {space (guid), probability, likelihood, reasoning}."""
        def rank(task_text):
            rankings = call_ollama(session.host, session.model, task_text, session.room_summaries,
                                   options=session.ollama_options)
            rankings = softmax_probabilities(validate_rankings(rankings, session.building), session.softmax_temp)
            return [{"space": self.key_to_guid[r["room"]], "probability": r["probability"],
                     "likelihood": r["likelihood"], "reasoning": r["reasoning"]} for r in rankings]
        return rank

    def run(self, session, interp, user_text):
        """One find_object or go_to_room task. Returns 'bye' or None, like
        SearchSession.run_task."""
        state, args = self.world.copy(), self.args
        if interp["task_type"] == "go_to_room":
            room = self.key_to_guid[interp["resolved_room"]]
            set_task(state, "go_to", target=room, text=user_text)
            task = CompoundTask("go_to", (room,))
            print(f"Task type: go_to_room -> {session._label(interp['resolved_room'])} (HTN)")
        else:
            target = interp["target"] or user_text
            mode = "all" if interp.get("quantity") == "all" else "single"
            restricted = self.key_to_guid.get(interp.get("resolved_room"))
            recorded = [self.key_to_guid[r] for r in interp["rooms_with_target"]]
            set_task(state, "find_object", target, mode, restricted, recorded, user_text)
            task = CompoundTask("find_object", (target, mode))
            print(f"Task type: find_object -> target = '{target}', mode = {mode} (HTN)"
                  + (f", restricted to {session._label(interp['resolved_room'])}" if restricted else "")
                  + (f", recorded in {', '.join(session._label(r) for r in interp['rooms_with_target'])}"
                     if recorded else ""))
            print("At each searched room, give an image path standing in for the camera "
                  "('skip' = no image, 'bye' = end the session).")

        logger = RunLogger(RESULTS_DIR, user_text)
        logger.write_json("connectivity.json", connectivity_report(state, self.stairs))
        executor = PlanExecutor(self.planner, state, BeliefModel(), self.perception, SimulatedNavigation(),
                                self.make_ranker(session), logger, args.miss_rate, args.detect_threshold,
                                args.residual_mass, log=print)
        summary = executor.run(task)
        logger.write_summary({
            "task": {"name": task.name, "args": list(task.args), "text": user_text},
            "robot": self.robot.to_state(self.world.robot["space"]),
            "settings": {**state.settings, "miss_rate": args.miss_rate, "detect_threshold": args.detect_threshold,
                         "residual_mass": executor.residual_mass, "llm_temperature": args.llm_temperature,
                         "seed": args.seed, "model": args.model},
            **summary.to_dict(),
            "found_labels": [executor.label(s) for s in summary.found],
            "visit_order_labels": [executor.label(s) for s in summary.visit_order],
        })
        self.world = executor.state                     # robot position and door states carry over

        where = ", ".join(executor.label(s) for s in summary.found)
        print(f"\nOutcome: {summary.outcome}" + (f" in {where}" if where else "")
              + (f" ({summary.reason})" if summary.reason else "")
              + f"; {len(summary.visit_order)} room(s) searched, {summary.travelled_m} m, "
                f"{summary.iterations} plan(s). Log: {summary.log_dir}")
        self._print_probability(summary, executor)
        visited = ", ".join(executor.label(s) for s in summary.visit_order) or "no room"
        session._remember(user_text, f"{task.name} via HTN: {summary.outcome}"
                          + (f" in {where}" if where else "") + f"; searched {visited}")
        return "bye" if summary.outcome == "aborted" else None

    @staticmethod
    def _print_probability(summary, executor):
        """Where the probability ended up, and every room the robot cannot reach."""
        p = summary.probability
        if p:
            print("Probability (prior / posterior): "
                  + " | ".join(f"{k.replace('_', ' ')} {v['prior']:.3f} / {v['posterior']:.3f} ({v['rooms']} rooms)"
                               for k, v in p.items()))
        if summary.unreachable:
            print(f"Unreachable for this robot ({len(summary.unreachable)} room(s)):")
            for r in summary.unreachable:
                prob = f"p={r['prior']:.3f}  " if r["prior"] is not None else ""
                print(f"  {prob}{executor.label(r['space']):28s} storey {r['storey']}  {r['code']}")
