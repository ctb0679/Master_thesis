#!/usr/bin/env python3
"""
interactive_search.py — testing-convenience REPL for the full pipeline:
(a0) task interpretation, (a) semantic ranking with softmax probabilities,
(b) optimal path ordering, (c) image-based detection at each stop, in an
interpret -> plan -> navigate -> detect loop. Type a task, get a plan, walk
through it; type 'bye' to end the session.

Flow per input:
  1. The model classifies the text into one intent, using the last few
     exchanges as context so follow-ups like "go there" resolve:
       find_object     - physically search for something
       go_to_room      - navigate to a room (synonyms/typos tolerated)
       building_query  - a question about rooms/equipment, answered from the
                         building model without moving
       none            - chit-chat, robot questions, gibberish, flags; gets a
                         short reply, never a search
  2. find_object: if the user restricted the search to one room, only that
     room is checked. Else if the building model records the target as
     equipment somewhere, the robot goes there first to verify. Only if it
     isn't recorded (or wasn't confirmed there) does it rank every room,
     convert the scores to a softmax probability distribution, and visit the
     most likely rooms until --prob_mass of the probability is covered.
  3. go_to_room: resolves the room and navigates, no detection.
  4. Detection stops the walk as soon as the target is found.

Status: this is a testing tool, not the final robot interface — but it's built
with an OOP skeleton (SearchSession, VisualDetector) on purpose, since it's
expected to grow into that interface later. The class boundaries here
shouldn't need to change when that happens, only gain real navigation/robot-
control internals in place of the manual prompts.

No robot integration yet: "arriving" at a location is simulated — the script
prompts you for an image file path standing in for what the robot's camera
would capture there. This is the one seam that will need to change first when
real camera integration exists; everything else (ranking, planning, detection
call shape) should carry over unchanged.

Reuses load_building / build_room_summaries / call_ollama / validate_rankings /
softmax_probabilities / interpret_task / optimal_visit_order / strip_json_fences
from rank_candidates.py rather than duplicating already-tested logic. rank_candidates.py must be in the same
directory (or on PYTHONPATH).

Usage:
    python3 interactive_search.py --cleaned_json ../../data/cleaned_ifc/synthetic_office_cleaned.json
"""

import argparse
import base64
import json
import sys
from pathlib import Path

from rank_candidates import (
    load_building, build_room_summaries, call_ollama, validate_rankings,
    softmax_probabilities, select_candidates, strip_json_fences,
    get_centroid_lookup, optimal_visit_order, resolve_start_room,
    interpret_task, validate_interpretation, DEFAULT_MODEL,
    DEFAULT_OLLAMA_HOST, DEFAULT_TEMPERATURE, DEFAULT_PROB_MASS,
    DEFAULT_MIN_PROB,
)
import urllib.request


class VisualDetector:
    """Stage (c): sends an image + task context to the VLM, asks whether the
    task's target is visible. Kept as its own class, separate from the
    ranking/planning logic, because it's a genuinely different kind of call
    (image payload, different response shape, different model role) — and
    this is exactly the boundary real camera-capture code will plug into
    later, replacing the manual file-path prompt in SearchSession below."""

    DETECTION_SYSTEM_PROMPT = (
        "You are a visual inspection module for a search robot. You will be "
        "given a photo taken at a specific room during a search task. "
        "Determine whether the task's target is visible in the image. "
        "Respond with ONLY a JSON object, no other text, no markdown fences: "
        '{"found": <true/false>, "confidence": <float 0.0-1.0>, '
        '"description": "<one short sentence on what you see relevant to the task>"}'
    )

    def __init__(self, host, model):
        self.host = host
        self.model = model

    def detect(self, image_path, task_description, room_name):
        path = Path(image_path)
        if not path.exists():
            return {"error": f"image file not found: {image_path}"}

        try:
            image_b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
        except OSError as e:
            return {"error": f"could not read image file: {e}"}

        user_prompt = (
            f"Task: {task_description}\n"
            f"Current location: {room_name}\n"
            f"Does this image show the target of the task?"
        )
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.DETECTION_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt, "images": [image_b64]},
            ],
            "stream": False,
            "think": False,
            "format": "json",
        }
        req = urllib.request.Request(
            f"{self.host}/api/chat", data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                raw = json.loads(resp.read().decode("utf-8"))
            content = raw["message"]["content"]
        except Exception as e:
            return {"error": f"Ollama call failed: {e}"}

        try:
            cleaned = strip_json_fences(content)
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return {"error": f"could not parse model response as JSON: {content[:200]}"}


class SearchSession:
    """Orchestrates one full task: interpret the input (a0), rank candidates
    (a), plan the route (b), then walk the route prompting for an image at
    each stop and running detection (c). Holds building data + model config
    as instance state rather than threading them through every call — this
    is the seam future robot-control code should extend (e.g. subclass or
    inject a real navigation/camera backend), not rebuild from scratch."""

    def __init__(self, cleaned_json_path, start_room, prob_mass, min_prob, temperature, model, host):
        self.building = load_building(cleaned_json_path)
        # Don't assume a room name like "Reception" exists — that's true only
        # for the synthetic office model. Fall back to the first room in this
        # building's own JSON if the requested one isn't there (or none was
        # given), so a new IFC file never hard-fails over this.
        self.start_room = resolve_start_room(self.building, start_room)
        self.prob_mass = prob_mass
        self.min_prob = min_prob
        self.temperature = temperature
        self.history = []  # (user_text, outcome) pairs, fed back to the interpreter
        self.model = model
        self.host = host
        self.centroid_lookup = get_centroid_lookup(self.building)
        self.room_summaries = build_room_summaries(self.building)
        self.long_names = {s["name"]: s.get("long_name") or "" for s in self.building["spaces"]}
        self.detector = VisualDetector(host, model)

    # --- stage (a0): understand the input ---------------------------------

    def interpret(self, user_text):
        """Classify the raw input before doing anything else: is it a task at
        all, what kind, what is the target, and does the building model
        already record where that target is."""
        raw = interpret_task(self.host, self.model, user_text, self.room_summaries, self.history)
        return validate_interpretation(raw, self.building)

    def _remember(self, user_text, outcome):
        self.history.append((user_text, outcome))
        self.history = self.history[-6:]

    # --- stages (a)+(b): rank + plan ---------------------------------------

    def rank_and_plan(self, task_description, exclude=()):
        """Semantic ranking -> softmax probabilities -> threshold -> optimal
        visit order. Split out from run_task so future callers (e.g. a real
        robot loop) can get the plan without the interactive prompts.

        `exclude` lists rooms already searched and ruled out. They are told
        to the model AND removed before the softmax, so the probability mass
        is redistributed over the rooms still in play — otherwise a room the
        IFC data strongly favours would keep ~all the mass even after the
        robot has physically checked it and found nothing."""
        exclude = set(exclude)
        note = None
        if exclude:
            note = ("The robot has ALREADY searched the following room(s) and the target "
                    "was NOT there, so give them 0.0: " + ", ".join(sorted(exclude)) + ".")
        rankings = call_ollama(self.host, self.model, task_description, self.room_summaries, note)
        rankings = validate_rankings(rankings, self.building)
        rankings = [r for r in rankings if r["room"] not in exclude]
        rankings = softmax_probabilities(rankings, self.temperature)

        candidates = select_candidates(rankings, self.prob_mass, self.min_prob)
        plan = optimal_visit_order(candidates, self.centroid_lookup, self.start_room) if candidates else []
        return rankings, plan

    def plan_known_rooms(self, rooms, note):
        """Plan a route through rooms the building model already lists the
        target in — no ranking needed, the IFC data is the evidence."""
        entries = [{"room": r, "likelihood": 1.0, "probability": 1.0, "reasoning": note} for r in rooms]
        return optimal_visit_order(entries, self.centroid_lookup, self.start_room)

    # --- stage (c): walk the plan -------------------------------------------

    def _label(self, room):
        ln = self.long_names.get(room)
        return f"{room} ({ln})" if ln and ln != room else room

    def _print_plan(self, plan, title):
        print(f"\n=== {title} (start = {self._label(self.start_room)}) ===")
        for i, stop in enumerate(plan, 1):
            print(f"  {i}. {self._label(stop['room']):28s} p={stop['probability']:.3f}  "
                  f"+{stop['distance_from_previous_m']}m")

    def walk(self, plan, task_description, detect=True):
        """Visit each stop in order. With detect=True, prompt for an image
        standing in for the camera capture and run detection; stop as soon
        as the target is found. Returns 'found', 'bye' (user ended the whole
        session), or None (walked everything without finding it)."""
        if detect:
            print("\nStarting inspection. At each stop, supply an image path standing "
                  "in for the robot's camera capture ('skip' to move on without one, "
                  "'bye' to end the whole session).")

        for i, stop in enumerate(plan, 1):
            room = stop["room"]
            print(f"\n[{i}/{len(plan)}] Arrived at {self._label(room)} "
                  f"(p={stop['probability']:.3f}: {stop['reasoning']})")
            if not detect:
                continue

            answer = input("  Image path (or 'skip'/'bye'): ").strip()
            if answer.lower() == "bye":
                print("Ending session.")
                return "bye"
            if answer.lower() == "skip" or not answer:
                print(f"  Skipped {room} — no detection performed.")
                continue

            result = self.detector.detect(answer, task_description, self._label(room))
            if "error" in result:
                print(f"  DETECTION ERROR: {result['error']}")
                continue

            verdict = "FOUND" if result.get("found") else "not found"
            print(f"  Detection: {verdict} (confidence={result.get('confidence', '?')})")
            print(f"  {result.get('description', '')}")
            if result.get("found"):
                print(f"\nTarget found in {self._label(room)} after {i} stop(s).")
                return "found"

        print(f"\nPlan complete ({len(plan)} location(s) visited)"
              + (", target not found." if detect else "."))
        return None

    # --- full flow ------------------------------------------------------------

    def run_task(self, user_text):
        """Full interactive flow for one input. Returns 'bye' if the user
        ended the whole session mid-walk, else None."""
        print("\nInterpreting input...")
        interp = self.interpret(user_text)
        kind = interp["task_type"]

        if kind == "none":
            reply = interp["answer"] or ("I can find objects, go to rooms, or answer "
                                         "questions about the building.")
            print(reply)
            self._remember(user_text, f"not a task ({interp['reason']}); replied: {reply}")
            return None

        if kind == "building_query":
            answer = interp["answer"] or "The building model has no information on that."
            print(answer)
            self._remember(user_text, f"answered from building model: {answer}")
            return None

        if kind == "go_to_room":
            room = interp.get("resolved_room")
            if not room:
                print(f"Understood 'go to {interp['target']}', but I can't pin that to a single "
                      f"room in this building ({interp['reason']}). Known rooms: "
                      + ", ".join(self._label(s["name"]) for s in self.building["spaces"]))
                self._remember(user_text, f"could not resolve room '{interp['target']}'")
                return None
            print(f"Task type: go_to_room -> {self._label(room)}")
            plan = self.plan_known_rooms([room], "navigation target")
            self._print_plan(plan, "Navigation plan")
            outcome = self.walk(plan, user_text, detect=False)
            self._remember(user_text, f"navigated to {self._label(room)}")
            return outcome

        # kind == "find_object"
        target = interp["target"] or user_text
        known = interp["rooms_with_target"]
        restricted = interp.get("resolved_room")
        print(f"Task type: find_object -> target = '{target}'"
              + (f", restricted to {self._label(restricted)}" if restricted else ""))

        if restricted:
            plan = self.plan_known_rooms([restricted], f"user asked to check this room for '{target}'")
            self._print_plan(plan, "Visit plan (room specified by user)")
            outcome = self.walk(plan, user_text, detect=True)
            self._remember(user_text, f"searched {self._label(restricted)} for '{target}': "
                           + ("found" if outcome == "found" else "not found"))
            return "bye" if outcome == "bye" else None

        if known:
            print(f"The building model records '{target}' in: "
                  + ", ".join(self._label(r) for r in known)
                  + ". Going there directly to verify.")
            plan = self.plan_known_rooms(known, f"'{target}' recorded here in the IFC model")
            self._print_plan(plan, "Visit plan (from building model)")
            outcome = self.walk(plan, user_text, detect=True)
            if outcome == "found":
                self._remember(user_text, f"found '{target}' at its recorded location")
                return None
            if outcome == "bye":
                return "bye"
            print(f"\n'{target}' was not confirmed at its recorded location(s); "
                  "falling back to semantic search over the remaining rooms.")

        print("\nQuerying VLM for room probabilities...")
        rankings, plan = self.rank_and_plan(user_text, exclude=known)

        print(f"\n=== Room probabilities (softmax, T={self.temperature}) ===")
        for r in rankings:
            print(f"  p={r['probability']:.3f}  score={r['likelihood']:.2f}  "
                  f"{self._label(r['room']):28s} {r['reasoning']}")

        if not plan:
            print(f"\nNo room reached the minimum probability ({self.min_prob}).")
            self._remember(user_text, f"searched for '{target}': no plausible room")
            return None

        covered = sum(s["probability"] for s in plan)
        self._print_plan(plan, f"Visit plan ({len(plan)} room(s) covering {covered:.0%} of the probability)")
        outcome = self.walk(plan, user_text, detect=True)
        visited = ", ".join(self._label(s["room"]) for s in plan)
        self._remember(user_text, f"searched for '{target}' in {visited}: "
                       + ("found" if outcome == "found" else "not found"))
        return "bye" if outcome == "bye" else None


def main():
    parser = argparse.ArgumentParser(description="Interactive interpret->plan->navigate->detect loop.")
    parser.add_argument("--cleaned_json", required=True)
    parser.add_argument(
        "--start_room", default=None,
        help="Room the robot starts in. If omitted, or if the given name "
             "isn't in this building, falls back to the first room listed "
             "in the cleaned JSON (a warning/note is printed either way).",
    )
    parser.add_argument(
        "--prob_mass", type=float, default=DEFAULT_PROB_MASS,
        help=f"Visit the most likely rooms until they cover this much of the "
             f"probability mass (default {DEFAULT_PROB_MASS}). Scales with the "
             f"building: a sharp distribution gives 1-2 rooms, a flat one many.",
    )
    parser.add_argument(
        "--min_prob", "--threshold", dest="min_prob", type=float, default=DEFAULT_MIN_PROB,
        help=f"Never visit a room whose own probability is below this "
             f"(default {DEFAULT_MIN_PROB}). --threshold is accepted as an alias.",
    )
    parser.add_argument(
        "--temperature", type=float, default=DEFAULT_TEMPERATURE,
        help=f"Softmax temperature applied to the model's 0-1 scores (default "
             f"{DEFAULT_TEMPERATURE}). Lower = sharper distribution.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--ollama_host", default=DEFAULT_OLLAMA_HOST)
    args = parser.parse_args()

    session = SearchSession(args.cleaned_json, args.start_room, args.prob_mass,
                            args.min_prob, args.temperature, args.model, args.ollama_host)

    print("Interactive search session. Type a task, or 'bye' to exit.")
    while True:
        task = input("\nTask: ").strip()
        if task.lower() in ("bye", "exit", "quit"):
            print("Goodbye.")
            break
        if not task:
            continue
        if session.run_task(task) == "bye":
            break


if __name__ == "__main__":
    main()
