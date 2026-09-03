#!/usr/bin/env python3
"""
interactive_search.py — testing-convenience REPL for the full pipeline:
(a) semantic ranking, (b) optimal path ordering, (c) image-based detection at
each stop, in a plan -> navigate -> detect loop. Type a task, get a plan, walk
through it; type 'bye' to end the session.

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
optimal_visit_order / strip_json_fences from rank_candidates.py rather than
duplicating already-tested logic. rank_candidates.py must be in the same
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
    strip_json_fences, get_centroid_lookup, optimal_visit_order,
    DEFAULT_MODEL, DEFAULT_OLLAMA_HOST,
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
    """Orchestrates one full task: rank candidates (a), plan the route (b),
    then walk the route prompting for an image at each stop and running
    detection (c). Holds building data + model config as instance state
    rather than threading them through every call — this is the seam future
    robot-control code should extend (e.g. subclass or inject a real
    navigation/camera backend), not rebuild from scratch."""

    def __init__(self, cleaned_json_path, start_room, threshold, model, host):
        self.building = load_building(cleaned_json_path)
        self.start_room = start_room
        self.threshold = threshold
        self.model = model
        self.host = host
        self.centroid_lookup = get_centroid_lookup(self.building)
        self.detector = VisualDetector(host, model)

    def rank_and_plan(self, task_description):
        """Stages (a)+(b) only — split out from run_task so future callers
        (e.g. a real robot loop) can get the plan without immediately walking
        through interactive prompts."""
        summaries = build_room_summaries(self.building)
        rankings = call_ollama(self.host, self.model, task_description, summaries)
        rankings = validate_rankings(rankings, self.building)
        rankings.sort(key=lambda r: r["likelihood"], reverse=True)

        candidates = [r for r in rankings if r["likelihood"] >= self.threshold]
        plan = optimal_visit_order(candidates, self.centroid_lookup, self.start_room) if candidates else []
        return rankings, plan

    def run_task(self, task_description):
        """Full interactive flow for one task: rank, plan, print, then walk
        the plan prompting for images. Returns 'bye' if the user ended the
        whole session mid-walk, else None."""
        print("\nQuerying VLM for room likelihoods...")
        rankings, plan = self.rank_and_plan(task_description)

        print("\n=== Full semantic ranking ===")
        for r in rankings:
            print(f"  {r['likelihood']:.2f}  {r['room']:12s}  {r['reasoning']}")

        if not plan:
            print(f"\nNo rooms met the likelihood threshold ({self.threshold}).")
            return None

        print(f"\n=== Visit plan (start = {self.start_room}) ===")
        for i, stop in enumerate(plan, 1):
            print(f"  {i}. {stop['room']:12s} likelihood={stop['likelihood']:.2f}  "
                  f"+{stop['distance_from_previous_m']}m")

        print("\nStarting inspection. At each stop, supply an image path standing "
              "in for the robot's camera capture ('skip' to move on without one, "
              "'bye' to end the whole session).")

        for i, stop in enumerate(plan, 1):
            room = stop["room"]
            print(f"\n[{i}/{len(plan)}] Arrived at {room} "
                  f"(likelihood={stop['likelihood']:.2f}: {stop['reasoning']})")
            answer = input("  Image path (or 'skip'/'bye'): ").strip()
            if answer.lower() == "bye":
                print("Ending session.")
                return "bye"
            if answer.lower() == "skip" or not answer:
                print(f"  Skipped {room} — no detection performed.")
                continue

            result = self.detector.detect(answer, task_description, room)
            if "error" in result:
                print(f"  DETECTION ERROR: {result['error']}")
                continue

            verdict = "FOUND" if result.get("found") else "not found"
            print(f"  Detection: {verdict} (confidence={result.get('confidence', '?')})")
            print(f"  {result.get('description', '')}")

        print(f"\nInspection plan complete ({len(plan)} location(s) visited).")
        return None


def main():
    parser = argparse.ArgumentParser(description="Interactive plan->navigate->detect loop.")
    parser.add_argument("--cleaned_json", required=True)
    parser.add_argument("--start_room", default="Reception")
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--ollama_host", default=DEFAULT_OLLAMA_HOST)
    args = parser.parse_args()

    session = SearchSession(args.cleaned_json, args.start_room, args.threshold,
                             args.model, args.ollama_host)

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
