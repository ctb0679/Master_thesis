#!/usr/bin/env python3
"""
test_interpreter.py — batch checker for the task-understanding stage (a0).

Runs many inputs through interpret_task + validate_interpretation WITHOUT
planning or walking, and prints one line per input showing how it was
classified. Use it to sanity-check a new building or a new set of phrasings
in one go instead of typing them into the REPL one at a time.

Usage:
    python3 test_interpreter.py --cleaned_json ../../data/cleaned_ifc/IDAC_model_cleaned.json
    python3 test_interpreter.py --cleaned_json ... --inputs my_phrases.txt   # one input per line
    python3 test_interpreter.py --cleaned_json ... --expect                  # also check built-in expectations

Optional expectations: a line may be written as
    <input text>  =>  <expected intent>
in the inputs file; mismatches are flagged and counted. The built-in list
carries expectations for the generic phrasings that should hold for any
building.
"""

import argparse
import sys
import time

from rank_candidates import (
    load_building, build_room_summaries, interpret_task, validate_interpretation,
    DEFAULT_MODEL, DEFAULT_OLLAMA_HOST,
)

# (input, expected intent or None). Generic enough to hold for any building.
BUILTIN = [
    # --- not tasks ---
    ("hello", "none"),
    ("thanks", "none"),
    ("what can you do?", "none"),
    ("what is your name?", "none"),
    ("never mind", "none"),
    ("--start_room ''", "none"),
    ("asdf qwer", "none"),
    ("tell me a joke", "none"),
    # --- questions about the building (answer, don't move) ---
    ("is there a storge room?", "building_query"),
    ("what rooms are there?", "building_query"),
    ("how many offices are there?", "building_query"),
    ("what's in the kitchen?", "building_query"),
    ("which rooms have monitors?", "building_query"),
    ("where is the printer?", "building_query"),
    ("does the building have a lab?", "building_query"),
    ("is there a fire extinguisher anywhere?", "building_query"),
    # --- navigation ---
    ("go to the meeting room", "go_to_room"),
    ("take me to the storage room", "go_to_room"),
    ("navigate to the lobby", "go_to_room"),
    ("go to the kitchn", None),             # typo test; go_to_room only if the building has a kitchen
    ("move to office 2", "go_to_room"),
    ("go to the cafeteria", None),          # building-dependent: go_to_room w/o room, or none
    ("go to the office", None),             # ambiguous in most buildings
    # --- physical search ---
    ("find the fire extinguisher", "find_object"),
    ("find the printer", "find_object"),
    ("look for a first aid kit", "find_object"),
    ("locate the source of the water leak", "find_object"),
    ("search for my laptop", "find_object"),
    ("can you find the coffee machine", "find_object"),
    ("check if there is a projector in the meeting room", "find_object"),
    ("is the coffee machine in the kitchen?", None),   # query or restricted search both defensible
    ("find the meeting room", "go_to_room"),
]


def load_inputs(path):
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=>" in line:
                text, expected = line.rsplit("=>", 1)
                items.append((text.strip(), expected.strip() or None))
            else:
                items.append((line, None))
    return items


def main():
    ap = argparse.ArgumentParser(description="Batch-check task interpretation.")
    ap.add_argument("--cleaned_json", required=True)
    ap.add_argument("--inputs", help="file with one input per line (optional '=> expected_intent')")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--ollama_host", default=DEFAULT_OLLAMA_HOST)
    args = ap.parse_args()

    building = load_building(args.cleaned_json)
    summaries = build_room_summaries(building)
    long_names = {s["name"]: s.get("long_name") or "" for s in building["spaces"]}

    def label(room):
        ln = long_names.get(room)
        return f"{room} ({ln})" if ln and ln != room else (room or "-")

    items = load_inputs(args.inputs) if args.inputs else BUILTIN
    mismatches = 0
    for text, expected in items:
        t0 = time.time()
        r = validate_interpretation(interpret_task(args.ollama_host, args.model, text, summaries), building)
        dt = time.time() - t0
        ok = "  " if expected is None or r["task_type"] == expected else "!!"
        if ok == "!!":
            mismatches += 1
        print(f"{ok} {dt:4.1f}s {text!r:50s} -> {r['task_type']:15s}"
              f" target={r['target']!r:22s} room={label(r.get('resolved_room')):28s}"
              f" rooms_with_target={r['rooms_with_target']}")
        extra = r["answer"] if r["task_type"] in ("building_query", "none") else r["reason"]
        print(f"          {extra}")
        if ok == "!!":
            print(f"          EXPECTED {expected}")

    print(f"\n{len(items)} inputs, {mismatches} mismatch(es) against expectations.")
    sys.exit(1 if mismatches else 0)


if __name__ == "__main__":
    main()
