#!/usr/bin/env python3
"""
rank_candidates.py — Stage (a)+(b) of the plan/navigate/detect/update loop.

(a) Semantic candidate ranking: sends the cleaned building JSON + a natural-
    language task description to the VLM (text-only at this stage — no image)
    and asks it to assign a likelihood score to EVERY room, not just a binary
    include/exclude list. This directly implements the thesis's required
    "probabilistic reasoning framework... ranking candidate locations based
    on likelihood" (task description section 3), not just a filter.

(b) Proximity ordering: among rooms above a likelihood threshold, orders them
    into a visit sequence via greedy nearest-neighbor from a fixed start
    point (Reception), using the centroid_xy data clean_ifc.py already
    extracts. This is a heuristic, not an optimal TSP solve — fine for the
    small room counts here (9-15 rooms); flag if candidate counts grow large
    enough that greedy nearest-neighbor's known suboptimality starts to
    matter.

Standalone: only stdlib + a running local Ollama server. No imports from
clean_ifc.py or any other pipeline stage — takes a cleaned JSON file path as
input, same decoupling contract as the cleaning script.

Usage:
    python3 rank_candidates.py \
        --cleaned_json ../../data/cleaned_ifc/synthetic_office_cleaned.json \
        --task "Find the source of a water leak in the building" \
        --start_room Reception \
        --threshold 0.3
"""

import argparse
import json
import math
import sys
import urllib.request
import urllib.error

DEFAULT_MODEL = "qwen3-vl:30b-a3b-instruct-q4_K_M"
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
MAX_RETRIES = 3

SYSTEM_PROMPT = """You are a spatial reasoning module for an indoor search robot. \
You will be given a JSON list of rooms in a building — each with a "room" \
field (the exact identifier), its function, and the equipment inside it — \
and a search task. For EVERY room in the list, assign a likelihood score \
from 0.0 to 1.0 that the target of the task would be found in that room, \
based on the room's function and equipment — not just the room name. \
Include rooms with likelihood 0.0 if they are clearly irrelevant; do not \
omit any room.

CRITICAL: the "room" value in your response must be copied EXACTLY, \
character-for-character, from the "room" field you were given — not the \
function, not a paraphrase, not the function appended to the name.

Respond with ONLY a JSON object in this exact shape, no other text, no \
markdown fences:
{"rankings": [{"room": "<room field copied exactly>", "likelihood": <float 0.0-1.0>, "reasoning": "<one short sentence>"}]}
"""


def load_building(path):
    with open(path) as f:
        return json.load(f)


def build_room_summaries(building):
    """Compact per-room JSON block for the prompt. Structured as JSON with an
    explicit 'room' field — not free-text bracket notation like
    "Kitchen [Kitchen / Break Room]" — because that earlier format caused the
    model to copy the whole bracketed line back as the room identifier
    (confirmed failure mode, not hypothetical: every single room got dropped
    on the first real test run because of exactly this ambiguity).

    Deliberately omits centroid_xy — proximity is a separate, later stage;
    giving raw coordinates to the LLM at the semantic-ranking step risks it
    treating spatial closeness as a relevance signal, which conflates two
    things that should stay independent (a room can be near and still wrong,
    or far and still correct)."""
    equip_by_room = {}
    for item in building.get("equipment", []):
        equip_by_room.setdefault(item["room"], []).append(item["name"])

    rooms = []
    for space in building["spaces"]:
        name = space["name"]
        func = space["long_name"] or space["object_type"] or "(unspecified function)"
        equip = equip_by_room.get(name, [])
        rooms.append({"room": name, "function": func, "equipment": equip})
    return json.dumps(rooms, indent=None)


def strip_json_fences(text):
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```", 2)
        text = text[1] if len(text) > 1 else text[0]
        if text.lstrip().startswith("json"):
            text = text.lstrip()[4:]
    text = text.strip()
    # fallback: extract from first '{' to last '}' if there's leading/trailing prose
    if not text.startswith("{"):
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            text = text[start:end + 1]
    return text


def call_ollama(host, model, task_description, room_summaries):
    url = f"{host}/api/chat"
    user_prompt = f"Task: {task_description}\n\nRooms:\n{room_summaries}"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        body = {
            "model": model,
            "messages": messages,
            "stream": False,
            "think": False,   # Qwen3 models: must be explicit, defaults vary by build
            "format": "json",  # hint only — not fully reliable, hence retry loop below
        }
        req = urllib.request.Request(
            url, data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                raw = json.loads(resp.read().decode("utf-8"))
            content = raw["message"]["content"]
        except urllib.error.URLError as e:
            print(f"ERROR: could not reach Ollama at {host} — is it running? ({e})", file=sys.stderr)
            sys.exit(1)
        except (KeyError, json.JSONDecodeError) as e:
            last_error = f"Unexpected Ollama response shape: {e}"
            continue

        cleaned = strip_json_fences(content)
        try:
            parsed = json.loads(cleaned)
            if "rankings" not in parsed or not isinstance(parsed["rankings"], list):
                raise ValueError("Missing or malformed 'rankings' key")
            return parsed["rankings"]
        except (json.JSONDecodeError, ValueError) as e:
            last_error = str(e)
            # corrective retry: tell the model exactly what was wrong with its last output
            messages.append({"role": "assistant", "content": content})
            messages.append({
                "role": "user",
                "content": f"That was not valid JSON in the required shape ({e}). "
                            f"Respond with ONLY the JSON object, nothing else.",
            })
            print(f"WARNING: attempt {attempt} failed to parse ({e}), retrying...", file=sys.stderr)

    print(f"ERROR: failed to get valid JSON from the model after {MAX_RETRIES} attempts. "
          f"Last error: {last_error}", file=sys.stderr)
    sys.exit(1)


def validate_rankings(rankings, building):
    """Cross-check the model's response against the actual room list — flag
    rather than silently patch, per the 'no info loss / no silent guessing'
    rule used throughout this pipeline. Includes a conservative fuzzy-match
    fallback (case/whitespace only) for near-misses, since prompt-format
    mismatches have already been observed once in practice — better to
    recover an unambiguous near-miss than drop a whole room's data."""
    known_rooms = {s["name"] for s in building["spaces"]}
    known_lower = {r.lower().strip(): r for r in known_rooms}

    corrected = []
    for r in rankings:
        raw_name = r.get("room", "")
        if raw_name in known_rooms:
            corrected.append(r)
        elif raw_name.lower().strip() in known_lower:
            fixed = known_lower[raw_name.lower().strip()]
            print(f"NOTE: auto-corrected case/whitespace mismatch: "
                  f"'{raw_name}' -> '{fixed}'", file=sys.stderr)
            corrected.append({**r, "room": fixed})
        else:
            print(f"WARNING: model returned unrecognized room name '{raw_name}' "
                  f"— no confident match, dropping this entry.", file=sys.stderr)

    seen_rooms = {r["room"] for r in corrected}
    missing = known_rooms - seen_rooms
    if missing:
        print(f"WARNING: model omitted {len(missing)} room(s) from its ranking: "
              f"{sorted(missing)}. Treating as likelihood 0.0 — verify this is correct.",
              file=sys.stderr)
        for room in missing:
            corrected.append({"room": room, "likelihood": 0.0, "reasoning": "(omitted by model)"})

    return corrected


def get_centroid_lookup(building):
    return {s["name"]: s["centroid_xy"] for s in building["spaces"]}


def euclidean(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def order_by_proximity(candidates, centroid_lookup, start_room):
    """Greedy nearest-neighbor from start_room through all candidates.
    Not optimal TSP — acceptable at these room counts, re-evaluate if
    candidate lists grow large."""
    start_xy = centroid_lookup.get(start_room)
    if start_xy is None:
        print(f"ERROR: start room '{start_room}' not found in building, or has no centroid.",
              file=sys.stderr)
        sys.exit(1)

    remaining = list(candidates)
    ordered = []
    current_xy = start_xy
    while remaining:
        remaining.sort(key=lambda c: euclidean(current_xy, centroid_lookup[c["room"]]))
        nxt = remaining.pop(0)
        dist = euclidean(current_xy, centroid_lookup[nxt["room"]])
        nxt = dict(nxt, distance_from_previous_m=round(dist, 2))
        ordered.append(nxt)
        current_xy = centroid_lookup[nxt["room"]]
    return ordered


def main():
    parser = argparse.ArgumentParser(description="Rank and order candidate rooms for a search task.")
    parser.add_argument("--cleaned_json", required=True)
    parser.add_argument("--task", required=True, help="Natural language task description")
    parser.add_argument("--start_room", default="Reception")
    parser.add_argument("--threshold", type=float, default=0.3,
                         help="Minimum likelihood to be included in the visit plan")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--ollama_host", default=DEFAULT_OLLAMA_HOST)
    parser.add_argument("--output", default=None, help="Optional path to save the plan as JSON")
    args = parser.parse_args()

    building = load_building(args.cleaned_json)
    summaries = build_room_summaries(building)

    print(f"Task: {args.task}")
    print(f"Model: {args.model}")
    print("Querying VLM for room likelihoods...\n")

    rankings = call_ollama(args.ollama_host, args.model, args.task, summaries)
    rankings = validate_rankings(rankings, building)
    rankings.sort(key=lambda r: r["likelihood"], reverse=True)

    print("=== Full semantic ranking ===")
    for r in rankings:
        print(f"  {r['likelihood']:.2f}  {r['room']:12s}  {r['reasoning']}")

    candidates = [r for r in rankings if r["likelihood"] >= args.threshold]
    if not candidates:
        print(f"\nNo rooms met the likelihood threshold ({args.threshold}). "
              f"Nothing to order or visit.")
        sys.exit(0)

    centroid_lookup = get_centroid_lookup(building)
    plan = order_by_proximity(candidates, centroid_lookup, args.start_room)

    print(f"\n=== Visit plan (threshold >= {args.threshold}, start = {args.start_room}) ===")
    for i, stop in enumerate(plan, 1):
        print(f"  {i}. {stop['room']:12s} likelihood={stop['likelihood']:.2f}  "
              f"+{stop['distance_from_previous_m']}m  — {stop['reasoning']}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump({"task": args.task, "full_ranking": rankings, "visit_plan": plan}, f, indent=2)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
