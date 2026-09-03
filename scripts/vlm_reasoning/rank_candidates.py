#!/usr/bin/env python3
"""rank_candidates.py — Stage (a)+(b), final version (Held-Karp optimal ordering)."""

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
            "model": model, "messages": messages, "stream": False,
            "think": False, "format": "json",
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
    known_rooms = {s["name"] for s in building["spaces"]}
    known_lower = {r.lower().strip(): r for r in known_rooms}
    corrected = []
    for r in rankings:
        raw_name = r.get("room", "")
        if raw_name in known_rooms:
            corrected.append(r)
        elif raw_name.lower().strip() in known_lower:
            fixed = known_lower[raw_name.lower().strip()]
            print(f"NOTE: auto-corrected case/whitespace mismatch: '{raw_name}' -> '{fixed}'", file=sys.stderr)
            corrected.append({**r, "room": fixed})
        else:
            print(f"WARNING: model returned unrecognized room name '{raw_name}' — dropping.", file=sys.stderr)

    seen_rooms = {r["room"] for r in corrected}
    missing = known_rooms - seen_rooms
    if missing:
        print(f"WARNING: model omitted {len(missing)} room(s): {sorted(missing)}. Treating as 0.0.", file=sys.stderr)
        for room in missing:
            corrected.append({"room": room, "likelihood": 0.0, "reasoning": "(omitted by model)"})
    return corrected


def get_centroid_lookup(building):
    return {s["name"]: s["centroid_xy"] for s in building["spaces"]}


def euclidean(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def optimal_visit_order(candidates, centroid_lookup, start_room, max_exact=18):
    n = len(candidates)
    if n == 0:
        return []
    if n > max_exact:
        print(f"WARNING: {n} candidates exceeds max_exact={max_exact}, falling back to greedy.", file=sys.stderr)
        return _greedy_nearest_neighbor(candidates, centroid_lookup, start_room)

    start_xy = centroid_lookup.get(start_room)
    if start_xy is None:
        print(f"ERROR: start room '{start_room}' not found or has no centroid.", file=sys.stderr)
        sys.exit(1)

    xy = [centroid_lookup[c["room"]] for c in candidates]
    dist_from_start = [euclidean(start_xy, p) for p in xy]
    dist = [[euclidean(xy[i], xy[j]) for j in range(n)] for i in range(n)]

    FULL = (1 << n) - 1
    INF = float("inf")
    dp = [[INF] * n for _ in range(1 << n)]
    parent = [[-1] * n for _ in range(1 << n)]
    for i in range(n):
        dp[1 << i][i] = dist_from_start[i]
    for mask in range(1 << n):
        for i in range(n):
            if dp[mask][i] == INF or not (mask & (1 << i)):
                continue
            for j in range(n):
                if mask & (1 << j):
                    continue
                new_mask = mask | (1 << j)
                cost = dp[mask][i] + dist[i][j]
                if cost < dp[new_mask][j]:
                    dp[new_mask][j] = cost
                    parent[new_mask][j] = i

    best_end = min(range(n), key=lambda i: dp[FULL][i])
    best_cost = dp[FULL][best_end]
    order = []
    mask, i = FULL, best_end
    while i != -1:
        order.append(i)
        prev = parent[mask][i]
        mask ^= (1 << i)
        i = prev
    order.reverse()

    ordered = []
    prev_xy = start_xy
    raw_total = 0.0
    for idx in order:
        c = candidates[idx]
        d = euclidean(prev_xy, xy[idx])
        raw_total += d
        ordered.append(dict(c, distance_from_previous_m=round(d, 2)))
        prev_xy = xy[idx]
    assert abs(raw_total - best_cost) < 1e-6, "path reconstruction mismatch"
    return ordered


def _greedy_nearest_neighbor(candidates, centroid_lookup, start_room):
    start_xy = centroid_lookup[start_room]
    remaining = list(candidates)
    ordered = []
    current_xy = start_xy
    while remaining:
        remaining.sort(key=lambda c: euclidean(current_xy, centroid_lookup[c["room"]]))
        nxt = remaining.pop(0)
        d = euclidean(current_xy, centroid_lookup[nxt["room"]])
        ordered.append(dict(nxt, distance_from_previous_m=round(d, 2)))
        current_xy = centroid_lookup[nxt["room"]]
    return ordered
