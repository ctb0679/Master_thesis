#!/usr/bin/env python3
"""rank_candidates.py — Stage (a)+(b), final version (Held-Karp optimal ordering)."""

import argparse
import difflib
import json
import math
import re
import sys
import urllib.request
import urllib.error

DEFAULT_MODEL = "qwen3.6:27b"
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


def chat_json(host, model, system_prompt, user_prompt, validate=None):
    """Send one system+user exchange to Ollama and return the parsed JSON
    object the model replies with. `validate(parsed)` may raise ValueError
    to reject a structurally-wrong answer; on parse/validation failure the
    error is fed back to the model and it is asked again, up to MAX_RETRIES.
    Shared by the ranking call and the task-interpretation call so both get
    the same retry/repair behaviour."""
    url = f"{host}/api/chat"
    messages = [
        {"role": "system", "content": system_prompt},
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
            if not isinstance(parsed, dict):
                raise ValueError("Top-level JSON value must be an object")
            if validate is not None:
                validate(parsed)
            return parsed
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


def call_ollama(host, model, task_description, room_summaries, extra_context=None):
    """Stage (a): ask the model for a raw 0.0-1.0 relevance score per room.
    `extra_context` is an optional sentence prepended to the task, e.g. to
    tell the model which rooms have already been searched and ruled out."""
    user_prompt = f"Task: {task_description}\n\nRooms:\n{room_summaries}"
    if extra_context:
        user_prompt = f"{extra_context}\n\n{user_prompt}"

    def _check(parsed):
        if "rankings" not in parsed or not isinstance(parsed["rankings"], list):
            raise ValueError("Missing or malformed 'rankings' key")

    return chat_json(host, model, SYSTEM_PROMPT, user_prompt, validate=_check)["rankings"]


# --- Task interpretation (stage a0) -----------------------------------------

INTERPRET_SYSTEM_PROMPT = """You are the task-understanding module of an indoor search robot. \
You will be given the raw text a user typed, a JSON list of the rooms in the \
building (each with an exact "room" identifier, its function, and the equipment \
recorded in it by the building model), and possibly the last few exchanges of \
the conversation.

Classify the text into exactly one of these intents:
- "find_object": the user wants the robot to physically locate an object, or \
the source of some condition, by going and looking (e.g. "find the fire \
extinguisher", "look for the printer", "locate the water leak", "check whether \
the coffee machine is in the kitchen").
- "go_to_room": the user wants the robot to move to a specific room, with nothing \
to look for (e.g. "go to the meeting room", "take me to office 2", "go there").
- "building_query": a QUESTION about the building, its rooms or their recorded \
equipment that can be answered from the room list without moving (e.g. "is there \
a storage room?", "how many offices are there?", "what's in the kitchen?", \
"which rooms have monitors?", "where is the printer?", "what rooms are there?"). \
Questions are NOT movement commands: "is there a storage room?" must be answered, \
not driven to. BUT a request to check / verify / confirm / see whether something \
is actually there ("check if there is a projector in the meeting room", "verify \
the printer is in room 1") is find_object, even when the building model records \
it — the user wants it physically verified, not looked up.
- "none": anything else — greetings, small talk, questions about the robot itself, \
thanks, cancellations ("never mind", "stop"), gibberish, command-line flags, or \
text the robot cannot act on. Never invent a task the user did not state.

Field rules:
- "target": for find_object the object (short noun phrase); for go_to_room the \
room phrase the user used; otherwise "".
- "room": a "room" identifier copied EXACTLY from the list, or "". For go_to_room \
it is the single room the user means. For find_object it is set ONLY if the user \
restricted the search to one particular room ("is the coffee machine in the \
kitchen?", "check office 2 for a laptop"); otherwise "". Match room phrases \
against BOTH the identifier and the function text, the way a person would: \
"meeting room" matches a room whose function says "Seminar / Meeting Room", \
"storage room" matches "Storage", "the lobby" matches "Reception / Entrance \
Lobby", and small typos like "kitchn", "storge", "ofice 2" still resolve. Use "" \
if no room is a reasonable match, or if several rooms fit equally ("the office" \
when there are many offices).
- "rooms_with_target" (find_object only): identifiers, copied EXACTLY, of rooms \
whose equipment list explicitly records the target — a printer model number \
counts as a printer. Only use equipment actually listed; never guess from the \
room's function alone. Empty list otherwise.
- "answer": for building_query, a concise answer (1-3 sentences) grounded ONLY \
in the room list, naming rooms as "<identifier> (<function>)". If the data does \
not record what was asked, say so plainly and mention that the user can say \
"find <object>" to have the robot search physically. For none, one short \
friendly sentence: respond to what was said and remind the user the robot can \
find objects, go to rooms, or answer questions about the building. For \
find_object and go_to_room, "".
- "reason": one short sentence explaining the classification.
- Use the recent exchanges to resolve references such as "there", "it", "that \
room", "the same one", "yes do that".

Respond with ONLY a JSON object in this exact shape, no other text, no markdown \
fences:
{"task_type": "<find_object|go_to_room|building_query|none>", "target": "<string>", "room": "<room or empty>", "rooms_with_target": ["<room>", ...], "answer": "<string>", "reason": "<one short sentence>"}
"""

TASK_TYPES = ("find_object", "go_to_room", "building_query", "none")


def format_history(history, max_turns=3):
    """Render the last few (user_text, outcome) pairs for the interpreter so
    follow-ups like 'go there' can be resolved. Empty string if none."""
    if not history:
        return ""
    lines = []
    for user_text, outcome in history[-max_turns:]:
        lines.append(f"- user: {user_text!r}\n  robot: {outcome}")
    return "Recent exchanges (oldest first):\n" + "\n".join(lines)


def interpret_task(host, model, user_text, room_summaries, history=None):
    """Stage (a0): classify the raw user input before doing any ranking, so
    non-tasks are rejected instead of being scored, questions are answered
    instead of driven to, and targets the building model already records
    can be routed to directly. `history` is a list of (user_text, outcome)
    pairs from earlier in the session."""
    parts = []
    hist = format_history(history)
    if hist:
        parts.append(hist)
    parts.append(f"User text: {user_text!r}")
    parts.append(f"Rooms:\n{room_summaries}")
    user_prompt = "\n\n".join(parts)

    def _check(parsed):
        if parsed.get("task_type") not in TASK_TYPES:
            raise ValueError(f"'task_type' must be one of {TASK_TYPES}")
        for key in ("target", "room", "answer"):
            if not isinstance(parsed.get(key, ""), (str, type(None))):
                raise ValueError(f"'{key}' must be a string")
        if not isinstance(parsed.get("rooms_with_target", []), (list, type(None))):
            raise ValueError("'rooms_with_target' must be a list")

    return chat_json(host, model, INTERPRET_SYSTEM_PROMPT, user_prompt, validate=_check)


_ROOM_STOPWORDS = {"the", "a", "an", "room", "area", "space", "to", "go", "into"}


def _norm_tokens(text):
    return [t for t in re.split(r"[^a-z0-9]+", text.lower()) if t and t not in _ROOM_STOPWORDS]


def fuzzy_room_match(query, spaces, min_ratio=0.75):
    """Code-side fallback for resolving a room the user named loosely.
    Compares the query against each room's id, its long name, and each
    '/'-, '-' or ','-separated segment of the long name, using both a
    token-containment test (all meaningful query tokens present) and
    difflib similarity, so 'storage room' -> 'Storage', 'meeting room' ->
    'Seminar / Meeting Room', 'kitchn' -> 'Kitchen'. Returns the room id or
    None when nothing clears min_ratio or the best two candidates belong to
    different rooms with practically the same score (ambiguous)."""
    q_tokens = _norm_tokens(query)
    if not q_tokens:
        return None
    q = " ".join(q_tokens)

    scored = []
    for sp in spaces:
        cands = [sp["name"]]
        ln = sp.get("long_name") or ""
        if ln:
            cands.append(ln)
            cands.extend(seg for seg in re.split(r"[/,\-]", ln) if seg.strip())
        best = 0.0
        for c in cands:
            c_tokens = _norm_tokens(c)
            if not c_tokens:
                continue
            c_norm = " ".join(c_tokens)
            if all(t in c_tokens for t in q_tokens):
                ratio = 1.0
            else:
                # per-token best similarity, so a typo in one word doesn't sink the whole phrase
                per_tok = [max(difflib.SequenceMatcher(None, qt, ct).ratio() for ct in c_tokens)
                           for qt in q_tokens]
                ratio = max(sum(per_tok) / len(per_tok),
                            difflib.SequenceMatcher(None, q, c_norm).ratio())
            best = max(best, ratio)
        scored.append((best, sp["name"]))

    scored.sort(reverse=True)
    top_ratio, top_room = scored[0]
    if top_ratio < min_ratio:
        return None
    if len(scored) > 1 and scored[1][1] != top_room and top_ratio - scored[1][0] < 0.05:
        return None  # two different rooms fit equally badly — don't guess
    return top_room


def validate_interpretation(interp, building):
    """Normalise a raw interpretation: drop room ids the model invented, and
    for go_to_room resolve which room is meant. Resolution order: the id
    the model chose (it sees the room list and handles synonyms/typos the
    way a person would) -> exact id / long-name match -> fuzzy_room_match
    as a code-side safety net."""
    spaces = building["spaces"]
    known = {s["name"] for s in spaces}
    by_lower = {s["name"].lower().strip(): s["name"] for s in spaces}
    for s in spaces:
        if s.get("long_name"):
            by_lower.setdefault(s["long_name"].lower().strip(), s["name"])

    out = {
        "task_type": interp.get("task_type", "none"),
        "target": (interp.get("target") or "").strip(),
        "rooms_with_target": [],
        "answer": (interp.get("answer") or "").strip(),
        "reason": interp.get("reason") or "",
        "resolved_room": None,
    }
    for r in interp.get("rooms_with_target", []) or []:
        r = str(r).strip()
        if r in known:
            out["rooms_with_target"].append(r)
        elif r.lower() in by_lower:
            out["rooms_with_target"].append(by_lower[r.lower()])
        else:
            print(f"WARNING: interpreter returned unknown room '{r}' — dropping.", file=sys.stderr)
    out["rooms_with_target"] = list(dict.fromkeys(out["rooms_with_target"]))  # dedupe, keep order

    model_room = str(interp.get("room") or "").strip()
    if out["task_type"] == "go_to_room":
        # navigation: the model's pick, else the target phrase, else fuzzy
        target = out["target"]
        resolved = None
        if model_room in known:
            resolved = model_room
        elif model_room.lower() in by_lower:
            resolved = by_lower[model_room.lower()]
        elif target in known:
            resolved = target
        elif target.lower().strip() in by_lower:
            resolved = by_lower[target.lower().strip()]
        else:
            resolved = fuzzy_room_match(target, spaces) or fuzzy_room_match(model_room, spaces)
            if resolved:
                print(f"NOTE: fuzzy-matched '{target or model_room}' -> room '{resolved}'.", file=sys.stderr)
        out["resolved_room"] = resolved
    elif out["task_type"] == "find_object" and model_room:
        # search restricted to one room: only trust an id we can verify
        if model_room in known:
            out["resolved_room"] = model_room
        elif model_room.lower() in by_lower:
            out["resolved_room"] = by_lower[model_room.lower()]
        else:
            out["resolved_room"] = fuzzy_room_match(model_room, spaces)
    return out


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


DEFAULT_TEMPERATURE = 0.1


def softmax_probabilities(rankings, temperature=DEFAULT_TEMPERATURE):
    """Turn the model's raw per-room scores into a proper probability
    distribution over rooms: p_i = exp(s_i / T) / sum_j exp(s_j / T).

    The raw scores live in [0, 1], so with T=1 the distribution would be
    nearly flat (exp(1)/exp(0) is only ~2.7x). A temperature of 0.1 spreads
    them out so that a 0.7 vs 0.4 score becomes a ~20x ratio, which matches
    how the scores are meant to be read. Lower T -> sharper, higher T ->
    flatter. Adds a "probability" key to each entry (raw score is kept as
    "likelihood") and returns the list sorted by probability, descending."""
    if not rankings:
        return []
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    scaled = [float(r["likelihood"]) / temperature for r in rankings]
    m = max(scaled)  # subtract max for numerical stability
    exps = [math.exp(s - m) for s in scaled]
    total = sum(exps)
    out = [dict(r, probability=e / total) for r, e in zip(rankings, exps)]
    out.sort(key=lambda r: r["probability"], reverse=True)
    return out


DEFAULT_PROB_MASS = 0.9
DEFAULT_MIN_PROB = 0.01


def select_candidates(rankings, prob_mass=DEFAULT_PROB_MASS, min_prob=DEFAULT_MIN_PROB):
    """Choose which rooms to visit from a probability distribution. Take rooms
    in descending probability until their cumulative mass reaches
    `prob_mass` (so a sharp distribution yields 1-2 rooms and a flat one
    yields many — the selection scales with the building instead of relying
    on a fixed per-room cutoff), and never include a room whose own
    probability is below `min_prob`. `rankings` must already be sorted by
    probability, descending, as softmax_probabilities returns it."""
    chosen, cumulative = [], 0.0
    for r in rankings:
        if r["probability"] < min_prob:
            break
        chosen.append(r)
        cumulative += r["probability"]
        if cumulative >= prob_mass:
            break
    return chosen


def get_centroid_lookup(building):
    return {s["name"]: s["centroid_xy"] for s in building["spaces"]}


def resolve_start_room(building, requested=None):
    """Pick a start room that's guaranteed to exist in this building, instead
    of assuming a name like "Reception" that only happens to exist in the
    synthetic office model. If `requested` is given and is a real room, use
    it as-is; otherwise fall back to the first room listed in the cleaned
    JSON (stable/deterministic) and print a note so the fallback is visible
    rather than silent."""
    spaces = building.get("spaces", [])
    if not spaces:
        print("ERROR: building has no spaces/rooms at all — cannot pick a start room.", file=sys.stderr)
        sys.exit(1)

    room_names = {s["name"] for s in spaces}
    if requested and requested in room_names:
        return requested

    fallback = spaces[0]["name"]
    if requested:
        print(f"WARNING: start room '{requested}' not found in this building — "
              f"falling back to first room '{fallback}'. Pass --start_room to pick a different one "
              f"(known rooms: {sorted(room_names)}).", file=sys.stderr)
    else:
        print(f"NOTE: no --start_room given — using first room in building: '{fallback}'.", file=sys.stderr)
    return fallback


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
