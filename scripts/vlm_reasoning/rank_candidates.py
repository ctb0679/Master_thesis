#!/usr/bin/env python3
"""rank_candidates.py — Stage (a)+(b), final version (Held-Karp optimal ordering)."""

import argparse
import difflib
import json
import math
import os
import re
import sys
import urllib.request
import urllib.error

DEFAULT_MODEL = "qwen3.5:9b"
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
MAX_RETRIES = 4
# Seconds to wait for one Ollama reply. A 9B model on CPU reads ~10 prompt
# tokens/s, so a ~3k-token prompt alone takes ~5 min; override with the
# OLLAMA_TIMEOUT environment variable.
REQUEST_TIMEOUT = float(os.environ.get("OLLAMA_TIMEOUT", "600"))

# Ollama sampling/runtime options sent with every request, so runs are
# reproducible (greedy decoding, fixed seed) and the context window is large
# enough for the full room list instead of Ollama's small default.
DEFAULT_LLM_TEMPERATURE = 0.0
DEFAULT_SEED = 42
DEFAULT_NUM_CTX = 32768        # this machine's Ollama default; set explicitly so runs
                               # elsewhere do not fall back to a 2-4k window
CHARS_PER_TOKEN = 3.0          # conservative prompt-size estimate (measured ~3.4)
DEFAULT_NUM_PREDICT = 512
TOKENS_PER_RANKED_ROOM = 60    # ranking reply budget per room (measured ~35, with margin)
DEFAULT_REPEAT_PENALTY = 1.0


def build_ollama_options(llm_temperature=DEFAULT_LLM_TEMPERATURE, seed=DEFAULT_SEED):
    """The `options` dict passed to every Ollama /api/chat call. Only the
    temperature and seed are exposed on the command line; the rest are fixed."""
    return {
        "temperature": llm_temperature,
        "seed": seed,
        "num_ctx": DEFAULT_NUM_CTX,
        "num_predict": DEFAULT_NUM_PREDICT,
        "repeat_penalty": DEFAULT_REPEAT_PENALTY,
    }


def add_ollama_args(parser):
    """Register the shared --llm_temperature / --seed flags on a parser;
    turn the parsed values into options with build_ollama_options."""
    parser.add_argument(
        "--llm_temperature", type=float, default=DEFAULT_LLM_TEMPERATURE,
        help=f"Sampling temperature for every Ollama call (default "
             f"{DEFAULT_LLM_TEMPERATURE} = greedy/deterministic).",
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help=f"Sampling seed for every Ollama call (default {DEFAULT_SEED}).",
    )

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
    """Load a building for ranking/planning. Accepts both the v1
    `*_cleaned.json` format and the `<name>_llm.json` view written by
    clean_ifc.py; the latter is converted in memory to the v1 shape the rest
    of this module reads (see from_llm_view)."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if "rooms" in data and "spaces" not in data:
        return from_llm_view(data)
    return data


def room_keys(rooms):
    """Map each room's GUID to the identifier the pipeline shows the model
    and the user. That is the room's IFC Name when it is unique in the
    building; duplicates get the function appended, and the GUID as a last
    resort, so two rooms never share an identifier. `rooms` are dicts with
    guid/name/function, as in both <name>_llm.json and <name>_full.json, so
    the ranking side and building_query.py agree on the ids."""
    def base(r):
        return str(r.get("name") or r.get("function") or r["guid"]).strip()

    counts = {}
    for r in rooms:
        counts[base(r)] = counts.get(base(r), 0) + 1
    keys, used = {}, set()
    for r in rooms:
        key = base(r)
        if counts[key] > 1 and r.get("function"):
            key = f"{key} ({r['function']})"
        if key in used:
            key = f"{key} [{r['guid']}]"
        used.add(key)
        keys[r["guid"]] = key
    return keys


def from_llm_view(data):
    """Adapter: <name>_llm.json -> the v1 dict shape (spaces / long_name /
    centroid_xy / equipment[].room as a room name). Rooms are keyed by
    room_keys; the GUID is kept on every record. Equipment whose room is
    unknown is dropped from the per-room lists, as v1 never listed it."""
    keys = room_keys(data["rooms"])
    spaces = [{
        "guid": r["guid"],
        "name": keys[r["guid"]],
        "long_name": r.get("function"),
        "object_type": None,
        "function_source": r.get("function_source"),
        "storey": r.get("storey"),
        "area_m2": r.get("area_m2"),
        "centroid_xy": (r.get("centroid") or [0.0, 0.0])[:2],
    } for r in data["rooms"]]
    equipment = [{
        "guid": e["guid"],
        "name": e["name"],
        "ifc_class": e.get("ifc_class"),
        "type": e.get("type"),
        "room": keys[e["room"]],
        "room_source": e.get("room_source"),
    } for e in data.get("equipment", []) if e.get("room") in keys]
    adjacency = [{**a, "a": keys.get(a["a"], a["a"]), "b": keys.get(a["b"], a["b"])}
                 for a in data.get("adjacency", [])]
    return {"source_format": "llm_view", "meta": data.get("meta", {}),
            "spaces": spaces, "equipment": equipment, "adjacency": adjacency}


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


class LLMError(RuntimeError):
    """A model call that produced no usable answer (Ollama unreachable, timed
    out, or kept replying in the wrong shape). Raised instead of exiting, so
    an interactive session can report it and take the next input."""


def _post(host, path, body, timeout):
    req = urllib.request.Request(
        f"{host}{path}", data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


_MODEL_MAX_CTX = {}


def model_max_context(host, model):
    """The model's trained context length from /api/show (cached), or None."""
    if model not in _MODEL_MAX_CTX:
        limit = None
        try:
            info = _post(host, "/api/show", {"model": model}, 30).get("model_info") or {}
            limit = next((int(v) for k, v in info.items() if k.endswith(".context_length")), None)
        except (urllib.error.URLError, TimeoutError, ValueError, OSError):
            pass
        _MODEL_MAX_CTX[model] = limit
    return _MODEL_MAX_CTX[model]


def fit_context(host, model, messages, options):
    """Options whose num_ctx holds the prompt plus the reply budget. If the
    prompt does not fit, Ollama silently drops its START (the system prompt
    with all instructions), so the window is doubled until it fits, up to
    the model's own maximum."""
    est = sum(len(m.get("content", "")) for m in messages) / CHARS_PER_TOKEN
    need = int(est + options.get("num_predict", DEFAULT_NUM_PREDICT) + 256)
    ctx = options.get("num_ctx", DEFAULT_NUM_CTX)
    if need <= ctx:
        return options
    while ctx < need:
        ctx *= 2
    limit = model_max_context(host, model)
    if limit:
        if need > limit:
            raise LLMError(f"the request (~{int(est)} prompt tokens plus reply) is larger than "
                           f"{model}'s maximum context of {limit} tokens")
        ctx = min(ctx, limit)
    print(f"NOTE: prompt ~{int(est)} tokens; raising num_ctx to {ctx}.", file=sys.stderr)
    return {**options, "num_ctx": ctx}


def close_truncated_json(text):
    """Complete JSON the model stopped early on. qwen3.5:9b under Ollama
    0.30.7 sometimes ends a reply with done_reason "stop" a few characters
    short of valid JSON (e.g. '...}]' without the final '}', or inside the
    last string), even with format="json" or a JSON schema. Closes an open
    string and any open arrays/objects; returns the parsed value, or None
    when the text is cut where it cannot be closed (after a key, mid-number)."""
    stack, in_str, esc = [], False, False
    for ch in text:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]":
            if not stack or stack.pop() != ch:
                return None
    if not stack and not in_str:
        return None                      # nothing to close: a different error
    if in_str:
        text = (text[:-1] if esc else text) + '"'
    else:
        text = text.rstrip().rstrip(",")
    try:
        return json.loads(text + "".join(reversed(stack)))
    except json.JSONDecodeError:
        return None


def chat_json(host, model, system_prompt, user_prompt, validate=None, options=None):
    """Send one system+user exchange to Ollama and return the parsed JSON
    object the model replies with. `validate(parsed)` may raise ValueError
    to reject a structurally-wrong answer; on parse/validation failure the
    error is fed back to the model and it is asked again, up to MAX_RETRIES.
    Shared by every text call (interpretation, ranking, building questions)
    so all get the same retry/repair behaviour. `options` defaults to
    build_ollama_options(). Raises LLMError if no usable answer comes back."""
    options = dict(options if options is not None else build_ollama_options())
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        options = fit_context(host, model, messages, options)
        body = {
            "model": model, "messages": messages, "stream": False,
            "think": False, "format": "json", "options": options,
        }
        try:
            raw = _post(host, "/api/chat", body, REQUEST_TIMEOUT)
            content = raw["message"]["content"]
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:300]
            raise LLMError(f"Ollama returned HTTP {e.code}: {detail}") from e
        except urllib.error.URLError as e:
            raise LLMError(f"could not reach Ollama at {host} — is it running? ({e.reason})") from e
        except (TimeoutError, OSError) as e:
            raise LLMError(f"Ollama did not answer within {REQUEST_TIMEOUT:.0f}s ({e}). On a "
                           f"CPU-only machine raise the OLLAMA_TIMEOUT environment variable.") from e
        except (KeyError, TypeError, json.JSONDecodeError) as e:
            last_error = f"unexpected Ollama response shape: {e}"
            continue

        ctx = options.get("num_ctx", DEFAULT_NUM_CTX)
        if raw.get("prompt_eval_count", 0) + raw.get("eval_count", 0) >= ctx - 8:
            # prompt + reply filled the window: Ollama truncated or shifted the
            # context, so the reply was made without part of the prompt
            options = {**options, "num_ctx": ctx * 2}
            last_error = f"context window of {ctx} tokens overflowed"
            print(f"WARNING: attempt {attempt}: {last_error}, retrying with num_ctx={ctx * 2}...",
                  file=sys.stderr)
            continue

        if raw.get("done_reason") == "length":
            # Cut off at num_predict. Feeding the truncated text back as
            # "invalid JSON" made a greedy model repeat the same cut-off reply;
            # instead retry with a doubled budget (greedy decoding gives the
            # same tokens up to the old cap, so results stay reproducible).
            budget = options.get("num_predict", DEFAULT_NUM_PREDICT)
            options = {**options, "num_predict": budget * 2}
            last_error = f"reply cut off at num_predict={budget} tokens"
            print(f"WARNING: attempt {attempt}: {last_error}, retrying with "
                  f"num_predict={budget * 2}...", file=sys.stderr)
            continue

        cleaned = strip_json_fences(content)
        try:
            try:
                parsed = json.loads(cleaned)
            except json.JSONDecodeError:
                parsed = close_truncated_json(cleaned) if raw.get("done_reason") == "stop" else None
                if parsed is None:
                    raise
                print("NOTE: model stopped just short of valid JSON; closed it.", file=sys.stderr)
            if not isinstance(parsed, dict):
                raise ValueError("Top-level JSON value must be an object")
            if validate is not None:
                validate(parsed)
            return parsed
        except (json.JSONDecodeError, ValueError, KeyError, TypeError, AttributeError) as e:
            last_error = str(e)
            messages.append({"role": "assistant", "content": content})
            messages.append({
                "role": "user",
                "content": f"That was not valid JSON in the required shape ({e}). "
                            f"Respond with ONLY the JSON object, nothing else.",
            })
            print(f"WARNING: attempt {attempt} failed to parse ({e}), retrying...", file=sys.stderr)

    raise LLMError(f"no valid answer from the model after {MAX_RETRIES} attempts "
                   f"(last error: {last_error})")


def _rank_once(host, model, task_description, rooms, extra_context, options):
    user_prompt = f"Task: {task_description}\n\nRooms:\n{json.dumps(rooms)}"
    if extra_context:
        user_prompt = f"{extra_context}\n\n{user_prompt}"

    def _check(parsed):
        if "rankings" not in parsed or not isinstance(parsed["rankings"], list):
            raise ValueError("Missing or malformed 'rankings' key")

    # One ranking entry is ~35 tokens, so the reply grows with the building
    # (IDAC: 15 rooms overran the fixed 512). Scale the budget to the room count.
    options = dict(options or build_ollama_options())
    options["num_predict"] = max(options.get("num_predict", DEFAULT_NUM_PREDICT),
                                 TOKENS_PER_RANKED_ROOM * len(rooms) + 64)
    return chat_json(host, model, SYSTEM_PROMPT, user_prompt, validate=_check,
                     options=options)["rankings"]


def call_ollama(host, model, task_description, room_summaries, extra_context=None, options=None):
    """Stage (a): ask the model for a raw 0.0-1.0 relevance score per room.
    `extra_context` is an optional sentence prepended to the task, e.g. to
    tell the model which rooms have already been searched and ruled out.
    Rooms the model skips are asked about once more on their own, instead of
    silently scoring 0.0 (seen: IDAC room 3, 'Workspace', skipped once)."""
    rooms = json.loads(room_summaries)
    rankings = _rank_once(host, model, task_description, rooms, extra_context, options)
    got = {str(r.get("room", "")).strip().lower() for r in rankings if isinstance(r, dict)}
    missing = [r for r in rooms if str(r["room"]).strip().lower() not in got]
    if missing and len(missing) < len(rooms):
        print(f"NOTE: model skipped {len(missing)} room(s) "
              f"({', '.join(str(r['room']) for r in missing)}); asking for those again.", file=sys.stderr)
        rankings = rankings + _rank_once(host, model, task_description, missing, extra_context, options)
    return rankings


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
- "quantity" (find_object only): "all" if the user wants every instance found \
("find all fire extinguishers", "where are all the chairs"), otherwise "one".
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
{"task_type": "<find_object|go_to_room|building_query|none>", "target": "<string>", "room": "<room or empty>", "rooms_with_target": ["<room>", ...], "quantity": "<one|all>", "answer": "<string>", "reason": "<one short sentence>"}
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


def interpret_task(host, model, user_text, room_summaries, history=None, options=None):
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

    return chat_json(host, model, INTERPRET_SYSTEM_PROMPT, user_prompt, validate=_check,
                     options=options)


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
        "quantity": "all" if str(interp.get("quantity") or "").strip().lower() == "all" else "one",
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
    by_room = {}
    for r in rankings:
        if not isinstance(r, dict):
            print(f"WARNING: model returned a non-object ranking entry {r!r} — dropping.", file=sys.stderr)
            continue
        # models sometimes return the room as a number (1 for "1") or the
        # score as a string ("0.8"); normalise instead of crashing
        raw_name = str(r.get("room", "")).strip()
        try:
            likelihood = min(max(float(r.get("likelihood", 0.0)), 0.0), 1.0)
        except (TypeError, ValueError):
            print(f"WARNING: unreadable likelihood {r.get('likelihood')!r} for '{raw_name}' — using 0.0.",
                  file=sys.stderr)
            likelihood = 0.0
        if raw_name in known_rooms:
            room = raw_name
        elif raw_name.lower() in known_lower:
            room = known_lower[raw_name.lower()]
            print(f"NOTE: auto-corrected case/whitespace mismatch: '{raw_name}' -> '{room}'", file=sys.stderr)
        else:
            print(f"WARNING: model returned unrecognized room name '{raw_name}' — dropping.", file=sys.stderr)
            continue
        entry = {"room": room, "likelihood": likelihood, "reasoning": str(r.get("reasoning") or "")}
        if room not in by_room or likelihood > by_room[room]["likelihood"]:   # duplicates: keep the best
            by_room[room] = entry
    corrected = list(by_room.values())

    seen_rooms = set(by_room)
    missing = known_rooms - seen_rooms
    if missing:
        print(f"WARNING: model omitted {len(missing)} room(s): {sorted(missing)}. Treating as 0.0.", file=sys.stderr)
        for room in missing:
            corrected.append({"room": room, "likelihood": 0.0, "reasoning": "(omitted by model)"})
    return corrected


DEFAULT_SOFTMAX_TEMP = 0.1


def softmax_probabilities(rankings, softmax_temp=DEFAULT_SOFTMAX_TEMP):
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
    if softmax_temp <= 0:
        raise ValueError("softmax_temp must be > 0")
    scaled = [float(r["likelihood"]) / softmax_temp for r in rankings]
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


def optimal_visit_order(candidates, centroid_lookup, start_room, max_exact=18, distance=None):
    """Shortest open tour from start_room over the candidates (Held-Karp).
    `distance(room_a, room_b) -> metres` defaults to the straight line
    between centroid_lookup centroids; the HTN passes door-graph route
    lengths instead."""
    n = len(candidates)
    if n == 0:
        return []
    if distance is None:
        def distance(a, b):
            return euclidean(centroid_lookup[a], centroid_lookup[b])
    if n > max_exact:
        print(f"WARNING: {n} candidates exceeds max_exact={max_exact}, falling back to greedy.", file=sys.stderr)
        return _greedy_nearest_neighbor(candidates, centroid_lookup, start_room, distance)

    start_xy = centroid_lookup.get(start_room)
    if start_xy is None:
        raise ValueError(f"start room '{start_room}' not found or has no centroid")

    rooms = [c["room"] for c in candidates]
    dist_from_start = [distance(start_room, r) for r in rooms]
    dist = [[distance(rooms[i], rooms[j]) for j in range(n)] for i in range(n)]

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
    prev = start_room
    raw_total = 0.0
    for idx in order:
        c = candidates[idx]
        d = distance(prev, rooms[idx])
        raw_total += d
        ordered.append(dict(c, distance_from_previous_m=round(d, 2)))
        prev = rooms[idx]
    assert abs(raw_total - best_cost) < 1e-6, "path reconstruction mismatch"
    return ordered


def _greedy_nearest_neighbor(candidates, centroid_lookup, start_room, distance):
    remaining = list(candidates)
    ordered = []
    current = start_room
    while remaining:
        remaining.sort(key=lambda c: distance(current, c["room"]))
        nxt = remaining.pop(0)
        d = distance(current, nxt["room"])
        ordered.append(dict(nxt, distance_from_previous_m=round(d, 2)))
        current = nxt["room"]
    return ordered
