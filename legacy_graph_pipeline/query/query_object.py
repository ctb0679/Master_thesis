"""
Stage 4: query router + all three query handlers.

Three query types, routed by an LLM call (not keyword rules — those don't
generalize, as the earlier German-keyword classifier proved):
  1. direct_room_lookup    -> match query against actual room names (JSON-based)
  2. factual_count_query   -> Text-to-Cypher against Neo4j (schema from load_graph.py)
  3. object_location_search -> ground-truth Equipment lookup (JSON-based) first,
                                falling back to category-based probabilistic
                                ranking only for objects not actually modeled

Requires (for factual_count_query only): pip install neo4j python-dotenv,
a running Neo4j instance, and the graph already loaded via load_graph.py.

Usage:
    Interactive (recommended):
        python3 query_object.py <extracted_graph.json>

    Single-shot (backward compatible with the old CLI):
        python3 query_object.py <extracted_graph.json> "find the fuse box"
"""

import sys
import os
import re
import json
import requests
from pathlib import Path
from neo4j import GraphDatabase
from dotenv import load_dotenv

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen3.6:27b"  # NEVER ":latest" -- that resolves to the 35B/24GB model and OOMs the 4090

INTENTS = ["direct_room_lookup", "factual_count_query", "object_location_search"]

# -- Neo4j setup (mirrors load_graph.py exactly -- same .env location, same
#    schema assumptions. Connection is lazy: only opened when a
#    factual_count_query actually needs it.) --
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = os.environ.get("NEO4J_USER")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD")

_driver = None


def get_neo4j_driver():
    global _driver
    if _driver is None:
        if not NEO4J_USER or not NEO4J_PASSWORD:
            raise RuntimeError(
                "NEO4J_USER / NEO4J_PASSWORD not found. Check that a .env file "
                "exists at the project root with both set, matching what "
                "docker-compose.yml used to start the container."
            )
        _driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    return _driver


# Cypher keywords that mutate data or the schema. The LLM is asked for
# read-only queries only, but this is enforced in code, not trusted from
# the prompt -- a hallucinated CREATE/DELETE must never reach the driver.
_WRITE_KEYWORDS = re.compile(
    r"\b(CREATE|MERGE|DELETE|DETACH|SET|REMOVE|DROP|LOAD\s+CSV|CALL\s+apoc)\b",
    re.IGNORECASE,
)


def is_write_query(cypher):
    return bool(_WRITE_KEYWORDS.search(cypher))


CYPHER_SYSTEM_PROMPT = """You translate a natural-language question about a building
into a single read-only Cypher query for Neo4j.

Graph schema (this is the ENTIRE schema -- do not invent labels, relationship
types, or properties beyond what is listed here):

Nodes:
  (:Space {guid: string, number: string, function: string,
           function_category: string | null, area_m2: float})
  (:Door {guid: string, name: string})
  (:Window {guid: string, name: string})
  (:Storey {name: string, guid: string})
  (:Equipment {guid: string, name: string, raw_name: string,
               ifc_type: string, object_type: string | null})

Relationships:
  (:Space)-[:ON_STOREY]->(:Storey)
  (:Door)-[:ON_STOREY]->(:Storey)
  (:Door)-[:BOUNDS]->(:Space)
  (:Window)-[:BOUNDS]->(:Space)
  (:Equipment)-[:ON_STOREY]->(:Storey)
  (:Equipment)-[:LOCATED_IN]->(:Space)

Notes:
- "function" is the raw room name/type as authored (may be German, English,
  abbreviated). "function_category" is a normalized category, may be null if
  Stage 2 classification hasn't been run on that space.
- There is no property that means "room type" other than `function` /
  `function_category` -- match against those, don't invent a `type` property.
- Equipment nodes come from IfcFurniture (desks, chairs, closets, shelving,
  tables, printers -- real furniture with proper IFC semantics) and
  IfcBuildingElementProxy (a generic catch-all class used for items the
  authoring tool didn't map to a proper IFC type, e.g. monitors, projection
  screens) in the source IFC model. `ifc_type` records which of the two it
  came from -- this is a technical/provenance detail, NOT what a query
  should filter on. A question like "how many monitors" should match on
  `name`, never on `ifc_type`.
- `name` is a cleaned label (raw name up to the first ':'); `raw_name` is
  the original string and may look like "Family:Type:ElementId" with
  repeated substrings. There is no controlled vocabulary for these names --
  always match with case-insensitive substring search, e.g.
  `WHERE toLower(eq.name) CONTAINS toLower('monitor')`, never exact string
  equality.
- Not every Equipment node has a LOCATED_IN relationship to a Space -- some
  items only have ON_STOREY (their room-level placement wasn't captured in
  the source IFC model). Absence of LOCATED_IN means "room unknown", not
  "doesn't exist" -- don't filter it out unless the query is specifically
  about room-level location.
- Only MATCH/WHERE/RETURN/WITH/ORDER BY/COUNT/etc -- never CREATE, MERGE,
  DELETE, SET, REMOVE, or DROP. This is a read-only query.
- Prefer simple patterns. For existence checks use
  `MATCH (x) WHERE <condition> RETURN count(x) > 0 AS result`, NOT the
  `EXISTS { ... }` subquery form -- that form requires a bound variable
  inside the subquery (`EXISTS { MATCH (e:Equipment) WHERE e.name ... }`,
  never `MATCH (:Equipment) WHERE .name ...` with no variable) and has
  produced syntax errors here. The simple count-based form is just as
  correct and less error-prone.
- The user message below will include the ACTUAL room list (number and
  function) and the ACTUAL unique equipment/furniture item names present in
  this building. ALWAYS match against these real values instead of guessing
  a substring from the question's own wording -- e.g. if the question says
  "workshop" but the real room list only has "Workspace", use "Workspace"
  (the real value) if it's clearly what's meant, or match nothing if it
  genuinely isn't among the real values. For equipment, prefer exact
  equality or `IN [...]` against the given real names over CONTAINS when
  you can identify which real name(s) the question refers to -- e.g. a
  question about "printers" should match the real name that IS a printer
  (a brand/model name you recognize), not a literal CONTAINS 'printer'
  substring search, which will miss items named after their model number.
- ALWAYS generate a fresh query based solely on the CURRENT question below.
  Do not reuse, copy, or adapt a Cypher query from an earlier turn, even if
  an earlier question looks similar (e.g. "how many offices" after "are
  there more than 5 offices" is a DIFFERENT question needing a plain
  count, not the earlier boolean comparison). Only carry over filter terms
  from earlier turns when the current question explicitly references them
  ("that room", "the same floor", "those items").
- You may be given prior turns in this conversation for context (e.g. to
  resolve "now count doors on that floor" referring to a storey named in an
  earlier turn) -- but treat them as background, not a template to copy.

Return your answer as structured JSON with a single Cypher query string."""

CYPHER_SCHEMA = {
    "type": "object",
    "properties": {
        "cypher": {"type": "string"},
    },
    "required": ["cypher"],
}

ANSWER_SYSTEM_PROMPT = """You are given a user's question about a building and the raw
result data that answers it. Write a short, direct, natural-language answer
(1-2 sentences) as if you were simply told the answer -- like talking to a
person, not describing a data structure.

Rules:
- Never mention Cypher, JSON, Neo4j, "intent", "results", "query", databases,
  or any other implementation detail. The person asking doesn't know or care
  how the answer was produced.
- If the data shows zero/none/not found, say that plainly and directly.
- If there's an "error" field, explain the underlying problem in plain
  language and, if it implies an action (e.g. a setup step), say what to do
  -- but still without naming Cypher/Neo4j/JSON.
- Be concise. Don't pad with pleasantries or repeat the question back."""

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
    },
    "required": ["answer"],
}

# Ollama's schema-constrained "format" field is unreliable in practice --
# it's known to sometimes ignore the schema, use different key names, or
# wrap output in markdown fences (see ollama/ollama#8063 and similar). This
# instruction is a second line of defense on top of "format", not a
# replacement for it.
JSON_OUTPUT_INSTRUCTION = (
    "\n\nRespond with ONLY a single valid JSON object matching the schema. "
    "Do not include markdown code fences (no ```), no explanation, and no "
    "text of any kind before or after the JSON object."
)

# How many past (user, assistant) turn-pairs to carry into each new LLM call.
# Kept small on purpose -- this is context for reference resolution
# ("now check the second floor"), not a full transcript.
MAX_HISTORY_TURNS = 6

ROUTER_SYSTEM_PROMPT = """You are routing a natural-language question about a building
to the correct handler, based on the ACTUAL rooms that exist in this building
(given below).

Choose exactly one intent:
- direct_room_lookup: the query is asking to find/locate a room/space itself
  (not a count of rooms, not an object inside a room), and a room matching it
  plausibly exists in the given room list (e.g. "find the lab" when a room
  named "Lab" exists, or "find the washroom" when a WC-type room exists).
- factual_count_query: the query asks for a count, list, or aggregate fact
  about ANYTHING in the building -- rooms, doors, windows, AND equipment/
  furniture/objects (e.g. "how many offices", "list all meeting rooms",
  "how many doors in the basement", "how many monitors are there", "how
  many chairs in room 3"). The deciding signal is the question SHAPE
  ("how many", "list", "count") -- if it's asking for a number or list of
  matching items, it is ALWAYS factual_count_query, regardless of whether
  the thing being counted is a room or an object.
- object_location_search: the query asks WHERE a specific object would
  likely be found -- an open-ended location search, not a count -- e.g.
  "find the fuse box", "find the first aid kit", "where would tools be".
  These objects will never appear as a room name themselves. Only use this
  when the query is asking for a location/likelihood, not a number.

You may be given prior turns in this conversation for context (e.g. to
resolve follow-up queries like "now check the second floor"). Route based on
the CURRENT query, using prior turns only to resolve ambiguous references.

Return your answer as structured JSON."""

ROUTER_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {"type": "string", "enum": INTENTS},
    },
    "required": ["intent"],
}

DIRECT_LOOKUP_SYSTEM_PROMPT = """You are matching a query to actual rooms in a building.
You are given the query and the full list of room names (raw, as authored --
may be in German, English, abbreviated, etc). Decide which room(s), if any,
satisfy the query. If nothing plausibly matches, say found=false and leave
matches empty -- do not force a weak match.

You may be given prior turns in this conversation for context (e.g. to
resolve follow-up queries like "now check the second floor")."""

DIRECT_LOOKUP_SCHEMA = {
    "type": "object",
    "properties": {
        "found": {"type": "boolean"},
        "matches": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["found", "matches"],
}

OBJECT_SEARCH_SYSTEM_PROMPT = """You are estimating, from general world knowledge, how
likely it is that a given object would be found in each of the listed room
types. Score every room type from 0.0 (never found there) to 1.0 (very
likely found there) for the given object. Be specific to the object, not
generic -- a fuse box and a first aid kit have very different likely
locations. This is for a physical search task: a robot will visit rooms in
order of your ranking, so a full, honest distribution across ALL given room
types is more useful than a single confident guess.

The room types given may be normalized categories (e.g. "Office",
"Technical/Utility") or raw, un-normalized room names taken directly from
the building model (e.g. "Printer room", "Corridor", "Lab") -- reason about
whichever kind you're given the same way: using ordinary knowledge of what
that kind of room is typically used for and what's commonly kept there.

Examples of the reasoning expected (illustrative only, not this building):
- Object "fire extinguisher", rooms ["Corridor", "Office 1", "Kitchen",
  "Server room"] -> corridors and kitchens are common extinguisher
  locations by code/convention, server rooms often have their own
  suppression, offices rarely: roughly
  {"Corridor": 0.8, "Kitchen": 0.7, "Server room": 0.5, "Office 1": 0.2}
- Object "first aid kit", rooms ["Lobby", "Workshop", "Storage", "Lab"] ->
  workshops and labs commonly keep first aid kits due to injury risk,
  lobbies sometimes have one at reception, storage rooms rarely:
  {"Workshop": 0.7, "Lab": 0.7, "Lobby": 0.4, "Storage": 0.2}

Score every room type given, even at 0.0 -- don't omit any.

You may be given prior turns in this conversation for context (e.g. to
resolve follow-up queries like "now check the second floor")."""

OBJECT_SEARCH_SCHEMA = {
    "type": "object",
    "properties": {
        "category_scores": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "category": {"type": "string"},
                    "likelihood": {"type": "number"},
                },
                "required": ["category", "likelihood"],
            },
        },
    },
    "required": ["category_scores"],
}

EQUIPMENT_LOOKUP_SYSTEM_PROMPT = """You are matching a request to find a physical object
against the actual furniture/equipment items present in this building (given
below, as a list of item names -- these are the exact names authored in the
source BIM model, may include brand/model names). Use real-world knowledge
to recognize what a brand/model name actually is -- e.g. "Canon iR C2620" is
a printer/copier, "Kinnarps Oberon" is an office desk/table line. Decide
which item name(s), if any, satisfy the request. If nothing in the list
plausibly matches, say found=false and leave matches empty -- do not force a
weak match; this building may genuinely not contain the requested item."""

EQUIPMENT_LOOKUP_SCHEMA = {
    "type": "object",
    "properties": {
        "found": {"type": "boolean"},
        "matches": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["found", "matches"],
}


def _extract_json(content):
    """Best-effort recovery of a JSON object from a chat response that should
    be pure JSON (via 'format') but, on some Ollama/model combinations, comes
    back as free text -- e.g. wrapped in ```json fences, or with leading/
    trailing prose. Raises ValueError with the original content if nothing
    parseable is found."""
    text = content.strip()

    # strip ```json ... ``` or ``` ... ``` fences
    fence_match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # last resort: grab the first balanced {...} span in the text
    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    raise ValueError(content)


def call_llm(system_prompt, history, user_content, schema, _is_retry=False):
    """history is a list of {"role": "user"|"assistant", "content": str} turn
    dicts, shared across intents so later queries can reference earlier
    answers regardless of which handler produced them."""
    messages = (
        [{"role": "system", "content": system_prompt + JSON_OUTPUT_INSTRUCTION}]
        + history
        + [{"role": "user", "content": user_content}]
    )
    resp = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "messages": messages,
            "format": schema,
            "think": False,  # qwen3.6 thinks by default; with structured
                              # output this leaves "content" empty unless
                              # explicitly disabled -- see model card notes
            "options": {"temperature": 0},
            "stream": False,
        },
        timeout=300,
    )
    if not resp.ok:
        raise RuntimeError(
            f"Ollama returned {resp.status_code} for model {MODEL!r}: {resp.text.strip()}"
        )
    content = resp.json()["message"]["content"]

    parse_error = None
    parsed = None
    try:
        parsed = _extract_json(content)
    except ValueError:
        parse_error = f"not valid JSON: {content!r}"
    else:
        missing = [k for k in schema.get("required", []) if k not in parsed]
        if missing:
            parse_error = f"valid JSON but missing required key(s) {missing}: {content!r}"

    if parse_error is None:
        return parsed

    if not _is_retry:
        # one corrective retry: show the model its own bad output and ask
        # again, before giving up. Handles the "format" schema being
        # silently ignored, or the model using different key names, on the
        # first pass.
        corrective = (
            f"Your previous response was invalid -- {parse_error}\n\n"
            f"Respond again with ONLY a single valid JSON object containing "
            f"exactly these required keys: {schema.get('required')}. No "
            "markdown fences, no explanation, no extra text."
        )
        return call_llm(system_prompt, history, corrective, schema, _is_retry=True)

    raise RuntimeError(
        f"Ollama returned invalid output for model {MODEL!r} "
        f"(after one retry) -- {parse_error}"
    )


def route_query(query, spaces, history):
    room_names = sorted(set(s["function"] for s in spaces if s["function"]))
    user_content = f"Rooms in this building:\n{json.dumps(room_names, ensure_ascii=False)}\n\nQuery: {query}"
    result = call_llm(ROUTER_SYSTEM_PROMPT, history, user_content, ROUTER_SCHEMA)
    intent = result.get("intent")
    if intent not in INTENTS:
        raise RuntimeError(f"Router returned an unrecognized intent: {result!r}")
    return intent


# Conversation history has been observed causing this local model to
# generate output about the PREVIOUS turn's topic instead of the current
# question -- e.g. "are there more than 8 offices?" produced a printer-
# lookup Cypher query, because the prior turn's answer happened to mention
# "printer". This isn't limited to raw-JSON history (already fixed once);
# plain-English history triggered it too. Given how costly a silently wrong
# count/answer is compared to losing a "now check the second floor"
# convenience, history is OFF by default per-query and only included when
# the query itself contains an explicit backward-referring cue.
#
# The cue pattern is deliberately narrow (multi-word phrases, not single
# common words). An earlier version matched bare "there"/"it", which fired
# on "is THERE a meeting room" and "are THERE more than 8 offices" -- the
# exact existential phrasing that caused the original bug -- which would
# have included history on precisely the queries that must not get it.
# False negatives here (a genuine follow-up phrased too loosely to match)
# just lose a convenience; false positives reintroduce the actual bug.
_CONTEXT_CUE_PATTERN = re.compile(
    r"\bthat (room|floor|space|item|object|building|one)\b|"
    r"\bthose (rooms|items|objects|spaces|ones)\b|"
    r"\b(the )?same (room|floor|space|one)\b|"
    r"\b(the )?previous (room|floor|question|answer|one)\b|"
    r"\bagain\b|\bearlier\b|\binstead\b",
    re.IGNORECASE,
)


def needs_context(query):
    return bool(_CONTEXT_CUE_PATTERN.search(query))


def handle_direct_lookup(query, spaces, equipment, history):
    room_names = sorted(set(s["function"] for s in spaces if s["function"]))
    user_content = f"Rooms in this building:\n{json.dumps(room_names, ensure_ascii=False)}\n\nQuery: {query}"
    result = call_llm(DIRECT_LOOKUP_SYSTEM_PROMPT, history, user_content, DIRECT_LOOKUP_SCHEMA)
    matches = result.get("matches", [])
    found = result.get("found", bool(matches))

    if not found or not matches:
        return {"intent": "direct_room_lookup", "found": False, "rooms": []}

    matched_spaces = [s for s in spaces if s["function"] in matches]
    return {"intent": "direct_room_lookup", "found": True, "rooms": matched_spaces}


def handle_object_search(query, spaces, equipment, history):
    # Step 1: ground truth first. If the requested object is literally
    # modeled in this building's Equipment data, look it up directly rather
    # than guessing -- a printer that's actually in the BIM model has a
    # known location; probabilistic reasoning is for objects that were
    # never modeled at all (e.g. "first aid kit", "fuse box" -- items these
    # buildings generally don't model as discrete BIM elements).
    unique_names = sorted(set(e["name"] for e in equipment if e.get("name")))
    if unique_names:
        lookup_content = (
            f"Physical items present in this building:\n"
            f"{json.dumps(unique_names, ensure_ascii=False)}\n\nRequest: {query}"
        )
        lookup_result = call_llm(
            EQUIPMENT_LOOKUP_SYSTEM_PROMPT, [], lookup_content, EQUIPMENT_LOOKUP_SCHEMA
        )
        matched_names = set(lookup_result.get("matches", [])) if lookup_result.get("found") else set()

        if matched_names:
            space_by_guid = {s["guid"]: s for s in spaces}
            found_items = []
            for e in equipment:
                if e.get("name") in matched_names:
                    sp = space_by_guid.get(e.get("located_in_space"))
                    found_items.append({
                        "item": e["name"],
                        "room_number": sp["number"] if sp else None,
                        "room_function": sp["function"] if sp else None,
                        "room_known": sp is not None,
                    })
            return {
                "intent": "object_location_search",
                "source": "ground_truth",
                "found_items": found_items,
            }

    # Step 2: the object isn't modeled -- fall back to probabilistic
    # reasoning. Prefer the normalized function_category when Stage 2
    # classification has run (fewer, cleaner categories); otherwise reason
    # directly over the raw `function` field, which is always populated
    # from Stage 1 extraction alone and needs no classification step. This
    # means an unanticipated object query ("first aid kit", or literally
    # anything else not explicitly handled) always gets a real world-
    # knowledge-based estimate, not a hard stop -- classification improves
    # the estimate when available, it was never a hard requirement for one.
    categories = sorted(set(s["function_category"] for s in spaces if s.get("function_category")))
    if categories:
        location_field = "function_category"
        location_values = categories
    else:
        location_field = "function"
        location_values = sorted(set(s["function"] for s in spaces if s.get("function")))

    if not location_values:
        return {
            "intent": "object_location_search",
            "ranked_rooms": [],
            "note": "No room information is available in this building's data to "
                    "reason about.",
        }

    user_content = (
        f"Object to find: {query}\n\n"
        f"Room types present in this building:\n{json.dumps(location_values, ensure_ascii=False)}"
    )
    result = call_llm(OBJECT_SEARCH_SYSTEM_PROMPT, history, user_content, OBJECT_SEARCH_SCHEMA)
    category_scores = result.get("category_scores", [])
    if not category_scores:
        return {"intent": "object_location_search", "ranked_rooms": [],
                "note": f"LLM response had no category_scores: {result!r}"}

    likelihood_by_value = {c["category"]: c["likelihood"] for c in category_scores}

    ranked = []
    for s in spaces:
        val = s.get(location_field)
        if val and val in likelihood_by_value:
            ranked.append({
                "room": s["function"],
                location_field: val,
                "likelihood": likelihood_by_value[val],
            })
    ranked.sort(key=lambda r: r["likelihood"], reverse=True)

    return {
        "intent": "object_location_search",
        "source": "probabilistic",
        "reasoning_basis": location_field,
        "ranked_rooms": ranked,
    }


def _run_cypher(driver, cypher):
    """Returns (records, error_string). Never raises -- execution failures
    are handed back as data so callers can decide whether to retry."""
    try:
        with driver.session() as session:
            return session.run(cypher).data(), None
    except Exception as e:
        return None, str(e)


def handle_factual_count(query, spaces, equipment, history):
    try:
        driver = get_neo4j_driver()
    except RuntimeError as e:
        return {"intent": "factual_count_query", "error": str(e)}

    try:
        with driver.session() as session:
            node_count = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
    except Exception as e:
        return {
            "intent": "factual_count_query",
            "error": f"Could not connect to Neo4j at {NEO4J_URI}: {e}. "
                     "Is the container running (docker compose up -d)?",
        }

    if node_count == 0:
        return {
            "intent": "factual_count_query",
            "error": "Neo4j graph is empty -- run "
                     "'python3 load_graph.py <extracted_graph.json>' first.",
        }

    # Give the generator the ACTUAL room list and ACTUAL unique equipment
    # names, not just the schema shape. Without this it can only guess real
    # values from the question's own wording -- which produced two
    # concrete failures: "workshop" (typo/near-miss for the real room
    # "Workspace") matched nothing since it never saw the real name to
    # reconcile against, and "office 1" got matched against `s.number`
    # (which is really just "13") instead of `s.function` (which really is
    # "Office 1") because it never saw what those fields actually contain.
    room_list = [{"number": s["number"], "function": s["function"]} for s in spaces]
    unique_equipment_names = sorted(set(e["name"] for e in equipment if e.get("name")))
    user_content = (
        f"Rooms in this building (number, function):\n"
        f"{json.dumps(room_list, ensure_ascii=False)}\n\n"
        f"Unique equipment/furniture item names in this building:\n"
        f"{json.dumps(unique_equipment_names, ensure_ascii=False)}\n\n"
        f"Query: {query}"
    )
    result = call_llm(CYPHER_SYSTEM_PROMPT, history, user_content, CYPHER_SCHEMA)
    # defensive: schema enforcement has been unreliable on this Ollama build,
    # so the model has been seen using "query" instead of the required "cypher"
    cypher = result.get("cypher") or result.get("query")
    if not cypher:
        return {
            "intent": "factual_count_query",
            "error": f"LLM response didn't contain a Cypher query under any "
                     f"expected key: {result!r}",
        }

    if is_write_query(cypher):
        return {
            "intent": "factual_count_query",
            "error": "Generated Cypher contained a write/schema-mutating keyword "
                      "and was blocked before execution.",
            "generated_cypher": cypher,
        }

    records, exec_error = _run_cypher(driver, cypher)
    if exec_error is not None and not is_write_query(cypher):
        # one corrective retry: show the model the exact Neo4j error and ask
        # it to fix the query. Handles genuine syntax mistakes (e.g. an
        # unbound EXISTS { } subquery) the same way the JSON-format retry
        # already handles malformed structured output.
        corrective = (
            f"{user_content}\n\n"
            f"Your previous query failed to execute:\n{cypher}\n\n"
            f"Error: {exec_error}\n\n"
            "Write a corrected read-only Cypher query that fixes this error."
        )
        retry_result = call_llm(CYPHER_SYSTEM_PROMPT, history, corrective, CYPHER_SCHEMA)
        retry_cypher = retry_result.get("cypher") or retry_result.get("query")
        if retry_cypher and not is_write_query(retry_cypher):
            cypher = retry_cypher
            records, exec_error = _run_cypher(driver, cypher)
        elif retry_cypher and is_write_query(retry_cypher):
            return {
                "intent": "factual_count_query",
                "error": "Corrected Cypher contained a write/schema-mutating keyword "
                          "and was blocked before execution.",
                "generated_cypher": retry_cypher,
            }

    if exec_error is not None:
        return {
            "intent": "factual_count_query",
            "error": f"Cypher execution failed: {exec_error}",
            "generated_cypher": cypher,
        }

    return {
        "intent": "factual_count_query",
        "generated_cypher": cypher,
        "results": records,
    }


HANDLERS = {
    "direct_room_lookup": handle_direct_lookup,
    "factual_count_query": handle_factual_count,
    "object_location_search": handle_object_search,
}


def synthesize_answer(query, result, history):
    user_content = f"Question: {query}\n\nRaw result data: {json.dumps(result, ensure_ascii=False)}"
    answer_result = call_llm(ANSWER_SYSTEM_PROMPT, history, user_content, ANSWER_SCHEMA)
    return answer_result.get("answer") or json.dumps(result, ensure_ascii=False)


def run_query(query, spaces, equipment, history, debug=False):
    # History is only handed to the LLM calls when the query itself has a
    # backward-referring cue -- see needs_context(). The persisted `history`
    # list in the caller still accumulates every turn regardless, so a
    # later query that DOES reference "that" still has it available.
    effective_history = history if needs_context(query) else []

    intent = route_query(query, spaces, effective_history)
    result = HANDLERS[intent](query, spaces, equipment, effective_history)
    answer = synthesize_answer(query, result, effective_history)

    if debug:
        print(f"[debug] routed as: {intent}")
        print(f"[debug] context used: {bool(effective_history)}")
        print(f"[debug] {json.dumps(result, indent=2, ensure_ascii=False)}")
    print(answer)

    return result, answer


def repl(spaces, equipment):
    print("Interactive query mode -- type 'exit' or 'quit' to stop.")
    print("Type 'debug' to toggle showing the routed intent and raw data.\n")
    history = []
    debug = False
    while True:
        try:
            query = input("query> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not query:
            continue
        if query.lower() in ("exit", "quit"):
            break
        if query.lower() == "debug":
            debug = not debug
            print(f"debug mode {'on' if debug else 'off'}\n")
            continue

        try:
            result, answer = run_query(query, spaces, equipment, history, debug=debug)
        except Exception as e:
            print(f"Error: {e}")
            print()
            continue

        # history keeps the natural-language answer, NOT the raw structured
        # result -- but note this is now only ever fed back to the LLM when
        # a later query contains a context cue (see needs_context).
        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": answer})
        # keep only the last MAX_HISTORY_TURNS turn-pairs
        history[:] = history[-(MAX_HISTORY_TURNS * 2):]
        print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage:')
        print('  Interactive: python3 query_object.py <extracted_graph.json>')
        print('  Single-shot: python3 query_object.py <extracted_graph.json> "your query"')
        sys.exit(1)

    path = sys.argv[1]
    with open(path) as f:
        building = json.load(f)
    spaces = building["spaces"]
    equipment = building.get("equipment", [])

    try:
        if len(sys.argv) >= 3:
            # backward-compatible single-shot mode -- shows full debug output
            # since this path is typically used for scripted testing/validation
            query = sys.argv[2]
            print(f"Query: {query!r}")
            run_query(query, spaces, equipment, history=[], debug=True)
        else:
            repl(spaces, equipment)
    finally:
        if _driver is not None:
            _driver.close()
