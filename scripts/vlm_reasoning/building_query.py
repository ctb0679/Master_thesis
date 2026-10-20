#!/usr/bin/env python3
"""
building_query.py — answers building_query questions ("how big is the window
in the meeting room?", "what colour are the office chairs?", "how many
monitors are there?", "which is the largest office?") from <name>_full.json.

The full JSON is far too large for the prompt (~3 MB for IDAC), so a question
goes through three steps:

  1. Plan (LLM): the model sees a compact catalogue (rooms, element types
     with counts, colours, materials, numeric fields) and turns the question
     into a structured query: an action (details / list / count / sum /
     average / max / min / unanswerable), a target (rooms or elements),
     filters, and a field. It never has to list objects itself.
  2. Execute (Python): the filters are applied to every record and counts,
     sums and extremes are computed exactly. Only the matching records, cut
     down to the useful fields, are kept.
  3. Answer (LLM): the model phrases an answer from the question and the
     computed result alone, and says so when the model does not record it.

Usage (stand-alone, for checking questions without the REPL):
    python3 building_query.py --full_json ../../data/cleaned_ifc/IDAC_Model_full.json \
        --question "how many office chairs are there?"
    python3 building_query.py --full_json ... --questions qs.txt   # one per line
    add --show_plan to print the plan and the computed result as well
"""

import argparse
import colorsys
import json
import math
import sys
from pathlib import Path

from rank_candidates import (
    chat_json, build_ollama_options, add_ollama_args, format_history, room_keys,
    DEFAULT_MODEL, DEFAULT_OLLAMA_HOST,
)

# Classes with no physical meaning for a question (voids, ports).
SKIP_CLASSES = {"IfcOpeningElement", "IfcVirtualElement", "IfcDistributionPort"}
# Property keys that only repeat the identity of the record.
SKIP_PROPS = {"GlobalId", "Name", "Description", "Tag", "ObjectType"}

MAX_DETAILS = 8        # full records returned for "details"
MAX_ITEMS = 60         # compact records returned for "list" / per-item values
MAX_PROPERTIES = 40    # properties kept per record in "details"

ACTIONS = ("details", "list", "count", "sum", "average", "max", "min", "unanswerable")
TARGETS = ("rooms", "elements")
NUMERIC_FIELDS = {
    "rooms": {
        "area_m2": "m²", "height_m": "m", "volume_m3": "m³", "perimeter_m": "m",
        "length_m": "m", "width_m": "m", "doors": "", "windows": "", "equipment_count": "",
    },
    "elements": {"length_m": "m", "width_m": "m", "height_m": "m"},
}
FILTER_KEYS = ("room_ids", "room_function_contains", "ifc_class", "type_contains",
               "colour", "material_contains")


# --- turning raw records into facts ------------------------------------------

def colour_name(rgb):
    """Plain-language name for an IFC surface colour (RGB in 0..1), so the
    model and the colour filter work with words instead of numbers. Fixed
    rules, so the same colour always gets the same name."""
    r, g, b = (max(0.0, min(1.0, float(c))) for c in rgb[:3])
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    if l < 0.12:
        return "black"
    if l > 0.93:
        return "white"
    hue = h * 360
    if hue < 15 or hue >= 345:
        name = "red"
    elif hue < 40:
        name = "orange"
    elif hue < 70:
        name = "yellow"
    elif hue < 170:
        name = "green"
    elif hue < 200:
        name = "cyan"
    elif hue < 260:
        name = "blue"
    elif hue < 300:
        name = "purple"
    else:
        name = "pink"
    if s < 0.08:
        if l < 0.35:
            return "dark grey"
        return "grey" if l < 0.7 else "light grey"
    if s < 0.25:
        tint = {"red": "reddish", "orange": "brownish", "yellow": "yellowish", "green": "greenish",
                "cyan": "bluish", "blue": "bluish", "purple": "purplish", "pink": "pinkish"}[name]
        return f"{tint} grey"
    if name in ("red", "orange", "yellow") and l < 0.4:
        return "brown"
    if l < 0.3:
        return f"dark {name}"
    if l > 0.75:
        return f"light {name}"
    return name


def short_label(text):
    """'Office Chair:Office Chair' -> 'Office Chair'; Revit library paths
    'A\\B\\Black' -> 'Black'."""
    text = str(text or "").replace("\\", "/").split("/")[-1].strip()
    parts = [p.strip() for p in text.split(":") if p.strip()]
    parts = list(dict.fromkeys(parts))
    return ": ".join(parts)


def type_label(e):
    if e.get("type_object") and e["type_object"].get("name"):
        return short_label(e["type_object"]["name"])
    name = str(e.get("name") or "")
    # Revit instance names are "Family:Type:ElementId"; drop the id
    parts = name.split(":")
    if len(parts) > 1 and parts[-1].strip().isdigit():
        name = ":".join(parts[:-1])
    return short_label(name) or e["ifc_class"]


def material_names(materials):
    if not materials:
        return []
    names = []
    for key in ("constituents", "layers", "profiles"):
        for m in materials.get(key) or []:
            names.append(m.get("material") or m.get("name"))
    if materials.get("kind") == "material":
        names.append(materials.get("name"))
    out = [short_label(n) for n in names if n and "<unnamed>" not in str(n).lower()]
    return list(dict.fromkeys(n for n in out if n))


def surfaces(appearance):
    out, seen = [], set()
    for a in appearance or []:
        if not a.get("rgb"):
            continue
        style = short_label(a.get("name")) or "(unnamed style)"
        entry = (style, colour_name(a["rgb"]))
        if entry not in seen:
            seen.add(entry)
            out.append({"style": entry[0], "colour": entry[1],
                        "rgb": [round(c, 3) for c in a["rgb"][:3]]})
    return out


def _r(v, nd=3):
    return round(v, nd) if isinstance(v, float) else v


def footprint_size(geometry):
    """(length, width, height) in metres. Plan extents are taken in the
    object's own axes (footprint rotated by -yaw) so a rotated desk is not
    reported larger than it is; falls back to the world bounding box."""
    if not geometry:
        return None
    bbox = geometry.get("bbox")
    height = (bbox["max"][2] - bbox["min"][2]) if bbox else None
    pts = []

    def collect(x):
        if isinstance(x, list) and len(x) == 2 and all(isinstance(c, (int, float)) for c in x):
            pts.append(x)
        elif isinstance(x, list):
            for y in x:
                collect(y)

    collect(geometry.get("footprint"))
    if pts:
        a = -math.radians(geometry.get("yaw_deg") or 0.0)
        ca, sa = math.cos(a), math.sin(a)
        xs = [p[0] * ca - p[1] * sa for p in pts]
        ys = [p[0] * sa + p[1] * ca for p in pts]
        dx, dy = max(xs) - min(xs), max(ys) - min(ys)
    elif bbox:
        dx, dy = bbox["max"][0] - bbox["min"][0], bbox["max"][1] - bbox["min"][1]
    else:
        return None
    return round(max(dx, dy), 3), round(min(dx, dy), 3), (round(height, 3) if height is not None else None)


def flat_properties(rec, skip_set_names=()):
    out = {}
    for group in ("psets", "qtos"):
        for set_name, props in (rec.get(group) or {}).items():
            if set_name in skip_set_names or not isinstance(props, dict):
                continue
            for k, v in props.items():
                if k in SKIP_PROPS or v is None or isinstance(v, (dict, list)):
                    continue
                out[f"{set_name}.{k}"] = _r(v)
    return out


class BuildingFacts:
    """Every room and physical element of <name>_full.json as a flat record
    with the fields questions are about: room(s), size, colours, materials,
    properties. Room ids come from rank_candidates.room_keys, so they are
    the same ids the interpreter and ranking use."""

    def __init__(self, full):
        self.room_key = room_keys(full["spaces"])
        rooms_by_guid = {s["guid"]: s for s in full["spaces"]}

        # element -> rooms: containment, door links, and space boundaries
        # (the only link windows/walls/columns have to a room)
        elem_rooms = {}
        for b in full.get("space_boundaries") or []:
            if b.get("space") in self.room_key and b.get("element"):
                elem_rooms.setdefault(b["element"], set()).add(self.room_key[b["space"]])
        neighbours = {k: set() for k in self.room_key.values()}
        for e in (full.get("adjacency") or {}).get("edges", []):
            a, b = self.room_key.get(e["a"], e["a"]), self.room_key.get(e["b"], e["b"])
            for x, y in ((a, b), (b, a)):
                if x in neighbours:
                    neighbours[x].add(y)

        self.elements = []
        for e in full.get("elements") or []:
            if e["ifc_class"] in SKIP_CLASSES:
                continue
            rooms = set(elem_rooms.get(e["guid"], ()))
            if e.get("room") in self.room_key:
                rooms.add(self.room_key[e["room"]])
            for g in (e.get("door") or {}).get("rooms") or []:
                rooms.add(self.room_key.get(g, g))
            attrs = e.get("attributes") or {}
            size = footprint_size(e.get("geometry"))
            length, width, height = size if size else (None, None, None)
            if attrs.get("OverallWidth"):          # doors and windows: use the IFC values
                width = round(attrs["OverallWidth"], 3)
                length = None
            if attrs.get("OverallHeight"):
                height = round(attrs["OverallHeight"], 3)
            surf = surfaces(e.get("appearance"))
            self.elements.append({
                "id": e["guid"],
                "class": e["ifc_class"],
                "type": type_label(e),
                "name": e.get("name"),
                "rooms": sorted(rooms, key=self._room_sort),
                "storey": e.get("storey"),
                "length_m": length, "width_m": width, "height_m": height,
                "position_m": [_r(c) for c in (e.get("geometry") or {}).get("centroid") or []],
                "materials": material_names(e.get("materials")),
                "colours": sorted({s["colour"] for s in surf}),
                "surfaces": surf,
                "door": {k: v for k, v in (e.get("door") or {}).items()
                         if k in ("operation_type", "clear_width_m", "is_external")} or None,
                "properties": flat_properties(e, skip_set_names={e.get("name")}),
            })

        self.rooms = []
        for s in full["spaces"]:
            key = self.room_key[s["guid"]]
            inside = [x for x in self.elements if key in x["rooms"]]
            equip = {}
            for x in inside:
                if x["class"] not in ("IfcWall", "IfcColumn", "IfcSlab", "IfcDoor", "IfcWindow"):
                    equip[x["type"]] = equip.get(x["type"], 0) + 1
            props = flat_properties(s)
            size = footprint_size(s.get("geometry"))
            self.rooms.append({
                "id": key,
                "function": s.get("function"),
                "storey": s.get("storey"),
                "area_m2": _r(s.get("area_m2")),
                "height_m": _r(s.get("clear_height_m")),
                "volume_m3": _r(_first(props, ("Qto_SpaceBaseQuantities.NetVolume",
                                               "Qto_SpaceBaseQuantities.GrossVolume",
                                               "Dimensions.Volume"))),
                "perimeter_m": _r(_first(props, ("Qto_SpaceBaseQuantities.NetPerimeter",
                                                 "Qto_SpaceBaseQuantities.GrossPerimeter",
                                                 "Dimensions.Perimeter"))),
                "length_m": size[0] if size else None,
                "width_m": size[1] if size else None,
                "floor_elevation_m": _r(s.get("floor_elevation_m")),
                "doors": sum(1 for x in inside if x["class"] == "IfcDoor"),
                "windows": sum(1 for x in inside if x["class"] == "IfcWindow"),
                "equipment_count": sum(equip.values()),
                "equipment": equip,
                "connected_to": sorted(neighbours.get(key, ()), key=self._room_sort),
                "properties": props,
            })
        self.functions = {r["id"]: r["function"] for r in self.rooms}

    @staticmethod
    def _room_sort(k):
        return (0, int(k), k) if str(k).isdigit() else (1, 0, str(k))

    def catalogue(self):
        """What the planner sees: the vocabulary of the building, not the
        records. Grows with the number of distinct types, not instances."""
        types = {}
        for x in self.elements:
            t = types.setdefault((x["class"], x["type"]), {"count": 0, "colours": set()})
            t["count"] += 1
            t["colours"].update(x["colours"])
        materials = sorted({m for x in self.elements for m in x["materials"]})
        return {
            "rooms": [{"id": r["id"], "function": r["function"]} for r in self.rooms],
            "element_types": [{"class": c, "type": t, "count": v["count"],
                               "colours": sorted(v["colours"])}
                              for (c, t), v in sorted(types.items())],
            "materials": materials[:80],
            "numeric_fields": {k: list(v) for k, v in NUMERIC_FIELDS.items()},
            "room_record_fields": ["function", "storey", "area_m2", "height_m", "volume_m3",
                                   "perimeter_m", "length_m", "width_m", "floor_elevation_m",
                                   "doors", "windows", "equipment (type counts)", "connected_to",
                                   "properties (IFC property sets)"],
            "element_record_fields": ["class", "type", "rooms", "storey", "length_m", "width_m",
                                      "height_m", "position_m", "materials", "colours",
                                      "surfaces (style + colour per part)", "door",
                                      "properties (IFC property sets)"],
        }

    # --- step 2: execute a plan --------------------------------------------

    def select(self, target, filters):
        f = {k: [str(v).lower().strip() for v in (filters.get(k) or []) if str(v).strip()]
             for k in FILTER_KEYS}

        def any_in(needles, hay):
            hay = [str(h).lower() for h in hay if h]
            return any(n in h for n in needles for h in hay)

        def room_ok(room_ids):
            if f["room_ids"] and not set(room_ids) & set(filters.get("room_ids") or []):
                return False
            if f["room_function_contains"] and not any_in(
                    f["room_function_contains"], [self.functions.get(r) for r in room_ids]):
                return False
            return True

        if target == "rooms":
            return [r for r in self.rooms if room_ok([r["id"]])]
        out = []
        for x in self.elements:
            if (f["room_ids"] or f["room_function_contains"]) and not room_ok(x["rooms"]):
                continue
            if f["ifc_class"] and x["class"].lower() not in f["ifc_class"]:
                continue
            if f["type_contains"] and not any_in(f["type_contains"], [x["type"], x["name"]]):
                continue
            if f["colour"] and not any_in(f["colour"], x["colours"]):
                continue
            if f["material_contains"] and not any_in(
                    f["material_contains"], x["materials"] + [s["style"] for s in x["surfaces"]]):
                continue
            out.append(x)
        return out

    def compact(self, rec, target, field=None):
        if target == "rooms":
            c = {"id": rec["id"], "function": rec["function"]}
        else:
            c = {"type": rec["type"], "class": rec["class"], "rooms": rec["rooms"],
                 "colours": rec["colours"]}
        if field:
            c[field] = rec.get(field)
        return c

    def execute(self, plan):
        action, target = plan["action"], plan["target"]
        if action == "unanswerable":
            return {"matched": 0, "note": "the planner found nothing in the building model that answers this"}
        matched = self.select(target, plan.get("filters") or {})
        result = {"matched": len(matched)}
        field = plan.get("field") or None
        unit = NUMERIC_FIELDS[target].get(field, "") if field else ""

        if action == "details":
            result["records"] = [self._trim(r) for r in matched[:MAX_DETAILS]]
            if len(matched) > MAX_DETAILS:
                result["note"] = f"only the first {MAX_DETAILS} of {len(matched)} matches are shown"
        elif action == "list":
            rows = [self.compact(r, target, field) for r in matched]
            if field:
                rows.sort(key=lambda r: (r.get(field) is None, -(r.get(field) or 0)))
            result["items"] = rows[:MAX_ITEMS]
            if len(rows) > MAX_ITEMS:
                result["note"] = f"only {MAX_ITEMS} of {len(rows)} matches are listed"
        elif action == "count":
            result["count"] = len(matched)
            group = plan.get("group_by") or ""
            if group in ("room", "type"):
                counts = {}
                for r in matched:
                    keys = (r.get("rooms") or [r.get("id")]) if group == "room" else [r.get("type") or r.get("function")]
                    for k in keys or ["(no room)"]:
                        label = f"{k} ({self.functions[k]})" if group == "room" and k in self.functions else k
                        counts[label] = counts.get(label, 0) + 1
                result["count_by_" + group] = counts
                if group == "room" and any(len(r.get("rooms") or []) > 1 for r in matched):
                    result["note"] = "an element on a room boundary (door, window, wall) counts for each room it touches"
            if target == "elements":
                result["colours_of_matches"] = _tally(c for r in matched for c in r["colours"])
        else:  # sum / average / max / min over a numeric field
            have = [r for r in matched if isinstance(r.get(field), (int, float))]
            result.update({"field": field, "unit": unit, "with_value": len(have),
                           "without_value": len(matched) - len(have)})
            if have:
                vals = [r[field] for r in have]
                if action == "sum":
                    result["sum"] = round(sum(vals), 3)
                elif action == "average":
                    result["average"] = round(sum(vals) / len(vals), 3)
                else:
                    best = max(vals) if action == "max" else min(vals)
                    result[action] = best
                    result["records_with_" + action] = [self.compact(r, target, field)
                                                       for r in have if r[field] == best]
                ordered = sorted(have, key=lambda r: r[field], reverse=(action != "min"))
                result["values"] = [self.compact(r, target, field) for r in ordered[:MAX_ITEMS]]
        return result

    @staticmethod
    def _trim(rec):
        rec = dict(rec)
        props = rec.get("properties") or {}
        if len(props) > MAX_PROPERTIES:
            rec["properties"] = dict(list(props.items())[:MAX_PROPERTIES])
        rec.pop("name", None)
        return {k: v for k, v in rec.items() if v not in (None, [], {}, "")}


def _first(d, keys):
    for k in keys:
        if d.get(k) is not None:
            return d[k]
    return None


def _tally(items):
    out = {}
    for i in items:
        out[i] = out.get(i, 0) + 1
    return out


# --- the two LLM calls ---------------------------------------------------------

PLAN_SYSTEM_PROMPT = """You are the query planner of an indoor robot's building \
database. You will be given a user's question about the building, a catalogue \
of what the building model contains (rooms with their ids and functions, \
element types with counts and colours, materials, and the fields each record \
has), and possibly the last few exchanges of the conversation.

Do NOT answer the question. Translate it into one query that a program will \
run over every record. The program does the matching, counting and arithmetic \
exactly, so never try to list or count objects yourself.

Fields:
- "action": one of
  "details" - full records of a few specific rooms or elements (size, colours, \
materials, properties, what a room contains, which rooms it connects to);
  "list"    - which rooms / elements match (optionally with one numeric field);
  "count"   - how many match (optionally "group_by": "room" or "type");
  "sum" | "average" | "max" | "min" - over one numeric "field";
  "unanswerable" - the catalogue clearly has nothing that could answer it.
- "target": "rooms" or "elements" (furniture, equipment, doors, windows, walls, \
columns are elements).
- "filters": all optional, each a list; values within a list are OR-ed, \
different keys are AND-ed, empty lists mean no restriction:
  "room_ids": room ids copied EXACTLY from the catalogue (for elements: the room \
they are in);
  "room_function_contains": words matched case-insensitively against the room \
function, e.g. ["office"] for all offices, ["meeting"] for "Meeting room";
  "ifc_class": exact IFC class names from the catalogue, e.g. ["IfcWindow"];
  "type_contains": words matched case-insensitively against element type names. \
Use the words that actually occur in the catalogue's type names, including \
other languages and brand names: a "chair" is "Office Chair", a "printer" may be \
"Canon_iR", a "cupboard" may be "Closet", a "table" may also be "Desk";
  "colour": colour words from the catalogue (black, white, grey, blue, ...);
  "material_contains": words matched against material and surface-style names.
- "field": a numeric field from catalogue.numeric_fields[target] for \
sum/average/max/min (and optionally list); "" otherwise.
- "group_by": "room", "type" or "".
- "reason": one short sentence.

Rules:
- "how big / what size / dimensions" of a specific object or room -> "details".
- "which room is largest / smallest" -> "max" / "min" with field "area_m2".
- "total area of the offices" -> "sum", target "rooms", field "area_m2".
- Resolve "it", "there", "that room" from the recent exchanges.
- Rooms are identified by their id; match room phrases against BOTH id and \
function the way a person would ("the meeting room" -> the room whose function \
is "Meeting room"; "office 3" is the room whose function is "Office 3", not \
necessarily the room with id "3").
- Prefer a filter that is too broad over one that misses objects.

Respond with ONLY a JSON object in this exact shape, no other text:
{"action": "...", "target": "rooms|elements", "filters": {"room_ids": [], "room_function_contains": [], "ifc_class": [], "type_contains": [], "colour": [], "material_contains": []}, "field": "", "group_by": "", "reason": "..."}
"""

ANSWER_SYSTEM_PROMPT = """You are the voice of an indoor robot answering a \
question about its building. You will be given the question, the query that \
was run over the building model, and the result the program computed.

Rules:
- Use ONLY the result. Copy numbers exactly as given, with their units, \
rounded sensibly (e.g. 2 decimals for metres). Never recompute or estimate.
- Name rooms as "<id> (<function>)".
- If "matched" is 0 or the needed value is missing, say plainly that the \
building model does not record it, and if it is an object, mention the user \
can say "find <object>" to have the robot look physically.
- Colours apply to parts of an object ("surfaces"); say which part has which \
colour when that matters, e.g. a chair with a black frame and a blue-grey seat.
- If the result says only some matches are shown, say the answer covers \
those.
- 1-4 sentences, no lists of raw ids, no GUIDs.

Respond with ONLY a JSON object: {"answer": "<text>"}
"""


class BuildingQueryEngine:
    """plan (LLM) -> execute (Python) -> answer (LLM) over <name>_full.json."""

    def __init__(self, full_json_path, host, model, options=None):
        with open(full_json_path, encoding="utf-8") as f:
            full = json.load(f)
        self.facts = BuildingFacts(full)
        self.host, self.model = host, model
        self.options = options if options is not None else build_ollama_options()
        self.catalogue = json.dumps(self.facts.catalogue(), ensure_ascii=False, separators=(",", ":"))
        self.room_ids = {r["id"] for r in self.facts.rooms}
        self.class_names = {x["class"] for x in self.facts.elements}

    def plan(self, question, history=None):
        # catalogue first: it is the same for every question, so Ollama can
        # reuse the already-processed prefix of the previous planner prompt
        parts = [f"Catalogue:\n{self.catalogue}"]
        hist = format_history(history)
        if hist:
            parts.append(hist)
        parts.append(f"Question: {question!r}")

        def _check(p):
            if p.get("action") not in ACTIONS:
                raise ValueError(f"'action' must be one of {ACTIONS}")
            if p["action"] != "unanswerable" and p.get("target") not in TARGETS:
                raise ValueError(f"'target' must be one of {TARGETS}")
            if not isinstance(p.get("filters", {}), dict):
                raise ValueError("'filters' must be an object")
            for k, v in (p.get("filters") or {}).items():
                if k not in FILTER_KEYS:
                    raise ValueError(f"unknown filter '{k}', allowed: {FILTER_KEYS}")
                if not isinstance(v, list):
                    raise ValueError(f"filter '{k}' must be a list")
            if p["action"] in ("sum", "average", "max", "min"):
                allowed = NUMERIC_FIELDS[p["target"]]
                if p.get("field") not in allowed:
                    raise ValueError(f"'field' for {p['target']} must be one of {list(allowed)}")

        plan = chat_json(self.host, self.model, PLAN_SYSTEM_PROMPT, "\n\n".join(parts),
                         validate=_check, options=self.options)
        return self._clean_plan(plan)

    def _clean_plan(self, plan):
        filters = plan.get("filters") or {}
        ids = []
        by_lower = {r.lower(): r for r in self.room_ids}
        for r in filters.get("room_ids") or []:
            r = str(r).strip()
            if r in self.room_ids:
                ids.append(r)
            elif r.lower() in by_lower:
                ids.append(by_lower[r.lower()])
            else:
                print(f"WARNING: query plan named unknown room '{r}' — dropping.", file=sys.stderr)
        filters["room_ids"] = ids
        cls = {c.lower(): c for c in self.class_names}
        filters["ifc_class"] = [cls.get(str(c).lower(), str(c)) for c in filters.get("ifc_class") or []]
        plan["filters"] = {k: filters.get(k) or [] for k in FILTER_KEYS}
        plan["field"] = plan.get("field") or ""
        plan["group_by"] = plan.get("group_by") or ""
        if plan.get("field") and plan.get("target") in NUMERIC_FIELDS \
                and plan["field"] not in NUMERIC_FIELDS[plan["target"]]:
            plan["field"] = ""
        return plan

    def answer(self, question, history=None, verbose=False):
        """Returns (answer_text, plan, result)."""
        plan = self.plan(question, history)
        result = self.facts.execute(plan)
        if verbose:
            print("PLAN:  " + json.dumps(plan, ensure_ascii=False))
            print("RESULT: " + json.dumps(result, ensure_ascii=False)[:3000])
        user_prompt = (f"Question: {question!r}\n\n"
                       f"Query: {json.dumps(plan, ensure_ascii=False)}\n\n"
                       f"Result: {json.dumps(result, ensure_ascii=False)}")

        def _check(p):
            if not isinstance(p.get("answer"), str) or not p["answer"].strip():
                raise ValueError("'answer' must be a non-empty string")

        reply = chat_json(self.host, self.model, ANSWER_SYSTEM_PROMPT, user_prompt,
                          validate=_check, options=self.options)
        return reply["answer"].strip(), plan, result


def find_full_json(path):
    """<name>_llm.json -> sibling <name>_full.json if it exists, else None."""
    p = Path(path)
    if p.name.endswith("_full.json"):
        return str(p)
    if p.name.endswith("_llm.json"):
        full = p.with_name(p.name[:-len("_llm.json")] + "_full.json")
        if full.exists():
            return str(full)
    return None


def main():
    ap = argparse.ArgumentParser(description="Answer questions about a building from <name>_full.json.")
    ap.add_argument("--full_json", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--question")
    g.add_argument("--questions", help="file with one question per line ('#' = comment)")
    ap.add_argument("--show_plan", action="store_true", help="print the query plan and computed result")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--ollama_host", default=DEFAULT_OLLAMA_HOST)
    add_ollama_args(ap)
    args = ap.parse_args()

    engine = BuildingQueryEngine(args.full_json, args.ollama_host, args.model,
                                 build_ollama_options(args.llm_temperature, args.seed))
    if args.question:
        questions = [args.question]
    else:
        with open(args.questions, encoding="utf-8") as f:
            questions = [l.strip() for l in f if l.strip() and not l.lstrip().startswith("#")]
    for q in questions:
        print(f"\nQ: {q}")
        text, _, _ = engine.answer(q, verbose=args.show_plan)
        print(f"A: {text}")


if __name__ == "__main__":
    main()
