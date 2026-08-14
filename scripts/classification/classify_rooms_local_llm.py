"""
Stage 2: classify room LongName strings into functional categories using a
local Qwen3.6-27B model served via Ollama.

Reads Stage 1 extraction output(s), classifies each *unique* raw room name
once, then writes the category back onto every matching space.

Usage:
    python3 classify_rooms_local_llm.py <extracted1.json> [extracted2.json ...]

Example:
    python3 scripts/classification/classify_rooms_local_llm.py \\
        data/extracted/ac20_extracted.json \\
        data/extracted/fantasy_office_1_extracted.json
"""

import sys
import json
import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen3.6:latest"

CATEGORIES = [
    "Office",
    "Meeting/Seminar",
    "Technical/Utility",
    "Sanitary",
    "Laboratory",
    "Circulation",
    "Storage",
    "Reception/Lobby",
    "Other",
]

SYSTEM_PROMPT = f"""You are classifying room names extracted from architectural BIM/IFC models.
The names may be in German, English, or mixed, and may include occupant names,
floor/location qualifiers, or abbreviations alongside the actual room function.

Classify each room name into exactly one of these categories:
{", ".join(CATEGORIES)}

You will be given a list of N room names. You must classify ALL of them and
return exactly N results, one per input, in the same order. Do not skip,
merge, summarize, or truncate the list."""

RESPONSE_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "input": {"type": "string"},
            "category": {"type": "string", "enum": CATEGORIES},
        },
        "required": ["input", "category"],
    },
}

BATCH_SIZE = 15


def classify_batch(names):
    schema = dict(RESPONSE_SCHEMA)
    schema["minItems"] = len(names)
    schema["maxItems"] = len(names)

    resp = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": "Classify these " + str(len(names)) + " room names:\n"
                    + json.dumps(names, ensure_ascii=False),
                },
            ],
            "format": schema,
            "options": {"temperature": 0},
            "stream": False,
        },
        timeout=300,
    )
    resp.raise_for_status()
    content = resp.json()["message"]["content"]
    parsed = json.loads(content)

    if len(parsed) != len(names):
        raise ValueError(f"Expected {len(names)} results, got {len(parsed)}.\nRaw: {content}")
    return parsed


def classify_unique_names(unique_names):
    all_results = []
    names = sorted(unique_names)
    for i in range(0, len(names), BATCH_SIZE):
        batch = names[i:i + BATCH_SIZE]
        print(f"Classifying batch {i // BATCH_SIZE + 1} ({len(batch)} rooms)...")
        all_results.extend(classify_batch(batch))
    return {r["input"]: r["category"] for r in all_results}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 classify_rooms_local_llm.py <extracted1.json> [extracted2.json ...]")
        sys.exit(1)

    paths = sys.argv[1:]
    buildings = {}
    unique_names = set()

    for p in paths:
        with open(p) as f:
            buildings[p] = json.load(f)
        for s in buildings[p]["spaces"]:
            if s["function"]:
                unique_names.add(s["function"])

    print(f"{len(unique_names)} unique room names across {len(paths)} file(s)\n")

    name_to_category = classify_unique_names(unique_names)

    for p, building in buildings.items():
        for s in building["spaces"]:
            s["function_category"] = name_to_category.get(s["function"], "Other")
        with open(p, "w") as f:
            json.dump(building, f, indent=2, ensure_ascii=False)
        print(f"Updated {p} with function_category for {len(building['spaces'])} spaces")

    print(f"\n{'Input':35s} -> Category")
    print("-" * 55)
    for name in sorted(name_to_category):
        print(f"{name:35s} -> {name_to_category[name]}")
