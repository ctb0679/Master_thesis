# Master Thesis — VLM-Based Object Search in Indoor Environments

This repository implements a pipeline that lets a vision-language model (VLM)
reason about where a target object is likely to be found inside a building,
given only a structured (IFC-derived) description of its rooms and equipment,
then plans an efficient route to check the most likely rooms and visually
confirms the target using camera images.

The pipeline has four stages:

0. **Task interpretation** — the raw input is first classified by a local LLM
   (via [Ollama](https://ollama.com)) into one intent, using the last few
   exchanges as context so follow-ups like "go there" resolve:
   - `find_object`: physically search for something. If the user restricted
     the search to one room ("check the kitchen for a coffee machine") only
     that room is visited. Otherwise, if the building model already records
     the target as equipment somewhere (a printer model number counts as a
     printer) those rooms are visited first, and only if it isn't confirmed
     there does the pipeline fall through to ranking.
   - `go_to_room`: navigate to a room, given by id, long name, synonym, or
     with small typos. Ambiguous requests ("the office" when there are
     several) are refused with the room list.
   - `building_query`: a question about rooms or equipment ("is there a
     storage room?", "how many offices?", "where is the printer?") is answered
     from the building model without moving.
   - `none`: greetings, small talk, questions about the robot, gibberish,
     stray flags. Gets a short reply, never a search.

   `scripts/vlm_reasoning/test_interpreter.py` runs a batch of phrasings
   through this stage (no planning or walking) and flags misclassifications,
   so a new building or a new set of inputs can be checked in one go.
1. **Semantic ranking** — given the task and a JSON summary of every room's
   function and equipment, the LLM assigns a raw 0–1 relevance score to every
   room. Those scores are then converted to a proper probability distribution
   over rooms with a softmax (`p_i = exp(s_i/T) / Σ_j exp(s_j/T)`,
   temperature `T` = 0.1 by default, so the probabilities sum to 1). Rooms
   already searched and ruled out are removed before the softmax so the mass
   is redistributed over the rooms still in play.
2. **Route planning** — the most likely rooms are taken in descending
   probability until they cover `--prob_mass` of the distribution (default
   0.9), skipping any room below `--min_prob` (default 0.01). This scales with
   the building: a sharp distribution yields one or two rooms, a flat one
   many. The chosen rooms are ordered into a visit sequence (optimal for
   small room counts via Held-Karp, greedy nearest-neighbor as a fallback for
   larger buildings), starting from a start room. If `--start_room` is omitted or names a room that doesn't exist
   in the building, the first room listed in the cleaned JSON is used.
3. **Visual confirmation** — at each planned stop, an image (standing in for
   a robot's camera capture) is sent to a vision-capable model, which judges
   whether the target is actually visible.

An earlier iteration of this project (Neo4j graph + Cypher queries) is kept
under [legacy_graph_pipeline/](legacy_graph_pipeline/) for reference but is
no longer the active approach — see [Legacy graph pipeline](#legacy-graph-pipeline-archived) below.

## Repository layout

```
data/
  raw_ifc/            Source .ifc building models
  cleaned_ifc/         Cleaned JSON summaries produced by clean_ifc.py
  images/              Sample photos used as stand-ins for camera captures
scripts/
  generate_synthetic_office_ifc.py   Builds a synthetic test building (IFC4X3)
  ifc_cleaning/
    clean_ifc.py        IFC -> lightweight JSON (spaces, equipment, doors)
  vlm_reasoning/
    rank_candidates.py    Ranking/planning logic (library, used by interactive_search.py)
    rank_candidates2.py   Standalone CLI for stages (a)+(b): rank + plan only
    interactive_search.py Full interactive REPL: interpret -> rank -> plan -> visual detect
    test_interpreter.py   Batch checker for the interpretation stage (no walking)
    old_rank_candidates*.py  Superseded versions, kept for reference
legacy_graph_pipeline/ Archived Neo4j-based approach (extraction, classification, query)
testing_vlm.py          One-off smoke test for a local Ollama vision model
requirements.txt        Actual pipeline dependencies (ifcopenshell, ollama)
venv/                   Local virtual environment (not portable — see Setup)
```

## Prerequisites

- **Python 3.12** (the existing `venv/` was built with 3.12; a fresh clone
  should build its own rather than reuse the committed one — see Setup).
- **[Ollama](https://ollama.com)** running locally (`http://localhost:11434`
  by default), with at least:
  - a text/reasoning model for ranking, e.g. `qwen3.6:27b`
  - a vision-capable model for the detection stage, e.g.
    `qwen3-vl:30b-a3b-instruct`

  Pull whichever models you intend to use:
  ```bash
  ollama pull qwen3.6:27b
  ```
- **ifcopenshell** (only needed for generating/cleaning IFC files, not for
  the ranking/planning/detection scripts, which are stdlib-only).

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

The legacy Neo4j pipeline needs a few more packages if you plan to use it:
`pip install neo4j python-dotenv requests`.

## Usage

### 1. Generate or bring your own IFC model

A synthetic test building (9 rooms, doors, furniture/equipment) can be
generated with:

```bash
python3 scripts/generate_synthetic_office_ifc.py
```

Note: the script currently writes to a hardcoded path
(`/home/claude/synthetic_office.ifc`) — edit the `OUT` variable at the
bottom of the script, or move the file into `data/raw_ifc/` afterwards.

Alternatively, use one of the existing files in `data/raw_ifc/` (e.g.
`synthetic_office.ifc`, `IDAC_Model.ifc`).

### 2. Clean the IFC into a JSON summary

```bash
python3 scripts/ifc_cleaning/clean_ifc.py \
  --input data/raw_ifc/synthetic_office.ifc \
  --output data/cleaned_ifc/synthetic_office_cleaned.json
```

This keeps spatial hierarchy, every `IfcSpace` (with name/function/area/
centroid), every piece of equipment (`IfcFurniture` / `IfcBuildingElementProxy`)
with its containing room, and every door with the rooms it connects — see the
module docstring in `clean_ifc.py` for the full keep/drop rationale.

### 3. Rank candidate rooms and plan a route (one-shot)

```bash
python3 scripts/vlm_reasoning/rank_candidates2.py \
  --cleaned_json data/cleaned_ifc/synthetic_office_cleaned.json \
  --task "Find the source of a water leak in the building" \
  --start_room Reception \
  --threshold 0.3 \
  --output results/leak_plan.json
```

This queries Ollama once for room likelihoods, then prints (and optionally
saves) a proximity-ordered visit plan — no image/detection stage.

### 4. Full interactive plan -> navigate -> detect loop

```bash
python3 scripts/vlm_reasoning/interactive_search.py \
  --cleaned_json data/cleaned_ifc/synthetic_office_cleaned.json \
  --start_room Reception

# or on a building whose room names you don't know yet — the start room
# falls back to the first room in the file:
python3 scripts/vlm_reasoning/interactive_search.py \
  --cleaned_json data/cleaned_ifc/IDAC_model_cleaned.json
```

Type anything at the prompt. Questions about the building are answered in
place; navigation requests drive to the room; search requests either go
straight to rooms where the building model records the target, or rank every
room, convert the scores to softmax probabilities, and plan a route over the
most likely rooms until `--prob_mass` (default 0.9) of the probability is
covered. The walk visits each stop asking for an image path (e.g. one of the
samples in `data/images/`) to stand in for a camera capture and runs visual
detection against it, stopping as soon as the target is found. Type `skip` to
move on without an image, or `bye` to end.

`--temperature` controls how sharp the softmax is (lower = the top-scored room
takes more of the mass); `--min_prob` (alias `--threshold`) drops rooms whose
own probability is negligible.

To check how a batch of phrasings is understood without walking anything:

```bash
python3 scripts/vlm_reasoning/test_interpreter.py \
  --cleaned_json data/cleaned_ifc/IDAC_model_cleaned.json
# or your own list, one per line, optionally "<text> => <expected intent>":
python3 scripts/vlm_reasoning/test_interpreter.py \
  --cleaned_json data/cleaned_ifc/IDAC_model_cleaned.json --inputs phrases.txt
```

`interactive_search.py` imports its ranking/planning functions from
`rank_candidates.py`, so both files must stay in the same directory (or
`rank_candidates.py` must be on `PYTHONPATH`).

## Legacy graph pipeline (archived)

`legacy_graph_pipeline/` contains an earlier approach that extracted IFC data
into a Neo4j graph and queried it with an LLM-assisted Cypher generator. It's
kept for reference and is not part of the active pipeline described above.
It requires Docker (`docker compose up` in that directory, using the
`NEO4J_USER`/`NEO4J_PASSWORD` in `.env.legacy`) plus `neo4j`, `python-dotenv`,
and `requests`.

## Notes

- `testing_vlm.py` is a minimal smoke test for confirming Ollama + a vision
  model are reachable — not part of the pipeline itself.
- `data/images/` holds sample photos (kitchen, offices, reception, seminar
  room) usable as stand-in camera captures when testing `interactive_search.py`.
