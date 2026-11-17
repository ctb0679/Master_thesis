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
     storage room?", "how many offices?", "where is the printer?", "how big
     are the meeting-room windows?", "what colour are the chairs?") is answered
     from the building model without moving. With a `<name>_full.json` this
     goes through `building_query.py`: the LLM turns the question into a
     structured query, Python runs it over every record (exact counts, sums,
     min/max), and the LLM phrases the answer from that result only.
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
  cleaned_ifc/         JSON produced by clean_ifc.py (<name>_full.json, <name>_llm.json;
                       *_cleaned.json are v1 outputs still read by vlm_reasoning/)
  images/              Sample photos used as stand-ins for camera captures
  <name>_preplanned.json  Pre-planned navbim map (copied from the Ubuntu PC), see section 6
  distances/           Walking-distance tables made from it (.json for code, .csv for Excel)
  robot/
    urdf/              Robot descriptions (*.urdf); a300.urdf = Clearpath A300 (expanded xacro)
    profiles/          <robot>.json derived from the URDF (width, capabilities, evidence); edit to correct
    door_states_IDAC.json  Assumed initial door states (all open + per-door overrides)
scripts/
  generate_synthetic_office_ifc.py   Builds a synthetic test building (IFC4X3)
  ifc_cleaning/
    clean_ifc.py        CLI: IFC -> <name>_full.json + <name>_llm.json
    make_grid.py        CLI: <name>_full.json -> nav2 occupancy grid (PNG + YAML) per storey
    ifc_extract/        Extractor package (audit, units, relations, geometry, exporter,
                        rooms_doors, obstacles, extractor); see its __init__.py
  navbim/
    navbim_distance_table.py  CLI: pre-planned navbim map JSON -> room-to-room walking
                        distance table (option A door to door, option B centre to centre)
    navbim_room_walks.py  Option B helper: walk from a room centre to its doors (room outline)
  vlm_reasoning/
    rank_candidates.py    Ranking/planning logic (library, used by interactive_search.py)
    search_planners/      Room ordering: held_karp.py (default), blr.py (BestLocalRatio),
                          selector.py (--search_algorithm); see its README.md
    rank_candidates2.py   Standalone CLI for stages (a)+(b): rank + plan only
    interactive_search.py Full interactive REPL: interpret -> rank -> plan -> visual detect
    htn/                  HTN planning layer (core framework, object-search domain, executor);
                          see docs/htn_design.md
    building_query.py     Answers building questions from <name>_full.json (plan -> compute -> answer)
    test_interpreter.py   Batch checker for the interpretation stage (no walking)
    old_rank_candidates*.py  Superseded versions, kept for reference
file_converters/        Standalone IFC <-> ifcJSON converters (ifc2json.py, json2ifc.py)
results/                Output location for saved plans; results/htn_runs/ = one folder per HTN run
docs/                   htn_design.md (HTN design + decisions), references/ (papers)
tests/htn/              pytest suite for the HTN layer (no Ollama needed; test_walking.py = navbim distances,
                        test_search_algorithm.py = --search_algorithm in the HTN)
tests/search_planners/  Held-Karp regression (baseline from before the refactor), BLR, selector, --no-htn
tests/navbim/           pytest suite for the walking-distance tables (no ROS needed)
navbim_setup_log.md     Every navbim setup change on the Ubuntu PC (what / why / undo)
legacy_graph_pipeline/ Archived Neo4j-based approach (extraction, classification, query)
testing_vlm.py          One-off smoke test for a local Ollama vision model
requirements.txt        Actual pipeline dependencies (ifcopenshell, ollama)
requirements-dev.txt    Test dependencies (pytest); pytest.ini sets the test path
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
- **ifcopenshell**, **shapely** and **numpy**: ifcopenshell only for generating/cleaning
  IFC files; shapely and numpy also for the HTN layer (approach points, URDF meshes).
  The `--no-htn` ranking/planning/detection path is stdlib-only.

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

### 2. Extract the IFC into JSON

```bash
python3 scripts/ifc_cleaning/clean_ifc.py \
  --input data/raw_ifc/IDAC_Model.ifc \
  --output-dir data/cleaned_ifc
```

Writes two files, named after the input:

- `<name>_full.json`: every rooted IFC entity except a short blacklist
  (annotations, grids, structural analysis), keyed by GlobalId. Top-level keys:
  `meta, frame, units, spatial, storeys, spaces, elements, types, groups,
  space_boundaries, adjacency, obstacles, audit`. Each record has its
  attributes, psets/qtos (with type inheritance), type object, resolved type,
  materials, colours, classifications, folded relations and geometry
  (centroid, 3D bbox, 2D footprint, yaw). All values are in m / m² / m³ /
  degrees whatever the file's units. Works on IFC2X3, IFC4 and IFC4X3.
- `<name>_llm.json`: compact view with `rooms` (function plus where it came
  from), `equipment` (room plus whether it was found by relation or geometry)
  and `adjacency` (room pairs via doors/virtual openings; external doors go to
  the node `"outside"`).

Doors with no space-boundary link and equipment with no spatial relation to a
room are placed geometrically (door normal probes / point-in-polygon), and this
is recorded. The `audit` key (also printed) counts classes, dropped psets,
geometry failures, rooms with no function field, doors/equipment placed only
geometrically and every caught exception. Use it as the model-quality measure.

| Flag | Meaning |
|---|---|
| `--input` | source `.ifc` (required) |
| `--output-dir` | where to write the JSON (default `.`) |
| `--no-llm-view` | write only `<name>_full.json` |
| `--drop-pset NAME` | drop a pset/qto by exact name (repeatable); the permanent list is `DROP_PSET_EXACT` in the script, initially empty. Pick names from `audit.pset_names` |
| `--threads N` | geometry threads (default: all cores) |

**Compatibility:** `rank_candidates.py`, `interactive_search.py` and
`test_interpreter.py` take either a `<name>_llm.json` or a v1
`*_cleaned.json` as `--cleaned_json`. `load_building()` converts the `_llm`
view in memory to the v1 shape (`from_llm_view`); rooms are identified by
their IFC `Name`, or by name plus function if two rooms share a name
(`room_keys`). `<name>_full.json` is not meant for the ranking prompt (3 MB
for IDAC); it is used by `building_query.py` and `make_grid.py`.

`clean_ifc.py` is a thin entry point; the logic is in `scripts/ifc_cleaning/ifc_extract/`
(module map in its `__init__.py`). It must stay next to `clean_ifc.py`.

Optional occupancy grids for nav2 (one PNG + YAML per storey, or one whole-model
grid if the model has no storeys):

```bash
python3 scripts/ifc_cleaning/make_grid.py \
  --full-json data/cleaned_ifc/IDAC_Model_full.json --resolution 0.05
```

Free = room and door footprints (254), occupied = floor-level obstacles (0),
unknown = everything else (128, which nav2 reads as unknown; the usual 205 would
load as free with `free_thresh: 0.25`). `--max-cells` (default 50M per storey)
refuses grids blown up by outlying geometry such as a Revit survey point, and
names the outlying objects. `--output-dir` defaults to the JSON's directory.

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

# new extractor output; building questions and the HTN use IDAC_Model_full.json alongside it.
# Search and navigation are planned by the HTN layer (section 5), which needs --miss_rate:
python3 scripts/vlm_reasoning/interactive_search.py \
  --cleaned_json data/cleaned_ifc/IDAC_Model_llm.json --miss_rate 0.2

# the v1 *_cleaned.json files have no door positions: use the old search there
python3 scripts/vlm_reasoning/interactive_search.py \
  --cleaned_json data/cleaned_ifc/synthetic_office_cleaned.json --no-htn

# order the ranked rooms with BestLocalRatio instead of Held-Karp (default);
# works with and without --no-htn, needs room areas
python3 scripts/vlm_reasoning/interactive_search.py \
  --cleaned_json data/cleaned_ifc/IDAC_Model_llm.json --miss_rate 0.2 --search_algorithm blr
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

`--softmax_temp` controls how sharp the softmax is (lower = the top-scored room
takes more of the mass); `--min_prob` (alias `--threshold`) drops rooms whose
own probability is negligible.

Every Ollama call (interpretation, ranking, visual detection) uses the same
sampling options so runs are reproducible: `--llm_temperature` (default 0.0,
greedy decoding) and `--seed` (default 42). The context window
(`num_ctx` 32768), `num_predict` (512) and `repeat_penalty` (1.0) are fixed in
`rank_candidates.py` (`build_ollama_options`).

All text calls go through `chat_json`, which keeps one input from ever ending
the session:

- `num_ctx` is raised (doubling, up to the model's own maximum from
  `/api/show`) when a prompt would not fit; Ollama would otherwise silently
  drop the start of the prompt, i.e. the instructions. A request larger than
  the model's maximum is refused with a clear message.
- The ranking call sets `num_predict` to 60 tokens per room (+64); a reply cut
  off at the budget (`done_reason: length`) is retried with the budget doubled.
- Replies the model ended a few characters short of valid JSON (seen with
  `qwen3.5:9b`, even with `format: json`) are closed and parsed instead of
  retried; anything else invalid is sent back to the model (up to 4 attempts).
- Rooms the ranking skips are asked about again in a short follow-up call.
- Failures raise `LLMError`; `interactive_search.py` reports them (and any
  unexpected error) and takes the next input. Ctrl-C cancels the current task;
  Ctrl-D / EOF ends the session.

When `--cleaned_json` is a `<name>_llm.json`, building questions are
answered from the `<name>_full.json` next to it (or from `--full_json`).
That covers sizes of rooms, windows, doors and furniture, colours (named from
the IFC RGB values, per part), materials, property-set values, what connects
to what, and exact counts / totals / largest / smallest. Without a full JSON
the interpreter answers from the room summary, as before. To try questions
on their own:

```bash
python3 scripts/vlm_reasoning/building_query.py \
  --full_json data/cleaned_ifc/IDAC_Model_full.json \
  --question "how many office chairs are in the meeting room?" --show_plan
# or --questions qs.txt (one per line); --show_plan prints the query and the computed result
```

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

### 5. HTN planning layer (default for search and navigation)

With a `<name>_llm.json` + `<name>_full.json`, `find_object` and `go_to_room`
are planned by a hierarchical task network (`scripts/vlm_reasoning/htn/`,
design and decisions in [docs/htn_design.md](docs/htn_design.md)), following
Tandon et al. (ECPPM 2026): a custom rule-based planner with operator and
method libraries, depth-first decomposition with backtracking, explicit
infeasibility reasons and JSON plans of `{"action", "robot", "args"}`.

- **Methods** for `find_object`, in priority order: the room the user named →
  rooms where the IFC records the target → the most likely rooms from the
  ranking model (asked once per task, only when needed).
- **Routes** go through doors (shortest path by length through door
  positions). The **visit order** is Held-Karp (default) or BLR
  (`--search_algorithm`, below) over the distance chosen with `--distance`:
  - `navbim` (default): real walking distances, room centre to room centre,
    from navbim's table `data/distances/<building>_preplanned_centre_walking_distances.json`
    (section 6; other file: `--distance_table`). For a pair whose navbim route
    has a door this robot cannot pass (closed, too narrow, outside), the
    `door_graph` length is used instead. Without a table for the building,
    the whole run uses `door_graph` and says so at start-up.
  - `door_graph`: route through the door positions in straight lines.
  - `euclidean`: straight line between the room centres.
- **Visit order formula, Held-Karp** (`--search_algorithm held_karp`, default;
  `search_planners/held_karp.py`, re-exported as `rank_candidates.optimal_visit_order`): with
  start room s, chosen rooms 1..n and distance d, find the order
  r1, ..., rn that minimises d(s, r1) + d(r1, r2) + ... + d(r(n-1), rn)
  (an open path: no return to s). Held-Karp computes this exactly:
  C({j}, j) = d(s, j) and C(S, j) = min over i in S without j of
  C(S without j, i) + d(i, j), where C(S, j) is the shortest walk from s that
  visits exactly the rooms in S and ends in j; the answer is min over j of
  C(all, j). Up to 18 rooms (n²·2ⁿ steps); above that greedy nearest
  neighbour. The order is made again after every searched room. Which rooms
  are in the list comes from the probabilities only, not from distance.
- **BLR** (`--search_algorithm blr`; `search_planners/blr.py`): BestLocalRatio
  (Izquierdo-Cordova et al., RoboCup 2016, Eq. 1): from the current room take
  the room with the highest P / (d · √A) (P probability, d distance, A floor
  area `area_m2`), continue from there. Same candidate rooms and same
  distances as Held-Karp; needs an area for every room (refused at start-up
  otherwise). Rooms recorded in the IFC are still ordered by Held-Karp (they
  have no probabilities yet). Details, conventions for zero distance and ties:
  [scripts/vlm_reasoning/search_planners/README.md](scripts/vlm_reasoning/search_planners/README.md).
- Each plan line shows the metres per leg and in total, and the run ends with
  "walked X m (navbim distances)"; `summary.json` has `distance`, `walking_m`
  and `walking_legs`.
- **Robot**: `data/robot/urdf/*.urdf` (`--robot <stem>` when there are several).
  Width and capabilities are derived once into `data/robot/profiles/<stem>.json`;
  a door is passable when it is at least width + 0.1 m wide, open (or the robot
  has `manipulate_door`) and not to the outside. Door states:
  `data/robot/door_states_IDAC.json` or `--door_states` (assumed values).
- **Single storey**: the robot plans only within its current storey (no stair
  or lift traversal). Rooms on other storeys, behind doors it cannot pass, or
  not connected by doors are unreachable; they keep their probability and are
  listed with the reason (`other_storey`, `door_too_narrow`,
  `missing_capability`, `exterior_not_allowed`, `unreachable_space`).
- **Connectivity report**: printed once at start-up and saved as
  `connectivity.json` in every run folder: groups of rooms connected through
  interior doors, rooms reachable from the start room, stairs not linked to any
  space, rooms with no door. Missing connections are never inferred.
- **Replanning**: after each searched room the belief is updated
  (a miss multiplies that room's probability by `--miss_rate`, then all rooms
  are renormalised) and the plan is made again. It stops when the target is
  found (`found` and confidence >= `--detect_threshold`, default 0.0), when no
  room is left (`--max_visits_per_room`, default 1), or when the unsearched
  reachable rooms hold less than `--residual_mass` (default 1 − `--prob_mass`).
- **Outcome**: `found`, `not_found_reachable_exhausted` (reachable rooms
  searched, but unreachable rooms still hold more than `--residual_mass`),
  `not_found`, `infeasible: <code>`, `reached` (go_to) or `aborted`. The final
  message and `summary.json` give the prior/posterior mass in reachable rooms
  searched, reachable rooms not searched and unreachable rooms.
- `--miss_rate` (the detector's P(not detected | present)) is required and has
  no default; it is recorded in each run's `summary.json`.
- Every plan is logged to `results/htn_runs/<timestamp>_<task>/iter_<nn>.json`
  (state, prior, posterior, chosen method, decomposition tree with rejected
  methods and reasons, plan, executed actions, observation), plus `summary.json`.
- The room probabilities are printed with the model's score and one-line
  reasoning per room, as in `--no-htn`, and saved as `rankings` in `summary.json`.
- `--no-htn` keeps the previous `SearchSession` behaviour as the evaluation baseline.

### 6. Walking distances from navbim (any IFC model)

Straight lines between room centres ignore walls. [navbim](https://github.com/janstueh/navbim)
(GPP-BIM, ROS 2 Jazzy, runs on the Ubuntu PC) plans real robot paths inside
every room on maps made from the IFC. Its "pre-planning" stores the length of
every door-to-door walk inside each room in the navbim map JSON. From that file
`scripts/navbim/navbim_distance_table.py` builds a table of walking distances
between all rooms, on Windows, without ROS.

Two ways to measure "room A to room B":

| `--method` | Distance | Note |
|---|---|---|
| `door` (option A, default) | nearest door of A → … → nearest door of B | only navbim's own planned lengths; two rooms sharing a door get **0 m** |
| `centre` (option B) | centre of A → door → … → door → centre of B | adds the walk inside A and B, computed from the room outlines in the same JSON (5 cm grid, 0.2 m robot clearance, walls respected); `--check` compares this method with navbim's own door-to-door walks (IDAC_Model: mean 0.15 m off, at most 0.4 m) |

**Steps for a new IFC model** (`<Name>` = the IFC file name without `.ifc`, no spaces).
Setup details, fixes and the build command are in [navbim_setup_log.md](navbim_setup_log.md).

On the **Ubuntu PC** (one terminal, in `~/Junaidali/navbim_ws`, after `source install/setup.bash`):

1. Put the IFC into navbim's bringup package and rebuild that package (the
   launch and the RViz BIM panel only find models there):
   ```bash
   cp <path>/<Name>.ifc src/navbim/navbim_bringup/bim/<Name>.ifc
   colcon build --symlink-install --packages-select navbim_bringup
   source install/setup.bash
   ```
2. Switch on automatic pre-planning in `src/navbim/navbim_bringup/params/navbim_params.yaml`,
   section `gpp_bim`: `pre_plan_paths: true` and `force_pre_plan_of_planned_paths: true`
   (no rebuild needed; this replaces the `ros2 action send_goal /pre_plan_edges ...`
   command, which hangs from a second terminal on this PC, see log B6).
3. Launch:
   ```bash
   ros2 launch navbim_bringup bringup_launch.py ifc:=<Name>
   ```
   The first time it prints `Navigation model folder not found: <path>` and
   builds the navigation model into that `<path>` (room/door graph + one map per
   room; takes a few minutes), then starts navigation. About 15 s later it
   pre-plans; wait for `Successfully pre-planned all <N> transition edges`.
   Optional: check paths in RViz (BIM panel → `<Name>`, "2D Pose Estimate", then
   "2D Goal Pose"). Then stop with Ctrl+C.
4. Set both pre-plan settings back to `false` (otherwise it re-plans at every launch).
5. Keep the model in the source folder, so a rebuild does not lose it:
   ```bash
   cp -r <path> src/navbim/navbim_bringup/nav_model/
   colcon build --symlink-install --packages-select navbim_bringup
   ```
   From now on, launching with `ifc:=<Name>` reuses this model
   (`force_nav_model_gen:=true` builds it again, e.g. after changing the IFC;
   then repeat steps 2-4).
6. Copy the pre-planned map into this repo and push it:
   ```bash
   cp src/navbim/navbim_bringup/nav_model/<Name>/<Name>.json <thesis repo>/data/<Name>_preplanned.json
   cd <thesis repo> && git add data/<Name>_preplanned.json && git commit -m "Add pre-planned <Name> map" && git push
   ```

On the **Windows PC** (in this repo, venv active):

7. `git pull`, then build the tables:
   ```bash
   python scripts/navbim/navbim_distance_table.py data/<Name>_preplanned.json                    # option A
   python scripts/navbim/navbim_distance_table.py data/<Name>_preplanned.json --method centre --check  # option B
   ```
   It prints the number of rooms, unreachable room pairs and 0 m pairs, and writes
   `data/distances/<Name>_preplanned_walking_distances[_centre].{json,csv}`.
   If it says "Run navbim PrePlanEdges first", step 3 did not finish pre-planning.
8. Use the table in code. Rooms are keyed by their IfcSpace GUID, the same key as
   in `data/cleaned_ifc/<Name>_llm.json` (made from the **same** IFC file with
   `clean_ifc.py`, section 2):
   ```python
   from navbim_distance_table import load_distance_function   # scripts/navbim on sys.path
   dist = load_distance_function("data/distances/<Name>_preplanned_centre_walking_distances.json")
   dist(guid_a, guid_b)    # metres, None = no walkable connection
   # e.g. rank_candidates.optimal_visit_order(..., distance=lambda a, b: dist(guid[a], guid[b]))
   ```
   The `.json` also holds, for every pair, the doors and rooms the walk goes
   through, so any number can be checked.

Options for option B: `--clearance` (robot radius in m; keep it equal to navbim's
`robot_radius`, 0.2 now), `--resolution` (grid cell, 0.05 m), `--wall` (wall
thickness kept around other rooms, 0.125 m). Limits: tested on the one-storey
IDAC_Model only; the "centre" of a big room is the middle of its outline, which
can be far from where a robot would actually stand; furniture is not in the
outlines.

The HTN layer (section 5) finds the option B table by the building name and
uses it for the visit order by default (`--distance navbim`), so after step 7
nothing else is needed: `interactive_search.py --cleaned_json data/cleaned_ifc/<Name>_llm.json ...`
prints "visit order uses navbim walking distances from ..." at start-up.

### Tests

```bash
pip install -r requirements-dev.txt
python -m pytest            # tests/htn: framework, IDAC domain, robot profiles, executor, reachability
                            # tests/navbim: walking-distance tables (options A and B)
                            # tests/search_planners: Held-Karp regression, BLR, selector, --no-htn session
```

## IFC <-> JSON converters

`file_converters/` holds a standalone converter between IFC (SPF) and the
ifcJSON format (`ifc2json.py`, `json2ifc.py`, helpers in `ifcjson/`). It is
separate from `clean_ifc.py`: it converts the whole model losslessly rather than
producing the lightweight room/equipment summary the pipeline consumes. See
[file_converters/README.md](file_converters/README.md) for install and usage.

## Legacy graph pipeline (archived)

`legacy_graph_pipeline/` contains an earlier approach that extracted IFC data
into a Neo4j graph and queried it with an LLM-assisted Cypher generator. It's
kept for reference and is not part of the active pipeline described above.
It requires Docker (`docker compose up` in that directory, using the
`NEO4J_USER`/`NEO4J_PASSWORD` in `.env.legacy`) plus `neo4j`, `python-dotenv`,
and `requests`.

## Notes

- `notes.txt` is the thesis log: what changed, why, rejected alternatives, and
  experiment results. It is updated alongside this README.
- `testing_vlm.py` is a minimal smoke test for confirming Ollama + a vision
  model are reachable — not part of the pipeline itself.
- `data/images/` holds sample photos (kitchen, offices, reception, seminar
  room) usable as stand-in camera captures when testing `interactive_search.py`.
