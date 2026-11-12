# navbim setup log (Ubuntu 24.04 + ROS 2 Jazzy)

Log of every change made to get navbim (GPP-BIM) running for the thesis.
Each entry says **what** was changed, **why**, and **how to undo it**.
Newest entries at the bottom. Keep adding to it.

Paths (on the Ubuntu PC):
- Workspace: `~/Junaidali/navbim_ws` (navbim repo cloned to `~/Junaidali/navbim_ws/src/navbim`)
- Backup / thesis files: `~/Junaidali/thesis_models`

---

## A. System changes (outside the navbim code)

| # | What | Why | Undo |
|---|------|-----|------|
| A1 | Installed ROS 2 Jazzy: `sudo apt install -y ros-jazzy-desktop ros-dev-tools` (+ `software-properties-common curl unzip htop nlohmann-json3-dev`) | Needed to run navbim | `sudo apt remove ros-jazzy-*` |
| A2 | Installed Python packages with `pip3 install --break-system-packages --ignore-installed ...` (list from navbim `Dockerfile`). Versions differ from the Dockerfile pins: numpy 1.26.4, dash 3.2.0, plotly 6.3.1 | navbim generator dependencies | `pip3 uninstall --break-system-packages <pkg>` |
| A3 | IfcConvert 0.8.5 → `/usr/local/bin/IfcConvert` (download zip from IfcOpenShell releases, `unzip -o ifcconvert.zip IfcConvert`, `sudo mv IfcConvert /usr/local/bin/ && sudo chmod +x /usr/local/bin/IfcConvert`) | Mesh export in the generator | `sudo rm /usr/local/bin/IfcConvert` |
| A4 | `rosdep install --from-paths src --ignore-src -y --skip-keys "slam_toolbox nlohmann_json"` | Install ROS dependencies of navbim | — |
| A5 | Extra 16 GB swap file `/swapfile2` (`fallocate`, `chmod 600`, `mkswap`, `swapon`). Not added to `/etc/fstab`, so it is not re-enabled after reboot, but the file stays on disk | The PC froze/restarted during `colcon build` (out of memory) | `sudo swapoff /swapfile2 && sudo rm /swapfile2` |

**Build command that works** (without it the PC runs out of memory):
```bash
cd ~/Junaidali/navbim_ws
MAKEFLAGS="-j4" colcon build --symlink-install --packages-skip nav2_system_tests --parallel-workers 4
source install/setup.bash
```

---

## B. Changes to the navbim source code

Run `cd ~/Junaidali/navbim_ws/src/navbim && git diff` to see all code changes at once.
`git checkout -- <file>` undoes a file completely. Then rebuild that package.

### B1. Topological map saved with "links" instead of "edges"
- **File:** `navbim_nav_model_gen/navbim_nav_model_gen/topology_manager.py`, line ~481 (`save_topological_map`)
- **Before:** `data = nx.node_link_data(topological_map)`
- **After:** `data = nx.node_link_data(topological_map, edges="links")`
- **Why:** newer NetworkX writes the key `"edges"`, but navbim's C++ loader (`navbim_topomap_server/src/topological_map_loader.cpp`) only reads `"links"`. The result was an empty graph ("0 vertices", "Topological graph not initialized").
- **Undo:** remove `, edges="links"`. After the change, the nav model must be generated again.

### B2. Removed the `IfcBuildingElement` block in the BIM server
- **File:** `navbim_bim_server/navbim_bim_server/bim_server.py`, `_build_element_cache` (~line 138)
- **Removed:**
  ```python
          # Also cache building elements
          for element in self.ifc_model.by_type('IfcBuildingElement'):
              if hasattr(element, 'GlobalId'):
                  self.element_cache[element.GlobalId] = element
  ```
- **Why:** IFC4X3 renamed `IfcBuildingElement` to `IfcBuiltElement`, so bim_server crashed with "Entity with name 'IfcBuildingElement' not found in schema 'IFC4X3_ADD2'". Nothing is lost, because the loop over `IfcElement` just above already includes these elements.
- **Note:** commenting the lines out by hand first broke the indentation (`IndentationError` at `def get_element_info_callback`). The block was then deleted with `sed`. Check with `python3 -m py_compile <file>`.
- **Undo:** paste the 4 lines back (8 spaces indent), then `colcon build --symlink-install --packages-select navbim_bim_server`.
- **Better alternative later (works for IFC2X3/IFC4 too):** use `IfcBuiltElement` when the schema is IFC4X3, otherwise `IfcBuildingElement`.

### B3. Topomap server no longer saves at shutdown
- **File:** `navbim_bringup/params/navbim_params.yaml`, section `topomap_server`
- **Before:** `save_at_shutdown: true` → **After:** `save_at_shutdown: false`
- **Why:** when loading failed, the server overwrote the JSON with an empty graph at shutdown.
- **Undo:** set it back to `true`. Note: PrePlanEdges still saves through its own `SaveTopologicalMap` call.

### B4. Floor smoothing switched off (test, 2026-11-12)
- **File:** `navbim_bringup/params/navbim_params.yaml`, line 53
- **Before:** `enable_floor_smoothing: true` → **After:** `enable_floor_smoothing: false`
- **Why:** to check whether smoothing causes the path to cut through walls. Result: no, the path is the same without it. Left off for now, because the distance table should use the planned path.
- **Undo:** `sed -i 's/enable_floor_smoothing: false/enable_floor_smoothing: true/' <file>` and restart navbim. No rebuild is needed (symlink-install).

### B5. Costmap parameters were never applied (FIXED and verified 2026-11-12)
- **File:** `navbim_bringup/params/navbim_params.yaml`, section starting at line ~123
- **Problem:** the section is written as `global_costmap: → global_costmap: → ros__parameters:`, which ROS reads as the node `/global_costmap/global_costmap`. But the room planner creates the costmap node as plain `/global_costmap` (`navbim_multi_costmap_2d/src/multi_costmap_2d_ros.cpp` line 46 passes an empty local namespace). So **none** of these settings are applied, and defaults are used instead. Evidence in the bringup log: `Using plugin "obstacle_layer"` appears, although the yaml lists only static_layer and inflation_layer.
- **Effect:** the default `track_unknown_space: false` makes the static layer turn "unknown" cells (value 100 in the room PNGs) into **free** cells (`nav2_costmap_2d/plugins/static_layer.cpp` line 274). The Workspace map marks the Printer_room/Storage block as unknown (checked: value 100), so the planner walked through it. Also `robot_radius` is 0.1 instead of 0.2, and the inflation settings are not used.
- **Check:** `ros2 param get /global_costmap track_unknown_space` hangs (the node runs inside room_planner_server and doesn't answer parameter requests), so instead check the bringup log: after the fix there's no `Using plugin "obstacle_layer"` line.
- **Result:** the Workspace → Office_1 path now goes around the Printer_room/Storage block (left side), as it should.
- **Fix:** remove one nesting level (`global_costmap: → ros__parameters:`), with a backup `navbim_params.yaml.bak` made first.
- **Undo:** `cp navbim_params.yaml.bak navbim_params.yaml`
- **Note:** `clearance_costmap` (clearance_server) has no section at all, so it also runs on defaults. That's not important for the path length.

### B6. Pre-planning started automatically at startup (2026-11-12)
- **File:** `navbim_bringup/params/navbim_params.yaml`, section `gpp_bim`, lines ~37-38
- **Before:** `pre_plan_paths: false`, `force_pre_plan_of_planned_paths: false` → **After:** both `true`
- **Why:** `ros2 action send_goal /pre_plan_edges ...` from a second terminal hangs at "Sending goal" (navbim never logs "Received pre-plan edges goal request"). `ros2 param get` also hung. Both terminals have the same env (`RMW` empty = Fast DDS, `ROS_DOMAIN_ID=6`, `ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET`). With these settings, gpp_bim runs pre-planning by itself 15 s after activation (`gpp_bim_server.cpp` ~line 199), without the terminal command.
- **Undo:** set both back to `false` after pre-planning is done, otherwise it re-plans on every launch. A backup of the JSON before pre-planning is in `IDAC_Model.before_preplan.json`.
- **Result (2026-11-12):** "Successfully pre-planned all 44 transition edges (26.67 ms)". The JSON was saved (64 KB → 551 KB), and `planned_distance` now has real values (about 1.5–17.8 m) instead of -1. Settings should be set back to `false` now.
- **Still open:** commands from a second terminal don't reach navbim. Possibly other ROS PCs on the lab network use domain 6 (RANGE=SUBNET). Try `export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST` in every terminal (incl. the one that launches navbim) + `ros2 daemon stop`.

---

## C. Files added

| # | What | Why | Undo |
|---|------|-----|------|
| C1 | `~/Junaidali/thesis_models/IDAC_Model.ifc` and generated model `~/Junaidali/thesis_models/nav_model/IDAC_Model/` | First generation run (now kept as **backup**) | delete folder |
| C2 | `~/Junaidali/thesis_models/check_doors.py` | Prints which rooms each door connects (reads `"links"` from the JSON) | delete |
| C3 | Copied the IFC to `navbim_bringup/bim/IDAC_Model.ifc` and the model to `navbim_bringup/nav_model/IDAC_Model/`, then `colcon build --symlink-install --packages-select navbim_bringup` | The RViz BimPanel only looks in navbim_bringup's install `share` folder (`navbim_rviz_plugins/src/bim_panel.cpp` lines 67, 78, 129-137) and ignores custom paths | delete both, rebuild `navbim_bringup` |

| C4 | `Master_thesis/data/IDAC_Model_preplanned.json` (commit dde6b5f, pulled on the Windows PC 2026-11-12): copy of the pre-planned navbim JSON. 31 nodes, 103 links, 44/44 transition edges with `planned_distance` + `path` | Input for the room-to-room distance table script (works without ROS) | `git rm` |

Launch now: `ros2 launch navbim_bringup bringup_launch.py ifc:=IDAC_Model`

---

## D. Verified results

- **2026-11-12:** generator output for IDAC_Model: 31 nodes, 15 rooms (all IfcSpaces, no combined rooms), 15 doors. 14 interior doors each link 2 correct rooms. The exterior door `2BOvfVeQT2E8wiKZrVEfNM` links only Workspace (dead end; that's fine, the robot must not leave the building).
- **2026-11-12:** bringup log shows `Topological map loaded successfully: 31 vertices, 103 edges`. 15 room costmaps created for `Floor_0`.
- **2026-11-12:** after the B2 fix, bim_server starts: "Successfully loaded IFC file with 249 elements", configured and activated. The full bringup runs with no errors (`ros2 launch navbim_bringup bringup_launch.py ifc:=IDAC_Model`).
- **2026-11-12:** RViz check: BimPanel shows IDAC_Model / Floor_0. Room maps show only with the Map display's **Durability Policy = Transient Local** (maps are published once). Most test paths look correct. One path (start in upper-middle open area → goal in the upper-right room) looks wrong: it cuts through the small two-room block and seems to cross the wall into the goal room. Under investigation (see E).
- The planner prints `Path distance: <m>` for every plan (gpp_bim_server.cpp ~line 738). We can use this for the distance table.
- Harmless warnings: `Robot is out of bounds of the costmap` (no robot pose yet), matplotlib `Unable to import Axes3D`, pip PATH warnings.

---

## E. Open / next steps

- [x] Confirm bim_server starts after the B2 fix (2026-11-12)
- [x] Check in RViz: BimPanel → IDAC_Model / Floor_0, the TopomapPanel graph, test paths (mostly OK)
- [x] Smoothing ruled out (B4)
- [x] FIXED by B5 (2026-11-12). Workspace → Office_1 path: the room order is correct (Workspace → door 27 → Corridor → door 28 → Meeting_room → door 23 → Office_1), but inside **Workspace** the path cuts straight through the Printer_room/Storage block (x 28.8–31.4, y 20.2–25.3) instead of going around it on the left (x < 28.66). Suspect: Workspace's own room map (`/maps/Floor_0/Workspace/map`) shows that block as free. Likely causes in `occupancy_map_manager.py`: floor-slab cells are marked free before the polygon check (~line 586), and BFS dilation (~line 615) grows by `max_width_of_walls` into neighbouring rooms. Check the Workspace map.
- [x] PrePlanEdges done via auto pre-plan (B6), 44/44 edges, 2026-11-12. Old note: Run PrePlanEdges (`ros2 action send_goal /pre_plan_edges navbim_msgs/action/PrePlanEdges "{force_replan: true}" --feedback`). This **writes into** `IDAC_Model.json`, so back it up first
- [ ] Script: 15×15 room-to-room walking-distance table keyed by IfcSpace GUID for Held-Karp
- [ ] Set robot size parameters from the URDF
- [ ] Delete `/swapfile2` when no longer needed (A5)
