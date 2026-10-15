# Video-to-3D pipeline — one entry point per thing you might want to do.
#
# The tests are plain scripts, not pytest cases: each prints its own pass lines
# and exits non-zero on failure. `make test` runs them that way so it works in a
# bare venv. `pip install -e '.[dev]'` adds pytest if you want `make test-pytest`.

PY      ?= .venv/bin/python
VENV    ?= .venv
SYS_PY  ?= python3.11

# The existing .venv was made by uv and has no pip in it, so `.venv/bin/pip`
# is not a thing that exists here. Use uv when it is installed (it is, and it is
# what made this venv); fall back to `python -m pip` only for a plain venv.
UV := $(shell command -v uv 2>/dev/null)
ifdef UV
  INSTALL = uv pip install --python $(PY)
else
  INSTALL = $(PY) -m pip install
endif

.DEFAULT_GOAL := help
.PHONY: help setup install install-all test test-pytest lint verify clean clean-venv \
        kaggle-ping kaggle-deploy sim-build sim-validate sim-render room-scene room-view room-play room-serve visualize video-index trim-videos tree

help:  ## Show this help
	@echo "Video-to-3D — make targets"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "First time:  make setup"

# --- environment ------------------------------------------------------------

$(VENV)/bin/python:
ifdef UV
	uv venv --python $(SYS_PY) $(VENV)
else
	$(SYS_PY) -m venv $(VENV)
endif

setup: $(VENV)/bin/python  ## Create .venv and install the pinned dependency set
	$(INSTALL) -r requirements.txt
	$(INSTALL) -e .
	@echo ""
	@echo "Done. Activate with:  source $(VENV)/bin/activate"
	@echo "Then check it:        make test"

install:  ## Reinstall the v3d package only (editable)
	$(INSTALL) -e .

install-all:  ## Install with every optional extra (texture, llm, dev)
	$(INSTALL) -e '.[texture,llm,dev]'

# --- checks -----------------------------------------------------------------

test:  ## Run the geometry/pose regression tests (no pytest needed)
	$(PY) tests/test_quality_knobs.py
	$(PY) tests/test_pose_conventions.py

test-pytest:  ## Same tests through pytest (needs `make install-all`)
	$(PY) -m pytest -q

lint:  ## Ruff check (needs `make install-all`)
	$(PY) -m ruff check src scripts tests

verify:  ## Run comprehensive verification suite (tests, static scan, imports)
	$(PY) tests/run_all_checks.py

verify-geometry:  ## Check CUDA / torch / v3d import health on this machine
	$(PY) scripts/tools/verify_geometry_setup.py

# --- remote GPU -------------------------------------------------------------

kaggle-ping:  ## Check the Kaggle proxy is alive (reads .env)
	scripts/kaggle/kx.sh ping

kaggle-deploy:  ## Upload src/v3d + the paper checkout to the run kernel
	scripts/kaggle/deploy_code.sh

# --- simulation -------------------------------------------------------------

SCENE ?= outputs-full-factory/robot_sim

# --- room + robot ----------------------------------------------------------
# The reconstruction is Z-down and its mesh is a shell, so neither gravity nor
# collision works until both are fixed. ROOM_ROBOT points at a unitree_robots/<name>
# directory holding origin/ and meshes/; without it the scene still builds, with the
# control sphere only. Set ROOM_ROBOT=<path> or ROBOT_DIR in environment.
ROOM       ?= outputs-full-factory/delivery/OUTPUT_B1/mesh/room_300k.glb
ROOM_GT    ?= outputs-full-factory/delivery/OUTPUT_B1/metadata/ground_truth_boxes.json
ROOM_ROBOT ?= $(or $(ROBOT_DIR),$(wildcard ./robot_assets))

room-scene:  ## Room mesh + robot -> textured MJCF, physics-checked and rendered
	$(PY) scripts/sim/make_room_scene.py \
	  --room $(ROOM) --gt $(ROOM_GT) --out $(ROOM_OUT) $(if $(ROOM_ROBOT),--robot $(ROOM_ROBOT),)

# POLICY points to an exported ONNX policy (e.g. from RSL-RL).
# Set POLICY=<path> or ROBOT_POLICY in environment, or override on the command line.
POLICY ?= $(or $(ROBOT_POLICY),$(wildcard ./policies/policy.onnx))
CMD    ?= 0 0 0

room-play:  ## Run an exported RSL-RL policy in the room (CMD="vx vy wz", default stand)
	@if [ -z "$(POLICY)" ]; then \
	  echo "Error: POLICY is not set and no default policy ONNX found."; \
	  echo "Specify POLICY=<path/to/policy.onnx> or set ROBOT_POLICY in environment."; \
	  exit 1; \
	fi
	$(PY) scripts/sim/play_policy.py --scene $(ROOM_OUT)/room_scene.xml \
	  --policy $(POLICY) --command $(CMD) --seconds 10 \
	  --render $(ROOM_OUT)/policy_play.png

visualize:  ## Gather every visualizable output into visualize/ with an index page
	$(PY) scripts/tools/collect_visualize.py --out visualize

SECS ?= 7

trim-videos:  ## Cut every visualize/ video to its first $(SECS) seconds
	$(PY) scripts/tools/trim_videos.py --seconds $(SECS)
	$(PY) scripts/tools/make_video_index.py

video-index:  ## Rebuild just visualize/videos.html
	$(PY) scripts/tools/make_video_index.py

PORT ?= 8080

room-serve:  ## Render the policy run to video and serve it at http://127.0.0.1:$(PORT)
	@if [ -z "$(POLICY)" ]; then \
	  echo "Error: POLICY is not set and no default policy ONNX found."; \
	  echo "Specify POLICY=<path/to/policy.onnx> or set ROBOT_POLICY in environment."; \
	  exit 1; \
	fi
	$(PY) scripts/sim/serve_room.py --scene $(ROOM_OUT)/room_scene.xml \
	  --policy $(POLICY) --command $(CMD) --port $(PORT)

room-view:  ## Open the built room scene in the MuJoCo viewer
	$(PY) -m mujoco.viewer --mjcf=$(ROOM_OUT)/room_scene.xml

sim-build:  ## Compile scene_layout.json into a MuJoCo MJCF scene
	$(PY) scripts/sim/make_unitree_scene.py --layout $(SCENE)/scene_layout.json --out-dir $(SCENE)

sim-validate:  ## Physics settling / penetration audit on the built scene
	$(PY) scripts/sim/validate_unitree_scene.py

sim-render:  ## Render inspection stills and a 360 orbit mp4
	$(PY) scripts/sim/render_unitree_scene.py --scene $(SCENE)/mujoco --out $(SCENE)/videos

presentation-serve:  ## Serve the Twin Data Presentation at http://localhost:8080
	$(PY) -m http.server 8080 --directory output/Twin_Data_Presentation

# --- housekeeping -----------------------------------------------------------

clean:  ## Remove caches and compiled Python
	find . -path ./.venv -prune -o -name '__pycache__' -type d -print0 2>/dev/null | xargs -0 rm -rf
	rm -rf .pytest_cache .ruff_cache .mypy_cache
	find . -path ./.venv -prune -o -name '.DS_Store' -print0 2>/dev/null | xargs -0 rm -f

clean-venv:  ## Delete the virtualenv entirely
	rm -rf $(VENV)

tree:  ## Print the repository layout, minus vendored and generated trees
	@find . -maxdepth 2 -type d \
	  -not -path './.git*' -not -path './.venv*' -not -path './outputs-*' \
	  -not -path './input/*' -not -path './depth-anything-3/*' \
	  -not -path './Hunyuan3D-2/*' -not -path './source-reference/*' \
	  -not -path '*/__pycache__*' | sort
