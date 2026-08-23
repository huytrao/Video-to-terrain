# AGENTS.md — engineering manual for AI assistants

Read this before changing anything. It states the rules that are not visible from
the code alone, and the places where a plausible-looking change is wrong.

## Engineering philosophy

1. **Fail loudly and immediately.** Never swallow an exception into a silent
   fallback. Missing depth, a rotation with negative determinant, an invalid
   intrinsic matrix — raise a descriptive `RuntimeError` or `ValueError` at the
   point of discovery. A wrong answer that looks right costs a GPU day.
2. **Preserve units and physical meaning.** Depth from Depth-Anything-3 is an
   *estimate* in DA3 units. Label it `unit="da3_estimate"` until a physical tape
   measurement calibrates it through `--metres-per-unit`.
3. **Resolve paths dynamically.** `Path(__file__).resolve().parents[N]`, never a
   hardcoded absolute directory. Note the depth: files in `scripts/<group>/` are
   **two** levels below the repo root, so the root is `parents[2]`.
4. **Zero regressions.** `make test` must print `all checks passed` for both
   suites before you call anything done.

## Repository map

```text
src/v3d/            The package. Importable as `v3d` after `make setup`.
  config.py           PipelineConfig, SceneLayout, layout detection
  paper_runner.py     Stage dispatcher + CLI entry point (`v3d`)
  ingest/             Video decode, focal pooling, intrinsic/extrinsic math
  geometry/           Adaptive quality knobs, PGSR build, PyTorch3D shim
  semantic/           VLM prompting and multi-view label voting
  scene/              Plane RANSAC, box decomposition, scene schema
  export/             GLB, terrain heightfield, Blender, visualisation
  validate/           Stage J physics checks, ground-truth box comparison
  data/               ScanNet++ and HuggingFace tar dataset access

scripts/            Executables. The subdirectory says WHERE a script runs.
  kernel/             Inside the Kaggle GPU kernel. No internet there.
  kaggle/             Locally, talking to Kaggle over the Jupyter proxy.
  deploy/             Locally, deploying to an SSH GPU server.
  sim/                MuJoCo scene build / validate / render.
  adapter/            HoliSpatial -> Hunyuan3D adapter, numbered 01..10.
  tools/              Standalone utilities with no pipeline position.
  patches/            Patches held against upstream, not yet landed.

docs/               All documentation. Index at docs/README.md.
  architecture/       How the system is shaped
  guides/             How to run a given thing
  reference/          Facts to look up, and the record of how they were learned
  state/              Live run state; changes every session

tests/              Regression tests. Plain scripts, not pytest cases.
configs/            Checked-in configuration.
skills/             Python helpers agents may import (VLM, vision, search).
```

Vendored upstream, fetched rather than authored here — do not "fix" these:
`depth-anything-3/`, `Hunyuan3D-2/`, `source-reference/`,
`kaggle-runtime-toolkit-agent-skills/`.

## Coordinate conventions

Three frames meet in this codebase. Never mix them without an explicit transform.

| Frame | Axes | Appears in |
| :--- | :--- | :--- |
| OpenCV | `+X` right, `+Y` down, `+Z` forward | Image plane, video frames, COLMAP |
| OpenGL | `+X` right, `+Y` up, `-Z` forward | NeRF, 3DGS, DA3, WebGL viewers |
| MuJoCo / ROS | `+X` forward, `+Y` left, `+Z` up | Simulation, URDF, RL world frame |

Camera-to-world conversion:

$$c2w_{gl} = c2w_{cv} \cdot \operatorname{diag}(1, -1, -1, 1)$$

- **Determinant invariant**: $\det(R) = +1.0 \pm 10^{-5}$. A negative determinant
  means a mirrored scene and must be rejected, not corrected.
- **Intrinsic scaling** on resize from $(W_o, H_o)$ to $(W_p, H_p)$:
  $$f_x' = f_x \tfrac{W_p}{W_o},\quad f_y' = f_y \tfrac{H_p}{H_o},\quad
    c_x' = c_x \tfrac{W_p}{W_o},\quad c_y' = c_y \tfrac{H_p}{H_o}$$

## Quality rails

In `src/v3d/geometry/quality.py` the constants are **derived from scene depth
statistics**, never hardcoded. Changing a number here changes output quality, so
change the derivation, not the result.

1. **Voxel size** from median depth $d_{med}$, focal length $f$, and pixel
   footprint, with a memory guard of 6000 voxels across the scene diagonal:
   $$v = \operatorname{clamp}\left(1.75 \tfrac{d_{med}}{f},\; \tfrac{D_{scene}}{6000},\; \tfrac{D_{scene}}{2500}\right)$$
2. **Fusion distance** from the 98th depth percentile: $d_{fuse} = P_{98}(d) \cdot 1.05$
3. **Mask distance**: $d_{mask} = 1.5 \cdot d_{fuse}$

## Kaggle remote execution

The GPU stages run on a Kaggle kernel behind an HTTP/WebSocket Jupyter proxy.

1. **Proxy URL**: `https://kkb-production.jupyter-proxy.kaggle.net/k/<KERNEL_ID>/<JWT>/proxy`
   The path *is* a JWT — it is a credential with an expiry, not configuration.
   It lives in `.env` (`KAGGLE_WITHOUT_INTERNET` / `KAGGLE_HAS_INTERNET`) and
   expires in 1–2 hours.
2. **The run kernel has no internet.** Every wheel and model weight must be
   pre-staged as a Kaggle dataset. Code goes up via `scripts/kaggle/deploy_code.sh`.
3. **`kx.sh run` takes a LOCAL path** and uploads it to `_run/<basename>` before
   executing it remotely. The `scripts/kernel/` grouping is therefore local
   organisation only — it does not change any remote path.
4. **Never read `.env` by sourcing it.** A key whose name contains a dash is not
   a valid shell identifier, and sourcing once echoed a live credential into a
   terminal log. Use `scripts/kaggle/kenv.sh`, which greps only the keys needed.
5. **Uploads over ~24 MB are silently truncated** by the Contents API. Use
   `scripts/kaggle/kupload.py`, which chunks and verifies a SHA-256 computed on
   the kernel over the reassembled file.

## Before you finish

- [ ] `make test` prints `all checks passed` for both suites.
- [ ] No hardcoded absolute paths; `parents[N]` depth matches the file's location.
- [ ] No generated artefacts staged for git (`outputs-*/`, `*.log`, `.DS_Store`).
- [ ] `docs/state/SESSION_STATE.txt` updated if stage status changed.
- [ ] Nothing secret added to a tracked file — `.env`, proxy URLs, API keys, and
      `server-connect.txt` are gitignored for a reason.
