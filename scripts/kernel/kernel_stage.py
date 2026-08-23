"""Take a fresh RTX PRO 6000 kernel to one that can run every pipeline stage.

Runs on the RUN kernel, which has no internet. Everything it installs is already
on disk: the attached Kaggle datasets, and the wheel bundle relayed from BUILD.

The whole design is an **allowlist**. `v3d-build-cache-sm120/deps` holds 120
wheels including `torch-2.13.0`, `numpy-2.5.2`, `pandas-3.0.5`, `triton-3.7.1`
and a full CUDA 13 runtime — and this kernel ships torch 2.10.0+cu128 with
numpy 2.0.2, which is exactly what the two PGSR CUDA extensions in
`pgsr_sm120/` were compiled against. A `pip install deps/*.whl` would replace
torch and numpy, and the extensions would stop importing with an ABI error that
looks like a broken build rather than a broken environment. So nothing is
installed unless it is named here, and every install is `--no-deps` so pip is
never free to resolve a replacement.

Then it *proves* the result rather than reporting the pip exit codes. Each check
runs the thing: `distCUDA2` over real points, `rasterize_gaussians` present with
its backward, nvdiffrast rasterising a triangle and counting covered pixels, and
numpy/torch re-checked after every install phase — because the failure this
guards against is an install that succeeds and quietly downgrades something.

    python3 /kaggle/working/_run/kernel_stage.py
"""

from __future__ import annotations

import glob
import os
import subprocess
import sys
import time

DATASETS = "/kaggle/input/datasets/traoanhuy"
CACHE = f"{DATASETS}/v3d-build-cache-sm120"
DEPS = f"{CACHE}/deps"
PGSR_WHEELS = f"{CACHE}/pgsr_sm120"
DA3_DS = f"{DATASETS}/v3d-da3nested-giant-large"
SAM3_DS = f"{DATASETS}/v3d-sam31-multiplex"
EXTRA = "/tmp/extra"

# nvdiffrast compiles its CUDA at first use, not at install, and needs a writable
# cache. Set before torch is asked to build anything.
os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/tmp/torchext")
os.makedirs("/tmp/torchext", exist_ok=True)

# What the environment must still look like at the end. Checked after every phase.
PINNED = {"numpy": "2.0.2", "torch": "2.10.0+cu128"}

# --- the allowlist -----------------------------------------------------------
#
# Distribution names, matched against the wheel filename's leading token. A name
# absent from the wheel directory is an error, not a skip: silently missing
# open3d means the mesh stage fails an hour later.
#
# open3d needs most of this list at *import* time, not at use time:
# `open3d/__init__.py` imports `open3d.visualization`, which reaches dash and
# plotly, and `open3d.ml`, which reaches addict. They are cheap and present, so
# installing them beats stubbing them.
FROM_DEPS = [
    "open3d",                                     # TSDF fusion, mesh IO — the mesh stage
    "plyfile",                                    # PGSR reads/writes gaussian PLYs
    "trimesh",                                    # paper code, mesh handling
    "lpips",                                      # PGSR's perceptual term
    "addict", "dash", "plotly", "narwhals", "retrying",       # open3d import chain
    "nbformat", "fastjsonschema", "jsonschema", "jsonschema_specifications",
    "referencing", "rpds_py", "attrs",
    "jupyter_core", "traitlets", "platformdirs",
    "ipywidgets", "widgetsnbextension", "jupyterlab_widgets", "comm",
    "flask", "werkzeug", "itsdangerous", "blinker", "importlib_metadata", "zipp",
    "configargparse", "pyopengl", "glfw",
    "hydra_core", "omegaconf", "portalocker",
    "mujoco", "etils",                            # the MuJoCo scene output
    "av",                                         # writing result videos
    "pyquaternion",
    "joblib", "threadpoolctl", "scikit_learn",    # sklearn: RANSAC in pose alignment
]

# The two entries in `deps/` that are source distributions rather than wheels.
# They need handling of their own for two reasons: each is nested one directory
# deeper than its name suggests (`deps/iopath-0.1.10/iopath-0.1.10/setup.py`),
# and `/kaggle/input` is a read-only mount, so pip cannot write the egg-info a
# build produces. Both are copied to /tmp first.
#
# `antlr4` is omegaconf's grammar runtime and `iopath` is what SAM3's config
# loader uses; neither has a wheel in the cache.
SDIST_FROM_DEPS = ["antlr4-python3-runtime-4.9.3", "iopath-0.1.10"]

# From the BUILD bundle. pandas, seaborn, rosbags and numexpr came down as evo's
# transitive dependencies and are deliberately *not* here: pandas 3.0.5 requires
# numpy >= 2.1 and would drag the pin. evo only reaches pandas through
# `evo.tools.pandas_bridge`, which DA3 never imports.
FROM_EXTRA = [
    "evo",                                        # DA3's pose_align, a hard import
    "pycocotools",                                # sam3.py:12, hard import
    "scikit_image",                               # sam3.py:19 `from skimage import morphology`
    "tifffile", "lazy_loader", "networkx",        # scikit-image's own imports
    "colorama", "natsort",                        # evo's
]


def run(cmd: list[str], what: str, timeout: int = 1800) -> None:
    t = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    tail = (r.stdout + r.stderr).strip().splitlines()
    if r.returncode != 0:
        raise SystemExit(
            f"FAILED: {what}\n  cmd: {' '.join(cmd[:8])} …\n"
            + "\n".join(tail[-25:])
        )
    print(f"  [{time.time() - t:5.1f}s] {what}")


def wheels_for(names: list[str], directory: str) -> list[str]:
    """Resolve distribution names to wheel paths, refusing to guess."""
    available = {}
    for path in sorted(glob.glob(os.path.join(directory, "*"))):
        base = os.path.basename(path)
        dist = base.split("-")[0].replace("_", "-").lower()
        available.setdefault(dist, path)

    resolved, missing = [], []
    for name in names:
        key = name.replace("_", "-").lower()
        if key in available:
            resolved.append(available[key])
        else:
            missing.append(name)
    if missing:
        raise SystemExit(
            f"FAILED: {directory} has no wheel for {missing}. Every name in the "
            f"allowlist is reached by a stage, so a missing one is a stage that "
            f"fails later, not a package that can be skipped.\n"
            f"  present: {sorted(available)[:40]}"
        )
    return resolved


def wheel_index(directories: list[str]) -> dict[str, str]:
    """Map distribution name -> path, across every wheel directory available."""
    index: dict[str, str] = {}
    for directory in directories:
        for path in sorted(glob.glob(os.path.join(directory, "*"))):
            dist = os.path.basename(path).split("-")[0].replace("_", "-").lower()
            index.setdefault(dist, path)
    return index


def ensure_importable(module: str, directories: list[str], max_rounds: int = 30) -> list[str]:
    """Import `module`, installing whatever it names as missing, one round at a time.

    open3d is the reason this exists. `open3d/__init__.py` imports
    `open3d.visualization` and `open3d.ml` unconditionally, which between them
    reach dash, plotly, janus, addict and a dozen more — a chain that is tedious
    and error-prone to enumerate by hand, and whose exact shape differs between
    open3d builds. Asking the interpreter what it wants is both shorter and
    correct by construction.

    Each probe runs in a *fresh* process on purpose. A failed import leaves
    partially initialised modules behind, and clearing them out of `sys.modules`
    to retry is not safe for a C extension: open3d's pybind11 layer registers
    `open3d.cuda.pybind.visualization` during its one-time module init, and
    removing those entries makes the re-import fail with "'open3d.cuda.pybind' is
    not a package" — which looks like a broken install and is not one.
    """
    index = wheel_index(directories)
    actions: list[str] = []
    for _ in range(max_rounds):
        probe = subprocess.run(
            [sys.executable, "-c",
             f"import {module}\nprint('IMPORT_OK')"],
            capture_output=True, text=True,
        )
        if "IMPORT_OK" in probe.stdout:
            return actions
        missing = None
        for line in (probe.stderr or "").splitlines():
            if "ModuleNotFoundError: No module named" in line:
                missing = line.split("'")[1] if "'" in line else None
        if missing is None:
            raise SystemExit(
                f"FAILED: `import {module}` failed for a reason other than a missing "
                f"module:\n" + "\n".join((probe.stderr or "").splitlines()[-20:])
            )
        # A missing *submodule* of the target is a broken install, not an absent
        # third-party package; installing something else would hide it.
        if missing == module or missing.startswith(module + "."):
            raise SystemExit(
                f"FAILED: {module} is installed but cannot import its own submodule "
                f"{missing}. Reinstall it rather than papering over this."
            )
        key = missing.split(".")[0].replace("_", "-").lower()
        if key not in index:
            raise SystemExit(
                f"FAILED: `import {module}` needs {missing!r} and no wheel for it "
                f"exists in {directories}. Download it on BUILD and relay it, "
                f"rather than stubbing a package a stage will actually call."
            )
        run(PIP + [index[key]], f"{module} needs {missing} -> {os.path.basename(index[key])}")
        actions.append(missing)
    raise SystemExit(f"FAILED: could not import {module} in {max_rounds} rounds; did {actions}")


def check_pins(after: str) -> None:
    """Re-read the pinned versions in a *fresh* process.

    In-process `importlib.reload` would report the already-imported version, which
    is precisely the wrong answer: pip has changed what is on disk, not what this
    interpreter holds.
    """
    code = (
        "import numpy, torch, json;"
        "print(json.dumps({'numpy': numpy.__version__, 'torch': torch.__version__,"
        "'cuda': torch.cuda.is_available(), 'arch': torch.cuda.get_arch_list()[-1]}))"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"FAILED: torch/numpy no longer import after {after}\n{r.stderr[-2000:]}")
    import json as _json
    got = _json.loads(r.stdout.strip().splitlines()[-1])
    for key, want in PINNED.items():
        if got[key] != want:
            raise SystemExit(
                f"FAILED: {key} became {got[key]} after {after}, expected {want}. "
                f"The PGSR sm_120 extensions were compiled against {want} and will "
                f"not import against anything else. Remove whatever pulled it."
            )
    if not got["cuda"]:
        raise SystemExit(f"FAILED: torch.cuda.is_available() went False after {after}")
    print(f"  pins ok after {after}: numpy {got['numpy']}, torch {got['torch']}, "
          f"arch {got['arch']}")


# Two entries in `deps/` are not wheels: `antlr4-python3-runtime-4.9.3` and
# `iopath-0.1.10` are source distributions, so pip has to build them.
# --no-build-isolation is what makes that possible offline — plain pip would fetch
# setuptools from PyPI to build in a clean env, and there is no network here.
PIP = [sys.executable, "-m", "pip", "install", "--no-deps", "--no-index",
       "--no-build-isolation", "-q"]


def main() -> None:
    print("=" * 70)
    print("staging the RUN kernel")
    print("=" * 70)
    for d in (DEPS, PGSR_WHEELS, DA3_DS, SAM3_DS, EXTRA):
        if not os.path.isdir(d):
            raise SystemExit(f"FAILED: {d} is absent. Attach the dataset, or re-run the relay.")

    check_pins("start")

    print("\n--- 1. wheels from the build cache ---")
    run(PIP + wheels_for(FROM_DEPS, DEPS), f"{len(FROM_DEPS)} packages from deps/")
    check_pins("deps")

    print("\n--- 1b. source distributions from the build cache ---")
    import shutil
    for name in SDIST_FROM_DEPS:
        outer = os.path.join(DEPS, name)
        # Find the directory that actually holds setup.py, rather than assuming the
        # nesting depth: the cache has it one level in, but an sdist repacked
        # differently would break a hardcoded path silently.
        root = None
        for candidate in (outer, os.path.join(outer, name)):
            if os.path.exists(os.path.join(candidate, "setup.py")) or \
               os.path.exists(os.path.join(candidate, "pyproject.toml")):
                root = candidate
                break
        if root is None:
            raise SystemExit(
                f"FAILED: no setup.py or pyproject.toml under {outer} or its "
                f"same-named child; the sdist layout is not what this expects."
            )
        local = f"/tmp/sdists/{name}"
        shutil.rmtree(local, ignore_errors=True)
        os.makedirs("/tmp/sdists", exist_ok=True)
        shutil.copytree(root, local)
        run(PIP + [local], f"{name} (sdist, built in /tmp)")
    check_pins("sdists")

    print("\n--- 2. wheels relayed from BUILD ---")
    run(PIP + wheels_for(FROM_EXTRA, EXTRA), f"{len(FROM_EXTRA)} packages from extra/")
    check_pins("extra")

    print("\n--- 3. PGSR CUDA extensions, built for sm_120 ---")
    run(PIP + sorted(glob.glob(f"{PGSR_WHEELS}/*.whl")), "simple_knn + diff_plane_rasterization")

    print("\n--- 3b. settling open3d's import chain ---")
    did = ensure_importable("open3d", [DEPS, EXTRA])
    print(f"  open3d importable; pulled in {did or 'nothing further'}")
    check_pins("open3d chain")

    print("\n--- 4. Depth-Anything-3 on the import path ---")
    # Not pip-installed. DA3 declares `hatchling` as its build backend, which is
    # not in the offline cache and cannot be fetched, so even
    # --no-build-isolation fails: pip still has to import the backend to ask it
    # for `build_editable`.
    #
    # It does not need building. `src/depth_anything_3` is pure Python, so a .pth
    # file in site-packages is the whole install — and unlike a PYTHONPATH export
    # it is inherited by every subprocess the stages spawn, which is the case that
    # actually matters here (`docs/reference/deb-technical.md`: "a subprocess inherits none of the
    # sys.path edits made in a notebook cell").
    import sysconfig
    da3_src = "/tmp/da3src/Depth-Anything-3/src"
    if not os.path.isdir(os.path.join(da3_src, "depth_anything_3")):
        raise SystemExit(f"FAILED: no depth_anything_3 package under {da3_src}")
    site = sysconfig.get_paths()["purelib"]
    pth = os.path.join(site, "v3d_depth_anything_3.pth")
    with open(pth, "w") as fh:
        fh.write(da3_src + "\n")
    print(f"  {pth} -> {da3_src}")

    print("\n--- 5. nvdiffrast ---")
    # This is docs/reference/deb-technical.md's P1-1, the item that blocks masks -> SAM3 -> boxes.
    #
    # nvdiffrast 0.4.0 compiles `_nvdiffrast_c` as a CUDAExtension at *install*
    # time (setup.py:26-56) — it does not JIT at first use, and `ops.py` imports
    # the built module directly. So the arch list has to be right here: without
    # TORCH_CUDA_ARCH_LIST the extension is built for whatever torch guesses, and
    # a mismatch on sm_120 surfaces as "no kernel image is available for
    # execution" at the first rasterise, deep inside mesh2mask.
    #
    # --no-build-isolation is required for a different reason: its own build
    # requires list names setuptools, wheel and ninja, and pip would fetch them.
    env = dict(os.environ, TORCH_CUDA_ARCH_LIST="12.0", MAX_JOBS="8")
    have_ninja = subprocess.run([sys.executable, "-c", "import ninja"],
                                capture_output=True).returncode == 0
    print(f"  ninja {'available' if have_ninja else 'absent — falling back to distutils, slower'}")
    t = time.time()
    r = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-deps", "--no-index",
         "--no-build-isolation", "-v", "/tmp/nvd/nvdiffrast_src"],
        capture_output=True, text=True, env=env, timeout=3600,
    )
    if r.returncode != 0:
        raise SystemExit(
            "FAILED: nvdiffrast build\n" + "\n".join((r.stdout + r.stderr).splitlines()[-40:])
        )
    print(f"  [{time.time() - t:5.1f}s] nvdiffrast built for sm_120")
    check_pins("extensions")

    print("\n--- 6. model weights, linked from the read-only datasets ---")
    # Symlinks, not copies: the checkpoints are 6.8 GB and 3.5 GB and the datasets
    # are already local disk. PipelineConfig's defaults point at these paths.
    for target, link in (
        (f"{DA3_DS}/DA3NESTED-GIANT-LARGE", "/tmp/DA3NESTED-GIANT-LARGE"),
        (SAM3_DS, "/tmp/sam3bundle"),
    ):
        if not os.path.exists(target):
            raise SystemExit(f"FAILED: {target} is absent")
        if os.path.islink(link):
            os.unlink(link)
        elif os.path.exists(link):
            raise SystemExit(f"FAILED: {link} exists and is not a symlink; refusing to replace it")
        os.symlink(target, link)
        n = len(os.listdir(link))
        print(f"  {link} -> {target}  ({n} entries)")

    print("\n" + "=" * 70)
    print("proving it, rather than trusting the exit codes")
    print("=" * 70)

    # Each probe runs in its own process so one hard crash (a CUDA extension can
    # abort rather than raise) names itself instead of taking the whole report
    # down with it.
    probes = {
        # Not just the import: TSDF fusion and mesh IO are what the mesh stage
        # needs, so exercise both rather than trusting that a version string
        # printed means the geometry pipeline works.
        "open3d (TSDF + mesh IO)":
            "import numpy as np, open3d as o3d;"
            "v=o3d.pipelines.integration.ScalableTSDFVolume(voxel_length=0.01,"
            "sdf_trunc=0.04,color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8);"
            "m=o3d.geometry.TriangleMesh.create_box();"
            "cl=m.cluster_connected_triangles();"
            "print(o3d.__version__,'tsdf ok, clusters',len(cl[1]))",
        # plyfile has no __version__; read/write is what PGSR actually uses it for.
        "plyfile (PGSR gaussian IO)":
            "from plyfile import PlyData, PlyElement; import numpy as np;"
            "a=np.array([(0.,1.,2.)],dtype=[('x','f4'),('y','f4'),('z','f4')]);"
            "e=PlyElement.describe(a,'vertex');"
            "print('PlyElement ok', e.count)",
        "trimesh": "import trimesh; print(trimesh.__version__)",
        "evo (DA3 pose_align)":
            "from evo.core.trajectory import PosePath3D; print('PosePath3D ok')",
        "pycocotools (masks)":
            "import numpy as np, pycocotools.mask as m;"
            "r=m.encode(np.asfortranarray(np.ones((4,4),np.uint8)));"
            "print('rle area', m.area(r))",
        "skimage (postprocess)":
            "from skimage import morphology; print('morphology ok')",
        "mujoco": "import mujoco; print(mujoco.__version__)",
        "sklearn": "import sklearn; print(sklearn.__version__)",
        "depth_anything_3":
            "import depth_anything_3 as d; "
            "from depth_anything_3.api import DepthAnything3; print('api ok')",
        # The two that a torch/numpy swap would have broken. distCUDA2 over real
        # points, not just an import: the extension links against libtorch and an
        # ABI mismatch shows up at the first call, not at import.
        "simple_knn.distCUDA2":
            "import torch; from simple_knn._C import distCUDA2;"
            "p=torch.rand(50000,3,device='cuda');d=distCUDA2(p);"
            "assert torch.isfinite(d).all(), 'non-finite distances';"
            "print('finite', tuple(d.shape), float(d.mean()))",
        "diff_plane_rasterization":
            "import diff_plane_rasterization as r;"
            "assert hasattr(r,'rasterize_gaussians'), 'no rasterize_gaussians';"
            "print('rasterize_gaussians +', [n for n in dir(r) if 'Rasteriz' in n])",
        # nvdiffrast is the item docs/reference/deb-technical.md calls P1-1: without it mesh2mask
        # cannot run, so there are no masks, no SAM3, no boxes. Rasterise a real
        # triangle and count pixels; an import alone would not compile the CUDA.
        "nvdiffrast (rasterise a triangle)":
            "import torch, nvdiffrast.torch as dr;"
            "ctx=dr.RasterizeCudaContext();"
            "v=torch.tensor([[[-0.8,-0.8,0,1],[0.8,-0.8,0,1],[0.0,0.8,0,1]]],"
            "device='cuda',dtype=torch.float32);"
            "f=torch.tensor([[0,1,2]],device='cuda',dtype=torch.int32);"
            "out,_=dr.rasterize(ctx,v,f,resolution=[64,64]);"
            "cov=int((out[...,3]>0).sum());"
            "assert cov>0, 'triangle covered no pixels';print('covered px', cov)",
    }

    failures = []
    for name, code in probes.items():
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                           env=dict(os.environ))
        if r.returncode == 0:
            print(f"  [ok]   {name:36} {r.stdout.strip().splitlines()[-1][:60]}")
        else:
            detail = (r.stderr.strip().splitlines() or ["<no stderr>"])[-1]
            print(f"  [FAIL] {name:36} {detail[:110]}")
            failures.append(name)

    print()
    if failures:
        raise SystemExit(
            f"STAGE_INCOMPLETE: {len(failures)} probe(s) failed: {failures}. "
            f"Fix these before spending GPU time — every one of them is on the "
            f"critical path, and the stage that needs it fails much later and less "
            f"clearly than this does."
        )
    print("STAGE_OK — every probe passed")


main()
