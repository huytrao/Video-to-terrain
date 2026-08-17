"""Cutaway renders — the interior, which no delivered video has ever shown.

Both existing turntables put the camera outside the object radius, so every
frame is the outside of the ceiling and walls. This cuts the near half of the
room away and rotates the cut with the camera, so the furniture inside is the
subject.

Runs on the Kaggle GPU kernel, and must be launched as a **detached
subprocess**, not inside the IPython kernel:

    p = subprocess.Popen([sys.executable, "/kaggle/working/_run/kernel_cutaway.py"],
                         stdout=open(LOG, "w"), stderr=subprocess.STDOUT,
                         cwd="/kaggle/working", start_new_session=True)

Inside the notebook kernel, `open3d/__init__.py` detects Jupyter and calls
`enable_webrtc()` on a half-initialised module, which fails; detaching also
means a dropped websocket costs nothing.

Everything it needs already survives in /kaggle/working: the decimated mesh, the
annotations, the optimised cameras, and the three cached wheels — so it needs no
internet and no rebuild.
"""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
import time
import types

W = "/kaggle/working"
DELIVER = f"{W}/deliver_out"
MESH = f"{DELIVER}/mesh/room_300k.glb"
BOXES = f"{DELIVER}/metadata/ground_truth_boxes.json"
CAMS = f"{DELIVER}/metadata/cameras_optimised.json"
OUT_VIZ = f"{DELIVER}/viz"
OUT_VID = f"{DELIVER}/videos"
FRAMES = f"{DELIVER}/_cutaway_frames"
STUBS = f"{W}/stubs"
WHEELS = f"{W}/wheels"

# Offscreen rendering, decided before open3d loads its backend.
os.environ.setdefault("EGL_PLATFORM", "surfaceless")

# open3d/__init__.py imports open3d.visualization and open3d.ml unconditionally.
# Those reach dash (the interactive plotly viewer) and addict (the ml config
# loader) at module level — neither of which anything here calls, and neither of
# which can be pip-installed on a kernel with no internet. A permissive stub
# satisfies the import and raises nothing; if the inert code is ever reached it
# returns empty containers rather than pretending to work.
STUB = '''"""Auto-generated stub for `{name}` — not installed, and this kernel is offline.

Enough import surface for a module-level `from {name} import X` to succeed.
Nothing here computes anything. Written by scripts/kernel_cutaway.py.
"""


class _Any(dict):
    """Permissive: usable as a value, a callable, a class, or a dict."""

    def __init__(self, *a, **k):
        try:
            dict.__init__(self, *a, **k)
        except Exception:
            dict.__init__(self)

    def __getattr__(self, item):
        if item.startswith("__"):
            raise AttributeError(item)
        return _Any()

    def __setattr__(self, item, value):
        self[item] = value

    def __call__(self, *a, **k):
        return _Any()


def __getattr__(name):
    if name.startswith("__"):
        raise AttributeError(name)
    return _Any
'''


def _stub_module(dotted: str) -> None:
    """Write a stub package for `dotted`, creating parent packages as needed."""
    parts = dotted.split(".")
    d = STUBS
    for depth, part in enumerate(parts):
        d = os.path.join(d, part)
        os.makedirs(d, exist_ok=True)
        init = os.path.join(d, "__init__.py")
        if not os.path.exists(init):
            with open(init, "w") as fh:
                fh.write(STUB.format(name=".".join(parts[: depth + 1])))


# open3d/__init__.py:163 runs a bare `import open3d.ml`, which pulls in the whole
# Open3D-ML tree — SemanticKITTI, torch dataset base classes, addict, yapf — none
# of which offscreen rendering touches. Stubbing that many third-party packages
# just to import a subtree nothing calls is the wrong trade, so the subtree itself
# is replaced by an inert module. open3d already treats it as optional:
# `_insert_pybind_names(skip_names=("ml",))` at line 129 skips it deliberately.
BLOCKED = ("open3d.ml",)


def _apply_blocks() -> None:
    """Pre-seed sys.modules so `import <blocked>` is satisfied without executing it.

    An empty `__path__` makes it a package with no contents, so an import of
    anything *under* a blocked name fails loudly instead of returning a stub.
    """
    for dotted in BLOCKED:
        existing = sys.modules.get(dotted)
        if existing is not None and getattr(existing, "_v3d_inert", False):
            continue
        mod = types.ModuleType(dotted)
        mod.__doc__ = f"Inert placeholder for {dotted} — see scripts/kernel_cutaway.py."
        mod.__path__ = []
        mod._v3d_inert = True
        sys.modules[dotted] = mod


def _install_from_cache(module: str) -> bool:
    """Install a real wheel for `module` if one is cached. Prefer this to a stub."""
    if not os.path.isdir(WHEELS):
        return False
    prefix = module.split(".")[0].replace("_", "-").lower()
    for name in sorted(os.listdir(WHEELS)):
        if not name.endswith(".whl"):
            continue
        dist = name.split("-")[0].replace("_", "-").lower()
        if dist != prefix:
            continue
        r = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-deps", "--no-index", "-q",
             os.path.join(WHEELS, name)],
            capture_output=True, text=True,
        )
        if r.returncode == 0:
            return True
        print(f"  cached wheel {name} failed to install: {r.stderr.strip()[-400:]}")
    return False


def _probe_missing(module: str) -> str | None:
    """Import `module` in a fresh process; name the dependency it wants, or None.

    A throwaway process rather than a try/except in this one. A failed import
    leaves partially initialised modules behind, and clearing them out of
    `sys.modules` to retry is *not* safe for a C extension: open3d's pybind11
    layer registers `open3d.cuda.pybind.visualization` and friends during its
    one-time module init, and removing those entries makes them unrecoverable —
    the re-import then fails with "'open3d.cuda.pybind' is not a package",
    which looks like a broken install and is not one.
    """
    code = (
        "import sys\n"
        f"sys.path.insert(0, {STUBS!r})\n"
        "try:\n"
        f"    import {module}\n"
        "except ModuleNotFoundError as exc:\n"
        "    print('V3D_MISSING:' + (exc.name or ''))\n"
        "    raise SystemExit(0)\n"
        "print('V3D_IMPORT_OK')\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    for line in r.stdout.splitlines():
        if line.startswith("V3D_MISSING:"):
            return line[len("V3D_MISSING:"):]
        if line == "V3D_IMPORT_OK":
            return None
    raise SystemExit(
        f"probing `import {module}` failed with something other than a missing "
        f"module:\n--- stdout ---\n{r.stdout[-2000:]}\n--- stderr ---\n{r.stderr[-4000:]}"
    )


def ensure_importable(module: str, max_rounds: int = 40) -> list[str]:
    """Make `module` importable, satisfying each dependency it names in turn.

    One call instead of one round trip to this kernel per missing module.
    Returns what it had to do, so the log states plainly which parts are inert.
    """
    if STUBS not in sys.path:
        sys.path.insert(0, STUBS)
    actions: list[str] = []
    for _ in range(max_rounds):
        missing = _probe_missing(module)
        if missing is None:
            # Every dependency is on disk; this process imports it exactly once.
            importlib.import_module(module)
            return actions
        # A missing *submodule* of the target is a broken install, not an absent
        # third-party package. Don't paper over it with a stub.
        if missing == module or missing.startswith(module + "."):
            raise SystemExit(
                f"{module} is installed but incomplete: it cannot import its own "
                f"submodule {missing}. Reinstall it; a stub would hide the fault."
            )
        if _install_from_cache(missing):
            actions.append(f"wheel:{missing}")
        else:
            _stub_module(missing)
            actions.append(f"stub:{missing}")
    raise SystemExit(f"could not import {module} in {max_rounds} rounds; did: {actions}")


def ensure_open3d() -> None:
    """Install from the cached wheel, then import it. The kernel has no internet."""
    if not any(os.path.exists(os.path.join(p, "open3d")) for p in sys.path if p):
        wheel = f"{WHEELS}/open3d-0.19.0-cp312-cp312-manylinux_2_31_x86_64.whl"
        if not os.path.exists(wheel):
            raise SystemExit(f"open3d absent and no cached wheel at {wheel}")
        t = time.time()
        r = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-deps", "--no-index", "-q", wheel],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            raise SystemExit(f"pip install failed: {r.stdout[-2000:]}{r.stderr[-2000:]}")
        print(f"installed open3d from cache in {time.time() - t:.0f}s", flush=True)
    did = ensure_importable("open3d")
    print(f"open3d imported; dependency fixes: {did or 'none needed'}", flush=True)


def check_framing(viz, paths, min_fill: float = 0.88) -> None:
    """Require the subject to actually fill the frame, measured in pixels.

    The first cutaway stills were correct in every other respect and put the
    room in the middle third of the frame: 29% of the height, 12% of the pixels,
    with the ground-truth boxes running off the edges. Geometry-side framing
    cannot see that — only the rendered image can.

    A subject that reaches the corners defeats the background sampling, so
    `image_content_extent` raises rather than guess. That is the one failure that
    means the framing is *too* tight rather than too loose, so it counts as a
    pass and says so.
    """
    worst = None
    for p in paths:
        name = os.path.basename(str(p))
        try:
            m = viz.image_content_extent(p)
        except ValueError as exc:
            print(f"  {name}: fills the frame corner to corner ({exc})", flush=True)
            continue
        fill = max(m["width_fraction"], m["height_fraction"])
        print(f"  {name}: fill={fill:.3f} "
              f"w={m['width_fraction']} h={m['height_fraction']} "
              f"px={m['coverage_fraction']} edge={m['touches_edge']}", flush=True)
        if worst is None or fill < worst[1]:
            worst = (name, fill)
    if worst and worst[1] < min_fill:
        raise SystemExit(
            f"framing still loose: {worst[0]} fills only {worst[1]:.1%} of its "
            f"frame, wanted >= {min_fill:.0%}. The solver guarantees "
            f"{1 / 1.06:.1%} on the binding axis, so either the renderer is not "
            f"honouring set_projection or the content being framed is not the "
            f"content being drawn."
        )


def main() -> None:
    for p in (MESH, BOXES, CAMS):
        if not os.path.exists(p):
            raise SystemExit(f"missing input: {p}")
    ensure_open3d()

    sys.path.insert(0, f"{W}/src")
    from v3d.export import visualize as viz
    from v3d.validate.gt_boxes import load_gt

    print("selftest:", viz.selftest(), flush=True)

    boxes = load_gt(BOXES)
    objects = [b for b in boxes if not b.is_shell]
    print(f"{len(boxes)} annotations, {len(objects)} of them objects", flush=True)

    # A .glb is Y-up; the annotations are not. Resolve it against the boxes
    # rather than assuming, and let it raise if they cannot be reconciled.
    mesh = viz.load_mesh_in_box_frame(MESH, boxes=boxes)
    print("mesh triangles:", len(mesh.triangles), flush=True)

    os.makedirs(OUT_VIZ, exist_ok=True)
    os.makedirs(OUT_VID, exist_ok=True)

    # Open3D writes a .glb it cannot read back (see bugs-found B9), so the
    # delivery gets a PLY too: no up-axis convention, no chunked container,
    # nothing to misinterpret — and already in the annotation frame, so a later
    # stage needs no correction at all. Verified by reading it back, because a
    # write that cannot verify its own result is exactly how the .glb shipped
    # broken in the first place.
    import open3d as o3d

    ply = f"{DELIVER}/mesh/room_300k_boxframe.ply"
    o3d.io.write_triangle_mesh(ply, mesh, write_ascii=False)
    back = viz.read_triangle_mesh(ply, quiet=True)
    if (len(back.vertices), len(back.triangles)) != (len(mesh.vertices), len(mesh.triangles)):
        raise SystemExit(
            f"{ply} does not read back: wrote {len(mesh.vertices):,}v/"
            f"{len(mesh.triangles):,}t, read {len(back.vertices):,}v/{len(back.triangles):,}t"
        )
    print(f"wrote {os.path.basename(ply)} ({os.path.getsize(ply) / 1e6:.1f}MB), "
          f"read back {len(back.vertices):,} verts intact", flush=True)

    # Stills first: four sections at high resolution, cheap, and they show
    # immediately whether the cut is on the right side of the room.
    t = time.time()
    stills = viz.cutaway_views(
        mesh, OUT_VIZ, n_views=4, boxes=boxes, cameras_json=CAMS,
        max_triangles=None, stem="cutaway",
    )
    print(f"{len(stills)} stills in {time.time() - t:.0f}s", flush=True)
    check_framing(viz, stills)

    # Then the same thing without boxes, so the geometry is judged on its own.
    t = time.time()
    plain = viz.cutaway_views(
        mesh, OUT_VIZ, n_views=4, boxes=None, cameras_json=CAMS,
        max_triangles=None, stem="cutaway_nobox",
    )
    print(f"{len(plain)} plain stills in {time.time() - t:.0f}s", flush=True)
    check_framing(viz, plain)

    # Frames are reused across runs so a dropped connection is cheap, which means
    # a framing change would silently keep every old frame. Clear them.
    if os.path.isdir(FRAMES):
        stale = [f for f in os.listdir(FRAMES) if f.endswith(".png")]
        for f in stale:
            os.remove(os.path.join(FRAMES, f))
        print(f"cleared {len(stale)} stale frames from {FRAMES}", flush=True)

    t = time.time()
    video = viz.cutaway_turntable(
        mesh, f"{OUT_VID}/07_cutaway_interior.mp4",
        n_frames=72, boxes=boxes, cameras_json=CAMS,
        max_triangles=None, frame_dir=FRAMES,
    )
    print(f"video {video} {os.path.getsize(video) / 1e6:.1f}MB in {time.time() - t:.0f}s",
          flush=True)
    check_framing(viz, [f"{FRAMES}/0000.png", f"{FRAMES}/0018.png", f"{FRAMES}/0036.png"])
    print("CUTAWAY_OK", flush=True)


main()
