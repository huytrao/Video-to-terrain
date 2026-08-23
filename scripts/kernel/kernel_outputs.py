"""Every remaining deliverable, and an mp4 for each one.

Runs after the mesh exists and needs no semantics, which is the point: the object
cut is geometric, so this whole stage is unblocked by the gated SAM3 checkpoint.
Semantics, when it arrives, *labels* these objects rather than creating them.

What it produces. Every artefact gets an mp4, and every frame of every mp4 is
stamped with its units, because these lengths are in DA3's arbitrary units and a
frame shared out of context is exactly where an unlabelled number gets read as
metres:

    02  point cloud turntable    the 8 M-point cloud, seen
    03  mesh turntable           the fused surface on its own
    04  height map               the terrain a locomotion policy trains on
    05  slope map                where the floor is walkable
    06  step height map          the curriculum's input distribution
    07  mesh-guided masks        what render.py rasterised from the fused mesh
    08  SAM3 masks               text-prompted instances, coloured and named
    09  semantic 3D boxes        the objects, over the surface they came from
    10  MuJoCo rollout           the scene simulating, orbited, with |qvel| shown

(01 is the depth video, written by the DA3 stage.)

07 and 08 answer different questions and neither substitutes for the other: the
mesh-guided masks say where the *geometry* is trusted, the SAM3 masks say what the
*semantics* found. A class SAM3 never matched cannot become a box, so 08 is the
only place that failure is visible.

Each artefact is written, then *read back and measured*. Three of the visualisation
bugs in this project's history passed every automated check and were found only by
opening the PNG, so a render is verified against the geometry it claims to show
rather than against the fact that ffmpeg exited zero.

Every failure is caught per-output. One missing renderer should cost one video,
not the whole delivery.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, "/kaggle/working/src")
os.environ.setdefault("EGL_PLATFORM", "surfaceless")

SCENE = os.environ.get("V3D_SCENE", "small_factory")
WORK = Path(os.environ.get("V3D_WORK_ROOT", "/tmp/run"))
OUT = Path(os.environ.get("V3D_DELIVER", "/tmp/deliver"))
VID = OUT / "videos"
SCALE_MODE = os.environ.get("V3D_SCALE_MODE", "assumed")

import numpy as np  # noqa: E402

RESULTS: dict[str, object] = {}
FAILURES: dict[str, str] = {}

# Every frame of every video carries this. The reconstruction is metrically
# unanchored — a room and a photograph of a room twice its size produce identical
# images — so a viewer looking at a 0.9-unit crate must not read it as 0.9 m. This
# project has already shipped a room measured 8.6 m tall with a validator passing it
# 16/16, and the mechanism was exactly a unit claim nobody wrote down.
SCALE_CAPTION = ("lengths in DA3 units, NOT metres (scale = assumed)"
                 if SCALE_MODE == "assumed" else "lengths in metres")


def stamp(img, title: str, extra: str = "") -> None:
    """Burn the title and the units into a frame, in place.

    A caption in a filename or a README is a caption that gets separated from the
    picture the moment anyone shares one frame of it.
    """
    import cv2

    h, w = img.shape[:2]
    scale = max(0.45, min(0.85, w / 1600))
    pad = int(10 * scale) + 4
    lines = [title] + ([extra] if extra else []) + [SCALE_CAPTION]
    box_h = int((len(lines) * 26 + 12) * scale)
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
    y = int(20 * scale) + 6
    for i, line in enumerate(lines):
        # The units line is the one that must not be skimmed past, so it is amber.
        colour = (90, 200, 255) if line == SCALE_CAPTION else (255, 255, 255)
        cv2.putText(img, line, (pad, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55 * scale / 0.6, colour, 1, cv2.LINE_AA)
        y += int(24 * scale)


def class_colour(name: str) -> tuple[int, int, int]:
    """A stable colour per class, so the same object is the same colour throughout.

    Hashed rather than assigned from a palette in iteration order: the class list
    differs between the mesh-guided pass and SAM3, and a colour that shifts between
    two videos of the same scene is actively misleading.
    """
    import hashlib

    h = int(hashlib.sha1(name.encode()).hexdigest()[:6], 16)
    # Keep it bright: dark overlays vanish against a dim factory floor.
    return (60 + h % 180, 60 + (h >> 8) % 180, 60 + (h >> 16) % 180)


def restamp_video(path, title: str, extra: str = "") -> "Path | None":
    """Decode a rendered video, stamp every frame, re-encode in place.

    `visualize.turntable_video` renders through Open3D and writes its own frames to a
    temporary directory, so there is no hook to caption them on the way past. Rather
    than reimplement the renderer, the units are burned in afterwards — the caption
    has to travel with the pixels, because a single frame shared out of context is
    exactly where a unitless number gets read as metres.
    """
    import cv2

    path = Path(path)
    if not path.exists():
        return None
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 24
    fd = path.parent / f"_{path.stem}_stamp"
    fd.mkdir(parents=True, exist_ok=True)
    for old in fd.glob("*.png"):
        old.unlink()
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        stamp(frame, title, extra)
        cv2.imwrite(str(fd / f"{n:05d}.png"), frame)
        n += 1
    cap.release()
    if not n:
        return None
    return encode(fd, path, fps=int(round(fps)))


def point_cloud_turntable(cloud_path, out, n_frames: int = 90,
                          size=(1280, 900), max_points: int = 1_500_000):
    """Orbit a point cloud offscreen.

    `visualize.turntable_video` reads its argument with `read_triangle_mesh`, which
    returns an empty mesh for a point-cloud PLY and would render a blank orbit
    without erroring — so the cloud gets its own path here rather than being squeezed
    through that one.
    """
    import cv2
    import open3d as o3d

    cloud_path = Path(cloud_path)
    pcd = o3d.io.read_point_cloud(str(cloud_path))
    total = len(pcd.points)
    if total == 0:
        return None
    if total > max_points:
        pcd = pcd.uniform_down_sample(max(1, total // max_points))
    pts = np.asarray(pcd.points)
    centre = pts.mean(axis=0)
    radius = float(np.linalg.norm(pts - centre, axis=1).max()) * 1.35

    r = o3d.visualization.rendering.OffscreenRenderer(*size)
    r.scene.set_background([1.0, 1.0, 1.0, 1.0])
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    mat.point_size = 2.0
    r.scene.add_geometry("cloud", pcd, mat)

    el = np.radians(25.0)
    fd = Path(out).parent / f"_{Path(out).stem}_frames"
    fd.mkdir(parents=True, exist_ok=True)
    for old in fd.glob("*.png"):
        old.unlink()
    for i in range(n_frames):
        az = 2 * np.pi * i / n_frames
        eye = centre + radius * np.array(
            [np.cos(az) * np.cos(el), np.sin(az) * np.cos(el), np.sin(el)])
        r.scene.camera.look_at(centre.tolist(), eye.tolist(), [0, 0, 1])
        img = np.asarray(r.render_to_image())
        frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        stamp(frame, f"DA3 point cloud  ({total:,} points, "
                     f"{len(pcd.points):,} shown)",
              f"{cloud_path.stat().st_size / 1e6:.0f} MB on disk")
        cv2.imwrite(str(fd / f"{i:05d}.png"), frame)
    return encode(fd, Path(out), fps=20)


def step(name: str):
    """Wrap a stage so calling it records its result and cannot kill the others.

    The body runs when the returned function is CALLED, not when it is decorated.
    Executing inside the decorator — which an earlier version of this did — meant
    every step ran once at definition time and again at its explicit call site,
    doubling the GPU and render cost while looking merely slow.
    """
    def deco(fn):
        def run():
            print(f"\n{'=' * 78}\n{name}\n{'=' * 78}", flush=True)
            t = time.time()
            try:
                RESULTS[name] = fn()
                print(f"  [ok] {name} in {time.time() - t:.0f}s", flush=True)
            except Exception as exc:
                FAILURES[name] = f"{type(exc).__name__}: {exc}"
                print(f"  [FAIL] {name}: {type(exc).__name__}: {exc}", flush=True)
                traceback.print_exc()
            return RESULTS.get(name)
        run.__name__ = fn.__name__
        return run
    return deco


def encode(frames_dir: Path, out: Path, fps: int = 20) -> Path | None:
    pngs = sorted(frames_dir.glob("*.png"))
    if not pngs:
        return None
    out.parent.mkdir(parents=True, exist_ok=True)
    # Renumber into a contiguous sequence: ffmpeg's %05d pattern silently stops at
    # the first gap, producing a video of the first few frames that looks fine.
    for i, p in enumerate(pngs):
        want = frames_dir / f"seq_{i:05d}.png"
        if p != want:
            p.rename(want)
    r = subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps),
         "-i", str(frames_dir / "seq_%05d.png"),
         # libx264 with yuv420p requires both dimensions even, and a frame sized
         # from an aspect ratio is odd half the time — this failed at 1280x1275.
         # Cropping to even here rather than at every call site means no future
         # video can hit it: one pixel is not worth a lost artefact.
         "-vf", "crop=trunc(iw/2)*2:trunc(ih/2)*2",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(out)],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg: {r.stderr[-500:]}")
    if not out.exists() or out.stat().st_size == 0:
        raise RuntimeError(f"ffmpeg exited 0 but {out} is missing or empty")
    return out


def main() -> None:
    from v3d.config import PipelineConfig, detect_layout
    from v3d import paper_runner as pr

    cfg = PipelineConfig()
    layout = detect_layout(SCENE, cfg.data_root)
    model_dir = cfg.gs_root / layout.scene_id
    pruned = model_dir / "mesh" / "tsdf_fusion_pruned.ply"
    fused = model_dir / pr.MESH_PATH_REL
    mesh_path = pruned if pruned.exists() else fused
    if not mesh_path.exists():
        raise SystemExit(f"no mesh at {pruned} or {fused}; run the mesh stage first")
    # Two meshes, deliberately: the full one for geometry (the object cut and the
    # height map need every triangle) and the decimated one for rendering. A
    # turntable of 38 M triangles takes ~45 min for 90 frames and is
    # indistinguishable from the 1.5 M version at 1280x900.
    deliver_mesh = WORK / "mesh" / f"{SCENE}_deliver.ply"
    render_mesh = deliver_mesh if deliver_mesh.exists() else mesh_path
    print(f"analysis mesh {mesh_path.name} "
          f"({mesh_path.stat().st_size/1e6:.0f} MB)")
    print(f"render   mesh {render_mesh.name} "
          f"({render_mesh.stat().st_size/1e6:.0f} MB)")

    OUT.mkdir(parents=True, exist_ok=True)
    VID.mkdir(parents=True, exist_ok=True)
    print(f"scene {SCENE}\nmesh  {mesh_path}\nout   {OUT}")

    # Camera centres, read through the pipeline's own pose convention. Needed to
    # orient the up axis from evidence rather than from an assumption.
    w2c, K, n_images = pr._read_transforms(layout)
    centres = np.stack([np.linalg.inv(m)[:3, 3] for m in w2c])
    print(f"poses {n_images}, camera extent "
          f"{np.linalg.norm(centres.max(0) - centres.min(0)):.3f}")

    # ---------------------------------------------------------------- objects --
    @step("cut objects out of the room")
    def _cut():
        from v3d.scene import decompose as dc

        res = dc.cut(mesh_path, OUT, camera_centres=centres, scale_mode=SCALE_MODE)
        rep = dc.audit(res)
        path = dc.write(res, rep, OUT)
        print(f"  up axis {np.round(res.up_axis, 4).tolist()} "
              f"({res.up_axis_evidence['rule']}, "
              f"{res.up_axis_evidence['plane_source']})")
        print(f"  room height estimate {res.up_axis_evidence['room_height_estimate']:.3f} "
              f"({'DA3 units' if SCALE_MODE == 'assumed' else 'm'})")
        for p in res.shell_planes:
            print(f"    plane {p.role:16} pts {p.points:>9,} span_frac {p.span_frac:.3f} "
                  f"verticality {p.verticality:.3f}")
        d = res.diagnostics
        print(f"  shell removed {d['shell_triangles_removed']:,} of "
              f"{d['mesh_triangles']:,} triangles ({d['shell_fraction']:.1%})")
        print(f"  components {d['components_after_shell_removal']:,} -> "
              f"{d['objects_kept']} objects "
              f"(dropped {d['dropped_below_size_fraction']} small, "
              f"{d['dropped_too_thin']} thin)")
        print(f"  audit {rep['overall']}")
        for c in rep["checks"]:
            print(f"    [{c['status']}] {c['check']}: {c['detail'][:120]}")
        return {"json": str(path), "objects": len(res.objects),
                "audit": rep["overall"], "up_axis": res.up_axis,
                "floor_offset": res.floor_offset,
                "diagnostics": d}

    objects_doc = json.loads((OUT / "objects.json").read_text()) \
        if (OUT / "objects.json").exists() else None

    # ---------------------------------------------------------------- terrain --
    @step("height map, traversability and clearances")
    def _terrain():
        from v3d.export import terrain as tr

        if objects_doc is None:
            raise RuntimeError("the object cut did not produce objects.json")
        up = np.array(objects_doc["up_axis"], dtype=float)
        floor = float(objects_doc["floor_offset"])
        hm = tr.height_map(mesh_path, up, floor)
        hm["floor_offset"] = floor
        trav = tr.traversability(hm)
        clear = tr.clearances(objects_doc["objects"], objects_doc["room_extent"],
                              objects_doc["up_axis"], floor)
        png = tr.write_heightfield_png(hm, OUT / "terrain" / "heightfield.png")
        np.save(OUT / "terrain" / "height_map.npy", hm["grid"])
        np.save(OUT / "terrain" / "slope_deg.npy", trav["slope_deg"])
        np.save(OUT / "terrain" / "step.npy", trav["step"])
        summ = tr.summary(hm, trav, clear, SCALE_MODE)
        (OUT / "terrain" / "terrain.json").write_text(json.dumps(summ, indent=2))

        s = summ["traversability"]
        print(f"  grid {hm['nu']}x{hm['nv']} cell {hm['cell_size']:.5f}, "
              f"holes {hm['hole_fraction']:.2%}")
        print(f"  height range {hm['height_range'][0]:.3f} .. {hm['height_range'][1]:.3f}")
        print(f"  floor-like {s['floor_like_fraction']:.1%} of the room")
        c = s["curriculum"]
        print(f"  curriculum: flat<5deg {c['flat_fraction_under_5deg']:.1%}, "
              f"traversable<20deg {c['traversable_fraction_under_20deg']:.1%}, "
              f"step p95 {c['max_step_p95']:.4f}, slope p95 {c['max_slope_p95_deg']:.1f}deg")
        print(f"  clearances: min {clear.get('gap_min')}, "
              f"median {clear.get('gap_median')}, "
              f"overlapping pairs {clear.get('overlapping_pairs')}")

        # MJCF, and then actually compile and step it — an XML that does not load
        # is not a deliverable, and "physics explodes on the first step" is a
        # failure this project has specifically had before.
        mjcf = tr.write_mjcf(OUT / "mujoco" / "scene.xml", hm,
                             objects_doc["objects"], "heightfield.png",
                             scale_mode=SCALE_MODE)
        subprocess.run(["cp", str(png), str(OUT / "mujoco" / "heightfield.png")],
                       check=False)
        import mujoco

        model = mujoco.MjModel.from_xml_path(str(mjcf))
        data = mujoco.MjData(model)
        for _ in range(600):
            mujoco.mj_step(model, data)
        finite = bool(np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel)))
        maxv = float(np.abs(data.qvel).max()) if data.qvel.size else 0.0
        print(f"  MJCF {mjcf}: {model.nbody} bodies, {model.ngeom} geoms; "
              f"600 steps -> finite {finite}, max |qvel| {maxv:.3f}")
        if not finite:
            raise RuntimeError("physics diverged: qpos/qvel went non-finite")
        summ["mujoco"] = {"xml": str(mjcf), "bodies": int(model.nbody),
                          "geoms": int(model.ngeom), "steps": 600,
                          "finite": finite, "max_abs_qvel": maxv}
        (OUT / "terrain" / "terrain.json").write_text(json.dumps(summ, indent=2))
        return summ

    # ----------------------------------------------------------------- videos --
    @step("height map and slope videos")
    def _terrain_video():
        import cv2

        hm = np.load(OUT / "terrain" / "height_map.npy")
        slope = np.load(OUT / "terrain" / "slope_deg.npy")
        step_a = np.load(OUT / "terrain" / "step.npy")
        made = []

        def sweep(arr, name, label, cmap=cv2.COLORMAP_TURBO, vmax=None):
            lo = float(np.nanmin(arr))
            hi = float(vmax if vmax is not None else np.nanpercentile(arr, 99))
            norm = np.clip((arr - lo) / max(hi - lo, 1e-9), 0, 1)
            img = cv2.applyColorMap((norm * 255).astype(np.uint8), cmap)
            img = cv2.resize(img, (1280, int(1280 * arr.shape[0] / arr.shape[1])),
                             interpolation=cv2.INTER_NEAREST)
            fd = VID / f"_{name}_frames"
            fd.mkdir(parents=True, exist_ok=True)
            for old in fd.glob("*.png"):
                old.unlink()
            # A scan line sweeping across, with the profile under it: a still image
            # of a height map hides whether the relief is real or is reconstruction
            # noise, and a moving cross-section does not.
            h, w = img.shape[:2]
            n = 120
            for i in range(n):
                f = img.copy()
                x = int(i / (n - 1) * (w - 1))
                cv2.line(f, (x, 0), (x, h - 1), (255, 255, 255), 2)
                col = arr[:, int(i / (n - 1) * (arr.shape[1] - 1))]
                prof = np.nan_to_num(col, nan=lo)
                pts = [(int(j / max(len(prof) - 1, 1) * (w - 1)),
                        int(h - 1 - (v - lo) / max(hi - lo, 1e-9) * (h * 0.28)))
                       for j, v in enumerate(prof)]
                for a, b in zip(pts, pts[1:]):
                    cv2.line(f, a, b, (255, 255, 255), 1)
                cv2.putText(f, f"{label}   range {lo:.3f}..{hi:.3f}"
                               + ("   (DA3 units, not metres)" if SCALE_MODE == "assumed" else " m"),
                            (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.imwrite(str(fd / f"{i:05d}.png"), f)
            v = encode(fd, VID / f"{name}.mp4", fps=20)
            if v:
                made.append(str(v))

        sweep(hm, "04_height_map", "height map")
        sweep(slope, "05_slope_deg", "slope (degrees)", vmax=45.0)
        sweep(step_a, "06_step_height", "step height to neighbour")
        return made

    @step("point cloud, mesh and object turntables")
    def _turntables():
        from v3d.export import visualize as viz

        made = []
        print(f"  open3d selftest: {viz.selftest()}")

        # turntable_video takes a PATH and reads the mesh itself; handing it a mesh
        # object silently produces an empty scene.
        v = viz.turntable_video(str(render_mesh), str(VID / "03_mesh_turntable.mp4"),
                                n_frames=90)
        restamp_video(v, "fused TSDF mesh (no boxes)",
                      f"{mesh_path.name}")
        made.append(str(v))
        print(f"  mesh turntable -> {v}")

        cloud = layout.point_cloud
        if cloud.exists():
            v = point_cloud_turntable(cloud, VID / "02_point_cloud_turntable.mp4")
            if v:
                made.append(str(v))
                print(f"  point cloud turntable -> {v}")
        return made

    @step("semantic 3D boxes video")
    def _boxes_video():
        from v3d.export import visualize as viz

        # Prefer the labelled boxes from scene.json; fall back to the geometric cut,
        # which needs no semantics and therefore always exists.
        scene_path = WORK / "scene" / SCENE / "scene.json"
        raw, source = [], None
        if scene_path.exists():
            doc = json.loads(scene_path.read_text())
            # Real schema, read off the file build_scene writes:
            #   {id, category, box:{center,size,(axes)}, confidence, mesh_path, ...}
            for o in doc.get("objects", []):
                box = o.get("box") or {}
                c = box.get("center") or o.get("centre") or o.get("center")
                e = box.get("size") or o.get("size_m") or o.get("extent")
                a = box.get("axes") or box.get("R") or o.get("axes")
                if c and e:
                    raw.append((c, e, a,
                                o.get("category") or o.get("label") or "object"))
            if not raw and doc.get("objects"):
                raise RuntimeError(
                    f"scene.json has {len(doc['objects'])} objects but none exposed a "
                    f"box; keys on the first are {sorted(doc['objects'][0])}. Refusing "
                    f"to fall back to the geometric cut silently, which would report "
                    f"'geometric' for a run that has semantics."
                )
            source = "scene.json (semantic, labelled)"
        if not raw and objects_doc:
            raw = [(o["centre"], o["extent"], o.get("obb_axes"),
                    o.get("label", "unlabelled")) for o in objects_doc["objects"]]
            source = "objects.json (geometric cut, unlabelled)"
        if not raw:
            raise RuntimeError("no boxes from either scene.json or objects.json")

        # turntable_video's `boxes` wants objects exposing .center/.size/.axes, and
        # draws them as wireframes so the surface behind stays visible — which is the
        # whole point of showing a box over a mesh.
        class Box:
            __slots__ = ("center", "size", "axes", "label", "is_shell")

            def __init__(self, c, s, a, label):
                self.center = np.asarray(c, dtype=float)
                self.size = np.asarray(s, dtype=float)
                ax = np.asarray(a, dtype=float) if a is not None else np.eye(3)
                self.axes = ax if ax.shape == (3, 3) else np.eye(3)
                self.label = label
                self.is_shell = label in {"wall", "floor", "ceiling", "room"}

        boxes = [Box(*r) for r in raw]
        print(f"  {len(boxes)} boxes from {source}")
        by_label: dict[str, int] = {}
        for b in boxes:
            by_label[b.label] = by_label.get(b.label, 0) + 1
        for lbl, n in sorted(by_label.items(), key=lambda kv: -kv[1])[:12]:
            print(f"    {lbl:26} {n:>4}")

        v = viz.turntable_video(str(render_mesh), str(VID / "09_semantic_boxes.mp4"),
                                n_frames=90, boxes=boxes)
        print(f"  {v}")
        # The renderer writes its own frames, so the units caption goes on afterwards
        # by re-stamping the encoded video's frames.
        restamped = restamp_video(v, f"semantic 3D boxes  ({len(boxes)} objects)",
                                 f"from {source}")
        return {"video": str(restamped or v), "boxes": len(boxes), "source": source,
                "by_label": by_label}

    @step("mesh-guided mask and SAM3 overlay videos")
    def _mask_videos():
        import cv2

        made = []
        frames = sorted(layout.images_dir.glob("*.JPG"))

        def frame_for(stem: str):
            for ext in (".JPG", ".jpg", ".png"):
                p = layout.images_dir / f"{stem}{ext}"
                if p.exists():
                    return cv2.imread(str(p))
            return None

        # --- mesh-guided masks (what render.py rasterised from the fused mesh) ---
        mask_files = sorted(layout.mask_dir.glob("*")) if layout.mask_dir.is_dir() else []
        mask_files = [p for p in mask_files if p.suffix.lower() in
                      {".png", ".jpg", ".jpeg", ".npy"}]
        if mask_files:
            fd = VID / "_masks_frames"
            fd.mkdir(parents=True, exist_ok=True)
            for old in fd.glob("*.png"):
                old.unlink()
            pick = mask_files
            if len(pick) > 200:
                idx = np.linspace(0, len(pick) - 1, 200).round().astype(int)
                pick = [pick[i] for i in sorted(set(idx.tolist()))]
            n = 0
            for p in pick:
                rgb = frame_for(p.stem)
                if rgb is None:
                    continue
                m = (np.load(p) if p.suffix == ".npy"
                     else cv2.imread(str(p), cv2.IMREAD_UNCHANGED))
                if m is None:
                    continue
                if m.ndim == 3:
                    m = m[..., 0]
                if m.shape[:2] != rgb.shape[:2]:
                    m = cv2.resize(m, (rgb.shape[1], rgb.shape[0]),
                                   interpolation=cv2.INTER_NEAREST)
                covered = m > 0
                tint = rgb.copy()
                tint[covered] = (0.45 * tint[covered] +
                                 0.55 * np.array([80, 220, 120])).astype(np.uint8)
                out = cv2.resize(tint, (rgb.shape[1] // 2, rgb.shape[0] // 2))
                stamp(out, f"mesh-guided mask  {p.stem}",
                      f"mesh coverage {covered.mean() * 100:.1f}% of the frame")
                cv2.imwrite(str(fd / f"{n:05d}.png"), out)
                n += 1
            if n:
                v = encode(fd, VID / "07_mesh_guided_masks.mp4", fps=12)
                made.append(str(v))
                print(f"  mesh-guided masks: {n} frames -> {v}")
        else:
            print("  no mesh-guided masks on disk (mesh2mask has not run)")

        # --- SAM3 text-prompted masks, coloured and labelled per class ----------
        index = WORK / "sam_masks" / SCENE / SCENE / "mask_index.json"
        if not index.exists():
            print(f"  no SAM3 masks yet ({index} absent) — run the semantics stage")
            return made

        import pycocotools.mask as mask_utils

        idx_doc = json.loads(index.read_text())
        # Schema: {scene, image_root, mask_root, items:[{image,label,mask_path,
        # mask_rle:{size,counts},bbox,score}], missing_images}
        entries = idx_doc["items"] if isinstance(idx_doc, dict) and "items" in idx_doc \
            else (idx_doc if isinstance(idx_doc, list) else [])
        # Group by image so every mask on a frame is drawn in one pass.
        by_image: dict[str, list] = {}
        for e in entries:
            if not isinstance(e, dict):
                continue
            key = e.get("image") or e.get("file_name") or e.get("image_name") or ""
            by_image.setdefault(Path(str(key)).stem, []).append(e)
        print(f"  SAM3 index: {len(entries)} masks over {len(by_image)} images")

        fd = VID / "_sam3_frames"
        fd.mkdir(parents=True, exist_ok=True)
        for old in fd.glob("*.png"):
            old.unlink()
        stems = sorted(by_image)
        if len(stems) > 200:
            sel = np.linspace(0, len(stems) - 1, 200).round().astype(int)
            stems = [stems[i] for i in sorted(set(sel.tolist()))]
        n = 0
        for stem in stems:
            rgb = frame_for(stem)
            if rgb is None:
                continue
            canvas = rgb.copy()
            drawn = []
            for e in by_image[stem]:
                label = str(e.get("label") or e.get("class") or "?")
                rle = e.get("mask_rle") or e.get("segmentation") or e.get("rle")
                m = None
                if isinstance(rle, dict) and "counts" in rle:
                    r = dict(rle)
                    if isinstance(r["counts"], str):
                        r["counts"] = r["counts"].encode()
                    try:
                        m = mask_utils.decode(r).astype(bool)
                    except Exception:
                        m = None
                if m is None or not m.any():
                    continue
                if m.shape[:2] != canvas.shape[:2]:
                    m = cv2.resize(m.astype(np.uint8),
                                   (canvas.shape[1], canvas.shape[0]),
                                   interpolation=cv2.INTER_NEAREST).astype(bool)
                col = np.array(class_colour(label), dtype=float)
                canvas[m] = (0.5 * canvas[m] + 0.5 * col).astype(np.uint8)
                # Outline plus a text anchor at the mask's centroid: a tint alone
                # does not say which class it is, and colour-only legends are
                # unreadable for anyone with a colour deficiency.
                cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL,
                                         cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(canvas, cs, -1, tuple(int(c) for c in col), 2)
                ys, xs = np.nonzero(m)
                sc = e.get("score")
                tag = f"{label} {sc:.2f}" if isinstance(sc, (int, float)) else label
                drawn.append((tag, int(xs.mean()), int(ys.mean()), col))
            out = cv2.resize(canvas, (canvas.shape[1] // 2, canvas.shape[0] // 2))
            for label, cx, cy, col in drawn[:24]:
                cv2.putText(out, label, (cx // 2, cy // 2), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(out, label, (cx // 2, cy // 2), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, tuple(int(c) for c in col), 1, cv2.LINE_AA)
            stamp(out, f"SAM3 masks  {stem}", f"{len(drawn)} instances this frame")
            cv2.imwrite(str(fd / f"{n:05d}.png"), out)
            n += 1
        if n:
            v = encode(fd, VID / "08_sam3_masks.mp4", fps=12)
            made.append(str(v))
            print(f"  SAM3 overlay: {n} frames -> {v}")
        return made

    @step("MuJoCo physics rollout video")
    def _mujoco_rollout():
        import cv2
        import mujoco

        mjcf = OUT / "mujoco" / "scene.xml"
        if not mjcf.exists():
            raise RuntimeError(f"no MJCF at {mjcf}; the terrain step must run first")
        # EGL: the kernel has no display, and mujoco.Renderer needs a GL context.
        os.environ.setdefault("MUJOCO_GL", "egl")
        model = mujoco.MjModel.from_xml_path(str(mjcf))
        data = mujoco.MjData(model)
        W, H = 960, 540
        renderer = mujoco.Renderer(model, height=H, width=W)
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(model, cam)

        fd = VID / "_mujoco_frames"
        fd.mkdir(parents=True, exist_ok=True)
        for old in fd.glob("*.png"):
            old.unlink()

        n_frames, steps_per_frame = 180, 10
        max_qvel = 0.0
        for i in range(n_frames):
            for _ in range(steps_per_frame):
                mujoco.mj_step(model, data)
            if data.qvel.size:
                max_qvel = max(max_qvel, float(np.abs(data.qvel).max()))
            # Orbit so the whole scene is seen rather than one corner of it.
            cam.azimuth = 360.0 * i / n_frames
            cam.elevation = -20.0
            renderer.update_scene(data, camera=cam)
            img = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
            stamp(img, f"MuJoCo rollout  t = {data.time:.2f} s",
                  f"{model.nbody - 1} bodies, {model.ngeom} geoms, "
                  f"max |qvel| {max_qvel:.2f}")
            cv2.imwrite(str(fd / f"{i:05d}.png"), img)
        renderer.close()

        finite = bool(np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel)))
        v = encode(fd, VID / "10_mujoco_rollout.mp4", fps=30)
        print(f"  {n_frames * steps_per_frame} steps, sim time {data.time:.2f}s, "
              f"finite {finite}, max |qvel| {max_qvel:.3f}")
        print(f"  {v}")
        if not finite:
            raise RuntimeError("physics diverged during the rollout")
        return {"video": str(v), "steps": n_frames * steps_per_frame,
                "sim_time": float(data.time), "max_abs_qvel": max_qvel,
                "finite": finite,
                "note": ("gravity is -9.81 in DA3 units, so the dynamics are "
                         "proportionally right and not physically calibrated"
                         if SCALE_MODE == "assumed" else "metric")}

    _cut()
    # objects.json only exists after the cut, so re-read it for the later steps.
    if (OUT / "objects.json").exists():
        objects_doc = json.loads((OUT / "objects.json").read_text())
    _terrain()
    _terrain_video()
    _turntables()
    _mask_videos()
    _boxes_video()
    _mujoco_rollout()

    report = {
        "scene": SCENE,
        "scale_mode": SCALE_MODE,
        "mesh": str(mesh_path),
        "results": RESULTS,
        "failures": FAILURES,
        "videos": sorted(str(p) for p in VID.glob("*.mp4")),
    }
    (OUT / f"outputs_{SCENE}.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"\n{'=' * 78}")
    print(f"videos: {len(report['videos'])}")
    for v in report["videos"]:
        print(f"  {Path(v).name}  {Path(v).stat().st_size / 1e6:.1f} MB")
    if FAILURES:
        print(f"\n{len(FAILURES)} step(s) failed: {list(FAILURES)}")
    print(f"\n  -> {OUT / f'outputs_{SCENE}.json'}")
    print("\nOUTPUTS_OK" if not FAILURES else "\nOUTPUTS_PARTIAL")


main()
