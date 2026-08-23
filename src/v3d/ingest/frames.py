"""Turn an arbitrary video into the frame set a reconstruction can actually use.

Three things decide whether a video reconstructs well, and none of them is the
frame count: frames must be *sharp*, they must be *correctly exposed*, and
consecutive kept frames must be separated by real *parallax*. A pan contributes
no parallax at all, so a thousand frames of a slow sweep triangulate no better
than one.

This module measures all three from the video itself and selects on the
measurements. Every threshold is a fraction of a quantity the clip supplies — the
clip's own median sharpness, the clip's own accumulated parallax — because a
threshold in absolute units is a threshold measured in one room, and this has to
work for any video.

Nothing here is specific to the input at hand: the selector is told how many
frames it may keep and what fraction of typical sharpness it must insist on, and
it finds the parallax spacing that satisfies both.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path

import cv2
import numpy as np


# --- what we measure ---------------------------------------------------------


@dataclass
class FrameStat:
    """One sampled frame, and how it compares to its predecessor."""

    index: int          # frame number in the source video
    time_s: float
    sharpness: float    # variance of the Laplacian; higher is sharper
    blown: float        # fraction of pixels at or near white
    dark: float         # fraction of pixels at or near black
    pan_px: float       # magnitude of the global image shift since the last sample
    parallax_px: float  # median residual motion once that global shift is removed


@dataclass
class FrameSpec:
    """How to choose frames. Every field is a ratio or a count, never a metre."""

    sample_stride: int = 3          # measure every Nth decoded frame
    max_frames: int = 600           # stay under the 800 that triggers chunking
    min_frames: int = 60
    min_sharpness_frac: float = 0.5   # of the clip's median sharpness
    max_blown: float = 0.12         # reject if >12% of pixels are clipped white
    max_dark: float = 0.60
    measure_width: int = 480        # motion is measured on a small copy; cheap and stable
    suffix: str = ".JPG"            # ScanNet++ uses uppercase JPG; the proven path
    jpeg_quality: int = 100

    def __post_init__(self) -> None:
        if self.min_frames > self.max_frames:
            raise ValueError(
                f"min_frames={self.min_frames} exceeds max_frames={self.max_frames}"
            )
        if not 0 < self.min_sharpness_frac <= 1:
            raise ValueError("min_sharpness_frac is a fraction of the clip median, in (0, 1]")


# --- measurement -------------------------------------------------------------


def measure(video: Path, spec: FrameSpec, progress: int = 300) -> tuple[list[FrameStat], dict]:
    """Walk the video once, reporting per-sample quality and inter-sample motion.

    Motion is split deliberately. The median flow vector is the part of the
    movement shared by the whole image — a pan, a rotation, a camera shake — and
    triangulation gains nothing from it. What is left after subtracting it varies
    with depth, which is precisely the signal a reconstruction needs. Reporting
    only the raw flow magnitude would rate a tripod pan as excellent input.
    """
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    declared = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not fps or fps <= 0 or not math.isfinite(fps):
        raise RuntimeError(f"{video} reports fps={fps!r}; cannot convert frames to time")

    stats: list[FrameStat] = []
    prev_small = None
    width = height = None
    decoded = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if decoded % spec.sample_stride == 0:
            if width is None:
                height, width = frame.shape[:2]
            elif (frame.shape[1], frame.shape[0]) != (width, height):
                # The transforms format carries one w/h for the whole scene and
                # the loader hard-asserts every image matches it, so a resolution
                # change mid-video has to be caught here rather than 200 frames
                # into a GPU stage.
                raise RuntimeError(
                    f"{video} changes resolution at frame {decoded}: "
                    f"{frame.shape[1]}x{frame.shape[0]} after {width}x{height}"
                )

            grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            small_h = max(2, round(spec.measure_width * height / width))
            small = cv2.resize(grey, (spec.measure_width, small_h),
                               interpolation=cv2.INTER_AREA)

            pan = parallax = 0.0
            if prev_small is not None:
                flow = cv2.calcOpticalFlowFarneback(
                    prev_small, small, None, 0.5, 3, 21, 3, 5, 1.2, 0
                )
                gx = float(np.median(flow[..., 0]))
                gy = float(np.median(flow[..., 1]))
                pan = math.hypot(gx, gy)
                residual = flow - np.array([gx, gy], np.float32)
                parallax = float(np.median(np.linalg.norm(residual, axis=2)))

            stats.append(FrameStat(
                index=decoded,
                time_s=decoded / fps,
                sharpness=float(cv2.Laplacian(grey, cv2.CV_64F).var()),
                blown=float((grey >= 250).mean()),
                dark=float((grey <= 12).mean()),
                pan_px=pan,
                parallax_px=parallax,
            ))
            prev_small = small
            if progress and len(stats) % progress == 0:
                print(f"    measured {len(stats)} samples ({decoded} frames decoded)")
        decoded += 1

    cap.release()
    if not stats:
        raise RuntimeError(f"{video} yielded no decodable frames")

    source = {
        "path": str(video),
        "frames_decoded": decoded,
        "frames_declared": declared,
        "fps": fps,
        "width": width,
        "height": height,
        "duration_s": decoded / fps,
        "samples": len(stats),
        "sample_stride": spec.sample_stride,
        # Measured at measure_width, so it is a relative quantity. Recorded so a
        # later reader knows the parallax figures are not in source pixels.
        "motion_measured_at_width": spec.measure_width,
    }
    return stats, source


# --- selection ---------------------------------------------------------------


@dataclass
class Selection:
    kept: list[FrameStat]
    parallax_target: float
    rejected_blur: int
    rejected_exposure: int
    rejected_spacing: int
    median_sharpness: float
    notes: list[str] = field(default_factory=list)


def _usable(stats: list[FrameStat], spec: FrameSpec) -> tuple[list[bool], float, int, int]:
    """Mark frames good enough to keep, on thresholds relative to this clip."""
    sharp = np.array([s.sharpness for s in stats], dtype=float)
    median_sharp = float(np.median(sharp))
    floor = spec.min_sharpness_frac * median_sharp

    usable, blur, exposure = [], 0, 0
    for s in stats:
        if s.sharpness < floor:
            usable.append(False)
            blur += 1
        elif s.blown > spec.max_blown or s.dark > spec.max_dark:
            usable.append(False)
            exposure += 1
        else:
            usable.append(True)
    return usable, median_sharp, blur, exposure


def _walk(stats: list[FrameStat], usable: list[bool], target: float) -> list[int]:
    """Keep a usable frame each time `target` parallax has accumulated since the last.

    Parallax accrues over *every* interval, including intervals whose frames were
    rejected for blur — the camera still travelled through them. So a blurred
    stretch is skipped without losing the baseline it contributed, and the next
    sharp frame is kept on the far side of it.
    """
    chosen: list[int] = []
    accumulated = 0.0
    for i, s in enumerate(stats):
        accumulated += s.parallax_px
        if not usable[i]:
            continue
        if not chosen:                      # first usable frame anchors the walk
            chosen.append(i)
            accumulated = 0.0
        elif accumulated >= target:
            chosen.append(i)
            accumulated = 0.0
    return chosen


def select(stats: list[FrameStat], spec: FrameSpec) -> Selection:
    """Choose frames spaced by parallax rather than by time.

    Even temporal sampling is the wrong default: it spends frames on the seconds
    where the camera hovered and starves the seconds where it moved. Spacing by
    accumulated parallax samples the *trajectory* evenly instead.

    The spacing that yields exactly `max_frames` is not known in advance, so it
    is searched for. Truncating an over-long list would have been simpler and
    would have silently discarded the end of the video — that is, part of the
    room.
    """
    usable, median_sharp, blur, exposure = _usable(stats, spec)
    n_usable = sum(usable)
    notes: list[str] = []

    if n_usable == 0:
        raise RuntimeError(
            f"no frame in {len(stats)} samples passed screening "
            f"(median sharpness {median_sharp:.1f}, floor "
            f"{spec.min_sharpness_frac * median_sharp:.1f}); the clip is unusable "
            f"as configured"
        )

    # target=0 keeps every usable frame: the ceiling on what selection can return.
    at_zero = _walk(stats, usable, 0.0)
    if len(at_zero) <= spec.max_frames:
        chosen, target = at_zero, 0.0
        if len(chosen) < spec.min_frames:
            notes.append(
                f"only {len(chosen)} frames survive screening, below min_frames="
                f"{spec.min_frames}; the reconstruction will be sparse and this is "
                f"a property of the input, not of the settings"
            )
    else:
        # Monotone in target: raising it can only keep fewer. Bisect on it.
        total_parallax = sum(s.parallax_px for s in stats)
        lo, hi = 0.0, max(total_parallax, 1e-6)
        chosen = at_zero
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            got = _walk(stats, usable, mid)
            if len(got) > spec.max_frames:
                lo = mid
            else:
                hi = mid
                chosen = got
            if len(got) == spec.max_frames:
                break
        target = hi
        if len(chosen) > spec.max_frames:      # bisection could not get under the cap
            raise RuntimeError(
                f"could not reduce selection below max_frames={spec.max_frames}; "
                f"got {len(chosen)}"
            )

    return Selection(
        kept=[stats[i] for i in chosen],
        parallax_target=target,
        rejected_blur=blur,
        rejected_exposure=exposure,
        rejected_spacing=n_usable - len(chosen),
        median_sharpness=median_sharp,
        notes=notes,
    )


# --- extraction --------------------------------------------------------------


def extract(video: Path, selection: Selection, images_dir: Path, spec: FrameSpec) -> list[str]:
    """Write the selected frames at full resolution, and verify what was written.

    Decoding is sequential rather than by seek: seeking a long-GOP H.264 stream
    lands on the wrong frame often enough that the pose attached to an image
    would describe a different image, which is a class of error no downstream
    check would catch.
    """
    wanted = {s.index: n for n, s in enumerate(selection.kept)}
    images_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video}")

    params = ([int(cv2.IMWRITE_JPEG_QUALITY), spec.jpeg_quality]
              if spec.suffix.lower() in {".jpg", ".jpeg"} else [])
    names: list[str | None] = [None] * len(selection.kept)
    shapes: set[tuple[int, int]] = set()
    decoded = 0

    while wanted:
        ok, frame = cap.read()
        if not ok:
            break
        slot = wanted.pop(decoded, None)
        if slot is not None:
            name = f"frame_{slot:05d}{spec.suffix}"
            out = images_dir / name
            if not cv2.imwrite(str(out), frame, params):
                raise RuntimeError(f"failed to write {out}")
            names[slot] = name
            shapes.add((frame.shape[1], frame.shape[0]))
        decoded += 1
    cap.release()

    if wanted:
        raise RuntimeError(
            f"video ended before {len(wanted)} selected frames were reached "
            f"(first missing source index {min(wanted)})"
        )
    if any(n is None for n in names):
        raise RuntimeError("internal: a selected slot received no image")
    if len(shapes) != 1:
        raise RuntimeError(f"extracted frames have differing sizes: {sorted(shapes)}")

    # Written, not merely returned: the loader asserts every image matches the
    # json's w/h, and a zero-byte or truncated write would pass that assert only
    # to fail inside the model.
    for name in names:
        p = images_dir / str(name)
        if p.stat().st_size == 0:
            raise RuntimeError(f"{p} was written empty")
    return [str(n) for n in names]


# --- the whole stage ---------------------------------------------------------


def ingest(video: Path, images_dir: Path, spec: FrameSpec | None = None) -> dict:
    """Measure, select, extract. Returns a manifest describing what happened."""
    spec = spec or FrameSpec()
    video = Path(video)
    if not video.exists():
        raise FileNotFoundError(video)

    print(f"  measuring {video.name} ...")
    stats, source = measure(video, spec)

    sharp = np.array([s.sharpness for s in stats])
    par = np.array([s.parallax_px for s in stats[1:]])
    pan = np.array([s.pan_px for s in stats[1:]])
    print(f"    {source['frames_decoded']} frames, {source['duration_s']:.2f} s, "
          f"{source['width']}x{source['height']} @ {source['fps']:.3f} fps")
    print(f"    sharpness  median {np.median(sharp):.1f}  p10 {np.percentile(sharp,10):.1f}  "
          f"max {sharp.max():.1f}")
    print(f"    motion     pan median {np.median(pan):.2f} px  "
          f"parallax median {np.median(par):.2f} px  "
          f"ratio {np.median(par)/max(np.median(pan),1e-9):.3f}")

    sel = select(stats, spec)
    print(f"  selected {len(sel.kept)} frames "
          f"(parallax spacing {sel.parallax_target:.2f} px); rejected "
          f"{sel.rejected_blur} blurred, {sel.rejected_exposure} mis-exposed, "
          f"{sel.rejected_spacing} too close")
    for note in sel.notes:
        print(f"    note: {note}")

    names = extract(video, sel, images_dir, spec)
    print(f"  wrote {len(names)} images to {images_dir}")

    return {
        "source": source,
        "spec": asdict(spec),
        "selection": {
            "kept": len(sel.kept),
            "parallax_target_px": sel.parallax_target,
            "rejected_blurred": sel.rejected_blur,
            "rejected_exposure": sel.rejected_exposure,
            "rejected_spacing": sel.rejected_spacing,
            "median_sharpness": sel.median_sharpness,
            "notes": sel.notes,
        },
        "frames": [
            {"file_path": name, "source_index": s.index, "time_s": s.time_s,
             "sharpness": s.sharpness, "blown": s.blown,
             "parallax_px": s.parallax_px}
            for name, s in zip(names, sel.kept)
        ],
    }


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("video", type=Path)
    ap.add_argument("--images-dir", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, help="where to write the ingest manifest")
    ap.add_argument("--max-frames", type=int, default=FrameSpec.max_frames)
    ap.add_argument("--min-frames", type=int, default=FrameSpec.min_frames)
    ap.add_argument("--stride", type=int, default=FrameSpec.sample_stride)
    ap.add_argument("--min-sharpness-frac", type=float, default=FrameSpec.min_sharpness_frac)
    ap.add_argument("--max-blown", type=float, default=FrameSpec.max_blown)
    ap.add_argument("--suffix", default=FrameSpec.suffix)
    args = ap.parse_args(argv)

    spec = FrameSpec(
        sample_stride=args.stride,
        max_frames=args.max_frames,
        min_frames=args.min_frames,
        min_sharpness_frac=args.min_sharpness_frac,
        max_blown=args.max_blown,
        suffix=args.suffix,
    )
    manifest = ingest(args.video, args.images_dir, spec)
    if args.manifest:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(manifest, indent=2))
        print(f"  manifest -> {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
