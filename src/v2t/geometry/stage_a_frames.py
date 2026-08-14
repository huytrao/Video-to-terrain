"""Stage A: Video preprocessing - extract frames, sample, metadata."""
from __future__ import annotations
import json
import subprocess
from pathlib import Path
from dataclasses import dataclass, asdict
import cv2
from src.v2t.config import CFG


@dataclass
class FrameMeta:
    frame_idx: int
    timestamp: float
    width: int
    height: int
    is_blurry: bool = False
    laplacian_var: float = 0.0


@dataclass
class CaptureMetadata:
    video_path: str
    total_frames: int
    fps: float
    duration: float
    width: int
    height: int
    sampled_frames: list[FrameMeta]
    sample_rate: int


def extract_frames(video_path: Path, frames_dir: Path, sample_rate: int = 5, max_frames: int = 100) -> CaptureMetadata:
    """Extract frames from video with sampling and blur detection."""
    frames_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0

    print(f"Video: {total_frames} frames, {fps:.2f} fps, {width}x{height}, {duration:.2f}s")

    sampled = []
    frame_idx = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_rate == 0:
            # Blur detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            is_blurry = bool(laplacian_var < 100.0)

            frame_name = f"frame_{saved_count:06d}.jpg"
            frame_path = frames_dir / frame_name
            cv2.imwrite(str(frame_path), frame)

            timestamp = frame_idx / fps if fps > 0 else 0
            sampled.append(FrameMeta(
                frame_idx=saved_count,
                timestamp=timestamp,
                width=width,
                height=height,
                is_blurry=is_blurry,
                laplacian_var=laplacian_var
            ))

            saved_count += 1
            if saved_count >= max_frames:
                break

        frame_idx += 1

    cap.release()

    meta = CaptureMetadata(
        video_path=str(video_path),
        total_frames=total_frames,
        fps=fps,
        duration=duration,
        width=width,
        height=height,
        sampled_frames=sampled,
        sample_rate=sample_rate
    )

    # Save metadata
    meta_path = frames_dir.parent / "capture_metadata.json"
    with open(meta_path, 'w') as f:
        json.dump(asdict(meta), f, indent=2)

    print(f"Extracted {saved_count} frames to {frames_dir}")
    print(f"Metadata saved to {meta_path}")
    return meta


def run_stage_a():
    """Entry point for Stage A."""
    paths = CFG.paths
    paths.frames.mkdir(parents=True, exist_ok=True)

    meta = extract_frames(
        CFG.input_video,
        paths.frames,
        sample_rate=CFG.frame_sample_rate,
        max_frames=CFG.max_frames
    )
    return meta


if __name__ == "__main__":
    run_stage_a()
