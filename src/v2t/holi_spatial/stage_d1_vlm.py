"""Stage D1: VLM Labeling via NVIDIA NIM API with 3-key rotation pool.

Reads frames from CFG.paths.frames, samples up to 20 frames, calls NVIDIA NIM
chat completions to identify visible objects with 2D bounding boxes,
outputs vlm_labels.json to CFG.paths.holi_spatial.
"""
from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Optional

from src.v2t.config import CFG
from src.v2t.nvidia_client import (
    call_vlm,
    is_keys_configured,
    _rotator,
)


# Robotics-oriented vocabulary (from holi-spatial expert)
ROBOTICS_VOCAB = [
    "floor", "wall", "ceiling", "door", "window", "table", "chair",
    "cabinet", "shelf", "stairs", "ramp", "obstacle", "plant", "box",
    "corridor", "hallway"
]

# Required categories that must always be included
REQUIRED_CATEGORIES = ["floor", "wall", "ceiling"]


def build_vlm_prompt() -> str:
    """Build the VLM prompt for object identification with 2D bounding boxes."""
    vocab_str = ", ".join(ROBOTICS_VOCAB)
    return (
        "You are a robotics vision system. Identify ALL visible objects, surfaces, "
        "and regions in this image. Use ONLY lowercase nouns from this vocabulary: "
        f"[{vocab_str}]. "
        "For each detected object, provide its category and 2D bounding box as "
        "[x_min, y_min, x_max, y_max] normalized to [0, 1] (top-left origin). "
        "Output as JSON array of objects: "
        '[{"category": "chair", "bbox": [0.1, 0.2, 0.4, 0.8]}, ...]. '
        "Include background surfaces (floor, wall, ceiling) if visible. "
        "No explanations, no extra text, only valid JSON array."
    )


def parse_vlm_response(response: str) -> list[dict]:
    """Parse VLM JSON response into list of objects with category and bbox."""
    # Try to extract JSON array from response
    try:
        # Find JSON array in response
        match = re.search(r'\[.*\]', response, re.DOTALL)
        if match:
            data = json.loads(match.group())
        else:
            data = json.loads(response)
    except json.JSONDecodeError:
        # Fallback: try comma-separated list (old format)
        labels = [label.strip().lower() for label in response.split(",")]
        filtered = [l for l in labels if l in ROBOTICS_VOCAB]
        return [{"category": l, "bbox": None} for l in filtered]

    # Validate and filter results
    objects = []
    for obj in data:
        if not isinstance(obj, dict):
            continue
        category = obj.get("category", "").strip().lower()
        if category not in ROBOTICS_VOCAB:
            continue
        bbox = obj.get("bbox")
        if bbox is not None:
            # Validate bbox format: [x_min, y_min, x_max, y_max] normalized
            if not (isinstance(bbox, list) and len(bbox) == 4):
                bbox = None
            else:
                try:
                    bbox = [float(b) for b in bbox]
                    # Clamp to [0, 1]
                    bbox = [max(0.0, min(1.0, b)) for b in bbox]
                except (ValueError, TypeError):
                    bbox = None
        objects.append({"category": category, "bbox": bbox})
    return objects


def get_frame_files(frames_dir: Path, max_frames: int = 20) -> list[Path]:
    """Get sorted list of frame files, sampling up to max_frames."""
    frame_files = sorted(frames_dir.glob("frame_*.jpg"))
    if not frame_files:
        frame_files = sorted(frames_dir.glob("frame_*.png"))

    if len(frame_files) > max_frames:
        # Uniform sampling
        step = len(frame_files) / max_frames
        indices = [int(i * step) for i in range(max_frames)]
        frame_files = [frame_files[i] for i in indices]

    return frame_files


def run_stage_d1() -> dict:
    """Run Stage D1: VLM labeling."""
    frames_dir = CFG.paths.frames
    holi_dir = CFG.paths.holi_spatial

    if not frames_dir.exists():
        print(f"WARNING: Frames directory not found: {frames_dir}")
        return _create_stub_output(holi_dir, stub_reason="frames_dir_missing")

    frame_files = get_frame_files(frames_dir, max_frames=20)
    if not frame_files:
        print(f"WARNING: No frame files found in {frames_dir}")
        return _create_stub_output(holi_dir, stub_reason="no_frames")

    # Check if we have valid API keys
    if not is_keys_configured():
        print("No valid NVIDIA API keys, using stub labels")
        return _create_stub_output(holi_dir, stub_reason="no_api_keys", frame_files=frame_files)

    # Process frames
    prompt = build_vlm_prompt()
    per_image = {}
    all_categories = set(REQUIRED_CATEGORIES)

    for i, frame_path in enumerate(frame_files):
        print(f"Processing frame {i+1}/{len(frame_files)}: {frame_path.name}")

        try:
            response = call_vlm(
                image=frame_path,
                prompt=prompt,
                model=CFG.nvidia_vlm_model,
                max_tokens=200,  # More tokens for JSON with bboxes
                temperature=0.1,
                max_retries=3,
            )
            if response is None:
                print(f"  Failed to get labels for {frame_path.name}, using stub")
                objects = [{"category": c, "bbox": None} for c in REQUIRED_CATEGORIES + ["table", "chair"]]
            else:
                objects = parse_vlm_response(response)
                # Always ensure required categories
                existing_cats = {obj["category"] for obj in objects}
                for req in REQUIRED_CATEGORIES:
                    if req not in existing_cats:
                        objects.append({"category": req, "bbox": None})
        except Exception as e:
            print(f"  Error processing {frame_path.name}: {e}")
            objects = [{"category": c, "bbox": None} for c in REQUIRED_CATEGORIES + ["table", "chair"]]

        per_image[frame_path.name] = objects
        for obj in objects:
            all_categories.add(obj["category"])
        print(f"  Objects: {objects}")

    # Build output
    output = {
        "scene": CFG.run_name,
        "categories": sorted(all_categories),
        "per_image": per_image,
        "stub": False,
    }

    # Save
    holi_dir.mkdir(parents=True, exist_ok=True)
    output_path = holi_dir / "vlm_labels.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Saved VLM labels to {output_path}")
    return output


def _create_stub_output(holi_dir: Path, stub_reason: str, frame_files: list[Path] | None = None) -> dict:
    """Create stub output when API is unavailable."""
    stub_objects = [
        {"category": "floor", "bbox": None},
        {"category": "wall", "bbox": None},
        {"category": "ceiling", "bbox": None},
        {"category": "table", "bbox": None},
        {"category": "chair", "bbox": None},
    ]
    stub_labels = [o["category"] for o in stub_objects]

    if frame_files is None:
        frame_files = []

    per_image = {f.name: stub_objects for f in frame_files}

    output = {
        "scene": CFG.run_name,
        "categories": stub_labels,
        "per_image": per_image,
        "stub": True,
        "stub_reason": stub_reason,
    }

    holi_dir.mkdir(parents=True, exist_ok=True)
    output_path = holi_dir / "vlm_labels.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Saved STUB VLM labels to {output_path} (reason: {stub_reason})")
    return output


if __name__ == "__main__":
    run_stage_d1()