#!/usr/bin/env python3
"""
Extract color video from ROS2 .db3 bag files.
Trích xuất video màu từ các file .db3 (ROS2 bag).

Usage:
    python3 extract_video_from_db3.py <bag1.db3> [bag2.db3 ...] --output <output_dir>
"""

import argparse
import sqlite3
import struct
import os
import sys
import numpy as np
import cv2
from pathlib import Path
from typing import Optional


# ─── CDR Deserializer for sensor_msgs/msg/Image ───────────────────────────────

def parse_ros2_image(data: bytes):
    """
    Parse a CDR-encoded sensor_msgs/msg/Image blob.

    CDR layout (little-endian):
      [0-3]   CDR header (00 01 00 00)
      [4-7]   Header.stamp.sec   (uint32)
      [8-11]  Header.stamp.nanosec (uint32)
      [12-15] frame_id length (uint32, includes null terminator)
      [16..]  frame_id bytes
      [...] Aligned to 4 bytes, then:
      height (uint32)
      width  (uint32)
      encoding length (uint32, includes null terminator)
      encoding bytes
      is_bigendian (uint8)
      <pad to 4>
      step (uint32)
      data array length (uint32)
      data bytes
    """
    offset = 4  # skip CDR header

    # stamp
    sec = struct.unpack_from('<I', data, offset)[0]; offset += 4
    nanosec = struct.unpack_from('<I', data, offset)[0]; offset += 4

    # frame_id
    fid_len = struct.unpack_from('<I', data, offset)[0]; offset += 4
    offset += fid_len  # skip frame_id bytes (includes null terminator)

    # Align to 4 bytes
    if offset % 4 != 0:
        offset += 4 - (offset % 4)

    # height, width
    height = struct.unpack_from('<I', data, offset)[0]; offset += 4
    width  = struct.unpack_from('<I', data, offset)[0]; offset += 4

    # encoding
    enc_len = struct.unpack_from('<I', data, offset)[0]; offset += 4
    encoding = data[offset:offset + enc_len - 1].decode('utf-8', errors='replace')
    offset += enc_len

    # is_bigendian
    is_bigendian = data[offset]; offset += 1

    # Align to 4 bytes
    if offset % 4 != 0:
        offset += 4 - (offset % 4)

    # step and pixel data
    step = struct.unpack_from('<I', data, offset)[0]; offset += 4
    arr_len = struct.unpack_from('<I', data, offset)[0]; offset += 4

    pixel_bytes = data[offset:offset + arr_len]

    return {
        'sec': sec,
        'nanosec': nanosec,
        'height': height,
        'width': width,
        'encoding': encoding,
        'is_bigendian': is_bigendian,
        'step': step,
        'pixels': pixel_bytes,
    }


def encoding_to_cv2(encoding: str, pixels: bytes, height: int, width: int) -> Optional[np.ndarray]:
    """Convert raw pixel bytes + encoding string to a BGR numpy array for OpenCV."""
    enc = encoding.lower()

    if enc == 'rgb8':
        arr = np.frombuffer(pixels, dtype=np.uint8).reshape((height, width, 3))
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    elif enc == 'bgr8':
        arr = np.frombuffer(pixels, dtype=np.uint8).reshape((height, width, 3))
        return arr.copy()

    elif enc in ('mono8', '8uc1'):
        arr = np.frombuffer(pixels, dtype=np.uint8).reshape((height, width))
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)

    elif enc == 'rgba8':
        arr = np.frombuffer(pixels, dtype=np.uint8).reshape((height, width, 4))
        return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)

    elif enc == 'bgra8':
        arr = np.frombuffer(pixels, dtype=np.uint8).reshape((height, width, 4))
        return cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)

    elif enc in ('16uc1', 'mono16'):
        # Depth image – normalize to 8-bit for visualization
        arr = np.frombuffer(pixels, dtype=np.uint16).reshape((height, width))
        arr_norm = cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return cv2.cvtColor(arr_norm, cv2.COLOR_GRAY2BGR)

    else:
        print(f"  [WARN] Unknown encoding '{encoding}', skipping frame.")
        return None


# ─── Main extraction logic ─────────────────────────────────────────────────────

def extract_video_from_bag(
    bag_path: str,
    output_dir: str,
    topic_name: str = '/device_0/sensor_1/Color_0/image/data',
    fps: float = 30.0,
    codec: str = 'mp4v',
) -> Optional[str]:
    """
    Extract all frames from a given image topic and write to an .mp4 file.

    Returns the output file path, or None on failure.
    """
    bag_path = Path(bag_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    bag_stem = bag_path.stem  # e.g. '20260810_150126'
    out_file = output_dir / f'{bag_stem}.mp4'

    print(f"\n{'='*60}")
    print(f"  Bag   : {bag_path}")
    print(f"  Topic : {topic_name}")
    print(f"  Output: {out_file}")
    print(f"{'='*60}")

    conn = sqlite3.connect(str(bag_path))
    cursor = conn.cursor()

    # Find topic id
    cursor.execute(
        "SELECT id FROM topics WHERE name = ?",
        (topic_name,)
    )
    row = cursor.fetchone()
    if row is None:
        print(f"  [ERROR] Topic '{topic_name}' not found in {bag_path.name}")
        print("  Available image topics:")
        cursor.execute("SELECT id, name FROM topics WHERE type LIKE '%Image%'")
        for r in cursor.fetchall():
            print(f"    {r}")
        conn.close()
        return None

    topic_id = row[0]
    cursor.execute(
        "SELECT COUNT(*) FROM messages WHERE topic_id = ?",
        (topic_id,)
    )
    total_frames = cursor.fetchone()[0]
    print(f"  Frames found: {total_frames}")

    if total_frames == 0:
        print("  [WARN] No frames found, skipping.")
        conn.close()
        return None

    # Read first frame to get dimensions
    cursor.execute(
        "SELECT data FROM messages WHERE topic_id = ? LIMIT 1",
        (topic_id,)
    )
    first_data = cursor.fetchone()[0]
    first_img = parse_ros2_image(bytes(first_data))
    height, width = first_img['height'], first_img['width']
    encoding = first_img['encoding']
    print(f"  Resolution: {width}x{height}, encoding: {encoding}")

    # Estimate FPS from timestamps if possible (use first 2 frames)
    cursor.execute(
        "SELECT data FROM messages WHERE topic_id = ? LIMIT 2",
        (topic_id,)
    )
    frames_2 = cursor.fetchall()
    if len(frames_2) == 2:
        f0 = parse_ros2_image(bytes(frames_2[0][0]))
        f1 = parse_ros2_image(bytes(frames_2[1][0]))
        t0 = f0['sec'] + f0['nanosec'] * 1e-9
        t1 = f1['sec'] + f1['nanosec'] * 1e-9
        dt = t1 - t0
        if 0 < dt < 1.0:
            fps = round(1.0 / dt, 2)
            print(f"  Detected FPS: {fps}")

    # Set up VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(out_file), fourcc, fps, (width, height))

    if not writer.isOpened():
        print(f"  [ERROR] Could not open VideoWriter for {out_file}")
        conn.close()
        return None

    # Stream frames
    cursor.execute(
        "SELECT data FROM messages WHERE topic_id = ? ORDER BY rowid",
        (topic_id,)
    )

    written = 0
    skipped = 0
    for i, (raw,) in enumerate(cursor):
        try:
            img_info = parse_ros2_image(bytes(raw))
            frame = encoding_to_cv2(
                img_info['encoding'],
                img_info['pixels'],
                img_info['height'],
                img_info['width'],
            )
            if frame is not None:
                writer.write(frame)
                written += 1
            else:
                skipped += 1
        except Exception as e:
            skipped += 1
            if skipped <= 5:
                print(f"  [WARN] Frame {i} error: {e}")

        # Progress
        if (i + 1) % 50 == 0 or (i + 1) == total_frames:
            pct = (i + 1) / total_frames * 100
            print(f"  Progress: {i+1}/{total_frames} ({pct:.1f}%)", end='\r')

    print(f"\n  Written: {written} frames, Skipped: {skipped}")
    writer.release()
    conn.close()

    if written == 0:
        out_file.unlink(missing_ok=True)
        return None

    size_mb = out_file.stat().st_size / 1024 / 1024
    print(f"  ✓ Saved: {out_file} ({size_mb:.1f} MB)")
    return str(out_file)


# ─── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Extract color video from ROS2 .db3 bag files.'
    )
    parser.add_argument(
        'bags', nargs='+',
        help='Path(s) to .db3 bag file(s)'
    )
    parser.add_argument(
        '--output', '-o', required=True,
        help='Output directory for .mp4 files'
    )
    parser.add_argument(
        '--topic', default='/device_0/sensor_1/Color_0/image/data',
        help='Image topic name (default: /device_0/sensor_1/Color_0/image/data)'
    )
    parser.add_argument(
        '--fps', type=float, default=30.0,
        help='Fallback FPS if cannot be auto-detected (default: 30)'
    )
    parser.add_argument(
        '--codec', default='mp4v',
        help='FourCC codec string (default: mp4v)'
    )
    args = parser.parse_args()

    results = []
    for bag in args.bags:
        out = extract_video_from_bag(
            bag_path=bag,
            output_dir=args.output,
            topic_name=args.topic,
            fps=args.fps,
            codec=args.codec,
        )
        results.append((bag, out))

    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    ok = [r for r in results if r[1] is not None]
    fail = [r for r in results if r[1] is None]
    for bag, out in ok:
        print(f"  ✓ {Path(bag).name}  →  {out}")
    for bag, _ in fail:
        print(f"  ✗ {Path(bag).name}  →  FAILED")
    print(f"\n  {len(ok)}/{len(results)} bags extracted successfully.")


if __name__ == '__main__':
    main()
