#!/bin/bash
# ==============================================================================
# DL3DV Dataset Downloader Helper Script
# Usage:
#   ./input/download.sh [HASH] [SUBSET] [FILE_TYPE] [RESOLUTION] [OUTPUT_DIR]
#
# Examples:
#   1. Tải video factory scene 4K (mặc định):
#      ./input/download.sh
#
#   2. Tải images+poses của 1 scene:
#      ./input/download.sh 23cdcd0cd5b44abb350ccbc6c780712d6b0c7af8143f8292f1d3e8ecfc9f4edd 1K images+poses 4K input/factory_scene_4k
#
#   3. Tải video của 1 hash khác:
#      ./input/download.sh <HASH> 1K video 4K input/<TÊN_FOLDER>
# ==============================================================================

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Sử dụng python trong .venv nếu có
if [ -f ".venv/bin/python" ]; then
    PYTHON_BIN=".venv/bin/python"
elif command -v python3 &>/dev/null; then
    PYTHON_BIN="python3"
else
    PYTHON_BIN="python"
fi

HASH="${1:-23cdcd0cd5b44abb350ccbc6c780712d6b0c7af8143f8292f1d3e8ecfc9f4edd}"
SUBSET="${2:-1K}"
FILE_TYPE="${3:-video}"      # 'video' | 'images+poses' | 'colmap_cache'
RESOLUTION="${4:-4K}"        # '4K' | '2K' | '960P' | '480P'
OUTPUT_DIR="${5:-input/factory_scene_4k}"

echo "=================================================="
echo "🚀 Đang tải dữ liệu từ Hugging Face DL3DV..."
echo "  Python:      $PYTHON_BIN"
echo "  Hash:        $HASH"
echo "  Subset:      $SUBSET"
echo "  File Type:   $FILE_TYPE"
echo "  Resolution:  $RESOLUTION"
echo "  Output Dir:  $OUTPUT_DIR"
echo "=================================================="

$PYTHON_BIN download.py \
  --odir "$OUTPUT_DIR" \
  --subset "$SUBSET" \
  --resolution "$RESOLUTION" \
  --file_type "$FILE_TYPE" \
  --hash "$HASH"
