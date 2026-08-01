#!/bin/bash

INPUT_DIR="${1:-test_videos}"
OUTPUT_DIR="${2:-${INPUT_DIR}_resized}"
WIDTH=832
HEIGHT=480

mkdir -p "$OUTPUT_DIR"

for f in "$INPUT_DIR"/*.mp4; do
  [ -f "$f" ] || continue
  fname=$(basename "$f")
  ffmpeg -i "$f" \
    -vf "scale=${WIDTH}:${HEIGHT}:force_original_aspect_ratio=disable" \
    -c:v libx264 -c:a copy \
    "$OUTPUT_DIR/$fname" \
    -y
done

echo "Done. Resized videos written to $OUTPUT_DIR/"