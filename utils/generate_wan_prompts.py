#!/usr/bin/env python3
"""Generate WAN2.2 text-to-video prompts from a folder of videos using the OpenAI API.

For every video in the input folder this script samples a handful of frames,
sends them to a vision-capable ChatGPT model, and asks the model to write a
prompt (plus a negative prompt) that could be used to regenerate a similar clip
with the WAN2.2 model.  Results are written to a JSONL file, one row per video.

WAN2.2 prompt conventions (see src/maxdiffusion/configs/base_wan_*.yml):
  * The positive prompt is a single, richly descriptive paragraph covering the
    subject, its action/motion, the scene/background, camera movement, lighting
    and overall aesthetic/style.  It reads like natural language, not tags.
  * The negative prompt lists qualities to avoid.  WAN ships a canonical default
    (overexposure, static footage, blur, subtitles, low quality, deformed hands,
    etc.); we reuse it as a base and let the model append clip-specific items.

Usage:
  export OPENAI_API_KEY=sk-...
  python utils/generate_wan_prompts.py \
      --video-dir droid_wan2.2_lowres_test \
      --output droid_wan2.2_lowres_test/prompts.jsonl \
      --model gpt-4o

Requires:  pip install openai opencv-python-headless
"""

import argparse
import base64
import json
import os
import sys
from pathlib import Path

import cv2

# The canonical WAN2.2 negative prompt (from configs/base_wan_1_3b.yml). The model
# is asked to build on top of this rather than reinvent it, so generated prompts
# stay consistent with what the training/eval configs already expect.
WAN_DEFAULT_NEGATIVE_PROMPT = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, "
    "paintings, images, static, overall gray, worst quality, low quality, JPEG "
    "compression residue, ugly, incomplete, extra fingers, poorly drawn hands, "
    "poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, "
    "still picture, messy background, three legs, many people in the background, "
    "walking backwards"
)

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".webm", ".mkv", ".m4v", ".gif"}

SYSTEM_PROMPT = """\
You are a prompt engineer for the WAN2.2 text-to-video diffusion model. You are \
shown several frames sampled in temporal order from a single short video clip. \
Your job is to write the prompt that would make WAN2.2 regenerate a video like \
this one.

Write a POSITIVE prompt that is one flowing, vivid paragraph (roughly 40-90 \
words). Describe, in this rough order:
  1. The main subject and its appearance.
  2. What the subject is doing (the action / motion over time — use the fact that
     the frames are a time sequence to infer movement).
  3. The setting / background and any secondary elements.
  4. Camera behaviour (e.g. static shot, slow pan, tracking shot, push-in, aerial).
  5. Lighting, mood, colour palette and overall visual/artistic style.
Use natural descriptive language, not comma-separated tags. Do not mention that
these are frames, do not mention timestamps, and do not add quotation marks.

Also write a NEGATIVE prompt: start from the provided WAN2.2 default negative
prompt and, if the clip warrants it, append a few extra clip-specific artefacts
to avoid. Keep the defaults intact.

Respond with ONLY a JSON object of the form:
{"prompt": "<positive prompt>", "negative_prompt": "<negative prompt>"}
"""


def sample_frames(video_path: Path, frame_skip: int, max_frames: int = 0) -> list[bytes]:
  """Keep every ``frame_skip``-th frame from a video as JPEG bytes.

  Mirrors the ``rgb_skip`` semantics of the WAN preprocessing pipeline: with
  frame_skip=N the 0th, Nth, 2Nth, ... frames are kept (e.g. frame_skip=3 turns
  15Hz DROID footage into 5Hz). Skipped frames use grab() only, so they are
  never decoded or JPEG-encoded. If max_frames > 0, stops after that many kept
  frames to bound API cost (a warning is logged by the caller when it triggers).
  """
  frame_skip = max(frame_skip, 1)
  cap = cv2.VideoCapture(str(video_path))
  if not cap.isOpened():
    raise RuntimeError(f"could not open video: {video_path}")

  frames: list[bytes] = []
  raw_idx = 0
  truncated = False
  try:
    while True:
      if raw_idx % frame_skip != 0:
        if not cap.grab():  # advance without decoding
          break
        raw_idx += 1
        continue
      ok, frame = cap.read()
      if not ok:
        break
      raw_idx += 1
      frames.append(_encode_jpeg(frame))
      if max_frames > 0 and len(frames) >= max_frames:
        truncated = True
        break
  finally:
    cap.release()

  if not frames:
    raise RuntimeError(f"no frames decoded from: {video_path}")
  return frames, truncated


def _encode_jpeg(frame) -> bytes:
  ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
  if not ok:
    raise RuntimeError("failed to JPEG-encode a frame")
  return buf.tobytes()


def build_messages(frames: list[bytes]) -> list[dict]:
  """Build the chat messages, embedding frames as base64 data URIs."""
  content = [
      {
          "type": "text",
          "text": (
              "These frames are sampled in order from one short video clip. "
              "Write the WAN2.2 prompt and negative prompt.\n\n"
              f"WAN2.2 default negative prompt to build on:\n{WAN_DEFAULT_NEGATIVE_PROMPT}"
          ),
      }
  ]
  for jpeg in frames:
    b64 = base64.b64encode(jpeg).decode("ascii")
    content.append(
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"},
        }
    )
  return [
      {"role": "system", "content": SYSTEM_PROMPT},
      {"role": "user", "content": content},
  ]


def generate_prompt(client, model: str, frames: list[bytes]) -> dict:
  """Call the OpenAI API and return {"prompt": ..., "negative_prompt": ...}."""
  response = client.chat.completions.create(
      model=model,
      messages=build_messages(frames),
      response_format={"type": "json_object"},
      temperature=0.7,
      max_tokens=600,
  )
  data = json.loads(response.choices[0].message.content)
  prompt = (data.get("prompt") or "").strip()
  negative = (data.get("negative_prompt") or WAN_DEFAULT_NEGATIVE_PROMPT).strip()
  if not prompt:
    raise RuntimeError("model returned an empty prompt")
  return {"prompt": prompt, "negative_prompt": negative}


def find_videos(video_dir: Path) -> list[Path]:
  return sorted(
      p for p in video_dir.rglob("*") if p.suffix.lower() in VIDEO_EXTENSIONS
  )


def load_done(output_path: Path) -> set[str]:
  """Return the set of video paths already present in an existing output file."""
  done: set[str] = set()
  if output_path.exists():
    with output_path.open() as f:
      for line in f:
        line = line.strip()
        if not line:
          continue
        try:
          done.add(json.loads(line)["video"])
        except (json.JSONDecodeError, KeyError):
          continue
  return done


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  parser.add_argument("--video-dir", required=True, type=Path, help="Folder of videos to caption.")
  parser.add_argument("--output", default="wan_prompts.jsonl", type=Path, help="Output JSONL file.")
  parser.add_argument("--model", default="gpt-4o", help="OpenAI vision model id.")
  parser.add_argument("--frame-skip", type=int, default=3,
                      help="Keep every Nth frame (rgb_skip semantics). Default 15.")
  parser.add_argument("--max-frames", type=int, default=100,
                      help="Cap on kept frames sent to the API, to bound cost. 0 = unlimited.")
  parser.add_argument("--api-key", default=None, help="OpenAI API key (defaults to $OPENAI_API_KEY).")
  parser.add_argument("--overwrite", action="store_true", help="Re-caption videos already in the output.")
  args = parser.parse_args()

  try:
    from openai import OpenAI
  except ImportError:
    print("error: the 'openai' package is required. Install with: pip install openai", file=sys.stderr)
    return 1

  api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
  if not api_key:
    print("error: set OPENAI_API_KEY or pass --api-key", file=sys.stderr)
    return 1

  if not args.video_dir.is_dir():
    print(f"error: not a directory: {args.video_dir}", file=sys.stderr)
    return 1

  videos = find_videos(args.video_dir)
  if not videos:
    print(f"no videos found under {args.video_dir}", file=sys.stderr)
    return 1

  client = OpenAI(api_key=api_key)
  done = set() if args.overwrite else load_done(args.output)
  # Append so re-runs resume where a previous run left off.
  mode = "w" if args.overwrite else "a"

  n_ok, n_skip, n_err = 0, 0, 0
  with args.output.open(mode) as out:
    for i, video in enumerate(videos, 1):
      key = str(video)
      if key in done:
        n_skip += 1
        continue
      print(f"[{i}/{len(videos)}] {video.name} ...", flush=True)
      try:
        frames, truncated = sample_frames(video, args.frame_skip, args.max_frames)
        if truncated:
          print(f"    note: hit --max-frames={args.max_frames}; only the first "
                f"{args.max_frames} kept frames were sent", flush=True)
        result = generate_prompt(client, args.model, frames)
      except Exception as exc:  # noqa: BLE001 - keep going on per-video failures
        n_err += 1
        print(f"    FAILED: {exc}", file=sys.stderr)
        continue
      row = {"video": key, **result}
      out.write(json.dumps(row, ensure_ascii=False) + "\n")
      out.flush()
      n_ok += 1
      print(f"    prompt: {result['prompt'][:100]}...", flush=True)

  print(f"\nDone. {n_ok} generated, {n_skip} skipped, {n_err} failed -> {args.output}")
  return 0


if __name__ == "__main__":
  sys.exit(main())
