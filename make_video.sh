#!/usr/bin/env bash
# =============================================================================
# make_video.sh – Convert a numbered image sequence to an MP4 video
#                 using libx264 (CPU only).
#
# Usage:
#   ./make_video.sh <input_pattern> <output_file> [fps] [height] [width] [crf]
#
#   <input_pattern> : e.g. "img_%05d.png"
#   <output_file>   : e.g. out.mp4
#   [fps]           : optional, default 30
#   [height] (H)    : optional, default -1
#   [width]  (W)    : optional, default -1
#                     - If H or W = -1, the missing dimension is
#                       auto-computed to keep aspect ratio.
#                     - If H = W = -1, no user resize is applied.
#   [crf]           : optional, default 18 (0–51, lower = better quality)
#
# Notes:
#   * Output dimensions are always forced to even numbers.
# =============================================================================
set -eu
IFS=$'\n\t'

# --- argument check -----------------------------------------------------------
if [[ $# -lt 2 || $# -gt 6 ]]; then
  echo "Usage: $0 <input_pattern> <output_file> [fps] [height] [width] [crf]" >&2
  exit 1
fi

INPUT_PATTERN=$1
OUTPUT_FILE=$2
FPS=${3:-30}
H=${4:--1}
W=${5:--1}
CRF=${6:-18}

# --- build scale filter -------------------------------------------------------
# Helper: force any dimension to the nearest even integer
auto_even() { printf 'ceil(%s/2)*2' "$1"; }

if [[ $H -eq -1 && $W -eq -1 ]]; then
  # No explicit resize; just make both dimensions even
  SCALE_EXPR_W=$(auto_even iw)
  SCALE_EXPR_H=$(auto_even ih)
elif [[ $H -eq -1 ]]; then
  # Width fixed, height auto
  SCALE_EXPR_W=$(auto_even "$W")
  SCALE_EXPR_H=-2          # FFmpeg: -2 ⇒ preserve aspect, height even
elif [[ $W -eq -1 ]]; then
  # Height fixed, width auto
  SCALE_EXPR_W=-2          # FFmpeg: -2 ⇒ preserve aspect, width even
  SCALE_EXPR_H=$(auto_even "$H")
else
  # Both dimensions specified
  SCALE_EXPR_W=$(auto_even "$W")
  SCALE_EXPR_H=$(auto_even "$H")
fi

SCALE_FILTER="scale=${SCALE_EXPR_W}:${SCALE_EXPR_H}"

# --- run ffmpeg ---------------------------------------------------------------
ffmpeg -hide_banner -loglevel error \
       -framerate "$FPS" \
       -i "$INPUT_PATTERN" \
       -vf "$SCALE_FILTER" \
       -c:v libx264 -pix_fmt yuv420p -crf "$CRF" \
       "$OUTPUT_FILE"


