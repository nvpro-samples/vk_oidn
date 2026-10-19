#!/bin/bash
# Copyright 2026 NVIDIA Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Validates vk_oidn on a set of denoising configurations:
#   - compares the output with the Open Image Denoise CPU device (reference), using the same
#     average relative error (SMAPE) and outlier criteria as oidnDenoise
#   - optionally checks that image view inputs/outputs give the same result as buffers
#
# Usage:
#   run_validation.sh <image dir> [--images]
#
# Environment variables:
#   VK_OIDN_TEST      path to vk_oidn_denoise_test (default: build/vk_oidn_denoise_test[.exe])
#   OIDN_REF_DENOISE  path to oidnDenoise of an official Open Image Denoise release (CPU device),
#                     used to create the reference images (ref_<config>.pfm) if missing
#   VK_OIDN_TEST_ARGS extra options for vk_oidn_denoise_test (e.g. --validation)
#
# The images can be generated with gen_images.py.

set -u
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
TEST="${VK_OIDN_TEST:-$ROOT_DIR/build/vk_oidn_denoise_test}"
REF="${OIDN_REF_DENOISE:-}"
TEST_ARGS="${VK_OIDN_TEST_ARGS:-}"

IMG_DIR="$1"
EXTRA="${2:-}"
cd "$IMG_DIR" || exit 1

failures=0

run() {
  local name=$1; shift
  if [ ! -f "ref_$name.pfm" ]; then
    if [ -z "$REF" ]; then
      echo "$name: missing ref_$name.pfm (set OIDN_REF_DENOISE)"; failures=$((failures+1)); return
    fi
    "$REF" -d cpu "$@" -o "ref_$name.pfm" > /dev/null 2>&1
  fi

  local output status result
  output=$("$TEST" "$@" $EXTRA $TEST_ARGS -o "sdk_$name.pfm" -r "ref_$name.pfm" 2>&1)
  status=$?
  result=$(echo "$output" | grep -E "Compare|Error:|alpha|Validation errors" | tr '\n' ' ')
  if [ $status -ne 0 ] || ! echo "$result" | grep -q -- "-> OK"; then
    failures=$((failures+1))
    echo "$name: exit code $status, last output lines:"
    echo "$output" | tail -n 8 | sed 's/^/    /'
  fi

  printf "%-16s %s\n" "$name" "$result"
}

run hdr            --hdr color.pfm
run hdr_alb        --hdr color.pfm --alb albedo.pfm
run hdr_alb_nrm    --hdr color.pfm --alb albedo.pfm --nrm normal.pfm
run hdr_calb_cnrm  --hdr color.pfm --alb albedo.pfm --nrm normal.pfm --clean_aux
run hdr_alb_nrm_f  --hdr color.pfm --alb albedo.pfm --nrm normal.pfm -q fast
run hdr_alb_nrm_b  --hdr color.pfm --alb albedo.pfm --nrm normal.pfm -q balanced
run hdr_is         --hdr color.pfm --alb albedo.pfm --nrm normal.pfm --is 0.5
run ldr_alb_nrm    --ldr ldr.pfm --alb albedo.pfm --nrm normal.pfm
run ldr_srgb_an    --ldr ldr.pfm --srgb --alb albedo.pfm --nrm normal.pfm
run hdr_half       --hdr color.pfm --alb albedo.pfm --nrm normal.pfm -t half
run hdr_tiled      --hdr color.pfm --alb albedo.pfm --nrm normal.pfm --maxmem 200

if [ $failures -eq 0 ]; then
  echo "All configurations passed"
else
  echo "$failures failure(s)"
fi
exit $failures
