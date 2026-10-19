#!/usr/bin/env python3
# Copyright 2026 NVIDIA Corporation
# SPDX-License-Identifier: Apache-2.0

"""Writes random benchmark inputs for vk_oidn_denoise_test, with the distributions of
oidnBenchmark: color uniform in [0, 100], albedo in [0, 1], normal in [-1, 1].
The denoising time does not depend on the pixel values.

usage: gen_bench_inputs.py [output directory, default: docs/benchmarks/inputs]"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'inputs')
os.makedirs(out, exist_ok=True)
rng = np.random.default_rng(1)
for w, h in ((1280, 720), (1920, 1080), (3840, 2160)):
  for name, lo, hi in (('color', 0.0, 100.0), ('albedo', 0.0, 1.0), ('normal', -1.0, 1.0)):
    data = rng.uniform(lo, hi, (h, w, 3)).astype('<f4')
    with open(os.path.join(out, '%s_%dx%d.pfm' % (name, w, h)), 'wb') as f:
      f.write(b'PF\n%d %d\n-1.0\n' % (w, h))
      f.write(data.tobytes())
print('wrote', out)
