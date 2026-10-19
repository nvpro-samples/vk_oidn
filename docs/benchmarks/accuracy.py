#!/usr/bin/env python3
# Copyright 2026 NVIDIA Corporation
# SPDX-License-Identifier: Apache-2.0

"""Accuracy of Open Image Denoise 2.5.1 CUDA and vk_oidn against the Open Image Denoise 2.5.1 CPU
device, for the configuration of the benchmarks (RT filter, hdr + albedo + normal, high and fast
quality), with the metric of oidnDenoise -r (average relative error and outliers, see
docs/validation_and_performance.md).

usage: accuracy.py <directory with color.pfm, albedo.pfm, normal.pfm>
       (e.g. images/hd from tests/scripts/gen_images.py images/hd 1920 1080). Paths: see bench_paths.py."""
import os
import subprocess
import sys
import numpy as np
from bench_paths import *

IMG = sys.argv[1]
OUT = os.path.join(HERE, 'accuracy_out')
OFFICIAL = os.path.join(OIDN_SDK, 'bin', 'oidnDenoise' + EXE)
os.makedirs(OUT, exist_ok=True)


def read_pfm(path):
  with open(path, 'rb') as f:
    assert f.readline().strip() == b'PF'
    w, h = map(int, f.readline().split())
    scale = float(f.readline())
    data = np.frombuffer(f.read(), dtype='<f4' if scale < 0 else '>f4')
  return np.flipud(data.reshape(h, w, 3)).astype(np.float64)


def compare(actual, ref):
  abs_err = np.abs(ref - actual)
  rel_err = abs_err / (np.abs(ref) + np.abs(actual) + 0.01)
  outliers = int(np.count_nonzero(~((abs_err <= 0.02) | (rel_err <= 0.05))))
  return rel_err.mean(), outliers


inputs = ['--hdr', os.path.join(IMG, 'color.pfm'), '--alb', os.path.join(IMG, 'albedo.pfm'),
          '--nrm', os.path.join(IMG, 'normal.pfm')]
for quality in ('high', 'fast'):
  outs = {}
  for name, cmd in [('cpu', [OFFICIAL, '-d', 'cpu']), ('cuda', [OFFICIAL, '-d', 'cuda']),
                    ('vk_oidn', [VK_OIDN_TEST])]:
    path = os.path.join(OUT, '%s_%s.pfm' % (name, quality))
    r = subprocess.run(cmd + inputs + ['-q', quality, '-o', path], capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(path):
      sys.exit('%s failed: %s %s' % (name, r.stdout[-500:], r.stderr[-500:]))
    outs[name] = read_pfm(path)
  ref = outs['cpu']
  for name in ('cuda', 'vk_oidn'):
    err, outliers = compare(outs[name], ref)
    print('%-5s %-10s avg relative error vs CPU %.6f, outliers %d' % (quality, name, err, outliers))
  err, outliers = compare(outs['vk_oidn'], outs['cuda'])
  print('%-5s vk_oidn vs CUDA: avg relative error %.6f, outliers %d' % (quality, err, outliers))
