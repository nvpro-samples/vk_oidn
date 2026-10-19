#!/usr/bin/env python3
# Copyright 2026 NVIDIA Corporation
# SPDX-License-Identifier: Apache-2.0

"""Main benchmark: GPU time per image, with the same method for every implementation: a warm-up
batch of N = 50 back-to-back denoisings, then a timed batch of 50, synchronized once. RT filter,
hdr + albedo + normal, FP32 images in device memory, 1280x720 / 1920x1080 / 3840x2160, high and
fast quality.

- Open Image Denoise 2.5.1 CUDA (official binaries): oidn_gpu_bench, built against the official
  2.5.1 SDK. Batched wall-clock time and CUDA events.
- vk_oidn: vk_oidn_denoise_test -n 50: batched wall-clock time and Vulkan timestamps.

Each configuration runs REPEATS times, interleaved; before each run the GPU must have been idle
(< 5%) for 10 s. Raw output: results/bench_gpu_raw.log; values: results/bench_gpu_results.json.
Paths: see bench_paths.py."""
import json
import os
import re
import statistics
import subprocess
import time
from bench_paths import *

REPEATS = 5
N = 50
RESULTS = os.path.join(HERE, 'results')


def oidn_cuda(res, quality):
  w, h = res.split('x')
  env = dict(os.environ, PATH=os.path.join(OIDN_SDK, 'bin') + os.pathsep + os.environ['PATH'])
  return [OIDN_GPU_BENCH, 'cuda', w, h, quality, str(N)], env


def vk_oidn(res, quality):
  return [VK_OIDN_TEST] + vk_oidn_inputs(res) + ['-q', quality, '-n', str(N)], dict(os.environ)


CONFIGS = [
  ('oidn_cuda', oidn_cuda),
  ('vk_oidn', vk_oidn),
]
# metric name in the output -> result key
METRICS = {'Batched wall-clock': 'batched', 'CUDA events': 'cuda_events', 'GPU time': 'timestamps',
           'Enqueue time': 'enqueue', 'Sync latency': 'sync_latency'}

os.makedirs(RESULTS, exist_ok=True)
results = {}
with open(os.path.join(RESULTS, 'bench_gpu_raw.log'), 'a', encoding='utf-8') as log:
  log.write('\n===== run started %s =====\n' % time.strftime('%Y-%m-%d %H:%M:%S'))
  for rep in range(REPEATS):
    for res in RESOLUTIONS:
      for quality in QUALITIES:
        for name, make in CONFIGS:
          wait_idle()
          cmd, env = make(res, quality)
          r = subprocess.run(cmd, env=env, capture_output=True, text=True)
          log.write('--- %s %s %s rep %d: %s\n%s%s\n' % (name, res, quality, rep, ' '.join(cmd), r.stdout, r.stderr))
          log.flush()
          found = []
          for label, key in METRICS.items():
            m = re.search(r'^%s: ([0-9.]+) msec' % label, r.stdout, re.M)
            if m:
              results.setdefault('%s|%s|%s|%s' % (name, key, res, quality), []).append(float(m.group(1)))
              found.append('%s %s' % (key, m.group(1)))
          print('%-10s %-9s %-4s rep %d: %s' % (name, res, quality, rep, ', '.join(found) or 'FAILED'), flush=True)

with open(os.path.join(RESULTS, 'bench_gpu_results.json'), 'w') as f:
  json.dump(results, f, indent=1)
print('\nmsec/image: median [min - max] of %d runs' % REPEATS)
for key in sorted(results):
  v = results[key]
  print('%-50s %.3f [%.3f - %.3f]' % (key, statistics.median(v), min(v), max(v)))
