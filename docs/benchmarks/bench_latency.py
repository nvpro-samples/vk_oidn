#!/usr/bin/env python3
# Copyright 2026 NVIDIA Corporation
# SPDX-License-Identifier: Apache-2.0

"""Secondary benchmark: per-image latency, the metric of Open Image Denoise's oidnBenchmark: for
each image, execute (record + submit for vk_oidn) and wait for the GPU; two warm-up runs, then at
least 0.5 s / 3 runs. RT filter, hdr + albedo + normal, FP32 images in device memory.

- Open Image Denoise 2.5.1 CUDA: the official oidnBenchmark.
- vk_oidn: vk_oidn_denoise_test --bench (also -n 50, whose GPU time is recorded as vk_oidn_gpu).

Each configuration runs REPEATS times, interleaved; before each run the GPU must have been idle
(< 5%) for 10 s. Raw output: results/bench_latency_raw.log; values:
results/bench_latency_results.json. Paths: see bench_paths.py."""
import json
import os
import re
import statistics
import subprocess
import time
from bench_paths import *

REPEATS = 5
RESULTS = os.path.join(HERE, 'results')


def oidn_benchmark(exe, device, res, quality):
  return [exe, '-d', device, '-r', 'RT.hdr_alb_nrm.%s$' % res, '-q', quality]


CONFIGS = [
  ('oidn_cuda', lambda r, q: oidn_benchmark(os.path.join(OIDN_SDK, 'bin', 'oidnBenchmark' + EXE), 'cuda', r, q), {}),
  ('vk_oidn', lambda r, q: [VK_OIDN_TEST] + vk_oidn_inputs(r) + ['-q', q, '-n', '50', '--bench'], {}),
]

os.makedirs(RESULTS, exist_ok=True)
results = {}
with open(os.path.join(RESULTS, 'bench_latency_raw.log'), 'a', encoding='utf-8') as log:
  log.write('\n===== run started %s =====\n' % time.strftime('%Y-%m-%d %H:%M:%S'))
  for rep in range(REPEATS):
    for res in RESOLUTIONS:
      for quality in QUALITIES:
        for name, make_cmd, env_add in CONFIGS:
          wait_idle()
          cmd = make_cmd(res, quality)
          out = subprocess.run(cmd, env=dict(os.environ, **env_add), capture_output=True, text=True).stdout
          log.write('--- %s %s %s rep %d: %s\n%s\n' % (name, res, quality, rep, ' '.join(cmd), out))
          log.flush()
          if name == 'vk_oidn':
            wall = re.search(r'Wall-clock time: ([0-9.]+) msec', out)
            gpu = re.search(r'GPU time: ([0-9.]+) msec', out)
            results.setdefault('vk_oidn|%s|%s' % (res, quality), []).append(float(wall.group(1)) if wall else float('nan'))
            results.setdefault('vk_oidn_gpu|%s|%s' % (res, quality), []).append(float(gpu.group(1)) if gpu else float('nan'))
            v = results['vk_oidn|%s|%s' % (res, quality)][-1]
          else:
            m = re.search(r'RT\.hdr_alb_nrm\.%s \.\.\. ([0-9.]+) msec/image' % res, out)
            v = float(m.group(1)) if m else float('nan')
            results.setdefault('%s|%s|%s' % (name, res, quality), []).append(v)
          print('%-10s %-9s %-4s rep %d: %8.3f ms' % (name, res, quality, rep, v), flush=True)

with open(os.path.join(RESULTS, 'bench_latency_results.json'), 'w') as f:
  json.dump(results, f, indent=1)
print('\nmsec/image: median [min - max] of %d runs' % REPEATS)
for key in sorted(results):
  v = results[key]
  print('%-40s %.3f [%.3f - %.3f]' % (key, statistics.median(v), min(v), max(v)))
