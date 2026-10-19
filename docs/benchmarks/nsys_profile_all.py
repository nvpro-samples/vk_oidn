#!/usr/bin/env python3
# Copyright 2026 NVIDIA Corporation
# SPDX-License-Identifier: Apache-2.0

"""Independent check of the GPU times with Nsight Systems: profiles the timed batch of 50
denoisings of every configuration of bench_gpu.py (REPEATS interleaved runs, idle GPU before each
run), and reads the GPU activity of the batch from the reports exported to SQLite.

- OIDN 2.5.1 CUDA: oidn_gpu_bench, nsys --trace=cuda; the kernels and memory operations launched
  between the two cudaEventRecord calls that delimit the timed batch.
- vk_oidn: vk_oidn_denoise_test -n 50, nsys --trace=vulkan --vulkan-gpu-workload=individual; the
  last command buffer, which holds the timed batch.

For each run: the span from the first to the last GPU operation of the batch and the time the GPU
was busy (union of the operations), divided by 50. Vulkan tracing on Windows needs the Nsight
Vulkan layer to be registered once (one Vulkan trace run as administrator). Paths: see
bench_paths.py; NSYS is the nsys executable. Output: results/nsys_raw.log, results/nsys_results.json."""
import json
import os
import sqlite3
import statistics
import subprocess
import time
from bench_paths import *

NSYS = env_path('NSYS')
REPEATS = 3
N = 50
RESULTS = os.path.join(HERE, 'results')
REPORTS = os.path.join(HERE, 'build', 'nsys')


def oidn_cuda(res, quality):
  w, h = res.split('x')
  env = 'PATH=%s;%s' % (os.path.join(OIDN_SDK, 'bin'), os.path.join(os.environ['SystemRoot'], 'System32'))
  return ['--trace=cuda', '-e', env], [OIDN_GPU_BENCH, 'cuda', w, h, quality, str(N)]


def vk_oidn(res, quality):
  return (['--trace=vulkan', '--vulkan-gpu-workload=individual'],
          [VK_OIDN_TEST] + vk_oidn_inputs(res) + ['-q', quality, '-n', str(N)])


def cuda_batch(con):
  """Kernels and memory operations launched between the two CUDA events of the timed batch."""
  tables = {r[0] for r in con.execute("select name from sqlite_master where type='table'")}
  first, last = [r[0] for r in con.execute('select correlationId from CUPTI_ACTIVITY_KIND_CUDA_EVENT order by correlationId')]
  ops = []
  for table in ('CUPTI_ACTIVITY_KIND_KERNEL', 'CUPTI_ACTIVITY_KIND_MEMCPY', 'CUPTI_ACTIVITY_KIND_MEMSET'):
    if table in tables:
      ops += con.execute('select start, end from %s where correlationId > ? and correlationId < ?' % table,
                         (first, last)).fetchall()
  return ops


def vulkan_batch(con):
  """The last command buffer of vk_oidn_denoise_test -n 50."""
  return con.execute('select start, end from VULKAN_WORKLOAD order by start desc limit 1').fetchall()


def span_and_busy(ops):
  ops = sorted(ops)
  busy, start, end = 0, ops[0][0], ops[0][1]
  for s, e in ops[1:]:
    if s > end:
      busy += end - start
      start = s
    end = max(end, e)
  busy += end - start
  return (ops[-1][1] - ops[0][0]) * 1e-6 / N, busy * 1e-6 / N, len(ops)


CONFIGS = [('oidn_cuda', oidn_cuda, cuda_batch), ('vk_oidn', vk_oidn, vulkan_batch)]

os.makedirs(RESULTS, exist_ok=True)
os.makedirs(REPORTS, exist_ok=True)
results = {}
with open(os.path.join(RESULTS, 'nsys_raw.log'), 'a', encoding='utf-8') as log:
  log.write('\n===== run started %s, %s =====\n' % (time.strftime('%Y-%m-%d %H:%M:%S'),
            subprocess.run([NSYS, '--version'], capture_output=True, text=True).stdout.strip()))
  for rep in range(REPEATS):
    for res in RESOLUTIONS:
      for quality in QUALITIES:
        for name, make, batch in CONFIGS:
          wait_idle()
          nsys_args, app = make(res, quality)
          report = os.path.join(REPORTS, '%s_%s_%s_%d' % (name, res, quality, rep))
          cmd = [NSYS, 'profile', '--sample=none', '--cpuctxsw=none', '-f', 'true', '-o', report] + nsys_args + app
          subprocess.run(cmd, capture_output=True, text=True)
          subprocess.run([NSYS, 'export', '--type', 'sqlite', '-f', 'true', '-o', report + '.sqlite', report + '.nsys-rep'],
                         capture_output=True, text=True)
          try:
            span, busy, count = span_and_busy(batch(sqlite3.connect(report + '.sqlite')))
            line = '%d GPU operations, span %.3f msec/image, GPU busy %.3f msec/image (%.2f%%)' % (
              count, span, busy, 100.0 * busy / span)
            results.setdefault('%s|span|%s|%s' % (name, res, quality), []).append(span)
            results.setdefault('%s|busy|%s|%s' % (name, res, quality), []).append(busy)
          except Exception as e:
            line = 'FAILED: %s' % e
          log.write('--- %s %s %s rep %d: %s\n%s\n' % (name, res, quality, rep, ' '.join(cmd), line))
          log.flush()
          print('%-9s %-9s %-4s rep %d: %s' % (name, res, quality, rep, line), flush=True)

with open(os.path.join(RESULTS, 'nsys_results.json'), 'w') as f:
  json.dump(results, f, indent=1)
print('\nmsec/image: median [min - max] of %d runs' % REPEATS)
for key in sorted(results):
  v = results[key]
  print('%-40s %.3f [%.3f - %.3f]' % (key, statistics.median(v), min(v), max(v)))
