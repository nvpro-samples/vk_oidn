#!/usr/bin/env python3
# Copyright 2026 NVIDIA Corporation
# SPDX-License-Identifier: Apache-2.0

"""Prints the Markdown tables of docs/validation_and_performance.md and the README from
results/bench_gpu_results.json and results/bench_latency_results.json: median [min - max] of the
runs, and the ratio of the medians."""
import json
import os
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, 'results')
ROWS = [(r, q) for q in ('high', 'fast') for r in ('1280x720', '1920x1080', '3840x2160')]


def load(name):
  with open(os.path.join(RESULTS, name)) as f:
    return json.load(f)


def cell(v):
  return '%.2f [%.2f-%.2f]' % (statistics.median(v), min(v), max(v))


def table(title, columns, values):
  """columns: list of (header, key function (res, quality) -> list of values)."""
  print('\n' + title + '\n')
  print('| Resolution | Quality | ' + ' | '.join(h for h, _ in columns) + ' | CUDA / vk_oidn |')
  print('|---' * (len(columns) + 3) + '|')
  for res, q in ROWS:
    cells = [cell(values[k(res, q)]) for _, k in columns]
    ratio = statistics.median(values[columns[0][1](res, q)]) / statistics.median(values[columns[-1][1](res, q)])
    print('| %s | %s | %s | %.2fx |' % (res, q, ' | '.join(cells), ratio))


gpu = load('bench_gpu_results.json')
table('GPU time (batched, ms/image)', [
  ('OIDN 2.5.1 CUDA', lambda r, q: 'oidn_cuda|batched|%s|%s' % (r, q)),
  ('vk_oidn', lambda r, q: 'vk_oidn|batched|%s|%s' % (r, q))], gpu)
diff = max(abs(x - y) for name, timer in (('oidn_cuda', 'cuda_events'), ('vk_oidn', 'timestamps'))
           for res, q in ROWS
           for x, y in zip(gpu['%s|%s|%s|%s' % (name, timer, res, q)], gpu['%s|batched|%s|%s' % (name, res, q)]))
print()
print('Largest difference between the GPU timers and the batched time: %.3f ms/image' % diff)
print()
print('Host enqueue time of OIDN CUDA (ms/image), against its GPU time:')
for res, q in ROWS:
  print('  %s %s: enqueue %s, GPU time %s' % (res, q, cell(gpu['oidn_cuda|enqueue|%s|%s' % (res, q)]),
                                           cell(gpu['oidn_cuda|batched|%s|%s' % (res, q)])))

lat = load('bench_latency_results.json')
table('Per-image latency (oidnBenchmark method, ms/image)', [
  ('OIDN 2.5.1 CUDA', lambda r, q: 'oidn_cuda|%s|%s' % (r, q)),
  ('vk_oidn', lambda r, q: 'vk_oidn|%s|%s' % (r, q))], lat)

nsys = load('nsys_results.json')
table('Nsight Systems: GPU busy time of the timed batch (ms/image; CUDA: kernels and memory operations only)', [
  ('OIDN 2.5.1 CUDA', lambda r, q: 'oidn_cuda|busy|%s|%s' % (r, q)),
  ('vk_oidn', lambda r, q: 'vk_oidn|busy|%s|%s' % (r, q))], nsys)

print('\nPer-run ratio CUDA / vk_oidn, GPU timers (the two run within the same interleaved round):')
for res, q in ROWS:
  c = gpu['oidn_cuda|cuda_events|%s|%s' % (res, q)]
  v = gpu['vk_oidn|timestamps|%s|%s' % (res, q)]
  ratios = [a / b for a, b in zip(c, v)]
  print('  %s %s: median %.2fx [%.2fx-%.2fx]' % (res, q, statistics.median(ratios), min(ratios), max(ratios)))
