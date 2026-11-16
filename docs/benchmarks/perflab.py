#!/usr/bin/env python3
# Copyright 2026 NVIDIA Corporation
# SPDX-License-Identifier: Apache-2.0

"""Self-contained vk_oidn benchmark for running on many GPUs (performance lab, new SKUs).

On the machine under test, it:
  1. optionally builds vk_oidn and the tensor core micro-benchmark (--build),
  2. writes the random inputs if they are missing (same as gen_bench_inputs.py),
  3. measures the GPU time per image of vk_oidn (vk_oidn_denoise_test -n N: a warm-up batch, then
     N back-to-back denoisings between two GPU timestamps), REPEATS runs interleaved over the
     resolutions and qualities; hdr + albedo + normal, FP32 images in device memory,
  4. records the per-layer times of one 1920x1080 high quality run (VK_OIDN_PROFILE=1),
  5. measures the cooperative matrix peak of the GPU (tests/microbench/peak.comp, register-only
     multiply-add chains, best of 1-8 independent chains and of 3 runs),
  6. computes the network work from the layer table of the packed weights, and the speed-of-light
     time (work / measured peak) and the fraction of the peak reached.

Unlike bench_gpu.py, it does not need Open Image Denoise. Results are printed and written as JSON
(--out). The exit code is non-zero if a denoising run fails; a missing peak (GPU without
VK_NV_cooperative_vector or VK_EXT_shader_float8, which the micro-benchmark harness requires) is
reported as null.

usage: perflab.py [--build] [--out results.json] [--repeats 5] [--runs 50]
                  [--res 1280x720,1920x1080,3840x2160] [--quality high,fast] [--no-peak] [--device N]

The work is the nominal 3x3 convolution count of the network (2 x 9 x padded input channels x
padded output channels per output pixel of each layer, without the sub-pixel decomposition, which
does 56% fewer multiply-adds on the upsampled sources): 565 GFLOP for the base network on a
1920x1088 tile. For tiled resolutions it uses the size of the first tile for every tile, an upper
bound (the last tiles can be smaller)."""
import argparse
import datetime
import json
import os
import platform
import re
import shutil
import statistics
import struct
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
EXE = '.exe' if sys.platform == 'win32' else ''
MICROBENCH = os.path.join(REPO, 'tests', 'microbench')


def run(cmd, env=None, cwd=None, check=False):
  r = subprocess.run(cmd, env=env, cwd=cwd, capture_output=True, text=True)
  if check and r.returncode != 0:
    sys.exit('command failed: %s\n%s%s' % (' '.join(cmd), r.stdout, r.stderr))
  return r


def find_exe(build_dir, name):
  for sub in ('Release', ''):
    path = os.path.join(build_dir, sub, name + EXE)
    if os.path.isfile(path):
      return path
  return None


def build(src, build_dir):
  print('building %s' % os.path.relpath(src, REPO) if src != REPO else 'building vk_oidn', flush=True)
  run(['cmake', '-S', src, '-B', build_dir, '-DCMAKE_BUILD_TYPE=Release'], check=True)
  run(['cmake', '--build', build_dir, '--config', 'Release', '--parallel'], check=True)


def write_inputs(directory, resolutions):
  import numpy as np  # only needed to create the inputs
  os.makedirs(directory, exist_ok=True)
  rng = np.random.default_rng(1)
  for res in resolutions:
    w, h = map(int, res.split('x'))
    for name, lo, hi in (('color', 0.0, 100.0), ('albedo', 0.0, 1.0), ('normal', -1.0, 1.0)):
      path = os.path.join(directory, '%s_%s.pfm' % (name, res))
      data = rng.uniform(lo, hi, (h, w, 3)).astype('<f4')  # always drawn, so the values do not depend on what exists
      if not os.path.isfile(path):
        with open(path, 'wb') as f:
          f.write(b'PF\n%d %d\n-1.0\n' % (w, h))
          f.write(data.tobytes())


def network_flops_per_tile(weights_path, tile_w, tile_h):
  """Nominal FLOPs of the convolutions for one tile, from the layer table of the packed weights
  (src/weights_format.h)."""
  with open(weights_path, 'rb') as f:
    blob = f.read()
  if blob[:8] != b'VKOIDNW1':
    raise ValueError('not a packed weights file: ' + weights_path)
  header_size, num_layers = struct.unpack_from('<II', blob, 12)
  scale = {-1: 1.0}  # resolution of each tensor relative to the tile (-1 = network input)
  flops = 0.0
  for i in range(num_layers):
    name, src0, src1, flags, c0, c1, cout = struct.unpack_from('<24siiIIII', blob, header_size + i * 64)
    conv_scale = scale[src0] * (2.0 if flags & 4 else 1.0)  # upsampled first source
    pixels = (tile_w * conv_scale) * (tile_h * conv_scale)
    flops += 2.0 * 9 * (c0 + c1) * cout * pixels
    scale[i] = conv_scale / (2.0 if flags & 2 else 1.0)  # pooled output
  return flops


def glslang():
  sdk = os.environ.get('VULKAN_SDK', '')
  for path in (os.path.join(sdk, 'Bin', 'glslangValidator' + EXE), os.path.join(sdk, 'bin', 'glslangValidator' + EXE),
               shutil.which('glslangValidator'), shutil.which('glslang')):
    if path and os.path.isfile(path):
      return path
  return None


def measure_peak(bench):
  """Cooperative matrix peak in TFLOPS (as tests/microbench/run_all.sh), or None per type."""
  compiler = glslang()
  if not bench or not compiler:
    print('peak: skipped (%s not found)' % ('glslangValidator' if bench else 'microbench harness'))
    return {}
  configs = {'fp16_fp16': ('-DTYPE_F16 -DACC_F16', 16), 'fp16_fp32': ('-DTYPE_F16 -DACC_F32', 16),
             'fp8_fp16': ('-DTYPE_FP8 -DACC_F16', 32), 'fp8_fp32': ('-DTYPE_FP8 -DACC_F32', 32)}
  peaks = {}
  with tempfile.TemporaryDirectory() as tmp:
    spv = os.path.join(tmp, 'peak.spv')
    for key, (defines, kk) in configs.items():
      best = None
      for nacc in (1, 2, 4, 8):
        r = run([compiler, '-V', '--target-env', 'vulkan1.3', '--quiet'] + defines.split() +
                ['-DNACC=%d' % nacc, '-o', spv, os.path.join(MICROBENCH, 'peak.comp')])
        if r.returncode != 0:
          continue
        # Best of 3 runs: clocks and power limits (laptops especially) make single runs noisy
        for _ in range(3):
          r = run([bench, spv, '8192', '1', '1', '--buf', '1', '--buf', '4', '--int', '256', '--iters', '10'])
          m = re.search(r'median ([0-9.]+) ms', r.stdout)
          if m:
            tflops = 2 * 16 * 16 * kk * 256 * nacc * 8192 * 4 / float(m.group(1)) / 1e9
            best = max(best or 0.0, tflops)
      peaks[key] = round(best, 1) if best else None
      print('peak %-10s %s TFLOPS' % (key, peaks[key]), flush=True)
  return peaks


def gpu_info(device):
  info = {}
  smi = shutil.which('nvidia-smi')
  if smi:
    r = run([smi, '--query-gpu=name,driver_version,clocks.max.sm,memory.total,power.limit', '--format=csv,noheader'])
    lines = [l.strip() for l in r.stdout.splitlines() if l.strip()]
    if lines:
      fields = [x.strip() for x in lines[min(device, len(lines) - 1)].split(',')]
      info = dict(zip(['name', 'driver', 'max_sm_clock', 'memory', 'power_limit'], fields))
  return info


def main():
  ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
  ap.add_argument('--build', action='store_true', help='configure and build vk_oidn and the micro-benchmark first')
  ap.add_argument('--vk-oidn-build', default=os.path.join(REPO, 'build'), help='vk_oidn build directory')
  ap.add_argument('--microbench-build', default=os.path.join(REPO, 'build_microbench'))
  ap.add_argument('--inputs', default=os.path.join(HERE, 'inputs'))
  ap.add_argument('--out', default=os.path.join(HERE, 'results', 'perflab.json'))
  ap.add_argument('--repeats', type=int, default=5)
  ap.add_argument('--runs', type=int, default=50, help='denoisings per timed batch')
  ap.add_argument('--res', default='1280x720,1920x1080,3840x2160')
  ap.add_argument('--quality', default='high,fast')
  ap.add_argument('--no-peak', action='store_true')
  ap.add_argument('--device', type=int, help='Vulkan device index (vk_oidn_denoise_test -d, default: its own choice)')
  args = ap.parse_args()
  resolutions = args.res.split(',')
  qualities = args.quality.split(',')

  if args.build:
    build(REPO, args.vk_oidn_build)
    if not args.no_peak:
      build(MICROBENCH, args.microbench_build)
  test = find_exe(args.vk_oidn_build, 'vk_oidn_denoise_test')
  if not test:
    sys.exit('vk_oidn_denoise_test not found in %s (use --build)' % args.vk_oidn_build)
  write_inputs(args.inputs, resolutions)

  def denoise(res, quality, extra=(), env=None):
    f = lambda n: os.path.join(args.inputs, '%s_%s.pfm' % (n, res))
    cmd = [test, '--hdr', f('color'), '--alb', f('albedo'), '--nrm', f('normal'), '-q', quality] + list(extra)
    if args.device is not None:
      cmd += ['-d', str(args.device)]
    return run(cmd, env=env)

  result = {
    'date': datetime.datetime.now().isoformat(timespec='seconds'),
    'host': platform.node(), 'os': platform.platform(), 'cpu': platform.processor() or platform.machine(),
    'gpu': gpu_info(args.device or 0),
    'vk_oidn_commit': run(['git', '-C', REPO, 'rev-parse', '--short', 'HEAD']).stdout.strip() or None,
    'method': 'GPU time per image (Vulkan timestamps), median of %d interleaved runs of %d back-to-back denoisings; '
              'hdr + albedo + normal, FP32 buffers' % (args.repeats, args.runs),
    'configs': {}, 'profile': None, 'peak_tflops': {},
  }

  # Static information of each configuration (weights, tiles, tile size) from a profiled run
  failed = False
  for res in resolutions:
    for quality in qualities:
      r = denoise(res, quality, ['-n', '2'], dict(os.environ, VK_OIDN_PROFILE='1'))
      dev = re.search(r'^Device: (.*?), cooperative matrices: (\w+)', r.stdout, re.M)
      w = re.search(r'^Weights: (\S+), tiles: (\d+), memory: ([0-9.]+) MB', r.stdout, re.M)
      t = re.search(r'^vk_oidn profile \(\S+, (\d+)x(\d+),', r.stdout, re.M)
      if r.returncode != 0 or not (dev and w and t):
        print('%s %s: FAILED\n%s%s' % (res, quality, r.stdout, r.stderr))
        failed = True
        continue
      result['device'] = dev.group(1)
      result['cooperative_matrices'] = dev.group(2) == 'yes'
      weights, tiles = w.group(1), int(w.group(2))
      tile_w, tile_h = int(t.group(1)), int(t.group(2))
      flops = network_flops_per_tile(os.path.join(REPO, 'weights', weights + '.vkoidnw'), tile_w, tile_h) * tiles
      result['configs']['%s|%s' % (res, quality)] = {
        'resolution': res, 'quality': quality, 'weights': weights, 'tiles': tiles, 'tile': '%dx%d' % (tile_w, tile_h),
        'memory_mb': float(w.group(3)), 'gflop': round(flops / 1e9, 1), 'times_ms': []}
      if res == '1920x1080' and quality == 'high':
        result['profile'] = {m.group(1): float(m.group(2))
                             for m in re.finditer(r'^\s+(\w+)\s+([0-9.]+) ms', r.stdout, re.M) if m.group(1) != 'total'}
  print('device: %s' % result.get('device'), flush=True)

  # Interleaved timed runs
  for rep in range(args.repeats):
    for key, cfg in result['configs'].items():
      r = denoise(cfg['resolution'], cfg['quality'], ['-n', str(args.runs)])
      m = re.search(r'^GPU time: ([0-9.]+) msec', r.stdout, re.M)
      if r.returncode != 0 or not m:
        print('%s rep %d: FAILED\n%s%s' % (key, rep, r.stdout, r.stderr))
        failed = True
        continue
      cfg['times_ms'].append(float(m.group(1)))
      print('%-16s rep %d: %s ms' % (key, rep, m.group(1)), flush=True)

  if not args.no_peak:
    result['peak_tflops'] = measure_peak(find_exe(args.microbench_build, 'bench'))

  # Summary: high quality accumulates in FP32, the other qualities in FP16 (with cooperative matrices)
  print('\n%-10s %-8s %-24s %10s %22s %9s %9s %8s' % ('res', 'quality', 'weights', 'GFLOP', 'ms median [min-max]',
                                                     'TFLOPS', 'SOL ms', 'of peak'))
  for cfg in result['configs'].values():
    v = cfg['times_ms']
    if not v:
      continue
    cfg['median_ms'], cfg['min_ms'], cfg['max_ms'] = statistics.median(v), min(v), max(v)
    cfg['tflops'] = round(cfg['gflop'] / cfg['median_ms'], 1)
    peak = result['peak_tflops'].get('fp16_fp32' if cfg['quality'] == 'high' else 'fp16_fp16')
    cfg['peak_tflops'] = peak
    cfg['sol_ms'] = round(cfg['gflop'] / peak, 3) if peak else None
    cfg['fraction_of_peak'] = round(cfg['tflops'] / peak, 3) if peak else None
    print('%-10s %-8s %-24s %10.1f %8.3f [%.3f-%.3f] %9.1f %9s %8s' % (
      cfg['resolution'], cfg['quality'], cfg['weights'], cfg['gflop'], cfg['median_ms'], cfg['min_ms'], cfg['max_ms'],
      cfg['tflops'], '%.3f' % cfg['sol_ms'] if peak else '-', '%.0f%%' % (100 * cfg['fraction_of_peak']) if peak else '-'))

  os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
  with open(args.out, 'w') as f:
    json.dump(result, f, indent=1)
  print('\nwrote %s' % args.out)
  return 1 if failed else 0


if __name__ == '__main__':
  sys.exit(main())
