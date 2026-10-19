# Copyright 2026 NVIDIA Corporation
# SPDX-License-Identifier: Apache-2.0

"""Locations of the programs compared by the benchmark scripts, from environment variables.

  OIDN_SDK         official Open Image Denoise 2.5.1 package (contains bin/ and lib/)
  VK_OIDN_BUILD    vk_oidn build directory (vk_oidn_denoise_test), default: <repo>/build
  OIDN_GPU_BENCH   oidn_gpu_bench executable, default: docs/benchmarks/build/oidn_gpu_bench(.exe)
  BENCH_INPUTS     directory of the inputs written by gen_bench_inputs.py, default: docs/benchmarks/inputs"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
EXE = '.exe' if sys.platform == 'win32' else ''


def env_path(name, default=None):
  value = os.environ.get(name, default)
  if value is None:
    sys.exit('set %s (see %s)' % (name, os.path.abspath(__file__)))
  return value


OIDN_SDK = env_path('OIDN_SDK')
VK_OIDN_BUILD = env_path('VK_OIDN_BUILD', os.path.join(REPO, 'build'))
OIDN_GPU_BENCH = env_path('OIDN_GPU_BENCH', os.path.join(HERE, 'build', 'oidn_gpu_bench' + EXE))
BENCH_INPUTS = env_path('BENCH_INPUTS', os.path.join(HERE, 'inputs'))
VK_OIDN_TEST = os.path.join(VK_OIDN_BUILD, 'vk_oidn_denoise_test' + EXE)
RESOLUTIONS = ['1280x720', '1920x1080', '3840x2160']
QUALITIES = ['high', 'fast']


def vk_oidn_inputs(res):
  f = lambda n: os.path.join(BENCH_INPUTS, '%s_%s.pfm' % (n, res))
  return ['--hdr', f('color'), '--alb', f('albedo'), '--nrm', f('normal')]


def gpu_utilization():
  out = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                       capture_output=True, text=True).stdout
  return int(out.strip().splitlines()[0])


def wait_idle(seconds=10):
  """Waits until the GPU utilization has been below 5% for the given time."""
  idle_since = None
  while True:
    if gpu_utilization() < 5:
      idle_since = idle_since or time.time()
      if time.time() - idle_since >= seconds:
        return
    else:
      idle_since = None
    time.sleep(1)
