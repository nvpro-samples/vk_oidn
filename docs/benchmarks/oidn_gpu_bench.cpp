// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// GPU time per image of the Open Image Denoise RT filter on its CUDA device (hdr + albedo +
// normal, FP32 images in device buffers), measured like vk_oidn_denoise_test: one warm-up batch of
// N back-to-back denoisings, then a timed batch of N, synchronized once at the end. Built against
// the official Open Image Denoise 2.5.1 SDK and run with its DLLs.
//
// Prints:
//   batched wall-clock: host time from the first executeAsync of the timed batch to the end of
//                       oidnSyncDevice, divided by N;
//   CUDA events:        GPU time between CUDA events recorded on the device's stream before and
//                       after the timed batch, divided by N;
//   enqueue time:       host time spent in the N executeAsync calls, divided by N (shows whether
//                       the batch is GPU bound, i.e. enqueue time < batched time);
//   sync latency:       oidnBenchmark's method, execute + sync per image (for cross-checking).
//
// Inputs are random, with oidnBenchmark's distributions: color [0,100], albedo [0,1], normal [-1,1].
//
// usage: oidn_gpu_bench cuda width height high|fast [N]

#include <OpenImageDenoise/oidn.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

using Clock = std::chrono::steady_clock;
static double seconds(Clock::time_point t0) { return std::chrono::duration<double>(Clock::now() - t0).count(); }

static void check(OIDNDevice device, const char* where)
{
  const char* message;
  if (oidnGetDeviceError(device, &message) != OIDN_ERROR_NONE)
  {
    fprintf(stderr, "error (%s): %s\n", where, message);
    exit(1);
  }
}

int main(int argc, char* argv[])
{
  if (argc < 5 || strcmp(argv[1], "cuda") != 0)
  {
    fprintf(stderr, "usage: oidn_gpu_bench cuda width height high|fast [N]\n");
    return 1;
  }
  const int W = atoi(argv[2]), H = atoi(argv[3]);
  const bool fast = strcmp(argv[4], "fast") == 0;
  const int N = argc > 5 ? atoi(argv[5]) : 50;

  int deviceID = 0;
  cudaStream_t stream = nullptr;
  if (cudaSetDevice(deviceID) != cudaSuccess || cudaStreamCreate(&stream) != cudaSuccess)
  {
    fprintf(stderr, "CUDA initialization failed\n");
    return 1;
  }
  OIDNDevice device = oidnNewCUDADevice(&deviceID, &stream, 1);
  check(device, "new device");
  oidnCommitDevice(device);
  check(device, "commit device");
  printf("Device: CUDA, %dx%d, quality %s, N = %d\n", W, H, fast ? "fast" : "high", N);

  // Random inputs, uploaded to device buffers
  const size_t numValues = size_t(W) * H * 3;
  const size_t byteSize = numValues * sizeof(float);
  std::mt19937 rng(1);
  auto upload = [&](float lo, float hi)
  {
    std::uniform_real_distribution<float> dist(lo, hi);
    std::vector<float> host(numValues);
    for (float& v : host)
      v = dist(rng);
    OIDNBuffer buffer = oidnNewBufferWithStorage(device, byteSize, OIDN_STORAGE_DEVICE); // like oidnBenchmark
    oidnWriteBuffer(buffer, 0, byteSize, host.data());
    return buffer;
  };
  OIDNBuffer color = upload(0.f, 100.f);
  OIDNBuffer albedo = upload(0.f, 1.f);
  OIDNBuffer normal = upload(-1.f, 1.f);
  OIDNBuffer output = oidnNewBufferWithStorage(device, byteSize, OIDN_STORAGE_DEVICE);
  check(device, "buffers");

  OIDNFilter filter = oidnNewFilter(device, "RT");
  oidnSetFilterImage(filter, "color", color, OIDN_FORMAT_FLOAT3, W, H, 0, 0, 0);
  oidnSetFilterImage(filter, "albedo", albedo, OIDN_FORMAT_FLOAT3, W, H, 0, 0, 0);
  oidnSetFilterImage(filter, "normal", normal, OIDN_FORMAT_FLOAT3, W, H, 0, 0, 0);
  oidnSetFilterImage(filter, "output", output, OIDN_FORMAT_FLOAT3, W, H, 0, 0, 0);
  oidnSetFilterBool(filter, "hdr", true);
  oidnSetFilterInt(filter, "quality", fast ? OIDN_QUALITY_FAST : OIDN_QUALITY_HIGH);
  oidnCommitFilter(filter);
  check(device, "commit filter");

  // Warm-up: one denoising, then one batch of N
  oidnExecuteFilter(filter);
  for (int i = 0; i < N; ++i)
    oidnExecuteFilterAsync(filter);
  oidnSyncDevice(device);
  check(device, "warm-up");

  // Timed batch
  cudaEvent_t events[2];
  cudaEventCreate(&events[0]);
  cudaEventCreate(&events[1]);
  cudaEventRecord(events[0], stream);
  Clock::time_point t0 = Clock::now();
  for (int i = 0; i < N; ++i)
    oidnExecuteFilterAsync(filter);
  const double enqueue = seconds(t0);
  cudaEventRecord(events[1], stream);
  oidnSyncDevice(device);
  const double batched = seconds(t0);
  check(device, "timed batch");
  printf("Batched wall-clock: %.3f msec/image\n", batched * 1000.0 / N);
  cudaEventSynchronize(events[1]);
  float msec = 0.f;
  cudaEventElapsedTime(&msec, events[0], events[1]);
  printf("CUDA events: %.3f msec/image\n", msec / N);
  printf("Enqueue time: %.3f msec/image\n", enqueue * 1000.0 / N);

  // oidnBenchmark's method: execute + sync per image, at least 0.5 s or 3 runs
  t0 = Clock::now();
  oidnExecuteFilter(filter);
  const int numRuns = std::max(int(0.5 / seconds(t0)), 3);
  t0 = Clock::now();
  for (int i = 0; i < numRuns; ++i)
    oidnExecuteFilter(filter);
  printf("Sync latency: %.3f msec/image (%d runs)\n", seconds(t0) * 1000.0 / numRuns, numRuns);

  oidnReleaseFilter(filter);
  for (OIDNBuffer b : {color, albedo, normal, output})
    oidnReleaseBuffer(b);
  oidnReleaseDevice(device);
  for (cudaEvent_t e : events)
    cudaEventDestroy(e);
  cudaStreamDestroy(stream);
  return 0;
}
