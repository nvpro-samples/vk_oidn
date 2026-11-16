# Benchmarks

Programs, scripts and raw results of the performance comparison between vk_oidn and Open Image
Denoise 2.5.1 (CUDA device). The protocol and the results are described in
[docs/validation_and_performance.md](../validation_and_performance.md#performance).

| File | Purpose |
|------|---------|
| [`oidn_gpu_bench.cpp`](oidn_gpu_bench.cpp), [`CMakeLists.txt`](CMakeLists.txt) | GPU time of OIDN's CUDA device, built against the official 2.5.1 SDK |
| [`bench_paths.py`](bench_paths.py) | locations of the compared programs (environment variables) |
| [`gen_bench_inputs.py`](gen_bench_inputs.py) | random inputs for vk_oidn, with `oidnBenchmark`'s distributions |
| [`bench_gpu.py`](bench_gpu.py) | main metric: GPU time per image (5 interleaved runs) |
| [`bench_latency.py`](bench_latency.py) | secondary metric: per-image latency, `oidnBenchmark` method (5 interleaved runs) |
| [`nsys_profile_all.py`](nsys_profile_all.py) | independent check with Nsight Systems (CUDA and Vulkan) |
| [`accuracy.py`](accuracy.py) | accuracy of each implementation against the OIDN CPU device |
| [`make_tables.py`](make_tables.py) | Markdown tables of the documentation, from `results/*.json` |
| [`perflab.py`](perflab.py) | self-contained run for any GPU (no OIDN needed): vk_oidn GPU time, per-layer times, measured tensor core peak and fraction of it reached |
| [`results/`](results) | raw logs and parsed values of the published runs |

The README chart ([`../images/make_perf_chart.py`](../images/make_perf_chart.py)) is generated from
`results/bench_gpu_results.json`.

## Running on other GPUs

[`perflab.py`](perflab.py) measures one machine with a single command, for performance labs and new
SKUs. It needs CMake, a C++ compiler, the Vulkan SDK (or `slangc` and `glslangValidator`), Python 3
with NumPy, and an NVIDIA driver:

```
python docs/benchmarks/perflab.py --build --out <machine>.json
```

It builds vk_oidn and the [tensor core micro-benchmark](../../tests/microbench), runs 5 interleaved
repetitions of 50 back-to-back denoisings at 1280x720, 1920x1080 and 3840x2160 in high and fast
quality, records the per-layer times of 1920x1080 high, and measures the cooperative matrix peak.
The work of each configuration is counted from the layer table of its weights (565 GFLOP for the
base network at 1920x1088), which gives the speed-of-light time (work / peak) and the fraction of
the peak reached. `--help` lists the options (resolutions, qualities, repetitions, device).

Example, RTX 3500 Ada Generation Laptop GPU (driver 621.24):

| Resolution | Quality | Weights | GFLOP | GPU time | TFLOPS | SOL | Of peak |
|------------|---------|---------|-------|----------|--------|-----|---------|
| 1920x1080 | high | rt_hdr_alb_nrm | 565 | 14.98 ms | 37.7 | 6.75 ms (83.7 TFLOPS) | 45% |
| 1920x1080 | fast | rt_hdr_alb_nrm_small | 292 | 8.89 ms | 32.9 | 3.43 ms (85.1 TFLOPS) | 39% |
