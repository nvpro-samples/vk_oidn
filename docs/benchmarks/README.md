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
| [`results/`](results) | raw logs and parsed values of the published runs |

The README chart ([`../images/make_perf_chart.py`](../images/make_perf_chart.py)) is generated from
`results/bench_gpu_results.json`.
