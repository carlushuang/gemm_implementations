gemm implementations
--------------------

cblas_sgemm use openblas build with no_avx512

status:
```
Intel(R) Xeon(R) Gold 6142:
     cblas_sgemm: gflops: 89.485577, cost_per_loop:1.502811ms
  cblas_sgemm_v1: gflops: 7.335408, cost_per_loop:18.332977ms
  cblas_sgemm_v2: gflops: 7.339242, cost_per_loop:18.323400ms
  cblas_sgemm_v3: gflops: 7.319943, cost_per_loop:18.371709ms
  cblas_sgemm_v4: gflops: 9.121355, cost_per_loop:14.743410ms
  cblas_sgemm_v5: gflops: 7.580218, cost_per_loop:17.740899ms
  cblas_sgemm_v6: gflops: 12.358202, cost_per_loop:10.881832ms
  cblas_sgemm_v7: gflops: 11.950769, cost_per_loop:11.252822ms
  cblas_sgemm_v8: gflops: 12.175192, cost_per_loop:11.045400ms
  cblas_sgemm_v9: gflops: 11.406770, cost_per_loop:11.789478ms

AMD Ryzen 7 2700X Eight-Core Processor:
     cblas_sgemm: gflops: 49.208146, cost_per_loop:2.732878ms
  cblas_sgemm_v1: gflops: 6.862054, cost_per_loop:19.597612ms
  cblas_sgemm_v2: gflops: 6.885630, cost_per_loop:19.530511ms
  cblas_sgemm_v3: gflops: 6.922516, cost_per_loop:19.426444ms
  cblas_sgemm_v4: gflops: 10.212351, cost_per_loop:13.168356ms
  cblas_sgemm_v5: gflops: 9.923053, cost_per_loop:13.552268ms
  cblas_sgemm_v6: gflops: 10.387328, cost_per_loop:12.946532ms
  cblas_sgemm_v7: gflops: 10.278183, cost_per_loop:13.084012ms
  cblas_sgemm_v8: gflops: 10.233747, cost_per_loop:13.140824ms
  cblas_sgemm_v9: gflops: 10.115501, cost_per_loop:13.294435ms

```
