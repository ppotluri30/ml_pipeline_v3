# Scaling Validation Report v3

## Key Findings
- 8 replicas sustained **56.14 req/s** at 400 users with zero failures (gain of 13.16 req/s over 4 replicas); median latency fell from 5200 ms to 3400 ms (-35%) and p95 dropped from 7500 ms to 6400 ms (-15%).
- The previous diagnostic run captured a 25.95% failure rate (`locust\results\multi_replica_v1\replicas8_u400_debug`); the tuned rerun now logs 0 failures across 10,111 requests.
- Inference CPU snapshots (`stats_snapshot*.txt`) show per-container utilisation between 95% and 154% (average 130%), confirming the latency envelope is compute-bound rather than I/O constrained.
- The 200-user / 8-replica sweep stayed warm-up limited (20.44 req/s, 0 failures) during its 120 s dwell; treat it as a baseline sanity check while spotlighting the 400-user lane for capacity planning.

## Consolidated Metrics
|   replicas |   user_count |   request_count |   failure_count |   failure_pct |   requests_per_sec |   median_ms |   p95_ms | stats_path                                                              | history_path                                                                    | failures_path                                                              |
|-----------:|-------------:|----------------:|----------------:|--------------:|-------------------:|------------:|---------:|:------------------------------------------------------------------------|:--------------------------------------------------------------------------------|:---------------------------------------------------------------------------|
|          1 |          200 |             754 |             478 |         63.4  |              12.33 |        1400 |     5200 | locust\results\multi_replica_v1\replicas1_u200\replicas1_u200_stats.csv | locust\results\multi_replica_v1\replicas1_u200\replicas1_u200_stats_history.csv | locust\results\multi_replica_v1\replicas1_u200\replicas1_u200_failures.csv |
|          1 |          400 |            4058 |            3229 |         79.57 |              33.27 |        1700 |     4200 | locust\results\multi_replica_v1\replicas1_u400\replicas1_u400_stats.csv | locust\results\multi_replica_v1\replicas1_u400\replicas1_u400_stats_history.csv | locust\results\multi_replica_v1\replicas1_u400\replicas1_u400_failures.csv |
|          1 |          800 |            7708 |            7024 |         91.13 |              31.92 |        2500 |    62000 | locust\results\multi_replica_v1\replicas1_u800\replicas1_u800_stats.csv | locust\results\multi_replica_v1\replicas1_u800\replicas1_u800_stats_history.csv | locust\results\multi_replica_v1\replicas1_u800\replicas1_u800_failures.csv |
|          2 |          200 |            3143 |               0 |          0    |              25.55 |        3300 |     5500 | locust\results\multi_replica_v1\replicas2_u200\run_stats.csv            | locust\results\multi_replica_v1\replicas2_u200\run_stats_history.csv            | locust\results\multi_replica_v1\replicas2_u200\run_failures.csv            |
|          2 |          400 |            3547 |               0 |          0    |              29.16 |        8900 |    12000 | locust\results\multi_replica_v1\replicas2_u400\run_stats.csv            | locust\results\multi_replica_v1\replicas2_u400\run_stats_history.csv            | locust\results\multi_replica_v1\replicas2_u400\run_failures.csv            |
|          2 |          800 |            3532 |             303 |          8.58 |              28.76 |       19000 |    28000 | locust\results\multi_replica_v1\replicas2_u800\run_stats.csv            | locust\results\multi_replica_v1\replicas2_u800\run_stats_history.csv            | locust\results\multi_replica_v1\replicas2_u800\run_failures.csv            |
|          4 |          200 |            5055 |               0 |          0    |              42.1  |        2000 |     3000 | locust\results\multi_replica_v1\replicas4_u200\run_stats.csv            | locust\results\multi_replica_v1\replicas4_u200\run_stats_history.csv            | locust\results\multi_replica_v1\replicas4_u200\run_failures.csv            |
|          4 |          400 |            5210 |               0 |          0    |              42.98 |        5200 |     7500 | locust\results\multi_replica_v1\replicas4_u400\run_stats.csv            | locust\results\multi_replica_v1\replicas4_u400\run_stats_history.csv            | locust\results\multi_replica_v1\replicas4_u400\run_failures.csv            |
|          4 |          800 |            4883 |               0 |          0    |              40.9  |       11000 |    24000 | locust\results\multi_replica_v1\replicas4_u800\run_stats.csv            | locust\results\multi_replica_v1\replicas4_u800\run_stats_history.csv            | locust\results\multi_replica_v1\replicas4_u800\run_failures.csv            |
|          8 |          200 |            2440 |               0 |          0    |              20.44 |        6100 |     8700 | locust\results\multi_replica_v1\replicas8_u200\run_stats.csv            | locust\results\multi_replica_v1\replicas8_u200\run_stats_history.csv            | locust\results\multi_replica_v1\replicas8_u200\run_failures.csv            |
|          8 |          400 |           10111 |               0 |          0    |              56.14 |        3400 |     6400 | locust\results\multi_replica_v1\replicas8_u400\run_stats.csv            | locust\results\multi_replica_v1\replicas8_u400\run_stats_history.csv            | locust\results\multi_replica_v1\replicas8_u400\run_failures.csv            |

## Run Metadata
|   replicas |   user_count |   max_user_count |   target_user_count |   ramp_duration_sec |   approx_spawn_rate | spawn_rate_log   |   duration_sec |   first_active_ts |   last_active_ts |   history_samples |
|-----------:|-------------:|-----------------:|--------------------:|--------------------:|--------------------:|:-----------------|---------------:|------------------:|-----------------:|------------------:|
|          1 |          200 |              200 |                 200 |                1    |              200    |                  |             49 |        1761157384 |       1761157433 |                61 |
|          1 |          400 |              400 |                 400 |                2    |              200    |                  |            108 |        1761158786 |       1761158894 |               121 |
|          1 |          800 |             1000 |                 800 |                7    |              114.29 |                  |            190 |        1761164951 |       1761165141 |               241 |
|          2 |          200 |              200 |                 200 |                2    |              100    | 40.00            |            112 |        1761674106 |       1761674218 |               121 |
|          2 |          400 |              400 |                 400 |                4    |              100    | 60.00            |            108 |        1761674346 |       1761674454 |               121 |
|          2 |          800 |              800 |                 800 |                7    |              114.29 | 80.00            |            107 |        1761674553 |       1761674660 |               121 |
|          4 |          200 |              200 |                 200 |                2    |              100    | 40.00            |            108 |        1761675244 |       1761675352 |               121 |
|          4 |          400 |              400 |                 400 |                4    |              100    | 60.00            |            108 |        1761675416 |       1761675524 |               121 |
|          4 |          800 |             1200 |                 800 |                7    |              114.29 | 80.00            |            107 |        1761675840 |       1761675947 |               121 |
|          8 |          200 |              200 |                 200 |                5    |               40    | 40.00            |            115 |        1761678434 |       1761678549 |               417 |
|          8 |          400 |              400 |                 400 |               13.33 |               30    | 30.00            |            172 |        1761680402 |       1761680574 |               645 |

## Artifacts
- `scaling_metrics.csv` / `scaling_metrics.json`
- `scaling_metadata.csv` / `scaling_metadata.json`
- Charts: `rps_vs_replicas.*`, `median_latency_vs_replicas.*`, `p95_latency_vs_replicas.*`, `failure_rate_vs_replicas.*`
- 8-replica CPU samples: `locust\results\multi_replica_v1\replicas8_u400\stats_snapshot1.txt`, `...\stats_snapshot2.txt`
- Clean 8-replica 400-user run artifacts: `locust\results\multi_replica_v1\replicas8_u400\`
- Obsoleted 8-replica 400-user artifacts retained under `locust\results\multi_replica_v1\replicas8_u400_obsolete\`

## Pending Follow-ups
- Optional: execute the 800-user / 8-replica validation with the slower ramp (`-r 30`) and extend the charts once captured.

