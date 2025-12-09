# Scaling Validation Report

All replica tiers (1, 2, 4) consolidated under `locust/results/multi_replica_v1`.

## Key Findings
- For 200 users: RPS improved by 29.77 and failure rate changed by 63.40 pts when scaling from 1 to 4 replicas.
- For 400 users: RPS improved by 9.71 and failure rate changed by 79.57 pts when scaling from 1 to 4 replicas.
- For 800 users: RPS improved by 8.98 and failure rate changed by 91.13 pts when scaling from 1 to 4 replicas.
- Metadata discrepancies detected:
  - Replica 1.0 / 800.0 users: max_user_count=1000.0
  - Replica 2.0 / 200.0 users: spawn mismatch log=40.0 approx=100.00
  - Replica 2.0 / 400.0 users: spawn mismatch log=60.0 approx=100.00
  - Replica 2.0 / 800.0 users: spawn mismatch log=80.0 approx=114.29
  - Replica 4.0 / 200.0 users: spawn mismatch log=40.0 approx=100.00
  - Replica 4.0 / 400.0 users: spawn mismatch log=60.0 approx=100.00
  - Replica 4.0 / 800.0 users: max_user_count=1200.0
  - Replica 4.0 / 800.0 users: spawn mismatch log=80.0 approx=114.29

## Consolidated Metrics
|   replicas |   user_count |   request_count |   failure_count |   failure_pct |   requests_per_sec |   median_ms |   p95_ms | stats_path                                                              | history_path                                                                    | failures_path                                                              |
|------------|--------------|-----------------|-----------------|---------------|--------------------|-------------|----------|-------------------------------------------------------------------------|---------------------------------------------------------------------------------|----------------------------------------------------------------------------|
|          1 |          200 |          754.00 |          478.00 |         63.40 |              12.33 |     1400.00 |  5200.00 | locust\results\multi_replica_v1\replicas1_u200\replicas1_u200_stats.csv | locust\results\multi_replica_v1\replicas1_u200\replicas1_u200_stats_history.csv | locust\results\multi_replica_v1\replicas1_u200\replicas1_u200_failures.csv |
|          1 |          400 |         4058.00 |         3229.00 |         79.57 |              33.27 |     1700.00 |  4200.00 | locust\results\multi_replica_v1\replicas1_u400\replicas1_u400_stats.csv | locust\results\multi_replica_v1\replicas1_u400\replicas1_u400_stats_history.csv | locust\results\multi_replica_v1\replicas1_u400\replicas1_u400_failures.csv |
|          1 |          800 |         7708.00 |         7024.00 |         91.13 |              31.92 |     2500.00 | 62000.00 | locust\results\multi_replica_v1\replicas1_u800\replicas1_u800_stats.csv | locust\results\multi_replica_v1\replicas1_u800\replicas1_u800_stats_history.csv | locust\results\multi_replica_v1\replicas1_u800\replicas1_u800_failures.csv |
|          2 |          200 |         3143.00 |            0.00 |          0.00 |              25.55 |     3300.00 |  5500.00 | locust\results\multi_replica_v1\replicas2_u200\run_stats.csv            | locust\results\multi_replica_v1\replicas2_u200\run_stats_history.csv            | locust\results\multi_replica_v1\replicas2_u200\run_failures.csv            |
|          2 |          400 |         3547.00 |            0.00 |          0.00 |              29.16 |     8900.00 | 12000.00 | locust\results\multi_replica_v1\replicas2_u400\run_stats.csv            | locust\results\multi_replica_v1\replicas2_u400\run_stats_history.csv            | locust\results\multi_replica_v1\replicas2_u400\run_failures.csv            |
|          2 |          800 |         3532.00 |          303.00 |          8.58 |              28.76 |    19000.00 | 28000.00 | locust\results\multi_replica_v1\replicas2_u800\run_stats.csv            | locust\results\multi_replica_v1\replicas2_u800\run_stats_history.csv            | locust\results\multi_replica_v1\replicas2_u800\run_failures.csv            |
|          4 |          200 |         5055.00 |            0.00 |          0.00 |              42.10 |     2000.00 |  3000.00 | locust\results\multi_replica_v1\replicas4_u200\run_stats.csv            | locust\results\multi_replica_v1\replicas4_u200\run_stats_history.csv            | locust\results\multi_replica_v1\replicas4_u200\run_failures.csv            |
|          4 |          400 |         5210.00 |            0.00 |          0.00 |              42.98 |     5200.00 |  7500.00 | locust\results\multi_replica_v1\replicas4_u400\run_stats.csv            | locust\results\multi_replica_v1\replicas4_u400\run_stats_history.csv            | locust\results\multi_replica_v1\replicas4_u400\run_failures.csv            |
|          4 |          800 |         4883.00 |            0.00 |          0.00 |              40.90 |    11000.00 | 24000.00 | locust\results\multi_replica_v1\replicas4_u800\run_stats.csv            | locust\results\multi_replica_v1\replicas4_u800\run_stats_history.csv            | locust\results\multi_replica_v1\replicas4_u800\run_failures.csv            |

## Run Metadata
|   replicas |   user_count |   max_user_count |   target_user_count |   ramp_duration_sec |   approx_spawn_rate |   spawn_rate_log |   duration_sec |   first_active_ts |   last_active_ts |   history_samples |
|------------|--------------|------------------|---------------------|---------------------|---------------------|------------------|----------------|-------------------|------------------|-------------------|
|       1.00 |       200.00 |           200.00 |              200.00 |                1.00 |              200.00 |           nan    |          49.00 |     1761157384.00 |    1761157433.00 |             61.00 |
|       1.00 |       400.00 |           400.00 |              400.00 |                2.00 |              200.00 |           nan    |         108.00 |     1761158786.00 |    1761158894.00 |            121.00 |
|       1.00 |       800.00 |          1000.00 |              800.00 |                7.00 |              114.29 |           nan    |         190.00 |     1761164951.00 |    1761165141.00 |            241.00 |
|       2.00 |       200.00 |           200.00 |              200.00 |                2.00 |              100.00 |            40.00 |         112.00 |     1761674106.00 |    1761674218.00 |            121.00 |
|       2.00 |       400.00 |           400.00 |              400.00 |                4.00 |              100.00 |            60.00 |         108.00 |     1761674346.00 |    1761674454.00 |            121.00 |
|       2.00 |       800.00 |           800.00 |              800.00 |                7.00 |              114.29 |            80.00 |         107.00 |     1761674553.00 |    1761674660.00 |            121.00 |
|       4.00 |       200.00 |           200.00 |              200.00 |                2.00 |              100.00 |            40.00 |         108.00 |     1761675244.00 |    1761675352.00 |            121.00 |
|       4.00 |       400.00 |           400.00 |              400.00 |                4.00 |              100.00 |            60.00 |         108.00 |     1761675416.00 |    1761675524.00 |            121.00 |
|       4.00 |       800.00 |          1200.00 |              800.00 |                7.00 |              114.29 |            80.00 |         107.00 |     1761675840.00 |    1761675947.00 |            121.00 |

## Artifacts
- `scaling_metrics.csv` / `scaling_metrics.json`
- `scaling_metadata.csv` / `scaling_metadata.json`
- Charts (`*.png`/`*.pdf`) in this folder