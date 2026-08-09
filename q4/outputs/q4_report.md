# Q4 Run Summary

## Recommended joint solution

```json
{
  "preferred_scenario": "baseline_joint",
  "selection_rule": "joint two-stage co-optimization minimizing operating cost under physical constraints; scenarios compare carbon budgets, price mechanisms, and renewable volatility",
  "operating_cost_CNY": 1337743076.401438,
  "carbon_tCO2": 1503647.377064048,
  "renewable_utilization": 1.0,
  "peak_net_import_sum_MW": 2730.5854,
  "mean_wait_hour": 0.3472157229761348,
  "mean_network_latency_ms": 25.537201684604586,
  "qos_loss": 0.6172204024333178,
  "hard_pass": true,
  "elapsed_seconds": 109.93101119995117,
  "fast_mode": true,
  "n_tasks_scheduled": 2137
}
```

## Scenario comparison

| scenario                  |   operating_cost_CNY |   carbon_tCO2 |   renewable_utilization |   peak_net_import_sum_MW |   mean_wait_hour |   mean_network_latency_ms | hard_pass   |
|:--------------------------|---------------------:|--------------:|------------------------:|-------------------------:|-----------------:|--------------------------:|:------------|
| baseline_local_first      |          1.3407e+09  |   1.50859e+06 |                1        |                  2568.44 |        0.0145063 |                    5      | True        |
| baseline_lowest_price     |          1.33687e+09 |   1.50646e+06 |                1        |                  2800    |        4.12213   |                   27.854  | True        |
| baseline_lowest_carbon    |          1.33769e+09 |   1.50348e+06 |                1        |                  2730.59 |        2.00562   |                   28.2096 | True        |
| baseline_joint            |          1.33774e+09 |   1.50365e+06 |                1        |                  2730.59 |        0.347216  |                   25.5372 | True        |
| carbon_100                |          1.33774e+09 |   1.50364e+06 |                1        |                  2730.59 |        0.347216  |                   25.5372 | True        |
| carbon_90                 |          1.33774e+09 |   1.50364e+06 |                1        |                  2730.59 |        0.347216  |                   25.5372 | True        |
| carbon_80                 |          1.33774e+09 |   1.50364e+06 |                1        |                  2730.59 |        0.347216  |                   25.5372 | True        |
| carbon_70                 |          1.33774e+09 |   1.50364e+06 |                1        |                  2730.59 |        0.347216  |                   25.5372 | True        |
| price_peak_valley_amplify |          1.21182e+09 |   1.63691e+06 |                1        |                  2620    |        0.349555  |                   25.7061 | True        |
| price_flat                |          1.45058e+09 |   1.45685e+06 |                1        |                  1907.7  |        0.340197  |                   25.496  | True        |
| price_carbon_linked       |          1.4569e+09  |   1.47529e+06 |                1        |                  2695.74 |        0.347216  |                   25.5372 | True        |
| re_minus20                |          1.60922e+09 |   1.65846e+06 |                1        |                  2800    |        0.347216  |                   25.5372 | True        |
| re_plus20                 |          1.10045e+09 |   1.3973e+06  |                0.994955 |                  2393.77 |        0.347216  |                   25.5372 | True        |


Method: two-stage co-optimization. Stage 1 schedules tasks with a multi-factor greedy proxy; Stage 2 solves regional storage–power LPs minimizing operating cost subject to optional carbon / peak / RE-utilization constraints.
