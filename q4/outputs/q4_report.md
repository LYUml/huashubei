# Q4 Run Summary

## Recommended joint solution

```json
{
  "preferred_scenario": "baseline_joint",
  "selection_rule": "joint two-stage co-optimization minimizing operating cost under physical constraints; scenarios compare carbon budgets, price mechanisms, and renewable volatility",
  "operating_cost_CNY": 1632012953.4730585,
  "carbon_tCO2": 1611850.536671484,
  "renewable_utilization": 1.0,
  "peak_net_import_sum_MW": 2730.5854,
  "mean_wait_hour": 0.27208,
  "mean_network_latency_ms": 25.89244,
  "qos_loss": 0.54637,
  "hard_pass": true,
  "elapsed_seconds": 1409.3668265342712,
  "fast_mode": false,
  "n_tasks_scheduled": 50000
}
```

## Scenario comparison

| scenario                  |   operating_cost_CNY |   carbon_tCO2 |   renewable_utilization |   peak_net_import_sum_MW |   mean_wait_hour |   mean_network_latency_ms | hard_pass   |
|:--------------------------|---------------------:|--------------:|------------------------:|-------------------------:|-----------------:|--------------------------:|:------------|
| baseline_local_first      |          1.70619e+09 |   1.74144e+06 |                1        |                  2669.65 |          0       |                   5.05704 | True        |
| baseline_lowest_price     |          1.60702e+09 |   1.68402e+06 |                1        |                  2800    |         77.1671  |                  28.0894  | True        |
| baseline_lowest_carbon    |          1.63219e+09 |   1.61596e+06 |                1        |                  2731.66 |         32.7886  |                  28.666   | True        |
| baseline_joint            |          1.63201e+09 |   1.61185e+06 |                1        |                  2730.59 |          0.27208 |                  25.8924  | True        |
| carbon_100                |          1.63201e+09 |   1.61185e+06 |                1        |                  2730.59 |          0.27208 |                  25.8924  | True        |
| carbon_90                 |          1.63219e+09 |   1.61596e+06 |                1        |                  2731.66 |         32.7886  |                  28.666   | True        |
| carbon_80                 |          1.63219e+09 |   1.61596e+06 |                1        |                  2731.66 |         32.7886  |                  28.666   | True        |
| carbon_70                 |          1.63219e+09 |   1.61596e+06 |                1        |                  2731.66 |         32.7886  |                  28.666   | True        |
| price_peak_valley_amplify |          1.50821e+09 |   1.7307e+06  |                1        |                  2631.25 |          0.27036 |                  25.8956  | True        |
| price_flat                |          1.76617e+09 |   1.59923e+06 |                1        |                  2095.41 |          0.26928 |                  25.9388  | True        |
| price_carbon_linked       |          1.76044e+09 |   1.59788e+06 |                1        |                  2709.82 |          0.27444 |                  25.8981  | True        |
| re_minus20                |          1.93343e+09 |   1.83374e+06 |                1        |                  2800    |          0.27208 |                  25.8924  | True        |
| re_plus20                 |          1.34742e+09 |   1.44002e+06 |                0.999454 |                  2585.41 |          0.27208 |                  25.8924  | True        |
| peak_cap_90               |          1.63209e+09 |   1.6123e+06  |                1        |                  2457.53 |          0.27208 |                  25.8924  | True        |


Method: two-stage co-optimization. Stage 1 schedules tasks with a multi-factor greedy proxy; Stage 2 solves regional storage–power LPs minimizing operating cost subject to optional carbon / peak / RE-utilization constraints.
