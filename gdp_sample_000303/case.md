# GDP example: sample_000303

Held-out test prediction (`reference_normalized`).

Truth **9.400**; visual **9.015**; final **9.405**.

## Prediction decomposition

Retrieval +9.3711 + visual directional adjustment -0.3566 = visual prediction 9.0146.
Visual prediction 9.0146 + query semantic adjustment +0.2564 + support semantic adjustment +0.1338 = final prediction 9.4048.

## 3 highest-weight supports

| Training region | Label | Weight | Weighted visual adjustment | Weighted semantic adjustment |
|---|---:|---:|---:|---:|
| sample_000162 | 9.800 | 0.3975 | -0.1045 | +0.0879 |
| sample_000207 | 9.300 | 0.2551 | -0.2041 | -0.0058 |
| sample_000336 | 9.900 | 0.0506 | +0.0033 | +0.0005 |

Shown supports carry 0.703 of total visual weight.
Weights are relative contributions, not calibrated confidence. Full profiles and all support terms are in `case.json`.
