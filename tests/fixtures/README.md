# Terminal-placement fixtures

`retained_terminal.json.gz` contains 800 terminal placement and metric records: eight policies at 25 origins on each of four decision cohorts. The fixtures preserve the recorded assignments, horizons and native metrics and are independent of the current search implementation.

`test_all_800_retained_terminal_metric_sets` reads the complete public Arrow inputs and recomputes all six native metrics from those assignments. These records are numerical regression fixtures; the paper results are under `results/`.
