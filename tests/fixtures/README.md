# Retained terminal-placement observations

`retained_terminal.json.gz` contains origins, horizons, terminal VM-to-host assignments, and recorded native metrics from four earlier `results/full_trace/<cohort>.json` files. It covers eight policies at 25 origins on four cohorts, for 800 metric sets.

`test_all_800_retained_terminal_metric_sets` recomputes six native metrics from each assignment using the original Arrow workloads and compares them with the recorded values. This checks the data reader and metric implementation. The fixture does not reproduce the earlier solver or add experimental evidence for the current method.
