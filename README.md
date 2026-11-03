# ADDIS ABABA UNIVERSITY
## College of Technology and Built Environment
### School of Information Technology and Engineering

**Course:** Selected Topics in AI | Assignment 1 Submission  
**Project Title:** Solving NP-Hard Problems Using Nature-Inspired Algorithms  
**Registered Metaheuristic Algorithm:** Simulated Annealing (SA)  
**Selected NP-Hard Problem & Instances:** Traveling Salesperson Problem (TSP) — att48, ch130, a280  
**Group Repository URL:** [To be completed by the group]

### Group Member Roster & Contribution Matrix

| No. | Full Student Name | Student ID | Assigned Task / Focus Area | Contribution |
|---|---|---|---|---:|
| 1 | [Group Leader Name] | [ID] | [Focus area] | [ ]% |
| 2 | [Member Name] | [ID] | [Focus area] | [ ]% |
| 3 | [Member Name] | [ID] | [Focus area] | [ ]% |
| 4 | [Member Name] | [ID] | [Focus area] | [ ]% |

> Replace every bracketed placeholder before submission. Contributions must sum
> to 100% and match the commit history.

# Simulated Annealing for the Traveling Salesperson Problem

## Repository layout

- `src/simulated_annealing.py` contains the generic from-scratch algorithm,
  temperature schedule, Metropolis acceptance rule, and result tracking.
- `src/tsp_problem.py` contains TSPLIB loading, Euclidean distances, tour cost,
  permutation validation, the 2-opt neighbour move, and the TSP solver.
- `src/baseline.py` contains the budget-matched random-restart 2-opt hill
  climbing baseline, plus a single-start nearest-neighbour reference.
- `data/` contains the downloader and source documentation.
- `experiments/` contains parameter tuning, 30-run experiment scripts,
  statistical summaries, plots, and `run_all.sh`.
- `results/` contains separate output directories for SA, baseline, tuning,
  summaries, and plots.

Each algorithm writes one trace table per run, for example
`results/sa/att48_seed2026.csv`, with columns `iteration`,
`evaluations_used`, `best_cost`, `elapsed_seconds`, and `temperature`.
`att48_summary.csv` is a compact one-row-per-run table containing the seed,
final cost, evaluations, and tour. Baseline traces use the same columns;
temperature is empty because hill climbing has no temperature schedule.

## Evaluation budget and cooling schedule

The default budget is exactly 50,000 objective evaluations per SA run. The
initial tour cost is evaluation 1, and each candidate move consumes one more
evaluation, so the loop performs 49,999 candidate moves. With 100 moves per
temperature block there are 500 blocks. The cooling rate is calculated as

```text
rate = (final_temperature / initial_temperature) ** (1 / 500)
```

so the temperature approaches the configured final value at the end of the
budget (the final block runs at approximately that value). The default final temperature is `0.25`; the tuning script also tests
`0.1` and `0.5`. When no initial temperature is supplied, `tsp_problem.py` samples
worsening 2-opt moves and chooses a value that accepts a typical worsening move
with probability about 0.8. The 128 sampling moves are calibration work and are
not counted in the optimization budget; the selected target is recorded in the
experiment configuration.

Each SA CSV row records the run result and a trace every 100 iterations,
including the first and last records. The trace contains iteration,
evaluations used, best-so-far cost, elapsed seconds, and temperature. TSP
distances are precomputed once, and 2-opt candidate costs use the exact
four-edge delta; the final tour is checked with a full cost calculation.

The budget-matched baseline uses random-restart 2-opt hill climbing. It accepts
only improving moves, restarts after a fixed number of non-improving attempts,
and keeps the best tour from 50,000 objective evaluations. Its trace uses the
same compact fields and interval as SA. A single nearest-neighbour construction
is retained as a quick reference, but is not the budget-matched comparison.

## Selected tuned settings

The final runs use settings selected from the 50,000-evaluation tuning study.
The initial grid used five runs per setting; the wider follow-up used two
additional validation runs per setting and was used for pattern confirmation
rather than treating one noisy minimum as definitive:

| Instance | Acceptance target | Initial-temperature scale | Final temperature | Moves/block |
|---|---:|---:|---:|---:|
| att48 | 0.7 | 0.75 | 0.1 | 200 |
| ch130 | 0.3 | 0.25 | 0.5 | 50 |
| a280 | 0.1 | 0.25 | 0.5 | 200 |

The wider follow-up sweep tests lower acceptance targets and smaller
temperature scales for `a280` and `ch130`. Tuning seeds begin at 9000 and are
separate from final-run seeds beginning at 2026. Results from five runs per
setting are noisy, so the settings are selected from consistent patterns.

## Results snapshot

Regenerate this table after the final tuned runs with
`python3 -m experiments.analyze_results`:

| Algorithm | Instance | Best | Mean | Median | Best gap | Mean gap |
|---|---|---:|---:|---:|---:|---:|
| SA | att48 | 10638 | 10902.87 | 10864 | 0.09% | 2.59% |
| SA | ch130 | 6419 | 6662.27 | 6657 | 5.06% | 9.04% |
| SA | a280 | 3587 | 3782.00 | 3788 | 39.08% | 46.65% |
| Random-restart 2-opt | att48 | 11514 | 12144.07 | 12135 | 8.34% | 14.26% |
| Random-restart 2-opt | ch130 | 7945 | 8708.00 | 8731.5 | 30.03% | 42.52% |
| Random-restart 2-opt | a280 | 4227 | 4539.00 | 4498.5 | 63.90% | 76.00% |

## Reproducibility environment

- Python: 3.12.3
- matplotlib: 3.6.3
- scipy: 1.11.4
- Platform: Linux x86_64
- Budget: 50,000 objective evaluations per run

## AI Use Statement

AI assistance was used for grammar review, cooling-schedule arithmetic,
2-opt delta-code review, repository structure suggestions, debugging, and
test planning. The group members reviewed, tested, and take responsibility for
the final algorithm, experiments, analysis, report, and presentation. Complete
this statement with the actual tools and contributions used by the group.

## Run the example

```bash
python3 -m src.tsp_problem
```

The assignment requires TSPLIB `att48`, `ch130`, and `a280`, with 30 independent runs, a
nearest-neighbour comparison, convergence plots, and comparison with published
best-known values or a permitted solver. The complete workflow is:

```bash
bash experiments/run_all.sh
```

The scripts are intended to run from the repository root using module syntax:

```bash
python3 -m experiments.run_sa
python3 -m experiments.run_baseline
python3 -m experiments.analyze_results
python3 -m experiments.make_plots
```

Run `python3 data/download_data.py` first if you want to inspect or cite the
downloaded benchmark files separately.

The tuning study also supports the documented bonus parameter-sensitivity
analysis. Generate its heat maps with:

```bash
python3 -m experiments.make_tuning_heatmaps
```

## Academic integrity

This code uses no metaheuristic or optimization library. The group must review,
understand, test, and explain every part of it, and must complete the cover
page, member contributions, AI-use statement, experiments, analysis, report,
and slides themselves according to the course rules.
