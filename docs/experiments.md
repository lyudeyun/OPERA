# Experiments

Settings of the experiments and how to run them. The systems and datasets are described in
[benchmarks.md](benchmarks.md), the performance metrics in [metrics.md](metrics.md).

## Settings

**Repair.** For every unsafe execution of the retraining set, NSGA-II searches for a control patch with
three objectives: maximize the robustness of the patched execution, and minimize the time length and
the L1-norm of the patch. Population size 40, 10 generations, so one NSGA-II run costs 40 x 10 = 400
system executions. A patch is a piecewise-cubic signal over k = 3 segments, so an individual has 6
decision variables: the start and end time of the patch and its values at the k + 1 segment endpoints
(the two outer ones are fixed to 0). If no individual satisfies the specification, the search is
repeated, up to 10 more times (`rep_budget`). In the configurations: `pop_size 40`, `max_gen 10`,
`M 3`, `V 6`, `N 4`, `rep_budget 10`.

**Selection strategies.** Among the satisfying individuals of the last generation that has any, the
patch is chosen by `st_id`:

| `st_id` | Paper | Chooses |
| --- | --- | --- |
| 1 | MinSat | the lowest robustness |
| 2 | MaxSat | the highest robustness |
| 3 | SimilarT | the shortest patch |
| 4 | SimilarI | the smallest L1-norm |
| 5 | Explore | one at random |

**Retraining and evaluation.** For each strategy, the controller is retrained 10 times on the repaired
retraining set and each retrained controller is evaluated on the test set.

**Baseline.** ISAR repairs the controller weights by simulated annealing, one unsafe execution at a
time, and rejects any change that makes a safe execution unsafe. Its verifier needs a white-box plant
model, so it is skipped and ISAR runs with the default hyperparameters of its implementation. Each
retraining set is downsampled to 100 traces (5 such sets) and ISAR runs 10 times on each, i.e., 50
runs per benchmark. The ISAR code is in a separate repository
([lyudeyun/ISAR](https://github.com/lyudeyun/ISAR)).

## Running a pipeline

Each step: a Python script in `test/` reads a configuration and writes the job scripts to
`test/scripts/`, then `make` in the root runs them. Example, AFC#2:

```
cd test
python3 gen_dataset.py config/AFC/gendata2     # retraining and test sets
python3 fastOpera.py config/AFC/repair2        # repair; once per st_id = 1..5
python3 retrain.py config/AFC/retrain2         # retrain, 10 times per strategy
python3 rep_eval_afc.py config/AFC/rep_eval2   # evaluate on the test set
```

Run `make` in the root after each command. Between repair and retraining, drop `_Start_1_End_10000`
from the name of the repaired set and normalize it with `util/preprocess.m` (this writes the `Nor_...`
file read by the retraining configuration); between retraining and evaluation, put the retrained
controllers into the system model with `util/assembleNNCtrl.m`. `bef_eval_<system>.py` evaluates the
original controller in the same way.

`config/<system>/*1` is the first controller of a system and `*2` the second. For MC, use
`pyOpera.py`, `retrainMC.py` and `rep_eval_mc.py` with the configurations in `config/MC/`.

Each step writes its outputs to the folder of the controller: the repaired retraining sets to
`repaired/dataset/`, the retrained controllers to `repaired/model/`, their evaluation results to
`repaired/eval/`, and the NSGA-II populations to `chromosome/`.

## Repair implementations

| Class (`src/`) | Generator (`test/`) | Model | Simulation |
| --- | --- | --- | --- |
| `fastOpera` | `fastOpera.py` | `*_RepFast.slx` | fastRob: the model is compiled once per worker |
| `opera` | `opera.py` | `*_Rep.slx` | Breach: the model is rebuilt for every simulation |
| `fastOperaCont` | `fastOperaCont.py` | `*_RepCont.slx` | fastRob |
| `operaCont` | `operaCont.py` | `*_RepCont.slx` | Breach |
| `pyOpera` | `pyOpera.py` | – | MC dynamics in MATLAB |

The fast and the Breach version of a class perform the same repair; the fast one is much quicker. In
`*_Rep` and `*_RepFast` models a switch puts the patch on the control signal; in `*_RepCont` models the
patch is added to the output of the controller, which avoids robustness inconsistencies of AFC's
variable-step solver. Only AFC#6 has a `*_RepCont` model (configuration `config/AFC/rq7/repair6_nn_repcont`).
Robustness is computed by Breach in all of them, so Breach is needed either way.

## Options

**Patch search space.** In a repair configuration,

- `use_custom_search_range 1` narrows the time range of the patch: the earliest start moves from the
  search start `opt_pos` towards the start of the violation T_l by the fraction `t_l_ratio` (alpha), and
  the latest end moves from T_r towards T_l by the fraction `t_r_ratio` (beta), both in [0, 1];
- `use_custom_patch_points 1` changes the number of segments k: set `N` = k + 1 and `V` = k + 3.

The two switches cannot both be on. `start_idx` and `end_idx` repair only a part of the retraining set.

**Larger controllers and longer horizons.** `config/AFC/rq7/` holds the configurations of AFC#3-#6 and
of AFC#2 with the specification over [0, 60] and [0, 120] (`*2_spec_t60`, `*2_spec_t120`), all with the
Explore strategy; the latter write to `repaired_rq7/T_60/` and `repaired_rq7/T_120/` of the AFC#2
folder.

**Patch range.** The patch range is `amp * [-(b-a), b-a]` for `oc_range = [a, b]` since October 2026;
the published AFC and WT results were obtained with the earlier range `amp * [a, b]` (and
`oc_range = [0.13, 1.66]` for AFC), so rerunning with the current configurations gives slightly
different numbers.
