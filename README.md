# OPERA

OPERA repairs the DNN controllers of AI-enabled cyber-physical systems. For each unsafe execution, it
searches with NSGA-II for a patch of the control signal that makes the execution satisfy its STL
specification, then retrains the controller on the repaired executions. The plant and the controller
are treated as black boxes.

This repository contains the code and the benchmarks (system models, controllers and datasets) of the
paper.

- [Benchmarks](docs/benchmarks.md): systems, specifications, controllers and datasets
- [Performance metrics](docs/metrics.md): the system-specific metrics of RQ4 and their normalization
- [Experiments](docs/experiments.md): settings, and how to run the experiments

## Requirements

- MATLAB with Simulink, Deep Learning Toolbox, Model Predictive Control Toolbox, Signal Processing
  Toolbox and DSP System Toolbox. The servers run R2021a; networks saved by a newer MATLAB do not load
  there.
- [Breach](https://github.com/decyphir/breach): run `mex -setup` and then `InstallBreach` in MATLAB.
  The configurations expect it in `~/git/breach`.
- Python 3, to generate the job scripts.
- [Git LFS](https://git-lfs.com): all `.mat` files are stored with LFS, so run `git lfs pull` after
  cloning.

## Quick start

Every step works the same way: a Python script in `test/` reads a configuration from `test/config/`
and writes one job script per run to `test/scripts/`; `make` in the root runs them all, writes the
outputs to the root and the logs to `output/`. Set the path under `addpath` in the configurations to
your copy of the repository first (they use `/home/deyun/git/OPERA_dev`).

For example, to repair AFC#2 with the Explore strategy:

```
cd test
python3 fastOpera.py config/AFC/repair2    # set st_id to 5 in the configuration
cd .. && make
```

The full pipeline (datasets, repair, retraining, evaluation) is in
[docs/experiments.md](docs/experiments.md#running-a-pipeline).

## Layout

```
.
├── benchmark/            systems, controllers and data, one folder per controller
├── src/                  OPERA (fastOpera.m for Simulink models, pyOpera.m for MC)
│   └── eval/             evaluation of the original and retrained controllers
├── test/
│   ├── *.py              job script generators
│   └── config/<system>/  their configurations
├── util/                 training, simulation (util/sim: fastRob) and helpers
├── NSGA-II/
├── docs/
└── Makefile              runs test/scripts/*
```

One folder per controller, for example AFC#2:

```
benchmark/AFC/AFC_mu_015_spec_1_FFNN_trainlm_30_30_30_Nor_Jan_6/
├── dataset/       retraining set (..._Tr.mat) and test set (..._Te.mat)
├── model/         system model with the original controller
├── chromosome/    written by the repair: NSGA-II populations, one file per repaired execution
└── repaired/      written by the later steps
    ├── dataset/   repaired retraining sets, one per selection strategy
    ├── model/     retrained controllers
    └── eval/      their results on the test set
```

## Notes

- `fastOpera` simulates through fastRob, which compiles the model once per worker; `opera` does the
  same repair but is much slower. Configurations ending in `_repcont` use the continuous-solver
  versions.
- fastRob leaves model copies (`fr_*`) in `~/git/breach/Ext/ModelsData`; delete them after local runs.
- AFC uses a variable-step solver, so its robustness can differ slightly between machines.
- On a cluster with the PJM scheduler, write the job scripts on the login node and run them with
  `pjsub submit_genkai.pjm.sh` (cores, parallel jobs and a parpool check are set when submitting).
