#!/bin/bash
#PJM -L rscgrp=a-batch
#PJM -L vnode-core=20
#PJM -L elapse=72:00:00
#PJM -j
#PJM -o opera_job_%j.log
#PJM -m e

# Runs the job scripts in test/scripts with make on Genkai (PJM).
#
# First write the job scripts on the login node, as on a local machine, e.g.
#   cd test && python3 fastOpera.py config/AFC/repair2 && cd ..
# then submit from the root of the repository:
#   pjsub submit_genkai.pjm.sh
#
# Options:
#   pjsub -L vnode-core=36 submit_genkai.pjm.sh       number of cores (default 20 above)
#   JOBS=3 pjsub -X submit_genkai.pjm.sh               job scripts run in parallel by make (default 1)
#   CORE_NUM=20 pjsub -X submit_genkai.pjm.sh          first check that parpool starts with 20 workers
#   (-X passes the environment variables to the job)
#
# Each job script starts a parpool of core_num workers (set in the configuration), so
# JOBS x core_num should not exceed vnode-core. Evaluation scripts already use all the cores:
# run them with JOBS=1.
#
# To be e-mailed when the job ends, add a line "#PJM --mail-list <address>" above.

set -euo pipefail

JOBS="${JOBS:-1}"
CORE_NUM="${CORE_NUM:-}"

# MATLAB: use the module system, or put a local installation on the PATH
module load matlab
# export PATH="$HOME/MATLAB/R2021a/bin:$PATH"

cd "${PJM_O_WORKDIR:-.}"

echo "repository: $(pwd)"
echo "job scripts: $(ls test/scripts | grep -cv '^\.gitkeep$'), make -j $JOBS, cores: $(nproc)"

if [ -n "$CORE_NUM" ]; then
	matlab -batch "addpath('util'); verify_parpool($CORE_NUM);"
fi

make -k -j "$JOBS" all
