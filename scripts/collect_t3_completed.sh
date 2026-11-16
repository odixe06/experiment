#!/usr/bin/env bash
set -euo pipefail
cd /home/odixe/project/fl_veremi_bert
export PATH=/home/odixe/Tools/miniconda3/envs/nckh/bin:$PATH
helper=.agents/skills/kaggle-training-notebook/scripts
for spec in minhtran0601/flb-t3-s41-p3/1 annhtnt/flb-t3-s42/1 trietbackup/flb-t3-s43/1; do
  owner=${spec%%/*}; rest=${spec#*/}; slug=${rest%/*}
  python "$helper/kaggle_as.py" "$owner" -- python "$helper/pull_kernel_output.py" "$spec" -p "runs/t3/pulls/$slug" -j 16
  seed=${slug#flb-t3-s}; seed=${seed%%-*}
  python scripts/audit_run.py "runs/t3/pulls/$slug/flb-t3-s$seed" --seed "$seed" --support runs/test_support.npy --out "runs/t3/audit-$slug.json"
  python scripts/audit_t3_state.py "runs/t3/pulls/$slug" --seed "$seed"
done
python "$helper/kaggle_as.py" minhtran0601 -- python "$helper/pull_kernel_output.py" minhtran0601/mh-k-b21-t341-r50/1 -p /home/odixe/project/mobihet/artifacts/t3-s41/mh-k-b21-t341-r50 -j 16
printf 'T3_COMPLETED_OUTPUTS_COLLECTED\n'
