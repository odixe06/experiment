#!/bin/bash
set -euo pipefail
export PATH=/home/odixe/Tools/miniconda3/envs/nckh/bin:$PATH
cd /home/odixe/project/fl_veremi_bert
python scripts/audit_t3_smoke.py runs/t3/smoke-resume 3
python scripts/run_t3_batch.py launch
python /home/odixe/project/fl_veremi_bert/scripts/watch_kaggle_batch.py --batch /home/odixe/project/fl_veremi_bert/runs/t3/batch.json --out /home/odixe/project/fl_veremi_bert/runs/t3/watch --interval 1200 --stop-on-error
echo T3_BERT_SESSIONS_TERMINAL
