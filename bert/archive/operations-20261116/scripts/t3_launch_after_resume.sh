#!/bin/bash
set -euo pipefail
export PATH=/home/odixe/Tools/miniconda3/envs/nckh/bin:$PATH
export V4_ROOT=/home/odixe/project/fl_veremi_bert/knowledge
cd /home/odixe/project/fl_veremi_bert
python scripts/audit_t3_smoke.py runs/t3/smoke 2
python -c "from pathlib import Path; import shutil; d=Path('runs/t3/smoke-resume'); assert not d.exists(); shutil.copytree('runs/t3/smoke',d)"
/home/odixe/Tools/miniconda3/envs/nckh/bin/python /home/odixe/project/fl_veremi_bert/scripts/run_local_guarded.py /home/odixe/Tools/miniconda3/envs/nckh/bin/python scripts/run_t3.py --method mobihet_compute --out runs/t3/smoke-resume --subset 100 --rounds 50 --until 3 --save-every 1 --compile --cache cache --set arm=none group_rows=1024 synth=copula synth_rows=64 q_codec=mean_fp16 kd_weight=fixed 'diagnostic_ids=[506,0,1571,929,3303,2048,4752,4269,504,1,1587,935,3304,2043,4754,4266,512,29,1566,904,3306,2052,4757,4267,1547,885,3318,2357,4748,4265,1548,886,4747,4279]' 'diagnostic_rounds=[1,2,5,10,15,20]' q_collect=online q_agg=count kd_support=off synth_target=soft anchor=off kd_bins=1 no_kd=false study_mode=class teacher_rows=1024 proxy_steps=200 real_kd=false real_kd_gate=false next_mode=stream r_kind=live synth_step_floor=16 w_boot=compact diagnostics=false teacher_diagnostics=false external_diagnostics=false cop_kmeans=16 cop_kmeans_w0=0.5 cop_kmeans_q=int8 cop_kmeans_lite=6 q_bits=8 km_bytes=true seed=41 'synth_tier_min=[4096,16384]' 'synth_tier_rows=[[64,64],[64,32],[48,16]]' synth_late_from=25
python scripts/audit_t3_smoke.py runs/t3/smoke-resume 3
python scripts/run_t3_batch.py launch
python /home/odixe/project/fl_veremi_bert/scripts/watch_kaggle_batch.py --batch /home/odixe/project/fl_veremi_bert/runs/t3/batch.json --out /home/odixe/project/fl_veremi_bert/runs/t3/watch --interval 1200 --stop-on-error
echo T3_BERT_SESSIONS_TERMINAL
