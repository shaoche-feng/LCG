#!/usr/bin/env bash
# Review package for the controller x model/data swap (read-only against the source runs; large files stay on the server).
set -eu
P=/home/bigking/anaconda3/envs/lcg/bin/python
W=/home/bigking/scfeng/LCG-diagnostic-gaussian-h10-ema-component-swap
O=$W/outputs/swap_20261027
NAME=component_swap_review_20261027
PK=$O/package/$NAME
rm -rf $O/package && mkdir -p $PK/arms $PK/smoke $PK/source
cp $O/REPORT_SWAP.md $O/progress.log $O/prep_progress.log $O/pipeline_exit_code.txt $O/main_checks.json $O/update0_eval_check.json $PK/
cp -a $O/results $O/plots $O/launch $O/logs $O/records $O/eval_prelearning $PK/
mkdir -p $PK/behavior_update0; for c in ctrl0 ctrl1; do mkdir -p $PK/behavior_update0/$c; cp $O/behavior_update0/$c/summary.json $PK/behavior_update0/$c/; done
for a in S00 S01 S10 S11; do
  mkdir -p $PK/arms/$a/checkpoints
  cp $O/arms/$a/{provenance.json,updates.jsonl,indices.jsonl,checkpoints.jsonl,final.json} $PK/arms/$a/
  cp -a $O/arms/$a/eval $PK/arms/$a/
  cp $O/arms/$a/behavior/summary.json $PK/arms/$a/behavior_summary.json
  cp $O/arms/$a/checkpoints/fixed_values_*.pt $PK/arms/$a/checkpoints/
done
cp $O/smoke/*.log $O/smoke/*.json $O/smoke/*.txt $PK/smoke/
for d in repro_trainer_path repro_trainer_path_explicit_bundle; do mkdir -p $PK/smoke/$d; cp $O/smoke/$d/*.json* $PK/smoke/$d/ 2>/dev/null || true; done
for a in S00 S01 S10 S11; do mkdir -p $PK/smoke/arms/$a; cp $O/smoke/arms/$a/{provenance.json,updates.jsonl,indices.jsonl,checkpoints.jsonl,final.json} $PK/smoke/arms/$a/; cp $O/smoke/arms/$a.log $PK/smoke/arms/; done
mkdir -p $PK/smoke/resume_S01/S01; cp $O/smoke/resume_S01/*.log $PK/smoke/resume_S01/; cp $O/smoke/resume_S01/S01/{updates.jsonl,final.json} $PK/smoke/resume_S01/S01/
cp -a $W/diagnostic_swap $PK/source/; rm -rf $PK/source/diagnostic_swap/__pycache__
(cd $W && cp --parents src/models/pmpo_beta.py config/agent/pmpo_beta.yaml tests/models/test_ema_critic.py $PK/source/)
(cd $W && git diff a13b57c > $PK/source/diff_vs_a13b57c_tracked.patch; git status --short > $PK/source/git_status.txt; git rev-parse HEAD > $PK/source/base_commit.txt; git branch --show-current >> $PK/source/base_commit.txt)
(cd $W && sha256sum src/models/pmpo_beta.py config/agent/pmpo_beta.yaml tests/models/test_ema_critic.py diagnostic_swap/*.py diagnostic_swap/*.sh) > $PK/source/source_sha256_at_packaging.txt
(find $O/arms $O/smoke -name "update_*.pt"; find $O/arms $O/behavior_update0 -name "*.npz"; echo $O/fixed_inputs_swap.pt; echo $O/update0_ctrl0.pt; echo $O/update0_ctrl1.pt) | sort | xargs sha256sum > $PK/SERVER_MANIFEST.sha256
find $PK -name __pycache__ -prune -exec rm -rf {} +
(cd $PK && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
cd $O/package && $P -c "import shutil; shutil.make_archive('$NAME', 'zip', '.', '$NAME')"
du -sh $PK $NAME.zip; sha256sum $NAME.zip; find $PK -type f | wc -l
