#!/usr/bin/env bash
# Review package for the replay-value ablation. Large checkpoints, traces and fixed inputs stay on the server
# (SERVER_MANIFEST.sha256 lists them with paths and hashes). Read-only against the arms.
set -eu
P=/home/bigking/anaconda3/envs/lcg/bin/python
W=/home/bigking/scfeng/LCG-diagnostic-gaussian-h10-replayvalue-ablation
A=$W/outputs/ablation_20261027
NAME=replayvalue_ablation_review_20261027
PK=$A/package/$NAME
rm -rf $A/package && mkdir -p $PK/source $PK/arms $PK/smoke
cp $A/REPORT.md $PK/
cp -a $A/results $A/plots $A/eval_shared $A/launch $A/logs $A/records $PK/
cp $A/progress.log $A/pipeline_exit_code.txt $A/main_checks.json $PK/
for a in A B C; do
  mkdir -p $PK/arms/$a/checkpoints
  cp $A/arms/$a/{provenance.json,updates.jsonl,indices.jsonl,checkpoints.jsonl,final.json} $PK/arms/$a/
  cp -a $A/arms/$a/eval $PK/arms/$a/
  cp $A/arms/$a/behavior/summary.json $PK/arms/$a/behavior_summary.json
  cp $A/arms/$a/checkpoints/fixed_values_*.pt $PK/arms/$a/checkpoints/
done
# smoke and pre-diagnostic validation (small files only)
cp $A/smoke/unit_tests_focused.txt $A/smoke/check_arms_smoke.json $A/smoke/*.log $PK/smoke/
for d in repro_C_path_replayvalue_smoke repro_A_path_scalar_ema_smoke; do mkdir -p $PK/smoke/$d; cp $A/smoke/$d/{final.json,provenance.json,updates.jsonl,indices.jsonl,checkpoints.jsonl} $PK/smoke/$d/; done
for a in A B C; do mkdir -p $PK/smoke/arms/$a; cp $A/smoke/arms/$a/{final.json,provenance.json,updates.jsonl,indices.jsonl,checkpoints.jsonl} $PK/smoke/arms/$a/; cp $A/smoke/arms/$a.log $PK/smoke/arms/; done
mkdir -p $PK/smoke/resume_C; cp $A/smoke/resume_C/*.log $A/smoke/resume_C/C/{final.json,provenance.json,checkpoints.jsonl} $PK/smoke/resume_C/
mkdir -p $PK/smoke/tools; cp -a $A/smoke/tf32_check $PK/smoke/; cp $A/smoke/tools/*.log $PK/smoke/tools/; cp -a $A/smoke/tools/eval $PK/smoke/tools/; cp $A/smoke/tools/behavior/summary.json $PK/smoke/tools/behavior_summary.json
# source: diagnostic scripts, patched controller, diffs
cp -a $W/diagnostic_ablation $PK/source/; rm -rf $PK/source/diagnostic_ablation/__pycache__
(cd $W && cp --parents src/models/pmpo_beta.py src/models/replay_value.py config/agent/pmpo_beta.yaml $PK/source/)
(cd $W && git diff a13b57c > $PK/source/diff_vs_a13b57c_tracked.patch; git status --short > $PK/source/git_status.txt; git rev-parse HEAD > $PK/source/base_commit.txt; git branch --show-current >> $PK/source/base_commit.txt)
cp $A/records/diagnostic_diff_vs_carried_pmpo_beta.patch $PK/source/
(cd $W && sha256sum src/models/pmpo_beta.py src/models/replay_value.py config/agent/pmpo_beta.yaml diagnostic_ablation/*.py diagnostic_ablation/*.sh) > $PK/source/source_sha256_at_packaging.txt
# large files kept on the server
(find $A/arms -name "update_*.pt"; find $A/arms -name "*.npz"; echo $A/fixed_inputs.pt; find $A/smoke -name "update_*.pt") | sort | xargs sha256sum > $PK/SERVER_MANIFEST.sha256
find $PK -name __pycache__ -prune -exec rm -rf {} +
(cd $PK && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
cd $A/package && $P -c "import shutil; shutil.make_archive('$NAME', 'zip', '.', '$NAME')"
du -sh $PK $NAME.zip; sha256sum $NAME.zip; find $PK -type f | wc -l
