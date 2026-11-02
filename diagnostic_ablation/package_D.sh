#!/usr/bin/env bash
# Review package for baseline follow-up D (read-only against the run; large files stay on the server).
set -eu
P=/home/bigking/anaconda3/envs/lcg/bin/python
W=/home/bigking/scfeng/LCG-diagnostic-gaussian-h10-replayvalue-ablation
B=$W/outputs/ablation_20261027/baseline_D
NAME=replayvalue_ablation_baselineD_review_20261027
PK=$B/package/$NAME
rm -rf $B/package && mkdir -p $PK/D/checkpoints $PK/smoke $PK/source
cp $B/REPORT_D.md $B/progress.log $B/pipeline_exit_code.txt $B/update0_identity.txt $PK/
cp -a $B/results $B/plots $B/launch $B/logs $B/records $B/eval_update0 $PK/
cp $B/D/{provenance.json,updates.jsonl,indices.jsonl,checkpoints.jsonl,final.json} $PK/D/
cp -a $B/D/eval $PK/D/
cp $B/D/behavior/summary.json $PK/D/behavior_summary.json
cp $B/D/checkpoints/fixed_values_*.pt $PK/D/checkpoints/
cp $B/smoke/*.log $B/smoke/check_D_smoke.json $PK/smoke/
mkdir -p $PK/smoke/D; cp $B/smoke/D/{provenance.json,updates.jsonl,indices.jsonl,checkpoints.jsonl,final.json} $PK/smoke/D/
cp -a $B/smoke/stochastic $PK/smoke/
cp -a $W/diagnostic_ablation $PK/source/; rm -rf $PK/source/diagnostic_ablation/__pycache__
(cd $W && cp --parents src/models/pmpo_beta.py src/models/replay_value.py config/agent/pmpo_beta.yaml $PK/source/)
(cd $W && git diff a13b57c > $PK/source/diff_vs_a13b57c_tracked.patch; git status --short > $PK/source/git_status.txt; git rev-parse HEAD > $PK/source/base_commit.txt; git branch --show-current >> $PK/source/base_commit.txt)
(cd $W && sha256sum src/models/pmpo_beta.py src/models/replay_value.py config/agent/pmpo_beta.yaml diagnostic_ablation/*.py diagnostic_ablation/*.sh) > $PK/source/source_sha256_at_packaging.txt
(find $B/D -name "update_*.pt"; find $B/D -name "*.npz"; find $B/smoke -name "update_*.pt"; echo $B/fixed_inputs_D.pt; echo $B/update0_from_snapshot.pt) | sort | xargs sha256sum > $PK/SERVER_MANIFEST.sha256
find $PK -name __pycache__ -prune -exec rm -rf {} +
(cd $PK && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
cd $B/package && $P -c "import shutil; shutil.make_archive('$NAME', 'zip', '.', '$NAME')"
du -sh $PK $NAME.zip; sha256sum $NAME.zip; find $PK -type f | wc -l
