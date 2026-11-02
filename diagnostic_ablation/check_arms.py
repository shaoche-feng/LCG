"""Pre-diagnostic checks over arm output folders A, B, C (smoke or main) and optional resume pair.
usage: python check_arms.py ROOT [RESUMED_C UNINTERRUPTED_C]   (ROOT holds A/, B/, C/)"""
import json
import sys
from pathlib import Path

import torch

root = Path(sys.argv[1])
res = {}
rows = {a: [json.loads(l) for l in open(root / a / "updates.jsonl")] for a in "ABC"}
idx = {a: [json.loads(l) for l in open(root / a / "indices.jsonl")] for a in "ABC"}
prov = {a: json.loads((root / a / "provenance.json").read_text()) for a in "ABC"}
cks = {a: [json.loads(l) for l in open(root / a / "checkpoints.jsonl")] for a in "ABC"}

# 1. identical starting learning state
rh = {a: prov[a]["restored_hashes"] for a in "ABC"}
res["identical_restored_state"] = all(rh[a] == rh["A"] for a in "BC")
res["restored_equals_snapshot"] = all(rh[a]["actor_critic"] == rh[a]["snapshot_actor_critic"] and rh[a]["pmpo_optimizers"] == rh[a]["snapshot_pmpo_optimizers"] for a in "ABC")
c0 = {a: next(c for c in cks[a] if c["diagnostic_update"] == 0) for a in "ABC"}
res["identical_update0_checkpoint"] = all(c0[a]["actor_critic_hash"] == c0["A"]["actor_critic_hash"] and c0[a]["pmpo_optimizers_hash"] == c0["A"]["pmpo_optimizers_hash"] for a in "BC")
res["restored_hash"] = rh["A"]
# 2. sampling layout
res["sampler"] = {a: (prov[a]["sampler"], prov[a]["sampler_batch_size"], prov[a]["sampler_seq_length"], prov[a]["grouped_sampling"]) for a in "ABC"}
res["A_32_independent_starts"] = all(len(r["batches"][0]) == 32 and all(s - st == 4 for _, st, s in r["batches"][0]) for r in idx["A"])
res["BC_8_segments_x_4_consecutive"] = all(len(r["batches"][0]) == 8 and len(r["starts"]) == 32 and all(s - st == 7 for _, st, s in r["batches"][0])
                                           and all(r["starts"][4 * i + k][1] == r["starts"][4 * i][1] + k for i in range(8) for k in range(4)) for a in "BC" for r in idx[a])
res["B_C_identical_first_batches_all_updates"] = all(b["batches"][0] == c["batches"][0] for b, c in zip(idx["B"], idx["C"])) and len(idx["B"]) == len(idx["C"])
res["B_C_identical_all_batches_fraction"] = sum(b["batches"] == c["batches"] for b, c in zip(idx["B"], idx["C"])) / len(idx["B"])
res["A_B_first_batch_identical_fraction"] = sum(a["starts"] == b["starts"] for a, b in zip(idx["A"], idx["B"])) / len(idx["A"])
checked = {a: [r for r in idx[a] if r.get("start_frames_checked")] for a in "ABC"}
res["start_frames_checked_updates"] = {a: len(checked[a]) for a in "ABC"}
res["start_frames_all_match"] = all(r["start_frames_match"] for a in "ABC" for r in checked[a])
res["batches_per_update"] = {a: sorted(set(r["n_batches"] for r in idx[a])) for a in "ABC"}
# 3. loss composition
res["A_has_no_replay_metrics"] = all("loss_repval" not in r for r in rows["A"])
res["B_no_replay_in_objective"] = all(r["replay_loss_in_objective"] == 0 and r["loss_repval_weighted"] == 0 and r["loss_value"] == r["loss_value_imag"] for r in rows["B"])
res["C_replay_in_objective"] = all(r["replay_loss_in_objective"] == 1 and abs(r["loss_value"] - r["loss_value_imag"] - r["loss_repval_weighted"]) <= 1e-5 * max(1, r["loss_value"])
                                   and abs(r["loss_repval_weighted"] - 0.3 * r["loss_repval"]) <= 1e-6 * max(1, r["loss_repval"]) for r in rows["C"])
res["update1_B_vs_C"] = {k: (rows["B"][0][k], rows["C"][0][k]) for k in ("loss_actor", "actor_grad_norm", "loss_value_imag", "value_grad_norm", "return_mean", "replay_target_mean")}
res["update1_B_C_actor_identical"] = rows["B"][0]["loss_actor"] == rows["C"][0]["loss_actor"] and rows["B"][0]["actor_grad_norm"] == rows["C"][0]["actor_grad_norm"]
res["update1_B_C_imagination_identical"] = rows["B"][0]["return_mean"] == rows["C"][0]["return_mean"] and rows["B"][0]["loss_value_imag"] == rows["C"][0]["loss_value_imag"]
res["replay_decomposition_max_error"] = max(r["replay_target_decomposition_error"] for a in "BC" for r in rows[a])
res["rollout_length"] = sorted(set(r["rollout_length"] for a in "ABC" for r in rows[a]))
# 4. frozen models, prior refreshes, counters
res["world_reward_unchanged_all_checkpoints"] = all(c["world_and_reward_models_unchanged"] for a in "ABC" for c in cks[a])
res["prior_refresh_relative_updates"] = {a: [r["diagnostic_update"] for r in rows[a] if r["prior_refresh_event"]] for a in "ABC"}
res["counters"] = {a: (rows[a][0]["pmpo_updates_counter"], rows[a][-1]["pmpo_updates_counter"], len(rows[a])) for a in "ABC"}
# 5. fixed-input identity at update 0 (rv190 traces were recorded with this same checkpoint)
fx = torch.load(prov["A"]["fixed_inputs"], weights_only=False)
fv0 = torch.load(root / "A/checkpoints/fixed_values_00000.pt", weights_only=False)
m = torch.tensor([s == "rv190" for s in fx["source"]])
tr, tp = torch.tensor(fx["v_real_trace"])[m], torch.tensor(fx["v_rep_trace"])[m]
res["update0_fixed_values_vs_trace_max_abs"] = dict(real_history=float((fv0["real_history"]["v_online"][m] - tr).abs().max()),
                                                    repeated_frame=float((fv0["repeated_frame"]["v_online"][m] - tp).abs().max()))
if len(sys.argv) > 3:
    a, b = (json.loads((Path(p) / "final.json").read_text()) for p in sys.argv[2:4])
    ra = [json.loads(l) for l in open(Path(sys.argv[2]) / "updates.jsonl")]
    rb = [json.loads(l) for l in open(Path(sys.argv[3]) / "updates.jsonl")]
    res["resume_identical_final_state"] = a["actor_critic_hash"] == b["actor_critic_hash"] and a["pmpo_optimizers_hash"] == b["pmpo_optimizers_hash"]
    res["resume_identical_metrics"] = ra == rb
    res["resumed_from"] = json.loads((Path(sys.argv[2]) / "provenance.json").read_text()).get("resumed_from")
keys_bool = [k for k, v in res.items() if isinstance(v, bool)]
res["all_boolean_checks_pass"] = all(res[k] for k in keys_bool)
print(json.dumps(res, indent=1))
