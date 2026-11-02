"""Checks for baseline follow-up arm D (smoke or main). usage: python check_D.py D_DIR ARM_A_DIR [SMOKE_D_DIR]
 - restored controller/optimizer state equals the baseline snapshot; world/reward models unchanged at every checkpoint
 - layout: BatchSampler, 32 independent 4-frame prompts per update; start frames match recorded indices where checked
 - same prompt-seeding convention as arm A (seed, mode); replay loss absent
 - update-0 fixed-input values equal the baseline's own epoch-190 behaviour traces (base190 states) exactly
 - the first 600 fixed inputs are identical to arm A's fixed inputs (shared states for the A/D comparison)
 - optional: main D's first updates equal the separate smoke D's"""
import json
import sys
from pathlib import Path

import torch

D, A = Path(sys.argv[1]), Path(sys.argv[2])
res = {}
prov = json.loads((D / "provenance.json").read_text())
provA = json.loads((A / "provenance.json").read_text())
rows = [json.loads(l) for l in open(D / "updates.jsonl")]
idx = [json.loads(l) for l in open(D / "indices.jsonl")]
cks = [json.loads(l) for l in open(D / "checkpoints.jsonl")]
rh = prov["restored_hashes"]
res["restored_equals_snapshot"] = rh["actor_critic"] == rh["snapshot_actor_critic"] and rh["pmpo_optimizers"] == rh["snapshot_pmpo_optimizers"]
res["world_reward_unchanged_all_checkpoints"] = all(c["world_and_reward_models_unchanged"] for c in cks)
res["sampler"] = [prov["sampler"], prov["sampler_batch_size"], prov["sampler_seq_length"], prov["grouped_sampling"]]
res["layout_32_independent_4_frame"] = prov["sampler"] == "BatchSampler" and not prov["grouped_sampling"] and all(
    len(r["batches"][0]) == 32 and all(s - st == 4 for _, st, s in r["batches"][0]) for r in idx)
res["same_prompt_convention_as_A"] = (prov["prompt_stream"], prov["prompt_seed"]) == (provA["prompt_stream"], provA["prompt_seed"]) == ("dedicated", 190000000)
res["replay_loss_absent"] = prov["replay_value_coef"] == 0.0 and all("loss_repval" not in r for r in rows)
res["controller_settings"] = {k: prov[k] for k in ("horizon", "num_envs", "slow_critic_coef", "slow_critic_rate", "max_grad_norm", "prior_refresh_interval")}
res["controller_settings_equal_A"] = res["controller_settings"] == {k: provA[k] for k in res["controller_settings"]}
checked = [r for r in idx if r.get("start_frames_checked")]
res["start_frames_checked_updates"] = len(checked)
res["start_frames_all_match"] = all(r["start_frames_match"] for r in checked)
res["rollout_length"] = sorted(set(r["rollout_length"] for r in rows))
res["counters"] = [rows[0]["pmpo_updates_counter"], rows[-1]["pmpo_updates_counter"], len(rows)]
res["prior_refresh_diagnostic_updates"] = [r["diagnostic_update"] for r in rows if r["prior_refresh_event"]]
fx = torch.load(prov["fixed_inputs"], weights_only=False)
fxA = torch.load(provA["fixed_inputs"], weights_only=False)
n = len(fxA["source"])
res["first_fixed_inputs_identical_to_A"] = all(torch.equal(fx[k][:n], fxA[k]) for k in ("real_history", "repeated_frame")) and fx["source"][:n] == fxA["source"]
fv0 = torch.load(D / "checkpoints/fixed_values_00000.pt", weights_only=False)
m = torch.tensor([s == "base190" for s in fx["source"]])
res["update0_base190_values_vs_trace_max_abs"] = dict(
    real_history=float((fv0["real_history"]["v_online"][m] - torch.tensor(fx["v_real_trace"])[m]).abs().max()),
    repeated_frame=float((fv0["repeated_frame"]["v_online"][m] - torch.tensor(fx["v_rep_trace"])[m]).abs().max()))
res["update0_values_reproduce_traces"] = max(res["update0_base190_values_vs_trace_max_abs"].values()) == 0.0
if len(sys.argv) > 3:
    S = Path(sys.argv[3])
    s = [json.loads(l) for l in open(S / "updates.jsonl")]
    si = [json.loads(l)["batches"] for l in open(S / "indices.jsonl")]
    res["smoke_updates_compared"] = len(s)
    res["main_equals_smoke"] = s == rows[:len(s)] and si == [r["batches"] for r in idx[:len(si)]]
res["all_boolean_checks_pass"] = all(v for v in res.values() if isinstance(v, bool))
print(json.dumps(res, indent=1))
sys.exit(0 if res["all_boolean_checks_pass"] else 1)
