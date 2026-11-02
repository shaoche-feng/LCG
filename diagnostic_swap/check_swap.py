"""Wiring checks for the four component-swap arms (smoke or main). usage: python check_swap.py ROOT (ROOT/S00 ... S11)
Arm Sxy = controller bundle of seed x, model/data bundle of seed y."""
import json
import sys
from pathlib import Path

import torch

root = Path(sys.argv[1])
ARMS = ["S00", "S01", "S10", "S11"]
prov = {a: json.loads((root / a / "provenance.json").read_text()) for a in ARMS}
rows = {a: [json.loads(l) for l in open(root / a / "updates.jsonl")] for a in ARMS}
idx = {a: [json.loads(l) for l in open(root / a / "indices.jsonl")] for a in ARMS}
cks = {a: [json.loads(l) for l in open(root / a / "checkpoints.jsonl")] for a in ARMS}
res = {}
cb = {a: prov[a]["controller_bundle_sha256"] for a in ARMS}
mb = {a: prov[a]["model_data_bundle_sha256"] for a in ARMS}
rh = {a: prov[a]["restored_hashes"] for a in ARMS}
res["controller_bundle_identical_S00_S01"] = cb["S00"] == cb["S01"]
res["controller_bundle_identical_S10_S11"] = cb["S10"] == cb["S11"]
res["controller_bundles_differ_between_seeds"] = cb["S00"] != cb["S10"]
res["model_data_bundle_identical_S00_S10"] = mb["S00"] == mb["S10"]
res["model_data_bundle_identical_S01_S11"] = mb["S01"] == mb["S11"]
res["model_data_bundles_differ_between_seeds"] = mb["S00"] != mb["S01"]
# controller pieces (actor, online/EMA critic, prior/initial actors, counter, private RNG; optimizers, schedulers) from the controller snapshot
res["controller_from_controller_snapshot"] = all(rh[a]["actor_critic"] == rh[a]["snapshot_actor_critic"] and rh[a]["pmpo_optimizers"] == rh[a]["snapshot_pmpo_optimizers"] for a in ARMS)
res["swapped_controller_differs_from_model_snapshot_controller"] = all(rh[a]["actor_critic"] != rh[a]["model_snapshot_actor_critic"] for a in ("S01", "S10"))
res["models_from_model_snapshot"] = all(rh[a]["world_and_reward_models"] == rh[a]["model_snapshot_world_and_reward_models"] for a in ARMS)
res["swapped_models_differ_from_controller_snapshot_models"] = all(rh[a]["world_and_reward_models"] != rh[a]["controller_snapshot_world_and_reward_models"] for a in ("S01", "S10"))
res["sources"] = {a: dict(controller=Path(prov[a]["snapshot"]).parents[1].name[-40:], model=Path(prov[a]["model_snapshot"]).parents[1].name[-40:],
                          controller_epoch=prov[a]["snapshot_epoch"], model_epoch=prov[a]["model_snapshot_epoch"]) for a in ARMS}
res["all_sources_epoch_90"] = all(prov[a]["snapshot_epoch"] == 90 and prov[a]["model_snapshot_epoch"] == 90 for a in ARMS)
res["dataset_state"] = {a: (prov[a]["dataset_num_episodes"], prov[a]["dataset_num_steps"]) for a in ARMS}
res["datasets_at_50k"] = all(prov[a]["dataset_num_steps"] == 50000 and prov[a]["dataset_num_episodes"] == 100 for a in ARMS)
# compatibility: identical interfaces and settings
keys = ("actor_distribution", "action_low", "action_high", "horizon", "num_envs", "slow_critic_coef", "slow_critic_rate", "max_grad_norm",
        "prior_refresh_interval", "sampler", "sampler_batch_size", "sampler_seq_length", "sample_weights", "grouped_sampling", "prompt_stream", "prompt_seed")
res["identical_settings_all_arms"] = all({k: prov[a][k] for k in keys} == {k: prov["S00"][k] for k in keys} for a in ARMS)
res["settings"] = {k: prov["S00"][k] for k in keys}
c0 = {a: next(c for c in cks[a] if c["diagnostic_update"] == 0) for a in ARMS}
res["update0_checkpoint_identical_within_controller"] = c0["S00"]["actor_critic_hash"] == c0["S01"]["actor_critic_hash"] and c0["S10"]["actor_critic_hash"] == c0["S11"]["actor_critic_hash"] \
    and c0["S00"]["pmpo_optimizers_hash"] == c0["S01"]["pmpo_optimizers_hash"] and c0["S10"]["pmpo_optimizers_hash"] == c0["S11"]["pmpo_optimizers_hash"]
res["world_reward_unchanged_all_checkpoints"] = all(c["world_and_reward_models_unchanged"] for a in ARMS for c in cks[a])
# sampling
res["layout_32_independent_4_frame"] = all(len(r["batches"][0]) == 32 and all(s - st == 4 for _, st, s in r["batches"][0]) for a in ARMS for r in idx[a])
for x, y in (("S00", "S10"), ("S01", "S11")):
    n = min(len(idx[x]), len(idx[y]))
    res[f"prompt_first_batch_identical_{x}_{y}_updates"] = sum(idx[x][i]["batches"][0] == idx[y][i]["batches"][0] for i in range(n))
    res[f"prompt_all_batches_identical_{x}_{y}_updates"] = sum(idx[x][i]["batches"] == idx[y][i]["batches"] for i in range(n))
    res[f"prompt_first_batch_identical_{x}_{y}_all"] = res[f"prompt_first_batch_identical_{x}_{y}_updates"] == n
checked = {a: [r for r in idx[a] if r.get("start_frames_checked")] for a in ARMS}
res["start_frames_checked_updates"] = {a: len(checked[a]) for a in ARMS}
res["start_frames_all_match"] = all(r["start_frames_match"] for a in ARMS for r in checked[a])
res["episodes_sampled_within_snapshot_boundary"] = all(e < prov[a]["dataset_num_episodes"] and stop <= 500 for a in ARMS for r in idx[a] for b in r["batches"] for e, _, stop in b)
res["counters"] = {a: (rows[a][0]["pmpo_updates_counter"], rows[a][-1]["pmpo_updates_counter"], len(rows[a])) for a in ARMS}
res["prior_refresh_diagnostic_updates"] = {a: [r["diagnostic_update"] for r in rows[a] if r["prior_refresh_event"]] for a in ARMS}
res["update1_same_controller_actor_grad_differs_by_bundle"] = {c: (rows[c + "0"][0]["actor_grad_norm"], rows[c + "1"][0]["actor_grad_norm"]) for c in ("S0", "S1")}
# fixed inputs at update 0: identical actions/values within a controller across bundles; equal to the controller's own traces
fx = torch.load(prov["S00"]["fixed_inputs"], weights_only=False)
fv = {a: torch.load(root / a / "checkpoints/fixed_values_00000.pt", weights_only=False) for a in ARMS}
res["update0_fixed_outputs_identical_within_controller"] = all(
    torch.equal(fv[x][k][q], fv[y][k][q]) for x, y in (("S00", "S01"), ("S10", "S11")) for k in ("real_history", "repeated_frame") for q in ("v_online", "v_ema", "action"))
dev = {}
for ctrl, arm in (("ctrl0_ep90", "S00"), ("ctrl1_ep90", "S11")):
    m = torch.tensor([s == ctrl for s in fx["source"]])
    dev[ctrl] = max(float((fv[arm]["real_history"]["v_online"][m] - torch.tensor(fx["v_real_trace"])[m]).abs().max()),
                    float((fv[arm]["repeated_frame"]["v_online"][m] - torch.tensor(fx["v_rep_trace"])[m]).abs().max()))
res["update0_values_vs_own_traces_max_abs"] = dev
res["update0_values_reproduce_own_traces"] = max(dev.values()) == 0.0
res["all_boolean_checks_pass"] = all(v for v in res.values() if isinstance(v, bool))
print(json.dumps(res, indent=1))
sys.exit(0 if res["all_boolean_checks_pass"] else 1)
