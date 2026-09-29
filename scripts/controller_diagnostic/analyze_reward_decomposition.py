import json
import statistics as st
import pathlib
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]  # scripts/controller_diagnostic/<file>.py -> repo root

with open(f"{REPO_ROOT}/outputs/diagnose_reward_decomposition/instrumented_results.json") as f:
    data = json.load(f)

REPORTED = {
    "stage3_seed3013": 634.043140055196,
    "stage3_seed3030": 224.2775240228603,
    "stage3b_seed3013": 641.8913244698215,
    "stage3b_seed3011": 177.87072312190554,
}

for key, d in data.items():
    steps = d["steps"]
    n = len(steps)
    print(f"\n{'='*90}\n{key}  (n_outer_steps={n}, total_return={d['total_return']:.3f}, reported={REPORTED[key]:.3f}, "
          f"match={abs(d['total_return']-REPORTED[key])<1e-2})\n{'='*90}")

    heights = [s["inner"][-1]["torso_height"] for s in steps]
    uprights = [s["inner"][-1]["torso_upright"] for s in steps]
    vels = [s["inner"][-1]["horizontal_velocity"] for s in steps]
    xs = [s["inner"][-1]["torso_x"] for s in steps]
    standings = [s["inner"][-1]["standing"] for s in steps]
    moves = [s["inner"][-1]["move_reward"] for s in steps]
    stand_rewards = [s["inner"][-1]["stand_reward"] for s in steps]
    rewards = [s["step_reward"] for s in steps]

    print(f"torso_height: mean={st.mean(heights):.3f} min={min(heights):.3f} max={max(heights):.3f}  "
          f"frac_below_1.2={sum(1 for h in heights if h<1.2)/n:.2f}  frac_below_0.6={sum(1 for h in heights if h<0.6)/n:.2f}")
    print(f"torso_upright: mean={st.mean(uprights):.3f} min={min(uprights):.3f}")
    print(f"horizontal_velocity: mean={st.mean(vels):.3f} min={min(vels):.3f} max={max(vels):.3f}")
    print(f"net displacement (x_end - x_start): {xs[-1]-xs[0]:.3f}  (total path length would need abs-diff sum)")
    print(f"standing term: mean={st.mean(standings):.3f}")
    print(f"move_reward: mean={st.mean(moves):.3f}  frac_near_zero(<0.05)={sum(1 for m in moves if m<0.05)/n:.2f}")
    print(f"stand_reward: mean={st.mean(stand_rewards):.3f}")
    print(f"per-step reward: mean={st.mean(rewards):.4f} (x1000 inner-steps~ sum={sum(rewards):.1f})")

    # identify collapse episodes: contiguous runs where torso_height < 0.9 (well below stand bound 1.2, margin 0.6 -> tolerance ~0 by height~0.6)
    collapse_runs = []
    in_collapse = False
    start = None
    for i, h in enumerate(heights):
        if h < 0.9 and not in_collapse:
            in_collapse = True
            start = i
        elif h >= 0.9 and in_collapse:
            in_collapse = False
            collapse_runs.append((start, i - 1))
    if in_collapse:
        collapse_runs.append((start, n - 1))
    print(f"collapse runs (torso_height<0.9): count={len(collapse_runs)}  total_steps_in_collapse={sum(b-a+1 for a,b in collapse_runs)} "
          f"({100*sum(b-a+1 for a,b in collapse_runs)/n:.1f}% of episode)")
    if collapse_runs:
        print(f"  runs: {collapse_runs[:15]}{'...' if len(collapse_runs)>15 else ''}")

    # reward earned during collapse vs non-collapse
    collapse_steps = set()
    for a, b in collapse_runs:
        collapse_steps.update(range(a, b + 1))
    reward_in_collapse = sum(rewards[i] for i in collapse_steps)
    reward_outside = sum(rewards) - reward_in_collapse
    print(f"reward earned during collapse steps: {reward_in_collapse:.2f}  outside collapse: {reward_outside:.2f}")
