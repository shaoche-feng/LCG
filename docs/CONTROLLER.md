# Controller settings and findings

Controllers tested so far for the DIAMOND world model on DM Control `walker_walk`, and what we learned. Repository
purpose, coding and access rules are in [AGENTS.md](AGENTS.md).

Status (2026-11-12): **no controller is stable yet.** The best single run is Gaussian-PMPO H10 + scalar EMA critic,
training seed 1 (final mean 646.9, 97% of seeds above 500), but the same configuration gives 337.4 and 480.9 on seeds 0
and 2.

## 1. Common setup

Unless a row says otherwise, every run uses:

| Setting | Value |
|---|---|
| Task / budget | `walker_walk`, 5k initial collection, 100k real steps (`collection.train.steps_per_epoch=500`), then 50 training-only epochs: 240 epochs in all, epoch 190 = collection end (100k) |
| World model | DIAMOND-derived diffusion model, `num_steps_conditioning=4`; reward/end LSTM model (`rew_end_model.training.seq_length=19` pinned when the horizon changes) |
| Controller updates | 100,600 PMPO updates (5,000 in the first epoch, 400 per later epoch), warm-up 100 |
| Imagination | `batch_size=32` rollouts per update from replay starts, horizon H (`world_model_env.horizon`); truncation bootstraps from the critic |
| Observation | K=4: the last four RGB frames concatenated along channels (`frame_stack=4`); not the batch size or horizon |
| PMPO | sign-only advantage groups, `alpha_pmpo=0.5`, `beta_kl=0.3` (KL to a frozen prior), `prior_refresh_interval=500`, `entropy_coef=0.01` |
| Learning | γ = 0.985, λ = 0.95, actor and value lr 1e-4, max grad norm 10 |
| Imagined policy history | repeated final frame (the reviewed `env_loop.py` patch), except where noted |
| Evaluation | eval30: 30 deterministic episodes, seeds 3001–3030, 500 steps; at 14 milestones (10k … 100k collection end, 100k final) |

Results below are final (epoch 240) eval30: **mean / median / fraction above 500**, unless marked.

## 2. Controllers tested

### Beta-PMPO (Beta actor, scalar critic, H = 15)

| Run | Change from common setup | Seed | Final mean / median / >500 | W&B |
|---|---|---|---|---|
| Stage 3 (historical) | entropy 0.01 (the reference Beta setting) | 0 | 561.4 / 595.8 / 83% | `ef22guzq` |
| Stage 3 repeat-frame reproduction | cleanup code + `env_loop.py` patch; reproduces the historical run's 30 returns exactly at every milestone | 0 | 561.4 / 595.8 / 83% | `mx970ez4` |
| Stage 3B | entropy 0.02 | 0 | — / 490.4 / 47% | — |
| Refresh 250 | `prior_refresh_interval` 500 → 250 (from Stage 3) | 0 | — / 210.9 / 3% | `18d5vcz5` |
| Replay-prompt history (cleanup `b976a6e`) | imagined history from the chronological replay prompt instead of repeated frame | 0 | 300.9 / 189.9 / 27% | `whe35fmy` |
| Replay-prompt history | as above | 1 | 246.0 / 253.8 / 0% | `3i6dvzj1` |
| Repeated-frame history | cleanup + patch | 1 | 321.3 / 272.2 / 7% | `4op80ok2` |
| Advantage-weighted Beta | raw PMPO loss weighted by \|advantage\| | 1 | 194.6 / 60.8 / 3% | `h6oh0qh6` |
| Beta + EMA critic | `slow_critic_coef=1.0`, `slow_critic_rate=0.02` | 0 | 395.0 / 260.3 / 40% | `betaemas0d429e7eb` |
| Beta + EMA critic | as above | 1 | 362.9 / 257.1 / 40% | `betaemas18d33bcb8` |
| Zero-shot adaptation | Stage 3's final world model frozen; fresh actor-critic trained only in imagination | 0 | bimodal; frac > 500 plateaus at 0.47–0.50 from ~10k updates | project `LCG-zero-shot-adaptation` |

### Gaussian-PMPO (tanh-transformed diagonal Gaussian actor, sign-only PMPO)

| Run | Change | Seed | Final mean / median / >500 | W&B |
|---|---|---|---|---|
| H15, no EMA | Gaussian actor (from the seed-1 repeated-frame Beta baseline) | 1 | 294.6 / 256.8 / 7% (peak 70k: 385.8 / 489.9 / 43%) | `onurwyaa` |
| H10, no EMA | horizon 10 (320 imagined transitions per update instead of 480) | 1 | 397.1 / 410.2 / 7% (collection end: 430.1 / 542.5 / 63%) | `6j20zi2z` |
| H8, no EMA | horizon 8 | 1 | 333.1 / 254.4 / 23% | `clfqlaeb` |
| H10, no EMA, frozen world model | continuation of `6j20zi2z` from epoch 190 with world/reward-model training stopped | 1 | 461.6 / 483.9 / 43% | `h10frozenwme7baa418` |
| **H10 + scalar EMA critic** | `slow_critic_coef=1.0`, `slow_critic_rate=0.02` | 0 | 337.4 / 265.4 / 23% | `h10emas04689e641` |
| **H10 + scalar EMA critic** | as above | 1 | **646.9 / 650.8 / 97%** | `h10ema5bc11033` |
| **H10 + scalar EMA critic** | as above | 2 | 480.9 / 542.7 / 57% | `h10emas292aea697` |
| H15 + scalar EMA critic | horizon 15 | 0 | 241.0 / 136.3 / 13% (collection end 341.8) | `h15emas0e0cf2239` |
| H15 + scalar EMA critic | horizon 15 | 1 | 372.0 / 384.9 / 33% | `h15emas189b01316` |
| H10 + two-hot EMA critic | 255-bin symexp two-hot critic, zero-initialized output layer, distributional EMA (CE) regularizer | 0 | 346.2 / 260.1 / 20% | `h10twohots0262d16e7` |
| H10 + two-hot EMA critic | as above | 1 | 356.2 / 347.2 / 0% | `h10twohots169ec6a24` |
| H10 + scalar EMA + replay value | critic also trained on real replay windows: `replay_value_coef=0.3`, `replay_starts=4` (8 segments × 4 starts) | 1 | 148.7 / 125.7 / 0% | `h10emarvs1f62ae16d` |

### DreamerV3-style controller (H = 15, seed 1)

DreamerV3 actor–critic adapted to this world model (two-hot critic, percentile return normalization, entropy 3e-4,
slow critic, replay value 0.3, γ = 0.997), Gaussian actor. W&B `dv3h15s1-b5a107f4`. **Halted at update 844** (0.8% of the
schedule) after strong action saturation: four of six action means moved into tanh's flat region (|μ| up to about 7).
Its policy scored the same as the untrained one (eval4 23.9 vs 25.1) and only held a prone posture. Too early to judge
the controller; not resumed.

### Bounded diagnostics (2,000 PMPO updates, world/reward models frozen, no new real data)

All from epoch-190 or epoch-90 snapshots of the H10 scalar-EMA runs; deterministic eval30 at the start and at update
2,000.

| Diagnostic | Arms | Result |
|---|---|---|
| Replay-value ablation (from `h10emarvs1f62ae16d`, epoch 190, start 281.2) | A independent starts / B grouped 8×4 / C grouped + replay 0.3 | 204.7 / 213.5 / 183.0. No grouping effect (B − A = +8.8); replay adds a borderline extra decline (C − B = −30.6, SE 15.4). All arms decline; calibration error rises from +8.8 to about +32. |
| Baseline follow-up D (from `h10ema5bc11033`, epoch 190) | arm-A settings | 582.0 → 573.0 deterministic, 524.7 → 539.2 stochastic; gait kept. The replay-value run's vulnerability is in its own learned state. |
| Controller × model/data swap (epoch 90 = 50k, seeds 0 and 1) | S00, S01, S10, S11 (controller seed × model/data seed) | 236.6 / 248.0 / 288.4 / 319.6 (starts 261.6 / 261.6 / 333.5 / 333.5). Every arm declines; none escapes sitting. The controller-1 advantage is the one it already had at 50k. |

## 3. Findings

### Behaviour and reward

1. **The reward permits a non-walking gait.** `walker_walk` reward is `stand_reward × move_reward`. The best policies use
   a crouched, forward-diving gait that falls and recovers 16–21 times per episode (returns 550–700). It does not
   alternate legs, and no run has produced sustained upright walking.
2. **Failures are absorbing low-reward modes, not small degradations.** Observed modes: seated scooting (≈250: torso
   upright at about 0.43 m, legs forward, moving 1.2–1.4 m/s by rocking); lying still, including on the back (20–135);
   a static split stance (≈160); seated rocking with backward falls (≈260–300); back-sliding at about 1.2 m/s (177–192).
   Evaluation returns are therefore bimodal; use fraction above 500, not the median, near the boundary.
3. **The seated mode is absorbing.** In the seed-1 Beta pair, 0 of 183 seated bouts were followed by an upright run.
   No get-up from sitting was ever sampled in imagination, and seated rollouts get advantages near zero.
4. **Backward falls decide the Stage 3 seed-0 difference.** Historical and current Stage 3 fall backward equally often
   (mostly at reset, set by the random initial pitch). Historical eventually recovers from all 15 backward falls, in 13
   of them by rolling to a forward tilt first; current recovers from none of 14 and back-slides. Historical is better on 29 of 30 seeds.
5. **Both world models lose a real get-up within about 5 steps of open loop,** and both critics overvalue backward-fallen
   states by 20–40. Since these errors are similar in both runs, the recovery difference sits in the policies.

### Training dynamics and data coverage

6. **Every run spends 20–40k steps in a seated-dominated phase.** Whether it escapes depends on early stochastic
   differences, not on the configuration: H10 scalar-EMA seed 1 had 3× more gait-like replay at 40–50k than seed 0
   (14–15% vs 5%), then 33–47%; seed 0 stayed 80–86% seated-like through 100k.
7. **The seated trap is self-reinforcing through the world model.** On identical gait contexts and actions, world models
   trained on seated-dominated data under-predict 15-step gait reward by 40–68% (seed 1's own model: 10–28%), so a
   trapped run gets little imagined credit for the gait it would need.
8. **Frozen-model continuations do not produce escape.** In eight arms across three diagnostics (2,000 updates each),
   no arm improved meaningfully and no seated episode recovered. Seed 1's real escape (50k → 70k: 333 → 518) happened
   while collecting new data and training the world model.
9. **Swapping model/data between seeds at 50k changes little.** Seed-1 model/data did not help the seed-0 controller
   (+11.3, SE 11.9, driven by two lying collapses). Seed-0 model/data may hurt the seed-1 controller in the upper tail
   (S11 − S10 = +31.3, SE 17.5), and under a foreign model the critic lowers its upright-state values by 13–14. Results
   follow the controller only through the advantage it already had.
10. **Several runs regress after collection ends.** H15 scalar EMA seed 0 drops abruptly 15–20 epochs after collection
    stops (episodes below 200: 0% at epoch 205, 67% at 210) while imagined λ-returns and values rise (47 → 84). Nothing
    logged moves clearly before the collapse. Stopping world-model training at collection end (frozen continuation of
    `6j20zi2z`) was mixed: −46/−65 at epochs 195–200, +54 to +131 at 220–240.

### Controller components

11. **Imagined policy history matters for Beta.** On seed 0, repeated-final-frame initialization (561.4 mean, 83% > 500)
    beats the chronological replay prompt (300.9, 27%) in a controlled comparison. On seed 1 the gain is smaller (+75.3
    mean, 7% vs 0% > 500). All later runs use repeated final frame.
12. **Entropy is non-monotonic** (Beta, seed 0): 0.01 → 83% > 500, 0.02 → 47%.
13. **Faster prior refresh hurt** (Beta, seed 0): refresh 250 → 3% > 500 vs 83% at 500. One seed per setting; may partly be
    trajectory luck.
14. **Advantage-weighted Beta PMPO is worse** than sign-only (seed 1 final mean 194.6 vs 321.3; low cluster is seated
    posture moving backward).
15. **Gaussian actor vs Beta (seed 1, H15):** ahead at 12 of 14 milestones and peaked higher (70k: 43% > 500), but regressed
    to about baseline by final (294.6 vs 321.3).
16. **Horizon (Gaussian, seed 1, no EMA): H10 is best.** H10 − H15 = +102.5 paired at final (25/30 wins); H8 is between.
    With EMA, H15 is worse than H10 on seed 1 and collapses after collection on seed 0. Longer horizons also mean more
    imagined transitions per update (480 vs 320), so horizon and volume are confounded.
17. **Scalar EMA critic regularization helps Gaussian, not clearly Beta.** At H15 seed 1 it beat no-EMA at both ends
    (final +77.3, 33% vs 7% > 500). For Beta, seed 1 gained (+41.7) and seed 0 lost to the strong reproduction baseline
    (395.0 vs 561.4). EMA lag is not a driver of failures: the online–EMA gap is 2–8% of the value spread and rises
    only after abrupt changes.
18. **H10 scalar EMA is the best configuration so far, but not seed-stable:** finals 337.4 / 646.9 / 480.9 (seeds 0/1/2,
    SD 127). Only seed 1 meets the success target.
19. **The two-hot critic did not help.** Seed 0 matched scalar EMA; seed 1 diverged at the first milestone (10k: −75.5
    paired) into seated rocking and never recovered (final 356.2, 0% > 500). Its critic was numerically healthy. The
    bundled change (head, loss, zero init, distributional EMA) cannot be separated with one seed each.
20. **Replay-value learning made things worse** (seed 1: final 148.7 vs 646.9; overestimation +52.6 vs −2.9). With 4
    consecutive starts per segment, real rewards are only 1.5–2.4% of the replay targets; the rest is imagination's own
    return at real states, so it did not anchor the critic and inflated with imagined returns after collection.

### Critic, world model and actor

21. **Critic targets are almost all bootstrapped value** (87–99% of the λ-target), and advantages are small differences
    (typically |A| < 2 on targets of 65–90). The sign of the PMPO update is decided by critic value differences between
    generated frames.
22. **Critics fit their imagined targets closely** (correlation 0.90–0.995) but are calibrated on real returns only where
    policy and data agree: seated states within ±1.5, seed 1's gait −2 to −6, static fallen states overvalued by +23 to
    +57 in 4 of 6 runs. On generated frames, values drift by up to ±21 within 15 imagined steps.
23. **World model:** one-step (teacher-forced) predictions and the reward model on real frames are accurate. Open-loop
    rollouts drift: by step 8 the image error reaches a frozen-frame baseline, and gait reward is under-predicted
    (pessimistic bias builds over steps 9–15). Imagination ranks fixed action candidates poorly against the simulator
    (Kendall τ −0.8 to +0.8). It does not hallucinate large reward from lying states.
24. **Imagined recovery may be hallucinated.** Imagined get-ups from back states did not happen when the same actions
    were replayed in the simulator (3 of 3 checked).
25. **Actor sensitivity to generated frames:** its outputs on generated frames differ from those on the matching real
    frames by 0.3–1.2 per action dimension by step 10–15, in all runs including successful ones.
26. **Action saturation is not a general failure driver** for Gaussian-PMPO (at most 4.6% near-bound actions in
    imagination, about 0% on real states; exception: two-hot seed 0 late, up to 36%). It was the failure of the
    DreamerV3-style controller. The float32 tanh/atanh round trip loses accuracy for |pre-tanh| ≳ 5 (shared by both
    Gaussian controllers).
27. **The scalar critic's gradient is almost always clipped** (pre-clip norm 90–350, 96–100% of updates clipped to 10);
    the two-hot critic (norm about 1) never is. Observed, not attributed.

### Ranking of limiting factors (controller diagnostic, 2026-10-26)

1. Data coverage and behavioural trapping (strongest, for the plateau).
2. Critic miscalibration in rarely trained failure states, uncorrected after collection (strongest, for post-collection
   collapse).
3. World-model rollout error, strongly coverage-dependent.
4. Actor sensitivity to generated vs real inputs (present everywhere).
5. Reward-model error (minor).
6. EMA lag and actor saturation (not supported).

## 4. Open questions and candidate next experiments

- **Why some seeds escape the seated trap.** More training seeds of H10 scalar EMA would show how often it escapes.
- **Live-training swap:** continue swapped controller/model-data pairings from epoch 90 with real collection and
  world-model training, to test whether escape follows the controller or the model/data once the loop is closed.
  (The epoch-90 snapshots were deleted with the server outputs, so this needs new runs.)
- **Coverage-targeted change** for the seated trap (no existing setting targets it cleanly).
- **Post-collection collapse:** stopping world-model training at collection end, tested with a matched control.
- **Refresh 250 vs 500** needs a second seed per setting; its KL analysis can only use W&B data now.
- **Two-hot critic:** separate head/loss, zero init and distributional EMA if revisited.

## 5. Where the evidence is

The detailed reports and review packages (`outputs/` in the local checkout) and all run outputs on kingplus0 were
deleted by 2026-11-12. What remains:

- **This document:** the findings and final numbers above.
- **[PMPO.md](PMPO.md):** controller design and the history-initialization comparison (seeds 0 and 1).
- **W&B** (entity `models-national-tsing-hua-university7947`): training curves and logged metrics of every run listed
  above, by W&B ID. Most runs are in project `lcg-stage3-seed1-interventions`. W&B has no checkpoints, datasets,
  behaviour traces or written reports.
- **Source code** on GitHub, below. Any result can be rerun from its branch, but runs are not guaranteed to reproduce
  exactly across seeds or GPUs.

**Source branches** (GitHub `shaoche-feng/LCG`): `dm-control-pmpo-cleanup` (baseline), `stage3-seed1-repeatframe-b976a6e`,
`stage3-seed1-replayhist-b976a6e`, `stage3-weighted-beta-seed1`, `stage3-beta-emacritic`, `stage3-gaussian-h10-emacritic`
(scalar EMA, used by all H10/H15 scalar-EMA runs), `stage3-gaussian-h10-twohot-emacritic`,
`stage3-gaussian-h10-ema-replayvalue`, `stage3-dreamerv3-controller`, `diagnostic-gaussian-h10-replayvalue-ablation`,
`diagnostic-gaussian-h10-ema-component-swap`. The plain Gaussian controller is commit `a13b57c` (in all Gaussian
branches).
