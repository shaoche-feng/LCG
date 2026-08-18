#! /usr/bin/env python
"""
LCG Stage 3.5 diagnostic: Monte Carlo variance-source and ranking-stability diagnostics for
Algorithm 3, following up on Stage 3's finding that the default M=3 estimator's candidate
ranking is nearly uncorrelated across seeds (mean Spearman ~0.10).

Does NOT modify src/lcg/*.py or the official method. Everything here is built by calling
lcg.gauss_newton.compute_vjp directly with explicit sigma/eps/xi (already a supported
parameter, no new VJP derivation) and lcg.candidate_score.candidate_score with different
`num_strata` values (already a supported parameter -- M *is* num_strata in this
implementation). Reuses the exact Stage 3 checkpoint/candidate-generation recipe.

Sections:
  A. Variance decomposition: fix (candidate, sigma, eps), resample only xi -- vs -- fix xi,
     resample (sigma, eps). Per-candidate CV and cross-repeat ranking (Spearman) stability
     for each source in isolation.
  B. Gaussian vs Rademacher xi, same fixed (sigma, eps) per candidate (paired comparison).
  C. Effective-rank estimate r_eff = (tr A)^2/tr(A^2) from repeated-xi score moments (exact
     for Gaussian probes: Var[xi^T A xi] = 2 tr(A^2), so r_eff = 2 * mean^2 / var).
  D. M in {1,3,6,12,24} vs a high-budget reference M_ref=96: Pearson/Spearman vs reference,
     top-k overlap, per-candidate CV, runtime/candidate.
  E. Independent-per-candidate probes vs common random numbers (CRN) shared across
     candidates, at fixed M=3: does CRN improve ranking stability at no extra VJP cost.

Usage:
    python scripts/diagnose_lcg_stage3_5.py
"""
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import torch

from data import Dataset, Episode
from envs.dm_control_env import make_dm_control_env
from lcg.candidate_score import candidate_score
from lcg.gauss_newton import compute_vjp
from lcg.precision import historical_precision, sample_valid_transitions
from lcg.sigma_strata import sample_sigma_stratum
from lcg.theta_s import selected_parameters
from models.diffusion import Denoiser, DenoiserConfig, DiffusionSampler, DiffusionSamplerConfig, SigmaDistributionConfig
from models.diffusion.inner_model import InnerModelConfig

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRATCH_BASE = Path(
    r"C:\Users\jerry\AppData\Local\Temp\claude\c--Users-jerry-Project-LCG"
    r"\974bd621-91fa-4766-822e-267b547a92c5\scratchpad"
)
DIAG_CHECKPOINT_PATH = SCRATCH_BASE / "lcg_diag_denoiser.pt"
TRAIN_DATASET_DIR = SCRATCH_BASE / "lcg_diag_train_dataset"
HELDOUT_DATASET_DIR = SCRATCH_BASE / "lcg_stage3_5_heldout_dataset"

DOMAIN_NAME, TASK_NAME = "cheetah", "run"
ENV_KWARGS = dict(size=64, camera_id=0, action_repeat=2, time_limit=1.0)
SIGMA_CFG = SigmaDistributionConfig(loc=-0.4, scale=1.2, sigma_min=0.002, sigma_max=20.0)
SAMPLER_CFG = DiffusionSamplerConfig(num_steps_denoising=3, sigma_min=2e-3, sigma_max=5.0, rho=7, order=1)

NUM_STEPS_CONDITIONING = 4
NUM_CANDIDATES = 20
H_D_B = 40
DAMPING = 1e-4
BETA = 1.0

R_SOURCE = 20          # repeats for the xi-only / (sigma,eps)-only variance decomposition
M_LIST = [1, 3, 6, 12, 24]
M_REF = 64
M_SWEEP_REPS = 3
CRN_REPS = 6
TOPK = 5


# --------------------------------------------------------------------------------------
# Setup (identical recipe to scripts/validate_lcg_stage3.py)
# --------------------------------------------------------------------------------------


def collect_episode(env, rng: np.random.Generator) -> Episode:
    obs_frames, act_list, rew_list, end_list, trunc_list = [], [], [], [], []
    raw_obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
    done = False
    while not done:
        action = rng.uniform(env.action_low, env.action_high).astype(np.float32)
        next_raw_obs, rew, terminated, truncated, _ = env.step(action)
        obs_frames.append(raw_obs)
        act_list.append(action)
        rew_list.append(rew)
        end_list.append(int(terminated))
        trunc_list.append(int(truncated))
        raw_obs = next_raw_obs
        done = terminated or truncated
    obs = torch.from_numpy(np.stack(obs_frames)).float().div(255).mul(2).sub(1).permute(0, 3, 1, 2).contiguous()
    act = torch.from_numpy(np.stack(act_list)).float()
    rew = torch.tensor(rew_list, dtype=torch.float32)
    end = torch.tensor(end_list, dtype=torch.uint8)
    trunc = torch.tensor(trunc_list, dtype=torch.uint8)
    return Episode(obs=obs, act=act, rew=rew, end=end, trunc=trunc, info={})


def build_dataset(directory: Path, name: str, target_num_steps: int, seed: int, include_stray: bool) -> Dataset:
    if directory.exists():
        shutil.rmtree(directory)
    dataset = Dataset(directory, name, cache_in_ram=True)
    if include_stray:
        stray_path = REPO_ROOT / "outputs" / "2026-08-17" / "15-35-46" / "dataset" / "train" / "000" / "00" / "0" / "0.pt"
        if stray_path.is_file():
            dataset.add_episode(Episode.load(stray_path))
    env = make_dm_control_env(domain_name=DOMAIN_NAME, task_name=TASK_NAME, **ENV_KWARGS)
    rng = np.random.default_rng(seed)
    while dataset.num_steps < target_num_steps:
        dataset.add_episode(collect_episode(env, rng))
    dataset.save_to_default_path()
    return dataset


def load_diagnostic_denoiser(device: torch.device) -> Denoiser:
    ckpt = torch.load(DIAG_CHECKPOINT_PATH, map_location=device, weights_only=False)
    inner_cfg = InnerModelConfig(
        img_channels=3, num_steps_conditioning=NUM_STEPS_CONDITIONING, cond_channels=256,
        depths=[2, 2, 2, 2], channels=[64, 64, 64, 64], attn_depths=[0, 0, 0, 0],
        continuous_action_dim=ckpt["action_dim"],
    )
    cfg = DenoiserConfig(inner_model=inner_cfg, sigma_data=0.5, sigma_offset_noise=0.3)
    denoiser = Denoiser(cfg).to(device)
    denoiser.load_state_dict(ckpt["denoiser"])
    denoiser.eval()
    print(f"Loaded Stage-2.6 diagnostic checkpoint (trained step={ckpt['step']}, action_dim={ckpt['action_dim']}).")
    return denoiser


def generate_candidates(denoiser, dataset, num_candidates, seed):
    sampler = DiffusionSampler(denoiser, SAMPLER_CFG)
    device = denoiser.device
    segment_ids = sample_valid_transitions(dataset, num_candidates, NUM_STEPS_CONDITIONING, seed=seed)
    candidates = []
    for segment_id in segment_ids:
        segment = dataset[segment_id]
        n = NUM_STEPS_CONDITIONING
        obs_window = segment.obs[:n].unsqueeze(0).to(device)
        act_window = segment.act[:n].unsqueeze(0).to(device)
        with torch.no_grad():
            y_star, _ = sampler.sample(obs_window, act_window)
        x_obs_flat = obs_window.reshape(1, -1, obs_window.shape[-2], obs_window.shape[-1])
        candidates.append((x_obs_flat, act_window, y_star.detach()))
    return candidates


# --------------------------------------------------------------------------------------
# Low-level single-sample score, built only from compute_vjp (Stage 1, unmodified)
# --------------------------------------------------------------------------------------


def rademacher_like(x: torch.Tensor) -> torch.Tensor:
    return (torch.randint(0, 2, x.shape, device=x.device).float() * 2 - 1)


def single_sample_score(denoiser, params, h_D, x_obs, x_act, y_star, sigma, eps, xi) -> float:
    y_sigma = (y_star + sigma.view(-1, 1, 1, 1) * eps).detach()
    v, _ = compute_vjp(denoiser, params, y_sigma, sigma, x_obs, x_act, xi=xi)
    return ((v * v) / h_D).sum().item()


def draw_unstratified_sigma(device):
    return sample_sigma_stratum(SIGMA_CFG, 0, 1, 1, device)  # num_strata=1 => full p_train(sigma)


def pearson_corr(a, b):
    a, b = torch.as_tensor(a, dtype=torch.float64), torch.as_tensor(b, dtype=torch.float64)
    a, b = a - a.mean(), b - b.mean()
    return ((a * b).sum() / (a.norm() * b.norm() + 1e-12)).item()


def spearman_corr(a, b):
    a, b = torch.as_tensor(a, dtype=torch.float64), torch.as_tensor(b, dtype=torch.float64)
    ra = torch.argsort(torch.argsort(a)).double()
    rb = torch.argsort(torch.argsort(b)).double()
    return pearson_corr(ra, rb)


def topk_overlap(scores, ref_scores, k):
    top = set(torch.topk(torch.as_tensor(scores), k).indices.tolist())
    top_ref = set(torch.topk(torch.as_tensor(ref_scores), k).indices.tolist())
    return len(top & top_ref) / k


# --------------------------------------------------------------------------------------
# Section A + B: variance decomposition (xi-only vs (sigma,eps)-only) + Gaussian/Rademacher
# --------------------------------------------------------------------------------------


def section_a_b(denoiser, params, h_D, candidates, device):
    print("\n" + "=" * 88)
    print("SECTION A: variance decomposition -- xi-only vs (sigma,eps)-only")
    print("=" * 88)
    n_c = len(candidates)

    # Fix one (sigma, eps) per candidate, reused across both the Gaussian- and
    # Rademacher-xi sweeps below for a fair paired comparison.
    fixed_sigma_eps = []
    for x_obs, x_act, y_star in candidates:
        sigma = draw_unstratified_sigma(device)
        eps = torch.randn_like(y_star)
        fixed_sigma_eps.append((sigma, eps))

    def xi_only_sweep(xi_fn, label):
        scores = torch.zeros(n_c, R_SOURCE)
        t0 = time.perf_counter()
        for r in range(R_SOURCE):
            for c, (x_obs, x_act, y_star) in enumerate(candidates):
                sigma, eps = fixed_sigma_eps[c]
                xi = xi_fn(y_star)
                scores[c, r] = single_sample_score(denoiser, params, h_D, x_obs, x_act, y_star, sigma, eps, xi)
        dt = time.perf_counter() - t0
        cv = (scores.std(dim=1) / scores.mean(dim=1).clamp_min(1e-12))
        corrs, scorrs = [], []
        for i in range(R_SOURCE):
            for j in range(i + 1, R_SOURCE):
                corrs.append(pearson_corr(scores[:, i], scores[:, j]))
                scorrs.append(spearman_corr(scores[:, i], scores[:, j]))
        print(f"\n  [{label}] xi resampled, (candidate,sigma,eps) fixed, {R_SOURCE} repeats, {n_c} candidates ({dt:.1f}s):")
        print(f"    per-candidate CV: mean={cv.mean().item():.4f} median={cv.median().item():.4f} max={cv.max().item():.4f}")
        print(f"    cross-repeat ranking stability: mean Pearson={np.mean(corrs):.4f}  mean Spearman={np.mean(scorrs):.4f}")
        return scores, cv, corrs, scorrs

    gauss_scores, gauss_cv, gauss_corrs, gauss_scorrs = xi_only_sweep(lambda y: torch.randn_like(y), "Gaussian xi")

    print("\n" + "=" * 88)
    print("SECTION B: Gaussian vs Rademacher xi (paired, same fixed sigma/eps)")
    print("=" * 88)
    rade_scores, rade_cv, rade_corrs, rade_scorrs = xi_only_sweep(rademacher_like, "Rademacher xi")
    print(f"\n  ratio of mean CV (Rademacher / Gaussian): {(rade_cv.mean() / gauss_cv.mean()).item():.4f}  "
          f"(theory: Rademacher variance <= Gaussian variance, i.e. ratio <= 1)")

    print("\n" + "=" * 88)
    print("SECTION A (cont.): (sigma,eps)-only, xi fixed per candidate")
    print("=" * 88)
    fixed_xi = [torch.randn_like(y_star) for _, _, y_star in candidates]
    scores_se = torch.zeros(n_c, R_SOURCE)
    for r in range(R_SOURCE):
        for c, (x_obs, x_act, y_star) in enumerate(candidates):
            sigma = draw_unstratified_sigma(device)
            eps = torch.randn_like(y_star)
            scores_se[c, r] = single_sample_score(denoiser, params, h_D, x_obs, x_act, y_star, sigma, eps, fixed_xi[c])
    cv_se = (scores_se.std(dim=1) / scores_se.mean(dim=1).clamp_min(1e-12))
    corrs_se, scorrs_se = [], []
    for i in range(R_SOURCE):
        for j in range(i + 1, R_SOURCE):
            corrs_se.append(pearson_corr(scores_se[:, i], scores_se[:, j]))
            scorrs_se.append(spearman_corr(scores_se[:, i], scores_se[:, j]))
    print(f"\n  [(sigma,eps)-only] xi fixed, (sigma,eps) resampled, {R_SOURCE} repeats, {n_c} candidates:")
    print(f"    per-candidate CV: mean={cv_se.mean().item():.4f} median={cv_se.median().item():.4f} max={cv_se.max().item():.4f}")
    print(f"    cross-repeat ranking stability: mean Pearson={np.mean(corrs_se):.4f}  mean Spearman={np.mean(scorrs_se):.4f}")

    print(f"\n  SUMMARY -- per-candidate CV: xi-only(Gaussian)={gauss_cv.mean().item():.4f}  "
          f"(sigma,eps)-only={cv_se.mean().item():.4f}  "
          f"[full-combination M=1 CV reported in Section D below]")
    print(f"  SUMMARY -- ranking stability (mean Spearman across repeats): "
          f"xi-only={np.mean(gauss_scorrs):.4f}  (sigma,eps)-only={np.mean(scorrs_se):.4f}")

    return gauss_scores


# --------------------------------------------------------------------------------------
# Section C: effective rank from repeated-xi score moments
# --------------------------------------------------------------------------------------


def section_c(gauss_scores: torch.Tensor):
    print("\n" + "=" * 88)
    print("SECTION C: effective rank r_eff = (tr A)^2 / tr(A^2), from repeated-xi moments")
    print("=" * 88)
    print("  For fixed (candidate, sigma, eps), r = xi^T A xi with xi ~ N(0,I_dy) and A fixed")
    print("  (A = 2 w(sigma) J diag(1/h_D) J^T, a d_y x d_y matrix -- note: this lives in the")
    print("  d_y-dimensional OUTPUT space, not the d_S-dimensional parameter space). For Gaussian")
    print("  xi: E[r]=tr(A), Var[r]=2 tr(A^2)  =>  r_eff = 2 * mean(r)^2 / var(r).")
    mean_r = gauss_scores.mean(dim=1)
    var_r = gauss_scores.var(dim=1, unbiased=True)
    r_eff = 2 * mean_r.square() / var_r.clamp_min(1e-30)
    print(f"\n  per-candidate r_eff: min={r_eff.min().item():.2f} median={r_eff.median().item():.2f} "
          f"mean={r_eff.mean().item():.2f} max={r_eff.max().item():.2f}")
    print(f"  for reference: d_y (denoiser output dim) = {3 * 64 * 64}, d_S (selected params) = 654851")
    return r_eff


# --------------------------------------------------------------------------------------
# Section D: M in {1,3,6,12,24} vs high-budget reference
# --------------------------------------------------------------------------------------


def section_d(denoiser, params, h_D, candidates, device):
    print("\n" + "=" * 88)
    print(f"SECTION D: M sweep vs reference (M_ref={M_REF})")
    print("=" * 88)
    n_c = len(candidates)

    t0 = time.perf_counter()
    ref_scores = torch.tensor([
        candidate_score(denoiser, params, h_D, SIGMA_CFG, x_obs, x_act, y_star, num_strata=M_REF)
        for x_obs, x_act, y_star in candidates
    ])
    t_ref = time.perf_counter() - t0
    print(f"  reference (M={M_REF}) computed: {t_ref:.1f}s total, {t_ref / n_c * 1000:.1f} ms/candidate")

    results = {}
    for M in M_LIST:
        rep_scores = torch.zeros(M_SWEEP_REPS, n_c)
        t0 = time.perf_counter()
        for r in range(M_SWEEP_REPS):
            for c, (x_obs, x_act, y_star) in enumerate(candidates):
                rep_scores[r, c] = candidate_score(denoiser, params, h_D, SIGMA_CFG, x_obs, x_act, y_star, num_strata=M)
        dt = time.perf_counter() - t0
        ms_per_candidate = dt / (M_SWEEP_REPS * n_c) * 1000

        pear = [pearson_corr(rep_scores[r], ref_scores) for r in range(M_SWEEP_REPS)]
        spear = [spearman_corr(rep_scores[r], ref_scores) for r in range(M_SWEEP_REPS)]
        overlap = [topk_overlap(rep_scores[r], ref_scores, TOPK) for r in range(M_SWEEP_REPS)]
        cv = (rep_scores.std(dim=0) / rep_scores.mean(dim=0).clamp_min(1e-12))

        results[M] = dict(pearson=np.mean(pear), spearman=np.mean(spear), overlap=np.mean(overlap),
                           cv=cv.mean().item(), ms_per_candidate=ms_per_candidate)
        print(f"\n  M={M:<3} ({M_SWEEP_REPS} reps x {n_c} candidates, {dt:.1f}s):")
        print(f"    Pearson vs ref:  {np.mean(pear):.4f}")
        print(f"    Spearman vs ref: {np.mean(spear):.4f}")
        print(f"    top-{TOPK} overlap vs ref: {np.mean(overlap):.4f}")
        print(f"    mean per-candidate CV (across the {M_SWEEP_REPS} reps): {cv.mean().item():.4f}")
        print(f"    runtime: {ms_per_candidate:.2f} ms/candidate")

    return results, ref_scores


# --------------------------------------------------------------------------------------
# Section E: independent probes vs common random numbers (CRN), fixed M=3
# --------------------------------------------------------------------------------------


def section_e(denoiser, params, h_D, candidates, device):
    print("\n" + "=" * 88)
    print("SECTION E: independent probes vs common random numbers (CRN), M=3")
    print("=" * 88)
    n_c = len(candidates)
    M = 3

    # Independent: each candidate gets its own fresh (sigma,eps,xi) per stratum per repeat.
    indep_scores = torch.zeros(CRN_REPS, n_c)
    t0 = time.perf_counter()
    for r in range(CRN_REPS):
        torch.manual_seed(1000 + r)
        for c, (x_obs, x_act, y_star) in enumerate(candidates):
            indep_scores[r, c] = candidate_score(denoiser, params, h_D, SIGMA_CFG, x_obs, x_act, y_star, num_strata=M)
    t_indep = time.perf_counter() - t0

    # CRN: draw M (sigma, eps-shape-template, xi) triples ONCE per repeat, reuse the SAME
    # sigma/xi and the SAME standard-normal draw for eps across every candidate in that
    # repeat (eps is applied to each candidate's own y*, so the corrupted y_sigma still
    # differs per candidate even though the underlying noise realization is shared).
    crn_scores = torch.zeros(CRN_REPS, n_c)
    t0 = time.perf_counter()
    y_shape = candidates[0][2].shape
    for r in range(CRN_REPS):
        torch.manual_seed(2000 + r)
        shared = []
        for m in range(M):
            sigma = sample_sigma_stratum(SIGMA_CFG, m, M, 1, device)
            eps = torch.randn(y_shape, device=device)
            xi = torch.randn(y_shape, device=device)
            shared.append((sigma, eps, xi))
        for c, (x_obs, x_act, y_star) in enumerate(candidates):
            per_stratum = []
            for sigma, eps, xi in shared:
                per_stratum.append(single_sample_score(denoiser, params, h_D, x_obs, x_act, y_star, sigma, eps, xi))
            crn_scores[r, c] = float(np.mean(per_stratum))
    t_crn = time.perf_counter() - t0

    def ranking_stability(scores):
        corrs, scorrs = [], []
        for i in range(CRN_REPS):
            for j in range(i + 1, CRN_REPS):
                corrs.append(pearson_corr(scores[i], scores[j]))
                scorrs.append(spearman_corr(scores[i], scores[j]))
        return np.mean(corrs), np.mean(scorrs)

    pear_indep, spear_indep = ranking_stability(indep_scores)
    pear_crn, spear_crn = ranking_stability(crn_scores)

    print(f"\n  independent probes ({CRN_REPS} reps, {t_indep:.1f}s): "
          f"mean Pearson={pear_indep:.4f}  mean Spearman={spear_indep:.4f}")
    print(f"  common random numbers ({CRN_REPS} reps, {t_crn:.1f}s): "
          f"mean Pearson={pear_crn:.4f}  mean Spearman={spear_crn:.4f}")
    print(f"  same VJP budget in both cases ({M} VJPs/candidate); "
          f"CRN {'improves' if spear_crn > spear_indep else 'does not improve'} ranking stability here")


# --------------------------------------------------------------------------------------


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    denoiser = load_diagnostic_denoiser(device)
    params = selected_parameters(denoiser)
    d_S = sum(p.numel() for p in params)

    train_dataset = build_dataset(TRAIN_DATASET_DIR, "lcg_stage3_5_train_for_hD", 1500, seed=0, include_stray=True)
    heldout_dataset = build_dataset(HELDOUT_DATASET_DIR, "lcg_stage3_5_heldout", 400, seed=555, include_stray=False)

    h_D = historical_precision(
        denoiser, params, train_dataset, SIGMA_CFG, B=H_D_B, num_strata=3,
        beta=BETA, damping=DAMPING, seed=0,
    )
    assert h_D.shape == (d_S,) and torch.all(h_D > 0)
    print(f"Frozen h_D: shape={tuple(h_D.shape)}, min={h_D.min().item():.4g}, max={h_D.max().item():.4g}")

    candidates = generate_candidates(denoiser, heldout_dataset, NUM_CANDIDATES, seed=42)
    print(f"Fixed candidate set: {len(candidates)} candidates (real held-out x*, DiffusionSampler-imagined y*).")

    before_state = {k: v.detach().clone() for k, v in denoiser.state_dict().items()}

    gauss_scores = section_a_b(denoiser, params, h_D, candidates, device)
    section_c(gauss_scores)
    results, ref_scores = section_d(denoiser, params, h_D, candidates, device)
    section_e(denoiser, params, h_D, candidates, device)

    for k, v in denoiser.state_dict().items():
        assert torch.equal(before_state[k], v), f"denoiser parameter {k} changed during Stage 3.5 diagnostics!"
    print("\n[integrity] denoiser parameters unchanged before vs after all Stage 3.5 diagnostics: OK")

    print("\n" + "=" * 88)
    print("SUMMARY")
    print("=" * 88)
    for M in M_LIST:
        r = results[M]
        print(f"  M={M:<3}  Spearman-vs-ref={r['spearman']:.3f}  top{TOPK}-overlap={r['overlap']:.3f}  "
              f"CV={r['cv']:.3f}  {r['ms_per_candidate']:.1f} ms/candidate")

    print("\nStage 3.5 diagnostic complete.")


if __name__ == "__main__":
    main()
