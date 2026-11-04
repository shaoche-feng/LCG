# Plan2Explore Adapted to DIAMOND: Design and Research Basis

Date: 2026-10-26\
Status: implemented and verified by tests and a bounded local GPU smoke (2026-10-26, commit `cd78477`); implementation notes in `docs/REWARD_FREE_EXPERIMENT_NOTES.md`, Step 6, "Plan2Explore-style lightweight ensemble disagreement". Not scientifically validated: no feature diagnostic, online pilot, downstream comparison or full run.\
Canonical repository path: `docs/PLAN2EXPLORE_DESIGN.md`. This corrects the earlier filename typos.

## 1. Purpose and scope

Build a Plan2Explore-inspired exploration signal for our DIAMOND-style diffusion world model. Keep one diffusion denoiser and add a small ensemble of feature predictors. Their disagreement becomes the intrinsic reward used to train the existing exploration controller.

This document explains what each reference studies, what its reported performance measures, and how its findings inform our design. It separates published evidence from proposed engineering choices.

The key unresolved questions are:

1. Which UNet feature or small combination of features should represent an observation?
2. At what diffusion noise level should that representation be extracted?
3. How should the conditional next-frame denoiser be used consistently as an encoder?
4. Can lightweight prediction disagreement identify useful uncertainty without excessive cost?

This is an adaptation, not an exact reproduction of original Plan2Explore. No cited paper below validates an optimal layer/noise pair for this particular adaptation.

## 2. Explain the terms before choosing a configuration

### Feature, layer, and noise level

A UNet layer produces an activation tensor, often shaped `[channels, height, width]`. This tensor is a feature map. Different layers can retain different information: local appearance, object position, body configuration, or broader structure.

The downsampling path reduces spatial resolution. The middle or deepest stage operates at low resolution. The upsampling path combines deeper features with skip connections to recover spatial detail. The deepest stage is not guaranteed to contain every useful detail, because skip connections carry information around it.

Diffusion noise level describes how much Gaussian corruption is associated with the denoiser input. It is different from environment time: an environment transition from t to t+1 is not a diffusion denoising step.

In the notation below, `t` denotes an environment step and `sigma` denotes diffusion noise. For EDM, a normally corrupted observed image is `o_t + sigma * epsilon`, followed by the model's existing preconditioning.

### What does feature performance mean in these papers?

Many papers use this procedure:

1. Feed an image through a diffusion model at a chosen noise level.
2. Extract a feature from a chosen layer.
3. Train or apply a downstream network using that feature.
4. Measure success on a task such as robot control, segmentation, or point matching.

Better results suggest that the feature exposes information useful for that task. They do not establish that the feature is best for next-step prediction or exploration.

For our baseline, a useful representation must retain changing scene information, support action-conditioned prediction, and produce informative disagreement between trained predictors. Recognition accuracy alone cannot establish all three.

## 3. Original Plan2Explore and the proposed adaptation

### Original method

Original Plan2Explore trains lightweight one-step predictors on a world model's state and the current action. They predict the next image encoder embedding. Variance between predicted means supplies the intrinsic reward. The world model generates the imagined trajectory; the small predictors supply its exploration reward. [R1]

### Proposed DIAMOND version

Let `E` be a feature extractor built from the current frozen DIAMOND denoiser. Let `z_t` be its representation of the current observation and available preceding context.

Each ensemble member learns:

```text
predictor_k(z_t, action_t) -> predicted_z_next_k
training target: z_next = E(next_observation, preceding_context)
```

Additional recent features or past actions may be needed if `z_t` does not preserve enough motion information. That is a possible extension, not an initial requirement.

The proposed loss is mean squared error against a detached next-feature target. The proposed raw intrinsic reward is the feature-wise average of the sample variance across predicted means:

```text
mean_prediction = mean(predictions across K members)
raw_reward = mean_over_features(
    sum_k((prediction_k - mean_prediction)^2) / (K - 1)
)
```

Keep raw disagreement separate from any configured reward scaling. The feature dimension, scaling statistics, variance convention, and applied reward divisor must be recorded.

The diffusion denoiser still generates the next imagined image. The lightweight ensemble does not replace the image generator and its mean prediction is not decoded into a frame.

| Component | Full diffusion ensemble baseline | Proposed Plan2Explore adaptation |
| --- | --- | --- |
| Main models | Several independently trained denoisers | One denoiser |
| Extra predictors | Full diffusion models | Small feature-prediction networks |
| Predicted quantity for disagreement | Next-image pixels | Next-observation features |
| Imagined next observation | Selected diffusion member | Single diffusion denoiser |
| Main additional training cost | Training every denoiser | Encoding replay and fitting small predictors |

## 4. Reference papers and their relevance

### R1. Planning to Explore via Self-Supervised World Models

- **Publication:** ICML 2020, Sekar et al.
- **Sources:** [paper](https://proceedings.mlr.press/v119/sekar20a/sekar20a.pdf), [project](https://ramanans1.github.io/plan2explore/).
- **What it studies:** reward-free exploration followed by adaptation to downstream tasks.
- **What performance means:** downstream control returns obtained using knowledge acquired during exploration.
- **Relevant mechanism:** a bootstrap ensemble predicts the next image embedding from the current model state and action. Disagreement between predicted means rewards exploration.
- **Relationship to our design:** supplies the predictor objective and disagreement mechanism. We need a replacement for its encoder embedding and latent state because our diffusion world model works differently.
- **What it does not establish:** which DIAMOND UNet layer, EDM noise level, or feature compression is best. Using the same feature definition for input and target is our simplifying proposal, not a requirement of the original method.

### R2. Pre-trained Text-to-Image Diffusion Models Are Versatile Representation Learners for Control

- **Method/publication:** Stable Control Representations (SCR), NeurIPS 2024, Gupta et al.
- **Source:** [published paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/8754ad68d557e633e5ce9dd02781e9cc-Paper-Conference.pdf).
- **What it studies:** visual features extracted from pretrained Stable Diffusion for robot manipulation and navigation policies.
- **What the quoted performance means:** manipulation success rate on Franka-Kitchen, not prediction loss or exploration quality.
- **Layer evidence:** Down[1-3] + Mid achieved 49.9% success, versus 41.6% for Mid alone.
- **Noise evidence:** timestep 0 achieved 49.9%, timestep 100 achieved 42.0%, and timestep 200 achieved 35.1%.
- **Relationship to our design:** motivates testing several spatial scales and low noise. Its spatial aggregation analysis also motivates retaining spatial structure during compression.
- **Limits:** Stable Diffusion features and policy training differ from our conditional denoiser and predictor ensemble. These block numbers and timesteps do not map directly to DIAMOND. The results do not prove that a particular two-layer subset is best.

### R3. Exploring Conditions for Diffusion Models in Robotic Control

- **Method/status:** ORCA, 2025 preprint.
- **Source:** [paper](https://arxiv.org/html/2510.15510v1).
- **What it studies:** adapting pretrained diffusion representations through learned conditioning prompts for imitation learning.
- **What performance means:** control scores from the resulting policies, including dm_control tasks.
- **Noise evidence:** timestep 0 had the highest average across its five dm_control tasks. Individual tasks differed: Walker walk scored 76.9 at timestep 0 and 78.6 at timestep 200. These are reported mean scores, not proof of a statistically resolved difference.
- **Relationship to our design:** provides evidence from a relevant environment family for considering low noise while retaining a moderate-noise comparison.
- **Limits:** learned prompts, imitation data, and the pretrained backbone differ from our reward-free setup. We are not proposing to import task-conditioned prompts or demonstrations.

### R4. Label-Efficient Semantic Segmentation with Diffusion Models

- **Publication:** ICLR 2022, Baranchuk et al.
- **Source:** [paper](https://arxiv.org/pdf/2112.03126).
- **What it studies:** using intermediate diffusion features to classify individual pixels into semantic regions or object parts.
- **What performance means:** intersection-over-union between predicted and annotated segmentation regions.
- **Layer evidence:** middle decoder/upsampling blocks were generally most informative. Deeper blocks favoured large structures, while shallower blocks retained information about smaller parts.
- **Noise evidence:** useful features generally occurred later in reverse denoising, where the image has less noise. The final method combined selected decoder blocks and DDPM timesteps 50, 150, and 250.
- **Relationship to our design:** motivates a middle upsampling feature as an alternative that could retain limb layout and boundaries.
- **Limits:** recognising a body part is not predicting its movement. The method uses segmentation annotations for evaluation; our exploration learner must not require them.

### R5. Emergent Correspondence from Image Diffusion

- **Method/publication:** DIFT, NeurIPS 2023, Tang et al.
- **Source:** [official implementation and extraction settings](https://github.com/Tsingularity/dift).
- **What it studies:** finding corresponding image locations using diffusion features.
- **What performance means:** accurate point correspondence or geometric matching, depending on the benchmark.
- **Layer/noise evidence:** the official Stable Diffusion semantic correspondence settings use timestep 261 and upsampling block 1. Geometric matching on HPatches uses timestep 0 and upsampling block 2.
- **Relationship to our design:** semantic identity and precise position can favour different representations. Dynamics prediction needs sensitivity to pose and movement, so settings selected for semantic matching should not be adopted automatically.
- **Limits:** the implementation's `ensemble_size=8` averages repeated feature extractions with noise samples. It is not an ensemble of eight independently trained dynamics predictors.

### R6. Diffusion Hyperfeatures: Searching Through Time and Space for Semantic Correspondence

- **Publication:** NeurIPS 2023, Luo et al.
- **Source:** [paper](https://arxiv.org/html/2305.14334).
- **What it studies:** learning to aggregate features across UNet layers and diffusion timesteps for point matching.
- **What performance means:** percentage of corresponding keypoints matched within a spatial tolerance.
- **Layer evidence:** informative features are distributed across layers and timesteps. Preferred layers differ between Stable Diffusion versions. Naively concatenating every layer performed worse than a useful single-layer baseline.
- **Relationship to our design:** supports a small, selected multiscale candidate rather than indiscriminately concatenating the entire UNet. Also argues against treating published layer indices as universal.
- **Limits:** the learned aggregation uses correspondence supervision and can require multiple denoising/inversion evaluations. We are not proposing that additional supervised training or inference cost for the initial baseline.

### R7. CleanDIFT: Diffusion Features without Noise

- **Publication:** CVPR 2025.
- **Source:** [paper](https://arxiv.org/html/2412.03439v3).
- **What it studies:** training a clean-input feature extractor by distilling features from a noisy diffusion teacher.
- **What performance means:** results on correspondence, depth estimation, segmentation, and classification.
- **Noise evidence:** ordinary diffusion features encode injected noise. Simply passing a clean image with a nonzero timestep degraded correspondence performance in its experiment; its trained clean-input model avoids that problem.
- **Relationship to our design:** motivates controlling extraction randomness and checking that disagreement is not dominated by corruption noise.
- **Limits:** its improvements require extra training. It does not show that setting epsilon to zero in our unmodified denoiser gives the same result, nor identify our best layer/noise pair.

### R8. Understanding Representation Dynamics of Diffusion Models via Low-Dimensional Modeling

- **Status:** arXiv:2502.05743.
- **Source:** [paper](https://arxiv.org/html/2502.05743v1).
- **What it studies:** theoretical and empirical explanations for how diffusion representation quality changes across noise scales, including DDPM/EDM experiments.
- **What performance means:** classification accuracy and segmentation quality obtained from the extracted representations.
- **Noise evidence:** representation quality can peak at intermediate noise. Its experiments also report advantages from clean-input extraction under their tested conventions.
- **Relationship to our design:** justifies retaining a moderate-noise candidate rather than assuming near-clean extraction always wins.
- **Limits:** its clean-input findings differ from the CleanDIFT correspondence experiment. Different backbones, tasks, and extraction conventions prevent a universal conclusion. Recognition results do not establish the optimum for action-conditioned uncertainty.

### R9. Diffusion for World Modeling: Visual Details Matter in Atari

- **Method/publication:** DIAMOND, NeurIPS 2024, Alonso et al.
- **Source:** [paper](https://arxiv.org/html/2405.12399v2).
- **What it studies:** learning a diffusion world model and training a policy in imagined pixel trajectories.
- **What performance means:** game returns obtained by its agent; this is not a feature-selection study.
- **Relevant architecture:** the denoiser receives the noisy image to be reconstructed, preceding observations, action conditioning, and diffusion-noise conditioning.
- **Relationship to our design:** explains why our feature extractor must specify both the image being encoded and its preceding context. A conditional next-frame denoiser cannot be treated as an ordinary image encoder without defining that convention.
- **Limits:** the paper does not establish a Plan2Explore representation or its best layer/noise setting.

### R10. Elucidating the Design Space of Diffusion-Based Generative Models

- **Method/publication:** EDM, NeurIPS 2022, Karras et al.
- **Sources:** [paper](https://arxiv.org/abs/2206.00364), [official preconditioning implementation](https://github.com/NVlabs/edm/blob/main/training/networks.py).
- **What it studies:** diffusion training and sampling design, including sigma-dependent network preconditioning.
- **Relationship to our design:** retain the existing denoiser's input scaling and noise embedding when extracting features. In standard EDM, noise conditioning includes `log(sigma) / 4`; therefore use a supported positive sigma for near-clean extraction.
- **Limits:** a DDPM timestep such as 100 or 261 is not numerically equivalent to EDM sigma. Likewise, sigma_data=0.5 is a preconditioning parameter, not evidence that sigma=0.5 gives the best feature.

## 5. A causal feature-extraction convention for DIAMOND

### Proposed convention

To encode an observed frame `o_t`:

1. Select the genuine preceding observation/action context that would have been used to predict `o_t`.
2. Form `o_t + sigma * epsilon_fixed` in the denoised-image input slot.
3. Apply the normal denoiser preconditioning and sigma conditioning.
4. Extract the selected UNet activations.
5. Apply a deterministic spatial compression and feature scaling to obtain `z_t`.

The preceding actions end at `a_(t-1)`. The candidate current action `a_t` enters the lightweight predictor separately. Do not accidentally use `o_(t+1)` when constructing its input.

The next-feature target is obtained by shifting this same procedure forward one transition. It may use the recorded next observation and action `a_t`, because those belong to the supervised training target's context.

```text
Input feature:  E(o_t, context before o_t)
Predictor:      f_k(input_feature, a_t)
Target feature: E(o_(t+1), context before o_(t+1))
```

This is a proposed, consistent use of the denoiser in its trained prediction role. It is not directly validated by the image-representation papers. Because action conditioning can also appear in the target feature, diagnostics should check that the representation contains changing visual information rather than merely making the action easy to recover.

### Training and imagination must agree

Use the same extraction function on recorded frames during predictor training and generated frames during imagination. Respect episode boundaries and retain enough preceding context at rollout initialization. Extra context may be required beyond the current imagination prompt.

The reward uses the heads' predictions before the actual next-feature target is known. Target extraction is required for predictor training, not for calculating disagreement itself.

Avoid three unvalidated shortcuts:

- Replacing the image being encoded with pure noise and assuming this reproduces DIFT-style image features.
- Feeding clean images with arbitrary nonzero sigma conditioning and assuming it is equivalent to matched noisy extraction.
- Reusing an activation from an arbitrary sampling step when its sigma, conditioning, or input convention differs from the training feature definition.

## 6. Provisional layer, compression, and noise choices

These are initial engineering choices, not established optima.

| Item | Proposed starting point | Rationale |
| --- | --- | --- |
| Input/target layer relationship | Same feature definition at consecutive times | Limits the initial design space; original Plan2Explore does not require this |
| First layer candidate | Last two downsampling scales, combined | Compact multiscale approximation motivated by control evidence |
| Main alternative | One middle upsampling feature | Spatial-detail evidence from segmentation and correspondence |
| Compression | Pool each selected map to a 4x4 grid, then concatenate | Retains coarse location while limiting predictor size |
| Initial sigma | 0.1 | A low-noise starting hypothesis; not a number proven by the papers |
| Comparison sigmas | A supported near-clean positive value and 0.5 | Tests geometric-detail and moderate-noise hypotheses |
| Corruption randomness | One reproducible Gaussian probe shared across heads | Prevents independent head inputs from confounding prediction disagreement |
| Feature scaling | Fixed training-data statistics during an epoch, with a variance floor | Makes coordinate units manageable and prevents unstable amplification |

Check actual block names, output shapes, sigma support, and preprocessing against the current implementation before coding these choices. Stable Diffusion feature resolutions and block numbers are not our layer specification.

Do not immediately collapse each map to one number per channel through global average pooling. A small spatial grid is the initial proposal because position matters. Do not add a jointly trained compression encoder initially: it would add another source of changing targets and would need an explicit anti-collapse design.

The reproducible probe convention must be saved with the run and reproduced during resume and imagination. It does not make the representation completely independent of noise; robustness to a second probe remains a diagnostic.

## 7. Proposed epoch workflow

Reuse the existing ExplorationTrainer and exploration controller.

1. Collect real data under the currently selected exploration policy.
2. Train the single denoiser with task rewards excluded.
3. Freeze the denoiser and its feature-extraction convention.
4. Encode eligible replay transitions into a temporary feature dataset.
5. Fit independently initialized lightweight predictor members, using configurable bootstrap sampling.
6. Freeze the denoiser and predictor ensemble during PMPO imagination.
7. Generate imagined frames with the denoiser; calculate intrinsic rewards from the predictor ensemble's next-feature disagreement.
8. Log raw/scaled rewards, predictor losses, feature statistics, and runtime; save resumable state and milestones through the existing workflow.

After the denoiser changes, regenerate any features tied to its old parameters. Do not mix stale cached targets with features from a newly trained denoiser.

For the initial implementation, reinitialize and refit the predictor ensemble after every denoiser training phase, using a deterministic epoch/member seed scheme. Do not warm-start heads across changed feature spaces. Section 12 defines configurable placeholder fitting settings; they are not a validated full-run schedule.

All feature fitting and selection must exclude downstream task rewards. The method uses our own learned denoiser; importing pretrained Stable Diffusion, demonstrations, or labelled segmentation data is not part of this proposal.

## 8. Later feature diagnostic before an online pilot

After implementation and its bounded correctness smoke, use an existing frozen denoiser and replay for feature selection. This diagnostic is a separate experiment and is not required to finish the implementation. Compare nine feature configurations rather than nine complete exploration runs:

| Feature candidate | Noise levels |
| --- | --- |
| Deepest downsampling/middle feature alone | Near-clean positive sigma, 0.1, 0.5 |
| Last two downsampling scales combined | Same three |
| One middle upsampling feature | Same three |

Use episode-separated fitting and validation transitions, consistent predictor budgets, and comparable predictor capacity. Match the feature dimension where practical; otherwise report differences in dimension and head parameter count.

For each configuration, check:

1. **Non-collapse:** variation across observations and retained spatial changes; an almost constant target must not win because it is easy to predict.
2. **Prediction beyond persistence:** compare next-feature prediction against copying the current feature.
3. **Action contribution:** compare with a predictor without the current action. Interpret weak one-step action effects cautiously rather than automatically rejecting the representation.
4. **Probe sensitivity:** repeat extraction on the same observations with another fixed Gaussian probe and compare that variation with actual observation changes.
5. **Disagreement versus error:** examine whether held-out prediction error increases with ensemble disagreement. This is a useful check, not proof of epistemic uncertainty or exploration quality.
6. **Cost:** report replay-encoding time, temporary dataset size, predictor-fitting time, feature-extraction time, and complete PMPO-update time.

Compare errors in consistently scaled feature coordinates and against baselines within each representation. Neither the smallest raw MSE nor the largest raw disagreement is a valid stand-alone selection rule.

Repeat the strongest candidate and a credible alternative at another denoiser maturity level if feasible. Early and mature models can expose different features. Then use a bounded online pilot to test the exploration loop and a downstream comparison to assess collected-data usefulness.

Do not use downstream rewards to choose the representation while describing the method as task-agnostic. Privileged simulator-state labels are not required for this initial diagnostic.

## 9. Efficiency and reproducibility

Feature extraction is not automatically free. A fixed-sigma encoding pass may be additional to ordinary image generation. Downsampling-only extraction may permit an early exit, but this depends on the model implementation and must be timed.

Encode replay in batches and avoid unnecessary duplicate encodings. During imagination, reuse the current representation only when it exactly matches the defined extraction convention. Keep the world model as the source of rollout images.

Exact resume needs the predictor parameters, optimizers if fitting is interrupted, update counters, bootstrap state, predictor RNG, probe convention, feature scaling statistics, layer/noise settings, and identity of the denoiser/replay behind any cache. A cache may instead be regenerated deterministically if that workflow is verified.

Benchmark both the complete PMPO phase and the complete epoch. Fast small heads do not guarantee a cheap baseline if feature extraction or replay encoding dominates.

## 10. Configuration decisions and current status

| Decision | Status |
| --- | --- |
| One denoiser plus lightweight predictor ensemble | Proposed baseline structure |
| Next-feature prediction and disagreement between predicted means | Based on original Plan2Explore |
| Keep the existing exploration controller | Proposed integration choice |
| Same feature definition for input and target | Initial simplification |
| Multiscale downsampling features | Preferred candidate; not validated |
| Middle upsampling feature | Main alternative; not validated |
| Sigma=0.1 | Provisional default only |
| Near-clean and sigma=0.5 comparisons | Proposed diagnostic |
| Spatial pooling to 4x4 | Provisional compression choice |
| Matched noisy extraction with a reproducible probe | Initial convention; requires validation |
| Number/width/depth of predictors | Configurable placeholders in Section 12; final experiment values undecided |
| Bootstrap | Configurable; proposed initial setting enabled |
| Predictor fitting budget and reward divisor | Configurable placeholders in Section 12; not tuned |
| Warm-start versus refit after denoiser changes | Fresh deterministic refit in the initial implementation |
| Online exploration and downstream benefit | Not established for this adaptation |

All method-specific settings should follow the repository's intrinsic-reward configuration conventions. Actual field names and migrations should be chosen after inspecting the current code. This document does not claim that any new configuration field or feature extractor already exists.

## 11. Relationship from research to the next decision

The literature supports a shortlist, not an exact optimum:

- Plan2Explore defines the prediction/disagreement mechanism.
- Control representation studies motivate low-noise, spatially structured features.
- Segmentation and correspondence studies provide the middle-decoder alternative.
- Noise studies motivate explicit, consistent extraction conventions and a moderate-noise comparison.
- DIAMOND and EDM determine the conditioning and preconditioning that our implementation must respect.

The immediate next step is implementation and bounded correctness validation under Section 12. The later frozen-model feature diagnostic can eliminate unsuitable representations and establish cost before an online pilot. The final scientific question remains whether the resulting exploration collects data that improves downstream learning.

## 12. Implementation contract for Claude

This section turns the preceding research discussion into a concrete first implementation. Earlier candidate comparisons remain future diagnostics, not permission to run a parameter sweep now.

### 12.1 Repository scope and naming

Read `docs/AGENTS.md`, the progress and notes documents, this design, and the actual exploration, intrinsic-reward, world-model, checkpoint, milestone, and downstream paths before editing.

Implement within the existing ExplorationTrainer and WorldModelEnv. Use a small method-specific module and a narrow feature-extraction interface where needed. Reuse existing bootstrap, RNG, scaling, logging, and checkpoint utilities when their semantics fit; avoid duplicating a trainer or creating a generic representation framework.

The intended selector is `intrinsic_reward=plan2explore_disagreement`, with a method-specific config at `config/intrinsic_reward/plan2explore_disagreement.yaml` and a Walker preset at `config/experiment/walker_plan2explore_100k.yaml`. Adapt to existing naming conventions if necessary, documenting the actual commands.

Keep all method settings under that intrinsic-reward configuration. Do not bring back the removed top-level `ensemble:` section. For this method, `num_members` counts lightweight predictors; the Agent and milestones still contain exactly one diffusion denoiser. Existing full-ensemble `num_members` continues to count denoisers.

### 12.2 Initial configurable settings

| Setting | Initial implementation value |
| --- | --- |
| Lightweight predictor members | 5; reject fewer than 2 |
| Bootstrap | true; episode-level Poisson(1) |
| Predictor network | Two hidden layers of width 256, nonlinear activation, linear next-feature output |
| Predictor optimizer | Adam, learning rate 1e-3 |
| Predictor fitting batch size | 256 |
| Fitting updates after each denoiser phase | 500 |
| Feature layers | Last two distinct downsampling resolutions, selected explicitly after architecture inspection |
| Feature pooling | Deterministic 4x4 adaptive average pooling per selected map, then concatenation |
| Feature sigma | 0.1, finite and strictly positive |
| Extraction noise | One fixed Gaussian tensor generated from a dedicated seed and saved or exactly reproducible |
| Feature scaling | Coordinate-wise training-data mean and standard deviation, with a configurable positive standard-deviation floor |
| Intrinsic reward scaling | Existing fixed-divisor semantics, divisor 1.0 initially; preserve raw reward logging |
| World-model compilation | Off in the initial preset; do not add a compilation project |

Noise convention as implemented (clarified 2026-10-26):
- **Formula:** the extractor uses the denoiser's own corruption, `o_t + feature_sigma * eps + sigma_offset_noise * eps_offset`. This combines pixel-noise sigma 0.1 with the denoiser's offset-noise scale 0.3, where `eps_offset` is one value per channel, constant over pixels.
- **Conditioning:** EDM preconditioning at `feature_sigma`, whose effective noise-conditioning sigma is sqrt(0.1² + 0.3²) ≈ 0.316.
- **Consequence:** "sigma 0.1" is therefore not a near-clean probe. Reducing the pixel-noise sigma alone leaves the effective sigma at ≥ 0.3 and the 0.3 offset shift in the input, while offset noise remains 0.3.
- **Implication for the diagnostic:** the near-clean comparison of Section 8 needs an explicit decision about the offset noise. It has not been implemented or run.

Predictor width/depth, learning rate, batch size, fitting budget, feature layers, pooling, sigma, probe seed, scaling floor, member count, bootstrap, and reward scaling must be configurable. These numerical settings are placeholders for a working baseline. They do not come from a tuned Plan2Explore reproduction and must not be presented as final experimental settings.

Do not copy Stable Diffusion block indices. Identify actual blocks in this UNet, record their tensor shapes, and place explicit selectors in the config. Validate selectors and pooling against a model-only shape check before collecting data. At minimum, the interface should permit one deepest/downsampling feature, the selected downsampling combination, and one middle upsampling feature. Adding that support does not authorize running the nine-configuration comparison.

Use a read-only activation path or a tightly scoped, cleaned-up hook. Ordinary denoising must retain its previous outputs and public behavior. If early-exit extraction is straightforward, use it; otherwise implement the correct full pass first and report its measured cost.

### 12.3 Context, valid transitions, and feature fitting

Implement the causal extraction convention in Section 5 literally: current-frame features use the real preceding context and past actions; the current candidate action is supplied separately to the predictors. Do not insert it into current-frame feature conditioning. Next-frame targets shift the same convention forward.

Account for the extra preceding context needed at imagination initialization without changing the existing PMPO observation history or action timing. If a complete context is unavailable, either use the repository's verified episode-start padding convention identically in fitting and imagination, or exclude that transition explicitly. Document and test the selected rule. Never cross episode boundaries to fabricate context.

Use only transitions with a verified real successor observation. Handle time-limit final observations and milestone-cut successors through the existing verified data access path. Do not train against blank padding or silently treat a missing successor as a real frame. Log eligible, excluded, and boundary-transition counts.

Fit feature-scaling statistics using eligible training replay only, never test data or task labels. Freeze them for the full predictor-fitting and PMPO phases of that epoch. Record near-constant feature coordinates and prevent their noise from being amplified without bound. Apply precisely the same pooling/scaling in replay encoding and imagination.

The denoiser must be in eval mode, detached, and unchanged while features and heads are fitted. Predictor fitting must update only the lightweight heads. The denoiser's optimizer and original trainability must be usable normally in the next denoiser phase. Preserve submodule-specific modes rather than restoring an entire hierarchy with a blanket train() call.

Each predictor needs independent initialization, sampling, and bootstrap behavior. Bootstrap multiplicities should be reproducible per epoch/member/episode and fixed throughout that refresh. Reuse the existing episode-level implementation when possible. When bootstrap support is empty, use an explicit logged fallback. With bootstrap disabled, members share eligible support but remain independently initialized and trained.

Use dedicated RNG state for probe generation, feature-cache construction if stochastic, head initialization, and head sampling. These operations must not accidentally consume the world-model or controller RNG streams. Use explicit DataLoader generators where applicable.

Build temporary features in batches on CPU or disk under the run's own outputs; do not keep the entire replay feature cache on the GPU. Prefer encoding each eligible observation/context once and indexing transitions into it. No task rewards or reward-bearing metadata may enter this cache. Episode identifiers may be retained solely for boundaries/bootstrap.

### 12.4 Intrinsic reward during imagination

For a given current state and action, all heads receive the same current feature and action. Score the variance of their predicted next-feature means using the K-1 convention in Section 3. Do not average over independent feature corruption per head.

Compute reward from those predictions; do not require the generated or real successor feature to calculate that transition's disagreement. Future frames may naturally influence later rewards after becoming current observations. Freeze all heads and the denoiser during PMPO updates; only the existing controller's parameters should change.

Expose the reward through the established intrinsic-reward interface with correctly aligned transition dimensions. Reuse the current horizon truncation, bootstrap-value, final-observation, and history behavior. The denoiser remains the sole next-image generator. Do not introduce a new actor, change the existing PMPO loss, or use feature prediction error as the reward.

### 12.5 Checkpoints, milestones, and cache lifecycle

Preserve the existing checkpoint granularity at completed epoch boundaries; a crash within an epoch may replay the unsaved work. Do not add mid-backward-pass or mid-fitting-update resume as a requirement for this implementation.

Save the completed predictor state needed to reproduce the run, extraction/scaling settings and statistics, probe identity, private RNG streams, bootstrap state or deterministic seed specification, fitting counters, and enough metadata to identify the source denoiser/replay feature version. Save optimizer state if it affects any supported continuation; state clearly if fitting always restarts from a completed boundary instead.

Validate the method and all behavior-changing settings before loading model weights. Refuse changed member count, bootstrap, architecture, feature selectors, sigma, probe, pooling/scaling, fitting configuration, or reward scaling on resume. Preserve compatibility with existing uniform, LCG, full-ensemble, and downstream checkpoints; use optional versioned method state or a justified format change with an explicit migration rather than silently reinterpreting old state.

Temporary caches may be regenerated instead of checkpointed, but must reproduce the same features and RNG behavior. Never silently reuse a cache after its denoiser, replay version, extraction settings, or scaling changes. A partially written cache must not be accepted as complete.

Milestones retain the usual single `denoiser.pt`, replay, and manifest semantics. Add provenance for this exploration method without requiring the downstream trainer to construct feature heads. Keep reward/task policy training fresh in downstream adaptation. Method-specific state belongs in exploration checkpoints; downstream still needs only the frozen denoiser and data.

### 12.6 Integration, logs, and tests

Uniform, LCG, and full diffusion ensemble behavior must remain unchanged. Their trainers must not construct unused Plan2Explore heads or feature caches. Preserve the accepted ensemble config migration and all downstream behavior.

The new Walker preset should match the existing baselines' interaction budget, denoiser schedule, milestones, and test-diagnostic protocol. Reuse the current exploration controller settings rather than choosing a new actor objective. Report any deliberate controller/config mismatch instead of silently introducing one.

Log predictor losses per member, fitting update counts, feature variation and scaling statistics, bootstrap fallback counts, raw/scaled reward summaries, reward finiteness, feature dimension/head parameter counts, encoding/fitting/scoring/full-PMPO time, and peak memory. Distinguish model-only extraction cost from complete update cost. Test diagnostics remain read-only and must not fit the feature scaler or predictors.

Add focused tests for:

- Correct layer extraction, shapes, sigma/preconditioning, probe reuse, pooling, and invalid-config rejection.
- A hand-computable disagreement example and the selected variance convention.
- Causality: changing an actual future frame cannot change the current feature or that step's predicted disagreement when current context/action/heads are held fixed.
- Correct action/history alignment, episode starts, final observations, and milestone-cut successors.
- Cached versus online features agreeing within explicitly justified floating-point precision.
- Actual head parameter updates and loss reduction on a learnable synthetic transition fixture; no denoiser updates during head fitting and no head updates during PMPO.
- Bootstrap on/off and fallback, deterministic fresh refitting, no stale-cache reuse, and private RNG isolation.
- Task-reward changes leaving features, losses, disagreement, updates, and sampled behavior unchanged under the same seeds.
- Uninterrupted versus supported crash/resume fidelity and refusal of changed method settings.
- Existing single-denoiser downstream loading and other intrinsic methods remaining intact.

Run the affected tests and full suite. Then run one bounded local Walker GPU smoke through the actual entry point, using a small schedule and a separate output directory. It should include real denoiser training, feature encoding, head fitting, PMPO updates, collection after the policy has updated, checkpoint/milestone publication, and crash/resume through the supported script. Verify counts, finite values, parameter isolation, and downstream milestone loading. Report rendering variation honestly; do not weaken exact state comparisons merely to pass.

The initial smoke is correctness and timing evidence, not learning-quality evidence. Do not run the feature sweep, online pilot, full 100k experiment, or downstream adaptation experiment as part of this implementation task. Do not access kingplus0, commit, or push.

### 12.7 Review delivery

Create `docs/changes/step6_plan2explore_review.zip` with the complete diff against the starting HEAD, changed source/config/tests, this updated design, progress/notes, a short report, resolved smoke configs, commands, stdout/stderr, exit codes, and comparison evidence.

The report must identify the exact layers/shapes and context convention implemented, the actual launch/override commands, all placeholder settings, bootstrap and scaling semantics, checkpoint/cache behavior, full-update and feature-preparation timings, source-file preservation checks, and remaining limitations. State separately what is implemented, smoke-verified, and still scientifically unvalidated.
