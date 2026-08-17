# Diffusion for World Modeling: Visual Details Matter in Atari (NeurIPS 2024 Spotlight)

[**TL;DR**] 💎 DIAMOND (DIffusion As a Model Of eNvironment Dreams) is a reinforcement learning agent trained entirely in a diffusion world model.

🌍 [Project Page](https://diamond-wm.github.io) • 🤓 [Paper](https://arxiv.org/pdf/2405.12399) • 𝕏 [Atari thread](https://x.com/EloiAlonso1/status/1793916382779982120) • 𝕏 [CSGO thread](https://x.com/EloiAlonso1/status/1844803606064611771) • 💬 [Discord](https://discord.gg/74vha5RWPg)

<div align='center'>
  RL agent playing in autoregressive imagination of Atari world models
  <br>
  <img alt="DIAMOND agent in WM" src="https://github.com/user-attachments/assets/eb6b72eb-73df-4178-8a3d-cdad80ff9152">

</div>

<div align='center'>
  Human player in CSGO world model (full quality video <a href="https://diamond-wm.github.io/static/videos/grid.mp4">here</a>)
  <br>
  <img alt="DIAMOND agent in WM" src="https://github.com/user-attachments/assets/dcbdd523-ca22-46a9-bb7d-bcc52080fe00">
</div>

Quick install to try our [pretrained world models](#try) using [miniconda](https://docs.anaconda.com/free/miniconda/miniconda-install/):

>```bash
>git clone https://github.com/eloialonso/diamond.git
>cd diamond
>conda create -n diamond python=3.10
>conda activate diamond
>pip install -r requirements.txt
>```

For Atari (world model + RL agent)

>```bash
>python src/play.py --pretrained
>```

For CSGO (world model only)

>```bash
>git checkout csgo
>python src/play.py
>```

And press `m` to take control (the policy is playing by default)!

**Warning**: Atari ROMs will be downloaded with the dependencies, which means that you acknowledge that you have the license to use them.

## CSGO


**Edit**: Check out the [csgo branch](https://github.com/eloialonso/diamond/tree/csgo) to try our DIAMOND's world model trained on *Counter-Strike: Global Offensive*!

```bash
git checkout csgo
python src/play.py
```
> Note on Apple Silicon you must enable CPU fallback for MPS backend with
> PYTORCH_ENABLE_MPS_FALLBACK=1 python src/play.py


<a name="quick_links"></a>
## Quick Links

- [Try our playable diffusion world models](#try)
- [Launch a training run](#launch)
- [DM Control Suite support](#dm_control)
- [Configuration](#configuration)
- [Visualization](#visualization)
  - [Play mode (default)](#play_mode)
  - [Dataset mode (add `-d`)](#dataset_mode)
  - [Other options, common to play/dataset modes](#other_options)
- [Run folder structure](#structure)
- [Results](#results)
- [Citation](#citation)
- [Credits](#credits)

<a name="try"></a>
## [⬆️](#quick_links) Try our playable diffusion world models

```bash
python src/play.py --pretrained
```

Then select a game, and world model and policy pretrained on Atari 100k will be downloaded from our [repository on Hugging Face Hub 🤗](https://huggingface.co/eloialonso/diamond) and cached on your machine.

Some things you might want to try:
- Press `m` to change the policy between the agent and human (the policy is playing by default).
- Press `↑/↓` to change the imagination horizon (default is 50 for playing).

To adjust the sampling parameters (number of denoising steps, stochasticity, order, etc) of the trained diffusion world model, for instance to trade off sampling speed and quality, edit the section `world_model_env.diffusion_sampler` in the file `config/trainer.yaml`.

See [Visualization](#visualization) for more details about the available commands and options.

<a name="launch"></a>
## [⬆️](#quick_links) Launch a training run

To train with the hyperparameters used in the paper on cuda:0, launch:
```bash
python src/main.py env.train.id=BreakoutNoFrameskip-v4 common.devices=0
```

This creates a new folder for your run, located in `outputs/YYYY-MM-DD/hh-mm-ss/`.

To resume a run that crashed, navigate to the fun folder and launch:

```bash
./scripts/resume.sh
```

<a name="dm_control"></a>
## [⬆️](#quick_links) DM Control Suite support

This fork extends DIAMOND from Atari to the [DeepMind Control Suite](https://github.com/google-deepmind/dm_control), a collection of continuous-control tasks built on MuJoCo physics. **The original Atari path described above is fully preserved and unaffected** — it remains the default.

- **Observations**: rendered RGB frames from `dm_control`'s MuJoCo renderer (not the default state observations such as `position`/`velocity`), flowing through the same observation pipeline as Atari.
- **Actions**: continuous, task-dependent action spaces. Action dimension and bounds (`low`/`high`) are inferred automatically from each task's `action_spec()` at environment-construction time — never hard-coded to a specific dimension or task, so the same code path supports any DM Control Suite task without per-task code changes.
- **Rewards**: raw continuous DM Control rewards are used directly end to end (world model, reward model, and actor-critic training) — no sign-clipping or classification, unlike the discrete-reward Atari path.

### Selecting Atari vs. DM Control

The environment is selected through Hydra's config group system, via the files in `config/env/`:

- `config/env/atari.yaml` (default) — the original Atari path.
- `config/env/dm_control.yaml` — the DM Control Suite path.

To run the **original Atari path** (unchanged):
```bash
python src/main.py env.train.id=BreakoutNoFrameskip-v4 common.devices=0
```

To run **DM Control's `cheetah-run`**:
```bash
python src/main.py env=dm_control env.train.domain_name=cheetah env.train.task_name=run common.devices=0
```

To select **another DM Control domain/task**, override `domain_name`/`task_name` with any valid `dm_control.suite` pair (e.g. `walker`/`walk`, `hopper`/`hop`):
```bash
python src/main.py env=dm_control env.train.domain_name=walker env.train.task_name=walk common.devices=0
```

Resuming a crashed run works identically for either environment type — `./scripts/resume.sh` requires no environment-specific flags in either case (see [Run folder structure](#structure)).

### DM Control configuration options

Set in `config/env/dm_control.yaml`, or overridden on the command line (e.g. `env.train.action_repeat=2`):

| Field | Meaning |
|---|---|
| `domain_name` / `task_name` | DM Control Suite task, e.g. `cheetah` / `run` |
| `size` | rendered observation height/width (square) |
| `camera_id` | which MuJoCo camera to render from |
| `action_repeat` | number of physics substeps executed per agent action |
| `time_limit` | episode duration in seconds (`null` = task default) |

### Additional dependency

DM Control support requires `dm_control` (which pulls in `mujoco`), on top of `requirements.txt`:
```bash
pip install dm_control
```

<a name="configuration"></a>
## [⬆️](#quick_links) Configuration

We use [Hydra](https://github.com/facebookresearch/hydra) for configuration management.

All configuration files are located in the `config` folder:

- `config/trainer.yaml`: main configuration file.
- `config/agent/default.yaml`: architecture hyperparameters.
- `config/env/atari.yaml`: Atari environment hyperparameters (default).
- `config/env/dm_control.yaml`: DM Control Suite environment hyperparameters (see [DM Control Suite support](#dm_control)).

You can turn on logging to [weights & biases](https://wandb.ai) in the `wandb` section of `config/trainer.yaml`.

Set `training.model_free=true` in the file `config/trainer.yaml` to "unplug" the world model and perform standard model-free reinforcement learning.

<a name="visualization"></a>
## [⬆️](#quick_links) Visualization

<a name="play_mode"></a>
### [⬆️](#quick_links) Play mode (default)

To visualize your last checkpoint, launch **from the run folder**:

```bash
python src/play.py
```

By default, you visualize the policy playing in the world model. To play yourself, or switch to the real environment, use the controls described below.

```txt
Controls (play mode)

(Game-specific commands will be printed on start up)

⏎   : reset environment

m   : switch controller (policy/human)
↑/↓ : imagination horizon (+1/-1)
←/→ : next environment [world model ←→ real env (test) ←→ real env (train)]

.   : pause/unpause
e   : step-by-step (when paused)
```

Add `-r` to toggle "recording mode" (works only in play mode). Every completed episode will be saved in `dataset/rec_<env_name>_<controller>`. For instance:

- `dataset/rec_wm_π`: Policy playing in world model.
- `dataset/rec_wm_H`: Human playing in world model.
- `dataset/rec_test_H`: Human playing in test real environment.

You can then use the "dataset mode" described in the next section to replay the stored episodes.

<a name="dataset_mode"></a>
### [⬆️](#quick_links) Dataset mode (add `-d`)

**In the run folder**, to visualize the datasets contained in the `dataset` subfolder, add `-d` to switch to "dataset mode":

```bash
python src/play.py -d
```

You can use the controls described below to navigate the datasets and episodes.

```txt
Controls (dataset mode)

m   : next dataset (if multiple datasets, like recordings, etc)
↑/↓ : next/previous episode
←/→ : next/previous timestep in episodes
PgUp: +10 timesteps
PgDn: -10 timesteps
⏎   : back to first timestep
```

<a name="other_options"></a>
### [⬆️](#quick_links) Other options, common to play/dataset modes

```txt
--fps FPS             Target frame rate (default 15).
--size SIZE           Window size (default 800).
--no-header           Remove header.
```

<a name="structure"></a>
## [⬆️](#quick_links) Run folder structure

Each new run is located at `outputs/YYYY-MM-DD/hh-mm-ss/`. This folder is structured as follows:

```txt
outputs/YYYY-MM-DD/hh-mm-ss/
│
└─── checkpoints
│   │   state.pt  # full training state
│   │
│   └─── agent_versions
│       │   ...
│       │   agent_epoch_00999.pt
│       │   agent_epoch_01000.pt  # agent weights only
│
└─── config
│   |   trainer.yaml
|
└─── dataset
│   │
│   └─── train
│   |   │   info.pt
│   |   │   ...
|   |
│   └─── test
│       │   info.pt
│       │   ...
│
└─── scripts
│   │   resume.sh
|   |   ...
|
└─── src
|   |   main.py
|   |   ...
|
└─── wandb
    |   ...
```

<a name="results"></a>
## [⬆️](#quick_links) Results

The file [results/data/DIAMOND.json](results/data/DIAMOND.json) contains the results for each game and seed used in the paper.

The DDPM code used for Section 5.1 of the paper can be found on the [ddpm](https://github.com/eloialonso/diamond/tree/ddpm) branch.

<a name="citation"></a>
## [⬆️](#quick-links) Citation

```text
@inproceedings{alonso2024diffusionworldmodelingvisual,
      title={Diffusion for World Modeling: Visual Details Matter in Atari},
      author={Eloi Alonso and Adam Jelley and Vincent Micheli and Anssi Kanervisto and Amos Storkey and Tim Pearce and François Fleuret},
      booktitle={Thirty-eighth Conference on Neural Information Processing Systems}}
      year={2024},
      url={https://arxiv.org/abs/2405.12399},
}
```

<a name="credits"></a>
## [⬆️](#quick_links) Credits

- [https://github.com/crowsonkb/k-diffusion/](https://github.com/crowsonkb/k-diffusion/)
- [https://github.com/huggingface/huggingface_hub](https://github.com/huggingface/huggingface_hub)
- [https://github.com/google-research/rliable](https://github.com/google-research/rliable)
- [https://github.com/pytorch/pytorch](https://github.com/pytorch/pytorch)
