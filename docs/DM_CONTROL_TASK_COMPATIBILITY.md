# dm_control task compatibility for multi-task reward labels

Checked 2026-10-15 against the local `lcg` environment: **dm_control 1.0.44, MuJoCo 3.11.0**
(`site-packages/dm_control/suite/`). Line numbers below refer to those files. Re-check this
table after upgrading dm_control.

## What "compatible" means

During collection one task (the first in `env.*.task_names`) drives the environment. Every
other task is a separate dm_control `Task` object that never steps, resets or renders; its
`get_reward` is evaluated on the collection physics after each action-repeat substep. A set
of tasks is compatible only if all of the following hold, so that the collected trajectory
and every recorded label are exactly what each task's own environment would produce:

1. **Same model and dynamics**: identical compiled MuJoCo model (also after reset, since
   some tasks edit the model in `initialize_episode`).
2. **Same reset distribution**: identical `initialize_episode`, drawing the same random
   numbers in the same order, so the choice of collection task does not change the data.
3. **Same action spec, control timestep, time limit and termination.**
4. **Reward reads only physics**: no task state set by the task's own reset, no model
   fields the reset writes per task, no randomness, no state changes.
5. **No step callbacks** that differ (`before_step`/`after_step`).

Facts shared by every supported group below: tasks come from the same factory model; same
`Task` class (only constructor constants differ); no task overrides `before_step` or
`after_step` (only `base.Task`'s, which recolours materials only when reward
visualization is on, and `suite.load` leaves it off); only `lqr` overrides
`get_termination` (`lqr.py:264`).

## Verified groups (`REWARD_LABEL_TASK_GROUPS`)

Any subset of one group may be selected, in any order; a single task is always allowed.

| Domain | Group | Why it qualifies (source) |
|---|---|---|
| acrobot | swingup, swingup_sparse | Same `Balance` (`acrobot.py:81`), reset draws two uniform angles (`:96`); reward reads tip-target distance and the fixed target site size (`:121`); only `sparse` differs. |
| cartpole | balance, balance_sparse | Same `Balance` with `swing_up=False` (`cartpole.py:40,51`), same reset branch (`:177`); reward (`:223`) reads cart/pole state and `physics.control()`. |
| cartpole | swingup, swingup_sparse | As above with `swing_up=True` (`:62,73`). |
| hopper | stand, hop | Same `Hopper` (`hopper.py:88`), same reset (`:104`; `_timeout_progress` is never read); reward (`:119`) reads height, speed and `physics.control()`, which the collection step sets. |
| humanoid | stand, walk, run, run_pure_state | Same `Humanoid` (`humanoid.py:132`) and collision-free reset (`:152`); reward (`:183`) reads head height, uprightness, COM velocity, control. `run_pure_state` differs from `run` only in its state observation, which pixel collection does not use. |
| humanoid_CMU | stand, walk, run | Same `HumanoidCMU` (`humanoid_CMU.py:119`, reset `:136`, reward `:162`); only `move_speed` differs. |
| walker | stand, walk, run | Same `PlanarWalker` (`walker.py:101`), reset (`:118`); reward (`:139`) reads torso height/uprightness and COM velocity; only `move_speed` differs. |

## All installed domains and tasks

| Domain | Tasks | Verified groups | Unsupported combinations and reason |
|---|---|---|---|
| acrobot | swingup, swingup_sparse | {swingup, swingup_sparse} | — |
| ball_in_cup | catch | single task | — |
| cartpole | balance, balance_sparse, swingup, swingup_sparse, two_poles, three_poles | {balance, balance_sparse}; {swingup, swingup_sparse} | balance* with swingup*: different reset distributions (`cartpole.py:177`). two_poles, three_poles: different models (`_make_model`, `:105`). |
| cheetah | run | single task | — |
| dog | stand, walk, trot, run, fetch | none | stand+walk share a model, but the reward (`dog.py:317`) reads `_stand_height`/`_body_weight`, set only by the task's own reset (`:264`); a never-reset label task raises `AttributeError`. walk/trot/run: floor size = speed x 15 (`:61,110-136`), so different models and rendered pixels. fetch: ball and walls in the model, different reset (`:395`). |
| finger | spin, turn_easy, turn_hard | none | spin's reset changes hinge damping (`finger.py:139`). turn_easy+turn_hard: each reset writes its own target radius into the model (`:174`) and the reward reads it (`dist_to_target`, `:197`). |
| fish | upright, swim | none | Different resets: swim also draws a target position; each edits the target differently (`fish.py:108,144`). |
| hopper | stand, hop | {stand, hop} | — |
| humanoid | stand, walk, run, run_pure_state | {stand, walk, run, run_pure_state} | — |
| humanoid_CMU | stand, walk, run | {stand, walk, run} | — |
| lqr | lqr_2_1, lqr_6_2 | none | Different body/actuator counts; stiffness is drawn from the task's RNG when the model is built (`lqr.py:140`); has termination (`:264`). |
| manipulator | bring_ball, bring_peg, insert_ball, insert_peg | none | Each task removes different props from the model (`manipulator.py:46`); reset edits target poses in the model (`:186`). |
| pendulum | swingup | single task | — |
| point_mass | easy, hard | none | hard randomizes actuator gains in the model and draws extra random numbers at reset (`point_mass.py:86`). |
| quadruped | walk, run, escape, fetch | none | walk+run share task and reset but floor size differs (10 vs 100, `quadruped.py:55,97,109`): different models and rendered pixels. escape: terrain generated at reset (`:362`). fetch: ball and walls (`:426`). |
| reacher | easy, hard | none | Each reset writes its own target size into the model (`reacher.py:89`), which the reward reads (`:110`). |
| stacker | stack_2, stack_4 | none | Different numbers of boxes in the model (`stacker.py:41`). |
| swimmer | swimmer6, swimmer15 | none | Different link counts and action dimensions (`swimmer.py:83`). |
| walker | stand, walk, run | {stand, walk, run} | — |

19 domains, 51 tasks; 7 verified groups covering 18 tasks.

## Runtime evidence

- **Pair probe** (raw dm_control, every ordered same-domain pair, 3 episodes, same seed and
  actions): compiled model at construction and after each reset, qpos/qvel/act at every
  step, and label task reward on the collection physics vs the label task's own reward.
  Exactly the verified groups pass, plus dog stand/walk, which passes only because the
  probe resets the label environment (see the dog row).
- **Pixel probe**: with identical state, quadruped walk vs run differs in 331-1453 of 4096
  pixels at 64x64, and dog walk vs trot in 1874-2006, depending on camera.
- A short matching rollout is not proof: in the probe the reacher, finger turn and
  cartpole two_poles labels "matched" only because those sparse rewards stayed zero.
- **Tests** (`tests/collection/test_collection_task_labels.py`), per verified group:
  `DMControlEnv` labels equal each task's own environment stepped with the same seed and
  actions (qpos/qvel/act identical, action-repeat sums including a 2-of-3 final substep,
  truncation at the episode boundary, three episodes across unseeded resets, one started
  from the default pose so every label is non-zero); recording extra labels, or letting a
  different group task drive collection, leaves pixels, rewards, episode ends, the task
  RandomState and the global Python/NumPy/Torch RNG unchanged. Unsupported combinations
  are rejected. A negative control (temporarily allowing point_mass, reacher, finger turn,
  dog, quadruped, cartpole balance/swingup, fish) makes these tests fail.

## Experiment design and Plan2Explore

**Agreed design.** One exploration run per verified group: one dataset and one shared
denoiser, with every task's reward recorded separately. At each saved milestone, the
frozen denoiser and matching dataset train a fresh reward predictor and policy for each
task. Domain and task selection stay configurable. We start with Walker Stand/Walk/Run.
The main experiment covers verified groups only.

**How Plan2Explore compares** (paper and official TF1 code, checked 2026-10-15):

- *Standard benchmark* (20 DMC tasks, Table 1): the official code runs one task per run,
  e.g. `tasks: [walker_walk]`. The run explores without reward, logs its own task reward,
  and trains a reward head that is cut off from the world model. Zero-shot evaluation
  covers that one task.
- *Cheetah transfer experiment* (Section 5.3, Figure 5): one reward-free exploration of
  Cheetah serves several tasks (run forward, run backward, flip forward, flip backward).
  Each task's reward labels the same exploration data, and a policy is trained in
  imagination. This is what our design does, restricted to groups whose tasks share
  exactly the same physics, resets and observations. (Taken from the paper text; not
  traced in the code.)

**Earlier 16-task reference list** (Plan2Explore without acrobot, cheetah, cup,
pendulum): not required. Its tasks in verified groups are walker stand/walk/run, hopper
stand/hop, cartpole balance/balance_sparse and swingup/swingup_sparse. finger, reacher and
quadruped have no verified group (see the table above).

Sources: [paper](https://arxiv.org/abs/2005.05960) (Table 1, Section 5.3, Figure 5,
appendix); [code](https://github.com/ramanans1/plan2explore) (`README.md`,
`plan2explore/scripts/tasks.py`, `scripts/configs.py`, `training/utility.py`).

## Adding a domain

Read the domain source against criteria 1-5, run the pair probe and the group tests, then
add the group to `REWARD_LABEL_TASK_GROUPS` and a row here.
