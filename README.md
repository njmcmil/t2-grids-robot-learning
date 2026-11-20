# GridS for LIBERO-Goal

This repository trains and evaluates policies for two simulated LIBERO-Goal tasks: put the bowl on the stove, and put the bowl on the plate. Success is `env.check_success()`.

The compact CNN lives in `t2/`. It is a small shared network, about 71k to 82k parameters, trained from random weights. ResNet-18 through layer 3, with a shared 2-layer transformer, lives in `resnet/`. That policy predicts one new action on every control step. The 8-action scripts look up a stored action sequence. They are not a learned policy.

## Layout

```
t2/                         compact CNN model, data, training, simulation
scripts/                    train and evaluation commands for both packages
configs/                    split, normalization, benchmark, and ResNet configs
notebooks/                  setup walkthrough, then notebooks 01 through 07
runs/<name>/best_model.pt  ten selected compact CNN weight files
results/final/              compact CNN rollout JSON and MP4
results/comparison.csv
results/aggregate.json
results/resnet_heldout/summary.json
results/resnet_protocol/summary.json
results/resnet_film.json
weights/resnet_heldout/     five ResNet held-out weight files
weights/resnet_protocol/    five ResNet seed-0 protocol weight files
weights/resnet_film.pt
results/final/resnet_videos/replay/              three demonstration replays
results/final/resnet_videos/policy/              twelve autonomous ResNet clips
LIBERO/libero/datasets/libero_goal/
```

`configs/final_selection.json` names the ten compact weight files:

```
runs/vision_only_seed0_20261117T213106_238623Z/best_model.pt
runs/vision_only_seed1_20261117T213210_698169Z/best_model.pt
runs/dense_seed0_20261117T211545_858127Z/best_model.pt
runs/dense_seed1_20261117T211813_215453Z/best_model.pt
runs/grids_seed0_20261117T212203_540220Z/best_model.pt
runs/grids_seed1_20261117T212424_721238Z/best_model.pt
runs/grids_film_seed0_20261117T212845_384875Z/best_model.pt
runs/grids_film_seed1_20261117T212956_979682Z/best_model.pt
runs/uniform_seed0_20261117T212318_683813Z/best_model.pt
runs/uniform_seed1_20261117T212534_924986Z/best_model.pt
```

The split seed is 42. Episodes are grouped by initial position, excluding time and velocities, rounded to four decimals: 28 train, 6 validation, 6 test. Groups do not cross splits. The compact CNN pairs observation `t` with `actions[t + 1]` and unscales the action head with the train-only mean and scale in `configs/data_preparation/action_normalization.json`. ResNet-18 pairs the action stored at the restored state and passes the action head through `tanh`.

## Setup

Compact CNN pins are in `requirements-setup.txt` and `libero-version.txt`.

| Package | Version |
| --- | --- |
| Python | 3.10 |
| torch | 2.14.1 |
| torchvision | 0.29.1 |
| mujoco | 2.3.7 |
| robosuite | 1.4.0 |
| gym | 0.25.2 |
| bddl | 1.0.1 |
| h5py | 3.16.0 |
| numpy | 1.22.4 |
| transformers | 4.21.1 |
| LIBERO | `8f1084e3132a39270c3a13ebe37270a43ece2a01` |

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements-setup.txt
```

Import `torch` and `torchvision` before LIBERO. `t2/simulation.py` does that. The recorded compact runs used one RTX 5070, CUDA 13.0, and PyTorch 2.14.1+cu130.

The ResNet-18 code was run with Python 3.10.12, torch 2.5.1+cu121, torchvision 0.20.1+cu121, mujoco 2.3.7, robosuite 1.4.0, gym 0.25.2, bddl 1.0.1, libero 0.1.0, h5py 3.16.0, numpy 2.2.6, and transformers 4.57.6, on one RTX A6000. Headless rendering needs EGL. `resnet/simulate.py` sets `MUJOCO_GL`, `PYOPENGL_PLATFORM`, and `MUJOCO_EGL_DEVICE_ID` when they are unset. Keep `MUJOCO_EGL_DEVICE_ID` aligned with `CUDA_VISIBLE_DEVICES`.

```bash
export CUDA_VISIBLE_DEVICES=0
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID=0
```

## Train

`scripts/train_policy.py` trains one compact variant and writes `runs/<variant>_seed<seed>_<timestamp>/best_model.pt`. Defaults in `t2/train.py` are 10 epochs, batch 32, Adam at learning rate 0.001, and gradient clip 10. The saved file is the lowest validation mean squared error. Variants are `vision_only`, `dense`, `grids`, `grids_film`, and `uniform`.

```bash
python scripts/train_policy.py --variant vision_only --seed 0 --epochs 10
python scripts/train_policy.py --variant vision_only --seed 1 --epochs 10
python scripts/train_policy.py --variant dense --seed 0 --epochs 10
python scripts/train_policy.py --variant dense --seed 1 --epochs 10
python scripts/train_policy.py --variant grids --seed 0 --epochs 10
python scripts/train_policy.py --variant grids --seed 1 --epochs 10
python scripts/train_policy.py --variant grids_film --seed 0 --epochs 10
python scripts/train_policy.py --variant grids_film --seed 1 --epochs 10
python scripts/train_policy.py --variant uniform --seed 0 --epochs 10
python scripts/train_policy.py --variant uniform --seed 1 --epochs 10
```

`--smoke` runs one epoch on 32 examples. There are no pretrained robot-policy weights.

`scripts/train_resnet.py` has four stages. Held-out and protocol training use AdamW at `1e-4`, weight decay `1e-4`, batch 32, and gradient clip 1. `--epochs` defaults to 50. ImageNet-1K initializes ResNet-18 through layer 3. Variants other than `vision_bc` pool the goal-word bytes.

```bash
python scripts/train_resnet.py replay
python scripts/train_resnet.py train --epochs 50
python scripts/train_resnet.py protocol --epochs 50
python scripts/train_resnet.py film --epochs 50
```

`replay` restores demonstration states and does not train. `train` writes `weights/resnet_heldout/<variant>.pt`. `protocol` reads `configs/resnet_protocol.json` and writes `weights/resnet_protocol/<variant>.pt`. `film` trains only `grids_film` with frozen all-MiniLM-L6-v2 and writes `weights/resnet_film.pt`. Those stages do not loop seeds 0 and 1 and do not use Adam at 0.001. The two-seed table below is a recorded evaluation, not a rerun of this script.

On the RTX 5070, each selected compact 10-epoch run took about 67–83 seconds of train plus validation. `grids_film` seed 0 took 73.9 seconds. On an RTX A6000, a ResNet FiLM epoch took about 5.5–6.6 seconds.

## Eval

```bash
python scripts/evaluate_selected.py
```

This loads the ten compact weight files, checks their sha256 values, and rolls each on the 20 benchmark states per task in `configs/evaluation_protocol.json`. The timeout is 300 steps and the success target is 0.5. Outputs go to `results/final/<variant>_seed<seed>/`, `results/comparison.csv`, and `results/aggregate.json`. Completed rollouts are skipped when the JSON already matches the protocol hash and the weight hash.

```bash
python scripts/evaluate_action_metric.py --preflight
python scripts/chunk_retrieval.py --variant grids --seed 0 --paired-40
```

Those two commands retrieve an 8-action sequence from the training demonstrations. They are not a learned policy.

```bash
python scripts/evaluate_resnet.py
python scripts/evaluate_resnet_protocol.py
python scripts/verify_models.py
```

`evaluate_resnet.py` reads `configs/resnet_heldout.json`: demos 30–49, environment seed 0, 600 steps, target 0.20. Ids 30–33 chose the epoch by validation error. Ids 34–49 were not used for training or for choosing weights. It writes `results/resnet_heldout/summary.json` and labeled clips under `results/final/resnet_videos/`. `evaluate_resnet_protocol.py` loads `weights/resnet_protocol/` and the 40 cases in `configs/resnet_protocol.json` (20 starts per task, 300 steps, one new action each step) and writes `results/resnet_protocol/summary.json`. `verify_models.py` checks a finite 7-D action, gradients, the token budget, and a language swap, then prints `All model checks passed. These checks do not establish task success.`

## Demo

`python scripts/evaluate_selected.py` writes an MP4 next to each compact rollout JSON. ResNet clips already in the repo are the three replays in `results/final/resnet_videos/replay/` and the twelve policy clips in `results/final/resnet_videos/policy/`.

```bash
python scripts/watch_resnet.py --size 512 --demo 0
```

That writes `results/final/resnet_videos/replay/watch_put_the_bowl_on_the_stove_demo_0.mp4`, `results/final/resnet_videos/replay/watch_put_the_bowl_on_the_plate_demo_0.mp4`, and `results/final/resnet_videos/replay/watch_both_demo_0.mp4`. Those files are separate from the three labeled replays.

A CPU shape check, with no simulator:

```bash
python - << 'PY'
import torch
from resnet.models import GridSamplerPolicy
from resnet.text import encode_task
torch.manual_seed(0)
policy = GridSamplerPolicy()
images = torch.zeros(1, 2, 3, 128, 128)
proprio = torch.zeros(1, 9)
token_ids = torch.tensor([encode_task("put the bowl on the stove")])
action, coords, shape = policy(images, proprio, token_ids, "grids_film")
print(tuple(action.shape), tuple(shape), tuple(coords.shape))
PY
```

Printed output is `(1, 7) (1, 33, 256) (1, 2, 16, 2)`.

## Notebooks

`notebooks/setup_walkthrough.ipynb` reads `~/.libero/config.yaml`. There are six suites. `libero_goal` has 10 tasks, each with a BDDL file and a `.pruned_init` of 50 states, dimension 79. It renders an initial state.

`notebooks/01_data_preperation.ipynb` uses 50 demos per bowl task, images of 128×128×3 uint8, and 7 finite actions. Observation `t` is paired with action `t+1`. Seed 42 keeps 40 episodes, split 28/6/6, with 10 unused. Initial-position groups stay inside one split. The train-only action mean and standard deviation are fit on 5,442 pairs. Images are scaled to 0–1, stored CHW, and the OpenGL orientation is kept. It saves `split_manifest.json` and `action_normalization.json`, tied by a SHA-256 hash.

`notebooks/02_behaivor_cloning.ipynb` is an earlier pilot, not the final vision-only policy: a 3-layer CNN, a 4×4 pool, and an MLP, with no language and 549,927 parameters. Training was batch 32, 10 epochs, Adam at `1e-3`, and MSE. The best epoch was 7, validation MSE 0.41500, on an RTX 5070, compared with a constant mean action.

`notebooks/03_policy_rollout.ipynb` loads those epoch-7 weights on a stove training scene for 300 steps. Recorded replay from the start failed. The final pre-action state of stove `demo_34` already met success. One development rollout finished the stove task in 80 steps.

`notebooks/04_dense_image_language.ipynb` is the shared compact model: a word embedding plus a GRU, 256 tokens per camera. Seed 0 validation MSE was 0.61968 at epoch 9. Seed 1 was 0.64212 at epoch 10. Both seed-0 training-scene rollouts timed out at 300 steps.

`notebooks/05_grids_and_uniform.ipynb` records GridS seed 0 validation MSE 0.50802 at epoch 9 and uniform seed 0 validation MSE 0.57645 at epoch 10. Both use 16 tokens per camera. It plots coordinates on stored training images.

`notebooks/06_grids_film.ipynb` records GridS-FiLM seed 0 validation MSE 0.51794 at epoch 9 and seed 1 validation MSE 0.61348 at epoch 9. A fresh vision-only run on the same compact net scored 0.61621 and 0.65571. Those two runs are the final compact vision-only policy. Ten runs were reviewed before the test.

`notebooks/07_evaluation_and_comparison.ipynb` froze the success target at 0.50 before the test. The test is 400 rollouts. Means across the two seeds, with the two seed rates in parentheses: uniform 12.5% (15/10), grids 10% (15/5), dense 3.75% (2.5/5), grids_film 3.75% (5/2.5), vision_only 3.75% (7.5/0). Latency met the target. Success did not.

## Results

Compact CNN counts are the `successes` field in `results/comparison.csv`, 20 rollouts per cell. `results/aggregate.json` averages the two tasks inside each seed, then averages seeds 0 and 1. The target is 0.5. Every variant is below it. `latency_target_met` is true against the 250 ms p95 target.

| Variant | Seed 0 plate | Seed 0 stove | Seed 1 plate | Seed 1 stove | Mean of the two seed rates |
| --- | ---: | ---: | ---: | ---: | ---: |
| vision_only | 0/20 | 3/20 | 0/20 | 0/20 | 0.0375 |
| dense | 0/20 | 1/20 | 1/20 | 1/20 | 0.0375 |
| grids | 5/20 | 1/20 | 0/20 | 2/20 | 0.10 |
| grids_film | 0/20 | 2/20 | 0/20 | 1/20 | 0.0375 |
| uniform | 1/20 | 5/20 | 0/20 | 4/20 | 0.125 |

ResNet held-out counts are in `results/resnet_heldout/summary.json`. Seed 0, demos 30–49, 600 steps, target 0.20. `dense_language` on the plate is 3/20 and misses the target. The other nine cells meet it.

| Variant | Stove | Plate | Total |
| --- | ---: | ---: | ---: |
| vision_bc | 4/20 | 13/20 | 17/40 |
| dense_language | 14/20 | 3/20 | 17/40 |
| vanilla_grids | 7/20 | 5/20 | 12/40 |
| grids_film | 19/20 | 5/20 | 24/40 |
| uniform_ablation | 13/20 | 7/20 | 20/40 |

Median action latency there is about 2.4 ms for `vision_bc` and `dense_language`, 2.8 ms for `uniform_ablation`, 2.9 ms for `vanilla_grids`, and 3.1 ms for `grids_film`.

`results/resnet_protocol/summary.json` is training seed 0 only. Each variant has 40 rollouts, timeout 300, and one new action per step.

| Variant | Stove | Plate | Total |
| --- | ---: | ---: | ---: |
| vision_bc | 16/20 | 3/20 | 19/40 |
| dense_language | 8/20 | 2/20 | 10/40 |
| vanilla_grids | 3/20 | 4/20 | 7/40 |
| grids_film | 13/20 | 17/20 | 30/40 |
| uniform_ablation | 8/20 | 1/20 | 9/40 |

The recorded two-seed evaluation used 10 epochs, batch 32, Adam 0.001, seeds 0 and 1, the 300-step benchmark, and one new action each step. Each seed is 40 rollouts. The mean is over 80 trials. `scripts/train_resnet.py` does not regenerate this table.

| Variant | Seed 0 | Seed 1 | Mean over 80 trials |
| --- | ---: | ---: | ---: |
| vision_bc | 22/40 | 13/40 | 35/80 (0.4375) |
| dense_language | 17/40 | 0/40 | 17/80 (0.2125) |
| vanilla_grids | 11/40 | 10/40 | 21/80 (0.2625) |
| grids_film | 18/40 | 10/40 | 28/80 (0.35) |
| uniform_ablation | 6/40 | 1/40 | 7/80 (0.0875) |

`results/resnet_film.json` is frozen MiniLM on the same held-out protocol. Stove 14/20, plate 10/20. Both meet 0.20. Median latency is about 6.5 ms and 6.6 ms. Swapping the two task strings changes the bilinear samples. `weights/resnet_film.pt` does not replace `weights/resnet_heldout/grids_film.pt`.

Open-loop replay can miss `env.check_success()` even when the stored final state succeeds. `results/final/resnet_videos/replay/demonstration_replay_put_the_bowl_on_the_plate_demo_0_seed0.mp4` is that case, and so is stove `demo_34` in the pilot notebook.

## Data

Download `put_the_bowl_on_the_stove_demo.hdf5` and `put_the_bowl_on_the_plate_demo.hdf5` from [yifengzhu-hf/LIBERO-datasets](https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets). These runs load only those two files. The rest of LIBERO-Goal is on the same page.

Place them here for the compact CNN:

```
LIBERO/libero/datasets/libero_goal/put_the_bowl_on_the_stove_demo.hdf5
LIBERO/libero/datasets/libero_goal/put_the_bowl_on_the_plate_demo.hdf5
```

LIBERO itself is `https://github.com/Lifelong-Robot-Learning/LIBERO.git` at commit `8f1084e3132a39270c3a13ebe37270a43ece2a01`. The ResNet dataset and BDDL paths are the constants in `resnet/data.py`. The frozen MiniLM files used by `weights/resnet_film.pt` are `sentence-transformers/all-MiniLM-L6-v2`.
