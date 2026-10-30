import os
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from torch.distributed import init_process_group, destroy_process_group
import torch.multiprocessing as mp

from exploration_trainer import ExplorationTrainer, check_no_run_outputs, default_run_dir
from utils import setup_visible_cuda_devices, skip_if_run_is_over


OmegaConf.register_new_resolver("eval", eval, replace=True)
# hydra.run.dir default (config/exploration.yaml): outputs/<method label>/<domain>/seed_<exploration seed>
OmegaConf.register_new_resolver("exploration_run_dir", default_run_dir, replace=True)


@hydra.main(config_path="../config", config_name="exploration", version_base="1.3")
def main(cfg: DictConfig) -> None:
    if not cfg.common.resume:  # a new run never reuses a run folder, finished (.run_is_over) or not
        check_no_run_outputs(Path.cwd())
    setup_visible_cuda_devices(cfg.common.devices)
    world_size = torch.cuda.device_count()
    root_dir = Path(hydra.utils.get_original_cwd())
    if world_size < 2:
        run(cfg, root_dir)
    else:
        mp.spawn(main_ddp, args=(world_size, cfg, root_dir), nprocs=world_size)


def main_ddp(rank: int, world_size: int, cfg: DictConfig, root_dir: Path) -> None:
    setup_ddp(rank, world_size)
    run(cfg, root_dir)
    destroy_process_group()


@skip_if_run_is_over
def run(cfg: DictConfig, root_dir: Path) -> None:
    trainer = ExplorationTrainer(cfg, root_dir)
    trainer.run()


def setup_ddp(rank: int, world_size: int) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "6006"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)


if __name__ == "__main__":
    main()
