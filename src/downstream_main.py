import hydra
from omegaconf import DictConfig, OmegaConf

from downstream_trainer import DownstreamTrainer, evaluate_saved_output
from utils import setup_visible_cuda_devices


OmegaConf.register_new_resolver("eval", eval)


@hydra.main(config_path="../config", config_name="downstream", version_base="1.3")
def main(cfg: DictConfig) -> None:
    setup_visible_cuda_devices(cfg.common.devices)
    mode = cfg.downstream.mode
    if mode == "evaluate":
        evaluate_saved_output(cfg)
    elif mode in ("train", "resume"):
        DownstreamTrainer(cfg, resume=mode == "resume").run()
    else:
        raise ValueError(f"downstream.mode must be train, resume or evaluate, got {mode!r}")


if __name__ == "__main__":
    main()
