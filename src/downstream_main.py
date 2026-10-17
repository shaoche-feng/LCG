import hydra
from omegaconf import DictConfig, OmegaConf

from downstream_trainer import DownstreamTrainer
from utils import setup_visible_cuda_devices


OmegaConf.register_new_resolver("eval", eval)


@hydra.main(config_path="../config", config_name="downstream", version_base="1.3")
def main(cfg: DictConfig) -> None:
    setup_visible_cuda_devices(cfg.common.devices)
    DownstreamTrainer(cfg).run()


if __name__ == "__main__":
    main()
