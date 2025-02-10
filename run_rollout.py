import os
import torch
import hydra
import wandb
import pytz
from datetime import datetime
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from template_trainer.utils import (
    set_seed,
)

from tqdm import tqdm

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


@torch.no_grad()
def run_rollout(cfg, model_class, dataset_class, trainer_class):
    """
    Run the training loop.

    Parameters
    ----------
    cfg : DictConfig
        Configuration object containing training parameters.
    """
    set_seed(cfg.base_seed)
    tc_rng = torch.Generator()
    tc_rng.manual_seed(cfg.base_seed)

    print(OmegaConf.to_yaml(cfg))

    if cfg.board:
        wandb.init(
            project=f"{cfg.project}-train",
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    # Model and dataset creation
    model = model_class(cfg.model)

    # Trainer creation
    trainer = trainer_class(model, cfg, tc_rng)
    # restore model
    trainer.restore(cfg.restore_dir, cfg.restore_step)

    # Data loaders creation
    test_datapipe = dataset_class(cfg.dataset, 0, cfg.base_seed, "rollout")
    test_loader = DataLoader(
        test_datapipe,
        batch_size=cfg.batch,
        num_workers=0,
        pin_memory=True,
    )
    # Printing meta info of the training
    time_stamp = datetime.now(pytz.timezone("America/Los_Angeles")).strftime("%Y%m%d-%H%M%S")
    print("stamp: {}".format(time_stamp))

    # Rollout loop starts
    for bi, batch in tqdm(enumerate(test_loader)):
        # to device
        batch = trainer.move_to_device(batch)
        # run rollout for 1 batch
        rollout_res = trainer.rollout(batch)
        # post-process the result
        trainer.post_process_rollout(batch, rollout_res, bi)

    # summarize results
    trainer.summarize_rollout()

    if cfg.board:
        wandb.finish()
