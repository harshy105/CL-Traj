import os
import torch
import subprocess
import time
import numpy as np
import re
import pytorch_lightning as pl

from typing import Optional
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, StochasticWeightAveraging, ModelCheckpoint
from datasets.nuscenes.nuscenes_devkit.eval.prediction.compute_metrics import compute_metrics

from evaluate.evaluate import NuScenesEvaluation
from evaluate.evaluate_deep_scenario import DeepScenarioEvaluation
from network.data_generator import TrajectoryGridDataModule
from network.net import Net
from config.train_config import DataStructureConfig, NetConfig, TrainingConfig
from config.config import SAVE_PATH, LOG_DIR


def train_net(
    model_name: str,
    save_dir: str,
    train_config: TrainingConfig,
    data_config: DataStructureConfig,
    net_config: NetConfig,
    pretrained_ckpt_name: Optional[str] = None,
):

    seed_everything(3407, workers=True)
    data_modules = TrajectoryGridDataModule(data_config=data_config, training_config=train_config)

    if pretrained_ckpt_name is not None:
        pretrained_ckpt_model_name = pretrained_ckpt_name.split("-epoch")[0]
        model = Net.load_from_checkpoint(
            save_dir + pretrained_ckpt_model_name + "/" + pretrained_ckpt_name,
            strict=True,
            data_config=data_config,
            net_config=net_config,
            train_config=train_config,
        )
    else:
        model = Net(data_config, net_config, train_config)

    callbacks = [
        LearningRateMonitor(logging_interval="epoch"),
        MinLREpochStopCallback(train_config.min_lr, patience=train_config.min_lr_patience_epochs),
    ]
    if train_config.use_swa:
        callbacks.append(
            StochasticWeightAveraging(
                swa_lrs=train_config.min_lr,
                swa_epoch_start=train_config.swa_epoch_start,
                annealing_epochs=train_config.swa_annealing_epochs,
            )
        )

    git_hash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode("ascii").strip()
    os.makedirs(name=save_dir + model_name + "/")
    
    if train_config.use_target_net and not train_config.use_scene_net:
        monitor_metric = "val_minADE"
        ckpt_name = (model_name + "-{epoch:02d}-{val_loss:.2f}-{val_minFDE:.2f}-{val_minADE:.2f}-" + git_hash)
    elif not train_config.use_target_net and train_config.use_scene_net:
        monitor_metric = "scene_val_minADE_1"
        ckpt_name = (model_name + "-{epoch:02d}-{scene_val_loss:.2f}-{scene_val_minFDE_1:.2f}-{scene_val_minADE_1:.2f}-" + git_hash)
    elif train_config.use_target_net and train_config.use_scene_net:
        monitor_metric = "val_minADE"
        ckpt_name = (model_name + "-{epoch:02d}-{val_loss:.2f}-{val_minFDE:.2f}-{val_minADE:.2f}-" + git_hash)
    else:
        raise ValueError("None of networks are being trained")
    callbacks.append(
        ModelCheckpoint(
            monitor=monitor_metric,
            dirpath=save_dir + model_name + "/",
            save_top_k=2,
            save_last=True,
            filename=ckpt_name,
            verbose=True,
        )
    )

    np.savez_compressed(f"{save_dir}{model_name}/{model_name}-{git_hash}", [train_config, data_config, net_config])

    trainer = Trainer(
        max_epochs=train_config.max_epochs,
        logger=TensorBoardLogger(LOG_DIR, version=model_name),
        callbacks=callbacks,
        devices=[0],
        accelerator="gpu",
        strategy="ddp",
        reload_dataloaders_every_n_epochs=1 if train_config.data_augmentation == "resample" else 0,
    )
    trainer.fit(model, data_modules)


class MinLREpochStopCallback(pl.Callback):
    def __init__(self, min_lr, patience):
        super().__init__()
        self.min_lr = min_lr
        self.patience = patience
        self.counter = 0

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        current_lr = trainer.optimizers[0].param_groups[0]["lr"]
        if current_lr <= self.min_lr:
            self.counter += 1
        else:
            self.counter = 0
        if self.counter >= self.patience:
            trainer.should_stop = True


if __name__ == "__main__":
    model_name = "v1_nuS"
    data_config = DataStructureConfig()
    net_config = NetConfig()
    train_config = TrainingConfig()
    model_name = f"{time.strftime('%Y-%m-%d-%H-%M-%S')}_{model_name}"
    save_dir = SAVE_PATH
    train_net(model_name, save_dir, train_config, data_config, net_config, train_config.pretrained_ckpt)
    # run quantitative evaluation
    ckpts = [f for f in os.listdir(save_dir + model_name + "/") if (".ckpt" in f) and (model_name in f)]
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using {DEVICE} device")
    best_ckpt = min(
        ckpts,
        key=lambda s: (
            (
                float(re.search(r"val_minADE=(-?[\d.]+)", s).group(1)),
                float(re.search(r"val_minFDE=(-?[\d.]+)", s).group(1)),
                float(re.search(r"val_loss=(-?[\d.]+)", s).group(1)),
            ) if train_config.use_target_net else (
                float(re.search(r"scene_val_minADE_1=(-?[\d.]+)", s).group(1)),
                float(re.search(r"scene_val_minFDE_1=(-?[\d.]+)", s).group(1)),
                float(re.search(r"scene_val_loss=(-?[\d.]+)", s).group(1)),
            )
        )
    )

    data_split = "val"
    for num_eval_recurr_steps in [1, 3]:
        if "deep_scenario" in data_config.db_folder:
            track_eval = DeepScenarioEvaluation(data_split, save_dir, model_name, best_ckpt, num_eval_recurr_steps=num_eval_recurr_steps)
        elif "nuScenes" in data_config.db_folder:
            track_eval = NuScenesEvaluation(data_split, save_dir, model_name, best_ckpt, num_eval_recurr_steps=num_eval_recurr_steps)
        track_eval.quant_evaluate()
        del track_eval
