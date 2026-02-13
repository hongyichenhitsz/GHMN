import os
import argparse
from pathlib import Path
import yaml
import torch
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from models.GHMN_backbone import GHMN


def main(args):

    pl.seed_everything(args.seed, workers=True)

    config_filepath = "configs/GHMN.yaml"  
    with open(config_filepath, "r") as f:
        hyperparams = yaml.load(f, Loader=yaml.FullLoader)
    model_args = hyperparams["model_args"]
    model_args['model_name']=args.model
    data_args = hyperparams["data_args"]
    data_args['vars'] = ['DEWP', 'MAX', 'MIN', 'MXSPD', 'SLP', 'WDSP']
    data_args['predict_vars'] = ['DEWP', 'MAX', 'MIN', 'MXSPD', 'SLP', 'WDSP']

    model = GHMN(model_args=model_args, data_args=data_args)
    model.setup()

    log_dir = Path("logs") / model_args["model_name"]
    log_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_diffusion_loss",
        mode="min",
    )

    early_stop_callback = EarlyStopping(
        monitor="val_diffusion_loss",
        patience=10,
        verbose=True,
        mode="min",
        check_on_train_epoch_end=False
    )

    wandb_logger = WandbLogger(
        project=args.project,
        name=f"{model_args['model_name']}",
        save_dir=str(log_dir),
        config={**model_args, **data_args}  
    )

    trainer = pl.Trainer(
        devices=[args.devices],
        accelerator=args.accelerator,
        strategy="auto",
        max_epochs=model_args["epochs"],
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stop_callback],
    )

    trainer.fit(model)
    trainer.test(model, ckpt_path="best")
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="GHMN")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--project", default="GHMN")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--accelerator", default="gpu")
    args = parser.parse_args()
    main(args)