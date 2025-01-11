from pl_wrappers import MAELightningWrapper
from archive.vit_pytorch.vit_3d import ViT
from archive.vit_pytorch import MAE
import random
import numpy as np
import argparse
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch

torch.set_float32_matmul_precision("high")  # Suppress warning on 4090

from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    EarlyStopping,
)
from pytorch_lightning.loggers import WandbLogger
from archive.utils.utils import parse_custom_args, get_datasets


def get_model(args):
    v = ViT(
        image_size=160,  # image size
        channels=1,
        frames=100,  # number of frames
        image_patch_size=10,  # image patch size
        frame_patch_size=10,  # frame patch size
        num_classes=args["encoder"]["num_classes"],
        dim=args["encoder"]["dim"],
        depth=args["encoder"]["depth"],
        heads=args["encoder"]["heads"],
        mlp_dim=args["encoder"]["mlp_dim"],
        dropout=args["encoder"]["dropout"],
        emb_dropout=args["encoder"]["emb_dropout"],
    )

    mae = MAE(
        encoder=v,
        masking_ratio=args["mae"][
            "masking_ratio"
        ],  # the paper recommended 75% masked patches
        decoder_dim=args["mae"][
            "decoder_dim"
        ],  # paper showed good results with just 512
        decoder_depth=args["mae"]["decoder_depth"],  # anywhere from 1 to 8
    )

    return mae


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        required=False,
        default="configs/default.yaml",
        help="Path to yaml config file",
    )
    args, unk_args = parser.parse_known_args()
    args = parse_custom_args(args, unk_args)

    do_test = False
    if args["data"]["test"]["img_dir"] and args["data"]["test"]["ann_dir"]:
        do_test = True
    print("Will include inference on test set?", do_test)

    # Set seed
    torch.manual_seed(args["training"]["seed"])
    torch.cuda.manual_seed(args["training"]["seed"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(args["training"]["seed"])
    np.random.seed(args["training"]["seed"])

    # Get model
    model = get_model(args)
    model = MAELightningWrapper(torch.compile(model), args)

    # Get datasets
    train_dataset, val_dataset, _ = get_datasets(args, use_sdf=False)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args["training"]["batch_size"],
        shuffle=True,
        pin_memory=True,
        num_workers=args["training"]["num_workers"],
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args["training"]["batch_size"],
        shuffle=False,
        pin_memory=True,
        num_workers=args["training"]["num_workers"],
    )

    # Logging stuff
    lr_monitor = LearningRateMonitor(logging_interval="step")
    checkpoint_callback = ModelCheckpoint(monitor="val/loss", mode="min")
    early_stop_callback = EarlyStopping(
        monitor="val/loss",
        mode="min",
        patience=int(args["lr_scheduler"]["patience"] * 2.2),
        verbose=True,
    )

    wandb_logger = WandbLogger(log_model=True, project="BowelSegmentation")
    wandb_logger.experiment.config.update(args)  # Log yaml file parameters

    # PL trainer
    trainer = pl.Trainer(
        accelerator="auto",
        devices="auto",
        max_epochs=args["training"]["epochs"],
        enable_progress_bar=True,
        log_every_n_steps=1,
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
        logger=wandb_logger,
        precision="16-mixed",
        accumulate_grad_batches=10,
        gradient_clip_val=0.5,
    )

    # Fit the model
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # Make sure everything is uploaded before evaluating, best model might not be uploaded otherwise!
    wandb_logger.experiment.finish()

    # TODO: Evaluate the model
