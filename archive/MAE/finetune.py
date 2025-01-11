import wandb

from archive.UNet.model.pl_wrappers import UnetLightningWrapper
from archive.UNet.train import load_unet_model
from archive.MAE.pl_wrappers import (
    MAELightningWrapper,
    ViTUnetLightingWrapper,
    ViTSegLightningWrapper,
)
from archive.vit_pytorch.vit_3d import ViT
from archive.vit_pytorch import MAE, ViTSeg
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


def get_mae_model(args):
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
        save_layers=args["encoder"]["save_layers"],
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

    api = wandb.Api()
    run = api.run(args["mae"]["run_link"])
    artifact = api.artifact(
        f"{run.entity}/{run.project}/model-{run.id}:best", type="model"
    )
    artifact_dir = artifact.download()
    mae_model = MAELightningWrapper.load_from_checkpoint(
        f"{artifact_dir}/model.ckpt", model=mae, args=args
    )

    # Return only encoder part of MAE model
    return mae_model.model


def load_vitunet_model(args, return_class=False):
    # Load mae from pre-trained
    mae = get_mae_model(args)

    # Move all decoder args to 'model' key
    unet_args = args.copy()
    for k, v in args["decoder"].items():
        unet_args["model"][k] = v

    # Load Unet decoder part
    decoder = load_unet_model(unet_args, decoder_only=True)

    if return_class:
        return ViTUnetLightingWrapper, mae, decoder

    model = ViTUnetLightingWrapper(mae=mae, decoder=decoder, args=args).train()
    return model


if __name__ == "__main__":
    # from archive.UNet.test import evaluate
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        required=False,
        default="configs/finetune_default.yaml",
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

    # Load model
    # mae_model = get_mae_model(args)
    # model = ViTSeg(
    #     encoder=mae_model.encoder,
    #     masking_ratio=0.0,  # No masking in finetuning stage
    #     decoder_dim=512,  # paper showed good results with just 512
    #     decoder_depth=6  # anywhere from 1 to 8
    # )
    # model = ViTSegLightningWrapper(model=torch.compile(model), args=args, use_sdf=False).train()
    # del mae_model
    model = load_vitunet_model(args)

    # Get datasets
    train_dataset, val_dataset, _ = get_datasets(args)

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
    monitor = "train/loss" if args["data"]["sanity_check"] else "val/loss"
    lr_monitor = LearningRateMonitor(logging_interval="step")
    checkpoint_callback = ModelCheckpoint(monitor=monitor, mode="min")
    early_stop_callback = EarlyStopping(
        monitor=monitor,
        mode="min",
        patience=int(args["lr_scheduler"]["patience"] * 2.1),
        verbose=True,
    )

    wandb_logger = WandbLogger(log_model=True, project="BowelSegmentation")
    wandb_logger.experiment.config.update(args)  # Log yaml file parameters

    # Calculate number of training batches
    n_batches = len(train_loader) // args["training"]["batch_size"]

    # PL trainer
    trainer = pl.Trainer(
        max_epochs=args["training"]["epochs"],
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
        logger=wandb_logger,
        precision="16-mixed",
        log_every_n_steps=50 if 50 <= n_batches else n_batches,
    )

    # Fit the model
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # Make sure everything is uploaded before evaluating, best model might not be uploaded otherwise!
    wandb_logger.experiment.finish()

    # Evaluate the model
    # TODO
    # results = evaluate(f"thomasvanorden/{wandb_logger.name}/{wandb_logger.version}", device='cuda', save_results=False, report_instance=False)
