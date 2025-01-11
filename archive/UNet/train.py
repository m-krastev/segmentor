import argparse
import torch

from archive.UNet.model.pl_wrappers import UnetLightningWrapper
from archive.UNet.model.model import DualHeadUNet3D
from archive.utils.utils import parse_custom_args, set_seed, main_train_loop


def load_unet_model(args, return_class=False, decoder_only=False):
    model = eval(args["model"]["name"])(
        in_channels=1,
        out_channels=1,
        dropout_prob=args["model"]["dropout"],
        f_maps=args["model"]["fmaps"],
        num_levels=args["model"]["num_levels"],
        dual_head=args["model"]["dual_head"],
    )

    # Delete encoder part of decoder
    if decoder_only:
        del model.encoders

    if return_class:
        return UnetLightningWrapper, model

    return UnetLightningWrapper(torch.compile(model), args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        required=False,
        default="configs/default_bowelseg.yaml",
        help="Path to yaml config file",
    )
    args, unk_args = parser.parse_known_args()
    args = parse_custom_args(args, unk_args)

    set_seed(args)
    model = load_unet_model(args)
    main_train_loop(model, args)
