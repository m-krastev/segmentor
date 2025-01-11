import argparse
from monai.networks.nets import SwinUNETR

from archive.ViT.pl_wrapper import SwinUNETRWrapper, SwinUNETRDualHead
from archive.utils.utils import parse_custom_args, set_seed, main_train_loop


def load_vit_model(args, return_class=False):
    pad_shape = (128, 160, 160)  # d, h, w (input should be padded for patches)

    # Determine single task or multi task
    model_cls = SwinUNETR
    if args["model"]["dual_head"]:
        model_cls = SwinUNETRDualHead

    model = model_cls(
        img_size=pad_shape,
        in_channels=1,
        out_channels=1,
        depths=args["model"]["depths"],
        num_heads=args["model"]["num_heads"],
        feature_size=args["model"]["feature_size"],
        norm_name=args["model"]["norm_name"],
        drop_rate=args["model"]["drop_rate"],
        attn_drop_rate=args["model"]["attn_drop_rate"],
        dropout_path_rate=args["model"]["dropout_path_rate"],
        normalize=args["model"]["normalize"],
        use_checkpoint=args["model"]["use_checkpoint"],
        downsample=args["model"]["downsample"],
        use_v2=args["model"]["use_v2"],
    )
    model.pad_shape = pad_shape

    if return_class:
        return SwinUNETRWrapper, model

    model = SwinUNETRWrapper(
        model=model,
        args=args,
    )
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        required=False,
        default="configs/defaults.yaml",
        help="Path to yaml config file",
    )
    args, unk_args = parser.parse_known_args()
    args = parse_custom_args(args, unk_args)

    set_seed(args)
    model = load_vit_model(args)
    main_train_loop(model, args)
