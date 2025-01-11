import torch
import pytorch_lightning as pl
from torch import nn
from torch.optim import Adam

from archive.UNet.model.pl_wrappers import UnetLightningWrapper


class MAELightningWrapper(pl.LightningModule):
    def __init__(self, model, args):
        super().__init__()
        self.model = model

        # Add to logging
        self.hparams.model_name = model.__class__.__name__

        self.__device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )  # TODO: should be handled better

        self.optimizer = Adam(self.parameters(), lr=args["optimizer"]["lr"])

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            "min",
            patience=args["lr_scheduler"]["patience"],
            verbose=True,
        )

    def forward(self, img):
        return self.model.forward(img)

    def common_step(self, batch):
        loss, img_recon = self(batch[0])  # [B, C, F, W, H]
        return loss, img_recon

    def training_step(self, batch, batch_idx):
        loss, _ = self.common_step(batch)

        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, img_recon = self.common_step(batch)

        if batch_idx == 0:
            self.logger.log_image(
                key="validation img slice 50",
                images=[batch[0][0][0][50], img_recon[0][50]],
                caption=["Image", "Reconstruced image"],
            )
        self.log("val/loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        loss, _ = self.common_step(batch)

        self.log("test/loss", loss)
        return loss

    def configure_optimizers(self):
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": self.scheduler,
            "monitor": "val/loss",
        }


class ViTSegLightningWrapper(UnetLightningWrapper):
    def __init__(self, model, args, use_sdf):
        super().__init__(model, args, use_sdf=use_sdf)

    def forward(self, img):
        output = self.model.forward(img).cuda()

        # Add dummy dimension for head index
        return output[:, None, ...]


class ViTUnetLightingWrapper(UnetLightningWrapper):
    def __init__(self, mae, decoder, args):
        super().__init__(decoder, args)
        self.mae = mae
        self.decoder = decoder

        # Add to logging
        self.hparams.model_name = mae.__class__.__name__ + decoder.__class__.__name__

        # Delete unused parts
        del self.model
        del self.mae.decoder
        del self.mae.enc_to_dec
        del self.mae.decoder_pos_emb
        del self.mae.to_pixels

        # Use in forward
        self.z_norm = args["enc_to_dec"]["z_norm"]

        # Freeze mae (encoder) part
        for param in list(self.mae.parameters())[: args["encoder"]["freeze_up_to"]]:
            param.requires_grad = False

        def build_block(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2,
            padding=0,
            output_padding=0,
            upsample=None,
            out_size=None,
        ):
            # Groupnorm
            modules = [nn.GroupNorm(8, num_channels=in_channels)]

            # Upsmaple
            conv_in = in_channels

            if upsample == "deconv":
                conv_in = out_channels
                modules.append(
                    nn.ConvTranspose3d(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        output_padding=output_padding,
                    )
                )
            elif upsample == "nearest":
                assert out_size is not None, "Should specify output dimensions"
                modules.append(nn.Upsample(size=out_size, mode="nearest"))
            elif upsample is not None:
                raise NotImplementedError(f"Upsampling {upsample} not implemented")

            # Conv
            modules.append(nn.Conv3d(conv_in, out_channels, 3, padding=1, stride=1))

            # ReLu
            modules.append(nn.ReLU(inplace=True))

            return nn.Sequential(*modules)

        # Build encoder to decoder path
        if args["enc_to_dec"]["upsample"] == "deconv":
            self.z12 = build_block(
                1024,
                1024 // 4,
                padding=(2, 4, 4),
                stride=3,
                output_padding=(0, 1, 1),
                upsample="deconv",
            )
            self.z9 = nn.Sequential(
                build_block(
                    1024,
                    1024 // 8,
                    padding=(2, 4, 4),
                    stride=3,
                    output_padding=(0, 1, 1),
                    upsample="deconv",
                ),
                build_block(1024 // 8, 1024 // 8, upsample="deconv"),
            )
            self.z6 = nn.Sequential(
                build_block(
                    1024,
                    1024 // 8,
                    padding=(2, 4, 4),
                    stride=3,
                    output_padding=(0, 1, 1),
                    upsample="deconv",
                ),
                build_block(1024 // 8, 1024 // 16, upsample="deconv"),
                build_block(1024 // 16, 1024 // 16, upsample="deconv"),
            )
        elif args["enc_to_dec"]["upsample"] == "nearest":
            self.z12 = build_block(
                1024, 1024 // 4, upsample="nearest", out_size=(25, 40, 40)
            )
            self.z9 = build_block(
                1024, 1024 // 8, upsample="nearest", out_size=(50, 80, 80)
            )
            self.z6 = build_block(
                1024, 1024 // 16, upsample="nearest", out_size=(100, 160, 160)
            )
        else:
            raise NotImplementedError(
                f"Upsample {args['enc_to_dec']['upsample']} not implemented"
            )

        self.z3 = build_block(
            1024,
            1024 // 2,
            kernel_size=1,
            stride=0,
            padding=0,
            output_padding=0,
            upsample=None,
        )

    def forward(self, x):
        tokens = self.mae.get_tokens(x, return_tokens_only=True)
        self.mae.encoder.transformer(tokens)

        vit_outputs = self.mae.encoder.transformer.layer_output
        batch_size = vit_outputs[0].shape[0]

        # Normalize vit outputs
        if self.z_norm:
            vit_outputs = [(a - a.mean()) / a.std() for a in vit_outputs]

        # In: [BS, 1024, 10, 16, 16]
        # Out
        # torch.Size([2, 512, 10, 16, 16])      z3
        # torch.Size([2, 256, 25, 40, 40])      z12
        # torch.Size([2, 128, 50, 80, 80])      z9
        # torch.Size([2, 64, 100, 160, 160])    z6

        x = self.z3(vit_outputs[3].reshape((batch_size, 1024, 10, 16, 16)))
        x_sdf = x.clone()

        encoding_features = (
            self.z12(vit_outputs[0].reshape((batch_size, 1024, 10, 16, 16))),
            self.z9(vit_outputs[1].reshape((batch_size, 1024, 10, 16, 16))),
            self.z6(vit_outputs[2].reshape((batch_size, 1024, 10, 16, 16))),
        )

        x_seg = self.decoder.model.decoder_seg(encoding_features, x)
        x_sdf = self.decoder.model.decoder_sdf(encoding_features, x_sdf)
        output = self.decoder.model.combine_output(x_seg, x_sdf)

        return output
