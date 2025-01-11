import torch
from torch import nn

from monai.networks.blocks import UnetOutBlock
from monai.networks.nets import SwinUNETR

from archive.UNet.model.pl_wrappers import UnetLightningWrapper


class SwinUNETRWrapper(UnetLightningWrapper):
    def __init__(self, model, args):
        super().__init__(model, args)
        self.pad_shape = (1, *model.pad_shape)  # c, d, h, w

    def forward(self, img):
        b, c, d, h, w = img.shape

        # SwinUNETR's input shape needs to be dividable by 2**5, so temporarily pad it
        # TODO: very memory inefficient because of copying, can be prevented by doing it in dataset
        padded_img = torch.zeros(
            (b, *self.pad_shape),
            dtype=img.dtype,
            layout=img.layout,
            device=img.device,
            requires_grad=img.requires_grad,
        )
        padded_img[:, :, :d, :h, :w] = img

        outputs = self.model(padded_img)

        # Unpad the output of the model
        outputs = outputs[:, :, :d, :h, :w]
        return outputs


class SwinUNETRDualHead(SwinUNETR):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sdf_out = nn.Sequential(
            UnetOutBlock(
                spatial_dims=3,
                in_channels=kwargs["feature_size"],
                out_channels=kwargs["feature_size"] // 2,
            ),
            UnetOutBlock(
                spatial_dims=3,
                in_channels=kwargs["feature_size"] // 2,
                out_channels=kwargs["out_channels"],
            ),
        )

    def forward(self, x_in):
        if not torch.jit.is_scripting():
            self._check_input_size(x_in.shape[2:])
        hidden_states_out = self.swinViT(x_in, self.normalize)
        enc0 = self.encoder1(x_in)
        enc1 = self.encoder2(hidden_states_out[0])
        enc2 = self.encoder3(hidden_states_out[1])
        enc3 = self.encoder4(hidden_states_out[2])
        dec4 = self.encoder10(hidden_states_out[4])
        dec3 = self.decoder5(dec4, hidden_states_out[3])
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)
        out = self.decoder1(dec0, enc0)

        # Dual head network
        out_copy = out.clone()
        logits_seg = self.out(out)

        logits_sdf = self.sdf_out(out_copy)

        combined_logits = torch.cat([logits_seg, logits_sdf], dim=1)  # B, 2, D, H, W
        return combined_logits
