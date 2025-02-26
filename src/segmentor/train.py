import argparse
import json

from lightning import LightningModule, Trainer
from torch.nn import functional as F
from torch.optim import AdamW

from utils.utils import create_base

from .model import Model, ModelArgs


class LitModel(LightningModule):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.save_hyperparameters(config)
        self.model = Model(config)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)[:, -1]
        loss = F.mse_loss(y_hat, y)
        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), lr=1e-3)
        return optimizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        required=False,
        default="configs/default.json",
        help="Path to JSON config file",
    )

    parser.add_argument(
        "--compile",
        action="store_true",
        help="Compile the model",
    )

    args, unk_args = parser.parse_known_args()
    with open(args.config_file) as f:
        config = json.load(f)

    model_args = create_base(ModelArgs, config)
    model = LitModel(model_args)
    trainer = Trainer()
    trainer.fit(model)
