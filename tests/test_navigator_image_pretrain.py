import tempfile
import unittest
from pathlib import Path

import torch

from navigator.config import Config
from navigator.models.memory import (
    NavigatorDenoisingAutoencoder,
    NavigatorVisualEncoder,
)


class NavigatorImagePretrainTest(unittest.TestCase):
    def test_autoencoder_preserves_patch_shape(self):
        model = NavigatorDenoisingAutoencoder(input_channels=5)
        patches = torch.rand(2, 5, 16, 16, 16)

        reconstruction = model(patches)

        self.assertEqual(reconstruction.shape, patches.shape)
        self.assertTrue(torch.isfinite(reconstruction).all())

    def test_five_channel_checkpoint_expands_to_policy_path_channel(self):
        source = NavigatorVisualEncoder(
            input_channels=5,
            context_features=0,
            output_features=16,
        )
        with torch.no_grad():
            for index, parameter in enumerate(source.spatial_state_dict().values()):
                parameter.fill_((index + 1) / 10)

        target = NavigatorVisualEncoder(
            input_channels=6,
            context_features=7,
            output_features=32,
        )
        project_before = {
            name: value.clone()
            for name, value in target.state_dict().items()
            if name.startswith("project.")
        }

        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint_path = Path(temporary_directory) / "encoder.pt"
            torch.save(
                {
                    "spatial_encoder_state_dict": source.spatial_state_dict(),
                    "label_files_read": False,
                },
                checkpoint_path,
            )
            target.load_spatial_checkpoint(checkpoint_path)

        source_state = source.spatial_state_dict()
        target_state = target.spatial_state_dict()
        torch.testing.assert_close(
            target_state["conv1.conv.weight"][:, :5],
            source_state["conv1.conv.weight"],
        )
        torch.testing.assert_close(
            target_state["conv1.conv.weight"][:, 5],
            torch.zeros_like(target_state["conv1.conv.weight"][:, 5]),
        )
        for name, source_value in source_state.items():
            if name != "conv1.conv.weight":
                torch.testing.assert_close(target_state[name], source_value)
        for name, value in project_before.items():
            torch.testing.assert_close(target.state_dict()[name], value)

    def test_checkpoint_must_contain_the_complete_spatial_encoder(self):
        target = NavigatorVisualEncoder(
            input_channels=5,
            context_features=0,
            output_features=16,
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint_path = Path(temporary_directory) / "incomplete.pt"
            torch.save(
                {
                    "spatial_encoder_state_dict": {
                        "conv1.conv.weight": target.conv1.conv.weight,
                    }
                },
                checkpoint_path,
            )
            with self.assertRaisesRegex(ValueError, "Incomplete pretrained"):
                target.load_spatial_checkpoint(checkpoint_path)

    def test_pretrained_encoder_requires_shared_recurrent_policy(self):
        with self.assertRaisesRegex(ValueError, "shared recurrent encoder"):
            Config(visual_encoder_checkpoint="encoder.pt")

        config = Config(
            memory_model="gru",
            visual_encoder_checkpoint="encoder.pt",
        )
        self.assertEqual(config.visual_encoder_checkpoint, "encoder.pt")

    def test_immutable_split_manifests_are_paired(self):
        with self.assertRaisesRegex(ValueError, "required together"):
            Config(train_case_ids_file="train.txt")
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            Config(
                train_case_ids_file="train.txt",
                val_case_ids_file="validation.txt",
                nnunet_train_case_ids_file="legacy-train.txt",
                nnunet_val_case_ids_file="legacy-validation.txt",
            )
        with self.assertRaisesRegex(ValueError, "masked direction_length"):
            Config(categorical_deterministic_decoding="direction_marginal_mode")
        with self.assertRaisesRegex(ValueError, "masked categorical"):
            Config(categorical_deterministic_decoding="projected_mean")


if __name__ == "__main__":
    unittest.main()
