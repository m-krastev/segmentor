import unittest

import torch
from tensordict import TensorDict
from tensordict.nn import set_composite_lp_aggregate
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE

from navigator.config import Config
from navigator.models import create_ppo_modules
from navigator.models.actor import ActorNetwork
from navigator.train import log_tensorboard


class NavigatorPpoSmokeTest(unittest.TestCase):
    def test_tensorboard_logger_only_writes_scalars(self):
        class Writer:
            def __init__(self):
                self.scalars = []

            def add_scalar(self, key, value, global_step):
                self.scalars.append((key, value, global_step))

        writer = Writer()
        log_tensorboard(
            writer,
            {
                "losses/policy_loss": torch.tensor(1.25),
                "validation/success_rate": 0.5,
                "ignored/vector": torch.tensor([1.0, 2.0]),
            },
            step=128,
        )

        self.assertEqual(
            writer.scalars,
            [
                ("losses/policy_loss", 1.25, 128),
                ("validation/success_rate", 0.5, 128),
            ],
        )

    def test_goal_prior_is_opt_in(self):
        config = Config(
            device="cpu",
            patch_size_mm=8,
            voxel_size_mm=1.0,
        )
        self.assertEqual(config.goal_action_prior, 0.0)

    def test_goal_prior_centers_initial_policy_on_endpoint_direction(self):
        actor = ActorNetwork(
            input_channels=4,
            context_features=5,
            goal_action_prior=8.0,
        )
        for parameter in actor.parameters():
            parameter.data.zero_()

        observation = torch.zeros(1, 4, 8, 8, 8)
        context = torch.tensor([[0.0, 0.0, 1.0, 0.0, 0.0]])
        alpha, beta = actor(observation, context)
        mode = (alpha - 1.0) / (alpha + beta - 2.0)

        self.assertGreater(mode[0, 0].item(), 0.9)
        self.assertAlmostEqual(mode[0, 1].item(), 0.5, places=5)
        self.assertAlmostEqual(mode[0, 2].item(), 0.5, places=5)

    def test_rollout_shaped_batch_supports_ppo_backward(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config = Config(
            device=str(device),
            patch_size_mm=8,
            voxel_size_mm=1.0,
        )
        set_composite_lp_aggregate(False).set()
        policy, value = create_ppo_modules(config, device)
        batch_size = torch.Size([2, 3])

        rollout = TensorDict(
            {
                "actor": torch.randn(
                    *batch_size,
                    config.observation_channels,
                    *config.patch_size_vox,
                    device=device,
                ),
                "context": torch.randn(
                    *batch_size,
                    config.context_features,
                    device=device,
                ),
                "next": TensorDict(
                    {
                        "actor": torch.randn(
                            *batch_size,
                            config.observation_channels,
                            *config.patch_size_vox,
                            device=device,
                        ),
                        "context": torch.randn(
                            *batch_size,
                            config.context_features,
                            device=device,
                        ),
                        "reward": torch.randn(*batch_size, 1, device=device),
                        "done": torch.zeros(*batch_size, 1, dtype=torch.bool, device=device),
                        "terminated": torch.zeros(*batch_size, 1, dtype=torch.bool, device=device),
                    },
                    batch_size=batch_size,
                    device=device,
                ),
            },
            batch_size=batch_size,
            device=device,
        )

        with torch.no_grad():
            policy(rollout)

        advantage = GAE(
            gamma=config.gamma,
            lmbda=config.gae_lambda,
            value_network=value,
            average_gae=True,
        )
        advantage(rollout)

        loss_module = ClipPPOLoss(
            actor_network=policy,
            critic_network=value,
            clip_epsilon=config.clip_epsilon,
            entropy_coeff=config.ent_coef,
            entropy_bonus=True,
            critic_coeff=config.vf_coef,
            loss_critic_type="smooth_l1",
            normalize_advantage=False,
        )
        losses = loss_module(rollout.reshape(-1))
        total_loss = losses["loss_objective"] + losses["loss_entropy"] + losses["loss_critic"]
        total_loss.backward()

        self.assertEqual(rollout["action"].shape, torch.Size([*batch_size, 3]))
        self.assertEqual(rollout["state_value"].shape, torch.Size([*batch_size, 1]))
        self.assertTrue(torch.isfinite(total_loss))
        self.assertTrue(
            all(
                parameter.grad is None or torch.isfinite(parameter.grad).all()
                for parameter in loss_module.parameters()
            )
        )


if __name__ == "__main__":
    unittest.main()
