import unittest

import torch
from tensordict import TensorDict
from tensordict.nn import set_composite_lp_aggregate
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from torchrl.envs.utils import ExplorationType
from torchrl.modules import set_recurrent_mode

from navigator.config import Config
from navigator.models import FeasibleCategorical, create_ppo_modules
from navigator.models.actor import ActorNetwork
from navigator.pretrain import _behavior_cloning_action
from navigator.train import (
    advance_periodic_threshold,
    deterministic_exploration_type,
    log_tensorboard,
    recurrent_minibatches,
    should_run_final_validation,
    validation_rank,
)


class NavigatorPpoSmokeTest(unittest.TestCase):
    def test_feasible_categorical_has_exact_masked_probabilities(self):
        logits = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
        mask = torch.tensor([[True, False, True, False]])
        distribution = FeasibleCategorical(logits, mask)

        expected = torch.softmax(torch.tensor([0.0, 2.0]), dim=0)
        torch.testing.assert_close(
            distribution.probs[0, [0, 2]],
            expected,
        )
        self.assertEqual(distribution.probs[0, 1].item(), 0.0)
        self.assertEqual(distribution.probs[0, 3].item(), 0.0)
        self.assertEqual(distribution.mode.item(), 2)
        self.assertAlmostEqual(
            distribution.log_prob(torch.tensor([2])).item(),
            expected[1].log().item(),
            places=6,
        )
        samples = distribution.sample((1024,))
        self.assertTrue(torch.isin(samples, torch.tensor([0, 2])).all())
        with self.assertRaisesRegex(ValueError, "at least one"):
            FeasibleCategorical(logits, torch.zeros_like(mask))

    def test_periodic_threshold_cannot_be_skipped(self):
        due, next_threshold = advance_periodic_threshold(399, 400, 400)
        self.assertFalse(due)
        self.assertEqual(next_threshold, 400)
        due, next_threshold = advance_periodic_threshold(403, 400, 400)
        self.assertTrue(due)
        self.assertEqual(next_threshold, 800)
        due, next_threshold = advance_periodic_threshold(1207, next_threshold, 400)
        self.assertTrue(due)
        self.assertEqual(next_threshold, 1600)

    def test_final_validation_runs_only_for_an_unscored_policy_state(self):
        self.assertFalse(should_run_final_validation(0, None))
        self.assertTrue(should_run_final_validation(65536, 49664))
        self.assertFalse(should_run_final_validation(65536, 65536))

    def test_behavior_cloning_action_statistic_is_explicit(self):
        distribution = type(
            "Distribution",
            (),
            {
                "mean": torch.tensor([0.25]),
                "mode": torch.tensor([0.75]),
            },
        )()
        torch.testing.assert_close(
            _behavior_cloning_action(distribution, "mean"),
            torch.tensor([0.25]),
        )
        torch.testing.assert_close(
            _behavior_cloning_action(distribution, "mode"),
            torch.tensor([0.75]),
        )
        with self.assertRaisesRegex(ValueError, "behavior_cloning_action_statistic"):
            Config(behavior_cloning_action_statistic="median")

    def test_deterministic_action_statistic_is_explicit(self):
        self.assertEqual(
            deterministic_exploration_type(
                Config(deterministic_action_statistic="mean")
            ),
            ExplorationType.MEAN,
        )
        self.assertEqual(
            deterministic_exploration_type(
                Config(deterministic_action_statistic="mode")
            ),
            ExplorationType.MODE,
        )
        with self.assertRaisesRegex(ValueError, "deterministic_action_statistic"):
            Config(deterministic_action_statistic="median")

    def test_categorical_actions_require_recurrent_mode_evaluation(self):
        with self.assertRaisesRegex(ValueError, "recurrent policy"):
            Config(
                action_distribution="categorical",
                deterministic_action_statistic="mode",
            )
        with self.assertRaisesRegex(ValueError, "deterministic mode"):
            Config(
                action_distribution="categorical",
                memory_model="gru",
            )
        with self.assertRaisesRegex(ValueError, "clean bounds-only"):
            Config(
                action_distribution="masked_categorical",
                memory_model="gru",
                deterministic_action_statistic="mode",
            )
        with self.assertRaisesRegex(ValueError, "deterministic mode"):
            Config(
                action_distribution="factorized_categorical",
                memory_model="gru",
            )

    def test_validation_rank_prioritizes_complete_traversal(self):
        incomplete = {
            "validation/traversal_success_rate": 0.0,
            "validation/endpoint_reach_rate": 1.0,
            "validation/avg_dice": 0.9,
            "validation/avg_endpoint_distance_mm": 0.0,
        }
        complete = {
            "validation/traversal_success_rate": 0.1,
            "validation/endpoint_reach_rate": 0.1,
            "validation/avg_dice": 0.4,
            "validation/avg_endpoint_distance_mm": 100.0,
        }

        self.assertGreater(validation_rank(complete), validation_rank(incomplete))

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

    def test_recurrent_minibatches_preserve_sequence_order(self):
        rollout = TensorDict(
            {"step": torch.arange(20)},
            batch_size=[20],
        )
        minibatches = list(
            recurrent_minibatches(
                rollout,
                sequence_length=4,
                batch_size=8,
            )
        )

        self.assertEqual(sum(batch.numel() for batch in minibatches), 20)
        for batch in minibatches:
            differences = batch["step"][:, 1:] - batch["step"][:, :-1]
            self.assertTrue(torch.equal(differences, torch.ones_like(differences)))

    def test_recurrent_actor_critic_share_encoder_and_support_ppo(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        set_composite_lp_aggregate(False).set()

        for memory_model in ("gru", "s5"):
            with self.subTest(memory_model=memory_model):
                config = Config(
                    device=str(device),
                    patch_size_mm=8,
                    voxel_size_mm=1.0,
                    memory_model=memory_model,
                    memory_hidden_size=16,
                    s5_state_size=16,
                    recurrent_sequence_length=4,
                )
                policy, value = create_ppo_modules(config, device)
                shared_parameters = {
                    id(parameter) for parameter in policy.parameters()
                } & {id(parameter) for parameter in value.parameters()}
                self.assertTrue(shared_parameters)

                batch_size = torch.Size([2, 4])
                current = {
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
                    "is_init": torch.zeros(
                        *batch_size,
                        1,
                        dtype=torch.bool,
                        device=device,
                    ),
                }
                current["is_init"][:, 0] = True
                next_data = {
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
                    "is_init": torch.zeros(
                        *batch_size,
                        1,
                        dtype=torch.bool,
                        device=device,
                    ),
                    "reward": torch.randn(*batch_size, 1, device=device),
                    "done": torch.zeros(
                        *batch_size,
                        1,
                        dtype=torch.bool,
                        device=device,
                    ),
                    "terminated": torch.zeros(
                        *batch_size,
                        1,
                        dtype=torch.bool,
                        device=device,
                    ),
                }
                if memory_model == "gru":
                    state_shape = (
                        *batch_size,
                        config.memory_num_layers,
                        config.memory_hidden_size,
                    )
                    current["recurrent_state"] = torch.zeros(
                        state_shape,
                        device=device,
                    )
                    next_data["recurrent_state"] = torch.zeros(
                        state_shape,
                        device=device,
                    )
                else:
                    state_shape = (*batch_size, config.s5_state_size, 2)
                    current["s5_state"] = torch.zeros(state_shape, device=device)
                    next_data["s5_state"] = torch.zeros(state_shape, device=device)

                rollout = TensorDict(
                    current
                    | {
                        "next": TensorDict(
                            next_data,
                            batch_size=batch_size,
                            device=device,
                        )
                    },
                    batch_size=batch_size,
                    device=device,
                )
                with set_recurrent_mode(True):
                    with torch.no_grad():
                        policy(rollout)
                    advantage = GAE(
                        gamma=config.gamma,
                        lmbda=config.gae_lambda,
                        value_network=value,
                        average_gae=True,
                        deactivate_vmap=True,
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
                    losses = loss_module(rollout)
                    total_loss = (
                        losses["loss_objective"]
                        + losses["loss_entropy"]
                        + losses["loss_critic"]
                    )
                total_loss.backward()

                self.assertTrue(torch.isfinite(total_loss))
                self.assertTrue(
                    all(
                        parameter.grad is None
                        or torch.isfinite(parameter.grad).all()
                        for parameter in loss_module.parameters()
                    )
                )

    def test_categorical_recurrent_policy_supports_ppo_backward(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        set_composite_lp_aggregate(False).set()
        config = Config(
            device=str(device),
            patch_size_mm=8,
            voxel_size_mm=1.0,
            memory_model="gru",
            memory_hidden_size=16,
            recurrent_sequence_length=4,
            action_distribution="categorical",
            deterministic_action_statistic="mode",
        )
        policy, value = create_ppo_modules(config, device)
        batch_size = torch.Size([2, 4])
        state_shape = (
            *batch_size,
            config.memory_num_layers,
            config.memory_hidden_size,
        )
        current = {
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
            "is_init": torch.zeros(
                *batch_size,
                1,
                dtype=torch.bool,
                device=device,
            ),
            "recurrent_state": torch.zeros(state_shape, device=device),
        }
        current["is_init"][:, 0] = True
        next_data = {
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
            "is_init": torch.zeros(
                *batch_size,
                1,
                dtype=torch.bool,
                device=device,
            ),
            "recurrent_state": torch.zeros(state_shape, device=device),
            "reward": torch.randn(*batch_size, 1, device=device),
            "done": torch.zeros(
                *batch_size,
                1,
                dtype=torch.bool,
                device=device,
            ),
            "terminated": torch.zeros(
                *batch_size,
                1,
                dtype=torch.bool,
                device=device,
            ),
        }
        rollout = TensorDict(
            current
            | {
                "next": TensorDict(
                    next_data,
                    batch_size=batch_size,
                    device=device,
                )
            },
            batch_size=batch_size,
            device=device,
        )
        with set_recurrent_mode(True):
            with torch.no_grad():
                policy(rollout)
            GAE(
                gamma=config.gamma,
                lmbda=config.gae_lambda,
                value_network=value,
                average_gae=True,
                deactivate_vmap=True,
            )(rollout)
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
            losses = loss_module(rollout)
            total_loss = (
                losses["loss_objective"]
                + losses["loss_entropy"]
                + losses["loss_critic"]
            )
        total_loss.backward()

        self.assertEqual(rollout["action"].shape, batch_size)
        self.assertEqual(rollout["action"].dtype, torch.int64)
        self.assertEqual(
            rollout["logits"].shape,
            torch.Size([*batch_size, config.categorical_action_count]),
        )
        self.assertTrue(torch.isfinite(total_loss))
        self.assertTrue(
            all(
                parameter.grad is None
                or torch.isfinite(parameter.grad).all()
                for parameter in loss_module.parameters()
            )
        )

    def test_masked_categorical_recurrent_policy_supports_exact_ppo_backward(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        set_composite_lp_aggregate(False).set()
        config = Config(
            device=str(device),
            patch_size_mm=8,
            voxel_size_mm=1.0,
            reward_supervised=True,
            memory_model="gru",
            memory_hidden_size=16,
            recurrent_sequence_length=4,
            action_distribution="masked_categorical",
            deterministic_action_statistic="mode",
        )
        policy, value = create_ppo_modules(config, device)
        batch_size = torch.Size([2, 4])
        state_shape = (
            *batch_size,
            config.memory_num_layers,
            config.memory_hidden_size,
        )
        action_mask = torch.ones(
            *batch_size,
            config.categorical_action_count,
            dtype=torch.bool,
            device=device,
        )
        action_mask[..., ::3] = False
        current = {
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
            "action_mask": action_mask,
            "is_init": torch.zeros(
                *batch_size,
                1,
                dtype=torch.bool,
                device=device,
            ),
            "recurrent_state": torch.zeros(state_shape, device=device),
        }
        current["is_init"][:, 0] = True
        next_data = {
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
            "action_mask": action_mask.clone(),
            "is_init": torch.zeros(
                *batch_size,
                1,
                dtype=torch.bool,
                device=device,
            ),
            "recurrent_state": torch.zeros(state_shape, device=device),
            "reward": torch.randn(*batch_size, 1, device=device),
            "done": torch.zeros(
                *batch_size,
                1,
                dtype=torch.bool,
                device=device,
            ),
            "terminated": torch.zeros(
                *batch_size,
                1,
                dtype=torch.bool,
                device=device,
            ),
        }
        rollout = TensorDict(
            current
            | {
                "next": TensorDict(
                    next_data,
                    batch_size=batch_size,
                    device=device,
                )
            },
            batch_size=batch_size,
            device=device,
        )
        with set_recurrent_mode(True):
            with torch.no_grad():
                policy(rollout)
            selected_is_valid = rollout["action_mask"].gather(
                -1,
                rollout["action"].unsqueeze(-1),
            )
            self.assertTrue(selected_is_valid.all())
            exact_log_prob = FeasibleCategorical(
                rollout["logits"],
                rollout["action_mask"],
            ).log_prob(rollout["action"])
            torch.testing.assert_close(
                rollout["action_log_prob"],
                exact_log_prob,
            )
            GAE(
                gamma=config.gamma,
                lmbda=config.gae_lambda,
                value_network=value,
                average_gae=True,
                deactivate_vmap=True,
            )(rollout)
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
            losses = loss_module(rollout)
            total_loss = (
                losses["loss_objective"]
                + losses["loss_entropy"]
                + losses["loss_critic"]
            )
        total_loss.backward()

        self.assertTrue(torch.isfinite(total_loss))
        self.assertTrue(
            all(
                parameter.grad is None
                or torch.isfinite(parameter.grad).all()
                for parameter in loss_module.parameters()
            )
        )

    def test_factorized_categorical_recurrent_policy_supports_ppo_backward(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        set_composite_lp_aggregate(False).set()
        config = Config(
            device=str(device),
            patch_size_mm=8,
            voxel_size_mm=1.0,
            memory_model="gru",
            memory_hidden_size=16,
            recurrent_sequence_length=4,
            action_distribution="factorized_categorical",
            deterministic_action_statistic="mode",
        )
        policy, value = create_ppo_modules(config, device)
        batch_size = torch.Size([2, 4])
        state_shape = (
            *batch_size,
            config.memory_num_layers,
            config.memory_hidden_size,
        )
        current = {
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
            "is_init": torch.zeros(
                *batch_size,
                1,
                dtype=torch.bool,
                device=device,
            ),
            "recurrent_state": torch.zeros(state_shape, device=device),
        }
        current["is_init"][:, 0] = True
        next_data = {
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
            "is_init": torch.zeros(
                *batch_size,
                1,
                dtype=torch.bool,
                device=device,
            ),
            "recurrent_state": torch.zeros(state_shape, device=device),
            "reward": torch.randn(*batch_size, 1, device=device),
            "done": torch.zeros(
                *batch_size,
                1,
                dtype=torch.bool,
                device=device,
            ),
            "terminated": torch.zeros(
                *batch_size,
                1,
                dtype=torch.bool,
                device=device,
            ),
        }
        rollout = TensorDict(
            current
            | {
                "next": TensorDict(
                    next_data,
                    batch_size=batch_size,
                    device=device,
                )
            },
            batch_size=batch_size,
            device=device,
        )
        with set_recurrent_mode(True):
            with torch.no_grad():
                policy(rollout)
            GAE(
                gamma=config.gamma,
                lmbda=config.gae_lambda,
                value_network=value,
                average_gae=True,
                deactivate_vmap=True,
            )(rollout)
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
            losses = loss_module(rollout)
            total_loss = (
                losses["loss_objective"]
                + losses["loss_entropy"]
                + losses["loss_critic"]
            )
        total_loss.backward()

        self.assertEqual(rollout["action"].shape, torch.Size([*batch_size, 3]))
        self.assertEqual(rollout["action"].dtype, torch.int64)
        self.assertEqual(
            rollout["logits"].shape,
            torch.Size(
                [
                    *batch_size,
                    3,
                    config.factorized_axis_action_count,
                ]
            ),
        )
        self.assertEqual(rollout["action_log_prob"].shape, batch_size)
        self.assertTrue(torch.isfinite(total_loss))
        self.assertTrue(
            all(
                parameter.grad is None
                or torch.isfinite(parameter.grad).all()
                for parameter in loss_module.parameters()
            )
        )


if __name__ == "__main__":
    unittest.main()
