"""Unit tests for the GRPO core primitives and trainer loop."""

from __future__ import annotations

import torch

from backend.training.dimension_rewards import DimensionRewardComputer
from backend.training.grpo import (
    LatentGaussianPolicy,
    SampledRollout,
    gaussian_kl,
    group_advantages,
    grpo_policy_loss,
)
from backend.training.grpo_trainer import (
    GRPOPolicy,
    GRPOTrainer,
    align_rtdetr_predictions,
)

# ---------------------------------------------------------------------------
# Core primitives
# ---------------------------------------------------------------------------


def test_group_advantages_is_mean_zero():
    rewards = torch.tensor([0.1, 0.2, 0.3, 0.4])
    adv = group_advantages(rewards)
    assert adv.shape == rewards.shape
    assert abs(adv.mean().item()) < 1e-6


def test_group_advantages_zero_std_returns_zeros():
    rewards = torch.tensor([0.5, 0.5, 0.5])
    adv = group_advantages(rewards)
    assert torch.all(adv == 0.0)


def test_group_advantages_orders_by_reward():
    rewards = torch.tensor([0.0, 1.0, 2.0])
    adv = group_advantages(rewards)
    assert adv[0] < adv[1] < adv[2]


def test_group_advantages_clip_bounds_advantage():
    rewards = torch.tensor([-1.0, 0.0, 0.0, 100.0])
    adv = group_advantages(rewards, clip=2.0)
    assert torch.all(adv <= 2.0 + 1e-6)
    assert torch.all(adv >= -2.0 - 1e-6)


def test_group_advantages_eps_stabilizes_tiny_std():
    # Nearly identical rewards -> tiny std -> the advantage explodes with the default
    # eps, but stays O(1) with a meaningful eps floor.
    rewards = torch.tensor([-0.3001, -0.3000, -0.2999, -0.3000])
    adv_small_eps = group_advantages(rewards, eps=1e-8)
    adv_floor = group_advantages(rewards, eps=1e-2)
    assert adv_small_eps.abs().max() > adv_floor.abs().max() * 10
    assert adv_floor.abs().max() < 1.0


def test_grpo_policy_loss_sign():
    # Positive advantage * positive log-prob -> negative loss (maximizing).
    loss = grpo_policy_loss(
        torch.tensor([1.0, 1.0]),
        torch.tensor([1.0, 1.0]),
    )
    assert loss.item() < 0.0


def test_grpo_policy_loss_detaches_advantage():
    # Gradient must not flow through the advantage (reward) path.
    adv = torch.tensor([1.0, 1.0], requires_grad=True)
    log_probs = torch.tensor([1.0, 1.0], requires_grad=True)
    loss = grpo_policy_loss(log_probs, adv)
    loss.backward()
    assert adv.grad is None
    assert log_probs.grad is not None


def test_gaussian_kl_identity_is_zero():
    kl = gaussian_kl(
        torch.tensor(-2.3),
        torch.tensor(-2.3),
    )
    assert abs(kl.item()) < 1e-6


def test_gaussian_kl_positive_when_different():
    kl = gaussian_kl(
        torch.tensor(-2.3),  # std ~ 0.1
        torch.tensor(-4.6),  # std ~ 0.01
    )
    assert kl.item() > 0.0


def test_latent_policy_sample_shape_and_grad():
    policy = LatentGaussianPolicy(log_std_init=-2.3)
    mean = torch.randn(2, 3, 4)
    z, log_prob = policy.sample(mean)
    assert z.shape == mean.shape
    assert log_prob.dim() == 0  # scalar

    # Gradient flows to log_std, not to mean (mean is not a parameter anyway).
    log_prob.backward()
    assert policy.log_std.grad is not None


def test_latent_policy_kl_penalty():
    policy = LatentGaussianPolicy(log_std_init=-2.3)
    kl = policy.kl_penalty(ref_log_std=-4.6)
    assert kl.item() > 0.0


# ---------------------------------------------------------------------------
# Alignment (Hungarian) with a synthetic matcher
# ---------------------------------------------------------------------------


def test_align_rtdetr_predictions():
    class FakeMatcher:
        def __call__(self, pred_boxes, pred_scores, bboxes, cls, gt_groups):
            # One image, match query 1 -> GT 0 and query 3 -> GT 1.
            return [(torch.tensor([1, 3]), torch.tensor([0, 1]))]

    pred_boxes = torch.randn(1, 5, 4)
    pred_scores = torch.randn(1, 5, 20)
    targets = {
        "cls": torch.tensor([9, 16]),
        "bboxes": torch.tensor([[0.1, 0.1, 0.2, 0.2], [0.3, 0.3, 0.1, 0.1]]),
        "gt_groups": [2],
    }
    aligned = align_rtdetr_predictions(pred_boxes, pred_scores, targets, FakeMatcher())
    assert aligned is not None
    predictions, aligned_targets = aligned
    assert predictions["logits"].shape == (2, 20)
    assert predictions["boxes"].shape == (2, 4)
    assert [t["labels"][0] for t in aligned_targets] == [9, 16]


def test_align_rtdetr_predictions_empty():
    class EmptyMatcher:
        def __call__(self, pred_boxes, pred_scores, bboxes, cls, gt_groups):
            return [(torch.tensor([]), torch.tensor([]))]

    aligned = align_rtdetr_predictions(
        torch.randn(1, 5, 4),
        torch.randn(1, 5, 20),
        {"cls": torch.tensor([]), "bboxes": torch.zeros(0, 4), "gt_groups": [0]},
        EmptyMatcher(),
    )
    assert aligned is None


# ---------------------------------------------------------------------------
# Trainer loop (fake policy)
# ---------------------------------------------------------------------------


class _FakePolicy(GRPOPolicy):
    """Deterministic-shape fake policy for exercising the loop end-to-end."""

    def __init__(self, latent: LatentGaussianPolicy, num_groups: int = 4):
        super().__init__()
        self.latent = latent
        self.num_groups = num_groups

    def sample_group(self, images, targets):
        rollouts = []
        for _ in range(self.num_groups):
            mean = torch.randn(3, 8)  # fake latent content
            z, log_prob = self.latent.sample(mean)
            boxes = z[:, :4].unsqueeze(0)  # (1, 3, 4)
            scores = torch.randn(1, 3, 20)
            rollouts.append(
                SampledRollout(log_prob=log_prob, boxes=boxes, scores=scores)
            )
        return rollouts

    def kl_penalty(self):
        return self.latent.kl_penalty(ref_log_std=-4.6)


def _fake_targets(n=3):
    return {
        "cls": torch.tensor([9, 16, 15][:n]),
        "bboxes": torch.tensor(
            [[0.1, 0.1, 0.2, 0.2], [0.3, 0.3, 0.1, 0.1], [0.5, 0.5, 0.2, 0.2]][:n]
        ),
        "gt_groups": [n],
    }


def test_trainer_step_moves_log_std():
    torch.manual_seed(0)
    latent = LatentGaussianPolicy(log_std_init=-2.3)
    policy = _FakePolicy(latent)
    reward_computer = DimensionRewardComputer(head_type="sigmoid")
    optimizer = torch.optim.Adam(latent.parameters(), lr=0.1)
    trainer = GRPOTrainer(
        policy,
        reward_computer,
        optimizer,
        num_groups=4,
        beta_kl=0.0,
        beta_reward=0.0,
        align_fn=None,  # index alignment
    )
    images = torch.randn(1, 3, 64, 64)
    before = latent.log_std.item()
    for _ in range(10):
        trainer.train_step(images, _fake_targets())
    after = latent.log_std.item()
    # The likelihood-ratio score yields a genuine (non-float-noise) gradient,
    # so log_std must move by a measurable amount over 10 steps.
    assert abs(after - before) > 1e-4


def test_trainer_step_reports_metrics():
    torch.manual_seed(0)
    latent = LatentGaussianPolicy()
    policy = _FakePolicy(latent)
    reward_computer = DimensionRewardComputer(head_type="sigmoid")
    optimizer = torch.optim.Adam(latent.parameters(), lr=0.1)
    trainer = GRPOTrainer(
        policy, reward_computer, optimizer, num_groups=4, beta_kl=0.01
    )
    metrics = trainer.train_step(torch.randn(1, 3, 64, 64), _fake_targets())
    assert "loss" in metrics
    assert "mean_reward" in metrics
    assert len(metrics["advantages"]) == 4


def test_trainer_step_with_beta_reward_runs():
    torch.manual_seed(0)
    latent = LatentGaussianPolicy()
    policy = _FakePolicy(latent)
    reward_computer = DimensionRewardComputer(head_type="sigmoid")
    optimizer = torch.optim.Adam(latent.parameters(), lr=0.1)
    trainer = GRPOTrainer(
        policy,
        reward_computer,
        optimizer,
        num_groups=4,
        beta_kl=0.0,
        beta_reward=0.5,
    )
    metrics = trainer.train_step(torch.randn(1, 3, 64, 64), _fake_targets())
    assert "loss" in metrics
