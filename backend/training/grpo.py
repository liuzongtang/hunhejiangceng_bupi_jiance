"""
Minimal GRPO (Group Relative Policy Optimization) primitives for detection.

GRPO replaces PPO's value-network baseline with a *group-relative* baseline: for
each image, sample ``G`` candidate detection sets, score them, and use the mean
and std of those ``G`` rewards as the baseline:

    A_g = (R_g - mean_g) / (std_g + eps)

The policy objective is the REINFORCE estimator on the latent action's log-prob:

    L_policy = -mean_g (A_g * log_prob_g)

plus an optional KL penalty toward a frozen reference policy (same latent mean,
a fixed reference noise scale) that stops the exploration temperature from
collapsing to zero (which would make the policy deterministic and kill the
group's variance) or exploding.

``LatentGaussianPolicy`` samples ``z ~ N(mean, exp(log_std))`` and reports the
*likelihood-ratio* log-density of the fixed sample. Critically, the log-density
is differentiated with the sampled ``z`` held fixed (``z.detach()``), NOT through
the reparametrization ``z = mean + std*eps``. The reparametrization view would
give a constant score ``d log pi / d log_std = -1`` and a zero REINFORCE gradient
(since advantages are mean-zero); the likelihood-ratio view gives the correct
score ``d log pi / d log_std = eps^2 - 1`` and ``d log pi / d mean = eps/std``,
which correlate with the reward and yield non-zero gradients for BOTH the noise
scale and the (non-detached) mean.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class SampledRollout:
    """One sampled rollout of a latent Gaussian policy.

    Attributes:
        log_prob: Scalar log-density of the sampled latent action, averaged over
            the latent dimensions; differentiable w.r.t. the policy parameters
            (``log_std`` and, if not detached, the latent ``mean``).
        boxes: Predicted boxes ``(B, N, 4)`` (xywh) for this rollout.
        scores: Predicted class logits ``(B, N, C)`` for this rollout.
    """

    log_prob: torch.Tensor
    boxes: torch.Tensor
    scores: torch.Tensor


def group_advantages(
    rewards: torch.Tensor,
    eps: float = 1e-8,
    clip: Optional[float] = None,
) -> torch.Tensor:
    """
    Compute group-relative advantages ``(R - mean) / (std + eps)``, optionally clipped.

    Args:
        rewards: ``(G,)`` scalar rewards for the group of rollouts.
        eps: Floor on the std denominator. A non-trivial value (e.g. ``1e-2``) keeps
            the advantage from exploding when the group's rewards are nearly
            identical (tiny std), which would otherwise amplify numerical noise into
            a huge pseudo-gradient. When ``std >> eps`` the result is the standard
            unit-variance advantage; when ``std << eps`` it shrinks toward zero.
        clip: If set, clamp advantages to ``[-clip, clip]`` as a final safety net.

    Returns:
        ``(G,)`` advantages, mean-zero before clipping (clipping may shift the mean).
    """
    mean = rewards.mean()
    std = rewards.std()
    adv = (rewards - mean) / (std + eps)
    if clip is not None:
        adv = adv.clamp(-clip, clip)
    return adv


def grpo_policy_loss(
    log_probs: torch.Tensor,
    advantages: torch.Tensor,
) -> torch.Tensor:
    """
    REINFORCE policy loss ``-mean(A * log_prob)``.

    Args:
        log_probs: ``(G,)`` per-rollout latent log-probabilities.
        advantages: ``(G,)`` group-relative advantages. Detached here — treated
            as constants, so no gradient flows through the reward.

    Returns:
        Scalar loss, differentiable w.r.t. the policy parameters via ``log_probs``.
    """
    return -(advantages.detach() * log_probs).mean()


def gaussian_kl(
    log_std: torch.Tensor,
    ref_log_std: torch.Tensor,
) -> torch.Tensor:
    """
    Per-dimension KL(N(0, std^2) || N(0, ref_std^2)).

    Both Gaussians share the same (deterministic) latent mean; only the noise
    scale differs between the policy and the reference. Closed form per
    dimension: ``log(ref_std / std) + (std^2 + 0) / (2 ref_std^2) - 1/2``.

    Args:
        log_std: Policy's log noise scale (scalar).
        ref_log_std: Reference policy's fixed log noise scale (scalar).

    Returns:
        Non-negative scalar (per-dimension) KL divergence.
    """
    std = log_std.exp()
    ref_std = ref_log_std.exp()
    return ref_log_std - log_std + (std * std) / (2.0 * ref_std * ref_std) - 0.5


class LatentGaussianPolicy(nn.Module):
    """
    A scalar-temperature latent Gaussian policy: ``z ~ N(mean, exp(log_std))``.

    The deterministic latent ``mean`` is supplied per-call by the model; only
    ``log_std`` (the exploration temperature) is a learned parameter here. The
    returned ``log_prob`` uses the likelihood-ratio score of the fixed sample,
    so REINFORCE yields a non-zero gradient w.r.t. ``log_std`` (and, if ``mean``
    is not detached, w.r.t. the mean).
    """

    def __init__(self, log_std_init: float = -2.3):  # std ~ 0.1
        super().__init__()
        self.log_std = nn.Parameter(torch.tensor(log_std_init, dtype=torch.float32))

    def sample(self, mean: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample ``z = mean + std * eps`` and return ``(z, log_prob)``.

        Args:
            mean: Deterministic latent ``(...,)`` tensor (the policy mean).

        Returns:
            ``(z, log_prob)`` where ``z`` has ``mean``'s shape and ``log_prob``
            is a scalar (mean over dims) differentiable w.r.t. ``log_std`` and,
            if ``mean`` carries a graph, w.r.t. the mean.
        """
        std = self.log_std.exp()
        eps = torch.randn_like(mean)
        z = mean + std * eps
        # Likelihood-ratio score: differentiate the log-density of the FIXED
        # sample z (z.detach()), not the reparametrization. Numerically
        # (z - mean)/std == eps, but autograd treats z as constant, yielding
        # d log_prob/d log_std = mean(eps^2) - 1 (non-constant, epsilon-dependent).
        resid = (z.detach() - mean) / std
        log_prob = -0.5 * resid.square().mean() - (
            self.log_std + 0.5 * math.log(2.0 * math.pi)
        )
        return z, log_prob

    def kl_penalty(self, ref_log_std: float) -> torch.Tensor:
        """Per-dimension KL toward a reference policy with the same mean and a fixed noise scale."""
        ref = torch.tensor(
            ref_log_std, dtype=self.log_std.dtype, device=self.log_std.device
        )
        return gaussian_kl(self.log_std, ref)
