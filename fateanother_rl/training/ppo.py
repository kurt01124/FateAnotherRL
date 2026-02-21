"""PPO loss computation for sequence-based training."""

import torch
import torch.nn.functional as F


def ppo_loss(
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    new_values: torch.Tensor,
    returns: torch.Tensor,
    new_entropy: torch.Tensor,
    advantages: torch.Tensor,
    clip_eps: float = 0.2,
    vf_coef: float = 0.5,
    ent_coef: float = 0.01,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute PPO clipped surrogate loss.

    Args:
        new_log_probs: (B, T) new policy log probabilities
        old_log_probs: (B, T) old policy log probabilities (from rollout)
        new_values:    (B, T) new value estimates
        returns:       (B, T) GAE returns (advantage + old_value)
        new_entropy:   (B, T) new policy entropy
        advantages:    (B, T) GAE advantages
        clip_eps:      PPO clipping epsilon
        vf_coef:       Value function loss coefficient
        ent_coef:      Entropy bonus coefficient

    Returns:
        total_loss: scalar
        stats: dict with component losses for logging
    """
    # Advantage normalization (across batch)
    adv = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # Policy loss (clipped surrogate)
    ratio = torch.exp(new_log_probs - old_log_probs)
    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
    policy_loss = -torch.min(surr1, surr2).mean()

    # Value loss (MSE)
    value_loss = F.mse_loss(new_values, returns)

    # Entropy bonus (negative = maximize entropy)
    entropy_loss = -new_entropy.mean()

    # Total loss
    total = policy_loss + vf_coef * value_loss + ent_coef * entropy_loss

    # Stats for logging
    with torch.no_grad():
        approx_kl = ((ratio - 1.0) - ratio.log()).mean().item()
        clip_fraction = ((ratio - 1.0).abs() > clip_eps).float().mean().item()

    stats = {
        "policy_loss": policy_loss.item(),
        "value_loss": value_loss.item(),
        "entropy": new_entropy.mean().item(),
        "entropy_loss": entropy_loss.item(),
        "approx_kl": approx_kl,
        "clip_fraction": clip_fraction,
        "total_loss": total.item(),
    }

    return total, stats


def get_entropy_coef(iteration: int, max_iterations: int = 100000) -> float:
    """Entropy coefficient schedule.

    Start low, decay further over time.
    """
    progress = iteration / max(max_iterations, 1)
    if progress < 0.3:
        return 0.01
    elif progress < 0.7:
        return 0.005
    else:
        return 0.001
