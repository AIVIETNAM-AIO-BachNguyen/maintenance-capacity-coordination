"""L2 diagnostics around the frozen L1 PPO recipe; no new learning objective."""
from __future__ import annotations

import math
import numpy as np
import torch
from torch import nn

from .jcst_learner_v1 import Encoding, Policy, Transition, tensors


def policy_diagnostic(policy: Policy, encoded: Encoding, *, deterministic: bool,
                      generator: torch.Generator | None = None) -> dict:
    c, w, flat, mask = tensors([encoded])
    with torch.no_grad():
        dist = policy.distribution(c, w, flat, mask)
        probabilities = dist.probs[0]
        legal = mask[0]
        ranked = torch.topk(dist.logits[0][legal], min(2, int(legal.sum())))
        ranked_prob = torch.topk(probabilities[legal], min(2, int(legal.sum())))
        if deterministic:
            index = int(dist.logits.argmax(dim=-1).item())
        else:
            if generator is None:
                raise ValueError("stochastic evaluation needs a separate action generator")
            index = int(torch.multinomial(probabilities, 1, generator=generator).item())
        logit_gap = float(ranked.values[0] - ranked.values[1]) if len(ranked.values) > 1 else 1e9
        probability_gap = (float(ranked_prob.values[0] - ranked_prob.values[1])
                           if len(ranked_prob.values) > 1 else 1e9)
        return {
            "index": index, "logprob": float(dist.log_prob(torch.tensor([index])).item()),
            "value": float(policy.value(flat).item()),
            "entropy": float(dist.entropy().item()),
            "wait_probability": float(probabilities[0].item()),
            "chosen_probability": float(probabilities[index].item()),
            "greedy_probability": float(probabilities.max().item()),
            "logit_gap": logit_gap, "probability_gap": probability_gap,
            "legal_count": int(legal.sum().item()),
        }


def _gradient_norm(parameters) -> float:
    return math.sqrt(sum(float(torch.sum(p.grad.detach().square()).item())
                         for p in parameters if p.grad is not None))


def _update_norm(parameters, before) -> float:
    return math.sqrt(sum(float(torch.sum((p.detach() - old).square()).item())
                         for p, old in zip(parameters, before)))


def _parameter_norm(parameters) -> float:
    return math.sqrt(sum(float(torch.sum(p.detach().square()).item()) for p in parameters))


def ppo_update_diagnostic(policy: Policy, optimizer: torch.optim.Optimizer,
                          data: list[Transition], *, epochs: int = 4,
                          minibatch: int = 128, lam: float = .95,
                          clip: float = .2, entropy_coef: float = .01) -> dict[str, float]:
    """Numerically identical update order to L1, with fixed sampled telemetry."""
    n = len(data)
    adv = np.zeros(n, dtype=np.float32)
    running = next_value = 0.0
    for i in range(n - 1, -1, -1):
        t = data[i]
        continuation = 0.0 if t.done else 1.0
        delta = t.reward + continuation * next_value - t.value
        running = delta + lam * continuation * running
        adv[i] = running
        next_value = t.value
    returns = adv + np.asarray([t.value for t in data], dtype=np.float32)
    normalized = (adv - adv.mean()) / (adv.std() + 1e-8)
    c, w, flat, mask = tensors([t.encoded for t in data])
    actions = torch.as_tensor([t.action_index for t in data], dtype=torch.long)
    old_lp = torch.as_tensor([t.logprob for t in data], dtype=torch.float32)
    target = torch.as_tensor(returns, dtype=torch.float32)
    advantage = torch.as_tensor(normalized, dtype=torch.float32)
    actor_parameters = list(policy.actor.parameters())
    critic_parameters = list(policy.critic.parameters())
    values: dict[str, list[float]] = {name: [] for name in (
        "loss", "actor_loss", "critic_loss", "entropy", "approx_kl", "clip_fraction",
        "actor_gradient_pre", "critic_gradient_pre", "actor_gradient_post",
        "critic_gradient_post", "actor_update", "critic_update",
        "actor_relative_update", "critic_relative_update")}
    for _ in range(epochs):
        for indices in torch.randperm(n).split(minibatch):
            dist = policy.distribution(c[indices], w[indices], flat[indices], mask[indices])
            new_lp = dist.log_prob(actions[indices])
            logratio = new_lp - old_lp[indices]
            ratio = torch.exp(logratio)
            surrogate = torch.minimum(ratio * advantage[indices],
                                      ratio.clamp(1 - clip, 1 + clip) * advantage[indices])
            value = policy.value(flat[indices])
            actor_loss = -surrogate.mean()
            critic_loss = .5 * (value - target[indices]).square().mean()
            entropy = dist.entropy().mean()
            loss = actor_loss + critic_loss - entropy_coef * entropy
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite PPO loss")
            old_actor = [p.detach().clone() for p in actor_parameters]
            old_critic = [p.detach().clone() for p in critic_parameters]
            actor_norm = _parameter_norm(actor_parameters)
            critic_norm = _parameter_norm(critic_parameters)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            values["actor_gradient_pre"].append(_gradient_norm(actor_parameters))
            values["critic_gradient_pre"].append(_gradient_norm(critic_parameters))
            nn.utils.clip_grad_norm_(policy.parameters(), .5)
            values["actor_gradient_post"].append(_gradient_norm(actor_parameters))
            values["critic_gradient_post"].append(_gradient_norm(critic_parameters))
            optimizer.step()
            actor_update = _update_norm(actor_parameters, old_actor)
            critic_update = _update_norm(critic_parameters, old_critic)
            values["actor_update"].append(actor_update)
            values["critic_update"].append(critic_update)
            values["actor_relative_update"].append(actor_update / max(actor_norm, 1e-12))
            values["critic_relative_update"].append(critic_update / max(critic_norm, 1e-12))
            values["loss"].append(float(loss.detach()))
            values["actor_loss"].append(float(actor_loss.detach()))
            values["critic_loss"].append(float(critic_loss.detach()))
            values["entropy"].append(float(entropy.detach()))
            values["approx_kl"].append(float(((ratio - 1) - logratio).mean().detach()))
            values["clip_fraction"].append(float(((ratio - 1).abs() > clip).float().mean().detach()))
    residual = target.numpy() - np.asarray([t.value for t in data], dtype=np.float32)
    variance = float(np.var(target.numpy()))
    result = {key: float(np.mean(value)) for key, value in values.items()}
    result.update(advantage_std=float(adv.std()), advantage_mean=float(adv.mean()),
                  return_mean=float(returns.mean()), return_std=float(returns.std()),
                  critic_prediction_mse=float(np.mean(residual**2)),
                  explained_variance=float(1 - np.var(residual) / variance) if variance > 0 else 0.0)
    return result
