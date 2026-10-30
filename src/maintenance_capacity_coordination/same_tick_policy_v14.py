"""Canonical increasing-edge matching PPO on unchanged N5/K3 physics.

PPO's random action is an ordered sequence. The environment receives only the
canonical final matching. Internal selections are never physical transitions.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from ht_pdm_fjsp.maintenance_dispatch import eligible_edges
from ht_pdm_fjsp.maintenance_framework_v1 import catalogue
from maintenance_capacity_coordination.scaled_coordination_v11 import actor_inputs_scaled, global_dim, local_dim

N_MACHINE, N_WORKER = 5, 3
LOCAL_DIM, GLOBAL_DIM, WORKER_DIM = local_dim(N_WORKER), global_dim(N_MACHINE, N_WORKER), 4

EDGES = tuple((m, j) for m in range(N_MACHINE) for j in range(N_WORKER))
STOP = len(EDGES)
ACTIONS = catalogue(N_MACHINE, N_WORKER)
MAX_TOKENS = N_WORKER + 1


def initial_edge_mask(config, admitted):
    """Present-state queued requests and physical compatibility, no future tape."""
    physical = eligible_edges(config, admitted.physical)
    return np.asarray([
        admitted.requests[m] is not None
        and admitted.requests[m].started_at is None
        and bool(physical[m][j])
        for m, j in EDGES
    ], dtype=bool)


def matching_from_sequence(sequence):
    used_m, used_w, matching = set(), set(), []
    stopped = False
    for token in sequence:
        token = int(token)
        if stopped:
            raise ValueError("token after STOP")
        if token == STOP:
            stopped = True
            continue
        if not 0 <= token < STOP:
            raise ValueError("invalid edge token")
        m, j = EDGES[token]
        if m in used_m or j in used_w:
            raise ValueError("duplicate machine or technician")
        used_m.add(m)
        used_w.add(j)
        matching.append((m, j))
    return tuple(sorted(matching))


def enumerate_sequences(edge_mask, *, canonical=True):
    """All positive-support terminal sequences, for small exact tests."""
    edge_mask = np.asarray(edge_mask, dtype=bool)
    if edge_mask.shape != (STOP,):
        raise ValueError("edge mask shape")
    results = []

    def visit(prefix, used_m, used_w, last_edge):
        options = [token for token, (m, j) in enumerate(EDGES)
                   if edge_mask[token] and m not in used_m and j not in used_w
                   and (not canonical or token > last_edge)]
        if not options:
            results.append(prefix)
            return
        results.append((*prefix, STOP))
        for token in options:
            m, j = EDGES[token]
            visit((*prefix, token), used_m | {m}, used_w | {j}, token)

    visit((), set(), set(), -1)
    return tuple(results)


class ContextActor(nn.Module):
    """C14/D14 have identical modules; only the context refresh switch differs."""

    def __init__(self, direct=False, canonical=True):
        super().__init__()
        self.direct = direct
        self.canonical = canonical
        if direct:
            self.edge_encoder = nn.Sequential(nn.Linear(LOCAL_DIM + WORKER_DIM + 2, 72),
                                              nn.Tanh(), nn.Linear(72, 32), nn.Tanh())
        else:
            self.machine_encoder = nn.Sequential(nn.Linear(LOCAL_DIM, 32), nn.Tanh(),
                                                 nn.Linear(32, 16), nn.Tanh())
            self.worker_encoder = nn.Sequential(nn.Linear(WORKER_DIM, 16), nn.Tanh(),
                                                nn.Linear(16, 16), nn.Tanh())
            self.edge_encoder = nn.Sequential(nn.Linear(34, 48), nn.Tanh(),
                                              nn.Linear(48, 32), nn.Tanh())
        # Selected provisional edges, candidate-specific residual mean/max, count.
        self.context = nn.Sequential(nn.Linear(97, 48), nn.Tanh(),
                                     nn.Linear(48, 32), nn.Tanh())
        self.edge_score = nn.Sequential(nn.Linear(64, 32), nn.Tanh(), nn.Linear(32, 1))
        self.stop_score = nn.Sequential(nn.Linear(32, 16), nn.Tanh(), nn.Linear(16, 1))
        self.register_buffer("machine_id", torch.as_tensor([m for m, _ in EDGES]), persistent=False)
        self.register_buffer("worker_id", torch.as_tensor([j for _, j in EDGES]), persistent=False)

    def encode(self, machine, worker):
        m, w = self.machine_id, self.worker_id
        duration = machine[:, m, 14 + w].unsqueeze(-1)
        restoration = machine[:, m, 14 + N_WORKER + w].unsqueeze(-1)
        if self.direct:
            machine_rep, worker_rep = machine, worker
            edge_input = torch.cat((machine[:, m], worker[:, w], duration, restoration), -1)
        else:
            machine_rep = self.machine_encoder(machine)
            worker_rep = self.worker_encoder(worker)
            edge_input = torch.cat((machine_rep[:, m], worker_rep[:, w], duration, restoration), -1)
        return machine_rep, worker_rep, self.edge_encoder(edge_input)

    def residual_summaries(self, edges, feasible):
        """Current and candidate-after-choice feasible edge embeddings."""
        batch = edges.shape[0]
        count = feasible.sum(1)
        current = feasible.to(edges.dtype)
        current_mean = (edges * current[..., None]).sum(1) / count.clamp(min=1)[..., None]
        current_max = edges.masked_fill(~feasible[..., None], -1e9).amax(1)
        current_max = torch.where((count > 0)[:, None], current_max, torch.zeros_like(current_max))
        different = ((self.machine_id[:, None] != self.machine_id[None, :])
                     & (self.worker_id[:, None] != self.worker_id[None, :]))
        residual = feasible[:, None, :] & different[None, :, :]
        if self.canonical:
            index = torch.arange(STOP, device=feasible.device)
            residual = residual & (index[None, :] > index[:, None])[None, :, :]
        residual_count = residual.sum(2)
        weights = residual.to(edges.dtype)
        residual_mean = torch.einsum("bej,bjd->bed", weights, edges) / residual_count.clamp(min=1)[..., None]
        residual_max = edges[:, None, :, :].expand(batch, STOP, STOP, edges.shape[-1])\
            .masked_fill(~residual[..., None], -1e9).amax(2)
        residual_max = torch.where((residual_count > 0)[..., None], residual_max,
                                   torch.zeros_like(residual_max))
        return (current_mean, current_max, count.to(edges.dtype) / STOP,
                residual_mean, residual_max, residual_count.to(edges.dtype) / STOP)

    def logits(self, encoded, initial_mask, used_m, used_w, selected, *, frozen_context=None):
        _, _, edges = encoded
        batch = edges.shape[0]
        feasible = initial_mask & ~used_m[:, self.machine_id] & ~used_w[:, self.worker_id]
        if self.canonical:
            index = torch.arange(STOP, device=feasible.device)
            last = torch.where(selected, index[None, :], -1).amax(1)
            feasible = feasible & (index[None, :] > last[:, None])
        if frozen_context is None:
            selected_sum = (edges * selected.to(edges.dtype)[..., None]).sum(1)
            cur_mean, cur_max, cur_count, after_mean, after_max, after_count = self.residual_summaries(edges, feasible)
            candidate_input = torch.cat((selected_sum[:, None, :].expand(batch, STOP, -1),
                                         after_mean, after_max, after_count[..., None]), -1)
            stop_input = torch.cat((selected_sum, cur_mean, cur_max, cur_count[:, None]), -1)
            candidate_context = self.context(candidate_input)
            stop_context = self.context(stop_input)
        else:
            candidate_context, stop_context = frozen_context
        edge_logits = self.edge_score(torch.cat((edges, candidate_context), -1)).squeeze(-1)
        stop_logit = self.stop_score(stop_context)
        logits = torch.cat((edge_logits, stop_logit), 1)
        mask = torch.cat((feasible, torch.ones((batch, 1), dtype=torch.bool, device=feasible.device)), 1)
        return logits.masked_fill(~mask, -1e9), feasible, (candidate_context, stop_context)


class SequencePolicy(nn.Module):
    def __init__(self, arm):
        super().__init__()
        if arm not in ("B14", "C14", "D14", "U14"):
            raise ValueError(arm)
        self.arm = arm
        self.actor = ContextActor(direct=arm == "B14", canonical=arm != "U14")
        self.critic = nn.Sequential(nn.Linear(GLOBAL_DIM, 64), nn.Tanh(),
                                    nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1))

    @property
    def actor_parameters(self):
        return sum(p.numel() for p in self.actor.parameters())

    @property
    def critic_parameters(self):
        return sum(p.numel() for p in self.critic.parameters())

    def value(self, pre_observation):
        return self.critic(pre_observation).squeeze(-1)

    def _path(self, machine, worker, initial_mask, sequences=None, *, deterministic=False):
        if machine.ndim == 2:
            machine, worker, initial_mask = machine[None], worker[None], initial_mask[None]
            if sequences is not None and sequences.ndim == 1:
                sequences = sequences[None]
        batch = machine.shape[0]
        encoded = self.actor.encode(machine, worker)
        used_m = torch.zeros((batch, N_MACHINE), dtype=torch.bool, device=machine.device)
        used_w = torch.zeros((batch, N_WORKER), dtype=torch.bool, device=machine.device)
        selected = torch.zeros((batch, STOP), dtype=torch.bool, device=machine.device)
        frozen = None
        logprob = torch.zeros(batch, device=machine.device)
        entropy = torch.zeros(batch, device=machine.device)
        forward_passes = torch.zeros(batch, dtype=torch.long, device=machine.device)
        generated = torch.full((batch, MAX_TOKENS), -1, dtype=torch.long, device=machine.device)
        live = torch.ones(batch, dtype=torch.bool, device=machine.device)
        for step in range(MAX_TOKENS):
            if not bool(live.any()):
                break
            logits, feasible, context = self.actor.logits(
                encoded, initial_mask, used_m, used_w, selected,
                frozen_context=frozen if self.arm == "D14" else None)
            if self.arm == "D14" and frozen is None:
                frozen = context
            choosing = live & feasible.any(1)
            if sequences is not None:
                tokens = sequences[:, step]
                if bool(((tokens >= 0) != choosing).any()):
                    raise ValueError("sequence has missing or extra token")
            else:
                distribution = Categorical(logits=logits)
                tokens = logits.argmax(1) if deterministic else distribution.sample()
                tokens = torch.where(choosing, tokens, -1)
            if bool(choosing.any()):
                distribution = Categorical(logits=logits)
                safe = tokens.clamp(min=0)
                valid = logits.gather(1, safe[:, None]).squeeze(1) > -1e8
                if bool((choosing & ~valid).any()):
                    raise ValueError("masked token in sequence")
                logprob = logprob + torch.where(choosing, distribution.log_prob(safe), 0.)
                entropy = entropy + torch.where(choosing, distribution.entropy(), 0.)
                forward_passes += choosing.long()
            generated[:, step] = tokens
            edge_choice = choosing & (tokens != STOP)
            safe_edge = tokens.clamp(min=0, max=STOP - 1)
            row = torch.arange(batch, device=machine.device)
            if bool(edge_choice.any()):
                selected[row[edge_choice], safe_edge[edge_choice]] = True
                used_m[row[edge_choice], self.actor.machine_id[safe_edge[edge_choice]]] = True
                used_w[row[edge_choice], self.actor.worker_id[safe_edge[edge_choice]]] = True
            live = edge_choice
        if sequences is not None and bool((generated != sequences).any()):
            raise ValueError("sequence differs from reconstructed path")
        return generated, logprob, entropy, forward_passes

    def sample(self, machine, worker, edge_mask, *, deterministic=False):
        tokens, logprob, entropy, passes = self._path(machine, worker, edge_mask, deterministic=deterministic)
        return tokens[0], logprob[0], entropy[0], int(passes[0])

    def logprob_entropy(self, machine, worker, edge_mask, sequences):
        _, logprob, entropy, _ = self._path(machine, worker, edge_mask, sequences)
        return logprob, entropy


@dataclass
class SequenceTransition:
    pre_obs: np.ndarray
    machine_obs: np.ndarray
    worker_obs: np.ndarray
    edge_mask: np.ndarray
    sequence: np.ndarray
    old_logprob: float
    value: float
    reward: float
    done: bool


def choose(policy, env, deterministic=False):
    pre = env.pre_observation()
    submissions = env.fixed_submissions("threshold")
    admitted = env.admitted(submissions)
    machine, worker = actor_inputs_scaled(env.config, admitted, env.tick)
    edge_mask = initial_edge_mask(env.config, admitted)
    with torch.no_grad():
        sequence, logprob, sequence_entropy, passes = policy.sample(
            torch.from_numpy(machine), torch.from_numpy(worker), torch.from_numpy(edge_mask),
            deterministic=deterministic)
        value = policy.value(torch.from_numpy(pre))
    tokens = tuple(int(x) for x in sequence.tolist() if x >= 0)
    matching = matching_from_sequence(tokens)
    index = env.actions.index(matching)
    if not env.matching_mask(submissions)[index]:
        raise RuntimeError("decoder produced an infeasible physical matching")
    return dict(pre_obs=pre, machine_obs=machine, worker_obs=worker,
                edge_mask=edge_mask, sequence=sequence.numpy(), tokens=tokens,
                matching_index=index, matching=matching, submissions=submissions,
                old_logprob=float(logprob), sequence_entropy=float(sequence_entropy),
                value=float(value), decoder_forward_passes=passes)


def ppo_update(policy, optimizer, transitions, bootstrap, *, epochs=4, minibatch=64,
               gamma=.99, gae_lambda=.95, clip=.2, entropy_coef=.01):
    count = len(transitions)
    advantages = np.zeros(count, dtype=np.float32)
    next_value, advantage = float(bootstrap), 0.
    for i in range(count - 1, -1, -1):
        item = transitions[i]
        continuation = 0. if item.done else 1.
        delta = item.reward + gamma * next_value * continuation - item.value
        advantage = delta + gamma * gae_lambda * continuation * advantage
        advantages[i] = advantage
        next_value = item.value
    returns = advantages + np.asarray([t.value for t in transitions], dtype=np.float32)
    pre = torch.as_tensor(np.stack([t.pre_obs for t in transitions]), dtype=torch.float32)
    machine = torch.as_tensor(np.stack([t.machine_obs for t in transitions]), dtype=torch.float32)
    worker = torch.as_tensor(np.stack([t.worker_obs for t in transitions]), dtype=torch.float32)
    mask = torch.as_tensor(np.stack([t.edge_mask for t in transitions]), dtype=torch.bool)
    sequences = torch.as_tensor(np.stack([t.sequence for t in transitions]), dtype=torch.long)
    old_logprobs = torch.as_tensor([t.old_logprob for t in transitions], dtype=torch.float32)
    standardized = torch.as_tensor((advantages - advantages.mean()) /
                                   (advantages.std() + 1e-8), dtype=torch.float32)
    returns_t = torch.as_tensor(returns)
    rng = np.random.default_rng(int(torch.randint(0, 2**31, ()).item()))
    losses, ratio_deviations = [], []
    for _ in range(epochs):
        for indices in np.array_split(rng.permutation(count), max(1, int(np.ceil(count / minibatch)))):
            idx = torch.as_tensor(indices, dtype=torch.long)
            logprob, entropy = policy.logprob_entropy(machine[idx], worker[idx], mask[idx], sequences[idx])
            ratio = torch.exp(logprob - old_logprobs[idx])
            surrogate = torch.minimum(ratio * standardized[idx],
                                      ratio.clamp(1 - clip, 1 + clip) * standardized[idx])
            value = policy.value(pre[idx])
            loss = -surrogate.mean() + .5 * (value - returns_t[idx]).square().mean() - entropy_coef * entropy.mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite sequence PPO loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), .5)
            optimizer.step()
            losses.append(float(loss.detach()))
            ratio_deviations.append(float((ratio.detach() - 1).abs().mean()))
    return {"ppo_loss_mean": float(np.mean(losses)),
            "sequence_ratio_abs_deviation_mean": float(np.mean(ratio_deviations)),
            "return_mean": float(np.mean(returns)),
            "advantage_mean": float(np.mean(advantages))}
