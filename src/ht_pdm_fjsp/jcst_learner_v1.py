"""JCST/1.1 complete-matching PPO actors and shared full-state critic.

S has demand/capacity cross messages. A uses the same modules and information,
but computes entity messages without cross-entity inputs. F scores each complete
matching directly from the flat state and action. All arms use one centralized
actor and the same full-information critic.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from .component_service_v1 import Action, Config, State, hazard, initial_state, legal_actions, machine_modes


def action_universe(config: Config) -> tuple[Action, ...]:
    # Every legal state catalogue is a subset of the fully idle catalogue.
    return legal_actions(config, initial_state(config))


@dataclass(frozen=True)
class Encoding:
    component: np.ndarray
    technician: np.ndarray
    flat: np.ndarray
    mask: np.ndarray


def encode(config: Config, state: State, universe: tuple[Action, ...],
           horizon: int = 48, tail: int = 16) -> Encoding:
    modes = machine_modes(state)
    mcount, kcount = config.machines, config.technicians
    comp = []
    for m, row in enumerate(state.components):
        for c, item in enumerate(row):
            spec = config.components[m][c]
            comp.append([
                item.age / 10.0, float(item.failed), hazard(spec, item.age),
                spec.scale / 10.0, spec.shape / 5.0,
                float(modes[m] == "service"), float(modes[m] == "breakdown"),
                *[config.duration[m][c][j] / 8.0 for j in range(kcount)],
            ])
    tech = []
    for j, job in enumerate(state.jobs):
        tech.append([
            float(job is None), 0.0 if job is None else job.remaining / 8.0,
            *[float(job is not None and job.machine == m) for m in range(mcount)],
            *[float(job is not None and job.component == c) for c in range(len(config.components[0]))],
            float(job is not None and job.kind == "PM"),
            float(job is not None and job.kind == "CM"),
            *[config.duration[m][c][j] / 8.0 for m in range(mcount)
              for c in range(len(config.components[m]))],
        ])
    component = np.asarray(comp, dtype=np.float32)
    technician = np.asarray(tech, dtype=np.float32)
    public = np.asarray([
        state.tick / horizon, horizon / 48.0, tail / 16.0,
        config.preventive_cost / 15.0, config.corrective_cost / 15.0,
        config.unavailable_cost / 15.0, config.failure_cost / 15.0,
    ], dtype=np.float32)
    flat = np.concatenate((component.ravel(), technician.ravel(), public))
    current = set(legal_actions(config, state))
    mask = np.asarray([action in current for action in universe], dtype=np.bool_)
    if not mask[0] or int(mask.sum()) != len(current):
        raise AssertionError("full legal catalogue/mask mismatch")
    return Encoding(component, technician, flat, mask)


class StructuredActor(nn.Module):
    def __init__(self, config: Config, universe: tuple[Action, ...], ablated: bool):
        super().__init__()
        self.ablated = ablated
        mcount = sum(map(len, config.components))
        kcount = config.technicians
        cdim = 7 + kcount
        wdim = 2 + config.machines + len(config.components[0]) + 2 + mcount
        self.component = nn.Sequential(nn.Linear(cdim, 32), nn.Tanh(), nn.Linear(32, 16), nn.Tanh())
        self.technician = nn.Sequential(nn.Linear(wdim, 32), nn.Tanh(), nn.Linear(32, 16), nn.Tanh())
        self.comp_message = nn.Sequential(nn.Linear(33, 32), nn.Tanh(), nn.Linear(32, 16), nn.Tanh())
        self.tech_message = nn.Sequential(nn.Linear(33, 32), nn.Tanh(), nn.Linear(32, 16), nn.Tanh())
        self.edge = nn.Sequential(nn.Linear(33, 32), nn.Tanh(), nn.Linear(32, 16), nn.Tanh())
        self.score = nn.Sequential(nn.Linear(kcount * 16 + 40, 64), nn.Tanh(),
                                   nn.Linear(64, 32), nn.Tanh(), nn.Linear(32, 1))
        slots_c = np.zeros((len(universe), kcount), dtype=np.int64)
        active = np.zeros((len(universe), kcount), dtype=np.float32)
        for i, action in enumerate(universe):
            for m, c, j in action:
                slots_c[i, j] = sum(len(config.components[x]) for x in range(m)) + c
                active[i, j] = 1.0
        self.register_buffer("slots_c", torch.as_tensor(slots_c), persistent=False)
        self.register_buffer("active", torch.as_tensor(active), persistent=False)
        durations = [config.duration[m][c] for m in range(config.machines)
                     for c in range(len(config.components[m]))]
        self.register_buffer("duration", torch.as_tensor(durations, dtype=torch.float32) / 8.0,
                             persistent=False)

    def forward(self, component: torch.Tensor, technician: torch.Tensor,
                flat: torch.Tensor) -> torch.Tensor:
        c = self.component(component)
        w = self.technician(technician)
        batch, nc, _ = c.shape
        nw = w.shape[1]
        cc = c[:, :, None, :].expand(batch, nc, nw, 16)
        ww = w[:, None, :, :].expand(batch, nc, nw, 16)
        dd = self.duration[None, :, :, None].expand(batch, nc, nw, 1)
        if self.ablated:
            mc = self.comp_message(torch.cat((cc, torch.zeros_like(ww), dd), -1))
            mw = self.tech_message(torch.cat((torch.zeros_like(cc), ww, dd), -1))
        else:
            both = torch.cat((cc, ww, dd), -1)
            mc = self.comp_message(both)
            mw = self.tech_message(both)
        c = c + mc.mean(dim=2)
        w = w + mw.mean(dim=1)
        selected_c = c[:, self.slots_c]
        selected_w = w[:, None, :, :].expand(batch, len(self.slots_c), nw, 16)
        selected_d = self.duration[self.slots_c,
                                   torch.arange(nw, device=component.device)[None, :]]
        selected_d = selected_d[None, :, :, None].expand(batch, -1, -1, -1)
        edges = self.edge(torch.cat((selected_c, selected_w, selected_d), -1))
        edges = edges * self.active[None, :, :, None]
        global_c = c.mean(dim=1)[:, None, :].expand(-1, len(self.slots_c), -1)
        global_w = w.mean(dim=1)[:, None, :].expand(-1, len(self.slots_c), -1)
        count = self.active.sum(dim=1)[None, :, None].expand(batch, -1, -1) / nw
        public = flat[:, None, -7:].expand(-1, len(self.slots_c), -1)
        return self.score(torch.cat((edges.flatten(start_dim=2), global_c, global_w,
                                     count, public), -1)).squeeze(-1)


class FlatActor(nn.Module):
    def __init__(self, config: Config, universe: tuple[Action, ...], flat_dim: int,
                 width: int = 70):
        super().__init__()
        edges = sum(map(len, config.components)) * config.technicians
        action_bits = torch.zeros((len(universe), edges + 1), dtype=torch.float32)
        for i, action in enumerate(universe):
            for m, c, j in action:
                offset = sum(len(config.components[x]) for x in range(m)) + c
                action_bits[i, offset * config.technicians + j] = 1.0
            action_bits[i, -1] = len(action) / config.technicians
        self.register_buffer("action_bits", action_bits, persistent=False)
        self.net = nn.Sequential(nn.Linear(flat_dim + edges + 1, width), nn.Tanh(),
                                 nn.Linear(width, width), nn.Tanh(), nn.Linear(width, 1))

    def forward(self, flat: torch.Tensor) -> torch.Tensor:
        batch, dim = flat.shape
        count = len(self.action_bits)
        x = torch.cat((flat[:, None, :].expand(batch, count, dim),
                       self.action_bits[None, :, :].expand(batch, count, -1)), -1)
        return self.net(x).squeeze(-1)


class Policy(nn.Module):
    def __init__(self, arm: str, config: Config, universe: tuple[Action, ...]):
        super().__init__()
        if arm not in ("S", "F", "A"):
            raise ValueError(arm)
        self.arm = arm
        self.universe = universe
        sample = encode(config, initial_state(config), universe)
        self.actor = (FlatActor(config, universe, len(sample.flat)) if arm == "F" else
                      StructuredActor(config, universe, arm == "A"))
        self.critic = nn.Sequential(nn.Linear(len(sample.flat), 64), nn.Tanh(),
                                    nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1))

    def logits(self, c: torch.Tensor, w: torch.Tensor, flat: torch.Tensor,
               mask: torch.Tensor) -> torch.Tensor:
        raw = self.actor(flat) if self.arm == "F" else self.actor(c, w, flat)
        return raw.masked_fill(~mask, -1e9)

    def distribution(self, c, w, flat, mask):
        return Categorical(logits=self.logits(c, w, flat, mask))

    def value(self, flat):
        return self.critic(flat).squeeze(-1)

    def parameter_counts(self) -> dict[str, int]:
        return {"actor": sum(p.numel() for p in self.actor.parameters()),
                "critic": sum(p.numel() for p in self.critic.parameters())}


def tensors(items: list[Encoding]) -> tuple[torch.Tensor, ...]:
    return tuple(torch.as_tensor(np.stack([getattr(x, name) for x in items]), dtype=dtype)
                 for name, dtype in (("component", torch.float32), ("technician", torch.float32),
                                     ("flat", torch.float32), ("mask", torch.bool)))


def choose(policy: Policy, encoded: Encoding, deterministic: bool = False):
    c, w, flat, mask = tensors([encoded])
    with torch.no_grad():
        dist = policy.distribution(c, w, flat, mask)
        index = int(torch.argmax(dist.logits, dim=-1).item() if deterministic else dist.sample().item())
        return index, float(dist.log_prob(torch.tensor([index])).item()), float(policy.value(flat).item())


@dataclass
class Transition:
    encoded: Encoding
    action_index: int
    logprob: float
    value: float
    reward: float
    done: bool


def ppo_update(policy: Policy, optimizer: torch.optim.Optimizer, data: list[Transition],
               *, epochs: int = 4, minibatch: int = 128, lam: float = 0.95,
               clip: float = 0.2, entropy_coef: float = 0.01) -> dict[str, float]:
    # gamma=1, because the primary objective is undiscounted physical H+tail cost.
    # Each update receives complete episodes; terminal reward includes tail once.
    n = len(data)
    adv = np.zeros(n, dtype=np.float32)
    running = 0.0
    next_value = 0.0
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
    losses = []
    for _ in range(epochs):
        for indices in torch.randperm(n).split(minibatch):
            dist = policy.distribution(c[indices], w[indices], flat[indices], mask[indices])
            new_lp = dist.log_prob(actions[indices])
            ratio = torch.exp(new_lp - old_lp[indices])
            surrogate = torch.minimum(ratio * advantage[indices],
                                      ratio.clamp(1 - clip, 1 + clip) * advantage[indices])
            value = policy.value(flat[indices])
            # Rewards are divided by 100 in the collector; critic targets use that scale.
            loss = -surrogate.mean() + 0.5 * (value - target[indices]).square().mean() \
                   - entropy_coef * dist.entropy().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite PPO loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()
            losses.append(float(loss.detach()))
    return {"loss": float(np.mean(losses)), "advantage_std": float(adv.std()),
            "return_mean": float(returns.mean())}
