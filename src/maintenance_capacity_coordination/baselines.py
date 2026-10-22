"""Adapters for the inherited fixed-requester baseline environment."""
from pathlib import Path

import torch
from sb3_contrib import MaskablePPO

from ht_pdm_fjsp.maintenance_framework_v1 import catalogue, vector_observation
from ht_pdm_fjsp.maintenance_framework_v14_policy import OpportunityCostActor
from ht_pdm_fjsp.maintenance_request_policies import request_action
from ht_pdm_fjsp.maintenance_requests import legal_matchings, submit_requests


def make_ppo(env, seed):
    return MaskablePPO(
        "MlpPolicy", env, seed=seed, device="cpu", n_steps=64,
        batch_size=32, n_epochs=2, policy_kwargs={"net_arch": [64, 64]},
        verbose=0,
    )


class V14Controller:
    def __init__(self, actor):
        self.actor = actor.cpu().eval()
        self.name = "v14_" + actor.mode
        self.forecast_transitions = 0

    @classmethod
    def from_checkpoint(cls, checkpoint: str | Path):
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        actor = OpportunityCostActor(payload["metadata"]["arm"])
        actor.load_state_dict(payload["state_dict"], strict=True)
        return cls(actor)

    def choose(self, config, state, tick, *, case, seed):
        if (config.machines, config.technicians) != (4, 2):
            raise ValueError("legacy V14 requires N4/K2")
        submissions = request_action(config, state, tick, "threshold")
        admitted = submit_requests(config, state, tick, submissions)
        actions = catalogue()
        legal = set(legal_matchings(config, admitted))
        mask = torch.tensor([action in legal for action in actions])
        observation = vector_observation(config, admitted, tick)
        with torch.no_grad():
            logits = self.actor(torch.from_numpy(observation).reshape(1, 56))[0]
            probabilities = torch.softmax(logits.masked_fill(~mask, -1e9), dim=0).cpu().numpy()
            index = int(probabilities.argmax())
        return submissions, actions[index]
