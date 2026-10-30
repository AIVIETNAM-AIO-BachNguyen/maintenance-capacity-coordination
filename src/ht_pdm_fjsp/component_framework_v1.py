"""JCST/1.1 controlled H intervals plus versioned fixed-risk tail adapter."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import time

from .component_service_v1 import Action, Config, State, initial_state, legal_actions, transition
from .component_controllers_v1 import Decision, RiskMatching, observe

ADAPTER_VERSION = "JCST/1.1-H-tail-risk-v1"


@dataclass(frozen=True)
class Step:
    state: State
    reward: float
    controlled_cost: float
    tail_cost: float
    done: bool
    records: tuple[dict, ...]


def advance(config: Config, state: State, action: Action, physical_seed: int,
            horizon: int = 48, tail: int = 16) -> Step:
    if type(horizon) is not int or horizon < 1 or type(tail) is not int or tail < 0:
        raise ValueError("horizon/tail")
    if state.tick >= horizon:
        raise ValueError("controlled horizon ended")
    result = transition(config, state, action, physical_seed)
    records = [dict(phase="H", tick=state.tick, before=asdict(state),
                    action=[list(x) for x in action],
                    catalogue=[[list(x) for x in a] for a in legal_actions(config,state)],
                    after=asdict(result.state),
                    cost=result.cost, breakdown=result.breakdown,
                    events=list(result.events))]
    tail_cost = 0.0
    next_state = result.state
    if next_state.tick == horizon:
        risk = RiskMatching()
        while next_state.tick < horizon + tail:
            before = next_state
            tail_action = risk.choose(observe(config, before, horizon, tail)).action
            tail_result = transition(config, before, tail_action, physical_seed)
            tail_cost += tail_result.cost
            records.append(dict(phase="tail", tick=before.tick, before=asdict(before),
                                action=[list(x) for x in tail_action],
                                catalogue=[[list(x) for x in a] for a in legal_actions(config,before)],
                                after=asdict(tail_result.state), cost=tail_result.cost,
                                breakdown=tail_result.breakdown,
                                events=list(tail_result.events)))
            next_state = tail_result.state
    return Step(next_state, -(result.cost + tail_cost), result.cost,
                tail_cost, next_state.tick == horizon + tail, tuple(records))


def run_episode(config: Config, controller, physical_seed: int,
                horizon: int = 48, tail: int = 16,
                state: State | None = None, record_callback=None) -> dict:
    episode_start = time.perf_counter()
    if state is None:
        state = initial_state(config)
    if state.tick != 0:
        raise ValueError("episode must start at tick zero")
    records: list[dict] = []
    totals = dict(root_actions_evaluated=0, forecast_scenarios=0,
                  simulated_transitions=0, selection_seconds=0.0, wall_seconds=0.0)
    h_cost = tail_cost = 0.0
    for _ in range(horizon):
        obs = observe(config, state, horizon, tail)
        decision: Decision = controller.choose(obs)
        if decision.action not in obs.actions:
            raise ValueError("controller selected action outside full catalogue")
        for key in totals:
            totals[key] += getattr(decision, key)
        step = advance(config, state, decision.action, physical_seed, horizon, tail)
        step.records[0]["decision"] = asdict(decision)
        for record in step.records:
            records.append(record)
            if record_callback is not None:
                record_callback(record)
        h_cost += step.controlled_cost
        tail_cost += step.tail_cost
        state = step.state
    return dict(seed=physical_seed, controller=controller.name,
                adapter_version=ADAPTER_VERSION, horizon=horizon, tail=tail,
                records=records, controlled_cost=h_cost, tail_cost=tail_cost,
                total_cost=h_cost + tail_cost, compute=totals,
                episode_wall_seconds=time.perf_counter() - episode_start,
                final_state=asdict(state), terminal_status="complete")
