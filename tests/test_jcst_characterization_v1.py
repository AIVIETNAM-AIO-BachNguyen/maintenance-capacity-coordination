"""Scientific registry invariants fixed before C1 outcomes."""
from ht_pdm_fjsp.component_service_v1 import legal_actions, validate_state
from ht_pdm_fjsp.jcst_characterization_v1 import CASE_NAMES, SEED_RANGES, case_config, physical_seed


def test_locked_cases_are_valid_and_include_expected_negative_controls():
    assert len(CASE_NAMES) == 13
    for name in CASE_NAMES:
        config,state=case_config(name)
        validate_state(config,state)
        assert () in legal_actions(config,state)
        assert all(not c.failed for row in state.components for c in row)
        assert all(job is None for job in state.jobs)
    homogeneous,_=case_config("homogeneous")
    assert homogeneous.components[0][0]==homogeneous.components[0][1]
    assert homogeneous.duration[0][0]==homogeneous.duration[0][1]
    surplus,state=case_config("surplus")
    assert surplus.technicians==surplus.machines
    assert any(len(a)==3 for a in legal_actions(surplus,state))
    restricted,state=case_config("restricted")
    assert all(row[1]==0 for machine in restricted.duration for row in machine[:1])
    assert all(row[0]==0 for machine in restricted.duration for row in machine[1:])
    scale4,state=case_config("scale4")
    assert len(legal_actions(scale4,state))==361


def test_role_seed_banks_are_disjoint_and_outside_forecast_namespace():
    values={physical_seed(role,i) for role,(_,count) in SEED_RANGES.items() for i in range(count)}
    assert len(values)==sum(count for _,count in SEED_RANGES.values())
    assert max(values)<2**64
    assert 104729 not in values


def test_campaign_audit_replays_forecast_and_rejects_corrupt_branch():
    import json
    import sys
    from dataclasses import asdict
    from pathlib import Path
    import pytest
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"experiments"))
    from jcst_campaign_audit_v1 import audit_episode
    from ht_pdm_fjsp.component_controllers_v1 import BoundedRollout
    from ht_pdm_fjsp.component_framework_v1 import run_episode
    from ht_pdm_fjsp.jcst_characterization_v1 import episode_metrics
    c,s=case_config("base")
    episode=run_episode(c,BoundedRollout(2,2,20000),104729,48,16,state=s)
    episode.update(initial_state=asdict(s),metrics=episode_metrics(c,episode),arm="rollout2")
    raw=json.loads(json.dumps(episode))
    assert audit_episode(asdict(c),raw)["records"]==64
    raw["records"][0]["decision"]["forecast_records"][0]["cost"]+=1
    with pytest.raises(ValueError,match="forecast branch"):
        audit_episode(asdict(c),raw)
