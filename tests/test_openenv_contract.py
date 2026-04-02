import pytest

from invoice_audit_env.env import InvoiceAuditEnv
from invoice_audit_env.models import Action, Observation, Reward


def test_reset_returns_observation():
    env = InvoiceAuditEnv()
    obs = env.reset(task_id="easy_single_mismatch", seed=42)
    assert isinstance(obs, Observation)


def test_step_returns_four_tuple_and_info_keys():
    env = InvoiceAuditEnv()
    env.reset(task_id="easy_single_mismatch", seed=42)
    action = Action(
        action_type="review_invoice",
        payload={"checks": ["amount", "tax_code"]},
        confidence=0.8,
        reasoning="validate fields",
    )
    obs, reward, done, info = env.step(action)
    assert isinstance(obs, Observation)
    assert isinstance(reward, Reward)
    assert isinstance(done, bool)
    assert isinstance(info, dict)
    assert set(["last_action_error", "step_num", "task_id", "grader_score"]).issubset(info.keys())


def test_state_matches_task_and_steps():
    env = InvoiceAuditEnv()
    env.reset(task_id="medium_policy_tangle", seed=123)
    state = env.state()
    assert state["task_id"] == "medium_policy_tangle"
    assert state["step_num"] == 0
