from invoice_audit_env.env import InvoiceAuditEnv
from invoice_audit_env.models import Action


def test_unsafe_approval_is_negative_reward():
    env = InvoiceAuditEnv()
    env.reset(task_id="hard_fraud_detection", seed=7)
    action = Action(action_type="approve_invoice", payload={"approval_code": "AUTO"}, confidence=0.95, reasoning="looks fine")
    _, reward, _, _ = env.step(action)
    assert reward.value < 0


def test_correct_escalation_is_positive_reward():
    env = InvoiceAuditEnv()
    env.reset(task_id="hard_fraud_detection", seed=7)
    action = Action(action_type="escalate_case", payload={"escalation_type": "fraud_review", "evidence_refs": ["dup_invoice"]}, confidence=0.9, reasoning="split pattern")
    _, reward, _, _ = env.step(action)
    assert reward.value > 0


def test_loop_penalty_triggers_and_max_steps_terminate():
    env = InvoiceAuditEnv()
    env.reset(task_id="easy_single_mismatch", seed=99)
    last_reward = None
    done = False
    for _ in range(6):
        _, reward, done, _ = env.step(Action(action_type="ask_vendor_question", payload={"question": "status?"}, confidence=0.2, reasoning="loop"))
        last_reward = reward
        if done:
            break
    assert done is True
    assert last_reward is not None
    assert last_reward.value <= 0
