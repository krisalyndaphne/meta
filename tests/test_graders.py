from invoice_audit_env.graders import grade_episode
from invoice_audit_env.tasks import TASK_FIXTURES


def test_perfect_paths_score_one():
    for task_id, task in TASK_FIXTURES.items():
        score = grade_episode(task_id, task.perfect_actions)
        assert score == 0.99


def test_worst_paths_score_zero():
    for task_id, task in TASK_FIXTURES.items():
        score = grade_episode(task_id, task.worst_actions)
        assert score == 0.01
