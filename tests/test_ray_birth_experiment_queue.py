import pytest
from scripts.queue_ray_birth_budget_experiment import predecessor_completed,experiment_command


def test_running_is_not_completed_and_failure_never_launches():
    assert not predecessor_completed(dict(state='running',returncode=None))
    assert predecessor_completed(dict(state='completed',returncode=0))
    with pytest.raises(RuntimeError):predecessor_completed(dict(state='failed',returncode=1))


def test_allocator_command_changes_only_output_and_explicit_budget():
    base=['python','train.py','-m','original','--allow-trainer-repair-resume','--iterations','3000']
    result=experiment_command(base,'new',8192)
    assert base[3]=='original' and result[3]=='new'
    assert result[-3:]==['--maximum-static-ray-births-per-event','8192','--allow-static-ray-birth-budget-ablation-resume']
    assert result[:3]+result[4:-3]==base[:3]+base[4:]
