import json
import pytest
import torch
from scripts.canopy_diagnostic_checkpoint import save_diagnostic_checkpoint, publish_diagnostic_metrics


def test_optimizer_state_roundtrip_and_checkpoint_is_never_overwritten(tmp_path):
    parameter = torch.tensor([.2], requires_grad=True); optimizer = torch.optim.Adam([parameter], lr=.05)
    parameter.square().sum().backward(); optimizer.step()
    path = tmp_path/'state.pth'
    payload = dict(diagnostic_only=True, candidate=parameter.detach(), optimizer_state=optimizer.state_dict(), step=1)
    save_diagnostic_checkpoint(path, payload)
    result = torch.load(path, weights_only=False)
    assert torch.equal(result['candidate'], parameter)
    assert torch.equal(result['optimizer_state']['state'][0]['exp_avg'], optimizer.state[parameter]['exp_avg'])
    with pytest.raises(FileExistsError): save_diagnostic_checkpoint(path, payload)
    assert not list(tmp_path.glob('*.tmp'))


def test_failed_save_does_not_publish_partial_checkpoint(tmp_path, monkeypatch):
    def broken_save(payload, handle):
        handle.write(b'partial'); raise OSError('simulated interrupted write')
    monkeypatch.setattr(torch, 'save', broken_save)
    with pytest.raises(OSError): save_diagnostic_checkpoint(tmp_path/'state.pth', dict(diagnostic_only=True))
    assert not list(tmp_path.iterdir())


def test_metrics_publish_rejects_nonfinite_payload_without_damaging_previous(tmp_path):
    path = tmp_path/'metrics.json'; publish_diagnostic_metrics(path, dict(step=1))
    with pytest.raises(ValueError): publish_diagnostic_metrics(path, dict(step=float('nan')))
    assert json.loads(path.read_text()) == dict(step=1)
