from copy import deepcopy
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))
from test_pmpo_integration import make_system
from models.pmpo_beta import PMPOOptimizers


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))])
def test_next_actual_imagined_update_resume(tmp_path, device):
    torch.set_num_threads(1)
    torch.backends.cudnn.deterministic = True
    model, _ = make_system(device, prior_refresh_interval=2)
    opt = PMPOOptimizers(model, 0)
    loss, _ = model()
    loss.backward()
    opt.step()
    path = tmp_path / "pmpo.pt"
    torch.save(dict(model=model.state_dict(), optimizer=opt.state_dict()), path)
    # Constructing a new process's models consumes global random numbers.
    restored, _ = make_system(device, prior_refresh_interval=2)
    restored_opt = PMPOOptimizers(restored, 0)
    saved = torch.load(path, map_location=device, weights_only=False)
    restored.load_state_dict(saved["model"])
    restored_opt.load_state_dict(saved["optimizer"])
    for _ in range(2):  # crosses a prior-refresh boundary as well
        opt.zero_grad()
        restored_opt.zero_grad()
        loss, metrics = model()
        actual, actual_metrics = restored()
        torch.testing.assert_close(actual, loss, rtol=0, atol=0)
        for key in metrics:
            torch.testing.assert_close(torch.as_tensor(actual_metrics[key]), torch.as_tensor(metrics[key]), rtol=0, atol=0)
        loss.backward()
        actual.backward()
        opt.step()
        restored_opt.step()
        for p, q in zip(model.parameters(), restored.parameters()):
            torch.testing.assert_close(p, q, rtol=0, atol=0)


def test_imagination_does_not_consume_global_rng():
    model, _ = make_system()
    before = torch.get_rng_state().clone()
    model.collect_imagination()
    assert torch.equal(torch.get_rng_state(), before)
