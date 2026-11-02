
import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from src.data.seg_multi_source import _InterleavedLoader, allocate_workers
from src.train.trainer import Trainer
from src.utils.checkpoint import load_weights


class SmallModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.5))

    def forward(self, image, **kwargs):
        return {"losses": (self.weight * image - 1).square().mean(), "metrics": {}}


def trainer(tmp_path, **overrides):
    settings = dict(lr=0.01, training_cfg=OmegaConf.create({"max_steps": 2, "save_ckpt_every": 0,
                                                          "log_every_step": 1, "grad_clip": 10}),
                    save_dir=str(tmp_path), device="cpu")
    settings.update(overrides)
    result = Trainer(**settings)
    result.build(SmallModel())
    return result


def test_accumulation_matches_large_batch(tmp_path):
    small, large = trainer(tmp_path / "small"), trainer(tmp_path / "large")
    first = small.step({"image": torch.tensor([1.]), "query": ["x"]}, accum_steps=2, accum_idx=0)
    assert small.global_step == 0
    assert first["loss"] == pytest.approx(0.25)  # logged loss is not divided by accumulation
    small.step({"image": torch.tensor([2.]), "query": ["x"]}, accum_steps=2, accum_idx=1)
    large.step({"image": torch.tensor([1., 2.]), "query": ["x", "x"]})
    assert small.global_step == large.global_step == 1
    torch.testing.assert_close(small.model.weight, large.model.weight)


def test_initialize_and_resume_are_distinct(tmp_path):
    original = trainer(tmp_path / "original")
    batch = {"image": torch.tensor([1.]), "query": ["x"]}
    original.step(batch)
    original.save()
    path = tmp_path / "original/model_step_1.pt"
    initialized, resumed = trainer(tmp_path / "init"), trainer(tmp_path / "resume")
    initialized.load_model(path)
    resumed.load_model(path, resume=True)
    assert initialized.global_step == 0
    assert not initialized.optim.state
    assert resumed.global_step == 1
    original.step(batch)
    resumed.step(batch)
    torch.testing.assert_close(original.model.weight, resumed.model.weight)


def test_nonfinite_loss_does_not_update_weights(tmp_path):
    run = trainer(tmp_path)
    before = run.model.weight.detach().clone()
    with pytest.raises(FloatingPointError):
        run.step({"image": torch.tensor([float("nan")]), "query": ["x"]})
    torch.testing.assert_close(run.model.weight, before)
    assert run.global_step == 0


def test_fit_counts_optimizer_steps_across_epochs(tmp_path):
    run = trainer(tmp_path, grad_accumulation_steps=2)
    run.fit([{"image": torch.tensor([1.]), "query": ["x"]}])
    assert run.global_step == 2
    assert run.epoch == 3


def test_worker_budget_and_source_plan():
    for budget in range(6):
        assert sum(allocate_workers(budget, {"a": 1, "b": 3}).values()) == budget
    with pytest.raises(ValueError):
        allocate_workers(3, {"a": float("nan")})
    loaders = {"a": [{"item": 1}], "b": [{"item": 2}]}
    first = _InterleavedLoader(loaders, fixed_steps=32, seed=4, motion_sources={"b"})
    second = _InterleavedLoader(loaders, fixed_steps=32, seed=4, motion_sources={"b"})
    assert list(first) == list(second)
    assert {row["has_motion"] for row in first} == {True, False}
    before = list(first)
    first.set_epoch(1)
    assert before != list(first)


def test_checkpoint_rejects_missing_projector(tmp_path):
    model = nn.Module()
    model.decoder = nn.Module()
    model.decoder.projector = nn.Linear(2, 2)
    path = tmp_path / "bad.pt"
    torch.save({"model": {}}, path)
    with pytest.raises(ValueError, match="required weights"):
        load_weights(model, path)
