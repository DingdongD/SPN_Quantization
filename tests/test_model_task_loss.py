import torch
import torch.nn as nn

from spn_quant.adapters.base import ModelSemanticAdapter
from spn_quant.qat.task_loss import (
    ModelTaskLossWeights,
    model_task_aware_loss,
)


class ToyPropagation(nn.Module):
    def forward(self, initial):
        first = initial + 0.25
        second = first + 0.5
        return second, (first, second)


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.initial = nn.Conv2d(1, 1, 1, bias=False)
        self.prop = ToyPropagation()

    def forward(self, value):
        initial = self.initial(value)
        prediction, states = self.prop(initial)
        return {
            "pred": prediction,
            "pred_init": initial,
            "pred_inter": states,
        }


class ToySemanticAdapter(ModelSemanticAdapter):
    MODEL_NAME = "toy"
    PROPAGATION_PATHS = ("prop",)

    def _input_signals(self, inputs):
        return {"rgb": inputs[0], "sparse_depth": inputs[0]}

    def _propagation_inputs(self, inputs):
        return {"initial_depth": inputs[0]}

    def _propagation_outputs(self, output):
        return {"propagation_state": output[1]}

    def _build_merge_adapters(self):
        return ()


def _capture(model, value):
    adapter = ToySemanticAdapter(model, strict=False)
    adapter.begin_task_capture()
    model(value)
    return adapter, adapter.task_capture()


def test_task_loss_uses_semantic_initial_depth_and_propagation_captures():
    student = ToyModel()
    teacher = ToyModel()
    student.initial.weight.data.fill_(1.0)
    teacher.initial.weight.data.fill_(0.5)
    value = torch.ones(1, 1, 2, 2)
    student_adapter, student_capture = _capture(student, value)
    with torch.no_grad():
        teacher_adapter, teacher_capture = _capture(teacher, value)
    target = torch.full((1, 1, 2, 2), 2.0)
    weights = ModelTaskLossWeights(
        depth=1.0,
        boundary=0.25,
        teacher=0.5,
        initial_depth=0.75,
        propagation=0.1,
    )

    loss = model_task_aware_loss(
        student_capture,
        teacher_capture,
        target,
        target > 0.0,
        weights,
        0.1,
    )

    assert torch.isfinite(loss.total)
    assert loss.propagation.ndim == 0
    assert loss.initial_depth.ndim == 0
    expected = (
        loss.depth + 0.25 * loss.boundary + 0.5 * loss.teacher
        + 0.75 * loss.initial_depth + 0.1 * loss.propagation)
    assert torch.equal(loss.total, expected)
    loss.total.backward()
    assert student.initial.weight.grad is not None
    assert teacher.initial.weight.grad is None
    student_adapter.close()
    teacher_adapter.close()


def test_semantic_task_capture_starts_fresh_for_each_forward():
    model = ToyModel()
    adapter = ToySemanticAdapter(model, strict=False)
    adapter.begin_task_capture()
    model(torch.ones(1, 1, 1, 1))
    first = adapter.task_capture()
    adapter.begin_task_capture()
    model(torch.full((1, 1, 1, 1), 2.0))
    second = adapter.task_capture()

    assert len(first.propagation_states) == 2
    assert len(second.propagation_states) == 2
    assert not torch.equal(first.initial_depth, second.initial_depth)
    adapter.close()
