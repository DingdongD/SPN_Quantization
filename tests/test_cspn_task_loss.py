import pytest
import torch

from spn_quant.qat.cspn_task_loss import (
    CSPNTaskLossWeights,
    cspn_task_aware_loss,
    depth_boundary_mask,
)


def _fixture():
    prediction = torch.tensor(
        [[[[1.0, 3.0], [2.0, 4.0]]]], requires_grad=True)
    target = torch.tensor([[[[1.0, 2.0], [2.0, 5.0]]]])
    teacher = torch.tensor(
        [[[[1.0, 2.5], [2.0, 4.5]]]], requires_grad=True)
    valid = torch.ones_like(target, dtype=torch.bool)
    return prediction, target, teacher, valid


def test_task_loss_is_exact_weighted_sum_and_teacher_is_detached():
    prediction, target, teacher, valid = _fixture()
    weights = CSPNTaskLossWeights(1.0, 0.25, 0.5, 0.1)

    result = cspn_task_aware_loss(
        prediction, target, valid, teacher,
        (prediction,), (teacher,), weights, 0.1)

    expected = (
        result["depth"] + 0.25 * result["boundary"]
        + 0.5 * result["teacher"] + 0.1 * result["propagation"])
    assert torch.equal(result["total"], expected)
    result["total"].backward()
    assert prediction.grad is not None
    assert teacher.grad is None


def test_boundary_mask_uses_required_metric_threshold():
    target = torch.tensor([[[[1.0, 1.05, 1.2], [1.0, 1.0, 1.0]]]])
    valid = torch.ones_like(target, dtype=torch.bool)

    mask = depth_boundary_mask(target, valid, 0.1)

    expected = torch.tensor(
        [[[[False, False, True], [False, False, True]]]])
    assert torch.equal(mask, expected)


def test_empty_boundary_and_zero_weights_retain_zero_gradient_path():
    prediction = torch.ones(1, 1, 2, 2, requires_grad=True)
    target = torch.ones_like(prediction)
    valid = torch.ones_like(target, dtype=torch.bool)
    weights = CSPNTaskLossWeights(0.0, 0.0, 0.0, 0.0)

    result = cspn_task_aware_loss(
        prediction, target, valid, target,
        (prediction,), (target,), weights, 0.1)

    assert result["boundary"].item() == 0.0
    assert result["total"].item() == 0.0
    result["total"].backward()
    assert torch.equal(prediction.grad, torch.zeros_like(prediction))


@pytest.mark.parametrize("field", (
    "prediction", "target", "teacher", "student_state", "teacher_state"))
def test_task_loss_rejects_nonfinite_tensors(field):
    prediction, target, teacher, valid = _fixture()
    student_state = prediction
    teacher_state = teacher
    values = {
        "prediction": prediction,
        "target": target,
        "teacher": teacher,
        "student_state": student_state,
        "teacher_state": teacher_state,
    }
    replacement = values[field].detach().clone()
    replacement.flatten()[0] = float("nan")
    values[field] = replacement

    with pytest.raises(ValueError, match="finite"):
        cspn_task_aware_loss(
            values["prediction"], values["target"], valid,
            values["teacher"], (values["student_state"],),
            (values["teacher_state"],),
            CSPNTaskLossWeights(1.0, 0.25, 0.5, 0.1), 0.1)


def test_task_loss_rejects_state_count_and_shape_mismatch():
    prediction, target, teacher, valid = _fixture()
    weights = CSPNTaskLossWeights(1.0, 0.25, 0.5, 0.1)

    with pytest.raises(ValueError, match="state count"):
        cspn_task_aware_loss(
            prediction, target, valid, teacher,
            (prediction,), (), weights, 0.1)
    with pytest.raises(ValueError, match="shape"):
        cspn_task_aware_loss(
            prediction, target, valid, teacher[..., :1],
            (prediction,), (teacher,), weights, 0.1)


def test_task_loss_requires_valid_weights_threshold_and_mask():
    prediction, target, teacher, valid = _fixture()
    with pytest.raises(ValueError, match="nonnegative"):
        CSPNTaskLossWeights(1.0, -0.1, 0.5, 0.1)
    with pytest.raises(ValueError, match="threshold"):
        depth_boundary_mask(target, valid, 0.0)
    invalid = torch.zeros_like(valid)
    with pytest.raises(ValueError, match="valid depth"):
        cspn_task_aware_loss(
            prediction, target, invalid, teacher,
            (prediction,), (teacher,),
            CSPNTaskLossWeights(1.0, 0.25, 0.5, 0.1), 0.1)
