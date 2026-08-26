import pytest
import torch

from scripts import smoke_nyu_selected_quantization as smoke
from spn_quant.qat.task_loss import ModelTaskCapture, ModelTaskLossWeights


def test_official_smoke_matrix_is_exact_and_ordered():
    assert smoke.SMOKE_METHODS == (
        "fp32",
        "rtn_w8a8",
        "rtn_w4a4",
        "qdrop_w6a6",
        "brecq_w6a6",
        "lsqplus_w4a4_step",
        "hawq_probe",
        "p3_t3_candidate",
    )


def test_prediction_assertion_requires_shape_finite_native_and_propagation():
    prediction = torch.ones(1, 1, 228, 304)
    rows = (
        {"signal": "state", "mse": 0.0},
        {
            "signal": "affinity_constraints",
            "coefficient_sum_max_error": 0.0,
            "contraction_violation_rate": 0.0,
        },
        {"signal": "anchor", "anchor_max_error": 0.0},
    )

    result = smoke.assert_official_prediction(
        "nlspn", "rtn_w4a4", prediction, True, rows,
        native_calls=18, official_propagation_calls=18)

    assert result["shape"] == [1, 1, 228, 304]
    assert result["finite"] == 1
    assert result["native_extension_calls"] == 18
    assert result["propagation_valid"] == 1


@pytest.mark.parametrize(
    "prediction, rows, native_calls, message",
    (
        (torch.ones(1, 1, 2, 3), (), 1, "shape"),
        (torch.full((1, 1, 228, 304), float("nan")), (), 1, "finite"),
        (torch.ones(1, 1, 228, 304), (), 0, "native"),
        (torch.ones(1, 1, 228, 304), (), 1, "propagation"),
    ),
)
def test_prediction_assertion_rejects_incomplete_hard_smoke(
        prediction, rows, native_calls, message):
    with pytest.raises(RuntimeError, match=message):
        smoke.assert_official_prediction(
            "completionformer", "method", prediction, False, rows,
            native_calls, official_propagation_calls=1)


def test_dyspn_prediction_requires_official_grid_propagation_calls():
    rows = (
        {"signal": "state", "mse": 0.0},
        {"signal": "affinity_constraints",
         "coefficient_sum_max_error": 0.0,
         "contraction_violation_rate": 0.0},
        {"signal": "anchor_injection", "anchor_max_error": 0.0},
    )

    with pytest.raises(RuntimeError, match="propagation operator"):
        smoke.assert_official_prediction(
            "dyspn", "method", torch.ones(1, 1, 228, 304), True,
            rows, native_calls=1, official_propagation_calls=0)


def test_smoke_parser_requires_explicit_runtime_and_output_controls():
    with pytest.raises(SystemExit):
        smoke.build_parser().parse_args([])


def test_lsqplus_smoke_uses_semantic_task_gradient_when_prediction_is_hard():
    scale = torch.tensor(1.0, requires_grad=True)
    target = torch.full((1, 1, 2, 2), 2.0)

    class Adapter(object):
        def __init__(self, capture):
            self.capture = capture

        def begin_task_capture(self):
            return None

        def task_capture(self):
            return self.capture()

    class Model(object):
        def __init__(self, output):
            self.output = output

        def __call__(self, value):
            del value
            return self.output()

        def eval(self):
            return self

    student_capture = lambda: ModelTaskCapture(
        prediction=scale.detach().expand_as(target),
        initial_depth=scale.expand_as(target),
        propagation_states=(scale.expand_as(target),),
    )
    teacher_capture = lambda: ModelTaskCapture(
        prediction=torch.ones_like(target),
        initial_depth=torch.zeros_like(target),
        propagation_states=(torch.zeros_like(target),),
    )
    student_model = Model(lambda: {"pred": scale.detach().expand_as(target)})
    teacher_model = Model(lambda: {"pred": torch.ones_like(target)})
    runtime = type("Runtime", (), {
        "prediction": staticmethod(lambda output: output["pred"]),
    })()
    weights = ModelTaskLossWeights(1.0, 0.25, 0.5, 0.1, 0.1)

    prediction, loss = smoke._lsqplus_task_forward(
        runtime, student_model, teacher_model,
        Adapter(student_capture), Adapter(teacher_capture),
        (target,), target, weights, 0.1)
    loss.total.backward()

    assert not prediction.requires_grad
    assert scale.grad is not None
    assert float(scale.grad.abs().item()) > 0.0
