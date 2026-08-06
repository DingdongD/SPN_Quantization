from scripts.run_nyu_strict_reconstruction import (
    maximum_primary_fold_error,
    parse_args,
)


def test_fold_validation_uses_primary_depth_output_error():
    teacher = {
        "max_abs_error": 4.0,
        "primary_max_abs_error": 1.0e-6,
    }
    student = {
        "max_abs_error": 6.0,
        "primary_max_abs_error": 2.0e-6,
    }

    assert maximum_primary_fold_error(teacher, student) == 2.0e-6


def test_runner_accepts_explicit_data_root():
    args = parse_args([
        "--run-dir", "run",
        "--method", "adaround_strict",
        "--data-root", "dataset",
    ])

    assert args.data_root == "dataset"
