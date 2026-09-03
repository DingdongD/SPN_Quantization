from pathlib import Path

from scripts import launch_nyu_cspn_lsqplus_hawq as launcher


def _args():
    config = Path(__file__).resolve().parents[1] / \
        "configs/cspn_lsqplus_hawq.json"
    return launcher.parse_args([
        "--config", str(config),
        "--checkpoint", "checkpoint.pt",
        "--data-root", "data",
        "--calibration-metadata", "calibration.json",
        "--mixed-source-root", "mixed",
        "--output-root", "output",
    ])


def test_phase_one_uses_all_four_distinct_gpus():
    phases = launcher.build_phases(_args())

    assert tuple(job.gpu for job in phases[0]) == (0, 1, 2, 3)
    assert tuple(job.name for job in phases[0]) == (
        "baselines", "lsqplus_w4a4", "lsqplus_w6a6", "hawq")


def test_hawq_trace_precedes_hawq_training_with_exact_assignment():
    hawq = launcher.build_phases(_args())[0][3]
    trace, training = hawq.commands

    assert Path(trace[1]).name == "run_nyu_cspn_hawq_trace.py"
    assignment_index = training.index("--assignment") + 1
    assert training[assignment_index] == str(
        Path("output/hawq_trace/selected_assignment.json").resolve())


def test_second_phase_evaluates_three_methods_on_their_training_gpus():
    phases = launcher.build_phases(_args())

    assert tuple(job.gpu for job in phases[1]) == (1, 2, 3)
    configurations = tuple(
        job.commands[0][job.commands[0].index("--configuration") + 1]
        for job in phases[1])
    assert configurations == (
        "LSQPLUS_W4A4", "LSQPLUS_W6A6", "HAWQ_MIXED_LE6")


def test_final_commands_aggregate_then_plot():
    phases = launcher.build_phases(_args())

    aggregate, plot = phases[2][0].commands
    assert "--aggregate" in aggregate
    assert Path(plot[1]).name == "plot_nyu_cspn_lsqplus_hawq.py"
