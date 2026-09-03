import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts import launch_nyu_three_model_quantization as launcher
from scripts.evaluate_nyu_selected_quantization import SELECTED_METHODS
from spn_quant.experiment_config import MODEL_ORDER


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_CONFIG = REPO_ROOT / \
    "configs/three_model_selected_quantization.json"
LAUNCH_SPEC = REPO_ROOT / \
    "configs/three_model_quantization_launch.json"


def _configuration():
    return launcher.load_launch_configuration(
        EXPERIMENT_CONFIG, LAUNCH_SPEC)


def test_launch_graph_respects_exact_method_dependencies():
    graph = launcher.build_launch_graph(_configuration())

    assert graph.predecessors("validate_static_inputs") == {
        "prepare_static_inputs"}
    assert graph.predecessors("p3_t3_mixed_ptq") == {
        "validate_static_inputs"}
    assert graph.predecessors("hawq_trace") == {
        "validate_static_inputs"}
    assert graph.predecessors("lsqplus_w4a4_qat") == {
        "validate_static_inputs"}
    assert graph.predecessors("lsqplus_w6a6_qat") == {
        "validate_static_inputs"}
    assert graph.predecessors("mixed_task_aware") == {
        "p3_t3_mixed_ptq", "validate_static_inputs"}
    assert graph.predecessors("hawq_allocation") == {"hawq_trace"}
    assert graph.predecessors("hawq_mixed_le6_qat") == {
        "hawq_allocation"}
    assert graph.predecessors("formal_artifacts") == {
        "selected_ptq",
        "hawq_mixed_le6_qat",
        "lsqplus_w6a6_qat",
        "lsqplus_w4a4_qat",
        "mixed_task_aware",
    }
    assert graph.predecessors("evaluate_fp32") == {"formal_artifacts"}


def test_graph_contains_all_models_and_exact_formal_methods():
    graph = launcher.build_launch_graph(_configuration())

    assert graph.models == MODEL_ORDER
    assert len(graph.jobs) == 70
    for model in MODEL_ORDER:
        formal = tuple(
            job.method for job in graph.jobs_for_model(model)
            if job.kind == "formal_evaluation")
        assert formal == SELECTED_METHODS
        assert graph.predecessors(
            "aggregate", model=model) == {
                "evaluate_%s" % SELECTED_METHODS[-1]}
    assert graph.predecessors("cross_model_summary") == {
        "%s:plot" % model for model in MODEL_ORDER}


def test_each_job_has_an_explicit_configured_cuda_device():
    configuration = _configuration()
    expected = dict(
        (model.model, model.device) for model in configuration.experiment.models)

    for job in launcher.build_jobs(configuration):
        assert job.device.startswith("cuda:")
        if job.model is not None:
            assert job.device == expected[job.model]
    assert len(set(expected.values())) == len(MODEL_ORDER)


def test_models_share_evaluation_identity_and_p3t3_quality_policy():
    configuration = _configuration()
    evaluation_indices = tuple(configuration.experiment.models[0].evaluation_indices)
    assert len(evaluation_indices) == 64
    assert evaluation_indices != tuple(range(64))
    assert all(
        tuple(model.evaluation_indices) == evaluation_indices
        for model in configuration.experiment.models)
    assert configuration.spec.p3_t3_policy == {
        "metric_aggregation": "mean_of_per_sample_rmse",
        "maximum_relative_rmse_loss": 0.10,
    }

    jobs = launcher.build_jobs(configuration)
    for model in MODEL_ORDER:
        p3 = next(job for job in jobs
                  if job.job_id == "%s:p3_t3_mixed_ptq" % model)
        assert "--maximum-relative-rmse-loss" in p3.command
        assert p3.command[p3.command.index(
            "--maximum-relative-rmse-loss") + 1] == "0.1"


def test_commands_use_only_declared_python_and_exact_devices():
    configuration = _configuration()
    model_python = dict(
        (model.model, str(model.python_executable))
        for model in configuration.experiment.models)

    for job in launcher.build_jobs(configuration):
        if job.kind in (
                "hawq_allocation", "artifact_index",
                "static_input_validation", "cross_model_summary"):
            assert job.command[0] == str(
                configuration.spec.orchestrator_python)
        else:
            assert job.command[0] == model_python[job.model]
        assert "CUDA_VISIBLE_DEVICES" not in job.environment
        if "--device" in job.command:
            assert job.command[job.command.index("--device") + 1] == \
                job.device


def test_only_qat_jobs_declare_deterministic_cublas_workspace():
    jobs = launcher.build_jobs(_configuration())

    for job in jobs:
        if job.kind == "selected_qat":
            assert job.environment["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
        else:
            assert "CUBLAS_WORKSPACE_CONFIG" not in job.environment


def test_commands_cover_exact_runner_inputs_and_artifact_outputs():
    configuration = _configuration()
    jobs = dict((job.job_id, job)
                for job in launcher.build_jobs(configuration))

    preparation = jobs["dyspn:prepare_static_inputs"]
    assert Path(preparation.command[1]).name == \
        "prepare_nyu_three_model_static_inputs.py"
    assert preparation.device == "cuda:0"
    assert "--device" in preparation.command
    assert len(preparation.produced_outputs) == 4

    validation = jobs["dyspn:validate_static_inputs"]
    assert Path(validation.command[
        validation.command.index("validate-static-inputs") - 1]).name == \
        "launch_nyu_three_model_quantization.py"
    assert set(preparation.produced_outputs + (preparation.output,)) <= \
        set(validation.inputs)

    p3 = jobs["dyspn:p3_t3_mixed_ptq"]
    assert Path(p3.command[1]).name == "run_nyu_model_p3t3_search.py"
    assert "--skip-conv-bn-fold" in p3.command
    assert "--fold-conv-bn" not in p3.command
    assert "--maximum-normalized-weight-cost" in p3.command
    assert "--maximum-normalized-activation-cost" in p3.command
    assert p3.output.name == "p3_t3_assignment.json"

    trace = jobs["nlspn:hawq_trace"]
    assert trace.command[trace.command.index("--phase") + 1] == "trace"
    allocation = jobs["nlspn:hawq_allocation"]
    assert allocation.command[
        allocation.command.index("--phase") + 1] == "allocate"
    assert "--device" not in allocation.command

    mixed = jobs["completionformer:mixed_task_aware"]
    assert mixed.command[mixed.command.index("--method") + 1] == \
        "mixed_task_aware"
    assert mixed.command[mixed.command.index("--p3-t3-assignment") + 1] == \
        str(jobs["completionformer:p3_t3_mixed_ptq"].output)
    assert mixed.command[mixed.command.index("--launch-spec") + 1] == \
        str(LAUNCH_SPEC)
    model_inputs = _configuration().spec.model_inputs["completionformer"]
    assert model_inputs.weight_cost_rows in mixed.inputs
    assert model_inputs.activation_cost_rows in mixed.inputs

    index = jobs["dyspn:formal_artifacts"]
    for method in SELECTED_METHODS:
        evaluation = jobs["dyspn:evaluate_%s" % method]
        assert evaluation.command[
            evaluation.command.index("--artifact-index") + 1] == \
            str(index.output)
        assert evaluation.command[
            evaluation.command.index("--launch-spec") + 1] == \
            str(LAUNCH_SPEC)


def test_selected_qat_commands_are_accepted_with_explicit_fold_policy():
    from scripts.train_nyu_selected_qat import build_parser

    jobs = launcher.build_jobs(_configuration())
    qat = next(job for job in jobs
               if job.job_id == "nlspn:lsqplus_w4a4_qat")

    parsed = build_parser().parse_args(qat.command[2:])

    assert parsed.fold_conv_bn is False
    assert parsed.device == "cuda:1"


def test_launch_spec_rejects_environment_or_interpreter_substitution(tmp_path):
    payload = json.loads(LAUNCH_SPEC.read_text(encoding="utf-8"))
    payload["model_environments"]["dyspn"][
        "CUDA_VISIBLE_DEVICES"] = "3"
    remapped = tmp_path / "remapped.json"
    remapped.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="CUDA_VISIBLE_DEVICES"):
        launcher.load_launch_configuration(EXPERIMENT_CONFIG, remapped)

    payload = json.loads(LAUNCH_SPEC.read_text(encoding="utf-8"))
    payload["orchestrator_python"] = "python"
    relative = tmp_path / "relative.json"
    relative.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="absolute"):
        launcher.load_launch_configuration(EXPERIMENT_CONFIG, relative)

    payload = json.loads(LAUNCH_SPEC.read_text(encoding="utf-8"))
    payload["orchestrator_python"] = str(
        (tmp_path / "missing-python").resolve())
    missing = tmp_path / "missing.json"
    missing.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="Python"):
        launcher.load_launch_configuration(EXPERIMENT_CONFIG, missing)

    payload = json.loads(LAUNCH_SPEC.read_text(encoding="utf-8"))
    payload["orchestrator_device"] = "cuda:auto"
    automatic = tmp_path / "automatic-device.json"
    automatic.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="indexed CUDA"):
        launcher.load_launch_configuration(EXPERIMENT_CONFIG, automatic)


def test_graph_rejects_generated_input_without_transitive_dependency(tmp_path):
    source = tmp_path / "source.json"
    destination = tmp_path / "destination.json"
    common = {
        "model": "dyspn",
        "kind": "test",
        "method": None,
        "device": "cuda:0",
        "command": (sys.executable, "-c", "pass"),
        "environment": {"PYTHONHASHSEED": "0"},
        "dependencies": (),
        "output_policy": "command_file",
    }
    producer = launcher.LaunchJob(
        job_id="dyspn:producer", name="producer", inputs=(), output=source,
        **common)
    consumer = launcher.LaunchJob(
        job_id="dyspn:consumer", name="consumer", inputs=(source,),
        output=destination, **common)

    with pytest.raises(ValueError, match="generated input lacks dependency"):
        launcher.LaunchGraph((producer, consumer))


def test_job_execution_persists_exact_provenance(tmp_path):
    source = tmp_path / "input.txt"
    source.write_text("revision-one\n", encoding="utf-8")
    output = tmp_path / "result.txt"
    manifest = tmp_path / "job.json"
    log = tmp_path / "job.log"
    command = (
        sys.executable,
        "-c",
        "from pathlib import Path; Path(%r).write_text('done\\n')" %
        str(output),
    )
    environment = {"PYTHONHASHSEED": "0"}
    job = launcher.LaunchJob(
        job_id="dyspn:probe",
        model="dyspn",
        name="probe",
        kind="test",
        method=None,
        device="cuda:0",
        command=command,
        environment=environment,
        inputs=(source,),
        output=output,
        dependencies=(),
        output_policy="command_file",
    )

    launcher.write_planned_job_manifest(job, manifest, log)
    planned = json.loads(manifest.read_text(encoding="utf-8"))
    assert planned["input_revisions"] == [{
        "path": str(source.resolve()),
        "size_bytes": source.stat().st_size,
        "sha256": launcher.file_sha256(source),
    }]
    launcher.execute_job(job, manifest, log)
    payload = json.loads(manifest.read_text(encoding="utf-8"))

    assert payload["state"] == "completed"
    assert payload["command"] == list(command)
    assert payload["environment"] == environment
    assert payload["input_revisions"] == [{
        "path": str(source.resolve()),
        "size_bytes": source.stat().st_size,
        "sha256": launcher.file_sha256(source),
    }]
    assert payload["output_path"] == str(output.resolve())
    assert payload["start_time_utc"].endswith("Z")
    assert payload["end_time_utc"].endswith("Z")
    assert payload["exit_status"] == 0
    assert output.read_text(encoding="utf-8") == "done\n"


def test_failed_job_records_nonzero_exit_status(tmp_path):
    manifest = tmp_path / "job.json"
    log = tmp_path / "job.log"
    output = tmp_path / "never-created.txt"
    job = launcher.LaunchJob(
        job_id="dyspn:failure",
        model="dyspn",
        name="failure",
        kind="test",
        method=None,
        device="cuda:0",
        command=(sys.executable, "-c", "raise SystemExit(7)"),
        environment={"PYTHONHASHSEED": "0"},
        inputs=(),
        output=output,
        dependencies=(),
        output_policy="command_file",
    )
    launcher.write_planned_job_manifest(job, manifest, log)

    with pytest.raises(launcher.JobExecutionError, match="status 7"):
        launcher.execute_job(job, manifest, log)

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["state"] == "failed"
    assert payload["exit_status"] == 7
    assert payload["start_time_utc"].endswith("Z")
    assert payload["end_time_utc"].endswith("Z")


def test_cross_model_summary_requires_exact_completed_matrix(tmp_path):
    summaries = []
    for model in MODEL_ORDER:
        path = tmp_path / (model + ".json")
        rows = [{
            "model": model,
            "method": method,
            "configuration": method,
            "samples": 64,
            "pooled_rmse": 0.1,
            "mean_sample_rmse": 0.1,
            "pooled_mae": 0.05,
            "pooled_abs_rel": 0.02,
            "pooled_irmse": 0.2,
        } for method in SELECTED_METHODS]
        path.write_text(json.dumps({
            "model": model,
            "methods": list(SELECTED_METHODS),
            "aggregate_metrics": rows,
        }), encoding="utf-8")
        summaries.append(path)
    output = tmp_path / "cross_model_summary.json"

    launcher.publish_cross_model_summary(tuple(summaries), output)

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["models"] == list(MODEL_ORDER)
    assert payload["methods"] == list(SELECTED_METHODS)
    assert len(payload["aggregate_metrics"]) == 30

    broken = json.loads(summaries[-1].read_text(encoding="utf-8"))
    broken["aggregate_metrics"].pop()
    summaries[-1].write_text(json.dumps(broken), encoding="utf-8")
    with pytest.raises(ValueError, match="method order"):
        launcher.publish_cross_model_summary(
            tuple(summaries), tmp_path / "broken.json")


def test_direct_execution_requires_an_explicit_operation():
    script = REPO_ROOT / "scripts/launch_nyu_three_model_quantization.py"

    completed = subprocess.run(
        (sys.executable, str(script)),
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 2
    assert "operation" in completed.stderr
