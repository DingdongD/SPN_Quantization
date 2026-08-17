import json
from pathlib import Path

import numpy as np
import pytest

from scripts import run_nlspn_in_memory_gop2_benchmark as runner


def test_array_pipe_round_trip_preserves_flow():
    flow = np.arange(2 * 3 * 4, dtype=np.float32).reshape(1, 2, 3, 4)
    encoded = runner.encode_array(flow)
    decoded = runner.decode_array(encoded)
    assert decoded.dtype == np.float32
    np.testing.assert_array_equal(decoded, flow)


def test_raft_parity_records_hard_tolerances():
    reference = np.ones((1, 2, 228, 304), dtype=np.float32)
    result = runner.compare_raft_flows(reference, reference.copy())
    assert result["passes"] is True
    assert result["max_abs"] == 0.0
    assert result["rmse"] == 0.0
    assert result["max_abs_tolerance"] == 1e-3
    assert result["rmse_tolerance"] == 1e-4


def test_raft_parity_rejects_changed_flow():
    reference = np.zeros((1, 2, 228, 304), dtype=np.float32)
    changed = reference.copy()
    changed[..., 10, 10] = 0.01
    with pytest.raises(RuntimeError, match="parity"):
        runner.require_raft_parity(reference, changed)


def test_raft_parity_rejects_wrong_shape():
    with pytest.raises(ValueError, match="shape"):
        runner.compare_raft_flows(
            np.zeros((1, 2, 228, 304), dtype=np.float32),
            np.zeros((1, 2, 10, 10), dtype=np.float32),
        )


def test_build_worker_command_uses_old_environment_and_pythonpath(tmp_path):
    command, environment = runner.build_worker_command(
        stage="raft-parity",
        data_root=tmp_path / "data",
        scene="scene",
        checkpoint=tmp_path / "best.pt",
        args_json=tmp_path / "args.json",
        raft_weights=tmp_path / "raft.pth",
        device="cuda:0",
    )
    assert command[:6] == [
        "conda", "run", "-n", "completionformer-py37", "python",
        str(runner.WORKER_PATH)]
    assert command[command.index("--stage") + 1] == "raft-parity"
    assert str(runner.REPO_ROOT) in environment["PYTHONPATH"]
    assert str(runner.NLSPN_ROOT / "src") in environment["PYTHONPATH"]


def test_parse_worker_pipe_payload_decodes_array():
    flow = np.ones((1, 2, 3, 4), dtype=np.float32)
    line = json.dumps({
        "array": runner.encode_array(flow),
        "shape": list(flow.shape),
        "dtype": str(flow.dtype),
        "weight_sha256": "abc",
    })
    array, metadata = runner.parse_worker_pipe_payload("noise\n" + line + "\n")
    np.testing.assert_array_equal(array, flow)
    assert metadata["weight_sha256"] == "abc"

