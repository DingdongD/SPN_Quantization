"""Frozen compressed-model sources for the RK3588 deployment study."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple


@dataclass(frozen=True)
class CompressedCandidate:
    model: str
    nas_candidate: str
    nas_depths: Tuple[int, ...]
    structured_candidate: str
    state_source: Path
    model_build_source: Path
    quantization_source: Path
    research_precision_policy: str
    rk3588_execution_precisions: Tuple[str, str]
    propagation_lowering: str
    standalone_checkpoint: bool


def candidate_inventory(data_root: Path) -> Dict[str, CompressedCandidate]:
    data_root = Path(data_root).resolve()
    profiles = data_root / "profile_logs"
    cspn_checkpoint = (
        data_root.parent / "CSPN" / "cspn_pytorch" / "output" /
        "nyu_converged_baselines" / "cspn_iter24" / "best.pt")
    candidates = (
        CompressedCandidate(
            model="cspn", nas_candidate="drop_s4",
            nas_depths=(2, 2, 2, 1),
            structured_candidate="depth_pruned_drop_s4",
            state_source=cspn_checkpoint,
            model_build_source=cspn_checkpoint,
            quantization_source=profiles / "four_model_nas_quant_fullval_v1" /
            "cspn" / "summary.json",
            research_precision_policy="decoder_stage2_W4_decoder_stage3_A6",
            rk3588_execution_precisions=("fp16", "w8a8"),
            propagation_lowering="native_convolution",
            standalone_checkpoint=False),
        CompressedCandidate(
            model="dyspn", nas_candidate="drop_s4",
            nas_depths=(3, 4, 6, 2),
            structured_candidate="bridge62_s5hidden62_s4hidden_50pct",
            state_source=profiles / "final_compressed_checkpoints" /
            "dyspn_drop_s4_s4hidden50_w4a6_s3w6_bf16_joint_v2.pt",
            model_build_source=profiles / "final_compressed_checkpoints" /
            "dyspn_drop_s4_s4hidden50_w4a6_s3w6_bf16_joint_v2.pt",
            quantization_source=profiles / "final_compressed_checkpoints" /
            "dyspn_drop_s4_s4hidden50_w4a6_s3w6_bf16_joint_v2.pt",
            research_precision_policy="S4_W4A6_S3_W6_BF16_PROPAGATION",
            rk3588_execution_precisions=("fp16", "w8a8"),
            propagation_lowering="grid_sample_legacy_and_packed",
            standalone_checkpoint=True),
        CompressedCandidate(
            model="nlspn", nas_candidate="drop_s4",
            nas_depths=(3, 4, 6, 2),
            structured_candidate="bridge62_s5hidden62_s4hidden_62p5pct",
            state_source=profiles / "joint_nas_prune_quant_search_v1" /
            "nlspn_s4hidden62p5_recovery_v1" / "best.pt",
            model_build_source=(
                data_root.parent / "CSPN" / "cspn_pytorch" / "output" /
                "nyu_converged_baselines" / "nlspn_iter18" / "best.pt"),
            quantization_source=profiles / "joint_nas_prune_quant_search_v1" /
            "nlspn_s4hidden62p5_recovered_s5w4_s4a6_taila6_full654" /
            "summary.json",
            research_precision_policy="S5_W4_S4_A6_TAIL_A6_BF16_PROPAGATION",
            rk3588_execution_precisions=("fp16", "w8a8"),
            propagation_lowering="grid_sample_legacy_and_packed",
            standalone_checkpoint=False),
        CompressedCandidate(
            model="completionformer", nas_candidate="drop_pvt_s3_s4",
            nas_depths=(3, 4, 3, 4, 3, 2),
            structured_candidate="mlp62_s4hidden62_s3hidden_20pct",
            state_source=profiles / "final_compressed_checkpoints" /
            "completionformer_drop_s3s4_s3hidden20_qkvw6_outw6_mlpw4_"
            "bf16_joint_v2.pt",
            model_build_source=profiles / "final_compressed_checkpoints" /
            "completionformer_drop_s3s4_s3hidden20_qkvw6_outw6_mlpw4_"
            "bf16_joint_v2.pt",
            quantization_source=profiles / "final_compressed_checkpoints" /
            "completionformer_drop_s3s4_s3hidden20_qkvw6_outw6_mlpw4_"
            "bf16_joint_v2.pt",
            research_precision_policy="QKV_W6_OUT_W6_MLP_W4_BF16_PROPAGATION",
            rk3588_execution_precisions=("fp16", "w8a8"),
            propagation_lowering="grid_sample_legacy_and_packed",
            standalone_checkpoint=True),
    )
    result = {candidate.model: candidate for candidate in candidates}
    for candidate in result.values():
        if not candidate.state_source.is_file():
            raise FileNotFoundError(candidate.state_source)
        if not candidate.model_build_source.is_file():
            raise FileNotFoundError(candidate.model_build_source)
        if not candidate.quantization_source.is_file():
            raise FileNotFoundError(candidate.quantization_source)
    return result
