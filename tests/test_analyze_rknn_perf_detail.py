import json
from pathlib import Path

from scripts.analyze_rknn_perf_detail import analyze_text, parse_perf_detail


DETAIL = """api_version=2.5.0
driver_version=0.9.8
run_duration_us=1234
perf_detail_begin
ID   OpType             DataType Target InputShape                               OutputShape            Cycles(DDR/NPU/Total)    Time(us)     MacUsage(%)          WorkLoad(0/1/2)      RW(KB)       FullName
1    Conv               FLOAT16  NPU    (1,4,8,8),(8,4,3,3)                     (1,8,8,8)              1/2/3                    100          50.0/0.0/0.0         100%/0%/0%            2            Conv:/encoder/Conv
2    GridSample         FLOAT16  CPU    (1,1,8,8),(1,8,8,2)                     (1,1,8,8)              0/0/0                    25                                0%/0%/0%              1            GridSample:/prop_layer/deform/GridSample
3    MatMul             FLOAT16  NPU    (1,2,4),(1,4,4)                         (1,2,4)                1/2/3                    40           60.0/0.0/0.0         100%/0%/0%            1            MatMul:/backbone/block/attn/qk/MatMul
4    Reshape            FLOAT16  NPU    (1,8,8,8),(4)                           (1,64,8)               0/0/0                    5                                 100%/0%/0%            1            Reshape:/decoder/Reshape
perf_detail_end
"""


def test_parse_perf_detail_uses_fixed_columns_with_blank_mac_usage():
    metadata, rows = parse_perf_detail(DETAIL)

    assert metadata["api_version"] == "2.5.0"
    assert metadata["run_duration_us"] == 1234
    assert len(rows) == 4
    assert rows[1]["target"] == "CPU"
    assert rows[1]["time_us"] == 25
    assert rows[1]["mac_usage"] == ""
    assert rows[1]["full_name"].endswith("/prop_layer/deform/GridSample")


def test_analyze_text_applies_scope_precedence_and_target_breakdown():
    result = analyze_text(DETAIL, model="completionformer")

    assert result["category_time_us"] == {
        "dense_compute": 100,
        "attention": 40,
        "sampling": 0,
        "propagation": 25,
        "elementwise_norm": 0,
        "layout_memory": 5,
        "other_unclassified": 0,
    }
    assert result["target_time_us"] == {"CPU": 25, "NPU": 145}
    assert result["category_target_time_us"]["propagation"] == {
        "CPU": 25,
        "NPU": 0,
    }
    assert result["operator_time_sum_us"] == 170
    assert result["placement"]["cpu_operator_count"] == 1
    assert result["placement"]["npu_operator_count"] == 3
    assert result["placement"]["cpu_operators"][0]["op_type"] == "GridSample"
    assert abs(sum(result["category_share_percent"].values()) - 100.0) < 1e-9


def test_explicit_propagation_id_range_recovers_compiler_fused_scope():
    result = analyze_text(
        DETAIL, model="cspn", propagation_id_range=(2, 3))

    assert result["category_time_us"]["propagation"] == 65
    assert result["category_time_us"]["attention"] == 0
    assert result["propagation_selector"] == {
        "scope_tokens": ["post_process_layer"],
        "operator_id_range_inclusive": [2, 3],
    }


def test_analysis_is_json_serializable():
    encoded = json.dumps(analyze_text(DETAIL, model="completionformer"))
    assert "RKNN_QUERY_PERF_DETAIL" in encoded
