from scripts.plot_nyu_cspn_rotation import (
    aggregate_results,
    display_config_label,
    register_arial_font,
)


def test_aggregate_selects_named_best_rotation_only():
    rows = [
        {"config": "FP32", "RMSE": "0.1", "MAE": "0.05",
         "ABS_REL": "0.01", "IRMSE": "0.02",
         "flat_RMSE": "0.08", "boundary_RMSE": "0.2"},
        {"config": "RTN_W4A4", "RMSE": "0.8", "MAE": "0.4",
         "ABS_REL": "0.1", "IRMSE": "0.2",
         "flat_RMSE": "0.7", "boundary_RMSE": "1.0"},
        {"config": "GROUP_W4A4", "RMSE": "0.6", "MAE": "0.3",
         "ABS_REL": "0.08", "IRMSE": "0.15",
         "flat_RMSE": "0.5", "boundary_RMSE": "0.9"},
        {"config": "RANDOM_both", "RMSE": "0.7", "MAE": "0.35",
         "ABS_REL": "0.09", "IRMSE": "0.17",
         "flat_RMSE": "0.6", "boundary_RMSE": "0.95"},
        {"config": "HADAMARD_both", "RMSE": "0.5", "MAE": "0.25",
         "ABS_REL": "0.07", "IRMSE": "0.12",
         "flat_RMSE": "0.4", "boundary_RMSE": "0.8"},
    ]

    summary, best = aggregate_results(rows)

    assert best == "HADAMARD_both"
    by_name = dict((row["config"], row) for row in summary)
    assert by_name["FP32"]["is_rotation"] == 0
    assert by_name["GROUP_W4A4"]["is_rotation"] == 0
    assert by_name["RANDOM_both"]["is_rotation"] == 1


def test_register_arial_font_uses_real_arial_file():
    name, path = register_arial_font()

    assert name == "Arial"
    assert path.name.lower() == "arial.ttf"


def test_display_config_label_wraps_long_names_without_rotation():
    assert display_config_label(
        "HADAMARD_layer4_signed_skip") == "Hadamard\nLayer4 Skip"
    assert display_config_label(
        "HADAMARD_GROUP_both") == "Hadamard + Group\nBoth"
