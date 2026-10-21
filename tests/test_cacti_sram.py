import pytest

from spn_quant.hardware.cacti import parse_cacti_output


CACTI_OUTPUT = """
Cache size                    : 4194304
Block size                    : 64
Cache banks (UCA)             : 16
---------- CACTI (version 7.0.0), Uniform Cache Access mode ----------
    Access time (ns): 1.25
    Cycle time (ns):  0.80
    Total dynamic read energy per access (nJ): 2.50
    Total dynamic write energy per access (nJ): 2.75
    Cache height x width (mm): 4.0 x 5.0
"""


def test_parse_cacti_output_extracts_sram_characteristics():
    result = parse_cacti_output(
        CACTI_OUTPUT, expected_capacity_bytes=4 * 1024 * 1024,
        output_width_bits=512)

    assert result.capacity_bytes == 4 * 1024 * 1024
    assert result.access_time_ns == 1.25
    assert result.cycle_time_ns == 0.80
    assert result.area_mm2 == 20.0
    assert result.read_energy_nj == 2.50
    assert result.write_energy_nj == 2.75
    assert result.output_width_bits == 512
    assert result.bank_count == 16


def test_parse_cacti_output_rejects_capacity_mismatch():
    with pytest.raises(ValueError, match="capacity"):
        parse_cacti_output(
            CACTI_OUTPUT, expected_capacity_bytes=2 * 1024 * 1024,
            output_width_bits=512)


def test_parse_cacti_output_rejects_incomplete_output():
    with pytest.raises(ValueError, match="missing"):
        parse_cacti_output(
            "Cache size : 4194304\n", expected_capacity_bytes=4 * 1024 * 1024,
            output_width_bits=512)


def test_parse_cacti_output_requires_byte_aligned_width():
    with pytest.raises(ValueError, match="width"):
        parse_cacti_output(
            CACTI_OUTPUT, expected_capacity_bytes=4 * 1024 * 1024,
            output_width_bits=510)
