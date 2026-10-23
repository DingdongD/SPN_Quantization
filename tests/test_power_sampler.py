from pathlib import Path

from scripts import sample_hardware_power as sampler
from spn_quant.energy import parse_power_trace


class FakeReader:
    source = "fake"
    scope = "test_rail"

    def __init__(self):
        self.value = 0

    def read_milliwatts(self):
        self.value += 1000
        return {"test": self.value}

    def close(self):
        pass


class ThreeSampleStop:
    def __init__(self):
        self.waits = 0

    def is_set(self):
        return self.waits >= 2

    def wait(self, interval):
        assert interval == 0.05
        self.waits += 1
        return self.is_set()


def test_sample_stream_writes_parseable_completed_trace(tmp_path: Path):
    output = tmp_path / "trace.jsonl"
    timestamps = iter([1, 2, 3])

    sampler.sample_stream(
        FakeReader(), output, interval_seconds=0.05,
        stop_event=ThreeSampleStop(),
        monotonic_ns=lambda: next(timestamps))

    parsed = parse_power_trace(output)
    assert parsed["metadata"]["source"] == "fake"
    assert parsed["metadata"]["scope"] == "test_rail"
    assert [sample["power_w"]["test"] for sample in parsed["samples"]] == [
        1.0, 2.0, 3.0]


def test_jtop_reader_extracts_total_and_named_ina3221_rails():
    values = sampler.jtop_power_milliwatts({
        "tot": {"power": 17000},
        "rail": {
            "GPU": {"power": 900},
            "SOC": {"power": 2700},
        },
    })

    assert values == {
        "total_module": 17000.0,
        "gpu": 900.0,
        "soc": 2700.0,
    }
