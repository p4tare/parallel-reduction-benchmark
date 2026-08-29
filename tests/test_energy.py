from pathlib import Path

from prbench.energy import RaplEnergyMeter, RaplZone


def test_rapl_wraparound(tmp_path: Path) -> None:
    energy = tmp_path / "energy_uj"
    energy.write_text("10")
    meter = RaplEnergyMeter()
    meter.zones = [RaplZone("pkg", energy, 100)]
    total, per_zone = meter.delta_j({"pkg": 90}, {"pkg": 10})
    assert abs(total - 20 / 1_000_000) < 1e-12
    assert per_zone["pkg"] == total
