from __future__ import annotations


class Adc:
    def __init__(
        self,
        num: int,
        dwell: float,  # s (spec: ns)
        delay: float,  # s (spec: us)
        freq: float,  # Hz
        phase: float,  # rad
    ) -> None:
        self.num = num
        self.dwell = dwell
        self.delay = delay
        self.freq = freq
        self.phase = phase

    @classmethod
    def parse(cls, line: str, version: int) -> tuple[int, Adc]:
        assert 120 <= version <= 149
        vals = line.split()
        assert len(vals) == 6

        adc_id = int(vals[0])
        return adc_id, cls(
            int(vals[1]),
            float(vals[2]) * 1e-9,
            int(vals[3]) * 1e-6,
            float(vals[4]),
            float(vals[5]),
        )

    def write(self, file, adc_id: int):
        file.write(
            f"{adc_id:4d} {self.num:7d} {round(self.dwell*1e9):7d} "
            f"{round(self.delay*1e6):7d} {self.freq:.6f} {self.phase:.6f}\n"
        )

    def get_duration(self) -> float:
        return self.num * self.dwell + self.delay

    def __repr__(self) -> str:
        return (
            f"ADC(num={self.num}, "
            f"dwell={self.dwell}, "
            f"delay={self.delay}, "
            f"freq={self.freq}, "
            f"phase={self.phase})"
        )


def parse_adcs(lines: list[str], version: int) -> dict[int, Adc]:
    tmp = {}
    for line in lines:
        key, value = Adc.parse(line, version)
        assert key > 0 and key not in tmp
        tmp[key] = value
    return tmp


def write_adcs(file, adcs: dict[int, Adc]):
    file.write(
        "\n[ADC]\n"
        "# ID     num   dwell   delay     freq    phase\n"
    )
    for adc_id, adc in adcs.items():
        adc.write(file, adc_id)
