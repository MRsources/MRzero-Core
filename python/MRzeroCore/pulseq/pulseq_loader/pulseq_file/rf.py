from __future__ import annotations
from numpy import ndarray


class Rf:
    def __init__(
        self,
        amp: float,
        mag_id: int,
        phase_id: int,
        time_id: int,
        delay: float,  # s (spec: us)
        freq: float,
        phase: float,
        shim_mag_id: int,
        shim_phase_id: int,
    ) -> None:
        self.amp = amp
        self.mag_id = mag_id
        self.phase_id = phase_id
        self.time_id = time_id
        self.delay = delay
        self.freq = freq
        self.phase = phase
        self.shim_mag_id = shim_mag_id
        self.shim_phase_id = shim_phase_id

    @classmethod
    def parse(cls, line: str, version: int) -> tuple[int, Rf]:
        assert 120 <= version <= 149

        vals = line.split()
        rf_id = int(vals.pop(0))
        rf = cls(
            float(vals.pop(0)),
            int(vals.pop(0)),
            int(vals.pop(0)),
            0 if version < 140 else int(vals.pop(0)),
            int(vals.pop(0)) * 1e-6,
            float(vals.pop(0)),
            float(vals.pop(0)),
            int(vals.pop(0)) if version in [139, 145] else 0,
            int(vals.pop(0)) if version in [139, 145] else 0,
        )
        assert len(vals) == 0
        return rf_id, rf

    def write(self, file, rf_id: int):
        file.write(
            f"{rf_id:4d} {self.amp:12g} {self.mag_id:4d} "
            f"{self.phase_id:4d} {self.time_id:4d} "
            f"{round(self.delay*1e6):7d} {self.freq:.6f} {self.phase:.6f}\n"
        )

    def get_duration(self, rf_raster_time: float, shapes: dict[int, ndarray]):
        if self.time_id != 0:
            last_sample = shapes[self.time_id][-1]
        else:
            last_sample = len(shapes[self.mag_id])
        return self.delay + last_sample * rf_raster_time

    def __repr__(self) -> str:
        return (
            f"RF(amp={self.amp}, "
            f"mag_id={self.mag_id}, "
            f"phase_id={self.phase_id}, "
            f"time_id={self.time_id}, "
            f"delay={self.delay}, "
            f"freq={self.freq}, "
            f"phase={self.phase})"
            f"mag_id={self.shim_mag_id}, "
            f"phase_id={self.shim_phase_id}"
        )


def parse_rfs(lines: list[str], version: int) -> dict[int, Rf]:
    tmp = {}
    for line in lines:
        key, value = Rf.parse(line, version)
        assert key > 0 and key not in tmp
        tmp[key] = value
    return tmp


def write_rfs(file, rfs: dict[int, Rf]):
    file.write(
        "\n[RF]\n"
        "# ID          amp  mag angl time   delay     freq    phase\n"
    )
    for rf_id, rf in rfs.items():
        rf.write(file, rf_id)
