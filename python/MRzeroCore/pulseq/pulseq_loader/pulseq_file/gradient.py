from __future__ import annotations
from numpy import ndarray


class Gradient:
    def __init__(
        self,
        amp: float,  # Hz / m
        shape_id: int,
        time_id: int,
        delay: float  # s (spec: us)
    ) -> None:
        self.amp = amp
        self.shape_id = shape_id
        self.time_id = time_id
        self.delay = delay

    @classmethod
    def parse(cls, line: str, version: int) -> tuple[int, Gradient]:
        assert 120 <= version <= 149
        vals = line.split()

        gradient_id = int(vals.pop(0))
        gradient = cls(
            float(vals.pop(0)),
            int(vals.pop(0)),
            int(vals.pop(0)) if version >= 140 else 0,  # default raster
            int(vals.pop(0)) * 1e-6,
        )
        assert len(vals) == 0
        return gradient_id, gradient

    def write(self, file, grad_id: int):
        file.write(
            f"{grad_id:4d} {self.amp:12g} {self.shape_id:4d} "
            f"{self.time_id:4d} {round(self.delay*1e6):7d}\n"
        )

    def get_duration(self, grad_raster_t: float, shapes: dict[int, ndarray]):
        if self.time_id != 0:
            last_sample = shapes[self.time_id][-1]
        else:
            last_sample = len(shapes[self.shape_id])
        return self.delay + last_sample * grad_raster_t

    def __repr__(self) -> str:
        return (
            f"Gradient(amp={self.amp}, "
            f"shape_id={self.shape_id}, "
            f"time_id={self.time_id}, "
            f"delay={self.delay})"
        )


def parse_gradients(lines: list[str], version: int) -> dict[int, Gradient]:
    tmp = {}
    for line in lines:
        key, value = Gradient.parse(line, version)
        assert key > 0 and key not in tmp
        tmp[key] = value
    return tmp


def write_grads(file, grads: dict[int, Gradient]):
    file.write(
        "\n[GRADIENTS]\n"
        "# ID          amp  mag time   delay\n"
    )
    for grad_id, trap in grads.items():
        trap.write(file, grad_id)
