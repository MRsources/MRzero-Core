from __future__ import annotations


class Trap:
    def __init__(
        self,
        amp: float,  # Hz / m
        rise: float,  # s (spec: us)
        flat: float,  # s (spec: us)
        fall: float,  # s (spec: us)
        delay: float,  # s (spec: us)
    ) -> None:
        self.amp = amp
        self.rise = rise
        self.flat = flat
        self.fall = fall
        self.delay = delay

    @classmethod
    def parse(cls, line: str, version: int) -> tuple[int, Trap]:
        assert 120 <= version <= 149
        vals = line.split()
        assert len(vals) == 6
        grad_id = int(vals[0])

        return grad_id, cls(
            float(vals[1]),
            int(vals[2]) * 1e-6,
            int(vals[3]) * 1e-6,
            int(vals[4]) * 1e-6,
            int(vals[5]) * 1e-6,
        )

    def write(self, file, grad_id: int):
        file.write(
            f"{grad_id:4d} {self.amp:12g} {round(self.rise*1e6):7d} "
            f"{round(self.flat*1e6):7d} {round(self.fall*1e6):7d} "
            f"{round(self.delay*1e6):7d}\n"
        )

    def get_duration(self):
        return self.delay + self.rise + self.flat + self.fall

    def __repr__(self) -> str:
        return (
            f"TRAP(amp={self.amp}, "
            f"rise={self.rise}, "
            f"flat={self.flat}, "
            f"fall={self.fall}, "
            f"delay={self.delay})"
        )


def parse_traps(lines: list[str], version: int) -> dict[int, Trap]:
    tmp = {}
    for line in lines:
        key, value = Trap.parse(line, version)
        assert key > 0 and key not in tmp
        tmp[key] = value
    return tmp


def write_traps(file, traps: dict[int, Trap]):
    file.write(
        "\n[TRAP]\n"
        "# ID          amp    rise    flat    fall   delay\n"
    )
    for grad_id, trap in traps.items():
        trap.write(file, grad_id)
