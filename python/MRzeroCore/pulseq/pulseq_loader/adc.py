from __future__ import annotations
import numpy as np
from .pulseq_file import PulseqFile, Block
from .helpers import integrate
from .spoiler import Spoiler


class Adc:
    def __init__(self, event_time: np.ndarray, gradm: np.ndarray, phase: float
                 ) -> None:
        self.event_time = event_time
        self.gradm = gradm
        self.phase = phase

    @classmethod
    def parse(cls, block: Block, pulseq: PulseqFile) -> tuple[Adc, Spoiler]:
        adc = pulseq.adcs[block.adc_id]
        time = np.concatenate([
            [0.0],
            adc.delay + np.arange(adc.num) * adc.dwell,
            [block.duration]
        ])

        gradm = np.zeros((adc.num + 1, 3))
        if block.gx_id != 0:
            grad = pulseq.grads[block.gx_id]
            gradm[:, 0] = np.diff([integrate(grad, pulseq, t) for t in time])
        if block.gy_id != 0:
            grad = pulseq.grads[block.gy_id]
            gradm[:, 1] = np.diff([integrate(grad, pulseq, t) for t in time])
        if block.gz_id != 0:
            grad = pulseq.grads[block.gz_id]
            gradm[:, 2] = np.diff([integrate(grad, pulseq, t) for t in time])

        event_time = np.diff(time)

        fov = pulseq.definitions.fov
        gradm[:, 0] *= fov[0]
        gradm[:, 1] *= fov[1]
        gradm[:, 2] *= fov[2]

        return (
            cls(event_time[:-1], gradm[:-1, :], adc.phase),
            Spoiler(event_time[-1], gradm[-1, :])
        )

    def __repr__(self) -> str:
        return f"ADC(event_time={self.event_time}, gradm={self.gradm})"
