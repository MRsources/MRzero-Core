from __future__ import annotations
import numpy as np
from .pulseq_file import PulseqFile, Block
from .helpers import total_gradm


class Spoiler:
    def __init__(
        self,
        duration: float,
        gradm: np.ndarray
    ) -> None:
        self.duration = duration
        self.gradm = gradm

    @classmethod
    def parse(cls, block: Block, pulseq: PulseqFile) -> Spoiler:
        fov = pulseq.definitions.fov
        gradm = np.zeros(3)
        if block.gx_id != 0:
            gradm[0] = total_gradm(pulseq.grads[block.gx_id], pulseq) * fov[0]
        if block.gy_id != 0:
            gradm[1] = total_gradm(pulseq.grads[block.gy_id], pulseq) * fov[1]
        if block.gz_id != 0:
            gradm[2] = total_gradm(pulseq.grads[block.gz_id], pulseq) * fov[2]

        return cls(block.duration, gradm)

    def __repr__(self) -> str:
        return (
            f"Spoiler(duration={self.duration*1e3:.1f}ms, "
            f"gradm={self.gradm})"
        )
