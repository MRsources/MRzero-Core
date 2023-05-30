from __future__ import annotations
import numpy as np
from .pulseq_file import PulseqFile, Gradient, Trap


def split_gradm(grad: Gradient | Trap, pulseq: PulseqFile, t: float
                ) -> tuple[float, float]:
    before = integrate(grad, pulseq, t)
    total = integrate(grad, pulseq, float("inf"))
    return (before, total - before)


def integrate(grad: Gradient | Trap, pulseq: PulseqFile, t: float) -> float:
    if isinstance(grad, Trap):
        t -= grad.delay
        total = grad.rise/2 + grad.flat + grad.fall/2

        if t <= 0.0:
            integral = 0.0
        elif t < grad.rise:
            integral = 0.5*t**2 / grad.rise
        elif t < grad.rise + grad.flat:
            t -= grad.rise
            integral = grad.rise/2 + t
        elif t < grad.rise + grad.flat + grad.fall:
            t = grad.rise + grad.flat + grad.fall - t
            integral = total - 0.5*t**2 / grad.fall
        else:
            integral = total
        return grad.amp * integral
    else:
        assert isinstance(grad, Gradient)
        # How many gradient samples are there before t?
        # We assume that adc aligns with the gradient raster, otherwise we
        # would have to integrate over an interpolatet version of the gradient
        shape = pulseq.shapes[grad.shape_id]
        raster_time = pulseq.definitions.grad_raster_time
        t -= grad.delay

        if grad.time_id != 0:
            time = pulseq.shapes[grad.time_id] * raster_time
        else:
            time = np.arange(len(shape)) * raster_time
        event_time = np.concatenate([np.diff(time), [raster_time]])
        t = min(t, time[-1])

        # Cut off all events after t
        mask = time < t
        time = time[mask]
        event_time = event_time[mask]
        shape = shape[mask]

        if len(time) == 0:
            return 0.0

        # Sum over all samples that end before t
        integral = np.sum(shape[:-1] * event_time[:-1])
        # Add the last sample that might only be partially before t
        integral += shape[-1] * (t - time[-1])

        return grad.amp * integral


def total_gradm(grad: Gradient | Trap, pulseq: PulseqFile) -> float:
    if isinstance(grad, Trap):
        return grad.amp * (grad.rise/2 + grad.flat + grad.fall/2)
    else:
        assert isinstance(grad, Gradient)
        shape = pulseq.shapes[grad.shape_id]
        raster_time = pulseq.definitions.grad_raster_time
        if grad.time_id != 0:
            event_time = np.concatenate([np.diff(shape), [1]]) * raster_time
        else:
            event_time = np.full((len(shape), ), raster_time)
        return np.sum(grad.amp * shape * event_time)
