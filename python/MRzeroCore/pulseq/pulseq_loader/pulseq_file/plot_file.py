from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
from .trap import Trap
from .gradient import Gradient
from .rf import Rf
from .adc import Adc

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from . import PulseqFile


def plot_file(seq: PulseqFile, figsize: tuple[float, float] | None = None):
    # Convert the sequence into a plottable format
    rf_plot = []
    adc_plot = []
    gx_plot = []
    gy_plot = []
    gz_plot = []
    t0 = [0.0]

    for block in seq.blocks.values():
        if block.rf_id != 0:
            rf_plot.append(get_rf(seq.rfs[block.rf_id], seq, t0[-1]))
        if block.adc_id != 0:
            adc_plot.append(get_adc(seq.adcs[block.adc_id], seq, t0[-1]))
        if block.gx_id != 0:
            gx_plot.append(get_grad(seq.grads[block.gx_id], seq, t0[-1]))
        if block.gy_id != 0:
            gy_plot.append(get_grad(seq.grads[block.gy_id], seq, t0[-1]))
        if block.gz_id != 0:
            gz_plot.append(get_grad(seq.grads[block.gz_id], seq, t0[-1]))
        t0.append(t0[-1] + block.duration)

    # Plot the aquired data
    plt.figure(figsize=figsize)

    ax1 = plt.subplot(311)
    plt.title("RF")
    for rf in rf_plot:
        ax1.plot(rf[0], rf[1].real, c="tab:blue")
        ax1.plot(rf[0], rf[1].imag, c="tab:orange")
    plt.grid()
    plt.ylabel("Hz")

    ax2 = plt.subplot(312, sharex=ax1)
    plt.title("ADC")
    for adc in adc_plot:
        ax2.plot(adc[0], adc[1], '.')
    for t in t0:
        plt.axvline(t, c="#0004")
    plt.grid()
    plt.ylabel("rad")

    ax3 = plt.subplot(313, sharex=ax1)
    plt.title("Gradients")
    for grad in gx_plot:
        ax3.plot(grad[0], grad[1], c="tab:blue")
    for grad in gy_plot:
        ax3.plot(grad[0], grad[1], c="tab:orange")
    for grad in gz_plot:
        ax3.plot(grad[0], grad[1], c="tab:green")
    plt.grid()
    plt.xlabel("time [s]")
    plt.ylabel("Hz/m")

    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.show()


def get_rf(rf: Rf, seq: PulseqFile, t0: float
           ) -> tuple[np.ndarray, np.ndarray]:
    if rf.time_id != 0:
        time = seq.shapes[rf.time_id]
    else:
        time = np.arange(len(seq.shapes[rf.mag_id]))

    time = t0 + rf.delay + (time + 0.5) * seq.definitions.rf_raster_time
    mag = rf.amp * seq.shapes[rf.mag_id]
    phase = rf.phase + 2*np.pi * seq.shapes[rf.phase_id]

    return time, mag * np.exp(1j * phase)


def get_adc(adc: Adc, seq: PulseqFile, t0: float
            ) -> tuple[np.ndarray, np.ndarray]:
    time = t0 + adc.delay + (np.arange(adc.num) + 0.5) * adc.dwell
    return time, adc.phase * np.ones(adc.num)


def get_grad(grad: Gradient | Trap, seq: PulseqFile, t0: float
             ) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(grad, Gradient):
        if grad.time_id != 0:
            time = seq.shapes[grad.time_id]
        else:
            time = np.arange(len(seq.shapes[grad.shape_id]))
        time = grad.delay + (time + 0.5) * seq.definitions.grad_raster_time
        shape = grad.amp * seq.shapes[grad.shape_id]
    else:
        assert isinstance(grad, Trap)
        time = grad.delay + np.array([
            0.0,
            grad.rise,
            grad.rise + grad.flat,
            grad.rise + grad.flat + grad.fall
        ])
        shape = np.array([0, grad.amp, grad.amp, 0])

    return t0 + time, shape
