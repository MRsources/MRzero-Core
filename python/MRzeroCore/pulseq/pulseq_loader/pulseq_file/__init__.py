from __future__ import annotations
from . import helpers
from .definitons import Definitions
from .block import parse_blocks, write_blocks, Block
from .rf import parse_rfs, write_rfs, Rf  # noqa
from .trap import parse_traps, write_traps, Trap
from .gradient import parse_gradients, write_grads, Gradient
from .adc import parse_adcs, write_adcs, Adc  # noqa

import matplotlib.pyplot as plt
import numpy as np

# Supports version 1.2.0 to 1.4.0, python representation is modeled after 1.4.0 with pTx
# Supports Martin Freudensprung's pTx extension as pulseq version 1.3.9 and 1.4.5


class PulseqFile:
    def __init__(self, file_name: str) -> None:
        sections = helpers.file_to_sections(file_name)

        assert "VERSION" in sections
        self.version = helpers.parse_version(sections.pop("VERSION"))
        assert 120 <= self.version <= 149

        # mandatory sections
        assert "BLOCKS" in sections
        assert self.version < 140 or "DEFINITIONS" in sections
        assert not (self.version >= 140 and "DELAYS" in sections)

        if "DEFINITIONS" in sections:
            self.definitions = Definitions.parse(
                sections.pop("DEFINITIONS"), self.version)
        else:
            self.definitions = Definitions({}, self.version)

        # Parse [RF], [GRADIENTS], [TRAP], [ADC], [SHAPES]
        # They are dicts of (ID, event) so return an empty dict if not present
        def maybe_parse(sec_name, parser):
            if sec_name not in sections:
                return {}
            else:
                return parser(sections.pop(sec_name), self.version)

        self.rfs = maybe_parse("RF", parse_rfs)
        self.grads = helpers.merge_dicts(
            maybe_parse("GRADIENTS", parse_gradients),
            maybe_parse("TRAP", parse_traps),
        )
        self.adcs = maybe_parse("ADC", parse_adcs)
        self.shapes = maybe_parse("SHAPES", helpers.parse_shapes)

        # Finally parse the blocks, some additional logic is needed to convert
        # 1.3.x sequences with delay events into the 1.4.0 format
        if self.version >= 140:
            self.blocks = parse_blocks(
                sections.pop("BLOCKS"), self.version,
                None, self.definitions.block_raster_time
            )
        else:
            delays = maybe_parse("DELAYS", helpers.parse_delays)
            self.blocks = parse_blocks(
                sections.pop("BLOCKS"), self.version,
                delays, None
            )

        # Inform if there are sections that were not parsed
        if len(sections) > 0:
            print(f"Some sections were ignored: {list(sections.keys())}")

        # Calculate block durations for 1.3.x sequences
        def calc_duration(block: Block) -> float:
            durs = [block.duration]  # delay event for 1.3.x

            if block.adc_id != 0:
                durs.append(self.adcs[block.adc_id].get_duration())

            if block.rf_id != 0:
                durs.append(self.rfs[block.rf_id].get_duration(
                    self.definitions.rf_raster_time, self.shapes
                ))

            grads = [
                self.grads.get(block.gx_id, None),
                self.grads.get(block.gy_id, None),
                self.grads.get(block.gz_id, None)
            ]

            for grad in grads:
                if isinstance(grad, Gradient):
                    durs.append(grad.get_duration(
                        self.definitions.grad_raster_time, self.shapes
                    ))
                if isinstance(grad, Trap):
                    durs.append(grad.get_duration())

            return max(durs)

        for block in self.blocks.keys():
            # We could check if 1.4.0 has set correct durations
            self.blocks[block].duration = calc_duration(self.blocks[block])

    def save(self, file_name: str):
        with open(file_name, "w") as out:
            out.write(
                "# Pulseq sequence definition file\n"
                "# Re-Exported by the MRzero pulseq interpreter\n"
            )
            helpers.write_version(out, 140)
            self.definitions.write(out)
            write_blocks(out, self.blocks, self.definitions.block_raster_time)
            write_rfs(out, self.rfs)
            write_traps(
                out,
                {k: v for k, v in self.grads.items() if isinstance(v, Trap)}
            )
            write_grads(
                out,
                {k: v for k, v in self.grads.items()
                    if isinstance(v, Gradient)}
            )
            write_adcs(out, self.adcs)
            helpers.write_shapes(out, self.shapes)

    def __repr__(self) -> str:
        return (
            f"PulseqFile(version={self.version}, "
            f"definitions={self.definitions}, "
            f"blocks={self.blocks}, "
            f"rfs={self.rfs}, "
            f"adcs={self.adcs}, "
            f"grads={self.grads}, "
            f"shapes={self.shapes})"
        )

    def plot(self, figsize: tuple[float, float] | None = None):
        # Convert the sequence into a plottable format
        rf_plot = []
        adc_plot = []
        gx_plot = []
        gy_plot = []
        gz_plot = []
        t0 = [0.0]

        for block in self.blocks.values():
            if block.rf_id != 0:
                rf_plot.append(get_rf(self.rfs[block.rf_id], self, t0[-1]))
            if block.adc_id != 0:
                adc_plot.append(get_adc(self.adcs[block.adc_id], self, t0[-1]))
            if block.gx_id != 0:
                gx_plot.append(get_grad(self.grads[block.gx_id], self, t0[-1]))
            if block.gy_id != 0:
                gy_plot.append(get_grad(self.grads[block.gy_id], self, t0[-1]))
            if block.gz_id != 0:
                gz_plot.append(get_grad(self.grads[block.gz_id], self, t0[-1]))
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


# Helper functions for plotting

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
