from __future__ import annotations
from .pulseq_file import PulseqFile  # noqa
from .pulse import Pulse
from .spoiler import Spoiler
from .adc import Adc


# ----- PULSES -----
# Simulated pulses are instantaneous. Pulseq blocks containing pulse events are
# split in the center of the pulse shape and converted into a gradient event
# before the pulse and a pulse + gradient event afterwards. This should result
# in a simulation that has no e.g. slice selection, but where the gradeint
# moments seen by refocussed, excited and unaffected magnetisation are correct.
# ----- ADC -----
# As stated in the specification, all samples are placed at (n + 0.5) * raster,
# which means when there is a gradient and an adc sample at the same time,
# the adc measurement sees only the first half of the gradient. TODO: check
# pulseq exporters and the Siemens interpreter to see if they respect the spec
# or if the adc samples should rather be placed at (n + 0 or 1) * raster.
# ----- GRADIENTS -----
# Only TRAP gradients are currently supported, arbitrary gradients would
# require to resample the gradient onto the ADC time grid. Note that the spec
# does not specify what should happen in between gradient samples, but there
# are two sensible options:
#  - Mimic the scanner, where interpolation is given by the electronics, but a
#    simple linear or bezier or similar interpolation should be a good choice
#  - Adhere to the implementation of the official pulseq exporter, which
#    probably assumes a piecewise constant gradient
# NOTE: As long as we don't simulate diffusion, we don't care about the shape
# of a gradient if it is not measured simultaneously.
# ----- Additional -----
# Simultaneous RF and ADC events are not supported but probably don't exist in
# pracitce anyways.


def intermediate(
    file: PulseqFile
) -> list[tuple[int, Pulse, list[Spoiler | Adc]]]:
    seq = []
    # Convert to intermediate representation
    for block in file.blocks.values():
        assert block.rf_id == 0 or block.adc_id == 0

        if block.rf_id != 0:
            seq += Pulse.parse(block, file)
        elif block.adc_id != 0:
            seq += Adc.parse(block, file)
        else:
            seq.append(Spoiler.parse(block, file))

    reps = []
    # [event_count, pulse, list of events]
    current = [0, None, []]  # Dummy for events before first pulse
    # Split into repetitions
    for block in seq:
        if isinstance(block, Pulse):
            current = [0, block, []]
            reps.append(current)
        else:
            if isinstance(block, Adc):
                current[0] += len(block.event_time)
            else:  # Spoiler
                current[0] += 1
            current[2].append(block)

    return reps
