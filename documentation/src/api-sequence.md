# MRI Sequence definitions

The MRzeroCore `Sequence` is a Python list of `Repetition`s, each of which is starting with an instantaneous `Pulse`.
To construct sequences, you can
- load from .seq files: [Pulseq Integration](../integration-pulseq.md)
- construct with pulseq-zero: [Pulseq-zero](../integration-pulseqzero.md)
- defined in the MR-zero directly:

```python
import MRzeroCore as mr0
import numpy as np

seq = mr0.Sequence()

# Iterate over the repetitions
for i in range(64):
    rep = seq.new_rep(2 + 64 + 2)
    
    # Set the pulse
    rep.pulse.usage = mr0.PulseUsage.EXCIT
    rep.pulse.angle = 5 * np.pi/180
    rep.pulse.phase = i**2 * 117*np.pi/180

    # Set encoding
    rep.gradm[:, :] = ...

    # Set timing
    rep.event_time[:] = ...

    # Set ADC
    rep.adc_usage[2:-2] = 1
    rep.adc_phase[2:-2] = np.pi - rep.pulse.phase
```

## `Sequence`

Defines an MRI sequence. Derived from a Python `list`.

```python
Sequence(repetitions, normlized_grads=True)
```

| Parameter | Description |
| --------- | ----------- |
| `repetition` | Iterable over `Repetition`s to initialize the list |
| `normalize_grads` | If set, simulation will scale gradients to match the phantom size. Simplifies writing sequences by assuming a FOV of 1. |

- `Sequence.cpu()`: move sequence data  to CPU
- `Sequence.cuda(device=None)`: move sequence data to selected or default CUDA device
- `Sequence.device` (property): the device of the repetition data
- `Sequence.clone()`: make a copy of the repetition with cloned
- `Sequence.new_rep(event_count)`: appends a `Repetition.zero(event_count)` to the sequence
- `Sequence.get_kspace()`: Returns the kspace trajectory of the measured signal as a single tensor (ADC events only, concatenated repetitions)
- `Sequence.get_full_kspace()`: Returns the full kspace trajectory as a list of per-repetition tensors (containing all events)
- `Sequence.plot_kspace_trajectory()`: Plot the result of `get_full_kspace()`
- `Sequence.get_contrast_mask(contrast)`: Returns a bool mask for the selected `contrast` which can be applied to the simulated signal or kspace
- `Sequence.get_contrasts()`: Returns a list of all used contrast IDs (apply for all repetitions)
- `Sequence.shift_contrasts(offset)`: Shift all contrast IDs by `offset` (apply for all repetitions)
- `get_duration()`: Return the total duration of the sequence in seconds

---

Sequences can be imported from [Pulseq](../integration-pulseq.md) .seq files or Siemens .dsv files:

```python
Sequence.import_file(file_name, exact_trajectories, print_stats, default_shim, ref_voltage, resolution)
```

| Parameter | Description |
| --------- | ----------- |
| `file_name` | Path to .seq file or .dsv folder + file stem |
| `exact_trajectories` | If true generates more events for accurate diffusion calculation |
| `print_stats` | Print more information during import |
| `default_shim` | `shim_array` for pulses that don't specify it themselves |
| `ref_voltage` | *.dsv only:* used to convert volts to RF amplitudes |
| `resolution` | *.dsv only:* defines samples per ADC block. If not set, this is determined by the .dsv time step + ADC duration. |


### `chain`

Helper function to combine multiple sequences into a multi-contrast sequence.

```python
chain(*sequences, oneshot=False)
```

Returns a new `sequence` object. If `oneshot` is set to:
- `True`, sequences are simply concatenated.
- `False`, the `adc_usage`s are shifted so that every sub-sequence has their own set of IDs.


## `Repetition`

Part of a `Sequence`, containing an RF pulse and a list of `event_count` events.

```python
Repetition(pulse: Pulse, event_time, gradm, adc_phase, adc_usage)
```

| Parameter | Description |
| --------- | ----------- |
| `pulse` | The pulse at the start of the repetition |
| `event_time` | Shape `[event_time]`: Durations in seconds |
| `gradm` | Shape `[event_time, 3]`: Gradient moments in 1/m |
| `adc_phase` | Shape `[event_time]`: ADC phases in radians |
| `adc_usage` | Shape `[event_time]`: 0/1 = ADC off/on, other values can be used to distinguish images in multi-contrast sequences |

- `Repetition.event_count` (property): number of events in this repetition
- `Repetition.cpu()`: move repetition data  to CPU
- `Repetition.cuda(device=None)`: move repetition data to selected or default CUDA device
- `Repetition.device` (property): the device of the repetition data
- `Repetition.zero(event_count)`: create a zero'd repetition with the given number of events
- `Repetition.clone()`: make a copy of the repetition with cloned
- `Repetition.get_contrasts()`: return a list of all used `adc_usage` values (except zero)
- `Repetition.shift_contrasts(offset)`: add `offset` to all `adc_usage` values (except zero)


## `Pulse`

Contains the definition of an instantaneous RF Pulse.

```python
Pulse(usage: PulseUsage, angle, phase, shim_array, selective)
```

| Parameter | Description |
| --------- | ----------- |
| `usage` | [`PulseUsage`](#pulseusage) - not used in simulation |
| `angle`, `phase` | Values in radians |
| `shim_array` | Channel amplitudes and phases for pTx - use `[[1, 0]]` for 1Tx |
| `selective` | Used by legacy exporters to emit slice-selective pulses |

- `Pulse.cpu()`: move pulse data to CPU
- `Pulse.cuda(device=None)`: move pulse data to selected or default CUDA device
- `Pulse.device` (property): the device of the pulse data
- `Pulse.zero()`: alternative constructor which zero-initializes everything
- `Pulse.clone()`: make a copy of the pulse with cloned


## `PulseUsage`

Used for computing the k-space trajectory or to select the pulse type in legacy exporters.

| Attribute | Value | Description |
| --------- | ----- | ----------- |
| `UNDEF` | `"undefined"` | No specified use case. |
| `EXCIT` | `"excitation"` | Will set the kspace position back to zero or the position stored by the last `STORE` pulse. |
| `REFOC` | `"refocussing"` |  Mirrors the kspace position. |
| `STORE` | `"storing"` | Stores the current kspace position. Can be used for DREAM-like sequences. |
| `FATSAT` | `"fatsaturation"` | Not handled differently by `get_kspace()`, but can be used by Pulseq exporters to emit a fat-saturation pulse. |
