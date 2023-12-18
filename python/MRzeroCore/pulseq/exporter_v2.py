import os
from types import SimpleNamespace
from ..sequence import Sequence, PulseUsage, Pulse
from . import helpers
import torch

import numpy as np
# HACK: pypulseq indirectly depends on these deprecated / removed values
np.float = float
np.int = int
np.complex = complex
import pypulseq as pp


# We support pTx with martins modified pulseq version 1.4.5
supports_ptx = (pp.Sequence.version_minor,
                pp.Sequence.version_revision) == (4, 5)


def rectify_flips(pulse: Pulse) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    angle = torch.as_tensor(pulse.angle).detach().cpu().numpy()
    phase = torch.as_tensor(pulse.phase).detach().cpu().numpy()
    shim_array = torch.as_tensor(pulse.shim_array).detach().cpu().numpy()

    if angle < 0:
        angle = -angle
        phase = phase + np.pi
    angle = np.fmod(angle)
    phase = np.fmod(phase)

    return angle, phase, shim_array


def make_block_pulse(flip_angle: np.ndarray, flip_phase: np.ndarray,
                     shim_array: np.ndarray,
                     duration: float, system: pp.Opts):
    ptx_args = {}
    if shim_array.shape[0] > 1:
        assert supports_ptx
        ptx_args["shim_array"] = shim_array

    return pp.make_block_pulse(
        flip_angle=flip_angle, phase_offset=flip_phase, duration=duration,
        system=system, **ptx_args
    )


def make_sinc_pulse(flip_angle: np.ndarray, flip_phase: np.ndarray,
                    shim_array: np.ndarray,
                    duration: float, slice_thickness: float,
                    apodization: float, time_bw_product: float,
                    system: pp.Opts):
    ptx_args = {}
    if shim_array.shape[0] > 1:
        assert supports_ptx
        ptx_args["shim_array"] = shim_array

    return pp.make_sinc_pulse(
        flip_angle=flip_angle, phase_offset=flip_phase, duration=duration,
        slice_thickness=slice_thickness, apodization=apodization,
        time_bw_product=time_bw_product, system=system,
        return_gz=True, **ptx_args
    )


def make_gauss_pulse(flip_angle: np.ndarray, flip_phase: np.ndarray,
                     shim_array: np.ndarray,
                     duration: float, slice_thickness: float,
                     apodization: float, time_bw_product: float,
                     system: pp.Opts, freq_offset: float):
    ptx_args = {}
    if shim_array.shape[0] > 1:
        assert supports_ptx
        ptx_args["shim_array"] = shim_array

    return pp.make_gauss_pulse(
        flip_angle=flip_angle, phase_offset=flip_phase, duration=duration,
        slice_thickness=slice_thickness, apodization=apodization,
        time_bw_product=time_bw_product, system=system, freq_offset=freq_offset,
        return_gz=True, **ptx_args
    )


# Modified versions of make_delay, make_adc and make_trapezoid that ensure that
# all events (and thus gradients) are on the gradient time raster. If they are
# not, the scanner crashes without hinting why

def make_delay(d: float) -> SimpleNamespace:
    """make_delay wrapper that rounds delay to the gradient time raster."""
    return pp.make_delay(round(d, 5))


def make_adc(num_samples: int, system: pp.Opts = pp.Opts(), dwell: float = 0,
             duration: float = 0, delay: float = 0, freq_offset: float = 0,
             phase_offset: float = 0) -> SimpleNamespace:
    """make_adc wrapper that modifies the delay such that the total duration
    is on the gradient time raster."""
    if dwell != 0:
        dwell = round(dwell, 5)
    if duration != 0:
        duration = round(duration, 5)

    delay = round((delay + duration), 5) - duration

    return pp.make_adc(
        num_samples=num_samples, system=system, dwell=dwell, duration=duration,
        delay=delay, freq_offset=freq_offset, phase_offset=phase_offset
    )


def make_trapezoid(channel: str, amplitude: float = 0, area: float = None, delay: float = 0, duration: float = 0,
                   flat_area: float = 0, flat_time: float = -1, max_grad: float = 0, max_slew: float = 0,
                   rise_time: float = 0, system: pp.Opts = pp.Opts()) -> SimpleNamespace:
    """make_trapezoid wrapper that rounds gradients to the raster."""
    raster = system.grad_raster_time
    if delay != 0:
        delay = round(delay / raster) * raster
    if rise_time != 0:
        rise_time = round(rise_time / raster) * raster
    if flat_time != -1:
        flat_time = round(flat_time / raster) * raster
    if duration != 0:
        duration = round(duration / raster) * raster

    return pp.make_trapezoid(
        channel=channel, amplitude=amplitude, area=area, delay=delay,
        duration=duration, flat_area=flat_area, flat_time=flat_time,
        max_grad=max_grad, max_slew=max_slew, rise_time=rise_time, system=system
    )


def pulseq_write_cartesian(seq_param: Sequence,
                           path: str,
                           FOV: tuple[float, float, float],
                           plot_seq=False,
                           write_data=True):
    """Export Sequence as pulseq file Version 1.3.1post1.

    This function creates a pulseq file where the exported seq files matches 
    the simulation.
    The SEQ file undergoes several modifications. 
    Firstly, the center of a pulse is placed at the beginning of the repetition. 
    To achieve this, the time of the last event within a repetition is reduced 
    by half the pulse length. 
    The gradient moment of the adc ramp (rising) is addtionally substracted 
    from the gradient in the event before an adc!
    The gradient moment of the adc ramp (falling) is addtionally substracted 
    from the gradient in the event after an adc!
    The simulation needs a rewinder of N_adc/2-1, since first a gradient is
    played out, than the adc point is measured. To match this to the exporter
    an additional half gradient moment of one adc needs to be added to the gradient
    before and after an adc!
    # Example for resolution 8! Rewinder goes to -5, after adc the gradient moment is +3.
    # ADC wrong index   /   -4.5 -3.5 -2.5 -1.5 -0.5 +0.5 +1.5 +2.5  \
    # Gradient Moment   / -5   -4   -3   -2   -1   +0   +1   +2   +3 \
    # Correct Gradient before adc by +deltakx/2!
    # Correct Gradient after adc by -deltakx/2!
    # ADC correct index   /    -4   -3   -2   -1   +0   +1   +2   +3     \
    # Gradient Moment     / -4.5 -3.5 -2.5 -1.5 -0.5 +0.5 +1.5 +2.5 +3.5 \

    Warnings
    --------
    The exporter only operates successfully when the phase-encoding gradient 
    directly precedes the ADC.
    Furthermore, gradients must also played out directly in the event after the ADCs,
    so that they can be handled correctly.

    Parameters
    ----------
    seq_param : Sequence
        Sequence that will be exported as pulseq file.
    path : String
        Location where the pulseq file is saved.
    FOV : tuple[float, float, float]
        FOV of the sequence.
    plot_seq : bool
        Plot the sequences.
    write_data : bool
        Writes the encoding of the sequence as command into the pulseq file.
        This is needed for easier reconstruction of measurements.


    Returns
    -------
    No return!
    """

    bdw_start = 0

    FOV = [item / 1000 for item in FOV]
    deltakx = 1.0 / FOV[0]
    deltaky = 1.0 / FOV[1]
    deltakz = 1.0 / FOV[2]

    system = pp.Opts(
        rf_ringdown_time=20e-6, rf_dead_time=100e-6, adc_dead_time=20e-6,
        max_grad=80, grad_unit="mT/m",
        max_slew=200, slew_unit="T/m/s"
    )
    seq = pp.Sequence(system)
    seq.set_definition("FOV", FOV)
    seq.add_block(make_delay(5.0))

    # import pdb; pdb.set_trace()
    for i, rep in enumerate(seq_param):  # Loop over all repetitions
        adc_start = 0

        # Loop over all events within one repetitions
        for event in range(rep.event_count):

            # No adc in event
            if torch.abs(rep.adc_usage[event]) == 0:
                if event == 0:
                    # First event of rep - apply pulse
                    (RFdurbefore, RFdur, RFdurafter) = create_pulse(
                        rep.pulse, system, seq, FOV
                    )
                elif event == rep.event_count - 1 and i < len(seq_param) - 2:
                    # Last event in a rep, except for the last rep in the seq:
                    # - calculate the duration of the following pulse
                    (RFdurbefore, RFdur, RFdurafter) = create_pulse(
                        rep.pulse, system, None, FOV
                    )

                # Substract the time of the rf pulse starting from the middle of the rf pulse to match simulation
                dur = rep.event_time[event].item() - RFdurafter

                # Substract the time of the rf pulse of the following repetition
                if event == rep.event_count - 1:  # Last event in rep
                    dur -= RFdurbefore

                if dur < 0:
                    raise Exception('Event Time too short! Event Time: Rep: ' + str(
                        i) + ', Event: ' + str(event), ', increase event_time by at least: ' + str(-dur))

                gx_gradmom = rep.gradm[event, 0].item()*deltakx
                gy_gradmom = rep.gradm[event, 1].item()*deltaky
                gz_gradmom = rep.gradm[event, 2].item()*deltakz
                idx_T = np.nonzero(torch.abs(rep.adc_usage))
                dur_adc = np.round(
                    torch.sum(rep.event_time[idx_T], 0).item(), decimals=5)

                if np.abs(gx_gradmom) > 0:
                    gx_adc_ramp = 0
                    if any(rep.adc_usage):
                        # Case: Gradient before adc
                        # Correct duration by the rise time
                        # Correct gradient by the gradient moment of adc ramp
                        if event < rep.event_count - 1 and torch.abs(rep.adc_usage[event+1]) and torch.abs(rep.gradm[event+1, 0]) > 0:

                            # ADC wrong         /   -4.5 -3.5 -2.5 -1.5 -0.5 +0.5 +1.5 +2.5  \
                            # Gradient Moment   / -5   -4   -3   -2   -1   +0   +1   +2   +3 \
                            # Correct Gradient before adc by +0.5 --> neg sign needed because the gradient correction is substracted (gx_gradmom-gx_adc_ramp)!
                            # Correct Gradient after adc by -0.5 --> pos sign is needed because the gradient correction is substracted (gx_gradmom-gx_adc_ramp)!
                            # ADC correct         /    -4   -3   -2   -1   +0   +1   +2   +3     \
                            # Gradient Moment     / -4.5 -3.5 -2.5 -1.5 -0.5 +0.5 +1.5 +2.5 +3.5 \

                            flat_area = torch.sum(rep.gradm[idx_T, 0]).item()*deltakx
                            gx_adc = make_trapezoid("x", flat_area=flat_area, flat_time=dur_adc, system=system)
                            # Correct gradient by half of the gradient moment of one adc moment!
                            gx_adc_ramp = (
                                gx_adc.amplitude*gx_adc.rise_time/2 -
                                torch.sign(rep.gradm[event+1, 0]).item() * deltakx/2
                            )
                            gx_rise_time = gx_adc.rise_time
                            dur -= gx_rise_time  # Correct timing for rise and fall time of adc gradient
                        # Case: Gradient after adc
                        # Correct duration by the fall time
                        # Correct gradient by the gradient moment of adc ramp
                        # Event after adc to reduce dur by fall time of adc gradient
                        elif event > 0 and torch.abs(rep.adc_usage[event-1]) and torch.abs(rep.gradm[event-1, 0]) > 0:
                            kwargs_for_gx = {"channel": 'x', "system": system, "flat_area": torch.sum(
                                rep.gradm[idx_T, 0]).item()*deltakx, "flat_time": dur_adc}
                            gx_adc = make_trapezoid(**kwargs_for_gx)
                            gx_adc_ramp = gx_adc.amplitude*gx_adc.fall_time/2 + \
                                torch.sign(
                                    rep.gradm[event-1, 0]).item() * deltakx/2
                            gx_fall_time = gx_adc.fall_time
                            dur -= gx_fall_time  # Correct timing for rise and fall time of adc gradient
                    kwargs_for_gx = {"channel": 'x', "system": system,
                                     "area": gx_gradmom-gx_adc_ramp, "duration": dur}
                    try:
                        gx = make_trapezoid(**kwargs_for_gx)
                    except:
                        raise Exception(
                            'Event Time too short (gx)! Event Time: Rep: ' + str(i) + ', Event: ' + str(event))
                if np.abs(gy_gradmom) > 0:
                    gy_adc_ramp = 0
                    if any(rep.adc_usage):
                        # Case: Gradient before adc
                        # Correct duration by the rise time
                        # Correct gradient by the gradient moment of adc ramp
                        if event < rep.event_count - 1 and torch.abs(rep.adc_usage[event+1]) and torch.abs(rep.gradm[event+1, 1]) > 0:
                            kwargs_for_gy = {"channel": 'y', "system": system, "flat_area": torch.sum(
                                rep.gradm[idx_T, 1]).item()*deltaky, "flat_time": dur_adc}
                            gy_adc = make_trapezoid(**kwargs_for_gy)
                            gy_adc_ramp = gy_adc.amplitude*gy_adc.rise_time/2 - \
                                torch.sign(
                                    rep.gradm[event+1, 1]).item() * deltaky/2
                            gy_rise_time = gy_adc.rise_time
                            dur -= gy_rise_time  # Correct timing for rise and fall time of adc gradient
                        # Case: Gradient after adc
                        # Correct duration by the fall time
                        # Correct gradient by the gradient moment of adc ramp
                        # Event after adc to reduce dur by fall time of adc gradient
                        elif event > 0 and torch.abs(rep.adc_usage[event-1]) and torch.abs(rep.gradm[event-1, 1]) > 0:
                            kwargs_for_gy = {"channel": 'y', "system": system, "flat_area": torch.sum(
                                rep.gradm[idx_T, 1]).item()*deltaky, "flat_time": dur_adc}
                            gy_adc = make_trapezoid(**kwargs_for_gy)
                            gy_adc_ramp = gy_adc.amplitude*gy_adc.fall_time/2 + \
                                torch.sign(
                                    rep.gradm[event-1, 1]).item() * deltaky/2
                            gy_fall_time = gy_adc.fall_time
                            dur -= gy_fall_time  # Correct timing for rise and fall time of adc gradient
                    kwargs_for_gy = {"channel": 'y', "system": system,
                                     "area": gy_gradmom-gy_adc_ramp, "duration": dur}
                    try:
                        gy = make_trapezoid(**kwargs_for_gy)
                    except:
                        raise Exception(
                            'Event Time too short (gy)! Event Time: Rep: ' + str(i) + ', Event: ' + str(event))
                if np.abs(gz_gradmom) > 0:
                    gz_adc_ramp = 0
                    if any(rep.adc_usage):
                        # Case: Gradient before adc
                        # Correct duration by the rise time
                        # Correct gradient by the gradient moment of adc ramp
                        if event < rep.event_count - 1 and torch.abs(rep.adc_usage[event+1]) and torch.abs(rep.gradm[event+1, 2]) > 0:
                            kwargs_for_gz = {"channel": 'z', "system": system, "flat_area": torch.sum(
                                rep.gradm[idx_T, 2]).item()*deltakz, "flat_time": dur_adc}
                            gz_adc = make_trapezoid(**kwargs_for_gz)
                            gz_adc_ramp = gz_adc.amplitude*gz_adc.rise_time/2 - \
                                torch.sign(
                                    rep.gradm[event+1, 2]).item() * deltakz/2
                            gz_rise_time = gz_adc.rise_time
                            dur -= gz_rise_time  # Correct timing for rise and fall time of adc gradient
                        # Case: Gradient after adc
                        # Correct duration by the fall time
                        # Correct gradient by the gradient moment of adc ramp
                        # Event after adc to reduce dur by fall time of adc gradient
                        elif event > 0 and torch.abs(rep.adc_usage[event-1]) and torch.abs(rep.gradm[event-1, 2]) > 0:
                            kwargs_for_gz = {"channel": 'z', "system": system, "flat_area": torch.sum(
                                rep.gradm[idx_T, 2]).item()*deltakz, "flat_time": dur_adc}
                            gz_adc = make_trapezoid(**kwargs_for_gz)
                            gz_adc_ramp = gz_adc.amplitude*gz_adc.fall_time/2 + \
                                torch.sign(
                                    rep.gradm[event-1, 2]).item() * deltakz/2
                            gz_fall_time = gz_adc.fall_time
                            dur -= gz_fall_time  # Correct timing for rise and fall time of adc gradient
                    kwargs_for_gz = {"channel": 'z', "system": system,
                                     "area": gz_gradmom-gz_adc_ramp, "duration": dur}
                    try:
                        gz = make_trapezoid(**kwargs_for_gz)
                    except:
                        raise Exception(
                            'Event Time too short (gz)! Event Time: Rep: ' + str(i) + ', Event: ' + str(event))

                if np.abs(gx_gradmom) > 0 and np.abs(gy_gradmom) > 0 and np.abs(gz_gradmom):
                    seq.add_block(gx, gy, gz)
                elif np.abs(gx_gradmom) > 0 and np.abs(gy_gradmom) > 0:
                    seq.add_block(gx, gy)
                elif np.abs(gx_gradmom) > 0 and np.abs(gz_gradmom) > 0:
                    seq.add_block(gx, gz)
                elif np.abs(gy_gradmom) > 0 and np.abs(gz_gradmom) > 0:
                    seq.add_block(gy, gz)
                elif np.abs(gx_gradmom) > 0:
                    seq.add_block(gx)
                elif np.abs(gy_gradmom) > 0:
                    seq.add_block(gy)
                elif np.abs(gz_gradmom) > 0:
                    seq.add_block(gz)
                else:
                    seq.add_block(make_delay(dur))
            else:  # adc mask == 1
                if adc_start == 1:
                    pass
                else:
                    adc_start = 1
                    if bdw_start == 0:
                        bwd = (1/rep.event_time[event]) / \
                            torch.sum(torch.abs(rep.adc_usage) > 0)
                        print('Bandwidth is %4d Hz/pixel' % (bwd))
                        seq.set_definition("Bandwidth", f"{int(bwd)} Hz/px")
                        bdw_start = 1

                    idx_T = np.nonzero(torch.abs(rep.adc_usage))
                    dur = np.round(
                        torch.sum(rep.event_time[idx_T], 0).item(), decimals=5)

                    rise_time_x = 0
                    rise_time_y = 0
                    rise_time_z = 0

                    gx_gradmom = torch.sum(rep.gradm[idx_T, 0]).item()*deltakx
                    if np.abs(gx_gradmom) > 0:
                        kwargs_for_gx = {
                            "channel": 'x', "system": system, "flat_area": gx_gradmom, "flat_time": dur}
                        gx = make_trapezoid(**kwargs_for_gx)
                        rise_time_x = gx.rise_time

                    gy_gradmom = torch.sum(rep.gradm[idx_T, 1]).item()*deltaky
                    if np.abs(gy_gradmom) > 0:
                        kwargs_for_gy = {
                            "channel": 'y', "system": system, "flat_area": gy_gradmom, "flat_time": dur}
                        gy = make_trapezoid(**kwargs_for_gy)
                        rise_time_y = gy.rise_time

                    gz_gradmom = torch.sum(rep.gradm[idx_T, 2]).item()*deltakz
                    if np.abs(gz_gradmom) > 0:
                        kwargs_for_gz = {
                            "channel": 'z', "system": system, "flat_area": gz_gradmom, "flat_time": dur}
                        gz = make_trapezoid(**kwargs_for_gz)
                        rise_time_z = gz.rise_time

                    # calculate correct delay to have same starting point of flat top
                    # np.round(rep.event_time[idx_T[0]].item()/2,decimals=6)  # heuristic delay, to be checked at scanner
                    shift = 0
                    x_delay = np.max(
                        [0, rise_time_y-rise_time_x, rise_time_z-rise_time_x])+shift
                    y_delay = np.max(
                        [0, rise_time_x-rise_time_y, rise_time_z-rise_time_y])+shift
                    z_delay = np.max(
                        [0, rise_time_x-rise_time_z, rise_time_y-rise_time_z])+shift

                    rise_time_x = 0
                    rise_time_y = 0
                    rise_time_z = 0

                    # adc gradient events are overwritten with correct delays
                    if np.abs(gx_gradmom) > 0:
                        kwargs_for_gx = {"channel": 'x', "system": system,
                                         "delay": x_delay, "flat_area": gx_gradmom, "flat_time": dur}
                        gx = make_trapezoid(**kwargs_for_gx)
                        rise_time_x = gx.rise_time
                    if np.abs(gy_gradmom) > 0:
                        kwargs_for_gy = {"channel": 'y', "system": system,
                                         "delay": y_delay, "flat_area": gy_gradmom, "flat_time": dur}
                        gy = make_trapezoid(**kwargs_for_gy)
                        rise_time_y = gy.rise_time
                    if np.abs(gz_gradmom) > 0:
                        kwargs_for_gz = {"channel": 'z', "system": system,
                                         "delay": z_delay, "flat_area": gz_gradmom, "flat_time": dur}
                        gz = make_trapezoid(**kwargs_for_gz)
                        rise_time_z = gz.rise_time

                    adc_delay = np.max(
                        [rise_time_x, rise_time_y, rise_time_z])+shift
                    kwargs_for_adc = {"num_samples": idx_T.size()[0], "duration": dur, "delay": (
                        adc_delay), "phase_offset": rf_ex.phase_offset - np.pi/4}
                    adc = make_adc(**kwargs_for_adc)

                    # dont play zero grads (cant even do FID otherwise)
                    if rep.adc_usage[event] == -1:
                        print("Dummie ADC played out")
                        if np.abs(gx_gradmom) > 0 and np.abs(gy_gradmom) > 0 and np.abs(gz_gradmom):
                            seq.add_block(gx, gy, gz)
                        elif np.abs(gx_gradmom) > 0 and np.abs(gy_gradmom) > 0:
                            seq.add_block(gx, gy)
                        elif np.abs(gx_gradmom) > 0 and np.abs(gz_gradmom) > 0:
                            seq.add_block(gx, gz)
                        elif np.abs(gy_gradmom) > 0 and np.abs(gz_gradmom) > 0:
                            seq.add_block(gy, gz)
                        elif np.abs(gx_gradmom) > 0:
                            seq.add_block(gx)
                        elif np.abs(gy_gradmom) > 0:
                            seq.add_block(gy)
                        elif np.abs(gz_gradmom) > 0:
                            seq.add_block(gz)
                    else:
                        if np.abs(gx_gradmom) > 0 and np.abs(gy_gradmom) > 0 and np.abs(gz_gradmom):
                            seq.add_block(gx, gy, gz, adc)
                        elif np.abs(gx_gradmom) > 0 and np.abs(gy_gradmom) > 0:
                            seq.add_block(gx, gy, adc)
                        elif np.abs(gx_gradmom) > 0 and np.abs(gz_gradmom) > 0:
                            seq.add_block(gx, gz, adc)
                        elif np.abs(gy_gradmom) > 0 and np.abs(gz_gradmom) > 0:
                            seq.add_block(gy, gz, adc)
                        elif np.abs(gx_gradmom) > 0:
                            seq.add_block(gx, adc)
                        elif np.abs(gy_gradmom) > 0:
                            seq.add_block(gy, adc)
                        elif np.abs(gz_gradmom) > 0:
                            seq.add_block(gz, adc)
                        else:
                            seq.add_block(adc)

    passes, report = seq.check_timing()
    if not passes:
        print("WARNING: Timing check failed:")
        for line in report:
            print(line, end="")

    if plot_seq:
        seq.plot()
    seq.write(path)

    append_header(path)

    if write_data:
        helpers.write_data_to_seq_file(seq_param, path)
        print('Added kspace & adc_usage information to the .seq file!')


def create_pulse(
        pulse: Pulse, system: pp.Opts, seq: pp.Sequence | None,
        FOV: tuple[float, float, float]
        ) -> tuple[float, float, float]:
    RFdurbefore = 0
    RFdur = 0
    RFdurafter = 0
    flip_angle, flip_phase, shim_array = rectify_flips(pulse)

    ###############################
    # global pulse
    if pulse.usage == PulseUsage.UNDEF:
        if torch.abs(pulse.angle) > 1e-8:

            RFdur = 1e-3
            rf = make_block_pulse(
                flip_angle, flip_phase, shim_array,
                RFdur, system
            )
            if seq:
                seq.add_block(rf)
                seq.add_block(make_delay(1e-4))

            RFdurbefore = rf.delay + RFdur/2
            RFdurafter = 1e-4 + RFdur/2 + rf.ringdown_time

    elif pulse.usage == PulseUsage.EXCIT or pulse.usage == PulseUsage.STORE:
        ###############################
        # excitation pulse
        if torch.abs(pulse.angle) > 1e-8:

            if not pulse.selective:
                RFdur = 1e-3
                rf = make_block_pulse(
                    flip_angle, flip_phase, shim_array,
                    RFdur, system
                )
                if seq:
                    seq.add_block(rf)
                    seq.add_block(make_delay(1e-4))

                RFdurbefore = rf.delay + RFdur/2
                RFdurafter = 1e-4 + RFdur/2 + rf.ringdown_time

            else:
                RFdur = 1e-3
                rf, gz, gzr = make_sinc_pulse(
                    flip_angle, flip_phase, shim_array,
                    RFdur, FOV[2], 0.15, 2, system
                )
                if seq:
                    seq.add_block(gzr)
                    seq.add_block(rf, gz)
                    seq.add_block(gzr)

                RFdurbefore = (
                    gzr.rise_time + gzr.flat_time + gzr.fall_time +
                    gz.delay + gz.rise_time + gz.flat_time/2
                )
                RFdurafter = (
                    gz.flat_time/2 + gz.fall_time +
                    gzr.rise_time + gzr.flat_time +
                    gzr.fall_time
                )

    elif pulse.usage == PulseUsage.REFOC:
        ###############################
        # refocusing pulse

        if torch.abs(pulse.angle) > 1e-8:
            RFdur = 1e-3

            if not pulse.selective:
                rf = make_block_pulse(
                    flip_angle, flip_phase, shim_array,
                    RFdur, system
                )
                seq.add_block(rf)
                seq.add_block(make_delay(1e-4))

                RFdurbefore = rf.delay + RFdur/2
                RFdurafter = 1e-4 + RFdur/2 + rf.ringdown_time
            else:
                rf, gz, gzr = make_sinc_pulse(
                    flip_angle, flip_phase, shim_array,
                    RFdur, FOV[2], 0.5, 4, system
                )
                if seq:
                    seq.add_block(gzr)
                    seq.add_block(rf, gz)
                    seq.add_block(gzr)

                RFdurbefore = (
                    gzr.rise_time + gzr.flat_time + gzr.fall_time +
                    gz.delay + gz.rise_time + gz.flat_time/2
                )
                RFdurafter = (
                    gz.flat_time/2 + gz.fall_time +
                    gzr.rise_time + gzr.flat_time +
                    gzr.fall_time
                )

    elif pulse.usage == PulseUsage.FATSAT:
        ###############################
        # fat saturation pulse
        RFdur = 6.120e-3
        dCurrFrequency = 123.2
        rf, gz, gzr = make_gauss_pulse(
            110*np.pi/180, 0, shim_array,
            RFdur, FOV[2], 0.5, 0.2, system, -3.3*dCurrFrequency
        )
        if seq:
            seq.add_block(gzr)
            seq.add_block(rf, gz)
            seq.add_block(gzr)

        RFdurafter = (
            gz.flat_time/2 + gz.fall_time +
            gzr.rise_time + gzr.flat_time + gzr.fall_time
        )

    return (RFdurbefore, RFdur, RFdurafter)


def append_header(path):
    """append version and definitions"""
    try:
        with open("../../../.git/ORIG_HEAD") as file:
            git_version = file.read()
    except FileNotFoundError:
        git_version = "unknown"

    with open(path, 'r') as fin:
        lines = fin.readlines()

    updated_lines = []
    updated_lines.append("# Pulseq sequence file\n")
    updated_lines.append("# Created by MRIzero/IMR/GPI pulseq converter\n")
    updated_lines.append("# MRzero-Core, git hash: " + git_version)
    updated_lines.append(f"# experiment_id: {os.path.split(path)[-2]}\n")
    updated_lines.append("# path: " + path + "\n")
    updated_lines.append("\n")

    updated_lines.extend(lines[3:])

    with open(path, 'w') as fout:
        fout.writelines(updated_lines)
