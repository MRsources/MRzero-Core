import numpy as np
# HACK for pypulseq that still uses the following, which was deprecated some time ago
np.float = float
np.int = int
np.complex = complex

import pypulseq as pp
from types import SimpleNamespace
import torch
from ..sequence import Sequence, PulseUsage, Pulse


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


def pulseq_write_cartesian(seq_param: Sequence, path: str, FOV: float,
                           plot_seq=False, num_slices=1, write_data=1):
    """Export the `seq_param` mr0 sequence as a Pulseq .seq file."""
    bdw_start = 0

    # save pulseq definition
    slice_thickness = np.max([8e-3, 5e-3*num_slices])
    MAXSLEW = 200
    FOV = FOV / 1000
    deltak = 1.0 / FOV
    deltakz = 1.0 / slice_thickness

    system = pp.Opts(
        rf_ringdown_time=20e-6, rf_dead_time=100e-6, adc_dead_time=20e-6,
        max_grad=80, grad_unit="mT/m", max_slew=MAXSLEW, slew_unit="T/m/s"
    )
    seq = pp.Sequence(system)
    seq.set_definition("FOV", [FOV, FOV, slice_thickness])

    seq.add_block(make_delay(5.0))

    # import pdb; pdb.set_trace()
    for i, rep in enumerate(seq_param):
        adc_start = 0
        flip_angle, flip_phase, shim_array = rectify_flips(rep.pulse)
        for event in range(rep.event_count):
            if torch.abs(rep.adc_usage[event]) == 0:
                RFdur = 0

                if event == 0:
                    if rep.pulse.usage == PulseUsage.UNDEF:
                        RFdur = 1e-3
                        if torch.as_tensor(rep.pulse.angle).abs().sum() > 1e-8:
                            rf = make_block_pulse(
                                flip_angle, flip_phase, shim_array,
                                RFdur, system
                            )
                            seq.add_block(rf)
                            seq.add_block(make_delay(1e-4))

                    elif (rep.pulse.usage == PulseUsage.EXCIT or
                          rep.pulse.usage == PulseUsage.STORE):
                        if torch.as_tensor(rep.pulse.angle).abs().sum() > 1e-8:
                            if not rep.pulse.selective:
                                RFdur = 1e-3
                                rf = make_block_pulse(
                                    flip_angle, flip_phase, shim_array,
                                    RFdur, system
                                )
                                seq.add_block(rf)
                            else:
                                RFdur = 1e-3
                                rf, gz, gzr = make_sinc_pulse(
                                    flip_angle, flip_phase, shim_array,
                                    RFdur, slice_thickness, 0.5, 4, system
                                )
                                seq.add_block(gzr)
                                seq.add_block(rf, gz)
                                seq.add_block(gzr)
                                RFdur = (gz.rise_time + gz.flat_time
                                         + gz.fall_time + gzr.rise_time
                                         + gzr.flat_time + gzr.fall_time)

                    elif rep.pulse.usage == PulseUsage.REFOC:
                        if torch.as_tensor(rep.pulse.angle).abs().sum() > 1e-8:
                            if not rep.pulse.selective:
                                RFdur = 1e-3
                                rf = make_block_pulse(
                                    flip_angle, flip_phase, shim_array,
                                    RFdur, system
                                )
                                seq.add_block(rf)
                            else:
                                RFdur = 1e-3
                                rf, gz, gzr = make_sinc_pulse(
                                    flip_angle, flip_phase, shim_array,
                                    RFdur, slice_thickness, 0.5, 4, system
                                )
                                seq.add_block(gzr)
                                seq.add_block(rf, gz)
                                seq.add_block(gzr)

                    elif rep.pulse.usage == PulseUsage.FATSAT:
                        # bandwidth: 200
                        RFdur = 6.120e-3
                        dCurrFrequency = 123.2
                        rf, gz, gzr = pp.make_gauss_pulse(
                            flip_angle=110*np.pi/180, system=system,
                            slice_thickness=slice_thickness, duration=RFdur,
                            freq_offset=-3.3*dCurrFrequency,
                            time_bw_product=0.2, apodization=0.5,
                            return_gz=True
                        )
                        seq.add_block(gzr)
                        seq.add_block(rf, gz)
                        seq.add_block(gzr)

                dur = rep.event_time[event].item() - RFdur
                if dur < 0:
                    raise Exception(
                        f"Event Time too short! Rep {i}, Event {event}, "
                        f"increase event_time by at least: {-dur}"
                    )

                gx_gradmom = rep.gradm[event, 0].item()*deltak
                gy_gradmom = rep.gradm[event, 1].item()*deltak
                gz_gradmom = rep.gradm[event, 2].item()*deltakz

                if np.abs(gx_gradmom) > 0:
                    gx_adc_ramp = 0
                    if any(rep.adc_usage):
                        if torch.abs(rep.adc_usage[event+1]) and torch.abs(rep.gradm[event+1, 0]) > 0:
                            kwargs_for_gx = {"channel": 'x', "system": system, "flat_area": rep.gradm[event+1, 0].item(
                            )*deltak, "flat_time": rep.event_time[event+1].item()}
                            gx_adc = make_trapezoid(**kwargs_for_gx)
                            gx_adc_ramp = gx_adc.amplitude*gx_adc.rise_time/2
                    kwargs_for_gx = {"channel": 'x', "system": system,
                                     "area": gx_gradmom-gx_adc_ramp, "duration": dur}
                    gx = make_trapezoid(**kwargs_for_gx)
                if np.abs(gy_gradmom) > 0:
                    gy_adc_ramp = 0
                    if any(rep.adc_usage):
                        if torch.abs(rep.adc_usage[event+1]) and torch.abs(rep.gradm[event+1, 1]) > 0:
                            kwargs_for_gy = {"channel": 'y', "system": system, "flat_area": rep.gradm[event+1, 1].item(
                            )*deltak, "flat_time": rep.event_time[event+1].item()}
                            gy_adc = make_trapezoid(**kwargs_for_gy)
                            gy_adc_ramp = gy_adc.amplitude*gy_adc.rise_time/2
                    kwargs_for_gy = {"channel": 'y', "system": system,
                                     "area": gy_gradmom-gy_adc_ramp, "duration": dur}
                    try:
                        gy = make_trapezoid(**kwargs_for_gy)
                    except Exception as e:
                        print(e)
                        print('Event Time too short! Event Time: Rep: ' +
                              str(i) + ', Event: ' + str(event))
                if np.abs(gz_gradmom) > 0:
                    gz_adc_ramp = 0
                    if any(rep.adc_usage):
                        if torch.abs(rep.adc_usage[event+1]) and torch.abs(rep.gradm[event+1, 2]) > 0:
                            kwargs_for_gz = {"channel": 'y', "system": system, "flat_area": rep.gradm[event+1, 2].item(
                            )*deltakz, "flat_time": rep.event_time[event+1].item()}
                            gz_adc = make_trapezoid(**kwargs_for_gz)
                            gz_adc_ramp = gz_adc.amplitude*gz_adc.rise_time/2
                    kwargs_for_gz = {"channel": 'z', "system": system,
                                     "area": gz_gradmom-gz_adc_ramp, "duration": dur}
                    try:
                        gz = make_trapezoid(**kwargs_for_gz)
                    except Exception as e:
                        print(e)
                        print(f"Event Time too short! Rep {i}, Event {event}")

                grads = []
                if gx_gradmom != 0:
                    grads.append(gx)
                if gy_gradmom != 0:
                    grads.append(gy)
                if gz_gradmom != 0:
                    grads.append(gz)

                if len(grads) > 0:
                    seq.add_block(*grads)
                else:
                    seq.add_block(make_delay(dur))

            else:  # adc mask == 1
                if adc_start == 1:
                    pass
                else:
                    adc_start = 1
                    if bdw_start == 0:
                        bwd = (1/rep.event_time[event]) / \
                            torch.sum(rep.adc_usage > 0)
                        print('Bandwidth is %4d Hz/pixel' % (bwd))
                        bdw_start = 1

                    idx_T = np.nonzero(torch.abs(rep.adc_usage))
                    dur = torch.sum(rep.event_time[idx_T], 0).item()

                    rise_time_x = 0
                    rise_time_y = 0
                    rise_time_z = 0

                    gx_gradmom = torch.sum(rep.gradm[idx_T, 0]).item()*deltak
                    if np.abs(gx_gradmom) > 0:
                        kwargs_for_gx = {
                            "channel": 'x', "system": system, "flat_area": gx_gradmom, "flat_time": dur}
                        gx = make_trapezoid(**kwargs_for_gx)
                        rise_time_x = gx.rise_time

                    gy_gradmom = torch.sum(rep.gradm[idx_T, 1]).item()*deltak
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
                    # rep.event_time[idx_T[0]].item()/2  # heuristic delay, to be checked at scanner
                    shift = 0.0
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
                        adc_delay), "phase_offset": rf.phase_offset - np.pi/4}
                    adc = make_adc(**kwargs_for_adc)

                    # dont play zero grads (cant even do FID otherwise)
                    grads = []
                    if gx_gradmom != 0:
                        grads.append(gx)
                    if gy_gradmom != 0:
                        grads.append(gy)
                    if gz_gradmom != 0:
                        grads.append(gz)

                    if len(grads) > 0:
                        seq.add_block(*grads, adc)
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
