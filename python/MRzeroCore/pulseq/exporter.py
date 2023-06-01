# -*- coding: utf-8 -*-
from types import SimpleNamespace
import sys
import torch
import numpy as np
# HACK for pypulseq that still uses the following, which was deprecated some time ago
np.float = float
np.int = int
np.complex = complex

from ..sequence import Sequence as Seq

# TODO: maybe replace with import pypulseq as pp
from pypulseq.Sequence.sequence import Sequence
from pypulseq.make_adc import make_adc as _make_adc
from pypulseq.make_delay import make_delay as _make_delay
from pypulseq.make_sinc_pulse import make_sinc_pulse
from pypulseq.make_trapezoid import make_trapezoid as _make_trapezoid
from pypulseq.make_gauss_pulse import make_gauss_pulse
from pypulseq.make_block_pulse import make_block_pulse
from pypulseq.opts import Opts


# TODO: This needs a cleanup before using it.
# Also, this only supports cartesian readouts. Clearly state that until we can
# provide a general verison.


def rectify_flips(flips):
    flip_angle = flips.angle.cpu()
    flip_phase = flips.phase.cpu()

    if flips.angle < 0:
        flip_angle = -flips.angle
        flip_phase = flips.phase + np.pi
        flip_phase = torch.fmod(flip_phase, 2*np.pi)
    return flip_angle.item(), flip_phase.item()


# Modified versions of make_delay, make_adc and make_trapezoid that ensure that
# all events (and thus gradients) are on the gradient time raster. If they are
# not, the scanner crashes without hinting why

def make_delay(d: float) -> SimpleNamespace:
    """make_delay wrapper that rounds delay to the gradient time raster."""
    return _make_delay(round(d / 10e-6) * 10e-6)


def make_adc(num_samples: int, system: Opts = Opts(), dwell: float = 0, duration: float = 0, delay: float = 0,
             freq_offset: float = 0, phase_offset: float = 0) -> SimpleNamespace:
    """make_adc wrapper that modifies the delay such that the total duration
    is on the gradient time raster."""
    # TODO: the total duration might not be on the gradient raster. If a
    # sequence with optimized ADC durations fails the timing check, implement
    # this functions to round the timing as necessary.

    return _make_adc(
        num_samples, system, dwell, duration, delay,
        freq_offset, phase_offset
    )


def make_trapezoid(channel: str, amplitude: float = 0, area: float = None, delay: float = 0, duration: float = 0,
                   flat_area: float = 0, flat_time: float = -1, max_grad: float = 0, max_slew: float = 0,
                   rise_time: float = 0, system: Opts = Opts()) -> SimpleNamespace:
    """make_trapezoid wrapper that rounds gradients to the raster."""
    raster = system.grad_raster_time
    if delay != -1:
        delay = round(delay / raster) * raster
    if rise_time != -1:
        rise_time = round(rise_time / raster) * raster
    if flat_time != -1:
        flat_time = round(flat_time / raster) * raster
    if duration != -1:
        duration = round(duration / raster) * raster

    return _make_trapezoid(
        channel, amplitude, area, delay, duration, flat_area, flat_time,
        max_grad, max_slew, rise_time, system
    )


nonsel = 0
if nonsel==1:
    slice_thickness = 200*1e-3
else:
    slice_thickness = 8e-3
    
def pulseq_write_EPG(seq_param, path, FOV, plot_seq=False):
    # save pulseq definition
    MAXSLEW = 200
    FOV = FOV / 1000
    deltak = 1.0 / FOV # /(2*np.pi) before v.2.1.0
    
    kwargs_for_opts = {"rf_ringdown_time": 20e-6, "rf_dead_time": 100e-6, "adc_dead_time": 20e-6, "max_grad": 80, "grad_unit": "mT/m", "max_slew": MAXSLEW, "slew_unit": "T/m/s"}
    system = Opts(**kwargs_for_opts)
    seq = Sequence(system)   
    
    seq.add_block(make_delay(5.0))
    
    # import pdb; pdb.set_trace()
    for i,rep in enumerate(seq_param):
        adc_start = 0
        flip_angle,flip_phase = rectify_flips(rep.pulse)
        for event in range(rep.event_count):
            ###############################
            ## global pulse
            if torch.abs(rep.adc_usage[event]) == 0:  
                RFdur=0
                if event == 0:
                    if rep.pulse.usage == Seq.PulseUsage.UNDEF:
                        RFdur=0
                        if np.abs(rep.pulse.angle) > 1e-8:
                
                            RFdur = 1*1e-3
                            kwargs_for_block = {"flip_angle": flip_angle, "system": system, "duration": RFdur, "phase_offset": flip_phase}
                            rf,_ = make_block_pulse(**kwargs_for_block)
                            
                            seq.add_block(rf)  
                            seq.add_block(make_delay(1e-4))
                                                    
                    elif (rep.pulse.usage == Seq.PulseUsage.EXCIT or
                          rep.pulse.usage == Seq.PulseUsage.STORE):
                        RFdur = 0
                        if np.abs(rep.pulse.angle) > 1e-8:
                            use = "excitation"
                            
                            if nonsel:
                                RFdur = 1*1e-3
                                kwargs_for_block = {"flip_angle": flip_angle, "system": system, "duration": RFdur, "phase_offset": flip_phase}
                                rf_ex,_ = make_block_pulse(**kwargs_for_block)
                                seq.add_block(rf_ex)     
                            else:
                                # alternatively slice selective:
                                use = "excitation"
                                RFdur = 1*1e-3
                                kwargs_for_sinc = {"flip_angle": flip_angle, "system": system, "duration": RFdur, "slice_thickness": slice_thickness, "apodization": 0.5, "time_bw_product": 4, "phase_offset": flip_phase}
                                rf_ex, gz, gzr= make_sinc_pulse(**kwargs_for_sinc)
                                seq.add_block(gzr)
                                seq.add_block(rf_ex, gz)
                                seq.add_block(gzr)
                                RFdur = gz.rise_time + gz.flat_time + gz.fall_time + gzr.rise_time + gzr.flat_time + gzr.fall_time                            
                    
                    elif rep.pulse.usage == Seq.PulseUsage.REFOC:
                        ###############################
                        ### refocusing pulse
                        use = "refocusing"
                        
                        RFdur = 0
                        
                        if np.abs(rep.pulse.angle) > 1e-8:
                          RFdur = 1*1e-3
                          
                          if nonsel:
                              kwargs_for_block = {"flip_angle": flip_angle, "system": system, "duration": RFdur, "phase_offset": flip_phase}
                              rf_ref,_ = make_block_pulse(**kwargs_for_block)
                              seq.add_block(rf_ref)         
                          else:
    
                              kwargs_for_sinc = {"flip_angle": flip_angle, "system": system, "duration": RFdur, "slice_thickness": slice_thickness, "apodization": 0.5, "time_bw_product": 4, "phase_offset": flip_phase}
                              rf_ref, gz_ref,gzr = make_sinc_pulse(**kwargs_for_sinc)
                              seq.add_block(gzr)
                              seq.add_block(rf_ref, gz_ref)
                              seq.add_block(gzr)
                              RFdur = gz_ref.rise_time + gz_ref.flat_time + gz_ref.fall_time 
        
                dur = rep.event_time[event].item() - RFdur
                if dur < 0:
                    raise Exception('Event Time too short! Event Time: Rep: '+ str(i) + ', Event: ' +str(event),', increase event_time by at least: ' + str(-dur))
                
                gx_gradmom = rep.gradm[event,0].item()*deltak
                gy_gradmom = rep.gradm[event,1].item()*deltak


                if np.abs(gx_gradmom)>0:
                    gx_adc_ramp = 0
                    if any(rep.adc_usage):
                        if torch.abs(rep.adc_usage[event+1]) and torch.abs(rep.gradm[event+1,0]) > 0:
                            kwargs_for_gx = {"channel": 'x', "system": system, "flat_area": rep.gradm[event+1,0].item()*deltak, "flat_time": rep.event_time[event+1].item()}
                            gx_adc = make_trapezoid(**kwargs_for_gx)
                            gx_adc_ramp = gx_adc.amplitude*gx_adc.rise_time/2
                    kwargs_for_gx = {"channel": 'x', "system": system, "area": gx_gradmom-gx_adc_ramp, "duration": dur}
                    gx = make_trapezoid(**kwargs_for_gx) 
                if np.abs(gy_gradmom)>0:
                    gy_adc_ramp = 0
                    if any(rep.adc_usage):
                        if torch.abs(rep.adc_usage[event+1]) and torch.abs(rep.gradm[event+1,0]) > 0:   
                            kwargs_for_gy = {"channel": 'y', "system": system, "flat_area": rep.gradm[event+1,0].item()*deltak, "flat_time": rep.event_time[event+1].item()}
                            gy_adc = make_trapezoid(**kwargs_for_gy)
                            gy_adc_ramp = gy_adc.amplitude*gy_adc.rise_time/2                    
                    kwargs_for_gy = {"channel": 'y', "system": system, "area": gy_gradmom-gy_adc_ramp, "duration": dur}
                    try:
                        gy = make_trapezoid(**kwargs_for_gy)
                    except Exception as e:
                        print(e)
                        print('Event Time too short! Event Time: Rep: '+ str(i) + ', Event: ' +str(event))                              
                            
                if np.abs(gx_gradmom) > 0 and np.abs(gy_gradmom) > 0:
                    seq.add_block(gx,gy)
                elif np.abs(gx_gradmom) > 0:
                    seq.add_block(gx)
                elif np.abs(gy_gradmom) > 0:
                    seq.add_block(gy)
                else:
                    seq.add_block(make_delay(dur))
            else: #adc mask == 1
                if adc_start == 1:
                    pass
                else:
                    adc_start = 1
                    idx_T = np.nonzero(torch.abs(rep.adc_usage))                
                    dur = torch.sum(rep.event_time[idx_T],0).item()
    
                    gx_gradmom = torch.sum(rep.gradm[idx_T,0]).item()*deltak                                        
                    kwargs_for_gx = {"channel": 'x', "system": system, "flat_area": gx_gradmom, "flat_time": dur}
                    gx = make_trapezoid(**kwargs_for_gx) 
                    
                    gy_gradmom = torch.sum(rep.gradm[idx_T,1]).item()*deltak
                    kwargs_for_gy = {"channel": 'y', "system": system, "flat_area": gy_gradmom, "flat_time": dur}
                    gy = make_trapezoid(**kwargs_for_gy)   

                    # calculate correct delay to have same starting point of flat top
                    x_delay = np.max([0,gy.rise_time-gx.rise_time])+rep.event_time[idx_T[0]].item()/2  # heuristic delay, to be checked at scanner
                    y_delay = np.max([0,gx.rise_time-gy.rise_time])+rep.event_time[idx_T[0]].item()/2
                    
                    # adc gradient events are overwritten with correct delays
                    kwargs_for_gx = {"channel": 'x', "system": system,"delay":x_delay, "flat_area": gx_gradmom, "flat_time": dur}
                    gx = make_trapezoid(**kwargs_for_gx)                      
                    kwargs_for_gy = {"channel": 'y', "system": system,"delay":y_delay, "flat_area": gy_gradmom, "flat_time": dur}
                    gy = make_trapezoid(**kwargs_for_gy)                       
                    
                    
                    adc_delay = np.max([gx.rise_time,gy.rise_time])
                    kwargs_for_adc = {"num_samples": idx_T.size()[0], "duration": dur, "delay":(adc_delay), "phase_offset": rf_ex.phase_offset - np.pi/4}
                    adc = make_adc(**kwargs_for_adc)    
                    
                    # dont play zero grads (cant even do FID otherwise)
                    if np.abs(gx_gradmom) > 0 and np.abs(gy_gradmom) > 0:
                        seq.add_block(gx,gy,adc)
                    elif np.abs(gx_gradmom) > 0:
                        seq.add_block(gx,adc)
                    elif np.abs(gy_gradmom) > 0:
                        seq.add_block(gy,adc)
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
    
    append_header(path, FOV,slice_thickness)

def pulseq_write_cartesian(seq_param, path, FOV, plot_seq=False, num_slices=1, write_data=1):
    """Export the `seq_param` mr0 sequence as a Pulseq .seq file."""
    bdw_start = 0
    
    # save pulseq definition
    slice_thickness = np.max([8e-3,5e-3*num_slices])
    MAXSLEW = 200
    FOV = FOV / 1000
    deltak = 1.0 / FOV # /(2*np.pi) before v.2.1.0
    deltakz = 1.0 / slice_thickness
    
    kwargs_for_opts = {"rf_ringdown_time": 20e-6, "rf_dead_time": 100e-6, "adc_dead_time": 20e-6, "max_grad": 80, "grad_unit": "mT/m", "max_slew": MAXSLEW, "slew_unit": "T/m/s"}
    system = Opts(**kwargs_for_opts)
    seq = Sequence(system)   
    
    seq.add_block(make_delay(5.0))
    
    # import pdb; pdb.set_trace()
    for i,rep in enumerate(seq_param):
        adc_start = 0
        flip_angle,flip_phase = rectify_flips(rep.pulse)
        for event in range(rep.event_count):
            ###############################
            ## global pulse
            if torch.abs(rep.adc_usage[event]) == 0:  
                RFdur=0
                if event == 0:
                    if rep.pulse.usage == Seq.PulseUsage.UNDEF:
                        RFdur=0
                        if torch.abs(rep.pulse.angle) > 1e-8:
                
                            RFdur = 1*1e-3
                            kwargs_for_block = {"flip_angle": flip_angle, "system": system, "duration": RFdur, "phase_offset": flip_phase}
                            rf = make_block_pulse(**kwargs_for_block)
                            
                            seq.add_block(rf)  
                            seq.add_block(make_delay(1e-4))
                                                    
                    elif (rep.pulse.usage == Seq.PulseUsage.EXCIT or
                          rep.pulse.usage == Seq.PulseUsage.STORE):
                        RFdur = 0
                        if torch.abs(rep.pulse.angle) > 1e-8:
                            use = "excitation"
                            
                            if nonsel:
                                RFdur = 1*1e-3
                                kwargs_for_block = {"flip_angle": flip_angle, "system": system, "duration": RFdur, "phase_offset": flip_phase}
                                rf_ex = make_block_pulse(**kwargs_for_block)
                                seq.add_block(rf_ex)     
                            else:
                                # alternatively slice selective:
                                use = "excitation"
                                RFdur = 1*1e-3
                                #kwargs_for_sinc = {"flip_angle": flip_angle, "system": system, "duration": RFdur, "slice_thickness": slice_thickness, "apodization": 0.5, "time_bw_product": 4, "phase_offset": flip_phase}
                                #rf_ex, gz, gzr= make_sinc_pulse(**kwargs_for_sinc)
                                kwargs_for_sinc = {"flip_angle": flip_angle, "system": system, "duration": RFdur, "slice_thickness": slice_thickness, "apodization": 0.15, "time_bw_product": 2, "phase_offset": flip_phase, "return_gz": True}
                                rf_ex, gz, gzr= make_sinc_pulse(**kwargs_for_sinc)
                                
                                #satPulse      = mr.makeSincPulse(fa_sat, 'Duration', tp, 'system', seq.sys,'timeBwProduct', 2,'apodization', 0.15); % philips-like sinc
                                #%satPulse      = mr.makeGaussPulse(fa_sat, 'Duration', t_p,'system',lims,'timeBwProduct', 0.2,'apodization', 0.5); % siemens-like gauss
                                
                                seq.add_block(gzr)
                                seq.add_block(rf_ex, gz)
                                seq.add_block(gzr)
                                RFdur = gz.rise_time + gz.flat_time + gz.fall_time + gzr.rise_time + gzr.flat_time + gzr.fall_time + gz.delay           
                                
                    
                    elif rep.pulse.usage == Seq.PulseUsage.REFOC:
                        ###############################
                        ### refocusing pulse
                        use = "refocusing"
                        
                        RFdur = 0
                        
                        if torch.abs(rep.pulse.angle) > 1e-8:
                          RFdur = 1*1e-3
                          
                          if nonsel:
                              kwargs_for_block = {"flip_angle": flip_angle, "system": system, "duration": RFdur, "phase_offset": flip_phase}
                              rf_ref = make_block_pulse(**kwargs_for_block)
                              seq.add_block(rf_ref)         
                          else:
    
                              kwargs_for_sinc = {"flip_angle": flip_angle, "system": system, "duration": RFdur, "slice_thickness": slice_thickness, "apodization": 0.5, "time_bw_product": 4, "phase_offset": flip_phase, "return_gz": True}
                              rf_ref, gz_ref,gzr = make_sinc_pulse(**kwargs_for_sinc)
                              seq.add_block(gzr)
                              seq.add_block(rf_ref, gz_ref)
                              seq.add_block(gzr)
                              RFdur = gz_ref.rise_time + gz_ref.flat_time + gz_ref.fall_time 
                              
                    elif rep.pulse.usage == Seq.PulseUsage.FATSAT:
                        ###############################
                        ### fat saturation pulse
                        use = "FatSat"
                        RFdur = 6.120*1e-3
                        dCurrFrequency = 123.2
                        kwargs_for_gauss = {"flip_angle": 110*np.pi/180, "system": system, "slice_thickness": slice_thickness, "duration": RFdur, "freq_offset": -3.3*dCurrFrequency, "time_bw_product": 0.2, "apodization": 0.5, "return_gz": True} # "bandwidth": 200
                        rf_ex, gz, gzr= make_gauss_pulse(**kwargs_for_gauss)
                        seq.add_block(gzr)
                        seq.add_block(rf_ex, gz)
                        seq.add_block(gzr)          
                        
                        
                dur = rep.event_time[event].item() - RFdur
                if dur < 0:
                    raise Exception('Event Time too short! Event Time: Rep: '+ str(i) + ', Event: ' +str(event),', increase event_time by at least: ' + str(-dur))
                
                gx_gradmom = rep.gradm[event,0].item()*deltak
                gy_gradmom = rep.gradm[event,1].item()*deltak
                gz_gradmom = rep.gradm[event,2].item()*deltakz

                if np.abs(gx_gradmom)>0:
                    gx_adc_ramp = 0
                    if any(rep.adc_usage):
                        if torch.abs(rep.adc_usage[event+1]) and torch.abs(rep.gradm[event+1,0]) > 0:
                            kwargs_for_gx = {"channel": 'x', "system": system, "flat_area": rep.gradm[event+1,0].item()*deltak, "flat_time": rep.event_time[event+1].item()}
                            gx_adc = make_trapezoid(**kwargs_for_gx)
                            gx_adc_ramp = gx_adc.amplitude*gx_adc.rise_time/2
                    kwargs_for_gx = {"channel": 'x', "system": system, "area": gx_gradmom-gx_adc_ramp, "duration": dur}
                    gx = make_trapezoid(**kwargs_for_gx) 
                if np.abs(gy_gradmom)>0:
                    gy_adc_ramp = 0
                    if any(rep.adc_usage):
                        if torch.abs(rep.adc_usage[event+1]) and torch.abs(rep.gradm[event+1,1]) > 0:   
                            kwargs_for_gy = {"channel": 'y', "system": system, "flat_area": rep.gradm[event+1,1].item()*deltak, "flat_time": rep.event_time[event+1].item()}
                            gy_adc = make_trapezoid(**kwargs_for_gy)
                            gy_adc_ramp = gy_adc.amplitude*gy_adc.rise_time/2                    
                    kwargs_for_gy = {"channel": 'y', "system": system, "area": gy_gradmom-gy_adc_ramp, "duration": dur}
                    try:
                        gy = make_trapezoid(**kwargs_for_gy)
                    except Exception as e:
                        print(e)
                        print('Event Time too short! Event Time: Rep: '+ str(i) + ', Event: ' +str(event))                              
                if np.abs(gz_gradmom)>0:
                    gz_adc_ramp = 0
                    if any(rep.adc_usage):
                        if torch.abs(rep.adc_usage[event+1]) and torch.abs(rep.gradm[event+1,2]) > 0:   
                            kwargs_for_gz = {"channel": 'y', "system": system, "flat_area": rep.gradm[event+1,2].item()*deltakz, "flat_time": rep.event_time[event+1].item()}
                            gz_adc = make_trapezoid(**kwargs_for_gz)
                            gz_adc_ramp = gz_adc.amplitude*gz_adc.rise_time/2                    
                    kwargs_for_gz = {"channel": 'z', "system": system, "area": gz_gradmom-gz_adc_ramp, "duration": dur}
                    try:
                        gz = make_trapezoid(**kwargs_for_gz)
                    except Exception as e:
                        print(e)
                        print('Event Time too short! Event Time: Rep: '+ str(i) + ', Event: ' +str(event))                              
                            
                if np.abs(gx_gradmom) > 0 and np.abs(gy_gradmom) > 0 and np.abs(gz_gradmom):
                    seq.add_block(gx,gy,gz)
                elif np.abs(gx_gradmom) > 0 and np.abs(gy_gradmom) > 0:
                    seq.add_block(gx,gy)
                elif np.abs(gx_gradmom) > 0 and np.abs(gz_gradmom) > 0:
                    seq.add_block(gx,gz)
                elif np.abs(gy_gradmom) > 0 and np.abs(gz_gradmom) > 0:
                    seq.add_block(gy,gz)
                elif np.abs(gx_gradmom) > 0:
                    seq.add_block(gx)
                elif np.abs(gy_gradmom) > 0:
                    seq.add_block(gy)
                elif np.abs(gz_gradmom) > 0:
                    seq.add_block(gz)
                else:
                    seq.add_block(make_delay(dur))
            else: #adc mask == 1
                if adc_start == 1:
                    pass
                else:
                    adc_start = 1
                    if bdw_start == 0: 
                        bwd = (1/rep.event_time[event])/torch.sum(rep.adc_usage>0)
                        print('Bandwidth is %4d Hz/pixel' % (bwd) )
                        bdw_start = 1
                    
                    idx_T = np.nonzero(torch.abs(rep.adc_usage))                
                    dur = torch.sum(rep.event_time[idx_T],0).item()
                    
                    rise_time_x = 0
                    rise_time_y = 0
                    rise_time_z = 0
                    
                    gx_gradmom = torch.sum(rep.gradm[idx_T,0]).item()*deltak
                    if np.abs(gx_gradmom) > 0:                                       
                        kwargs_for_gx = {"channel": 'x', "system": system, "flat_area": gx_gradmom, "flat_time": dur}
                        gx = make_trapezoid(**kwargs_for_gx)
                        rise_time_x = gx.rise_time
                    
                    gy_gradmom = torch.sum(rep.gradm[idx_T,1]).item()*deltak
                    if np.abs(gy_gradmom) > 0:
                        kwargs_for_gy = {"channel": 'y', "system": system, "flat_area": gy_gradmom, "flat_time": dur}
                        gy = make_trapezoid(**kwargs_for_gy)
                        rise_time_y = gy.rise_time
                    
                    gz_gradmom = torch.sum(rep.gradm[idx_T,2]).item()*deltakz
                    if np.abs(gz_gradmom) > 0:
                        kwargs_for_gz = {"channel": 'z', "system": system, "flat_area": gz_gradmom, "flat_time": dur}
                        gz = make_trapezoid(**kwargs_for_gz) 
                        rise_time_z = gz.rise_time

                    # calculate correct delay to have same starting point of flat top
                    shift = 0.0# rep.event_time[idx_T[0]].item()/2  # heuristic delay, to be checked at scanner
                    x_delay = np.max([0,rise_time_y-rise_time_x,rise_time_z-rise_time_x])+shift
                    y_delay = np.max([0,rise_time_x-rise_time_y,rise_time_z-rise_time_y])+shift
                    z_delay = np.max([0,rise_time_x-rise_time_z,rise_time_y-rise_time_z])+shift
                    
                    rise_time_x = 0
                    rise_time_y = 0
                    rise_time_z = 0
                    
                    # adc gradient events are overwritten with correct delays
                    if np.abs(gx_gradmom) > 0:
                        kwargs_for_gx = {"channel": 'x', "system": system,"delay":x_delay, "flat_area": gx_gradmom, "flat_time": dur}
                        gx = make_trapezoid(**kwargs_for_gx)
                        rise_time_x = gx.rise_time
                    if np.abs(gy_gradmom) > 0:    
                        kwargs_for_gy = {"channel": 'y', "system": system,"delay":y_delay, "flat_area": gy_gradmom, "flat_time": dur}
                        gy = make_trapezoid(**kwargs_for_gy)
                        rise_time_y = gy.rise_time
                    if np.abs(gz_gradmom) > 0:
                        kwargs_for_gz = {"channel": 'z', "system": system,"delay":z_delay, "flat_area": gz_gradmom, "flat_time": dur}
                        gz = make_trapezoid(**kwargs_for_gz)  
                        rise_time_z = gz.rise_time
                    
                    adc_delay = np.max([rise_time_x,rise_time_y,rise_time_z])+shift
                    kwargs_for_adc = {"num_samples": idx_T.size()[0], "duration": dur, "delay":(adc_delay), "phase_offset": rf_ex.phase_offset - np.pi/4}
                    adc = make_adc(**kwargs_for_adc)    
                    
                    # dont play zero grads (cant even do FID otherwise)
                    if np.abs(gx_gradmom) > 0 and np.abs(gy_gradmom) > 0 and np.abs(gz_gradmom):
                        seq.add_block(gx,gy,gz,adc)
                    elif np.abs(gx_gradmom) > 0 and np.abs(gy_gradmom) > 0:
                        seq.add_block(gx,gy,adc)
                    elif np.abs(gx_gradmom) > 0 and np.abs(gz_gradmom) > 0:
                        seq.add_block(gx,gz,adc)
                    elif np.abs(gy_gradmom) > 0 and np.abs(gz_gradmom) > 0:
                        seq.add_block(gy,gz,adc)
                    elif np.abs(gx_gradmom) > 0:
                        seq.add_block(gx,adc)
                    elif np.abs(gy_gradmom) > 0:
                        seq.add_block(gy,adc)
                    elif np.abs(gz_gradmom) > 0:
                        seq.add_block(gz,adc)
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
    
    append_header(path, FOV,slice_thickness)
    
    # TODO: This function is missing
    # if write_data:
    #     util.write_data_to_seq_file(seq_param,path)
    #     print('Add kspace & adc_usage information to seq File!')

def append_header(path, FOV,slice_thickness):
    # append version and definitions
    if sys.platform != 'linux':
        try:
            with open(r"\\141.67.249.47\MRTransfer\mrzero_src\.git\ORIG_HEAD") as file:
                git_version = file.read()
        except:
            git_version = ''
    with open(path, 'r') as fin:
        lines = fin.read().splitlines(True)

    updated_lines = []
    updated_lines.append("# Pulseq sequence file\n")
    updated_lines.append("# Created by MRIzero/IMR/GPI pulseq converter\n")
    if sys.platform != 'linux':
        updated_lines.append('# MRZero Version: 0.5, git hash: ' + git_version)
    if sys.platform == 'linux':
        updated_lines.append("# experiment_id: "+path.split('/')[-2]+"\n")
    else:
        updated_lines.append("# experiment_id: "+path.split('\\')[-2]+"\n")
    updated_lines.append('#' + path + "\n")
    updated_lines.append("\n")
    # updated_lines.append("[VERSION]\n")
    # updated_lines.append("major 1\n")
    # updated_lines.append("minor 2\n")   
    # updated_lines.append("revision 1\n")  
    # updated_lines.append("\n")    
    updated_lines.append("[DEFINITIONS]\n")
    updated_lines.append("FOV "+str(FOV)+" "+str(FOV)+" "+str(slice_thickness)+" \n")   
    updated_lines.append("\n")    
    
    updated_lines.extend(lines[3:])

    with open(path, 'w') as fout:
        fout.writelines(updated_lines)    