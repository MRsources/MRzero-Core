import matplotlib.pyplot as plt
import numpy as np
from pypulseq.calc_rf_center import calc_rf_center
from pypulseq.calc_duration import calc_duration
from pypulseq.Sequence.sequence import Sequence


def pulseq_plot(seq: Sequence, type: str = 'Gradient', time_range=(0, np.inf), time_disp: str = 's', clear=False, signal=0):
    """
    Plot `Sequence`.
    Parameters
    ----------
    type : str
        Gradients display type, must be one of either 'Gradient' or 'Kspace'.
    time_range : List
        Time range (x-axis limits) for plotting the sequence. Default is 0 to infinity (entire sequence).
    time_disp : str
        Time display type, must be one of `s`, `ms` or `us`.
    """

    valid_plot_types = ['Gradient', 'Kspace']
    valid_time_units = ['s', 'ms', 'us']
    if type not in valid_plot_types:
        raise Exception()
    if time_disp not in valid_time_units:
        raise Exception()

    fig2, fig1 = plt.figure(2), plt.figure(1)
    fig1_sp_list = fig1.get_axes(); fig2_sp_list = fig2.get_axes()

    if clear:
        for ax in fig1_sp_list + fig2_sp_list:
            ax.remove()
        fig1_sp_list = fig1.get_axes(); fig2_sp_list = fig2.get_axes()

    if len(fig1_sp_list) == 3:
        (sp11, sp12, sp13) = fig1_sp_list
    else:
        for ax in fig1_sp_list:
            ax.remove()

        sp11 = fig1.add_subplot(311)
        sp12, sp13 = fig1.add_subplot(312, sharex=sp11), fig1.add_subplot(313, sharex=sp11)

    if len(fig2_sp_list) == 3:
        fig2_sp_list = fig2_sp_list
    else:
        for ax in fig2_sp_list:
            ax.remove()

        fig2_sp_list = [fig2.add_subplot(311, sharex=sp11), fig2.add_subplot(312, sharex=sp11), fig2.add_subplot(313, sharex=sp11)]

    t_factor_list = [1, 1e3, 1e6]
    t_factor = t_factor_list[valid_time_units.index(time_disp)]
    t0 = 0
    t_adc = []
    for iB in range(1, len(seq.dict_block_events) + 1):
        block = seq.get_block(iB)
        is_valid = time_range[0] <= t0 <= time_range[1]
        if is_valid:
            if hasattr(block, 'adc'):
                adc = block.adc
                t = adc.delay + [(x * adc.dwell) for x in range(0, int(adc.num_samples))]
                sp11.plot((t0 + t), np.zeros(len(t)), 'rx')
                t_adc = np.append(t_adc, t0 + t)
            if hasattr(block, 'rf'):
                rf = block.rf
                tc, ic = calc_rf_center(rf)
                t = rf.t + rf.delay
                tc = tc + rf.delay
                sp12.plot(t_factor * (t0 + t), abs(rf.signal))
                sp12.fill_between(t_factor * (t0 + t), 0, abs(rf.signal),alpha=0.5)
                sp13.plot(t_factor * (t0 + t),
                          np.angle(rf.signal * np.exp(1j * rf.phase_offset) * np.exp(1j * 2 * np.pi * rf.t * rf.freq_offset)),
                          t_factor * (t0 + tc),
                          np.angle(rf.signal[ic] * np.exp(1j * rf.phase_offset) * np.exp(1j * 2 * np.pi * rf.t[ic] * rf.freq_offset)),
                          'xb')
                sp13.fill_between(t_factor * (t0 + t), 0, np.angle(rf.signal * np.exp(1j * rf.phase_offset) * np.exp(1j * 2 * np.pi * rf.t * rf.freq_offset)), alpha=0.5)
                sp13.fill_between(t_factor * (t0 + t), 0, np.angle(rf.signal[ic] * np.exp(1j * rf.phase_offset) * np.exp(1j * 2 * np.pi * rf.t[ic] * rf.freq_offset)), alpha=0.5)
            grad_channels = ['gx', 'gy', 'gz']
            for x in range(0, len(grad_channels)):
                if hasattr(block, grad_channels[x]):
                    grad = getattr(block, grad_channels[x])
                    if grad.type == 'grad':
                        # In place unpacking of grad.t with the starred expression
                        t = grad.delay + [0, *(grad.t + (grad.t[1] - grad.t[0]) / 2),
                                          grad.t[-1] + grad.t[1] - grad.t[0]]
                        waveform = np.array([grad.first, grad.last])
                        waveform = 1e-3 * np.insert(waveform, 1, grad.waveform)
                    else:
                        t = np.cumsum([0, grad.delay, grad.rise_time, grad.flat_time, grad.fall_time])
                        waveform = 1e-3 * grad.amplitude * np.array([0, 0, 1, 1, 0])
                    fig2_sp_list[x].plot(t_factor * (t0 + t), waveform)
        t0 += calc_duration(block)

    grad_plot_labels = ['x', 'y', 'z']
    sp11.set_ylabel('ADC'); sp11.grid()
    sp12.set_ylabel('RF mag (Hz)'); sp12.grid()
    sp13.set_ylabel('RF phase (rad)'); sp13.grid()
    [fig2_sp_list[x].set_ylabel(f'G{grad_plot_labels[x]} (kHz/m)') for x in range(3)]
    [fig2_sp_list[x].grid() for x in range(3)]
    # Setting display limits
    disp_range = t_factor * np.array([time_range[0], min(t0, time_range[1])])
    sp11.set_xlim(disp_range)
    sp12.set_xlim(disp_range)
    sp13.set_xlim(disp_range)
    [x.set_xlim(disp_range) for x in fig2_sp_list]

    sp11.plot((t0 + t), np.zeros(len(t)), 'rx')
    if np.size(signal)>1:
        sp11.plot(t_adc, np.real(signal), t_adc, np.imag(signal))
    plt.show()

    return sp11, t_adc
