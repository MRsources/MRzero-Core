import os
import time
from typing import Literal, Union, Optional
import torch
import numpy as np

import matplotlib.pyplot as plt
import pypulseq as pp


def get_signal_from_real_system(path: str, NRep: int, NRead: int,ncoils = 20, heuristic_shift = 4):
    """Wait for a TWIX file and return its data.

    This function assumes 20 recieve coils and a readout with equal number of
    ADC samples for all repetitions.

    Parameters
    ----------
    path : str
        Path to TWIX file
    NRep : int
        Number of repetitions of the measured sequence
    NRead : int
        Number of ADC samples per repetition

    Returns
    -------
    torch.tensor
        A (samples x coils) tensor with the signal extracted from the file
    """ 
    
    
    print('waiting for TWIX file from the scanner... ' + path)
    done_flag = False
    while not done_flag:
        if os.path.isfile(path):
            # read twix file
            print("TWIX file arrived. Reading....")

            time.sleep(0.2)
            raw = np.loadtxt(path)

            expected_size = NRep * ncoils * (NRead + heuristic_shift) * 2
            print(f"raw size: {raw.size}, expected size: {expected_size}")

            if raw.size != expected_size:
                print("get_signal_from_real_system: SERIOUS ERROR, "
                      "TWIX dimensions corrupt, returning zero array..")
                print(f"Try different nuber of coils (ncoils), maybe around {raw.size/(2*NRep*NRead + heuristic_shift*2*NRep)}.")
                raw = np.zeros((NRep, ncoils, NRead + heuristic_shift, 2))
                raw = raw[:, :, :NRead, 0] + 1j * raw[:, :, :NRead, 1]
            else:
                raw = raw.reshape([NRep, ncoils, NRead + heuristic_shift, 2])
                raw = raw[:, :, :NRead, 0] + 1j * raw[:, :, :NRead, 1]

            # raw = raw.transpose([1,2,0]) #ncoils,NRep,NRead
            raw = raw.transpose([0, 2, 1])  # NRep,NRead,NCoils
            raw = raw.reshape([NRep * NRead, ncoils])
            raw = np.copy(raw)
            done_flag = True

    return torch.tensor(raw, dtype=torch.complex64)


def insert_signal_plot(seq: pp.Sequence, signal: np.ndarray):
    """Insert a measured signal into a currently open pypulseq plot.

    Usage:
    ```
    seq.plot(plot_now=False)
    mr0.util.insert_signal_plot(seq, signal.numpy())
    plt.show()
    ```

    Parameters
    ----------
    seq : pypulseq.Sequence
        The sequence that is plotted and produced the signal
    signal : np.ndarray
        The signal that should be inserted into the ADC plot.
        Has to have the same amount of samples as the sequence itself.
    """
    remaining_signal = signal.flatten().tolist()
    t0 = 0
    time = []
    samples = []

    for iB in range(1, len(seq.block_events) + 1):
        block = seq.get_block(iB)
        if getattr(block, "adc", None):
            adc = block.adc
            count = int(adc.num_samples)
            time += [t0 + adc.delay + (t + 0.5) * adc.dwell for t in range(count)] + [float("nan")]
            samples += remaining_signal[:count] + [float("nan")]
            remaining_signal = remaining_signal[count:]
        t0 += pp.calc_duration(block)

    if len(time) != len(samples) + len(remaining_signal):
        print("Can't insert signal into pulseq plot:")
        print("Signal and sequence have different amount of ADC samples.")
    else:
        sp11 = plt.figure(1).get_axes()[0]

        sp11.plot(time, np.abs(samples),  label='abs')
        sp11.plot(time, np.real(samples), label='real', linewidth=0.5)
        sp11.plot(time, np.imag(samples), label='imag', linewidth=0.5)

        sp11.legend().remove()  # clear old legend
        sp11.legend(loc='upper right', bbox_to_anchor=(1.02, 1.0), fontsize='small')



def pulseq_plot(seq: pp.Sequence,
                type: Literal['Gradient', 'Kspace'] = 'Gradient',
                time_range: tuple[float, float] = (0, np.inf),
                time_disp: Literal['s', 'ms', 'us'] = 's',
                show_blocks=False,
                clear=False, signal=None, figid=(1, 2)):
    try:
        signal=signal.numpy()
    except:
        signal=signal
    try: 
        version = f"{pp.Sequence.version_major}.{pp.Sequence.version_minor}.{pp.Sequence.version_revision}" # this works for 1.4
    except:
        try:
            version=f"{pp.major}.{pp.minor}.{pp.revision}"  # this works for 1.3post1 and lower
        except:
            version=pp.__version__ # forgot in which version this worked, but here it is if everything else fails

    version=float(version[:3]) # to float major.minor
        
    if version>=1.4:
        # This works since pypulseq 1.4, which introduced the plot_now argument
        print("plotting with pulseq 1.4")
        if signal is None:
            sp_adc, t_adc = pulseq_plot_142(seq=seq,time_range=time_range,time_disp=time_disp,show_blocks=show_blocks)
        else:
            sp_adc, t_adc = pulseq_plot_142(seq=seq,plot_now=False,time_range=time_range,time_disp=time_disp,show_blocks=show_blocks)
            insert_signal_plot(seq, signal)
            plt.show()
    else:
        # This works for older pypulseq versions expect the newest dev branch
        print("plotting with pulseq <=1.3")
        sp_adc, t_adc = pulseq_plot_pre14(seq=seq,time_range=time_range,time_disp=time_disp,signal=signal)
    
    return sp_adc, t_adc

# This plot function is a modified version from the one provided by
# pypulseq 1.4.2, all changes are marked
# NOTE: the parameters have changed in the 1.4 version, maybe we should adapt them
def pulseq_plot_142(seq: pp.Sequence,
        label: str = str(),
        show_blocks: bool = False,
        save: bool = False,
        time_range=(0, np.inf),
        time_disp: str = "s",
        grad_disp: str = "kHz/m",
        plot_now: bool = True
    ) -> None:
        """
        Plot `Sequence`.

        Parameters
        ----------
        label : str, defualt=str()
            Plot label values for ADC events: in this example for LIN and REP labels; other valid labes are accepted as
            a comma-separated list.
        save : bool, default=False
            Boolean flag indicating if plots should be saved. The two figures will be saved as JPG with numerical
            suffixes to the filename 'seq_plot'.
        show_blocks : bool, default=False
            Boolean flag to indicate if grid and tick labels at the block boundaries are to be plotted.
        time_range : iterable, default=(0, np.inf)
            Time range (x-axis limits) for plotting the sequence. Default is 0 to infinity (entire sequence).
        time_disp : str, default='s'
            Time display type, must be one of `s`, `ms` or `us`.
        grad_disp : str, default='s'
            Gradient display unit, must be one of `kHz/m` or `mT/m`.
        plot_now : bool, default=True
            If true, function immediately shows the plots, blocking the rest of the code until plots are exited.
            If false, plots are shown when plt.show() is called. Useful if plots are to be modified.
        plot_type : str, default='Gradient'
            Gradients display type, must be one of either 'Gradient' or 'Kspace'.
        """
        from pypulseq.supported_labels_rf_use import get_supported_labels
        from pypulseq.calc_rf_center import calc_rf_center
        from pypulseq.Sequence.sequence import cumsum
        import math
        import matplotlib as mpl


        mpl.rcParams["lines.linewidth"] = 0.75  # Set default Matplotlib linewidth

        valid_time_units = ["s", "ms", "us"]
        valid_grad_units = ["kHz/m", "mT/m"]




        valid_labels = get_supported_labels()
        if (
            not all([isinstance(x, (int, float)) for x in time_range])
            or len(time_range) != 2
        ):
            raise ValueError("Invalid time range")
        if time_disp not in valid_time_units:
            raise ValueError("Unsupported time unit")

        if grad_disp not in valid_grad_units:
            raise ValueError(
                "Unsupported gradient unit. Supported gradient units are: "
                + str(valid_grad_units)
            )

        fig1, fig2 = plt.figure(1), plt.figure(2)
        ### change satrts here
        clear=False
        fig1_sp_list = fig1.get_axes()
        fig2_sp_list = fig2.get_axes()
        clear=False

        if clear:
            for ax in fig1_sp_list + fig2_sp_list:
                ax.remove()
            fig1_sp_list = fig1.get_axes()
            fig2_sp_list = fig2.get_axes()

        if len(fig1_sp_list) == 3:
            (sp11, sp12, sp13) = fig1_sp_list
        else:
            for ax in fig1_sp_list:
                ax.remove()

            # These 3 subplots are unchanged from pypulseq 1.2
            sp11 = fig1.add_subplot(311)
            sp12 = fig1.add_subplot(312, sharex=sp11)
            sp13 = fig1.add_subplot(313, sharex=sp11)

        if len(fig2_sp_list) == 3:
            fig2_sp_list = fig2_sp_list
            fig2_subplots = fig2_sp_list
        else:
            for ax in fig2_sp_list:
                ax.remove()

            # This is also straight from pypulseq 1.2
            fig2_sp_list = [
                fig2.add_subplot(311, sharex=sp11),
                fig2.add_subplot(312, sharex=sp11),
                fig2.add_subplot(313, sharex=sp11)
            ]
            fig2_subplots = fig2_sp_list
        ### change ends here
        t_factor_list = [1, 1e3, 1e6]
        t_factor = t_factor_list[valid_time_units.index(time_disp)]

        g_factor_list = [1e-3, 1e3 / seq.system.gamma]
        g_factor = g_factor_list[valid_grad_units.index(grad_disp)]

        t0 = 0
        t_adc = [] # >>>> Changed: store adc samples <<<<
        label_defined = False
        label_idx_to_plot = []
        label_legend_to_plot = []
        label_store = dict()
        for i in range(len(valid_labels)):
            label_store[valid_labels[i]] = 0
            if valid_labels[i] in label.upper():
                label_idx_to_plot.append(i)
                label_legend_to_plot.append(valid_labels[i])

        if len(label_idx_to_plot) != 0:
            p = parula.main(len(label_idx_to_plot) + 1)
            label_colors_to_plot = p(np.arange(len(label_idx_to_plot)))
            cycler = mpl.cycler(color=label_colors_to_plot)
            sp11.set_prop_cycle(cycler)


        # Block timings
        block_edges = np.cumsum([0] + [x[1] for x in sorted(seq.block_durations.items())])
        block_edges_in_range = block_edges[
            (block_edges >= time_range[0]) * (block_edges <= time_range[1])
        ]
        if show_blocks:
            for sp in [sp11, sp12, sp13, *fig2_subplots]:
                sp.set_xticks(t_factor * block_edges_in_range)
                sp.set_xticklabels(sp.get_xticklabels(), rotation=90)

        for block_counter in seq.block_events:
            block = seq.get_block(block_counter)
            is_valid = (time_range[0] <= t0 + seq.block_durations[block_counter]
                        and t0 <= time_range[1])
            if is_valid:
                if getattr(block, "label", None) is not None:
                    for i in range(len(block.label)):
                        if block.label[i].type == "labelinc":
                            label_store[block.label[i].label] += block.label[i].value
                        else:
                            label_store[block.label[i].label] = block.label[i].value
                    label_defined = True

                if getattr(block, "adc", None) is not None:  # ADC
                    adc = block.adc
                    # From Pulseq: According to the information from Klaus Scheffler and indirectly from Siemens this
                    # is the present convention - the samples are shifted by 0.5 dwell
                    t = adc.delay + (np.arange(int(adc.num_samples)) + 0.5) * adc.dwell
                    sp11.plot(t_factor * (t0 + t), np.zeros(len(t)), "rx")
                    
                    t_adc = np.append(t_adc, t0 + t) # >>>> Changed: store adc samples <<<<
                    sp13.plot(
                        t_factor * (t0 + t),
                        np.angle(
                            np.exp(1j * adc.phase_offset)
                            * np.exp(1j * 2 * np.pi * t * adc.freq_offset)
                        ),
                        "b.",
                        markersize=0.25,
                    )

                    if label_defined and len(label_idx_to_plot) != 0:
                        arr_label_store = list(label_store.values())
                        lbl_vals = np.take(arr_label_store, label_idx_to_plot)
                        t = t0 + adc.delay + (adc.num_samples - 1) / 2 * adc.dwell
                        _t = [t_factor * t] * len(lbl_vals)
                        # Plot each label individually to retrieve each corresponding Line2D object
                        p = itertools.chain.from_iterable(
                            [
                                sp11.plot(__t, _lbl_vals, ".")
                                for __t, _lbl_vals in zip(_t, lbl_vals)
                            ]
                        )
                        if len(label_legend_to_plot) != 0:
                            sp11.legend(p, label_legend_to_plot, loc="upper left")
                            label_legend_to_plot = []

                if getattr(block, "rf", None) is not None:  # RF
                    rf = block.rf
                    tc, ic = calc_rf_center(rf)
                    time = rf.t
                    signal = rf.signal
                    if abs(signal[0]) != 0:
                        signal = np.concatenate(([0], signal))
                        time = np.concatenate(([time[0]], time))
                        ic += 1

                    if abs(signal[-1]) != 0:
                        signal = np.concatenate((signal, [0]))
                        time = np.concatenate((time, [time[-1]]))

                    sp12.plot(t_factor * (t0 + time + rf.delay), np.abs(signal))
                    sp13.plot(
                        t_factor * (t0 + time + rf.delay),
                        np.angle(
                            signal
                            * np.exp(1j * rf.phase_offset)
                            * np.exp(1j * 2 * math.pi * time * rf.freq_offset)
                        ),
                        t_factor * (t0 + tc + rf.delay),
                        np.angle(
                            signal[ic]
                            * np.exp(1j * rf.phase_offset)
                            * np.exp(1j * 2 * math.pi * time[ic] * rf.freq_offset)
                        ),
                        "xb",
                    )

                grad_channels = ["gx", "gy", "gz"]
                for x in range(len(grad_channels)):  # Gradients
                    if getattr(block, grad_channels[x], None) is not None:
                        grad = getattr(block, grad_channels[x])
                        if grad.type == "grad":
                            # We extend the shape by adding the first and the last points in an effort of making the
                            # display a bit less confusing...
                            time = grad.delay + np.array([0, *grad.tt, grad.shape_dur])
                            waveform = g_factor * np.array(
                                (grad.first, *grad.waveform, grad.last)
                            )
                        else:
                            time = np.array(cumsum(
                                    0,
                                    grad.delay,
                                    grad.rise_time,
                                    grad.flat_time,
                                    grad.fall_time,
                            ))
                            waveform = (
                                g_factor * grad.amplitude * np.array([0, 0, 1, 1, 0])
                            )
                        fig2_subplots[x].plot(t_factor * (t0 + time), waveform)
            t0 += seq.block_durations[block_counter]

        grad_plot_labels = ["x", "y", "z"]
        sp11.set_ylabel("ADC")
        sp12.set_ylabel("RF mag (Hz)")
        sp13.set_ylabel("RF/ADC phase (rad)")
        sp13.set_xlabel(f"t ({time_disp})")
        for x in range(3):
            _label = grad_plot_labels[x]
            fig2_subplots[x].set_ylabel(f"G{_label} ({grad_disp})")
        fig2_subplots[-1].set_xlabel(f"t ({time_disp})")

        # Setting display limits
        disp_range = t_factor * np.array([time_range[0], min(t0, time_range[1])])
        [x.set_xlim(disp_range) for x in [sp11, sp12, sp13, *fig2_subplots]]

        # Grid on
        for sp in [sp11, sp12, sp13, *fig2_subplots]:
            sp.grid()

        fig1.tight_layout()
        fig2.tight_layout()
        if save:
            fig1.savefig("seq_plot1.jpg")
            fig2.savefig("seq_plot2.jpg")

        if plot_now:
            plt.show()

        return sp11, t_adc

# This plot function is a modified version from the one provided by
# pypulseq 1.2.0post1, all changes are marked
# NOTE: the parameters have changed in the 1.4 version, maybe we should adapt them
def pulseq_plot_pre14(seq: pp.Sequence,
                type: Literal['Gradient', 'Kspace'] = 'Gradient',
                time_range: tuple[float, float] = (0, np.inf),
                time_disp: Literal['s', 'ms', 'us'] = 's',
                clear=False, signal=0, figid=(1, 2)):
    """Modified pypulseq Sequence.plot() that includes an ADC signal.

    Parameters
    ----------
    seq : pypulseq.Sequence
        The sequence object to plot
    type : str
        Gradients display type, must be one of either 'Gradient' or 'Kspace'.
    time_range : (float, float)
        Time range (x-axis limits) for plotting the sequence.
        Default is 0 to infinity (entire sequence).
    time_disp : str
        Time display type, must be one of `s`, `ms` or `us`.
    """

    valid_plot_types = ['Gradient', 'Kspace']
    valid_time_units = ['s', 'ms', 'us']
    if type not in valid_plot_types:
        raise Exception()
    if time_disp not in valid_time_units:
        raise Exception()

    fig1, fig2 = plt.figure(figid[0]), plt.figure(figid[1])

# >>>> This is changed compared to pypulseq 1.2
    fig1_sp_list = fig1.get_axes()
    fig2_sp_list = fig2.get_axes()

    if clear:
        for ax in fig1_sp_list + fig2_sp_list:
            ax.remove()
        fig1_sp_list = fig1.get_axes()
        fig2_sp_list = fig2.get_axes()

    if len(fig1_sp_list) == 3:
        (sp11, sp12, sp13) = fig1_sp_list
    else:
        for ax in fig1_sp_list:
            ax.remove()

        # These 3 subplots are unchanged from pypulseq 1.2
        sp11 = fig1.add_subplot(311)
        sp12 = fig1.add_subplot(312, sharex=sp11)
        sp13 = fig1.add_subplot(313, sharex=sp11)

    if len(fig2_sp_list) == 3:
        fig2_sp_list = fig2_sp_list
    else:
        for ax in fig2_sp_list:
            ax.remove()

        # This is also straight from pypulseq 1.2
        fig2_sp_list = [
            fig2.add_subplot(311, sharex=sp11),
            fig2.add_subplot(312, sharex=sp11),
            fig2.add_subplot(313, sharex=sp11)
        ]
# <<<< End of change

    t_factor_list = [1, 1e3, 1e6]
    t_factor = t_factor_list[valid_time_units.index(time_disp)]
    t0 = 0
    t_adc = []
    N_adc = [0, 0]
# >>>> Changed
    try:
        block_events = seq.dict_block_events
    except AttributeError:
        block_events = seq.block_events

    for iB in range(1, len(block_events) + 1):
# <<<< End of change
        block = seq.get_block(iB)
        is_valid = time_range[0] <= t0 <= time_range[1]
        if is_valid:
            if getattr(block, 'adc', None) is not None:
                adc = block.adc
                t = [(adc.delay + x * adc.dwell) for x in range(0, int(adc.num_samples))]
                sp11.plot((t0 + t), np.zeros(len(t)), 'rx')
# >>>> Changed: store adc samples <<<<
                t_adc = np.append(t_adc, t0 + t)
                N_adc[1] += 1
                N_adc[0] += int(adc.num_samples)
            if getattr(block, 'rf', None) is not None:
                rf = block.rf
                tc, ic = pp.calc_rf_center(rf)
                t = rf.t + rf.delay
                tc = tc + rf.delay
                sp12.plot(t_factor * (t0 + t), abs(rf.signal))
                sp13.plot(
                    t_factor * (t0 + t),
                    np.angle(
                        rf.signal * np.exp(1j * rf.phase_offset) *
                        np.exp(1j * 2 * np.pi * rf.t * rf.freq_offset)
                    ),
                    t_factor * (t0 + tc),
                    np.angle(
                        rf.signal[ic] * np.exp(1j * rf.phase_offset) *
                        np.exp(1j * 2 * np.pi * rf.t[ic] * rf.freq_offset)
                    ),
                    'xb'
                )
# >>>> Changed
                sp12.fill_between(
                    t_factor * (t0 + t), 0,
                    abs(rf.signal),
                    alpha=0.5
                )
                sp13.fill_between(
                    t_factor * (t0 + t), 0,
                    np.angle(
                        rf.signal * np.exp(1j * rf.phase_offset) *
                        np.exp(1j * 2 * np.pi * rf.t * rf.freq_offset)
                    ),
                    alpha=0.5
                )
                sp13.fill_between(
                    t_factor * (t0 + t), 0,
                    np.angle(
                        rf.signal[ic] * np.exp(1j * rf.phase_offset) *
                        np.exp(1j * 2 * np.pi * rf.t[ic] * rf.freq_offset)
                    ),
                    alpha=0.5
                )
# <<<< End of change
            grad_channels = ['gx', 'gy', 'gz']
            for x in range(0, len(grad_channels)):
                if getattr(block, grad_channels[x], None) is not None:
                    grad = getattr(block, grad_channels[x])
                    if grad.type == 'grad':
                        t = grad.delay + [
                            0,
                            *(grad.t + (grad.t[1] - grad.t[0]) / 2),
                            grad.t[-1] + grad.t[1] - grad.t[0]
                        ]
                        waveform = np.array([grad.first, grad.last])
                        waveform = 1e-3 * np.insert(waveform, 1, grad.waveform)
                    else:
                        t = np.cumsum([
                            0, grad.delay, grad.rise_time,
                            grad.flat_time, grad.fall_time
                        ])
                        waveform = (
                            1e-3 * grad.amplitude *
                            np.array([0, 0, 1, 1, 0])
                        )
                    fig2_sp_list[x].plot(t_factor * (t0 + t), waveform)
        t0 += pp.calc_duration(block)

    grad_plot_labels = ['x', 'y', 'z']
    sp11.set_ylabel('ADC')
    sp12.set_ylabel('RF mag (Hz)')
    sp13.set_ylabel('RF phase (rad)')
    [fig2_sp_list[x].set_ylabel(
        f'G{grad_plot_labels[x]} (kHz/m)') for x in range(3)]

# >>>> Changed: added grid
    sp11.grid('on')
    sp12.grid('on')
    sp13.grid('on')
    [fig2_sp_list[x].grid('on') for x in range(3)]
# <<<< End of change
    # Setting display limits
    disp_range = t_factor * np.array([time_range[0], min(t0, time_range[1])])
    sp11.set_xlim(disp_range)
    sp12.set_xlim(disp_range)
    sp13.set_xlim(disp_range)
    [x.set_xlim(disp_range) for x in fig2_sp_list]

# >>>> Changed: Plot signal and adc samples perhaps?

    if np.size(signal) > 1:
        N_adc[0] = N_adc[0]/N_adc[1]
        if N_adc[0].is_integer():
            idx = np.arange(int(N_adc[0]),
                            int(N_adc[0])*N_adc[1],
                            int(N_adc[0]))
            t_adc_p = np.insert(t_adc, idx, np.nan, axis=None)
            signal = np.insert(signal, idx, np.nan, axis=None)

            sp11.plot(t_adc_p, np.abs(signal),  label='abs')

            sp11.plot(t_adc_p, np.real(signal), label='real', linewidth=0.5)
            sp11.plot(t_adc_p, np.imag(signal), label='imag', linewidth=0.5)

            sp11.legend(loc='right', bbox_to_anchor=(
                1.12, 0.5), fontsize='xx-small')
        else:
            print('Your ADCs seem to have different samples, '
                  'this cannot be plotted.')
            print(N_adc)


# <<<< End of change
    plt.show()

# New: return plot axes and adc time points
    return sp11, t_adc


def imshow(data: Union[np.ndarray, torch.Tensor], *args, **kwargs):
    """Alternative to matplotlib's `imshow`.
    
    This function applies quadratic coil combine on 4D data and prints 3D
    data as a grid of slices. Also prints x-axis horizontal and y vertiacl.
    
    Assumes data to be indexed [c, x, y, z]"""
    data = torch.as_tensor(data).detach().cpu()
    assert 2 <= data.ndim <= 4

    # Coil combine 4D data
    if data.ndim == 4:
        data = (data.abs()**2).sum(0)**0.5
    
    # Shape 3D data into grid
    if data.ndim == 3:
        rows = int(np.floor(data.shape[2]**0.5))
        cols = int(np.ceil(data.shape[2] / rows))
        
        tmp = data
        data = torch.zeros((tmp.shape[0] * cols, tmp.shape[1] * rows), dtype=tmp.dtype)
        for i in range(tmp.shape[2]):
            x = (i % cols)*tmp.shape[0]
            y = ((rows * cols - i - 1) // cols)*tmp.shape[1]
            data[x:x+tmp.shape[0], y:y+tmp.shape[1]] = tmp[:, :, i]

    return plt.imshow(data.T, *args, origin="lower", **kwargs)


DEFAULT_PHANTOM_URL = "https://github.com/MRsources/MRzero-Core/raw/main/documentation/playground_mr0/numerical_brain_cropped.mat"


def load_phantom(size: Optional[tuple[int, int]] = None,
                 url=DEFAULT_PHANTOM_URL,
                 dB0_fB0=(0, 1), dB1_fB1=(0, 1),
                 B0_polynomial: Optional[list[float]] = None):
    """Download a phantom from the given URL, which by default is a quantified
    phantom from https://github.com/MRsources/MRzero-Core.

    Parameters
    ----------
    size : (int, int) | None
        If a size is passed, the phantom is scaled to it
    url : str
        The URL to download the phantom from
    dB0_fB0 : (float, float)
        (offset, scaling) for the loaded B0 data: dB0_fB0[1] * (B0 + dB0_fB0[0])
    dB1_fB1 : (float, float)
        (offset, scaling) for the loaded B1 data: dB1_fB1[1] * (B1 + dB1_fB1[0])
    B0_polynomial : list[float] | None
        Construct a B0 map and add it to the loaded data. The polynomial is
        defined by [x, y, x^2, y^2, x*y], where x, y are in the range [-1, 1[
    
    Returns
    -------
    mr0.VoxelGridPhantom:
        Loaded phantom. Call .build() and pass the result to the simulation
    """
    from urllib.request import urlretrieve
    from MRzeroCore import VoxelGridPhantom
    import os
    phantom_name = url.split("/")[-1]

    if not os.path.exists(phantom_name):
        urlretrieve(url, phantom_name)
        print("Downloaded simulation data")

    if phantom_name.endswith(".mat"):
        phantom = VoxelGridPhantom.load_mat(phantom_name)
    elif phantom_name.endswith(".npz"):
        phantom = VoxelGridPhantom.brainweb(phantom_name)
    else:
        raise ValueError(f"Can't load {phantom_name}: unknown file extension")

    # Apply scaling - if requested
    if size is not None:
        phantom = phantom.interpolate(*size, 1)
    
    # Apply scaling
    phantom.B0 += dB0_fB0[0]
    phantom.B0 *= dB0_fB0[1]
    phantom.B1 += dB1_fB1[0]
    phantom.B1 *= dB1_fB1[1]

    if B0_polynomial is not None:
        x, y = torch.meshgrid(
            torch.linspace(-1, 1, phantom.B0.shape[0] + 1)[:-1],
            torch.linspace(-1, 1, phantom.B0.shape[1] + 1)[:-1]
        )

        if phantom.PD.shape[2] != 1:
            raise ValueError("load_phantom() with B0_polynomial currently only works with 2D phantoms")
        nB0 = torch.full_like(phantom.B0, B0_polynomial[0])[:, :, 0]

        if len(B0_polynomial) > 1:
            nB0 += x * B0_polynomial[1]
        if len(B0_polynomial) > 2:
            nB0 += y * B0_polynomial[2]
        if len(B0_polynomial) > 3:
            nB0 += x*x * B0_polynomial[3]
        if len(B0_polynomial) > 4:
            nB0 += y*y * B0_polynomial[4]
        if len(B0_polynomial) > 5:
            nB0 += x*y * B0_polynomial[5]

        phantom.B0 += nB0[:,:,None]

    return phantom


def simulate(seq, phantom=None, sim_size=None, accuracy=1e-3, noise_level=None):
    """Simulate the sequence with the phantom using PDG and default settings.
    
    Arguments
    ---------
    seq: mr0.Sequence | pypulseq.Sequence | str
        Sequence to simulate.
        Either an mr0 or a pypulseq sequence object or the path to an .seq file
    phantom: VoxelGridPhantom | CustomVoxelPhantom | str | None
        Phantom used for simulation. Either an mr0 phantom, the path to a
        .npz or .mat phantom or None to load a default phantom from GitHub
    sim_size: (int, int) | None
        If given, the center slice of the phantom is taken and scaled to size
    accuracy: float
        Used for simulation precision (min_emitted_signal and min_latent_signal)
    noise_level: float | None
        Normal distributed noise with the specifed std is added to the signal.

    Returns
    -------
    (torch.Tensor, torch.Tensor)
        A tuple containing (ADC signal, kspace)
    """
    import MRzeroCore as mr0

    # Phantom loading: use passed phantom, load from file or default from URL
    if phantom is None:
        phantom = load_phantom(sim_size)
    else:
        if isinstance(phantom, str):
            if phantom.endswith(".mat"):
                phantom = mr0.VoxelGridPhantom.load_mat(phantom)
            else:
                phantom = mr0.VoxelGridPhantom.load(phantom)

        if sim_size is not None:
            slices = phantom.PD.shape[2]
            phantom = phantom.slices([slices // 2])
            phantom = phantom.interpolate(sim_size[0], sim_size[1], 1)
    
    # Sequence loading: mr0, load from file, or write and load seq object
    if isinstance(seq, mr0.Sequence):
        pass
    elif isinstance(seq, str):
        seq = mr0.Sequence.import_file(seq)
    else:
        seq.write("tmp.seq")
        seq = mr0.Sequence.import_file("tmp.seq")

    # Run the simulation
    data = phantom.build()
    graph = mr0.compute_graph(seq, data, 2000, 1e-5)

    if torch.cuda.is_available():
        signal = mr0.execute_graph(
            graph, seq.cuda(), data.cuda(), accuracy, accuracy,
            print_progress=False
        ).cpu()
    else:
        signal = mr0.execute_graph(
            graph, seq, data, accuracy, accuracy,
            print_progress=False
        )

    # Add noise if requested
    if noise_level:
        signal += noise_level * torch.randn(*signal.shape, dtype=signal.dtype)

    return signal, seq.get_kspace()


def simulate_2d(seq, sim_size=None, noise_level=0, dB0=0, B0_scale=1, B0_polynomial=None):
    print("DEPRECATED: util.simulate_2d will be removed in the future.")
    print("Use util.simulate() instead (together with util.load_phantom() if necessary")

    # Copied and modified from https://github.com/pulseq/MR-Physics-with-Pulseq
    import urllib
    import MRzeroCore as mr0
    # Download .mat file for phantom if necessary
    sim_url = 'https://github.com/mzaiss/MRTwin_pulseq/raw/mr0-core/data/numerical_brain_cropped.mat'
    sim_filename = os.path.basename(sim_url)
    if not os.path.exists(sim_filename):
        print(f'Downloading {sim_url}...')
        urllib.request.urlretrieve(sim_url, sim_filename)
    
    # If seq is not a str, assume it is a sequence object and write to a temporary sequence file
    if not isinstance(seq, str):
        seq.write('tmp.seq')
        seq_filename = 'tmp.seq'
        adc_samples = int(seq.adc_library.data[1][0])
    else:
        seq_filename = seq
        tmp_seq = pp.Sequence()
        tmp_seq.read(seq_filename)
        adc_samples = int(tmp_seq.adc_library.data[1][0])
    
    # Create phantom from .mat file
    obj_p = mr0.VoxelGridPhantom.load_mat(sim_filename)
    if sim_size is not None:
        obj_p = obj_p.interpolate(sim_size[0], sim_size[1], 1)

    # Manipulate loaded data
    obj_p.B0 *= B0_scale
    obj_p.B0 += dB0
    
    if B0_polynomial is not None:
        x,y = torch.meshgrid(torch.linspace(-1,1,obj_p.PD.shape[0]),torch.linspace(-1,1,obj_p.PD.shape[1]))
        
        obj_p.B0 = B0_polynomial[0]
        if len(B0_polynomial) > 1:
            obj_p.B0 += x * B0_polynomial[1]
        if len(B0_polynomial) > 2:
            obj_p.B0 += y * B0_polynomial[2]
        if len(B0_polynomial) > 3:
            obj_p.B0 += x*x * B0_polynomial[3]
        if len(B0_polynomial) > 4:
            obj_p.B0 += y*y * B0_polynomial[4]
        if len(B0_polynomial) > 5:
            obj_p.B0 += x*y * B0_polynomial[5]
        
        obj_p.B0 = obj_p.B0[:,:,None]
    
    obj_p.D *= 0
    
    # Convert Phantom into simulation data
    obj_p = obj_p.build()

    # MR zero simulation
    seq0 = mr0.Sequence.import_file(seq_filename)
    
    # Remove temporary sequence file
    if not isinstance(seq, str):
        os.unlink('tmp.seq')
    
    # Simulate the sequence
    graph = mr0.compute_graph(seq0, obj_p, 200, 1e-5)
    signal = mr0.execute_graph(graph, seq0, obj_p)

    # Add noise to the simulated data
    if noise_level > 0:
        signal += noise_level * torch.randn(*signal.shape, dtype=signal.dtype)
    
    return signal

