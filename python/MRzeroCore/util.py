import os
import time
from typing import Literal, Union, Optional
import torch
import numpy as np

import matplotlib.pyplot as plt
import pypulseq as pp


def get_signal_from_real_system(path: str, NRep: int, NCol: int):
    """Wait for a TWIX file and return its data.

    This function assumes 20 recieve coils and a readout with equal number of
    ADC samples for all repetitions.

    Parameters
    ----------
    path : str
        Path to TWIX file
    NRep : int
        Number of repetitions of the measured sequence
    NCol : int
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

            ncoils = 20
            time.sleep(0.2)
            raw = np.loadtxt(path)

            heuristic_shift = 4
            expected_size = NRep * ncoils * (NCol + heuristic_shift) * 2
            print(f"raw size: {raw.size}, expected size: {expected_size}")

            if raw.size != expected_size:
                print("get_signal_from_real_system: SERIOUS ERROR, "
                      "TWIX dimensions corrupt, returning zero array..")
                raw = np.zeros((NRep, ncoils, NCol + heuristic_shift, 2))
                raw = raw[:, :, :NCol, 0] + 1j * raw[:, :, :NCol, 1]
            else:
                raw = raw.reshape([NRep, ncoils, NCol + heuristic_shift, 2])
                raw = raw[:, :, :NCol, 0] + 1j * raw[:, :, :NCol, 1]

            # raw = raw.transpose([1,2,0]) #ncoils,NRep,NCol
            raw = raw.transpose([0, 2, 1])  # NRep,NCol,NCoils
            raw = raw.reshape([NRep * NCol, ncoils])
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

        sp11.legend(loc='right', bbox_to_anchor=(1.12, 0.5), fontsize='xx-small')


# This plot function is a modified version from the one provided by
# pypulseq 1.2.0post1, all changes are marked
# NOTE: the parameters have changed in the 1.4 version, maybe we should adapt them
def pulseq_plot(seq: pp.Sequence,
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

    plt.imshow(data.T, *args, origin="lower", **kwargs)


PHANTOM_URL = "https://github.com/MRsources/MRzero-Core/raw/main/documentation/playground_mr0/numerical_brain_cropped.mat"


def load_phantom(size_x: Optional[int] = None, size_y: Optional[int] = None,phantom_url=PHANTOM_URL,dB0=(0,1),dB1=(0,1), B0_polynomial=None):
    """Download an default phantom from https://github.com/MRsources/MRzero-Core


    Parameters
    ----------
    size_x, size_y : int | None
        If a size is passed, the phantom is scaled to it.
    plot : bool
        If True, the phantom is plotted with matplotlib before returning it.
    
    Returns
    -------
    mr0.VoxelGridPhantom:
        Loaded phantom. Call .build() and pass the result to the simulation.
    """
    from urllib.request import urlretrieve
    from MRzeroCore import VoxelGridPhantom
    import os
    phantom_name = phantom_url.split("/")[-1]

    if not os.path.exists(phantom_name):
        urlretrieve(phantom_url, phantom_name)
        print("Downloaded simulation data")

    if phantom_name.endswith(".mat"):
        phantom = VoxelGridPhantom.load_mat(phantom_name)
    elif phantom_name.endswith(".npz"):
        phantom = VoxelGridPhantom.brainweb(phantom_name)
    else:
        raise ValueError(f"Can't load {phantom_name}: unknown file extension")

    if isinstance(size_x, int) and isinstance(size_y, int):
        phantom = phantom.interpolate(size_x, size_y, 1)
    elif not (size_x is None and size_y is None):
        raise TypeError(
            "Arguments 'size_x' and 'size_y' are expected to be both given "
            "(int) or both omitted (None), but got types "
            f"{type(size_x)} and {type(size_y)}"
        )

    # Manipulate loaded data
    phantom.B0 += dB0[0]
    phantom.B0 *= dB0[1]
    phantom.B1 += dB1[0]
    phantom.B1 *= dB1[1]

    if B0_polynomial is not None:
        x,y = torch.meshgrid(torch.linspace(-1,1,phantom.PD.shape[0]),torch.linspace(-1,1,phantom.PD.shape[1]))

        nB0= B0_polynomial[0]
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


def simulate(seq, phantom,accuracy=1e-3):
    """Simulate the sequence with the phantom using PDG and default settings.
    
    Arguments
    ---------
    seq: mr0.Sequence
        The MRI sequence in MR-zero's format
    phantom: VoxelGridPhantom | CustomVoxelPhantom
        The simulation data in its original form (without calling .build() on it)

    Returns
    -------
    torch.Tensor
        The ADC signal
    """
    import MRzeroCore as mr0
    data = phantom.build()
    graph = mr0.compute_graph(seq, data, 2000, 1e-5)
    if torch.cuda.is_available():
        signal = mr0.execute_graph(graph, seq.cuda(), data.cuda(), accuracy, accuracy, print_progress=False)
        signal = signal.cpu()
    else:
        signal = mr0.execute_graph(graph, seq, data, accuracy, accuracy, print_progress=False)
    return signal


def simulate_2d(seq, sim_size=None, noise_level=0,accuracy=1e-3):
    # Copied and modified from https://github.com/pulseq/MR-Physics-with-Pulseq
    import urllib
    import MRzeroCore as mr0
    
    
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
    
    # load standard phantom    
    if sim_size is not None:
        obj_p = load_phantom(size_x = sim_size[0], size_y = sim_size[1])
    else:
        obj_p = load_phantom()

    # Convert Phantom into simulation data
    obj_p = obj_p.build()

    # MR zero simulation
    seq0 = mr0.Sequence.import_file(seq_filename)
    
    # Remove temporary sequence file
    if not isinstance(seq, str):
        os.unlink('tmp.seq')
    
    # Simulate the sequence
    graph = mr0.compute_graph(seq0, obj_p, 2000, 1e-5)
    
    if torch.cuda.is_available():
        signal = mr0.execute_graph(graph, seq.cuda(), data.cuda(), accuracy, accuracy, print_progress=False)
        signal = signal.cpu()
    else:
        signal = mr0.execute_graph(graph, seq, data, accuracy, accuracy, print_progress=False)

    # Add noise to the simulated data
    if noise_level > 0:
        signal += noise_level * torch.randn(*signal.shape, dtype=signal.dtype)
    
    return signal
