import os

import ismrmrd
import numpy as np
import pypulseq as pp
import torch
from tqdm import tqdm

PULSEQ_MRD_FLAGS = {
    "NAV": ismrmrd.ACQ_IS_PHASECORR_DATA,
    "REV": ismrmrd.ACQ_IS_REVERSE,
    "REF": ismrmrd.ACQ_IS_PARALLEL_CALIBRATION,
    "IMA": ismrmrd.ACQ_IS_PARALLEL_CALIBRATION_AND_IMAGING,
    "NOISE": ismrmrd.ACQ_IS_NOISE_MEASUREMENT,
    "PMC": ismrmrd.ACQ_IS_RTFEEDBACK_DATA,
}


def sig_to_mrd(
    mrd_path: str,
    mr0_signal: torch.Tensor,
    seq: pp.Sequence,
    verbose: int = 0,
) -> None:
    """
    Convert and write MR0 simulated signal, Pulseq k-space trajectory and Pulseq metadata to an ISMRMRD dataset file.
    The output is suitable for downstream reconstruction workflows.

    Parameters
    ----------
    mrd_path : str  
        File path where the ISMRMRD file will be created/saved.
    mr0_signal : torch.Tensor, shape (N, C)
        The simulated signal array from MR0, where N is the number samples
        and C is the number of receiver channels.
    seq : pp.Sequence
        Pulseq sequence object.
    verbose : int, optional
        Logging verbosity level from 0 (silent) to 5 (most detailed). Default is 0.
    """
    #  Convert signal tensor to NumPy for ISMRMRD compatibility
    mr0_signal_numpy = mr0_signal.detach().cpu().numpy()

    # Overwrite the existing MRD file if it exists
    if os.path.exists(mrd_path):
        os.remove(mrd_path)
    dataset = ismrmrd.Dataset(mrd_path, create_if_needed=True, dataset_name="dataset")
    if verbose > 0:
        print(f"Create ISMRMRD dataset at '{mrd_path}'")

    # Write ISMRMRD header based on Pulseq metadata
    mrd_head = _seq_write_mrd_head(seq, verbose=verbose)
    mrd_head.acquisitionSystemInformation.receiverChannels = mr0_signal_numpy.shape[1]
    if verbose > 4:
        print(
            f"Set mrd header receiver channels to {mrd_head.acquisitionSystemInformation.receiverChannels}"
        )

    # Save the header to the dataset
    dataset.write_xml_header(ismrmrd.xsd.ToXML(mrd_head))

    # Calcuate the k-space trajectory and sample times
    kadc, _, _, _, _ = seq.calculate_kspace()

    # A single ADC object has a maximum sample limit that is defined in the sequence.
    # Usually longer readouts are written into multiple ADC objects with increasing 'set' encodes.
    # If no MaxAdcSegmentLength is defined, the default value is 2^64 so there will be no splitting into ADC sets.
    max_segment_samples = seq.definitions.get("MaxAdcSegmentLength", 2**64)
    max_segment_samples = int(max_segment_samples)  # pulseq uses floats

    # Get the ADC IDs from the sequence
    adc_ids = [int(items[5]) for _, items in seq.block_events.items() if items[5] != 0]

    # Extract ADC labels from the sequence
    labels = seq.evaluate_labels(evolution="adc")

    # Prepare process bar
    log = tqdm(
        total=len(adc_ids),
        desc="Write readouts to mrd file",
        unit="acq",
        disable=(verbose == 0),
    )

    # Write each ADC object to an mrd acquisition
    current_sample = 0
    current_traj_sample = 0
    scan_counter = 1
    for n_acq, adc_id in enumerate(adc_ids):
        # Read the number of samples and dwell time for the current ADC
        adc_obj = seq.adc_library.data.get(adc_id, None)
        if adc_obj is None:
            raise ValueError(f"ADC object with ID {adc_id} not found in the sequence.")

        num_samples_adc = int(adc_obj[0])
        dwell = float(adc_obj[1])

        # Extract signal and k-space trajectory for the current ADC. If it is a noise sample than no signal is available
        k_acq = kadc[:, current_traj_sample : current_traj_sample + num_samples_adc]
        current_traj_sample += num_samples_adc
        if "NOISE" in labels and labels["NOISE"][n_acq] > 0:
            s_acq = np.zeros(
                (num_samples_adc, mr0_signal_numpy.shape[1]), dtype=mr0_signal_numpy.dtype
            )
        else:
            s_acq = mr0_signal_numpy[current_sample : current_sample + num_samples_adc, :]
            current_sample += num_samples_adc
        

        # Split the k-space trajectory and data into multiple segments depending on the maximum segment length
        k_acq, s_acq = _split_adc_sets(k_acq, s_acq, max_segment_samples)
        num_set, num_samples_set, num_cha = s_acq.shape
        num_dim, _, _ = k_acq.shape

        if num_set > 1 and verbose > 3:
            print(f"Split ADC into {num_set} sets with {num_samples_set} samples each.")

        # Save each set as different acquisition
        for n_set in range(num_set):
            # Save acquisition header
            acq_labels = {key: int(value[n_acq]) for key, value in labels.items()}

            if (
                acq_labels.get("SET", 0) != 0 or acq_labels.get("SET", 0) != n_set
            ) and verbose > 1:
                print(
                    f"Pulseq SET label ({acq_labels.get('SET', 0)}) does not match segment index ({n_set}). Overwrite with segment index."
                )
            acq_labels["SET"] = n_set

            # Set the acquisition header
            header = ismrmrd.AcquisitionHeader()
            header.scan_counter = scan_counter
            header.number_of_samples = num_samples_set
            header.center_sample = num_samples_set // 2
            header.trajectory_dimensions = num_dim
            header.available_channels = num_cha
            header.active_channels = num_cha
            header.sample_time_us = dwell * 1e6
            # We don't know the orientation but we set it to something reasonable
            header.read_dir[0] = 1
            header.phase_dir[1] = 1
            header.slice_dir[2] = 1
            header = _labels_to_acq_head(header, acq_labels)

            # Write header to acquisition
            acquisition = ismrmrd.Acquisition()
            acquisition.setHead(header)

            # Write traj and data to acquisition
            acquisition.traj[:] = k_acq[:, n_set, :].T
            acquisition.data[:] = s_acq[n_set, ...].T

            dataset.append_acquisition(acquisition)

            if verbose > 4:
                print(
                    f"Acquisition {scan_counter} written:\n"
                    f"Header: {header}\n"
                    f"Data shape: {acquisition.data[:].shape}\n"
                    f"Traj shape: {acquisition.traj[:].shape}"
                )

            scan_counter += 1

        log.update(1)

    # Close the dataset
    dataset.close()


def _split_adc_sets(ktraj, data, max_segment_samples):
    """Split the k-space trajectory and signal data into multiple segments
    if the number of samples exceeds the maximum segment length.

    Parameters
    ----------
    ktraj : np.ndarray, shape (D, N)
        The k-space trajectory data with D dimensions (e.g., x, y, z) and N samples.
    data : np.ndarray, shape (N, C)
        The signal data with N samples and C channels.
    max_segment_samples : int
        The maximum number of samples allowed in a single ADC segment.

    Returns
    -------
    ktraj : np.ndarray, shape (D, S, M)
        The k-space trajectory data split into S segments, each with M samples.
    data : np.ndarray, shape (S, M, C)
        The signal data split into S segments with M samples and C channels.
    """
    num_dim, num_samples_ktraj = ktraj.shape
    num_samples_data, num_cha = data.shape

    if num_samples_ktraj != num_samples_data:
        raise ValueError(
            "Number of samples in k-space trajectory",
            f"({num_samples_ktraj}) and ",
            f"data ({num_samples_data}) do not match.",
        )
    # Split the data into multiple segments if the number
    # of samples exceeds the maximum segment length
    if num_samples_ktraj > max_segment_samples:
        ktraj = ktraj.reshape(num_dim, -1, max_segment_samples)
        data = data.reshape(-1, max_segment_samples, num_cha)
    else:
        ktraj = ktraj[:, None, :]
        data = data[None, ...]

    return ktraj, data


def _labels_to_acq_head(
    acq_head: ismrmrd.AcquisitionHeader, seq_labels: dict
) -> ismrmrd.AcquisitionHeader:
    """Writes the pulseq ADC labels to the ISMRMRD acquisition header."""
    # Set encodes
    acq_head.idx.kspace_encode_step_1 = seq_labels.get("LIN", 0)
    acq_head.idx.kspace_encode_step_2 = seq_labels.get("PAR", 0)
    acq_head.idx.average = seq_labels.get("AVG", 0)
    acq_head.idx.slice = seq_labels.get("SLC", 0)
    acq_head.idx.contrast = seq_labels.get("ECO", 0)
    acq_head.idx.phase = seq_labels.get("PHS", 0)
    acq_head.idx.repetition = seq_labels.get("REP", 0)
    acq_head.idx.set = seq_labels.get("SET", 0)
    acq_head.idx.segment = seq_labels.get("SEG", 0)

    # Set flags
    for key, item in PULSEQ_MRD_FLAGS.items():
        label_value = seq_labels.get(key, 0)
        if label_value > 0:
            acq_head.set_flag(item)

    return acq_head


def _seq_write_mrd_head(
    seq: pp.Sequence,
    verbose: int = 0,
) -> ismrmrd.xsd.ismrmrdHeader:
    """Writes the pulseq sequence definitions and labels to the ISMRMRD header."""
    
    def m_to_mm(x_in_m):
        return x_in_m * 1e3
    
    def s_to_ms(x_in_s):
        return x_in_s * 1e3

    mrd_seq_params = ismrmrd.xsd.sequenceParametersType()

    mrd_seq_params.TE = [s_to_ms(val) for val in _to_list(seq.definitions.get("TE", []))]
    mrd_seq_params.TR = [s_to_ms(val) for val in _to_list(seq.definitions.get("TR", []))]
    mrd_seq_params.TI =[s_to_ms(val) for val in _to_list(seq.definitions.get("TI", []))]
    if verbose > 4:
        print(
            f"Wrote TE/TR/TI to mrd header with {mrd_seq_params.TE}/{mrd_seq_params.TR}/{mrd_seq_params.TI}"
        )

    seq_res = seq.definitions.get("RES", [None, None, None])
    seq_fov = [m_to_mm(val) for val in seq.definitions.get("FOV", [None, None, None])]
    seq_labels = seq.evaluate_labels(evolution="adc")

    mrd_enc_params = ismrmrd.xsd.encodingType()

    mrd_enc_params.encodedSpace = ismrmrd.xsd.encodingSpaceType(
        matrixSize=ismrmrd.xsd.matrixSizeType(
            x=seq_res[0],
            y=seq_res[1],
            z=seq_res[2],
        ),
        fieldOfView_mm=ismrmrd.xsd.fieldOfViewMm(
            x=seq_fov[0],
            y=seq_fov[1],
            z=seq_fov[2],
        ),
    )

    mrd_enc_params.reconSpace = ismrmrd.xsd.encodingSpaceType(
        matrixSize=ismrmrd.xsd.matrixSizeType(
            x=seq_res[0],
            y=seq_res[1],
            z=seq_res[2],
        ),
        fieldOfView_mm=ismrmrd.xsd.fieldOfViewMm(
            x=seq_fov[0],
            y=seq_fov[1],
            z=seq_fov[2],
        ),
    )

    if verbose > 4:
        print(
            f"Wrote encode/recon FOV to mrd header with {mrd_enc_params.encodedSpace.fieldOfView_mm}/{mrd_enc_params.reconSpace.fieldOfView_mm}"
        )
        print(
            f"Wrote encode/recon RES to mrd header with {mrd_enc_params.encodedSpace.fieldOfView_mm}/{mrd_enc_params.reconSpace.fieldOfView_mm}"
        )

    mrd_enc_params.encodingLimits = _labels_to_encodinglimits(seq_labels)
    
    # The Lamour frequency is a required field in the ISMRMRD header
    exp = ismrmrd.xsd.experimentalConditionsType()
    exp.H1resonanceFrequency_Hz = int(seq.system.B0 * 42.5764 * 1e6)

    mrd_head = ismrmrd.xsd.ismrmrdHeader(
        experimentalConditions=exp,
        measurementInformation=ismrmrd.xsd.measurementInformationType(),
        acquisitionSystemInformation=ismrmrd.xsd.acquisitionSystemInformationType(),
        sequenceParameters=mrd_seq_params,
        encoding=[mrd_enc_params],
    )
    if verbose > 4:
        print(
            f"Wrote encoding limits to mrd header with {mrd_enc_params.encodingLimits}"
        )

    return mrd_head


def _labels_to_encodinglimits(
    seq_labels: dict,
):
    """Writes the pulseq ADC labels to the ISMRMRD encoding limits."""

    encoding_limits = ismrmrd.xsd.encodingLimitsType(
        kspace_encoding_step_1=_label_limits(seq_labels, "LIN"),
        kspace_encoding_step_2=_label_limits(seq_labels, "PAR"),
        average=_label_limits(seq_labels, "AVG"),
        slice=_label_limits(seq_labels, "SLC"),
        contrast=_label_limits(seq_labels, "ECO"),
        phase=_label_limits(seq_labels, "PHS"),
        repetition=_label_limits(seq_labels, "REP"),
        set=_label_limits(seq_labels, "SET"),
        segment=_label_limits(seq_labels, "SEG"),
    )

    return encoding_limits


def _label_limits(
    seq_labels: dict,
    label: str,
) -> ismrmrd.xsd.limitType:
    """Writes a pulseq ADC label to a ISMRMRD encoding limit type."""
    label_evol = seq_labels.get(label, [0])

    type_limits = ismrmrd.xsd.limitType(
        minimum=int(np.min(label_evol)),
        maximum=int(np.max(label_evol)),
        center=int((np.max(label_evol) - np.min(label_evol)) // 2 + 1 if label in ('LIN', 'PAR') else 0),
    )

    return type_limits


def _to_list(v):
    return v if isinstance(v, list) else [v]

