import time
import os
import io
import torch
import numpy as np
import base64
from ..sequence import Sequence

print("""----------------------------------------------------------
                         WARNING!
   Included helpers.py, which is not yet appropriately
 documented for MRzero-Core, neither are its dependencies
        included in the MRzero-Core requirements,
   and it is not exposed yet to the global mr0 import!
----------------------------------------------------------""")


# TODO: This is specific to GRE-like sequences, make it more general!
def get_signal_from_real_system(path, seq, NRep: float | None = None):
    if NRep is None:
        NRep = len(seq)
    NCol = torch.count_nonzero(seq[2].adc_usage).item()

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
            print("raw size: {} ".format(raw.size) + "expected size: {} ".format(
                "raw size: {} ".format(NRep*ncoils*(NCol+heuristic_shift)*2)))

            if raw.size != NRep*ncoils*(NCol+heuristic_shift)*2:
                print(
                    "get_signal_from_real_system: SERIOUS ERROR, TWIX dimensions corrupt, returning zero array..")
                raw = np.zeros((NRep, ncoils, NCol+heuristic_shift, 2))
                raw = raw[:, :, :NCol, 0] + 1j*raw[:, :, :NCol, 1]
            else:
                raw = raw.reshape([NRep, ncoils, NCol+heuristic_shift, 2])
                raw = raw[:, :, :NCol, 0] + 1j*raw[:, :, :NCol, 1]

            # raw = raw.transpose([1,2,0]) #ncoils,NRep,NCol
            raw = raw.transpose([0, 2, 1])  # NRep,NCol,NCoils
            raw = raw.reshape([NRep*NCol, ncoils])
            raw = np.copy(raw)
            done_flag = True

    return torch.tensor(raw, dtype=torch.complex64)


def write_data_to_seq_file(seq: Sequence, file_name: str):
    """Write all sequence data needed for reconstruction into a .seq file.

    The data is compressed, base64 encoded and inserted as a comment into the
    pulseq .seq file, which means it is ignored by all interpreters and only
    slightly increases the file size.

    Parameters
    ----------
    seq : Sequence
        Should be the sequence that was used to produce the .seq file
    file_name : str
        The file name to append the data to, it is not checked if this
        actually is a pulseq .seq file.
    """
    kspace = seq.get_kspace()
    adc_usage = torch.cat([rep.adc_usage[rep.adc_usage > 0] for rep in seq])

    # Transpose for more efficient compression (contiguous components)
    kspace_enc = np.ascontiguousarray(kspace.T.cpu().numpy())
    # Delta encoding (works very well for cartesian trajectories)
    kspace_enc[:, 1:] -= kspace_enc[:, :-1]
    # Reduce precision, don't need 32bit for a kspace
    kspace_enc = kspace_enc.astype(np.float16)

    # Compressing adc_usage
    assert -128 <= adc_usage.min() <= 127, "8 bit are not enough"
    adc_usage_enc = adc_usage.cpu().numpy().astype(np.int8)

    # Compress and encode with base64 to write as legal ASCII text
    buffer = io.BytesIO()
    np.savez_compressed(buffer, kspace=kspace_enc, adc_usage=adc_usage_enc)
    encoded = base64.b64encode(buffer.getvalue()).decode('ascii')

    # The pulseq Siemens interpreter has a bug in the comment code leading to
    # errors if comments are longer than MAX_LINE_WIDTH = 256. We split the
    # data into chunks of 250 bytes to be on the safe side.
    with open(file_name, "a") as file:
        for i in range(0, len(encoded), 250):
            file.write(f"\n# {encoded[i:i+250]}")
        file.write("\n")


def extract_data_from_seq_file(
    file_name: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extracts kspace and adc_usage written with ``write_data_to_seq_file``.

    Parameters
    ----------
    file_name : str
        The name of the file the kspace was previously written to.

    Returns
    -------
    The original kspace and the adc_usage. There might be a  loss of precision
    because the kspace is written as 16 bit (half precision) floats and the
    usage as 8 bit integer (-128 to 127), this could be changed.
    """
    try:
        with open(file_name, "r") as file:
            # Find the last n lines that start with a '#'
            lines = file.readlines()

            if lines[-1][-1:] != '\n':
                lines[-1] = lines[-1] + '\n'

            n = len(lines)
            while n > 0 and lines[n-1][0] == '#':
                n -= 1
            if n == len(lines):
                raise ValueError(
                    "No data comment found at the end of the file")

            # Join the parts of the comment while removing "# " and "\n"
            encoded = "".join(line[2:-1] for line in lines[n:])
            # print(encoded)
            decoded = base64.b64decode(encoded, validate=True)

            data = np.load(io.BytesIO(decoded))
            kspace = np.cumsum(data["kspace"].astype(np.float32), 1).T
            adc_usage = data["adc_usage"].astype(np.int32)

            return torch.tensor(kspace), torch.tensor(adc_usage)
    except Exception as e:
        raise ValueError("Could not extract data from .seq") from e


def load_measurement(
    seq_file: str,
    seq_dat_file: str,
    wait_for_dat: bool = False,
    twix: bool = False
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Loads the seq data from a .seq file and the signal from a .seq.dat file.

    This function waits for the .seq.dat file if it doesn't exist yet and
    ``wait_for_dat = True``.

    Parameters
    ----------
    seq_file : str
        Name of the (path to the) .seq file
    seq_dat_file : str
        Name of the (path to the) .seq.dat file
    wait_for_dat : bool
        Specifies if this function should wait for the .seq.dat file or throw
        an error if it doesn't exist

    Returns
    -------
    (Samples, 4) tensor containing the kspace stored in the .seq file and a
    (Samples, Coils) tensor containing the signal (for all coils)
    """

    kspace, adc_usage = extract_data_from_seq_file(seq_file)

    if wait_for_dat:
        print("Waiting for TWIX file...", end="")
        while not os.path.isfile(seq_dat_file):
            time.sleep(0.2)
        print(" arrived!")

    if twix:
        # Clone https://github.com/pehses/twixtools
        import twixtools
        twix = twixtools.read_twix(seq_dat_file)
        image_mdbs = [mdb for mdb in twix[-1]['mdb'] if mdb.is_image_scan()]

        n_line = 1 + max([mdb.cLin for mdb in image_mdbs])

        # assume that all data were acquired with same number of channels & columns:
        n_channel, n_column = image_mdbs[0].data.shape

        kspace_data = np.zeros(
            [n_line, n_channel, n_column], dtype=np.complex64)
        for mdb in image_mdbs:
            kspace_data[mdb.cLin] = mdb.data
        # For 32 Coils!
        signal = kspace_data.transpose(0, 2, 1).reshape(-1, 32)
    else:
        data = np.loadtxt(seq_dat_file)

        data = data[:, 0] + 1j*data[:, 1]

        # .dat files contain additional samples we need to remove. This is probably
        # a bug in the TWIX to text file converter.
        #
        # These additional samples might be at the and of every shot or ADC block,
        # in which case a possible solution would be to store the subdivision in
        # the .seq file.
        #
        # Or maybe we can just fix it when exporting .seq files :D
        #
        # For now, we detect the number of samples in a single ADC readout and
        # assume 20 coils. Might not work for irregular readouts.

        # We assume that there are no exact zeros in the actual signal
        adc_length = np.where(np.abs(data) == 0)[0][0]
        data = data.reshape([-1, 20, adc_length + 4])

        # Remove additional samples and reshape into samples x coils
        signal = data.transpose([0, 2, 1])[:, :adc_length, :].reshape([-1, 20])

    if kspace.shape[0] != signal.shape[0]:
        print(
            f"WARNING: the kspace contains {kspace.shape[0]} samples but the "
            f"loaded signal has {signal.shape[0]}. They are either not for the"
            " same measurement, or something went wrong loading the data."
        )

    return kspace, adc_usage, torch.tensor(signal, dtype=torch.complex64)
