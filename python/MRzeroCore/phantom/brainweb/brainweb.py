import os
import requests
import gzip
import numpy as np
import matplotlib.pyplot as plt
from enum import IntEnum
from dataclasses import dataclass


# Files downloaded from brainweb are cached in the directory of this file
CACHE_PATH = os.path.dirname(os.path.realpath(__file__))

# All 20 subject numbers provided by BrainWeb
SUBJECTS = [
    4, 5, 6, 18, 20, 38, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54
]


class TissueType(IntEnum):
    GRAY_MATTER = 0
    WHITE_MATTER = 1
    CSF = 2  # includes vessels
    FAT = 4  # includes skin, muscles, dura, connective
    # Missing: Skull / Marrow


@dataclass
class Tissue:
    # The names under which brainweb provides the files
    download_aliases: list[str]
    # Normalized value: water has a density of 1
    pd: float
    # Exponential relaxation constant [1/s]
    t1: float
    # Exponential relaxation constant [1/s]
    t2: float
    # Exponential relaxation constant [1/s]
    t2dash: float
    # Isometric diffusion coefficient [10^-3 mm^2/s]
    d: float


# The values used here and their sources can be found in phantom_values.txt
TISSUE_DATA = {
    TissueType.GRAY_MATTER: Tissue(["gry"], 0.8, 1.55, 0.09, 0.322, 0.83),
    TissueType.WHITE_MATTER: Tissue(["wht"], 0.7, 0.83, 0.07, 0.183, 0.65),
    TissueType.CSF: Tissue(["csf", "ves"], 1.0, 4.16, 1.65, 0.0591, 3.19),
    TissueType.FAT: Tissue(["fat", "mus", "m-s", "dura", "fat2"],
                           1.0, 0.374, 0.125, 0.0117, 0.1),
}


def download(subject: int, tissue_type: TissueType) -> np.ndarray:
    """Load all maps for a tissue from BrainWeb or from the cache.

    Returns the sum of all maps corresponding to the tissue (we don't split
    the tissue in as many parts as BrainWeb does). We still cache them
    separately as we might to want the individual segments in the future.
    """
    if subject not in SUBJECTS:
        raise ValueError(
            "Requested subject not in list of valid subjects",
            subject, SUBJECTS
        )

    # This is the returned data: The sum of all segemets for the tissue
    data = np.zeros((362, 434, 362))

    for alias in TISSUE_DATA[tissue_type].download_aliases:
        download_alias = f"subject{subject:02d}_{alias}"
        file_name = download_alias + ".i8.gz"  # 8 bit signed int, gnuzip
        file_dir = os.path.join(CACHE_PATH, f"subject{subject:02d}")
        file_path = os.path.join(file_dir, file_name)

        try:
            os.mkdir(file_dir)
        except FileExistsError:
            pass

        # Download and cache file if it doesn't exist yet
        if not os.path.exists(file_path):
            print(f"Downloading '{file_name}' ...")
            response = requests.post(
                "https://brainweb.bic.mni.mcgill.ca/cgi/brainweb1",
                data={
                    "do_download_alias": download_alias,
                    "format_value": "raw_byte",
                    "zip_value": "gnuzip"
                }
            )
            with open(file_path, "wb") as f:
                f.write(response.content)

        # Load the raw BrainWeb data and add it to the return array
        with gzip.open(file_path) as f:
            # BrainWeb says this data is unsigned, which is a lie
            tmp = np.frombuffer(f.read(), np.uint8) + 128
            # Vessel bugfix: most of background is 1 instead of zero
            if alias == "ves":
                tmp[tmp == 1] = 0
            data += tmp.reshape(362, 434, 362).swapaxes(0, 2).astype(float)

    return data / 255.0


def downsample(factor: int, array: np.ndarray):
    """Downsample the input array [1, 2, ...]x with "area" interpolation.

    Downsample functions found in libraries like pytorch are slow
    and often just do nearest neigbour interpolation."""
    shape = (np.array(array.shape) // factor) * factor
    # crop array to multiple of factor
    array = array[:shape[0], :shape[1], :shape[2]]

    # This could be faster with np.add.reduceat or similar built-in functions
    tmp = np.zeros(shape // factor)
    for x in range(factor):
        for y in range(factor):
            for z in range(factor):
                tmp += array[x::factor, y::factor, z::factor]

    return tmp / factor**3


class Phantom:
    def __init__(self, subject: int, pd: np.ndarray, t1: np.ndarray,
                 t2: np.ndarray, t2dash: np.ndarray, d: np.ndarray):
        self.subject = subject
        self.pd = pd
        self.t1 = t1
        self.t2 = t2
        self.t2dash = t2dash
        self.d = d

    @classmethod
    def load(cls, subject: int, scale: int = 1) -> Tissue:
        """Load a phantom from BrainWeb data.

        scale: downsampling factor [1, 2, ...]x"""
        gm = TISSUE_DATA[TissueType.GRAY_MATTER]
        gm_map = downsample(scale, download(subject, TissueType.GRAY_MATTER))
        wm = TISSUE_DATA[TissueType.WHITE_MATTER]
        wm_map = downsample(scale, download(subject, TissueType.WHITE_MATTER))
        csf = TISSUE_DATA[TissueType.CSF]
        csf_map = downsample(scale, download(subject, TissueType.CSF))
        fat = TISSUE_DATA[TissueType.FAT]
        fat_map = downsample(scale, download(subject, TissueType.FAT))

        return cls(
            subject,
            pd=(gm.pd * gm_map + wm.pd * wm_map +
                csf.pd * csf_map + fat.pd * fat_map),
            t1=(gm.t1 * gm_map + wm.t1 * wm_map +
                csf.t1 * csf_map + fat.t1 * fat_map),
            t2=(gm.t2 * gm_map + wm.t2 * wm_map +
                csf.t2 * csf_map + fat.t2 * fat_map),
            t2dash=(gm.t2dash * gm_map + wm.t2dash * wm_map +
                    csf.t2dash * csf_map + fat.t2dash * fat_map),
            d=(gm.d * gm_map + wm.d * wm_map +
               csf.d * csf_map + fat.d * fat_map),
        )

    def plot(self):
        """Plot the center slice of all maps of this phantom"""
        slice = self.pd.shape[2] // 2
        plots = [
            ("PD", self.pd), ("T_1", self.t1), ("T_2", self.t2),
            ("T_2'", self.t2dash), ("D", self.d)
        ]
        for (name, map) in plots:
            plt.figure(figsize=(7, 5))
            plt.title(f"${name}$")
            plt.imshow(map[:, :, slice].T, origin="lower")
            plt.colorbar()
            plt.show()

    def save(self):
        """Save the maps of this phantom as numpy .npz archive"""
        path = os.path.join(CACHE_PATH, f"subject{self.subject}.npz")
        np.savez_compressed(
            path,
            PD_map=self.pd,
            T1_map=self.t1,
            T2_map=self.t2,
            T2dash_map=self.t2dash,
            D_map=self.d
        )


def generate_phantom(subject: int, scale: int = 1):
    phantom = Phantom.load(subject, scale)
    phantom.save()


if __name__ == "__main__":
    print("Generating 20 BrainWeb phantoms at full resolution...")
    for i, subject in enumerate(SUBJECTS):
        print(f"[{i+1: 2d} / 20]: Subject{subject:02d}")
        generate_phantom(subject)
