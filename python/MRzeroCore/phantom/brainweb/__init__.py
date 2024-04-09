from typing import Literal
import json
import gzip
import requests
import os
import numpy as np


# Load the brainweb data file that contains info about tissues, subjects, ...
brainweb_data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                  "brainweb_data.json")
brainweb_data = json.load(open(brainweb_data_path))


def load_tissue(subject: int, alias: str, cache_dir: str) -> np.ndarray:
    download_alias = f"subject{subject:02d}_{alias}"
    file_name = download_alias + ".i8.gz"  # 8 bit signed int, gnuzip
    file_path = os.path.join(cache_dir, file_name)

    # Download and cache file if it doesn't exist yet
    if not os.path.exists(file_path):
        print(f"Downloading '{download_alias}'", end="", flush=True)
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
        print(" - ", end="")

    # Load the raw BrainWeb data and add it to the return array
    with gzip.open(file_path) as f:
        print(f"Loading {os.path.basename(file_path)}", end="", flush=True)
        # BrainWeb says this data is unsigned, which is a lie
        tmp = np.frombuffer(f.read(), np.uint8) + 128

    # Vessel bugfix: most of background is 1 instead of zero
    if alias == "ves":
        tmp[tmp == 1] = 0
    data = tmp.reshape(362, 434, 362).swapaxes(0, 2).astype(np.float32)

    print(" - done")
    return data / 255.0


def gen_noise(range: float, res: np.ndarray) -> np.ndarray:
    if range == 0:
        return 1
    else:
        freq = 20
        padded_res = (res + freq - 1) // freq * freq
        try:
            from perlin_numpy import generate_perlin_noise_3d
            noise = generate_perlin_noise_3d(padded_res, (freq, freq, freq))
        except:
            print("perlin_numpy@git+https://github.com/pvigier/perlin-numpy")
            print("is not installed, falling back to numpy.random.random()")
            noise = np.random.random(padded_res)
        return 1 + range * noise[:res[0], :res[1], :res[2]]


def downsample(array: np.ndarray, factor: int) -> np.ndarray:
    # crop array to multiple of factor
    shape = (np.array(array.shape) // factor) * factor
    array = array[:shape[0], :shape[1], :shape[2]]

    tmp = np.zeros(shape // factor)
    for x in range(factor):
        for y in range(factor):
            for z in range(factor):
                tmp += array[x::factor, y::factor, z::factor]

    return tmp / factor**3


def generate_brainweb_phantoms(
        output_dir: str,
        config: Literal["3T", "7T-noise", "3T-highres-fat"] = "3T"):
    """Generate BrainWeb phantom maps for the selected configuration.

    Raw tissue segmentation data is provided by the BrainWeb Database:
    http://www.bic.mni.mcgill.ca/brainweb/

    All tissue data etc. are stored in `brainweb_data.json`. To ensure
    consistent configurations and reproducible results, available configs are
    stored in this file as well. They specify which field strength to use,
    which tissues to include, as well as the downsampling and noise levels.

    The emitted files are compressed numpy files, which can be loaded with
    `np.load(file_name)`. They contain the following arrays:

     - `PD_map`: Proton Density [a.u.]
     - `T1_map`: T1 relaxation time [s]
     - `T2_map`: T2 relaxation time [s]
     - `T2dash_map`: T2' relaxation time [s]
     - `D_map`: Isotropic Diffusion coefficient [10^-3 mm² / s]
     - `tissue_XY`: Tissue segmentation for all included tissues

    Parameters
    ----------
    output_dir: str
        The directory where the generated phantoms will be stored to. In
        addition, a `cache` folder will be generated there too, which contains
        all the data downloaded from BrainWeb to avoid repeating the download
        for all configurations or when generating phantoms again.
    config: ["3T", "7T-noise", "3T-highres-fat"]
        The configuration for which the maps are generated.
    """
    config_data = brainweb_data["configs"][config]
    cache_dir = os.path.join(output_dir, "cache")

    try:
        os.makedirs(cache_dir)
    except FileExistsError:
        pass

    # Map resolution:
    res = np.array([362, 434, 362]) // config_data["downsample"]

    def noise() -> np.ndarray:
        return gen_noise(config_data["noise"], res)

    for subject in brainweb_data["subjects"]:
        print(f"Generating '{config}', subject {subject}")
        maps = {
            "FOV": np.array([0.181, 0.217, 0.181]),
            "PD_map": np.zeros(res, dtype=np.float32),
            "T1_map": np.zeros(res, dtype=np.float32),
            "T2_map": np.zeros(res, dtype=np.float32),
            "T2dash_map": np.zeros(res, dtype=np.float32),
            "D_map": np.zeros(res, dtype=np.float32),
        }

        for tissue in config_data["tissues"]:
            tissue_map = sum([
                load_tissue(subject, alias, cache_dir)
                for alias in brainweb_data["download-aliases"][tissue]
            ])
            tissue_map = downsample(tissue_map, config_data["downsample"])
            maps["tissue_" + tissue] = tissue_map

            field_strength = config_data["field-strength"]
            tissue_data = brainweb_data["tissues"][field_strength][tissue]

            # Separate noise maps is slower but uncorrelated.
            # Might be better for training or worse - could be configurable
            print("Adding tissue to phantom", end="", flush=True)
            maps["PD_map"] += tissue_data["PD"] * tissue_map * noise()
            maps["T1_map"] += tissue_data["T1"] * tissue_map * noise()
            maps["T2_map"] += tissue_data["T2"] * tissue_map * noise()
            maps["T2dash_map"] += tissue_data["T2'"] * tissue_map * noise()
            maps["D_map"] += tissue_data["D"] * tissue_map * noise()
            print(" - done")

        file = os.path.join(output_dir, f"subject{subject:02d}_{config}.npz")
        print(f"Saving to '{os.path.basename(file)}'", end="", flush=True)
        np.savez_compressed(file, **maps)
        print(" - done\n")


if __name__ == "__main__":
    print("This is for testing only, use generate_brainweb_phantoms directly!")
    file_dir = os.path.dirname(os.path.realpath(__file__))
    output_dir = os.path.join(file_dir, "output")

    for config in brainweb_data["configs"].keys():
        generate_brainweb_phantoms(output_dir, config)
