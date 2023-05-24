import os
import requests
import gzip
import numpy as np
import matplotlib.pyplot as plt
from enum import IntEnum


# Disable slow perlin noise
NO_NOISE = True

if not NO_NOISE:
    # Needs perlin_numpy: https://github.com/pvigier/perlin-numpy
    from perlin_numpy import generate_fractal_noise_3d, generate_perlin_noise_3d

# NOTE: tissues are blended linearly.
# Exp. fits might be better, but not in general and are more complicated

# The folder that contains this file and a cache of all tissues
BRAINWEB_PATH = os.path.dirname(os.path.realpath(__file__))
# Fat is optional bc. it only is of interest for supression & ofres not handled
INCLUDE_FAT = False
# The BrainWeb data is centered in the MAP_SIZE^3 volume
MAP_SIZE = 432 if INCLUDE_FAT else 432

if INCLUDE_FAT:
    print("WARNING: the maps include fat but don't export offresonance!")


# enumeration of all available BrainWeb tissues
class Tissue(IntEnum):
    CSF = 0
    GRAY_MATTER = 1
    WHITE_MATTER = 2
    FAT = 3
    MUSCLES = 4
    MUSCLES_SKIN = 5
    SKULL = 6
    VESSELS = 7
    CONNECTIVE = 8
    DURA = 9
    BONE_MARROW = 10


TISSUE_DOWNLOAD_ALIAS = [
    "csf", "gry", "wht", "fat", "mus",
    "m-s", "skl", "ves", "fat2", "dura", "mrw"
]

SUBJECTS = [
    4, 5, 6, 18, 20, 38, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54
]


def load(subject: int, tissue: Tissue) -> np.ndarray:
    download_alias = f"subject{subject:02d}_{TISSUE_DOWNLOAD_ALIAS[tissue]}"
    file_name = download_alias + ".i8.gz"  # 8 bit signed int, gnuzip
    file_dir = os.path.join(BRAINWEB_PATH, f"subject{subject:02d}")
    file_path = os.path.join(file_dir, file_name)
    try:
        os.mkdir(file_dir)  # create the cache folder (if it doesn't exist)
    except FileExistsError:
        pass

    # If the file is not cached yet, we will download it
    if not os.path.exists(file_path):
        print(f"Couldn't find {file_name}, downloading it...")
        response = requests.post(
            "https://brainweb.bic.mni.mcgill.ca/cgi/brainweb1",
            data={
                "do_download_alias": download_alias,
                "format_value": "raw_byte",
                "zip_value": "gnuzip",
            }
        )

        with open(file_path, "wb") as f:
            f.write(response.content)

    # Now the file is guaranteed to exist so we can load it
    with gzip.open(file_path) as f:
        # BrainWeb states the data is unsigned, but that's plain wrong
        data = np.frombuffer(f.read(), np.uint8) + 128

        # Coordinate system:
        #  - x points to right ear
        #  - y points to nose
        #  - z points to top of head
        # Indices are data[x, y, z]

        return data.reshape(362, 434, 362).swapaxes(0, 2)


for subject in SUBJECTS:
    tissue_GM = load(subject, Tissue.GRAY_MATTER)
    tissue_WM = load(subject, Tissue.WHITE_MATTER)
    # Vessels have a bug where nearly all background is 1 instead of 0
    vessels = load(subject, Tissue.VESSELS)
    vessels[vessels > 0] -= 1
    # Assuming vessels are mostly whater they should be similar to CSF
    tissue_CSF = load(subject, Tissue.CSF) + vessels
    if INCLUDE_FAT:
        tissue_FAT = (
            load(subject, Tissue.FAT) +
            load(subject, Tissue.MUSCLES) +
            load(subject, Tissue.MUSCLES_SKIN) +
            load(subject, Tissue.DURA) +
            load(subject, Tissue.CONNECTIVE)
        )

    # Downsample and pad to get 128^3 maps
    print("Downsample and center maps")

    def downsample(tensor: np.ndarray):
        # tensor shape must be a multiple of 3 for this to work - remove excess
        shape = (np.array(tensor.shape) // 3) * 3
        tensor = tensor[:shape[0], :shape[1], :shape[2]].astype(np.float32)
        # tensor = tensor[0::3, :, :] + tensor[1::3, :, :] + tensor[2::3, :, :]
        # tensor = tensor[:, 0::3, :] + tensor[:, 1::3, :] + tensor[:, 2::3, :]
        # tensor = tensor[:, :, 0::3] + tensor[:, :, 1::3] + tensor[:, :, 2::3]
        return tensor 

    tissue_GM = downsample(tissue_GM)
    tissue_WM = downsample(tissue_WM)
    tissue_CSF = downsample(tissue_CSF)
    if INCLUDE_FAT:
        tissue_FAT = downsample(tissue_FAT)

    # Find the extends of the brain to center it
    total = tissue_GM + tissue_WM + tissue_CSF
    if INCLUDE_FAT:
        total += tissue_FAT
    mask = total > 0.01
    test = mask.copy()
    x_indices, y_indices, z_indices = np.nonzero(mask)
    x_indices, y_indices, z_indices = np.where(total > 0.1)
    min_x = x_indices.min()
    max_x = x_indices.max() + 1
    min_y = y_indices.min()
    max_y = y_indices.max() + 1
    min_z = z_indices.min()
    max_z = z_indices.max() + 1

    # Warn if brain is too large
    length_x = max_x - min_x
    length_y = max_y - min_y
    length_z = max_z - min_z

    if length_x > MAP_SIZE or length_y > MAP_SIZE or length_z > MAP_SIZE:
        print(f"WARNING: Brain size = {length_x} x {length_y} x {length_z}")
        print(f"Maximum size is MAP_SIZE={MAP_SIZE}^3, maps will be truncated")

    # Center it (and cut to size if too large)
    length_x = min(MAP_SIZE, length_x)
    length_y = min(MAP_SIZE, length_y)
    length_z = min(MAP_SIZE, length_z)
    min_x = int((min_x + max_x) / 2 - length_x / 2)
    min_y = int((min_y + max_y) / 2 - length_y / 2)
    min_z = int((min_z + max_z) / 2 - length_z / 2)
    max_x = min_x + length_x
    max_y = min_y + length_y
    max_z = min_z + length_z

    def add_padding(data):
        # return data
        pad_x = (MAP_SIZE - length_x) // 2
        pad_y = (MAP_SIZE - length_y) // 2
        pad_z = (MAP_SIZE - length_z) // 2

        padded = np.zeros((MAP_SIZE, MAP_SIZE, MAP_SIZE), dtype=data.dtype)
        padded[
            pad_x:(pad_x+length_x),
            pad_y:(pad_y+length_y),
            pad_z:(pad_z+length_z)
        ] = data[min_x:max_x, min_y:max_y, min_z:max_z]
        return padded

    mask = add_padding(mask)
    total = add_padding(total)
    tissue_GM = add_padding(tissue_GM)
    tissue_WM = add_padding(tissue_WM)
    tissue_CSF = add_padding(tissue_CSF)

    tissue_GM[mask] /= total[mask]
    tissue_WM[mask] /= total[mask]
    tissue_CSF[mask] /= total[mask]
    tissue_GM[~mask] = 0
    tissue_WM[~mask] = 0
    tissue_CSF[~mask] = 0

    if INCLUDE_FAT:
        tissue_FAT = add_padding(tissue_FAT)
        tissue_FAT[mask] /= total[mask]
        tissue_FAT[~mask] = 0

    # Generate maps
    print("Generating maps...")

    def perlin():
        # Maps all have noise added in the range [-10%, 10%] of their mean
        return generate_perlin_noise_3d((MAP_SIZE, MAP_SIZE, MAP_SIZE), (4, 4, 4))
    
    def gaussian():
        return np.random.randn((MAP_SIZE, MAP_SIZE, MAP_SIZE))
        

    def gen_map(gm_val, wm_val, csf_val, fat_val):
        #assert NO_NOISE, "perlin_noise not yet re-implemented"
        if NO_NOISE:
            total = (
                tissue_GM * gm_val +
                tissue_WM * wm_val +
                tissue_CSF * csf_val
            )
            if INCLUDE_FAT:
                total += tissue_FAT * fat_val
        else:
            total = (
                tissue_GM * gm_val * (1 + 0.1 * perlin()) +
                tissue_WM * wm_val * (1 + 0.1 * perlin()) +
                tissue_CSF * csf_val * (1 + 0.1 * perlin())
            )
            if INCLUDE_FAT:
                total += tissue_FAT * fat_val ** (1 + 0.1 * perlin())
        return total

    T1_map = gen_map(1.55, 0.83, 4.16, 0.374)
    T2_map = gen_map(0.09, 0.07, 1.65, 0.125)
    # These are calculated from T2* (for which sources are sparse)
    T2dash_map = gen_map(0.322, 0.183, 0.0591, 0.0117)
    # These are completeley guessed
    PD_map = gen_map(0.8, 0.7, 1.0, 1.0)
    # Isometric diffusion in [10^-3 mm^2/s]
    D_map = gen_map(0.83, 0.65, 3.19, 0.1)

    # Plotting
    print("Plotting center slice of generated maps")

    def plot(title, image):
        plt.figure(figsize=(7, 5))
        plt.title(title)
        # NOTE: Plot this way so matplotlib does not rotate the image
        # First index: to the right, second index: to the top
        plt.imshow(image[:, :, 90].T, origin="lower")
        plt.colorbar()
        plt.show()

    plot("$PD$", PD_map)
    plot("$T_1$", T1_map)
    plot("$T_2$", T2_map)
    plot("$T_2'$", T2dash_map)
    plot("$D$", D_map)

    # Save generated data
    name = (
        f"subject{subject:02d}_fat.npz" if INCLUDE_FAT else
        f"subject{subject:02d}.npz"
    )
    file_name = os.path.join(BRAINWEB_PATH, "output", name)
    print(f"Saving maps to 'output/subject{subject:02d}.npz'")
    
    if INCLUDE_FAT:
        np.savez_compressed(
            file_name,
            T1_map=T1_map,
            T2_map=T2_map,
            T2dash_map=T2dash_map,
            PD_map=PD_map,
            D_map=D_map,
            tissue_WM = tissue_WM,
            tissue_GM = tissue_GM,
            tissue_CSF = tissue_CSF
        )
    else:        
        np.savez_compressed(
            file_name,
            T1_map=T1_map,
            T2_map=T2_map,
            T2dash_map=T2dash_map,
            PD_map=PD_map,
            D_map=D_map,
            tissue_WM = tissue_WM,
            tissue_GM = tissue_GM,
            tissue_CSF = tissue_CSF
        )
