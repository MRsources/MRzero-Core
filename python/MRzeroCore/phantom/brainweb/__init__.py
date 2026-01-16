import numpy as np
from pathlib import Path


def load_tissue(subject: int, alias: str, cache_dir: Path) -> np.ndarray:
    import os
    import requests
    import gzip

    download_alias = f"subject{subject:02d}_{alias}"
    file_name = download_alias + ".i8.gz"  # 8 bit signed int, gnuzip
    file_path = cache_dir / file_name

    # Download and cache file if it doesn't exist yet
    if not os.path.exists(file_path):
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
    
    # Convert to RAS+ [x, y, z] indexing and [0..1] range
    data = tmp.reshape(362, 434, 362).swapaxes(0, 2).astype(np.float32) / 255.0

    return data


def generate_B0_B1(mask):
    """Generate a somewhat plausible B0 and B1 map.

    Visually fitted to look similar to the numerical_brain_cropped
    """
    x_pos, y_pos, z_pos = np.meshgrid(
        np.linspace(-1, 1, mask.shape[0]),
        np.linspace(-1, 1, mask.shape[1]),
        np.linspace(-1, 1, mask.shape[2]),
        indexing="ij"
    )
    B1 = np.exp(-(0.4*x_pos**2 + 0.2*y_pos**2 + 0.3*z_pos**2))
    dist2 = (0.4*x_pos**2 + 0.2*(y_pos - 0.7)**2 + 0.3*z_pos**2)
    B0 = 7 / (0.05 + dist2) - 45 / (0.3 + dist2)

    # Normalize such that the average over the mask is 0 or 1
    
    B0 -= B0[mask].mean()
    B1 /= B1[mask].mean()
    B0[~mask] = 0
    B1[~mask] = 0

    return B0, B1


def generate_brainweb_phantoms(output_dir: str, subject_count: int | None = None):
    """Generate BrainWeb phantom maps for the selected configuration.

    Raw tissue segmentation data is provided by the BrainWeb Database:
    http://www.bic.mni.mcgill.ca/brainweb/

    The generated phantoms are stored as MR-zero
    [NIfTI phantoms](https://mrsources.github.io/MRzero-Core/nifti-overview.html).
    Settings for the generated phantoms are stored in `brainweb_data.json`.

    Parameters
    ----------
    output_dir: str
        The directory where the generated phantoms will be stored to. In
        addition, a `cache` folder will be generated there too, which contains
        all the data downloaded from BrainWeb to avoid repeating the download
        for all configurations or when generating phantoms again.
    subject_count: int
        Number of phantoms to generate. BrainWeb provides 20 different segmented
        phantoms. If you don't need as many, you can lower this number to only
        generate the first `count` phantoms. If `None`, all phantoms are generated.
    """
    import os
    import json
    from tqdm import tqdm
    import nibabel as nib
    from ..nifti_phantom import NiftiPhantom, NiftiTissue, NiftiRef, NiftiMapping

    cache_dir = Path(output_dir) / "cache"
    config_file = Path(__file__).parent / "brainweb_data.json"
    os.makedirs(cache_dir, exist_ok=True)

    # Load the brainweb data file that contains info about tissues and subjects
    with open(config_file) as f:
        config = json.load(f)

    if subject_count is None:
        subject_count = len(config["subjects"])

    # Voxel index to physical coordinates in millimeters:
    # Brainweb has 0.5 mm voxel size; we center the brain.
    affine = np.array(
        [
            [0.5, 0, 0, -90.5],  # X: Right
            [0, 0.5, 0, -108.5], # Y: Anterior
            [0, 0, 0.5, -90.5],  # Z: Superior
            [0, 0, 0, 0],        # ignored
        ]
    )


    number_of_downloads = subject_count * sum(len(maps) for maps in config["download-aliases"].values())
    number_of_saves = subject_count * 3  # density, B0, B1
    number_of_jsons = subject_count * len(config["fields"])
    total = number_of_downloads + number_of_saves + number_of_jsons

    with tqdm(total=total) as pbar:
        for subject in config["subjects"][:subject_count]:
            phantom_name = f"brainweb-subj{subject:02d}"
            phantom_dir = Path(output_dir) / phantom_name

            pbar.set_description(str(phantom_dir))
            os.makedirs(phantom_dir, exist_ok=True)

            tissue_indices = {}
            density_maps = []
            for tissue, density in config["density"].items():
                density_map = 0

                for alias in config["download-aliases"][tissue]:
                    pbar.set_postfix_str(f"loading 'subject{subject:02d}_{alias}'")
                    density_map += density * load_tissue(subject, alias, cache_dir)
                    pbar.update()

                tissue_indices[tissue] = len(density_maps)
                density_maps.append(density_map)

            pbar.set_postfix_str(f"saving '{phantom_name}.nii.gz'")
            nib.loadsave.save(
                nib.nifti1.Nifti1Image(np.stack(density_maps, -1), affine),
                phantom_dir / f"{phantom_name}.nii.gz"
            )
            pbar.update()

            B0, B1 = generate_B0_B1(sum(density_maps) > 0)
            pbar.set_postfix_str(f"saving '{phantom_name}_dB0.nii.gz'")
            nib.loadsave.save(
                nib.nifti1.Nifti1Image(B0[..., None], affine),
                phantom_dir / f"{phantom_name}_dB0.nii.gz"
            )
            pbar.update()
            pbar.set_postfix_str(f"saving '{phantom_name}_B1+.nii.gz'")
            nib.loadsave.save(
                nib.nifti1.Nifti1Image(B1[..., None], affine),
                phantom_dir / f"{phantom_name}_B1+.nii.gz"
            )
            pbar.update()

            for field in config["fields"]:
                phantom = NiftiPhantom.default(B0=field)
                pbar.set_postfix_str(f"saving '{phantom_name}-{field}T.json'")
                density_maps = []

                for tissue in config["tissues"]:
                    tissue_config = config["props"][str(field)][tissue]

                    phantom.tissues[tissue] = NiftiTissue(
                        density=NiftiRef(Path(f"{phantom_name}.nii.gz"), tissue_indices[tissue]),
                        T1=tissue_config["T1"],
                        T2=tissue_config["T2"],
                        T2dash=tissue_config["T2'"],
                        ADC=tissue_config["ADC"],
                        dB0=NiftiRef(Path(f"{phantom_name}_dB0.nii.gz"), 0),
                        B1_tx=[NiftiRef(Path(f"{phantom_name}_B1+.nii.gz"), 0)],
                        B1_rx=[1.0],
                    )
                    if tissue == "fat":
                        phantom.tissues["fat"].dB0 = NiftiMapping(
                            file=NiftiRef(Path(f"{phantom_name}_dB0.nii.gz"), 0),
                            func=config["dB0-fat-remap"][str(field)]
                        )

                phantom.save(phantom_dir / f"{phantom_name}-{field}T.json")
                pbar.update()
        
        pbar.close()
