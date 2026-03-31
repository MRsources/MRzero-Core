from .voxel_grid_phantom import VoxelGridPhantom
from .sim_data import SimData
from .nifti_phantom import NiftiPhantom, NiftiTissue, NiftiRef, NiftiMapping
from pathlib import Path
import torch
import numpy as np
from typing import Literal
from functools import lru_cache


class TissueDict(dict[str, VoxelGridPhantom]):
    @classmethod
    def load(cls, path: Path | str, config: NiftiPhantom | None = None):
        """Load a NIfTI phantom into a dictionary of tissues.
        
        This class is a Python dictionary where the keys are the names of the
        tissues (as written in the `phantom.json`) and the values are
        VoxelGridPhantoms (typically tissues).
        The dictionary is extended by additional methods for saving as NIfTI
        phantom, interpolation, slicing, combining all tissues into a single
        VoxelGridPhantom with weighted voxels, and finally for converting all
        tissues into `SimData` (with overlapping tissues / partial-volume).

        Parameters
        ----------
        path: Path | str
            Either the path to the `phantom.json` configuration file _or_ the
            directory where the NIfTIs are found (if config is provided).
        config: NiftiPhantom
            Optional configuration - can be used to load a NIfTI phantom config
            and modify it in-memory before loading the actual data.
        """
        if config:
            base_dir = Path(path)
        else:
            base_dir = Path(path).parent
            config = NiftiPhantom.load(path)

        # NOTE: Unit conversion is ignored. Currently no other than the default
        # units are supported (conversion factor 1); this might change in the future

        return TissueDict({
            name: load_tissue(tissue, base_dir)
            for name, tissue in config.tissues.items()
        })
    
    def save(self, path_to_json: str | Path, gyro=42.5764, B0=3.0):
        from pathlib import Path
        import os
        import nibabel as nib

        path_to_json = Path(path_to_json)
        base_name = path_to_json.stem
        base_dir = path_to_json.parent
        os.makedirs(base_dir, exist_ok=True)

        density = []
        T1 = []
        T2 = []
        T2dash = []
        ADC = []
        dB0 = []
        B1_tx = []
        B1_rx = []

        def save_tissue(tissue: VoxelGridPhantom):
            config = {}

            def save_map(name, map, nifti):
                ext = f"_{name}" if name != "density" else ""
                # Multi-channel data when setting the same property multiple times
                def set(value):
                    if name in config:
                        config[name].append(value)
                    else:
                        config[name] = value

                if map.std() < 1e-5:
                    set(float(map.mean()))
                else:
                    # Check if map is shared with other tissues
                    for idx, nifti_map in enumerate(nifti):
                        if torch.equal(map, nifti_map):
                            set(f"{base_name}{ext}.nii.gz[{idx}]")
                            return
                    # Not shared, write new map
                    set(f"{base_name}{ext}.nii.gz[{len(nifti)}]")
                    nifti.append(map)

            save_map("density", tissue.PD, density)
            save_map("T1", tissue.T1, T1)
            save_map("T2", tissue.T2, T2)
            save_map("T2'", tissue.T2dash, T2dash)
            save_map("ADC", tissue.D, ADC)
            save_map("dB0", tissue.B0, dB0)
            config["B1+"] = []
            for channel in tissue.B1:
                save_map("B1+", channel, B1_tx)
            config["B1-"] = []
            for channel in tissue.coil_sens:
                save_map("B1-", channel, B1_rx)

            return NiftiTissue.from_dict(config)

        # Generate all tissues (and fill the prop maps)
        tissues = {tissue: save_tissue(self[tissue]) for tissue in self.keys()}

        # Write the NIfTIs
        size = np.asarray(next(iter(self.values())).size)
        vs = 1000 * size / np.asarray(density[0].shape)
        affine = np.array(
            [
                [+vs[0], 0, 0, -size[0] / 2 * 1000],
                [0, +vs[1], 0, -size[1] / 2 * 1000],
                [0, 0, +vs[2], -size[2] / 2 * 1000],
                [0, 0, 0, 0],  # Row ignored
            ]
        )

        def save_nifti(prop, name):
            if len(prop) > 0:
                ext = f"-{name}" if name != "density" else ""
                file_name = base_dir / f"{base_name}{ext}.nii.gz"
                data = np.stack(prop, -1)

                print(f"Storing '{file_name}' - {data.shape}")
                nib.save(nib.nifti1.Nifti1Image(data, affine), file_name)
        
        save_nifti(density, "density")
        save_nifti(T1, "T1")
        save_nifti(T2, "T2")
        save_nifti(T2dash, "T2'")
        save_nifti(ADC, "ADC")
        save_nifti(dB0, "dB0")
        save_nifti(B1_tx, "B1+")
        save_nifti(B1_rx, "B1-")

        config = NiftiPhantom.default(gyro, B0)
        config.tissues = tissues
        config.save(path_to_json)
    
    def interpolate(self, x: int, y: int, z: int):
        return TissueDict({
            name: phantom.interpolate(x, y, z) for name, phantom in self.items()
        })
    
    def slices(self, slices: list[int]):
        return TissueDict({
            name: phantom.slices(slices) for name, phantom in self.items()
        })
    
    def combine(self) -> VoxelGridPhantom:
        """Combine individual maps to mixed-tissue (no partial volume) phantom."""
        phantoms = list(self.values())

        PD = sum(p.PD for p in phantoms)
        segmentation = [p.PD / PD for p in phantoms]
        
        from copy import deepcopy
        combined = deepcopy(phantoms[0])
        combined.PD = PD
        combined.T1 = sum(seg * p.T1 for seg, p in zip(segmentation, phantoms))
        combined.T2 = sum(seg * p.T2 for seg, p in zip(segmentation, phantoms))
        combined.T2dash = sum(seg * p.T2dash for seg, p in zip(segmentation, phantoms))
        combined.D = sum(seg * p.D for seg, p in zip(segmentation, phantoms))
        combined.B0 = sum(seg * p.B0 for seg, p in zip(segmentation, phantoms))
        combined.B1 = sum(seg[None, ...] * p.B1 for seg, p in zip(segmentation, phantoms))
        
        return combined

    def build(self, PD_threshold: float = 1e-6,
              voxel_shape: Literal["sinc", "box", "point"] = "sinc"
              ) -> SimData:
        data_list = [self[tissue].build(PD_threshold, voxel_shape) for tissue in self]

        kwargs = {
            "PD": torch.cat([obj.PD for obj in data_list]),
            "T1": torch.cat([obj.T1 for obj in data_list]),
            "T2": torch.cat([obj.T2 for obj in data_list]),
            "T2dash": torch.cat([obj.T2dash for obj in data_list]),
            "D": torch.cat([obj.D for obj in data_list]),
            "B0": torch.cat([obj.B0 for obj in data_list]),
            "B1": torch.cat([obj.B1 for obj in data_list], 1),
            "coil_sens": torch.cat([obj.coil_sens for obj in data_list], 1),
            "voxel_pos": torch.cat([obj.voxel_pos for obj in data_list], 0),
            "size": data_list[0].size,
            "nyquist": data_list[0].nyquist,
            "dephasing_func": data_list[0].dephasing_func,
        }

        # Only add tissue_masks if any object has it non-empty
        if any(obj.tissue_masks for obj in data_list):
            kwargs["tissue_masks"] = torch.stack([obj.tissue_masks for obj in data_list])

        return SimData(**kwargs)


# ============================
# Helpers for importing NIfTIs
# ============================

def load_tissue(config: NiftiTissue, base_dir: Path) -> VoxelGridPhantom:
    density, affine = load_file_ref(base_dir, config.density)
    size = np.abs(density.shape @ affine[:3, :3]) / 1000  # affine is in mm

    def lp(cfg):
        return torch.as_tensor(load_property(cfg, base_dir, density, affine))

    return VoxelGridPhantom(
        PD=torch.as_tensor(density),
        size=torch.as_tensor(size),
        T1=lp(config.T1),
        T2=lp(config.T2),
        T2dash=lp(config.T2dash),
        D=lp(config.ADC),
        B0=lp(config.dB0),
        B1=torch.stack([lp(cfg) for cfg in config.B1_tx], 0),
        coil_sens=torch.stack([lp(cfg) for cfg in config.B1_rx], 0),
    )


def load_property(config: float | NiftiRef | NiftiMapping,
                  base_dir: Path, density_mat: np.ndarray, target_affine: np.ndarray
                  ) -> np.ndarray:
    
    if isinstance(config, float):
        return np.full_like(density_mat, config)

    if isinstance(config, NiftiRef):
        data, affine = load_file_ref(base_dir, config)
        assert np.all(affine == target_affine)
        return data

    if isinstance(config, NiftiMapping):
        data, affine = load_mapping(base_dir, config)
        assert np.all(affine == target_affine)
        return data

    raise TypeError("Config must be a float, file_ref or mapping", type(config))

def load_mapping(base_dir: Path, file_mapping: NiftiMapping) -> tuple[np.ndarray, np.ndarray]:
    data, affine = load_file_ref(base_dir, file_mapping.file)

    # TODO - SAFETY: Don't use eval but a custom (imported?) expression parser.
    print(f"Executing mapping function: '{file_mapping.func}'")
    return eval(
        file_mapping.func,
        {"__builtins__": None},
        {
            "x": data,
            "x_min": data.min(),
            "x_max": data.max(),
            "x_mean": data.mean(),
            "x_std": data.std()
        }
    ), affine


def load_file_ref(base_dir: Path, file_ref: NiftiRef) -> tuple[np.ndarray, np.ndarray]:
    print(f"Loading NIfTI (file={file_ref.file_name}, index={file_ref.tissue_index})")
    file = file_ref.file_name
    index = file_ref.tissue_index
    if not file.is_absolute():
        file = (base_dir / file).resolve()

    data, affine = _load_cached(str(file))
    return data[:, :, :, index], affine


# Use a small cache to avoid reloading NIfTIs every time
@lru_cache(maxsize=20)
def _load_cached(file_name):
    import nibabel
    img = nibabel.loadsave.load(file_name)
    return np.asarray(img.dataobj), img.get_sform()
