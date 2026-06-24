from .voxel_grid_phantom import VoxelGridPhantom
from .sim_data import SimData
from .nifti_phantom import NiftiPhantom, NiftiTissue, NiftiRef, NiftiMapping, ResliceConfig
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Literal, Self
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
            name: load_tissue(tissue, base_dir, reslice=config.reslice_to)
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
        affine = np.asarray(next(iter(self.values())).affine)
        if affine.shape == (3, 4):
            affine = np.vstack([affine, [0, 0, 0, 1]], dtype=np.float32)

        def save_nifti(prop, name):
            if len(prop) > 0:
                ext = f"_{name}" if name != "density" else ""
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

        reslice_to = {
            "resolution": list(next(iter(self.values())).PD.shape),
            "affine": affine[:3, :4].tolist()
        }

        config = NiftiPhantom.default(gyro, B0)
        config.tissues = tissues
        config.reslice_to = ResliceConfig.from_dict(reslice_to)
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
        combined.coil_sens = sum(seg[None, ...] * p.coil_sens for seg, p in zip(segmentation, phantoms))
        
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
            "affine": data_list[0].affine,
            "nyquist": data_list[0].nyquist,
            "dephasing_func": data_list[0].dephasing_func,
            "recover_func": lambda data: self.recover(data),
            "tissue_masks": dict(zip(self.keys(), [obj.tissue_masks["combined"] for obj in data_list]))
        }

        return SimData(**kwargs)

    def recover(self, sim_data: SimData) -> Self:
        """Provided to :class:`SimData` to reverse the ``build()``"""
        
        assert sim_data.tissue_masks is not None
        
        tissues = list(sim_data.tissue_masks.keys())     
        tissue_begin = 0 # first tissue starts at index 0 in sparse tensors    
               
        def to_full(sparse):
            assert sparse.ndim < 3
            
            if sparse.ndim == 2:
                full = torch.zeros(
                    [sparse.shape[0], *mask.shape], dtype=sparse.dtype, device=mask.device)
                full[:, mask] = sparse
            else:
                full = torch.zeros(mask.shape, device=mask.device)
                full[mask] = sparse
            return full
        
        data_list = [] 
        for tissue in tissues:
            mask = sim_data.tissue_masks[tissue]     
            mask = mask.to(sim_data.device)  
           
            tissue_end = tissue_begin + torch.sum(mask)  # index of tissue end in sparse tensors         
            
            data_list.append(VoxelGridPhantom(
                                    to_full(sim_data.PD[tissue_begin:tissue_end]),
                                    to_full(sim_data.T1[tissue_begin:tissue_end]),
                                    to_full(sim_data.T2[tissue_begin:tissue_end]),
                                    to_full(sim_data.T2dash[tissue_begin:tissue_end]),
                                    to_full(sim_data.D[tissue_begin:tissue_end]),
                                    to_full(sim_data.B0[tissue_begin:tissue_end]),
                                    to_full(sim_data.B1[:, tissue_begin:tissue_end]),
                                    to_full(sim_data.coil_sens[:, tissue_begin:tissue_end]),
                                    sim_data.size,
                                    sim_data.affine,
                                )
                            )
            tissue_begin = tissue_end # next tissue in sparse tensors starts where last ended
        
        return TissueDict(dict(zip(tissues, data_list)))
    
    def plot(self, tissue="all", plot_masks=False, plot_slice="center", time_unit='s') -> None:
        """ 
        Plots the individual tissues of the PhantomDict 
        
        Parameters
        ----------
        tissue : str, default="all"
            Specifies which tissue(s) to plot.
            - ``"all"`` : Plot all tissues stored in the PhantomDict, one after another.
            - ``"combined"`` : Plot a combined phantom created from all tissues using :meth:`combine`.
            - any other string is interpreted as a key identifying a single tissue stored in the PhantomDict.
        plot_masks : bool
            Plot tissue masks (assumes they exist)
        slice : str | int
            If int, the specified slice is plotted. "center" plots the center
            slice and "all" plots all slices as a grid.
         time_unit : str
             Time unit to use for T1, T2, and T2' maps (default: 's'). Supported 's' and 'ms'.
        """
        
        if tissue == "all":
            print("Plot combined phatom")
            self.combine().plot(plot_masks, plot_slice, time_unit)
            
            fignum = max(plt.get_fignums(), default=1)
                        
            for name, t in self.items():
                print("Plot tissue: ", name)                
                t.plot(plot_masks, plot_slice, time_unit, f"Figure {fignum} - {name}")
            
        elif tissue == "combined":
            print("Plot combined tissue phantom")
            self.combine().plot(plot_masks, plot_slice, time_unit)
            
        else:            
            print("Plot tissue: ", tissue) 
            self[tissue].plot(plot_masks, plot_slice, time_unit)

# ============================
# Helpers for importing NIfTIs
# ============================

def load_tissue(config: NiftiTissue, base_dir: Path,
                reslice: ResliceConfig | None = None) -> VoxelGridPhantom:
    density, nifti_affine = load_file_ref(base_dir, config.density)

    def lp(cfg):
        return load_property(cfg, base_dir, density, nifti_affine)

    T1     = lp(config.T1)
    T2     = lp(config.T2)
    T2dash = lp(config.T2dash)
    ADC      = lp(config.ADC)
    B0     = lp(config.dB0)
    B1     = [lp(cfg) for cfg in config.B1_tx]
    coil   = [lp(cfg) for cfg in config.B1_rx]

    if reslice is None:
        target_shape = density.shape
        aff_mm = nifti_affine[:3, :]
    else:
        target_shape = tuple(reslice.resolution)
        aff_mm = np.array(reslice.affine, dtype=float)
        
    def rs(arr):
        return _resample_nifti(arr, nifti_affine, target_shape, aff_mm)
    size = target_shape * np.linalg.norm(aff_mm[:3, :3], axis=0) /1000 # np.abs(target_shape @ aff_mm[:3, :3]) / 1000
    density = rs(density)
    T1, T2, T2dash, ADC, B0 = rs(T1), rs(T2), rs(T2dash), rs(ADC), rs(B0)
    B1   = [rs(b) for b in B1]
    coil = [rs(c) for c in coil]

    return VoxelGridPhantom(
        PD=torch.as_tensor(density),
        size=torch.as_tensor(size),
        T1=torch.as_tensor(T1),
        T2=torch.as_tensor(T2),
        T2dash=torch.as_tensor(T2dash),
        D=torch.as_tensor(ADC),
        B0=torch.as_tensor(B0),
        B1=torch.stack([torch.as_tensor(b) for b in B1], 0),
        coil_sens=torch.stack([torch.as_tensor(c) for c in coil], 0),
        affine=torch.as_tensor(aff_mm),
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


def _resample_nifti(data: np.ndarray, nifti_affine: np.ndarray,
                    target_shape: tuple,
                    target_affine_mm: np.ndarray) -> np.ndarray:
    """Resample a 3D array onto a target grid via trilinear interpolation,
    averaging over the slice thickness using the source voxel size as step.

    Parameters
    ----------
    data:
        Source 3D numpy array (native NIfTI voxel space).
    nifti_affine:
        4×4 sform affine of the source NIfTI (mm units).
    target_shape:
        Output shape ``(nx, ny, nz)``.
    target_affine_mm:
        3×4 or 4×4 NIfTI-style affine of the target grid in mm.
        Maps target voxel ``[i, j, k]`` to physical coordinates in mm.
    """
    from scipy.ndimage import affine_transform

    A = np.array(target_affine_mm, dtype=float)
    A_rot   = A[:3, :3]
    A_trans = A[:3, 3]

    A_nifti_inv = np.linalg.inv(nifti_affine[:3, :3])

    # Voxel size of source phantom (column norms of affine rotation)
    src_voxel_mm    = np.linalg.norm(nifti_affine[:3, :3], axis=0)
    # Slice thickness and normal from Z-column of target affine
    slice_vec_mm    = A_rot[:, 2]
    slice_thickness = np.linalg.norm(slice_vec_mm)
    slice_unit      = slice_vec_mm / slice_thickness
    # Step size = source voxel projected onto slice normal
    step_mm   = float(np.dot(src_voxel_mm, np.abs(slice_unit)))
    n_samples = max(int(round(slice_thickness / step_mm)), 1)

    offsets_mm = np.arange(n_samples, dtype=float) * step_mm
    offsets_mm -= offsets_mm.mean()

    M      = A_nifti_inv @ A_rot
    o0     = A_nifti_inv @ (A_trans - nifti_affine[:3, 3])
    o_step = A_nifti_inv @ slice_unit   # offset change per mm along slice normal

    kwargs = dict(output_shape=tuple(target_shape), order=1,
                  mode='constant', cval=0.0, prefilter=False)

    # Convert once outside the loop
    if np.iscomplexobj(data):
        data_r = data.real.copy()
        data_i = data.imag.copy()
    else:
        data_r = data.astype(float, copy=False)
        data_i = None

    accumulator = np.zeros(target_shape,
                           dtype=np.complex128 if data_i is not None else float)

    for delta_mm in offsets_mm:
        o = o0 + delta_mm * o_step
        if data_i is not None:
            accumulator += (affine_transform(data_r, M, offset=o, **kwargs) +
                            1j * affine_transform(data_i, M, offset=o, **kwargs))
        else:
            accumulator += affine_transform(data_r, M, offset=o, **kwargs)

    return accumulator / n_samples

