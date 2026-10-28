from .voxel_grid_phantom import VoxelGridPhantom
from .sim_data import SimData
from .nifti_phantom import NiftiPhantom, NiftiTissue, NiftiRef, ResliceConfig, Patient
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Literal, Self


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
            name: load_tissue(tissue, base_dir, config.reslice_to, config.patient)
            for name, tissue in config.tissues.items()
        })
    
    def save(self, path_to_json: str | Path, gyro=42.5764, B0=3.0):
        """Save as BIfTI phantom: the `.json` and NIfTIs named after it.

        Maps that are (nearly) constant are stored as values in the `.json`.
        """
        import bifti

        path_to_json = Path(path_to_json)
        config = NiftiPhantom.default(gyro, B0)
        tissues = {}

        def np_map(t: torch.Tensor):
            return t.detach().cpu().numpy()

        for name, tissue in self.items():
            # Placeholder references, replaced by the files written by bifti
            ref = NiftiRef(Path(), 0)
            config.tissues[name] = NiftiTissue(
                density=ref, T1=ref, T2=ref, T2dash=ref, ADC=ref, dB0=ref,
                B1_tx=[ref] * tissue.B1.shape[0],
                B1_rx=[ref] * tissue.coil_sens.shape[0],
            )
            tissues[name] = bifti.NumpyTissue(
                density=np_map(tissue.PD),
                T1=np_map(tissue.T1),
                T2=np_map(tissue.T2),
                T2dash=np_map(tissue.T2dash),
                ADC=np_map(tissue.D),
                dB0=np_map(tissue.B0),
                B1_tx=np_map(tissue.B1),
                B1_rx=np_map(tissue.coil_sens),
                resliced=ResliceConfig(
                    affine=tissue.affine[:3, :].tolist(),
                    resolution=list(tissue.PD.shape),
                ),
            )

        bifti.NumpyPhantom(config, tissues).save(
            path_to_json.parent, path_to_json.name, constant_threshold=1e-5
        )
    
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
        # Segmented phantoms have background voxels where every tissue has
        # PD == 0. A plain ``p.PD / PD`` divides 0 / 0 -> NaN there; guard the
        # denominator so empty voxels get a weight of 0 instead.
        denom = torch.where(PD > 0, PD, torch.ones_like(PD))
        segmentation = [p.PD / denom for p in phantoms]

        def weighted_sum(values: list[torch.Tensor]) -> torch.Tensor:
            # ``torch.where`` keeps zero-weight voxels at 0 even when the tissue
            # carries a default (infinite) relaxation value, avoiding the
            # 0 * inf == NaN that would otherwise reach ``compute_graph``.
            total = None
            for seg, value in zip(segmentation, values):
                weight = seg
                while weight.ndim < value.ndim:
                    weight = weight[None, ...]
                term = torch.where(weight > 0, weight * value,
                                   torch.zeros_like(value))
                total = term if total is None else total + term
            return total

        from copy import deepcopy
        combined = deepcopy(phantoms[0])
        combined.PD = PD
        combined.T1 = weighted_sum([p.T1 for p in phantoms])
        combined.T2 = weighted_sum([p.T2 for p in phantoms])
        combined.T2dash = weighted_sum([p.T2dash for p in phantoms])
        combined.D = weighted_sum([p.D for p in phantoms])
        combined.B0 = weighted_sum([p.B0 for p in phantoms])
        combined.B1 = weighted_sum([p.B1 for p in phantoms])
        combined.coil_sens = weighted_sum([p.coil_sens for p in phantoms])

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
                reslice: ResliceConfig | None = None,
                patient: Patient | None = None) -> VoxelGridPhantom:
    """Load a tissue with the `bifti` reference loader into a VoxelGridPhantom.

    The voxel-to-scanner affine includes the phantom's patient position (if any).
    """
    import bifti.loader

    tissue = bifti.loader.load_tissue(config, base_dir, reslice)
    affine = bifti.to_scanner_affine(tissue.affine, patient)[:3, :]

    return VoxelGridPhantom(
        PD=torch.as_tensor(tissue.density),
        T1=torch.as_tensor(tissue.T1),
        T2=torch.as_tensor(tissue.T2),
        T2dash=torch.as_tensor(tissue.T2dash),
        D=torch.as_tensor(tissue.ADC),
        B0=torch.as_tensor(tissue.dB0),
        B1=torch.as_tensor(tissue.B1_tx),
        coil_sens=torch.as_tensor(tissue.B1_rx),
        affine=torch.as_tensor(affine),
    )
