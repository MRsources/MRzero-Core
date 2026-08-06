"""CUDA footprint_mean resampling via batched ``grid_sample`` (substeps + channels + depth)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def _index_to_grid(c: torch.Tensor, size: int) -> torch.Tensor:
    """Continuous voxel index → ``grid_sample`` coord (``align_corners=False``)."""
    return 2.0 * (c + 0.5) / float(size) - 1.0


def _build_footprint_grid(
    ref_shape: tuple[int, int, int],
    substeps: tuple[int, int, int],
    vox_to_vox: np.ndarray,
    source_shape: tuple[int, int, int],
    device: torch.device,
) -> torch.Tensor:
    """``(S, D_out, H_out, W_out, 3)`` grid for ``grid_sample`` (D=nz, H=ny, W=nx)."""
    st_i, st_j, st_k = substeps
    nx, ny, nz = ref_shape
    sx, sy, sz = source_shape
    s_count = st_i * st_j * st_k

    i = torch.arange(nx, device=device, dtype=torch.float32)
    j = torch.arange(ny, device=device, dtype=torch.float32)
    k = torch.arange(nz, device=device, dtype=torch.float32)

    flat_s = torch.arange(s_count, device=device)
    uj = flat_s % st_j
    ui = (flat_s // st_j) % st_i
    uk = flat_s // (st_i * st_j)

    off_i_s = -0.5 + (ui.to(torch.float32) + 0.5) / st_i
    off_j_s = -0.5 + (uj.to(torch.float32) + 0.5) / st_j
    off_k_s = -0.5 + (uk.to(torch.float32) + 0.5) / st_k

    x_ref = i.view(1, nx, 1, 1) + off_i_s.view(s_count, 1, 1, 1)
    y_ref = j.view(1, 1, ny, 1) + off_j_s.view(s_count, 1, 1, 1)
    z_ref = k.view(1, 1, 1, nz) + off_k_s.view(s_count, 1, 1, 1)
    x_ref = x_ref.expand(s_count, nx, ny, nz)
    y_ref = y_ref.expand(s_count, nx, ny, nz)
    z_ref = z_ref.expand(s_count, nx, ny, nz)

    vox = torch.as_tensor(vox_to_vox, device=device, dtype=torch.float32)
    homog = torch.stack(
        [x_ref, y_ref, z_ref, torch.ones_like(x_ref)],
        dim=-1,
    )
    src = homog @ vox.T
    sc_x, sc_y, sc_z = src[..., 0], src[..., 1], src[..., 2]

    grid = torch.stack(
        [
            _index_to_grid(sc_x, sx),
            _index_to_grid(sc_y, sy),
            _index_to_grid(sc_z, sz),
        ],
        dim=-1,
    )
    return grid.permute(0, 3, 2, 1, 4)


def _resample_footprint_torch(
    source_data: np.ndarray | torch.Tensor,
    vox_to_vox: np.ndarray | torch.Tensor,
    ref_shape: tuple[int, int, int],
    substeps: tuple[int, int, int],
    *,
    device: torch.device | None = None,
    grid: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Footprint-mean resample ``source_data`` onto ``ref_shape``.

    ``source_data`` is NIfTI array order ``(nx, ny, nz)`` or ``(nx, ny, nz, C)``.
    Optional ``grid`` avoids rebuilding the footprint grid (shared across same geometry).
    """
    if device is None:
        device = torch.device("cuda")
    
    vol = torch.as_tensor(source_data, device=device, dtype=torch.float32)
    if vol.ndim == 3:
        vol = vol.unsqueeze(-1)
    
    sx, sy, sz, channels = vol.shape
    s_count = substeps[0] * substeps[1] * substeps[2]

    # (C, sz, sy, sx) for grid_sample (N, C, D, H, W)
    src = vol.permute(3, 2, 1, 0).unsqueeze(0).expand(s_count, -1, -1, -1, -1)

    if grid is None:
        grid = _build_footprint_grid(ref_shape, substeps, vox_to_vox, (sx, sy, sz), device)
    else:
        grid = grid.to(device=device)

    with torch.no_grad():
        sampled = F.grid_sample(
            src,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        out = sampled.mean(dim=0)

    out_np = out.permute(3, 2, 1, 0)
    if channels == 1:
        return out_np[..., 0]
    return out_np

def resample_nifti(
    data: np.ndarray | torch.Tensor,
    nifti_affine: np.ndarray | torch.Tensor,
    target_shape: tuple,
    target_affine_mm: np.ndarray | torch.Tensor,
    device: torch.device = torch.device("cpu")
) -> np.ndarray:
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
    
    data = torch.as_tensor(data, device=device)
    nifti_affine = torch.as_tensor(nifti_affine, device=device, dtype=torch.float32)
    target_affine_mm = torch.as_tensor(target_affine_mm, device=device, dtype=torch.float32)

    # A constant field stays constant under any resampling. Short-circuit this
    # case so that constant defaults (e.g. T2 == inf for "no decay") survive:
    # feeding non-finite values to ``affine_transform`` would yield NaNs.
    flat = data.reshape(-1)
    if flat.numel() > 0 and torch.all(flat == flat[0]):
        const = torch.full(
            tuple(target_shape), flat[0], dtype=torch.float32, device=device
        )
        return const

    A_rot   = target_affine_mm[:3, :3]
    A_trans = target_affine_mm[:3, 3]
    A_nifti_inv = torch.linalg.inv(nifti_affine[:3, :3])

    # Source voxel size
    src_voxel_mm = torch.linalg.norm(nifti_affine[:3, :3], axis=0)

    # --- Substeps per axis
    substeps: list[int] = []
    for a in range(3):
        axis_vec_mm = A_rot[:, a]
        axis_len_mm = torch.linalg.norm(axis_vec_mm)
        axis_unit = axis_vec_mm / axis_len_mm
        step_mm = torch.dot(src_voxel_mm, torch.abs(axis_unit))
        n_samples = max(int(round((axis_len_mm / step_mm).item())), 1)
        substeps.append(n_samples)
    
    # --- vox_to_vox (Target-Voxel -> Source-Voxel)
    M  = A_nifti_inv @ A_rot
    o0 = A_nifti_inv @ (A_trans - nifti_affine[:3, 3])
    vox_to_vox = torch.cat([M, o0.unsqueeze(1)], dim=1)
    
    def _run(arr: torch.Tensor) -> torch.Tensor:
        return _resample_footprint_torch(
            arr,
            vox_to_vox,
            tuple(target_shape),
            tuple(substeps),
            device=device,
            grid=None,
        )
    
    if torch.is_complex(data):
        out_r = _run(data.real.contiguous())
        out_i = _run(data.imag.contiguous())
        return out_r + 1j * out_i

    return _run(data)


def apply_patient_orientation(
    affine: torch.Tensor | np.ndarray,
    patient_pos: str,
    shape: tuple[int, int, int],  # (nx, ny, nz)
) -> torch.Tensor:
    """Adjust a NIfTI-style affine (voxel-center convention) to reflect
    a patient-orientation flip (e.g. FFS -> HFS), without touching the
    underlying voxel data.

    Parameters
    ----------
    affine:
        4x4 affine, maps voxel index [i, j, k] to the physical mm
        coordinate of the voxel CENTER.
    patient_pos:
        "ffs" or "hfs".
    shape:
        (nx, ny, nz) of the voxel array the affine belongs to.
    """

    if patient_pos == "ffs":
        sign = torch.tensor([1., 1., 1.])
    elif patient_pos == "hfs":
        sign = torch.tensor([-1., 1., -1.])
    else:
        raise ValueError(f"Unsupported patient position '{patient_pos}'")
    
    affine = torch.as_tensor(affine, dtype=torch.float32)
    
    new_affine = affine.clone()

    # 1) Flip direction columns
    new_affine[:3, :3] = affine[:3, :3] * sign

    # 2) Correct translation: voxel-center convention needs (N-1),
    #    NOT N, since index 0 must map to the OLD position of the
    #    last voxel center (index N-1), not to the FOV edge.
    shape_t = torch.tensor(shape, dtype=torch.float32)
    mask = (sign == -1).float()
    shift = affine[:3, :3] @ ((shape_t - 1) * mask)
    new_affine[:3, 3] = affine[:3, 3] + shift
    
    # 3) Re-center: shift translation so the grid's true center
    #    maps to physical (0, 0, 0).
    #    If resolution is odd no extra shift is necessary!
    center_idx = shape_t // 2 + mask * sign * ((shape_t + 1) % 2)
    center = new_affine[:3, :3] @ center_idx + new_affine[:3, 3]
    new_affine[:3,3] -= center

    return new_affine

