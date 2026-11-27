"""Unit tests for isochromat_sim against PDG (execute_graph + fid_graph)."""

import sys
import types

import torch

# Allow importing MRzeroCore in environments without the compiled Rust _prepass
# extension (e.g. macOS ARM64), since isochromat_sim and fid_graph/execute_graph
# are pure Python/PyTorch.
if "MRzeroCore._prepass" not in sys.modules:
    try:
        import MRzeroCore._prepass
    except ImportError:
        _stub = types.ModuleType("MRzeroCore._prepass")
        _stub.PyDistribution = object
        sys.modules["MRzeroCore._prepass"] = _stub

import MRzeroCore as mr0


def _make_single_voxel_phantom(b1: torch.Tensor) -> mr0.SimData:
    """Create a 1x1x1 voxel phantom with long T1/T2/T2' and custom B1."""
    phantom = mr0.VoxelGridPhantom(
        PD=torch.ones(1, 1, 1),
        T1=torch.full((1, 1, 1), 100.0),
        T2=torch.full((1, 1, 1), 100.0),
        T2dash=torch.full((1, 1, 1), 100.0),
        D=torch.zeros(1, 1, 1),
        B0=torch.zeros(1, 1, 1),
        B1=b1,
        coil_sens=torch.ones(1, 1, 1, 1),
        affine=torch.diag(torch.tensor([10.0, 10.0, 10.0, 1.0]))[:3, :],
    )
    return phantom.build()


def _make_fid_sequence(
    angle: float = torch.pi / 2,
    phase: float = 0.0,
    normalized_grads: bool = False,
) -> mr0.Sequence:
    seq = mr0.Sequence(normalized_grads=normalized_grads)
    rep = seq.new_rep(1)
    rep.pulse.angle = torch.tensor(angle)
    rep.pulse.phase = torch.tensor(phase)
    rep.event_time[:] = torch.tensor([1e-3])
    rep.adc_usage[0] = 1
    return seq


def test_complex_b1_matches_pdg():
    """Complex B1+ should scale flip angle by |B1| and shift phase by angle(B1)."""
    torch.manual_seed(0)
    b1 = torch.full((1, 1, 1, 1), 1j, dtype=torch.complex64)
    data = _make_single_voxel_phantom(b1)
    seq = _make_fid_sequence(angle=torch.pi / 2, phase=0.25)

    sig_pdg = mr0.execute_graph(mr0.fid_graph(seq), seq, data, print_progress=False)
    sig_iso = mr0.isochromat_sim(seq, data, spin_count=200, print_progress=False)

    torch.testing.assert_close(sig_iso, sig_pdg, atol=2e-2, rtol=2e-2)


def test_ptx_b1_with_shim_array_matches_pdg():
    """Multi-coil B1+ with pulse.shim_array should match execute_graph."""
    torch.manual_seed(1)
    b1 = torch.tensor([[[[0.6 + 0.2j]]], [[[0.3 - 0.5j]]]], dtype=torch.complex64)
    data = _make_single_voxel_phantom(b1)
    seq = _make_fid_sequence(angle=torch.pi / 3, phase=0.1)
    seq[0].pulse.shim_array = torch.tensor([[1.0, 0.3], [0.8, -0.4]])

    sig_pdg = mr0.execute_graph(mr0.fid_graph(seq), seq, data, print_progress=False)
    sig_iso = mr0.isochromat_sim(seq, data, spin_count=200, print_progress=False)

    torch.testing.assert_close(sig_iso, sig_pdg, atol=2e-2, rtol=2e-2)


def test_normalized_grads_matches_pdg():
    """Sequence(normalized_grads=True) should scale gradm by 1/data.size like PDG."""
    torch.manual_seed(2)
    nx = 4
    fov_mm = 200.0  # 0.2 m FOV -> 5x difference if unscaled
    dx_mm = fov_mm / nx
    phantom = mr0.VoxelGridPhantom(
        PD=torch.ones(nx, 1, 1),
        T1=torch.full((nx, 1, 1), 100.0),
        T2=torch.full((nx, 1, 1), 100.0),
        T2dash=torch.full((nx, 1, 1), 100.0),
        D=torch.zeros(nx, 1, 1),
        B0=torch.zeros(nx, 1, 1),
        B1=torch.ones(1, nx, 1, 1, dtype=torch.complex64),
        coil_sens=torch.ones(1, nx, 1, 1),
        affine=torch.diag(torch.tensor([dx_mm, dx_mm, dx_mm, 1.0]))[:3, :],
    )
    data = phantom.build()

    seq = mr0.Sequence(normalized_grads=True)
    rep = seq.new_rep(2)
    rep.pulse.angle = torch.tensor(torch.pi / 2)
    rep.event_time[:] = torch.tensor([1e-3, 1e-3])
    # 1 full cycle across the 4-voxel FOV -> destructive interference -> ~0 signal
    rep.gradm[0, 0] = 1.0
    rep.adc_usage[1] = 1

    sig_pdg = mr0.execute_graph(mr0.fid_graph(seq), seq, data, print_progress=False)
    sig_iso = mr0.isochromat_sim(
        seq, data, spin_count=400, spin_dist="rand", print_progress=False
    )

    assert sig_pdg.abs().item() < 1e-4
    torch.testing.assert_close(sig_iso, sig_pdg, atol=5e-2, rtol=5e-2)


def test_spin_dist_r2_shape_and_accuracy():
    """spin_dist='r2' should work for arbitrary spin_count (not crash when != 3)."""
    data = _make_single_voxel_phantom(torch.ones(1, 1, 1, 1, dtype=torch.complex64))
    seq = _make_fid_sequence()

    sig_pdg = mr0.execute_graph(mr0.fid_graph(seq), seq, data, print_progress=False)
    sig_iso = mr0.isochromat_sim(
        seq,
        data,
        spin_count=50,
        spin_dist="r2",
        r2_seed=torch.tensor([0.1, 0.2, 0.3]),
        print_progress=False,
    )

    torch.testing.assert_close(sig_iso, sig_pdg, atol=2e-2, rtol=2e-2)


def test_custom_phantom_inf_nyquist_triggers_intravoxel_dephasing():
    """CustomVoxelPhantom (nyquist=inf) should use fallback voxel_size=0.1, not 0.0."""
    torch.manual_seed(3)
    phantom = mr0.CustomVoxelPhantom(
        pos=[[0.0, 0.0, 0.0]],
        T1=100.0,
        T2=100.0,
        T2dash=100.0,
        voxel_size=0.1,
        voxel_shape="box",
    )
    data = phantom.build()

    seq = mr0.Sequence(normalized_grads=False)
    rep = seq.new_rep(2)
    rep.pulse.angle = torch.tensor(torch.pi / 2)
    rep.event_time[:] = torch.tensor([1e-3, 1e-3])
    # 1 full cycle over a 0.1 m box voxel (gradm = 10 1/m) -> sinc(1) = 0
    rep.gradm[0, 0] = 10.0
    rep.adc_usage[1] = 1

    sig_iso = mr0.isochromat_sim(
        seq,
        data,
        spin_count=1000,
        spin_dist="rand",
        print_progress=False,
    )
    # Without fallback voxel_size=0.1, spin_pos is all zeros and |sig| == 1.0
    assert sig_iso.abs().item() < 0.1
