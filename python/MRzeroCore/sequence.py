from __future__ import annotations
from time import time
import warnings
import torch
import numpy as np
from enum import Enum
from typing import Iterable, Literal, Optional
import matplotlib.pyplot as plt

# TODO: if everything is working, deprecate old pulseq loader
from .pulseq.pulseq_loader import intermediate, PulseqFile, Adc, Spoiler
import pydisseqt


class PulseUsage(Enum):
    """:class:`Enum` of all pulse usages, needed for reconstruction.

    The simulation always simulates all magnetization pathways and will ignore
    the pulse usage. It is only used for reconstruction, by the
    :meth:`Sequence.get_kspace` method, to understand the role of a pulse.
    Additionally, Pulseq exportes might use the pulse usage to select the
    correct pulse type.

    Attributes
    ----------
    UNDEF : str
        ``"undefined"``: No specified use case.
    EXIT : str
        ``"excitation"``: Will set the kspace position back to zero
        or the position stored by the last `STORE` pulse.
    REFOC : str
        ``"refocussing"``: Mirrors the kspace position.
    STORE : str
        ``"storing"``: Stores the current kspace position.
        Can be used for DREAM-like sequences.
    FATSAT : str
        ``"fatsaturation"``: Not handled differently by the simulation, but
        can be used by Pulseq exporters to emit a fat-saturation pulse.
    """

    UNDEF = "undefined"
    EXCIT = "excitation"
    REFOC = "refocussing"
    STORE = "storing"
    FATSAT = "fatsaturation"


class Pulse:
    """Contains the definition of an instantaneous RF Pulse.

    Attributes
    ----------
    usage : PulseUsage
        Specifies how this pulse is used, needed only for reconstruction
    angle : torch.Tensor
        Flip angle in radians
    phase : torch.Tensor
        Pulse phase in radians
    pulse_freq: torch.Tensor, to be removed in the future!
        pulse frequency omega_1 = angle/duration
    freq_offset: torch.Tensor
        Frequency offset in Hz
	duration: torch.Tensor
		pulse duration in seconds
	grad: torch.Tensor (dim=3)
		gradient during the pulse in Hz/m per channel (x,y,z)
    off_ress: bool
        Specifies if the pulse should be simulated with the off-resonance treatment        
    shim_array : torch.Tensor
        Contains B1 mag and phase, used for pTx. 2D tensor([[1, 0]]) for 1Tx.
    selective : bool
        Specifies if this pulse should be slice-selective (z-direction)
    """

    def __init__(
        self,
        usage: PulseUsage,
        angle: torch.Tensor,
        phase: torch.Tensor,
        
        pulse_freq: torch.Tensor, # to be removed in the future
        freq_offset: torch.Tensor,
        duration: torch.Tensor,
        grad: torch.Tensor,
        off_res: bool,
        
        shim_array: torch.Tensor,
        selective: bool,
    ):
        """Create a Pulse instance."""
        self.usage = usage
        self.angle = angle
        self.phase = phase
        
        self.pulse_freq = pulse_freq # to be removed in the future
        self.freq_offset = freq_offset
        self.duration = duration
        self.grad = grad
        self.off_res = off_res
        
        self.shim_array = shim_array
        self.selective = selective

    def cpu(self) -> Pulse:
        """Move this pulse to the CPU and return it."""
        return Pulse(
            self.usage,
            torch.as_tensor(self.angle, dtype=torch.float32).cpu(),
            torch.as_tensor(self.phase, dtype=torch.float32).cpu(),
            
            torch.as_tensor(self.pulse_freq, dtype=torch.float32).cpu(), # to be removed in the future
            torch.as_tensor(self.freq_offset, dtype=torch.float32).cpu(),
            torch.as_tensor(self.duration, dtype=torch.float32).cpu(),
            torch.as_tensor(self.grad, dtype=torch.float32).cpu(),
            self.off_res,
            
            torch.as_tensor(self.shim_array, dtype=torch.float32).cpu(),
            self.selective
        )

    def cuda(self, device: int | None = None) -> Pulse:
        """Move this pulse to the specified CUDA device and return it."""
        return Pulse(
            self.usage,
            torch.as_tensor(self.angle, dtype=torch.float32).cuda(device),
            torch.as_tensor(self.phase, dtype=torch.float32).cuda(device),
            
            torch.as_tensor(self.pulse_freq, dtype=torch.float32).cuda(device), # to be removed in the future
            torch.as_tensor(self.freq_offset, dtype=torch.float32).cuda(device),
            torch.as_tensor(self.duration, dtype=torch.float32).cuda(device),
            torch.as_tensor(self.grad, dtype=torch.float32).cuda(device),
            self.off_res,
            
            torch.as_tensor(self.shim_array, dtype=torch.float32).cuda(device),
            self.selective
        )

    @property
    def device(self) -> torch.device:
        """Return the device this pulse is stored on."""
        return self.angle.device

    @classmethod
    def zero(cls):
        """Create a pulse with zero flip and phase."""
        return cls(
            PulseUsage.UNDEF,
            torch.zeros(1, dtype=torch.float32),
            torch.zeros(1, dtype=torch.float32),
            torch.zeros(1, dtype=torch.float32), # to be removed in the future
            torch.zeros(1, dtype=torch.float32),
            torch.zeros(1, dtype=torch.float32),
            torch.zeros(3, dtype=torch.float32),
            False,
            torch.asarray([[1, 0]], dtype=torch.float32),
            True
        )

    def clone(self) -> Pulse:
        """Return a cloned copy of self."""
        return Pulse(
            self.usage,
            self.angle.clone(),
            self.phase.clone(),
            
            self.pulse_freq.clone(), # to be removed in the future
            self.freq_offset.clone(),
            self.duration.clone(),
            self.grad.clone(),
            self.off_res,
            
            self.shim_array.clone(),
            self.selective
        )


class Repetition:
    """A :class:`Repetition` starts with a RF pulse and ends before the next.

    Attributes
    ----------
    pulse : Pulse
        The RF pulse at the beginning of this :class:`Repetition`
    event_time : torch.Tensor
        Duration of each event (seconds)
    gradm : torch.Tensor
        Gradient moment of every event, shape (:attr:`event_count`, 3)
    adc_phase : torch.Tensor
        Float tensor describing the adc phase, shape (:attr:`event_count`, 3)
    adc_usage: torch.Tensor
        Int tensor specifying which contrast a sample belongs to, shape
        (:attr:`event_count`, 3). Samples with ```adc_usage <= 0``` will not be
        measured. For single contrast sequences, just use 0 or 1.
    event_count : int
        Number of events in this :class:`Repetition`
    """

    def __init__(
        self,
        pulse: Pulse,
        event_time: torch.Tensor,
        gradm: torch.Tensor,
        adc_phase: torch.Tensor,
        adc_usage: torch.Tensor
    ):
        """Create a repetition based on the given tensors.

        Raises
        ------
        ValueError
            If not all tensors have the same shape or have zero elements.
        """
        if event_time.numel() == 0:
            raise ValueError("Can't create a repetition with zero elements")

        self.pulse = pulse
        self.event_count = event_time.numel()

        if event_time.shape != torch.Size([self.event_count]):
            raise ValueError(
                f"Wrong event_time shape {tuple(event_time.shape)}, "
                f"expected {(self.event_count, )}"
            )
        if gradm.shape != torch.Size([self.event_count, 3]):
            raise ValueError(
                f"Wrong gradm shape {tuple(gradm.shape)}, "
                f"expected {(self.event_count, 3)}"
            )
        if adc_phase.shape != torch.Size([self.event_count]):
            raise ValueError(
                f"Wrong adc_phase shape {tuple(adc_phase.shape)}, "
                f"expected {(self.event_count, )}"
            )
        if adc_usage.shape != torch.Size([self.event_count]):
            raise ValueError(
                f"Wrong adc_usage shape {tuple(adc_usage.shape)}, "
                f"expected {(self.event_count, )}"
            )

        self.event_time = event_time
        self.gradm = gradm
        self.adc_phase = adc_phase
        self.adc_usage = adc_usage
        # Per-event integer labels (pulseq LABELSET/LABELINC state at the time
        # each ADC fires). Populated by the pulseq_rs importer; empty otherwise.
        # Each tensor has shape (event_count,); values at non-ADC events are
        # meaningless and should be masked by adc_usage > 0.
        self.adc_labels: dict[str, torch.Tensor] = {}

    def cuda(self, device: int | None = None) -> Repetition:
        """Move this repetition to the specified CUDA device and return it."""
        return Repetition(
            self.pulse.cuda(device),
            self.event_time.cuda(device),
            self.gradm.cuda(device),
            self.adc_phase.cuda(device),
            self.adc_usage.cuda(device)
        )

    def cpu(self) -> Repetition:
        """Move this repetition to the CPU and return it."""
        return Repetition(
            self.pulse.cpu(),
            self.event_time.cpu(),
            self.gradm.cpu(),
            self.adc_phase.cpu(),
            self.adc_usage.cpu()
        )

    @property
    def device(self) -> torch.device:
        """Return the repetition this pulse is stored on."""
        return self.gradm.device

    @classmethod
    def zero(cls, event_count: int) -> Repetition:
        """Create a ``Repetition`` instance with everything set to zero.

        Parameters
        ----------
        event_count : int
            Number of events in the new :class:`Repetition`.

        Returns
        -------
        :class:`Repetition`
            A :class:`Repetition` that is part of this :class:`Sequence`.
        """
        return cls(
            Pulse.zero(),
            torch.zeros(event_count, dtype=torch.float32),
            torch.zeros((event_count, 3), dtype=torch.float32),
            torch.zeros(event_count, dtype=torch.float32),
            torch.zeros(event_count, dtype=torch.int32)
        )

    def clone(self) -> Repetition:
        """Create a copy of self with cloned tensors."""
        return Repetition(
            self.pulse.clone(),
            self.event_time.clone(),
            self.gradm.clone(),
            self.adc_phase.clone(),
            self.adc_usage.clone()
        )

    def get_contrasts(self) -> list[int]:
        """Return a sorted list of contrasts used by this ``Repetition``."""
        return sorted(self.adc_usage[self.adc_usage > 0].unique().tolist())

    def shift_contrasts(self, offset: int):
        """Increment all contrasts used by this repetition by ``offset``.

        Only operates on elements that are already larger than zero, so this
        function does not change which elements are measured.
        """
        self.adc_usage[self.adc_usage > 0] += offset


class Sequence(list):
    """Defines a MRI sequence.

    This extends a standard python list and inherits all its functions. It
    additionally implements MRI sequence specific methods.
    """

    def __init__(self, repetitions: Iterable[Repetition] = [],
                 normalized_grads: bool = True):
        """Create a ``Sequence`` instance by passing repetitions.

        Parameters
        ----------
        repetitions
            Initialize this Sequence directly with a list of repetitions
        normalized_grads : bool
            When this sequence is loaded from pulseq, this is set to `False`.
            The default of `True` flags this sequence for using normalized
            k-values, which is what is usually desired when building the
            sequence in mr0, using gradient steps of 1. If true, these
            gradients are then scaled to the phantom size on simulation.
            If false, no scaling happens and SI units are assumed.
        """
        super().__init__(repetitions)
        self.normalized_grads = normalized_grads

    def cuda(self) -> Sequence:
        """Move this sequence to the specified CUDA device and return it."""
        return Sequence([rep.cuda() for rep in self], self.normalized_grads)

    def cpu(self) -> Sequence:
        """Move this sequence to the CPU and return it."""
        return Sequence([rep.cpu() for rep in self], self.normalized_grads)

    @property
    def device(self) -> torch.device:
        """Return the sequence this pulse is stored on."""
        return self[0].device

    def clone(self) -> Sequence:
        """Return a deepcopy of self."""
        return Sequence([rep.clone() for rep in self], self.normalized_grads)

    def new_rep(self, event_count) -> Repetition:
        """Return a zeroed out repetition that is part of this ``Sequence``."""
        rep = Repetition.zero(event_count)
        self.append(rep)
        return rep

    def get_full_kspace(self) -> list[torch.Tensor]:
        """Compute the kspace trajectory produced by the gradient moments.

        This function relies on the values of ``Repetition.pulse_usage`` to
        determine which trajectory the sequence tries to achieve.

        The trajectory is 4-dimensional as it also includes dephasing time.

        Returns
        -------
        list[torch.Tensor]
            A tensor of shape (``event_count``, 4) for every repetition.
        """
        k_pos = torch.zeros(4, device=self.device)
        trajectory = []
        # Pulses with usage STORE store magnetisation and update this variable,
        # following excitation pulses will reset to stored instead of origin
        stored = torch.zeros(4, device=self.device)

        for rep in self:
            if rep.pulse.usage == PulseUsage.EXCIT:
                k_pos = stored
            elif rep.pulse.usage == PulseUsage.REFOC:
                k_pos = -k_pos
            elif rep.pulse.usage == PulseUsage.STORE:
                stored = k_pos

            rep_traj = k_pos + torch.cumsum(
                torch.cat([rep.gradm, rep.event_time[:, None]], 1),
                dim=0
            )
            k_pos = rep_traj[-1, :]
            trajectory.append(rep_traj)

        return trajectory

    def get_kspace(self) -> torch.Tensor:
        """Calculate the trajectory described by the signal of this sequence.

        This function returns only the kspace positions of the events that were
        actually measured (i.e. ``adc_usage > 0``) as one continuous tensor.
        The kspace includes the dephasing time as 4th dimension.

        Returns
        -------
        torch.Tensor
            Float tensor of shape (sample_count, 4)
        """
        # - Iterate over the full kspace and the sequence repetitions
        # - Mask the kspace to only retain samples that were measured
        # - Concatenate all repetitions and return the result
        return torch.cat([
            shot[rep.adc_usage > 0]
            for shot, rep in zip(self.get_full_kspace(), self)
        ])

    def get_contrast_mask(self, contrast: int) -> torch.Tensor:
        """Return a mask for a specific contrast as bool tensor.

        The returned tensor only contains measured events and is designed to be
        used together with ``get_kspace()`` or the simulated signal.

        Parameters
        ----------
        contrast : int
            The index for the contrast of which the signal mask is requested.

        Returns
        -------
        torch.Tensor
            The signal mask for the requested contrast as bool tensor

        Examples
        --------
        >>> signal = execute_graph(graph, seq, data)
        >>> kspace = seq.get_kspace()
        >>> mask = seq.get_contrast_mask(7)
        >>> contrast_reco = reco(signal[mask], kspace[mask])
        """
        return torch.cat(
            [rep.adc_usage[rep.adc_usage > 0] == contrast for rep in self]
        )

    def get_contrasts(self) -> list[int]:
        """Return a sorted list of all contrasts used by this ``Sequence``."""
        # flat list of all contrasts of all sequences
        tmp = [c for rep in self for c in rep.get_contrasts()]
        # Use a set to remove duplicates
        return sorted(list(set(tmp)))

    def shift_contrasts(self, offset: int):
        """Increment all offsets used by this sequence by ``offset``.

        Only operates on elements that are already larger than zero, so this
        function does not change which elements are measured. Modifies the
        sequence in-place, use :meth:`clone()` if you want to keep the original
        sequence as well.
        """
        for rep in self:
            rep.shift_contrasts(offset)

    def get_duration(self) -> float:
        """Calculate the total duration of self in seconds."""
        return sum(rep.event_time.sum().item() for rep in self)

    def get_adc_labels(self, name: str) -> torch.Tensor:
        """Return the values of a pulseq label for every measured ADC sample.

        The label is read from each event with ``adc_usage > 0`` and the
        result is concatenated across all repetitions, giving a 1-D
        ``int32`` tensor whose length equals the total number of measured
        samples in the sequence. Useful for reconstruction code that needs
        the per-sample ``lin`` / ``par`` / ``slc`` / … indices.

        Parameters
        ----------
        name : str
            Label name as used by pulseq: one of ``"slc"``, ``"seg"``,
            ``"rep"``, ``"avg"``, ``"set"``, ``"eco"``, ``"phs"``, ``"lin"``,
            ``"par"``, ``"acq"`` (counters), or ``"nav"``, ``"rev"``,
            ``"sms"``, ``"ref"``, ``"ima"``, ``"off"``, ``"noise"`` (flags,
            returned as ``0`` / ``1``).

        Raises
        ------
        KeyError
            If the sequence has no labels (i.e. it wasn't imported with the
            ``pulseq_rs`` backend) or the requested name doesn't exist.
        """
        chunks: list[torch.Tensor] = []
        for rep in self:
            if name not in rep.adc_labels:
                raise KeyError(
                    f"label {name!r} not available on this sequence; ensure "
                    f"it was imported with backend='pulseq_rs'"
                )
            mask = rep.adc_usage > 0
            chunks.append(rep.adc_labels[name][mask])
        if not chunks:
            return torch.zeros(0, dtype=torch.int32)
        return torch.cat(chunks)

    def get_label_changes(self, name: str) -> list[tuple[int, int]]:
        """Return repetition-wise changes of a pulseq label.

        Walks the sequence in order, collapses each repetition to the single
        value of ``name`` across all of its ADC samples, and emits a
        ``(rep_index, new_value)`` pair whenever that value differs from the
        previous one. The first repetition that carries an ADC is always
        emitted as the baseline.

        Repetitions without any ADC are skipped (they have no opinion on
        the label state). If a single repetition contains ADCs with
        different values for ``name``, this is treated as illegal — the
        intended use case is splitting the sequence on label changes, which
        requires one value per repetition.

        Parameters
        ----------
        name : str
            See :meth:`get_adc_labels`.

        Raises
        ------
        ValueError
            If any repetition contains multiple distinct values for ``name``
            among its ADC samples.
        KeyError
            If the label is not available on this sequence.
        """
        changes: list[tuple[int, int]] = []
        prev: Optional[int] = None
        for i, rep in enumerate(self):
            if name not in rep.adc_labels:
                raise KeyError(
                    f"label {name!r} not available on this sequence; ensure "
                    f"it was imported with backend='pulseq_rs'"
                )
            mask = rep.adc_usage > 0
            if not mask.any():
                continue
            vals = rep.adc_labels[name][mask]
            uniq = torch.unique(vals)
            if uniq.numel() != 1:
                raise ValueError(
                    f"label {name!r} has multiple values "
                    f"{uniq.tolist()} within repetition {i}; cannot collapse "
                    f"to a single per-rep value"
                )
            v = int(uniq.item())
            if v != prev:
                changes.append((i, v))
                prev = v
        return changes

    @classmethod
    def import_file(cls, file_name: str,
                    exact_trajectories: bool = True,
                    print_stats: bool = False,
                    default_shim: torch.Tensor = torch.asarray([[1, 0]], dtype=torch.float32),
                    ref_voltage: float = 300.0,
                    resolution: Optional[int] = None,
                    output_dir: Optional[str] = None,
                    backend: Literal["pydisseqt", "pulseq_rs"] = "pydisseqt",
                    larmor_hz: Optional[float] = None,
                    fov_scale: Optional[float] = None,
                    fov_pos: Optional[tuple[float, float, float]] = None,
                    fov_rot: Optional[tuple[float, float, float, float]] = None,
                    soft_delays: Optional[dict[str, float]] = None,
                    ) -> Sequence:
        """Import a pulseq .seq file or a bundle of .dsv files.

        Parameters
        ----------
        file_name : str
            The path to the pulseq .seq file that is imported or the .dsv file
            name stem (the part before _ADC.dsv, _GRX.dsv, etc...)
        exact_trajectories : bool
            If true, the gradients before and after the ADC blocks are imported
            exactly. If false, they are summed into a single event. Depending
            on the sequence, simulation might be faster if set to false, but
            the simulated diffusion changes with simplified trajectoreis.
        print_stats : bool
            If set to true, additional information is printed during import
        default_shim : Tensor
            The shim_array used for pulses that do not specify it themselves.
        ref_voltage : float
            If a .dsv file is imported, this is used to convert pulses from
            volts to angles. Make sure to use 1Tx systems for dsv simulation, 
            otherwise the ref_voltage will not match to the concerted flip angle!!
            A 1 ms block pulse of ref_voltage is a 180 ° flip
        resolution : int | None
            .dsv files do not contain data for the number of ADC samples.
            This is used to specify the number of samples per ADC block.
            If false, uses the .dsv time step as ADC dwell time
        backend : "pydisseqt" | "pulseq_rs"
            Parser used to read the .seq file. ``"pydisseqt"`` (default) is the
            legacy path. ``"pulseq_rs"`` uses the Rust pulseq-rs parser via the
            bundled extension; .dsv files always fall back to pydisseqt.
        larmor_hz, fov_scale, fov_pos, fov_rot, soft_delays
            Forwarded to the pulseq-rs interpreter when ``backend="pulseq_rs"``;
            ignored for the pydisseqt backend.

        Returns
        -------
        mr0.Sequence
            The imported file as mr0 Sequence
        """
        if not file_name.endswith(".seq") and backend == "pulseq_rs":
            warnings.warn(
                "pulseq_rs backend only supports .seq files; falling back "
                "to pydisseqt for DSV input.",
                stacklevel=2,
            )
            backend = "pydisseqt"

        if backend == "pulseq_rs":
            return cls._import_pulseq_rs(
                file_name, exact_trajectories, print_stats, default_shim,
                larmor_hz=larmor_hz, fov_scale=fov_scale,
                fov_pos=fov_pos, fov_rot=fov_rot, soft_delays=soft_delays,
            )
        elif backend == "pydisseqt":
            return cls._import_pydisseqt(
                file_name, exact_trajectories, print_stats, default_shim,
                ref_voltage, resolution,
            )
        else:
            raise ValueError(
                f"unknown backend {backend!r}; expected 'pydisseqt' or 'pulseq_rs'"
            )

    @classmethod
    def _import_pydisseqt(cls, file_name: str,
                          exact_trajectories: bool,
                          print_stats: bool,
                          default_shim: torch.Tensor,
                          ref_voltage: float,
                          resolution: Optional[int],
                          ) -> Sequence:
        start = time()
        if file_name.endswith(".seq"):
            parser = pydisseqt.load_pulseq(file_name)
        else:
            #try import of from dsv2pulseq import read_dsv
            try:
                from dsv2pulseq import read_dsv
                import os
            except ImportError:
                raise ImportError(
                    "To import .dsv files, please install the dsv2pulseq package"
                )
            seq_temp = read_dsv(file_name, ref_volt=ref_voltage)
            os.makedirs(output_dir, exist_ok=True)
            seq_name = os.path.basename(file_name) + '_dsv2seq.seq'
            seq_dsv_path = os.path.join(output_dir, seq_name)
            seq_pulseq_dsv = seq_temp.make_pulseq_sequence(seq_dsv_path)
            print(f"Saved .seq file to: {seq_dsv_path}")
            parser = pydisseqt.load_pulseq(seq_dsv_path)
            #parser = pydisseqt.load_dsv(file_name, ref_voltage, resolution)

        if print_stats:
            print(f"Importing the .seq file took {time() - start} s")
        start = time()
        seq = cls(normalized_grads=False)

        # We should do at least _some_ guess for the pulse usage
        def pulse_usage(angle: float) -> PulseUsage:
            if abs(angle) < 100 * np.pi / 180:
                return PulseUsage.EXCIT
            else:
                return PulseUsage.REFOC

        # Get time points of all pulses
        pulses = []  # Contains pairs of (pulse_start, pulse_end)
        tmp = parser.encounter("rf", 0.0)
        while tmp is not None:
            pulses.append(tmp)
            tmp = parser.encounter("rf", tmp[1])  # pulse_end

        # Iterate over all repetitions (we ignore stuff before the first pulse)
        for i in range(len(pulses)):
            # Calculate repetition start and end time based on pulse centers
            rep_start = (pulses[i][0] + pulses[i][1]) / 2
            if i + 1 < len(pulses):
                rep_end = (pulses[i + 1][0] + pulses[i + 1][1]) / 2
            else:
                rep_end = parser.duration()

            # Fetch additional data needed for building the mr0 sequence
            pulse = parser.integrate_one(pulses[i][0], pulses[i][1]).pulse
            shim = parser.sample_one(rep_start).pulse.shim
            
            # load pulse frequency-offset needed for potential treatment off off-resonance
            frequency = parser.sample_one(rep_start).pulse.amplitude # this only works for block pulses! should be removed in the future
            frequency_offset = parser.sample_one(rep_start).pulse.frequency

            adcs = parser.events("adc", rep_start, rep_end)

            # To simulate diffusion, we want to more exactly simulate gradient
            # trajectories between pulses and the ADC block
            if exact_trajectories:
                # First and last timepoint in repetition with a gradient sample
                first = pulses[i][1]
                last = (pulses[i + 1][0] if i + 1 < len(pulses) else rep_end)
                eps = 1e-6  # Move a bit past start / end of repetition
                # Gradient samples can be duplicated between x, y, z.
                # They are deduplicated after rounding to `precision` digits
                precision = 6

                if len(adcs) > 0:
                    grad_before = sorted(set([round(t, precision) for t in (
                        parser.events("grad x", first + eps, adcs[0] - eps) +
                        parser.events("grad y", first + eps, adcs[0] - eps) +
                        parser.events("grad z", first + eps, adcs[0] - eps)
                    )]))
                    grad_after = sorted(set([round(t, precision) for t in (
                        parser.events("grad x", adcs[-1] + eps, last - eps) +
                        parser.events("grad y", adcs[-1] + eps, last - eps) +
                        parser.events("grad z", adcs[-1] + eps, last - eps)
                    )]))
                    # Last repetition: no pulse, ignore [last, rep_end]
                    if i == len(pulses) - 1:
                        abs_times = [rep_start, first] + grad_before + adcs
                    else:
                        abs_times = ([rep_start, first] + grad_before + adcs +
                                     grad_after + [last, rep_end])
                    # Index of first ADC: -1 - we count spans between indices
                    adc_start = 2 + len(grad_before) - 1
                else:
                    grad = sorted(set([round(t, precision) for t in (
                        parser.events("grad x", first + eps, last - eps) +
                        parser.events("grad y", first + eps, last - eps) +
                        parser.events("grad z", first + eps, last - eps)
                    )]))
                    # Last repetition: no pulse, ignore [last, rep_end]
                    if i == len(pulses) - 1:
                        abs_times = [rep_start, first] + grad
                    else:
                        abs_times = [rep_start, first] + grad + [last, rep_end]
                    adc_start = None
            else:
                # No gradient samples, only adc and one final to the next pulse
                abs_times = [rep_start] + adcs + [rep_end]
                adc_start = 0

            event_count = len(abs_times) - 1
            samples = parser.sample(adcs)
            moments = parser.integrate(abs_times)

            if print_stats:
                print(
                    f"Rep. {i + 1}: {event_count} samples, of which "
                    f"{len(adcs)} are ADC (starting at {adc_start})"
                )

            # -- Now we build the mr0 Sequence repetition --
            rep = seq.new_rep(event_count)
            rep.event_time[:] = torch.as_tensor(np.diff(abs_times))
            
            rep.pulse.angle = pulse.angle
            rep.pulse.phase = pulse.phase                       
            
            # provide frequency and frequency-offset to pulse object needed for potential treatment off off-resonance
            rep.pulse.pulse_freq = 2*torch.pi * frequency # rad/s # may only work for block-pulses
            rep.pulse.freq_offset = frequency_offset      # Hz
            if rep.pulse.freq_offset != 0:
                 rep.pulse.off_res = True
            
            rep.pulse.usage = pulse_usage(pulse.angle)
            if shim is None:
                rep.pulse.shim_array = default_shim.clone()
            else:
                rep.pulse.shim_array = torch.as_tensor(shim)

            rep.gradm[:, 0] = torch.as_tensor(moments.gradient.x)
            rep.gradm[:, 1] = torch.as_tensor(moments.gradient.y)
            rep.gradm[:, 2] = torch.as_tensor(moments.gradient.z)

            if adc_start is not None:
                phases = np.pi / 2 - torch.as_tensor(samples.adc.phase)
                rep.adc_usage[adc_start:adc_start + len(adcs)] = 1
                rep.adc_phase[adc_start:adc_start + len(adcs)] = phases

        if print_stats:
            print(f"Converting the sequence to mr0 took {time() - start} s")
        return seq

    """credits @ Claude"""
    @classmethod
    def _import_pulseq_rs(cls, file_name: str,
                          exact_trajectories: bool,
                          print_stats: bool,
                          default_shim: torch.Tensor,
                          *,
                          larmor_hz: Optional[float] = None,
                          fov_scale: Optional[float] = None,
                          fov_pos: Optional[tuple[float, float, float]] = None,
                          fov_rot: Optional[tuple[float, float, float, float]] = None,
                          soft_delays: Optional[dict[str, float]] = None,
                          ) -> Sequence:
        from . import _prepass  # local import to avoid touching module init order

        start = time()
        interp = _prepass.load_pulseq_rs(
            file_name,
            larmor_hz=larmor_hz,
            fov_scale=fov_scale,
            fov_pos=fov_pos,
            fov_rot=fov_rot,
            soft_delays=soft_delays,
        )
        if print_stats:
            print(f"Importing the .seq file took {time() - start} s")
        start = time()

        blocks = list(interp.blocks)
        duration = interp.duration

        # Cache shape-times arrays per block so events_axis_in() doesn't keep
        # rebuilding them through the FFI.
        block_starts = [b.start for b in blocks]
        block_ends = [b.start + b.duration for b in blocks]
        grad_breakpoints = {  # absolute times of every breakpoint per (block, axis)
            "x": [None] * len(blocks),
            "y": [None] * len(blocks),
            "z": [None] * len(blocks),
        }
        for j, b in enumerate(blocks):
            for axis in ("x", "y", "z"):
                g = getattr(b, "g" + axis)
                if g is None:
                    continue
                t_off = b.start + g.delay
                grad_breakpoints[axis][j] = [t_off + t for t in g.shape_times()]

        adc_times_per_block = [None] * len(blocks)
        adc_phases_per_block = [None] * len(blocks)
        # Labels live at the block (ADC) level, so one dict per ADC block is
        # enough — every sample inside that block shares the same snapshot.
        adc_labels_per_block: list[Optional[dict[str, int]]] = [None] * len(blocks)
        for j, b in enumerate(blocks):
            if b.adc is None:
                continue
            adc_times_per_block[j] = [b.start + t for t in b.adc.sample_times()]
            adc_phases_per_block[j] = list(b.adc.sample_phases())
            adc_labels_per_block[j] = b.adc.labels()

        label_names: list[str] = []
        for lbl in adc_labels_per_block:
            if lbl is not None:
                label_names = list(lbl.keys())
                break

        def events_axis_in(t0: float, t1: float, axis: str) -> list[float]:
            out: list[float] = []
            bp = grad_breakpoints[axis]
            for j in range(len(blocks)):
                if block_starts[j] >= t1 or block_ends[j] <= t0:
                    continue
                times = bp[j]
                if times is None:
                    continue
                for t in times:
                    if t0 <= t <= t1:
                        out.append(t)
            return out

        def adcs_in(t0: float, t1: float) -> tuple[
            list[float], list[float], list[dict[str, int]]
        ]:
            times_out: list[float] = []
            phases_out: list[float] = []
            labels_out: list[dict[str, int]] = []
            for j in range(len(blocks)):
                if block_starts[j] >= t1 or block_ends[j] <= t0:
                    continue
                ts = adc_times_per_block[j]
                if ts is None:
                    continue
                ph = adc_phases_per_block[j]
                lbl = adc_labels_per_block[j]
                assert ph is not None and lbl is not None
                for k, t in enumerate(ts):
                    if t0 <= t <= t1:
                        times_out.append(t)
                        phases_out.append(ph[k])
                        labels_out.append(lbl)
            return times_out, phases_out, labels_out

        def integrate_axis(axis: str, t0: float, t1: float) -> float:
            if t1 <= t0:
                return 0.0
            moment = 0.0
            for j in range(len(blocks)):
                if block_starts[j] >= t1 or block_ends[j] <= t0:
                    continue
                g = getattr(blocks[j], "g" + axis)
                if g is None:
                    continue
                lo = max(t0, block_starts[j]) - block_starts[j]
                hi = min(t1, block_ends[j]) - block_starts[j]
                if hi > lo:
                    moment += g.integrate(lo, hi)
            return moment

        seq = cls(normalized_grads=False)

        # Discover RF pulses: each RF-bearing block produces one (start, end).
        pulses: list[tuple[float, float, int]] = []  # (pulse_start, pulse_end, block_idx)
        for j, b in enumerate(blocks):
            if b.rf is None:
                continue
            ps = b.start + b.rf.delay
            pe = ps + b.rf.shape_duration
            pulses.append((ps, pe, j))

        for i in range(len(pulses)):
            ps, pe, b_idx = pulses[i]
            rep_start = 0.5 * (ps + pe)
            if i + 1 < len(pulses):
                rep_end = 0.5 * (pulses[i + 1][0] + pulses[i + 1][1])
            else:
                rep_end = duration

            rf = blocks[b_idx].rf
            angle, phase = rf.integrate(0.0, rf.shape_duration)
            if rf.rf_use == "excitation":
                rf_usage = PulseUsage.EXCIT
            elif rf.rf_use == "saturation":
                rf_usage = PulseUsage.FATSAT
            elif rf.rf_use == "refocusing":
                rf_usage = PulseUsage.REFOC
            else:
                rf_usage = PulseUsage.UNDEF
            
            # TODO: I do not know if this is the expected format!
            rf_grad = torch.zeros(3, dtype=torch.float32)
            rf_hasgrad = False
            # BUG: this code is too simple - amp is the amplitude scaling and
            # ignores the shape. Use the following methods on gx/gy/gz instead:
            # - .shape_times() and .amp * .shape_amp(): return the full shape
            # - .sample(t): return the amplitude at one time point
            # - .integrate(t0, t1): return the gradient moment over the given period
            if blocks[b_idx].gx is not None:
                rf_hasgrad = True
                rf_grad[0] = blocks[b_idx].gx.amp
            if blocks[b_idx].gy is not None:
                rf_hasgrad = True
                rf_grad[1] = blocks[b_idx].gy.amp
            if blocks[b_idx].gz is not None:
                rf_hasgrad = True
                rf_grad[2] = blocks[b_idx].gz.amp

            # Shim handling: pulseq-rs always returns a list, with [(1.0, 0.0)]
            # meaning "no shim" - in that case fall back to the default.
            shims = rf.shims
            if (len(shims) == 1
                    and abs(shims[0][0] - 1.0) < 1e-12
                    and abs(shims[0][1]) < 1e-12):
                shim_arr = default_shim.clone()
            else:
                shim_arr = torch.as_tensor(shims, dtype=torch.float32)

            adcs, adc_phases, adc_label_snapshots = adcs_in(rep_start, rep_end)

            if exact_trajectories:
                first = pe
                last = pulses[i + 1][0] if i + 1 < len(pulses) else rep_end
                eps = 1e-6
                precision = 6

                if len(adcs) > 0:
                    grad_before = sorted({round(t, precision) for t in (
                        events_axis_in(first + eps, adcs[0] - eps, "x") +
                        events_axis_in(first + eps, adcs[0] - eps, "y") +
                        events_axis_in(first + eps, adcs[0] - eps, "z")
                    )})
                    grad_after = sorted({round(t, precision) for t in (
                        events_axis_in(adcs[-1] + eps, last - eps, "x") +
                        events_axis_in(adcs[-1] + eps, last - eps, "y") +
                        events_axis_in(adcs[-1] + eps, last - eps, "z")
                    )})
                    if i == len(pulses) - 1:
                        abs_times = [rep_start, first] + grad_before + adcs
                    else:
                        abs_times = ([rep_start, first] + grad_before + adcs +
                                     grad_after + [last, rep_end])
                    adc_start = 2 + len(grad_before) - 1
                else:
                    grad = sorted({round(t, precision) for t in (
                        events_axis_in(first + eps, last - eps, "x") +
                        events_axis_in(first + eps, last - eps, "y") +
                        events_axis_in(first + eps, last - eps, "z")
                    )})
                    if i == len(pulses) - 1:
                        abs_times = [rep_start, first] + grad
                    else:
                        abs_times = [rep_start, first] + grad + [last, rep_end]
                    adc_start = None
            else:
                abs_times = [rep_start] + adcs + [rep_end]
                adc_start = 0

            event_count = len(abs_times) - 1

            if print_stats:
                print(
                    f"Rep. {i + 1}: {event_count} samples, of which "
                    f"{len(adcs)} are ADC (starting at {adc_start})"
                )

            mom_x = np.empty(event_count, dtype=np.float64)
            mom_y = np.empty(event_count, dtype=np.float64)
            mom_z = np.empty(event_count, dtype=np.float64)
            for k in range(event_count):
                t0 = abs_times[k]
                t1 = abs_times[k + 1]
                mom_x[k] = integrate_axis("x", t0, t1)
                mom_y[k] = integrate_axis("y", t0, t1)
                mom_z[k] = integrate_axis("z", t0, t1)

            rep = seq.new_rep(event_count)
            rep.pulse.angle = torch.as_tensor(angle)
            rep.pulse.phase = torch.as_tensor(phase)
            rep.pulse.usage = rf_usage
            rep.pulse.shim_array = shim_arr
            rep.pulse.freq_offset = 2*torch.pi * rf.freq
            rep.pulse.duration = rf.shape_duration
            rep.pulse.grad = rf_grad
            if rep.pulse.freq_offset != 0 or rf_hasgrad:
                 rep.pulse.off_res = True

            rep.event_time[:] = torch.as_tensor(np.diff(abs_times))
            rep.gradm[:, 0] = torch.as_tensor(mom_x)
            rep.gradm[:, 1] = torch.as_tensor(mom_y)
            rep.gradm[:, 2] = torch.as_tensor(mom_z)

            if adc_start is not None:
                phases_t = np.pi / 2 - torch.as_tensor(adc_phases)
                rep.adc_usage[adc_start:adc_start + len(adcs)] = 1
                rep.adc_phase[adc_start:adc_start + len(adcs)] = phases_t

            if label_names and adc_label_snapshots and adc_start is not None:
                for name in label_names:
                    t = torch.zeros(event_count, dtype=torch.int32)
                    for k, snap in enumerate(adc_label_snapshots):
                        t[adc_start + k] = snap[name]
                    rep.adc_labels[name] = t

        if print_stats:
            print(f"Converting the sequence to mr0 took {time() - start} s")
        return seq

    @classmethod
    def from_seq_file(cls, file_name: str) -> Sequence:
        """Import a sequence from a pulseq .seq file.

        # This function is deprecated, use `Sequence.import_file` instead

        The importer currently minimizes the amount of used `mr0` sequence
        events. This can be problematic for diffusion weighted sequences,
        because gradients that are not directly measured by ADC samples can be
        removed from the sequence, even if they are important for the targeted
        contrast.

        Parameters
        ----------
        file_name : str
            Path to the imported .seq file

        Returns
        -------
        Sequence
            Imported sequence, converted to MRzero
        """
        raise DeprecationWarning(
            "WARNING: Use of deprecated Sequence.from_seq_file,"
            "use Sequence.import_file instead"
        )
        seq = Sequence(normalized_grads=False)
        for tmp_rep in intermediate(PulseqFile(file_name)):
            rep = seq.new_rep(tmp_rep[0])
            rep.pulse.angle = torch.as_tensor(
                tmp_rep[1].angle, dtype=torch.float32)
            rep.pulse.phase = torch.as_tensor(
                tmp_rep[1].phase, dtype=torch.float32)
            rep.pulse.shim_array = torch.as_tensor(
                tmp_rep[1].shim_array, dtype=torch.float32)

            if rep.pulse.angle > 100 * torch.pi/180:
                rep.pulse.usage = PulseUsage.REFOC
            else:
                rep.pulse.usage = PulseUsage.EXCIT

            # to combine multiple blocks into one repetition, we need to keep
            # track of the current event offset
            i = 0
            for block in tmp_rep[2]:
                if isinstance(block, Spoiler):
                    rep.event_time[i] = block.duration
                    rep.gradm[i, :] = torch.tensor(block.gradm)
                    i += 1
                else:
                    assert isinstance(block, Adc)
                    num = len(block.event_time)
                    rep.event_time[i:i+num] = torch.tensor(block.event_time)
                    rep.gradm[i:i+num, :] = torch.tensor(block.gradm)
                    rep.adc_phase[i:i+num] = torch.pi/2 - block.phase
                    rep.adc_usage[i:i+num] = 1
                    i += num
            assert i == tmp_rep[0]
        return seq

    def plot_kspace_trajectory(self,
                               figsize: tuple[float, float] = (5, 5),
                               plotting_dims: str = 'xy',
                               plot_timeline: bool = True
                               ):
        """Plot the kspace trajectory produced by self.

        Parameters
        ----------
        kspace : list[Tensor]
            The kspace as produced by ``Sequence.get_full_kspace()``
        figsize : (float, float), optional
            The size of the plotted matplotlib figure.
        plotting_dims : string, optional
            String defining what is plotted on the x and y axis ('xy' 'zy' ...)
        plot_timeline : bool, optional
            Plot a second subfigure with the gradient components per-event.
        """
        assert len(plotting_dims) == 2
        assert plotting_dims[0] in ['x', 'y', 'z']
        assert plotting_dims[1] in ['x', 'y', 'z']
        dim_map = {'x': 0, 'y': 1, 'z': 2}

        # TODO: We could (optionally) plot which contrast a sample belongs to,
        # currently we only plot if it is measured or not

        kspace = self.get_full_kspace()
        adc_mask = [rep.adc_usage > 0 for rep in self]

        cmap = plt.get_cmap('rainbow')
        plt.figure(figsize=figsize)
        if plot_timeline:
            plt.subplot(211)
        for i, (rep_traj, mask) in enumerate(zip(kspace, adc_mask)):
            kx = rep_traj[:, dim_map[plotting_dims[0]]]
            ky = rep_traj[:, dim_map[plotting_dims[1]]]

            plt.plot(kx, ky, c=cmap(i / len(kspace)))
            plt.plot(kx[mask], ky[mask], 'r.')
            plt.plot(kx[~mask], ky[~mask], 'k.')
        plt.xlabel(f"$k_{plotting_dims[0]}$")
        plt.ylabel(f"$k_{plotting_dims[1]}$")
        plt.grid()

        if plot_timeline:
            plt.subplot(212)
            event = 0
            for i, rep_traj in enumerate(kspace):
                x = torch.arange(event, event + rep_traj.shape[0], 1)
                event += rep_traj.shape[0]

                if i == 0:
                    plt.plot(x, rep_traj[:, 0], c='r', label="$k_x$")
                    plt.plot(x, rep_traj[:, 1], c='g', label="$k_y$")
                    plt.plot(x, rep_traj[:, 2], c='b', label="$k_z$")
                else:
                    plt.plot(x, rep_traj[:, 0], c='r', label="_")
                    plt.plot(x, rep_traj[:, 1], c='g', label="_")
                    plt.plot(x, rep_traj[:, 2], c='b', label="_")
            plt.xlabel("Event")
            plt.ylabel("Gradient Moment")
            plt.legend()
            plt.grid()

        plt.show()


def chain(*sequences: Sequence, oneshot: bool = False) -> Sequence:
    """Chain multiple sequences into one.

    This function modifies the contrast of the sequences so that they don't
    overlap by shifting them by the maximum contrast of the previous sequence.

    Parameters
    ----------
    *sequences : Sequence
        Arbitrary number of sequences that will be chained

    oneshot : bool
        If true, contrasts of individual sequences are not shifted, which means
        that duplicates are possible.

    Returns
    -------
    Sequence
        A single sequence

    Examples
    --------
    >>> seq_a = build_a_seq()
    ... seq_b = build_another_seq()
    ... seq_ab = chain(seq_a, seq_b)
    >>> seq_a.get_contrasts()
    [1, 3]
    >>> seq_b.get_contrasts()
    [2]
    >>> seq_ab.get_contrasts()
    [1, 3, 5]
    """
    combined = Sequence()
    contrast_offset = 0

    for seq in sequences:
        temp = seq.clone()
        temp.shift_contrasts(contrast_offset)
        if not oneshot:
            contrast_offset = max(temp.get_contrasts())
        for rep in temp:
            combined.append(rep)

    return combined
