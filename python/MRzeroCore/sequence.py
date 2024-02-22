from __future__ import annotations
from time import time
import torch
import numpy as np
from enum import Enum
from typing import Iterable
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
        shim_array: torch.Tensor,
        selective: bool,
    ):
        """Create a Pulse instance."""
        self.usage = usage
        self.angle = angle
        self.phase = phase
        self.shim_array = shim_array
        self.selective = selective

    def cpu(self) -> Pulse:
        """Move this pulse to the CPU and return it."""
        return Pulse(
            self.usage,
            torch.as_tensor(self.angle, dtype=torch.float32).cpu(),
            torch.as_tensor(self.phase, dtype=torch.float32).cpu(),
            torch.as_tensor(self.shim_array, dtype=torch.float32).cpu(),
            self.selective
        )

    def cuda(self, device: int | None = None) -> Pulse:
        """Move this pulse to the specified CUDA device and return it."""
        return Pulse(
            self.usage,
            torch.as_tensor(self.angle, dtype=torch.float32).cuda(device),
            torch.as_tensor(self.phase, dtype=torch.float32).cuda(device),
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
            torch.asarray([[1, 0]], dtype=torch.float32),
            True
        )

    def clone(self) -> Pulse:
        """Return a cloned copy of self."""
        return Pulse(
            self.usage,
            self.angle.clone(),
            self.phase.clone(),
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
            [rep.adc_usage[rep.adc_usage != 0] == contrast for rep in self]
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

    @classmethod
    def import_file(cls, file_name: str,
                    exact_trajectories: bool = True,
                    print_stats: bool = False
                    ) -> Sequence:
        """Import a pulseq .seq file.

        Parameters
        ----------
        file_name : str
            The path to the file that is imported
        exact_trajectories : bool
            If true, the gradients before and after the ADC blocks are imported
            exactly. If false, they are summed into a single event. Depending
            on the sequence, simulation might be faster if set to false, but
            the simulated diffusion changes with simplified trajectoreis.
        print_stats : bool
            If set to true, additional information is printed during import

        Returns
        -------
        mr0.Sequence
            The imported file as mr0 Sequence
        """
        start = time()
        parser = pydisseqt.load_pulseq(file_name)
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
            rep.pulse.angle = pulse.angle
            rep.pulse.phase = pulse.phase
            rep.pulse.usage = pulse_usage(pulse.angle)

            rep.event_time[:] = torch.as_tensor(np.diff(abs_times))

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
        print(
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
