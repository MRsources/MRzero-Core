"""Classes needed for sequence definition."""

from __future__ import annotations
import pickle
import torch
from enum import Enum
from typing import Iterable, Union
from MRzeroCore import util


class PulseUsage(Enum):
    """Enumerates all pulse usages, needed for reconstruction."""

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
    selective : bool
        Specifies if this pulse should be slice-selective (z-direction)
    """

    def __init__(
        self,
        usage: PulseUsage,
        angle: Union[float, torch.Tensor],
        phase: Union[float, torch.Tensor],
        selective: bool,
    ) -> None:
        """Create a Pulse instance.

        If ``angle`` and ``phase`` are floats they will be converted to
        torch tensors on the device given by util.get_device().
        """
        self.usage = usage
        self.angle = torch.as_tensor(angle, dtype=torch.float,
                                     device=util.get_device())
        self.phase = torch.as_tensor(phase, dtype=torch.float,
                                     device=util.get_device())
        self.selective = selective

    @classmethod
    def zero(cls):
        """Create a pulse with zero flip and phase."""
        return cls(PulseUsage.UNDEF, 0.0, 0.0, True)

    def clone(self) -> Pulse:
        """Return a cloned copy of self."""
        return Pulse(
            self.usage, self.angle.clone(), self.phase.clone(), self.selective)


class Repetition:
    """A ``Repetition`` starts with a RF pulse and ends just before the next.

    Attributes
    ----------
    pulse : Pulse
        The RF pulse at the beginning of this ``Repetition``
    event_time : torch.Tensor
        Duration of each event (seconds)
    gradm : torch.Tensor
        Gradient moment of every event, shape (``event_count``, 3)
    adc_phase : torch.Tensor
        Float tensor describing the adc rotation, shape (``event_count``, 3)
    adc_usage: torch.Tensor
        Int tensor specifying which contrast a sample belongs to, shape
        (``event_count``, 3). Samples with ``adc_usage <= 0`` will not be
        measured. For single contrast sequences, just use 0 or 1.
    event_count : int
        Number of events in this ``Repetition``
    """

    def __init__(
        self,
        pulse: Pulse,
        event_time: torch.Tensor,
        gradm: torch.Tensor,
        adc_phase: torch.Tensor,
        adc_usage: torch.Tensor
    ) -> None:
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

        self.event_time = util.set_device(event_time)
        self.gradm = util.set_device(gradm)
        self.adc_phase = util.set_device(adc_phase)
        self.adc_usage = util.set_device(adc_usage)

    @classmethod
    def zero(cls, event_count: int) -> Repetition:
        """Create a ``Repetition`` instance with everything set to zero."""
        return cls(
            Pulse.zero(),
            torch.zeros(event_count, dtype=torch.float),
            torch.zeros((event_count, 3), dtype=torch.float),
            torch.zeros(event_count, dtype=torch.float),
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
        return sorted(torch.unique(self.adc_usage).tolist())

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

    def __init__(self, repetitions: Iterable[Repetition] = []) -> None:
        """Create a ``Sequence`` instance by passing repetitions."""
        super().__init__(repetitions)

    def clone(self) -> Sequence:
        """Return a deepcopy of self."""
        return Sequence(rep.clone() for rep in self)

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
        k_pos = util.set_device(torch.zeros(4))
        trajectory = []
        # Pulses with usage STORE store magnetisation and update this variable,
        # following excitation pulses will reset to stored instead of origin
        stored = util.set_device(torch.zeros(4))

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
        used together with ``get_kspace()`` or the simulated signal:
        ```
        signal = execute_graph(graph, seq, data)
        kspace = seq.get_kspace()
        mask = seq.get_contrast_mask(7)
        contrast_reco = reco(signal[mask], kspace[mask])
        ```
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

    def save(self, file_name):
        """Save self to the given file.

        Can be loaded again by using :meth:`load`.

        Parameters
        ----------
        file_name : str
            The directory & file name the ``Sequence`` will be written to.
        """
        with open(file_name, 'wb') as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, file_name) -> Sequence:
        """Create a ``Sequence`` instance by loading it from a file.

        The file is expected to be created by using :meth:`save`.

        Parameters
        ----------
        file_name : str
            The directory & file name the ``Sequence`` will be read from.
        """
        with open(file_name, 'rb') as file:
            return pickle.load(file)


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
