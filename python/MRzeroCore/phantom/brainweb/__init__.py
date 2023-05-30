# The brainweb script contains tools for handling with Brainweb itself.
# Because the API is not final yet, we just expose the very basic
# download functionality here.
import os
from . import brainweb
from .brainweb import SUBJECTS


def generate_phantom(
        subject: int,
        output: str = os.path.join(os.getcwd(), "brainweb"),
        scale: int = 1
        ):
    """Generate an mr0 phantom from BrainWeb data.

    Downloads segmentation data from https://brainweb.bic.mni.mcgill.ca/
    and fills it with literature T1, T2, ... data to generate an mr0 phantom.

    Parameters
    ----------
    subject : int
        Subject ID. Use `mr0.brainweb.SUBJECTS` for a list of valid ID.
    output : str
        Path to output folder for the resulting phantoms ('subjectXX.npz').
        Will also be used to cache the downloaded segmentation data.
    scale : int
        Amount of downsampling, e.g., a value of 2 will produce half-res data.
    """
    brainweb.CACHE_PATH = output
    phantom = brainweb.Phantom.load(subject, scale)
    phantom.save()
