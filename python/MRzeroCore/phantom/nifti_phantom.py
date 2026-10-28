# The BIfTI phantom config is implemented by the official `bifti` package.
# This module only re-exports it under the names MRzero used before, so that
# existing code like `mr0.NiftiPhantom.load(...)` keeps working.
from bifti import (
    BiftiPhantom,
    BiftiTissue,
    NiftiMapping,
    NiftiRef,
    Patient,
    PhantomSystem,
    PhantomUnits,
    ResliceTo,
)

NiftiPhantom = BiftiPhantom
NiftiTissue = BiftiTissue
ResliceConfig = ResliceTo
