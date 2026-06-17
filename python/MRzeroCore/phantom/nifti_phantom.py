from dataclasses import dataclass, field
from typing import Literal, Any
from pathlib import Path


@dataclass
class PhantomUnits:
    gyro: Literal["MHz/T"]
    B0: Literal["T"]
    T1: Literal["s"]
    T2: Literal["s"]
    T2dash: Literal["s"]
    ADC: Literal["10^-3 mm^2/s"]
    dB0: Literal["Hz"]
    B1_tx: Literal["rel"]
    B1_rx: Literal["rel"]

    @classmethod
    def default(cls):
        return cls(
            gyro="MHz/T",
            B0="T",
            T1="s",
            T2="s",
            T2dash="s",
            ADC="10^-3 mm^2/s",
            dB0="Hz",
            B1_tx="rel",
            B1_rx="rel",
        )

    @classmethod
    def from_dict(cls, config: dict[str, str]):
        # Currently this is only for documentation and no other units than the
        # default units are supported. This definetly can change in the future
        # but the implementation has no priority right now
        default = cls.default()
        assert default.to_dict() == config, "Only default units are supported for now"
        return default

    def to_dict(self) -> dict[str, str]:
        return {
            "gyro": self.gyro,
            "B0": self.B0,
            "T1": self.T1,
            "T2": self.T2,
            "T2'": self.T2dash,
            "ADC": self.ADC,
            "dB0": self.dB0,
            "B1+": self.B1_tx,
            "B1-": self.B1_rx,
        }


@dataclass
class PhantomSystem:
    gyro: float
    B0: float

    @classmethod
    def from_dict(cls, config: dict[str, float]):
        return cls(**config)

    def to_dict(self) -> dict[str, float]:
        return {"gyro": self.gyro, "B0": self.B0}


@dataclass
class ResliceConfig:
    """Target resampling grid declared inside the phantom JSON."""
    resolution: list  # [nx, ny, nz] — output voxel count
    affine: list      # 3×4 NIfTI-style affine in mm (rotation×voxel_size | origin)

    @classmethod
    def from_dict(cls, config: dict):
        return cls(resolution=config["resolution"], affine=config["affine"])

    def to_dict(self) -> dict:
        return {"resolution": self.resolution, "affine": self.affine}


@dataclass
class NiftiRef:
    file_name: Path
    tissue_index: int

    @classmethod
    def parse(cls, config: str):
        import re

        regex = re.compile(r"(?P<file>.+?)\[(?P<idx>\d+)\]$")
        m = regex.match(config)
        if not m:
            raise ValueError("Invalid file_ref", m)
        return cls(file_name=Path(m.group("file")), tissue_index=int(m.group("idx")))

    def to_str(self) -> str:
        return f"{self.file_name}[{self.tissue_index}]"


@dataclass
class NiftiMapping:
    file: NiftiRef
    func: str

    @classmethod
    def parse(cls, config: dict[str, Any]):
        return cls(file=NiftiRef.parse(config["file"]), func=config["func"])

    def to_dict(self) -> dict[str, Any]:
        return {"file": self.file.to_str(), "func": self.func}


@dataclass
class NiftiTissue:
    density: NiftiRef
    T1: float | NiftiRef | NiftiMapping
    T2: float | NiftiRef | NiftiMapping
    T2dash: float | NiftiRef | NiftiMapping
    ADC: float | NiftiRef | NiftiMapping
    dB0: float | NiftiRef | NiftiMapping
    B1_tx: list[float | NiftiRef | NiftiMapping]
    B1_rx: list[float | NiftiRef | NiftiMapping]

    @classmethod
    def default(cls, density: NiftiRef):
        return cls.from_dict({"density": density})

    @classmethod
    def from_dict(cls, config: dict[str, Any]):
        def parse_prop(prop):
            if isinstance(prop, (float, int)):
                return float(prop)
            elif isinstance(prop, str):
                return NiftiRef.parse(prop)
            else:
                return NiftiMapping.parse(prop)

        return cls(
            density=NiftiRef.parse(config["density"]),
            T1=parse_prop(config.get("T1", float("inf"))),
            T2=parse_prop(config.get("T2", float("inf"))),
            T2dash=parse_prop(config.get("T2'", float("inf"))),
            ADC=parse_prop(config.get("ADC", 0.0)),
            dB0=parse_prop(config.get("dB0", 0.0)),
            B1_tx=[parse_prop(ch) for ch in config.get("B1+", [1.0])],
            B1_rx=[parse_prop(ch) for ch in config.get("B1-", [1.0])],
        )

    def to_dict(self) -> dict:
        def serialize_prop(prop):
            if isinstance(prop, (float, int)):
                return prop
            elif isinstance(prop, NiftiRef):
                return prop.to_str()
            elif isinstance(prop, NiftiMapping):
                return prop.to_dict()
            else:
                raise ValueError("Unsupported property type", type(prop))

        return {
            "density": self.density.to_str(),
            "T1": serialize_prop(self.T1),
            "T2": serialize_prop(self.T2),
            "T2'": serialize_prop(self.T2dash),
            "ADC": serialize_prop(self.ADC),
            "dB0": serialize_prop(self.dB0),
            "B1+": [serialize_prop(ch) for ch in self.B1_tx],
            "B1-": [serialize_prop(ch) for ch in self.B1_rx],
        }


@dataclass
class NiftiPhantom:
    file_type = "nifti_phantom_v1"
    units: PhantomUnits
    system: PhantomSystem
    tissues: dict[str, NiftiTissue]
    reslice_to: ResliceConfig | None = field(default=None)

    @classmethod
    def default(cls, gyro=42.5764, B0=3.0):
        return cls(PhantomUnits.default(), PhantomSystem(gyro, B0), {})

    @classmethod
    def load(cls, path: Path | str):
        import json

        with open(path, "r") as f:
            config = json.load(f)
        return cls.from_dict(config)

    def save(self, path: Path | str):
        import json
        import re
        import os

        path = Path(path)

        os.makedirs(path.parent, exist_ok=True)
        text = json.dumps(self.to_dict(), indent=2)
        # Compact arrays of numbers onto a single line
        text = re.sub(
            r'\[(\n\s+[-\d.eE+]+,?)+\n\s*\]',
            lambda m: '[' + ', '.join(re.findall(r'[-\d.eE+]+', m.group(0))) + ']',
            text
        )
        with open(path, "w") as f:
            f.write(text)

    @classmethod
    def from_dict(cls, config: dict):
        units = PhantomUnits.from_dict(config["units"])
        system = PhantomSystem.from_dict(config["system"])
        tissues = {
            name: NiftiTissue.from_dict(tissue)
            for name, tissue in config["tissues"].items()
        }
        reslice_to = (ResliceConfig.from_dict(config["reslice_to"])
                      if "reslice_to" in config else None)

        return cls(units, system, tissues, reslice_to)

    def to_dict(self) -> dict:
        d = {
            "file_type": self.file_type,
            "units": self.units.to_dict(),
            "system": self.system.to_dict(),
            "tissues": {
                name: tissue.to_dict() for name, tissue in self.tissues.items()
            },
        }
        if self.reslice_to is not None:
            d["reslice_to"] = self.reslice_to.to_dict()
        return d
