from __future__ import annotations


class Definitions:
    def __init__(self, defs: dict, version: int) -> None:
        # Use Siemens defaults if nothing is provided by the .seq file
        self.grad_raster_time = float(defs.pop("GradientRasterTime", 10e-6))
        self.rf_raster_time = float(defs.pop("RadiofrequencyRasterTime", 1e-6))
        self.adc_raster_time = float(defs.pop("AdcRasterTime", 0.1e-6))
        self.block_raster_time = float(defs.pop("BlockDurationRaster", 10e-6))

        if "FOV" in defs:
            fov_str = defs.pop("FOV").split()
            fov = (float(fov_str[0]), float(fov_str[1]), float(fov_str[2]))
            # The pulseq spec says nothing about FOV units before 1.4 and
            # mandates [mm] since 1.4. In reality, you can use arbitrary units
            # when building sequences, so we assume [m] for values < 1 and [mm]
            # for larger values. Should be safe bc. FOVs > 1m are unrealistic.
            if fov[0] > 1 or fov[1] > 1 or fov[2] > 1:
                assert version < 140, "Version 1.4 mandates FOV to be in [m]"
                fov = (fov[0] / 1000, fov[1] / 1000, fov[2] / 1000)
            self.fov = fov
        else:
            self.fov = (0.2, 0.2, 0.2)

        self.defs = defs

    @classmethod
    def parse(cls, lines: list[str], version: int):
        assert 120 <= version <= 149
        defs = {}

        for line in lines:
            item = line.split(maxsplit=1)
            assert len(item) == 2  # No support for defs without a value
            defs[item[0]] = item[1]

        if version >= 140:
            # Required in 1.4.0, could also use defaults if this violation of
            # the spec is common (but could give suprises if defaults change)
            assert "GradientRasterTime" in defs
            assert "RadiofrequencyRasterTime" in defs
            assert "AdcRasterTime" in defs
            assert "BlockDurationRaster" in defs

        return cls(defs, version)

    def write(self, file):
        file.write(
            f"\n[DEFINITIONS]\n"
            f"FOV {self.fov[0]} {self.fov[1]} {self.fov[2]}\n"
            f"GradientRasterTime {self.grad_raster_time}\n"
            f"RadiofrequencyRasterTime {self.rf_raster_time}\n"
            f"AdcRasterTime {self.adc_raster_time}\n"
            f"BlockDurationRaster {self.block_raster_time}\n"
        )
        for key, value in self.defs.items():
            file.write(f"{key} {value}\n")

    def __repr__(self) -> str:
        return (
            f"Definitions(fov={self.fov}, "
            f"grad_raster_time={self.grad_raster_time}, "
            f"rf_raster_time={self.rf_raster_time}, "
            f"adc_raster_time={self.adc_raster_time}, "
            f"block_raster_time={self.block_raster_time}, "
            f"defs={self.defs})"
        )
