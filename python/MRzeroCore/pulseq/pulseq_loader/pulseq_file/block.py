from __future__ import annotations


class Block:
    def __init__(
        self,
        duration: float,  # s (spec: 1.4.0: BlockRaster units, 1.3.x: computed)
        rf_id: int,
        gx_id: int,
        gy_id: int,
        gz_id: int,
        adc_id: int,
        ext_id: int,
    ) -> None:
        self.duration = duration
        self.rf_id = rf_id
        self.gx_id = gx_id
        self.gy_id = gy_id
        self.gz_id = gz_id
        self.adc_id = adc_id
        self.ext_id = ext_id

    @classmethod
    def parse(cls, line: str, version: int,
              delays: dict[int, float] | None,  # 1.3.x to comp duration
              block_duration_raster: float | None  # 1.4.0 to comp duration
              ) -> tuple[int, Block]:
        """
        If this function is used for 1.3.x, duration is only respects the delay
        event (if there is any). Calculate duration after finishing parsing
        (set to duration of longest event)
        """
        assert (
            (120 <= version < 140 and delays is not None) or
            (version >= 140 and block_duration_raster is not None)
        )

        vals = line.split()
        assert len(vals) == 7 if version < 130 else 8

        if version >= 140:
            assert block_duration_raster is not None
            duration = int(vals[1]) * block_duration_raster
        else:
            assert delays is not None
            duration = delays.get(int(vals[1]), 0.0)

        block_id = int(vals[0])
        return block_id, cls(
            duration,
            int(vals[2]),
            int(vals[3]),
            int(vals[4]),
            int(vals[5]),
            int(vals[6]),
            0 if version < 130 else int(vals[7]),
        )

    def write(self, file, block_id: int, block_raster_time: float):
        file.write(
            f"{block_id:4d} {round(self.duration / block_raster_time):7d} "
            f"{self.rf_id:4d} {self.gx_id:4d} {self.gy_id:4d} "
            f"{self.gz_id:4d} {self.adc_id:4d} {self.ext_id:4d}\n"
        )

    def __repr__(self) -> str:
        return (
            f"Block(duration={self.duration}, "
            f"rf_id={self.rf_id}, "
            f"gx_id={self.gx_id}, "
            f"gy_id={self.gy_id}, "
            f"gz_id={self.gz_id}, "
            f"adc_id={self.adc_id}, "
            f"ext_id={self.ext_id})"
        )


def parse_blocks(
    lines: list[str],
    version: int,
    delays: dict[int, float] | None,
    block_duration_raster: float | None
) -> dict[int, Block]:
    tmp = {}
    for line in lines:
        key, value = Block.parse(line, version, delays, block_duration_raster)
        assert key > 0 and key not in tmp
        tmp[key] = value
    return tmp


def write_blocks(file, blocks: dict[int, Block], block_raster_time: float):
    file.write(
        "\n[BLOCKS]\n"
        "# ID     DUR   RF   GX   GY   GZ  ADC  EXT\n"
    )
    for block_id, block in blocks.items():
        block.write(file, block_id, block_raster_time)
