from __future__ import annotations
import numpy as np


def merge_dicts(a: dict, b: dict):
    common = set(a.keys()).intersection(b.keys())
    if len(common) > 0:
        raise Exception(f"Can't merge dicts with common keys: {common}")

    return {**a, **b}


def file_to_sections(file_name: str) -> dict[str, list[str]]:
    sections = {}
    current_section: None | list = None

    def new_section(name):
        nonlocal current_section
        assert not ("[" in name or "]" in name)  # section name filtering error

        if name not in sections:
            sections[name] = []
        current_section = sections[name]

    for line in open(file_name):
        line = line.strip()
        if len(line) > 0 and line[0] != "#":
            if line[0] == "[":
                new_section(line[1:-1])
            else:
                assert current_section is not None
                current_section.append(line)

    return sections


def parse_version(lines: list[str]) -> int:
    assert len(lines) == 3
    major = None
    minor = None
    revision = None

    # This strips suffixes like "1.3.1post1"
    def to_int(s: str) -> int:
        s = s.split()[1]
        pos = 0
        while pos < len(s) and s[pos].isdigit():
            pos += 1
        return int(s[:pos])

    for line in lines:
        if line.startswith("major"):
            major = to_int(line)
        elif line.startswith("minor"):
            minor = to_int(line)
        elif line.startswith("revision"):
            revision = to_int(line)

    # We checked that there are 3 lines so each has to be set exactly once
    assert major is not None and minor is not None and revision is not None
    # Version is compressed to a 3 digit int
    assert 0 <= major <= 9 and 0 <= minor <= 9 and 0 <= revision <= 9
    return major * 100 + minor * 10 + revision


def write_version(file, version: int):
    file.write(
        f"\n[VERSION]\n"
        f"major {version // 100}\n"
        f"minor {version // 10 % 10}\n"
        f"revision {version % 10}\n"
    )


def parse_delays(lines: list[str], version: int) -> dict[int, float]:
    assert 120 <= version <= 149
    delays = {}

    for line in lines:
        vals = line.split()
        assert len(vals) == 2
        delay_id = int(vals[0])
        duration = int(vals[1]) * 1e-6

        assert delay_id > 0 and delay_id not in delays
        delays[delay_id] = duration

    return delays


def parse_shape(lines: list[str], version: int) -> tuple[int, np.ndarray]:
    assert 120 <= version <= 149
    assert len(lines) >= 3  # at least id, num and one sample

    shape_id = int(lines[0].split()[1])
    count = int(lines[1].split()[1])
    compressed = [float(line) for line in lines[2:]]

    # Uncompressed shapes are introduced with 1.4.0 but also used
    # for pTx pulses by Martin's pTx extension
    if len(compressed) == count and version >= 140:
        # No compression
        return shape_id, np.array(compressed)

    i = 0
    deriviate = []
    while i < len(compressed):
        if i < len(compressed) - 2 and compressed[i] == compressed[i+1]:
            RLE_count = compressed[i + 2]  # +2 for the samples marking RLE
            assert RLE_count == int(RLE_count) and RLE_count >= 0
            deriviate += [compressed[i]] * (int(RLE_count) + 2)
            i += 3
        else:
            deriviate.append(compressed[i])
            i += 1

    assert len(deriviate) == count, (
        f"Decompressed shape has len: {len(deriviate)}, expected: {count}"
    )

    return shape_id, np.array(deriviate).cumsum()


def parse_shapes(lines: list[str], version: int) -> dict[int, np.ndarray]:
    def is_new_shape(line: str) -> bool:
        return line.split()[0].lower() == "shape_id"

    shape_lines = []
    shapes = {}

    def new_shape(lines):
        key, value = parse_shape(lines, version)
        assert key > 0 and key not in shapes
        shapes[key] = value

    # Read in lines and parse old shape as soon as new one begins
    for line in lines:
        if is_new_shape(line) and len(shape_lines) > 0:
            new_shape(shape_lines)
            shape_lines = []

        shape_lines.append(line)

    if len(shape_lines) > 0:
        new_shape(shape_lines)

    return shapes


def write_shapes(file, shapes: dict[int, np.ndarray]):
    # TODO: compression
    file.write("\n[SHAPES]\n")
    for shape_id, shape in shapes.items():
        file.write(f"\nShape_ID {shape_id}\nNum_Uncompressed {len(shape)}\n")
        for sample in shape:
            file.write(f"{sample}\n")
