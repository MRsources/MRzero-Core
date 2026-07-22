# Pulseq-zero

Pulseq-zero allows to define MRI sequences with the Python[^1] port of Pulseq[^2]: PyPulseq[^3], and use them within MR-zero[^4].
This way they are deeply integrated in a differentiable digital twin, enabling not only the simulation of the defined sequence but also the efficient optimization of any sequence parameter and any loss function, using the power of PyTorch[^5] and gradient-descent with backpropagation.

Pulseq-zero uses PDG[^6], a fast, analytical and physically exact simulation model that calculates signals that are comparable to in-vivo measurements within seconds.
At the same time, the required changes to the sequence code are minimal; Pulseq-zero exports the optimized sequence works by simply using the installed PyPulseq without any interference.


## Table of contents

1. [General Information](#1-general-information)
2. [Usage](#2-usage)
3. [Development](#3-development)
4. [API](#4-api)
5. [References](#5-references)


## 1. General Information

Pulseq-zero can be cloned from this repository but is also hosted on [PyPI](https://pypi.org/project/pulseqzero/), install it locally with:
```bash
pip install pulseqzero
```

> [!NOTE]
> Pulseq-zero does not declare any runtime dependencies, but it expects `pypulseq`, `torch`, `MRzeroCore`, `numpy`, and `matplotlib` to already be in the environment.
> Pulseq-zero 1.0 is compatible with **PyPulseq 1.4.x and 1.5.x**. Version-specific kwargs (`freq_ppm`, `no_signal_scaling`, `use_block_cache`, etc.) are detected at import time via `inspect.signature`, so the same pulseq-zero installation adapts automatically to whichever PyPulseq version is in the environment.
>
> **Migration from 0.x:** the mode-switching facade has been removed. Replace `import pulseqzero; pp = pulseqzero.pp_impl` with `import pulseqzero as pp`, and drop any `with pulseqzero.mr0_mode():` wrappers — `seq.to_mr0()` and `seq.write()` both work unconditionally now.

Pulseq-zero was displayed at [ESMRMB 2024](https://www.esmrmb2024.org/)!
You can view the abstract here: [abstract/abstract.md](abstract/abstract.md).
This project is affiliated with MR-zero and PDG but none of the other technologies.
It relies on the following amazing projects:
- [Python](https://www.python.org/) is the programming language used for Pulseq-zero
- [Pulseq](https://pubmed.ncbi.nlm.nih.gov/27271292/) is a vendor-agnostic library and file format for sequence definition and transfer to real systems
- [PyPulseq](https://joss.theoj.org/papers/10.21105/joss.01725) is the port of Pulseq to Python
- [MR-zero](https://onlinelibrary.wiley.com/doi/full/10.1002/mrm.28727) is a digital twin of the full measurement and reconstruction pipeline for sequence optimization and discovery
- [PyTorch](https://arxiv.org/abs/1912.01703) is an ecosystem of tools for efficient tensor math with GPU accelleration, autograd through backpropagation and a wide variety of optimizers
- [PDG](https://onlinelibrary.wiley.com/doi/full/10.1002/mrm.30055) (short for Phase Distribution Graphs) is a state-of-the-art Bloch simulation that produces accurate MRI signals for any sequence, orders of magnitude faster than other approaches


## 2. Usage

Pulseq-zero is a drop-in replacement for PyPulseq: any existing PyPulseq script runs under pulseq-zero by swapping the import — or, if you don't want to touch the script at all, by installing a two-line `sys.modules` hijack in the driver (see [§2 · Reuse an unmodified PyPulseq script](#reuse-an-unmodified-pypulseq-script)). The same script then writes `.seq` files *and* is consumed differentiably by MR-zero — no context managers, no mode flags.

```python
import pulseqzero as pp

# Build the sequence exactly like a PyPulseq script.
seq = pp.Sequence()
seq.add_block(pp.make_delay(10e-3))
```

### Define the sequence as a function

Wrap the sequence code in a function so the same definition can drive both `.seq` export and MR-zero simulation / optimization:

```python
def my_gre_seq(TR, TE):
    seq = pp.Sequence()
    # ... create your sequence ...
    seq.add_block(pp.make_delay(TR - 3e-3))
    # ... more sequence creation ...
    return seq
```

### Reuse an unmodified PyPulseq script

You don't have to swap the import at all. If a sequence script still reads `import pypulseq as pp`, a two-line hijack in the driver redirects every `pypulseq` attribute lookup to pulseq-zero:

```python
import sys
import pulseqzero
sys.modules["pypulseq"] = pulseqzero

from my_existing_pypulseq_script import main  # its `import pypulseq as pp` now resolves to pulseqzero
seq = main()
seq.to_mr0()   # differentiable — no edits to the sequence script
```

The [demo/](demo/) workspace uses exactly this pattern: [demo/write_tse.py](demo/write_tse.py) is a near-verbatim copy of the PyPulseq 1.5 upstream TSE example and still imports `pypulseq`; [demo/main.py](demo/main.py) installs the hijack before importing it and gets a differentiable sequence for free.

**Caveats.** The hijack must run before the downstream script is first imported. `from pypulseq.submod import X` (submodule access, e.g. `pypulseq.convert`) will fail unless pulseq-zero mirrors `submod` — top-level `import pypulseq` and `from pypulseq import foo` both work. Calls to entry points pulseq-zero deliberately doesn't wrap (adiabatic pulses, sigpy, SLR, etc.) still raise `NotImplementedError` with a named workaround (see §4).

### Application

- **Export a `.seq` file and plot** (goes through PyPulseq under the hood; a one-off translation warning is emitted so you notice if it fires inside a hot loop):
  ```python
  seq = my_gre_seq(14e-3, 5e-3)
  seq.plot()
  seq.write("tse.seq")
  ```
- **Simulate with MR-zero**:
  ```python
  import MRzeroCore as mr0

  seq = my_gre_seq(14e-3, 5e-3).to_mr0()
  graph = mr0.compute_graph(seq, sim_data)
  signal = mr0.execute_graph(graph, seq, sim_data)
  reco = mr0.reco_adjoint(signal, seq.get_kspace())
  ```
- **Optimize sequence parameters with PyTorch**:
  ```python
  TR = torch.tensor(14e-3, requires_grad=True)
  TE = torch.tensor(5e-3, requires_grad=True)
  optimizer = torch.optim.Adam([TR, TE], lr=0.001)

  for _ in range(100):
      optimizer.zero_grad()
      seq = my_gre_seq(TR, TE).to_mr0()
      loss = my_loss(seq)
      loss.backward()
      optimizer.step()

  # After optimization: export using the same script.
  my_gre_seq(TR, TE).write("tse_optim.seq")
  ```


## 3. Development

The recommended dev toolchain is [uv](https://docs.astral.sh/uv/). Install it once ([install instructions](https://docs.astral.sh/uv/getting-started/installation/)), then run the demos straight from the repo root:

```bash
uv run demo/main.py        # end-to-end optimization demo (needs demo/brain.npz)
uv run demo/write_tse.py   # build a TSE sequence, plot it, and emit tse_pypulseq.seq
```

`uv run` resolves the `demo` workspace member defined in [pyproject.toml](pyproject.toml), installs the pinned demo deps (PyPulseq 1.5.0.post1, torch, MRzeroCore, matplotlib) into `.venv/`, and picks up the editable `pulseqzero` checkout — no manual `pip install -e .` step.

Good to know:

- **No test suite, no linter, no CI for correctness.** The two demo scripts *are* the acceptance gate: `demo/main.py` must complete 30 Adam iterations with non-NaN data loss and monotonically-decreasing SAR, and `demo/write_tse.py` must produce a `.seq` file that round-trips byte-for-byte against the pypulseq reference (the unified adapter guarantees this; see `to_pypulseq()` in [adapter/sequence.py](src/pulseqzero/adapter/sequence.py)).
- **PyTorch CUDA pin.** [demo/pyproject.toml](demo/pyproject.toml) references the CUDA 12.6 wheel index (`download.pytorch.org/whl/cu126`). Swap that index URL (or remove it) if you're on CPU-only or a different CUDA version.
- **Headless plotting.** `seq.plot()` forwards to pypulseq's plot and expects an interactive matplotlib backend. Export `MPLBACKEND=Agg` to run headless (Agg will render but not show — useful for CI-style runs).
- **Falling back to plain pip.** If you'd rather skip uv, `pip install --editable .` from the root still works — but the root declares no runtime deps, so you need `pypulseq==1.5.0.post1`, `torch`, `MRzeroCore`, `numpy`, and `matplotlib` already in the env (a venv created with `--system-site-packages` and an existing MR-zero install is the path of least resistance).
- **Warnings are single-fire.** `seq.to_pypulseq()` (and `seq.write()` / `seq.plot()` / other forwarders) emit a `UserWarning` the *first* time they run from a given call site, then stay quiet — Python's default warning filter dedupes by `(message, module, lineno)`. If the warning fires inside an optimization loop, move the call out of the loop.


## 4. API

Pulseq-zero provides the whole pypulseq 1.5 API, with some notable exceptions:
Some functions are not differentiable (like plotting) and just re-exports.
A few methods are not provided (like sigpy pulse optimization) as they are not
compatible with the approach taken by pulseq- zero.

The full API can be found in [TOC.md](TOC.md).

### Differentiable rounding

PyPulseq aligns many events to the block / gradient / ADC raster, which requires rounding — and rounding kills gradients. Pulseq-zero ships `pp.round` / `pp.ceil` / `pp.floor` that match PyTorch semantics but act like the identity function on the backward pass:

```python
my_param = torch.tensor(1.5, requires_grad=True)
some_calc = pp.round(torch.sin(my_param))
some_calc.backward()

assert some_calc == 1
assert my_param.grad == torch.cos(my_param)
```

Use these whenever you round a timing (or any sequence quantity) that flows from an optimization parameter. For plain numeric rounding outside optimization, `np.round` / `torch.round` are fine.

### `seq.to_mr0()` and `seq.write()`

Every `pulseqzero.Sequence` supports both paths unconditionally:

- `mr0_seq = seq.to_mr0()` — build an `MRzeroCore.Sequence` for PDG simulation / optimization.
- `seq.write("out.seq")` — translate the internal event graph through PyPulseq and emit a `.seq` file. A one-time `warnings.warn` is raised per call so you notice if it fires inside a hot loop (move it out of the optimizer).

If you need a native PyPulseq `Sequence` for a one-off exotic call, `seq.to_pypulseq()` is the explicit escape hatch.

### System (`Opts`) is configured once, on the `Sequence`

Pulseq-zero applies **one** system to the whole sequence: the `system=` you pass to `pp.Sequence(system=...)` (or `Opts.default` when you pass none). That system is authoritative — it is the one used for every event at `to_mr0()`, `write()`, and `to_pypulseq()`.

A `system=` handed to an individual `make_*` call is honored only for that call's construction-time math (e.g. deriving a slew-limited `rise_time`); it is **not** stored on the event and **not** carried to conversion/export. Consequently, pulseq-zero does **not** support mutating the system between calls, nor using a different/derated `Opts` for individual events: at export the `Sequence`'s system wins, so an event built under different limits can silently disagree with it (or fail PyPulseq's re-validation). If you genuinely need per-event limits, do that one-off through `seq.to_pypulseq()`.

This matches the universal pulseq idiom — create one `Opts` at the top, pass it to the `Sequence` and (by convention) to each `make_*` call; since it is the same object, everything stays consistent. Every script in the PyPulseq example suite checked (`write_gre`, `write_epi`, `write_tse`, `write_haste`, `write_radial_gre`, `write_ute`, `write_epi_se_rs`) creates exactly one `Opts` and reuses it throughout.

### Differentiability

Gradients flow through the following quantities end-to-end (set `requires_grad=True` and they thread through to `seq.to_mr0()`):

- RF `flip_angle`, `phase_offset`, `freq_offset`, `delay`
- ADC `phase_offset`, `freq_offset`, `delay`, `dwell`
- Gradient `amplitude` (trapezoidal and arbitrary), `rise_time`, `flat_time`, `fall_time`, `delay`
- Block / repetition / TR / TE durations

The following are **not** differentiable today (they affect pulse *shape*, which is materialized eagerly via PyPulseq):

- Pulse shape parameters: `duration` when used to shape the envelope, `time_bw_product`, `apodization`, `center_pos`, `slice_thickness` (as it feeds shape generation), `dwell` for pulses
- Gradient *waveform samples* for arbitrary gradients (the scale is differentiable, the samples aren't)
- `Opts` fields (max_grad, rasters, dead times) — intentionally numeric

Pulse-shape autograd can be added back per-factory via an opt-in flag if it ever becomes load-bearing.


## 5. References

[^1]: python programming language: https://www.python.org/
[^2]: Layton K et al: Pulseq: A rapid and hardware-independent pulse sequence prototyping framework. MRM 2017, [doi: 10.1002/mrm.26235](https://pubmed.ncbi.nlm.nih.gov/27271292/)
[^3]: Keerthi SR et al: PyPulseq: A Python Package for MRI Pulse Sequence Design. JOSS 2019, [doi: 10.21105/joss.01725](https://joss.theoj.org/papers/10.21105/joss.01725)
[^4]: Loktyushin A et al: MRzero - Automated discovery of MRI sequences using supervised learning. MRM 2021, [doi: 10.1002/mrm.28727](https://onlinelibrary.wiley.com/doi/full/10.1002/mrm.28727)
[^5]: Paszke A et al: PyTorch: An Imperative Style, High-Performance Deep Learning Library. arxiv 2019, [doi: 10.48550/arXiv.1912.01703](https://arxiv.org/abs/1912.01703)
[^6]: Endres J et al: Phase distribution graphs for fast, differentiable, and spatially encoded Bloch simulations of arbitrary MRI sequences. MRM 2024, [doi: 10.1002/mrm.30055](https://onlinelibrary.wiley.com/doi/full/10.1002/mrm.30055)