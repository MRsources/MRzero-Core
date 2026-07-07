use std::collections::HashMap;
use std::sync::Arc;

use num_complex::Complex64;
use pulseq_rs::int::{self, Quaternion, Transform};
use pulseq_rs::raw::RfUse;
use pulseq_rs::seq;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

// 1H @ 3 T - MR-zero is usually run at 3 T.
const DEFAULT_LARMOR_HZ: f64 = 3.0 * 42_577_468.8;

/// Interpreted pulseq sequence: a list of blocks with timing in absolute seconds.
/// Each block carries its own start time so Python doesn't have to re-scan.
#[pyclass(module = "_prepass")]
pub struct PyInterpSeq {
    #[pyo3(get)]
    name: Option<String>,
    #[pyo3(get)]
    fov: [f64; 3],
    #[pyo3(get)]
    duration: f64,
    #[pyo3(get)]
    blocks: Py<pyo3::types::PyList>,
}

#[pymethods]
impl PyInterpSeq {
    fn __len__(&self, py: Python) -> usize {
        self.blocks.as_ref(py).len()
    }

    fn __repr__(&self) -> String {
        format!(
            "PyInterpSeq(name={:?}, fov={:?}, duration={} s, n_blocks={})",
            self.name,
            self.fov,
            self.duration,
            Python::with_gil(|py| self.blocks.as_ref(py).len())
        )
    }
}

#[pyclass(module = "_prepass")]
pub struct PyBlock {
    #[pyo3(get)]
    start: f64,
    #[pyo3(get)]
    duration: f64,
    #[pyo3(get)]
    rf: Option<Py<PyRf>>,
    #[pyo3(get)]
    gx: Option<Py<PyGradient>>,
    #[pyo3(get)]
    gy: Option<Py<PyGradient>>,
    #[pyo3(get)]
    gz: Option<Py<PyGradient>>,
    #[pyo3(get)]
    adc: Option<Py<PyAdc>>,
}

#[pymethods]
impl PyBlock {
    fn __repr__(&self) -> String {
        let mut parts: Vec<&str> = Vec::new();
        if self.rf.is_some() {
            parts.push("rf");
        }
        if self.gx.is_some() {
            parts.push("gx");
        }
        if self.gy.is_some() {
            parts.push("gy");
        }
        if self.gz.is_some() {
            parts.push("gz");
        }
        if self.adc.is_some() {
            parts.push("adc");
        }
        format!(
            "PyBlock(start={}, duration={}, events=[{}])",
            self.start,
            self.duration,
            parts.join(", ")
        )
    }
}

#[pyclass(module = "_prepass")]
pub struct PyRf {
    #[pyo3(get)]
    amp: f64,
    #[pyo3(get)]
    phase: f64,
    #[pyo3(get)]
    freq: f64,
    #[pyo3(get)]
    delay: f64,
    #[pyo3(get)]
    center: f64,
    #[pyo3(get)]
    rf_use: &'static str,
    #[pyo3(get)]
    shape_duration: f64,
    /// Per-channel shim weights, each as (magnitude, phase [rad]). A missing
    /// shim is exposed as `[(1.0, 0.0)]`, matching pulseq-rs.
    #[pyo3(get)]
    shims: Vec<(f64, f64)>,
    shape: Arc<int::Shape<Complex64>>,
}

#[pymethods]
impl PyRf {
    /// Sparse breakpoint times of the underlying complex shape, in seconds.
    fn shape_times(&self) -> Vec<f64> {
        self.shape.time.clone()
    }

    /// Complex shape values at the breakpoints, returned as `(re, im)`.
    fn shape_amp(&self) -> (Vec<f64>, Vec<f64>) {
        let n = self.shape.amp.len();
        let mut re = Vec::with_capacity(n);
        let mut im = Vec::with_capacity(n);
        for c in &self.shape.amp {
            re.push(c.re);
            im.push(c.im);
        }
        (re, im)
    }

    /// Integrate the pulse over `[t0, t1]` (block-relative seconds).
    /// Returns `(flip [rad], phase [rad])`. The shape carries amp×exp(i·phase)
    /// in Hz; the flip integral is `2π · amp · ∫|shape| dt` over the window,
    /// and the phase is the argument of the complex moment (so this matches
    /// `pydisseqt.parser.integrate_one(...).pulse`).
    fn integrate(&self, t0: f64, t1: f64) -> (f64, f64) {
        let (re, im) = integrate_complex_shape(self.shape.as_ref(), t0, t1);
        let mom = Complex64::new(re, im);
        let flip = 2.0 * std::f64::consts::PI * self.amp * mom.norm();
        // Add the constant RF phase offset and the optional in-shape phase.
        let phase = if mom.norm() > 0.0 {
            self.phase + mom.arg()
        } else {
            self.phase
        };
        (flip, phase)
    }

    fn __repr__(&self) -> String {
        format!(
            "PyRf(amp={} Hz, phase={} rad, freq={} Hz, delay={} s, dur={} s, use={}, shims={})",
            self.amp,
            self.phase,
            self.freq,
            self.delay,
            self.shape_duration,
            self.rf_use,
            self.shims.len()
        )
    }
}

#[pyclass(module = "_prepass")]
pub struct PyGradient {
    /// FOV-scaled gradient amplitude `[Hz/m]`. The shape stores normalised values
    /// in `[-1, 1]`; multiply by `amp` to get instantaneous strength.
    #[pyo3(get)]
    amp: f64,
    #[pyo3(get)]
    delay: f64,
    #[pyo3(get)]
    shape_duration: f64,
    shape: Arc<int::Shape<f64>>,
}

#[pymethods]
impl PyGradient {
    fn shape_times(&self) -> Vec<f64> {
        self.shape.time.clone()
    }

    fn shape_amp(&self) -> Vec<f64> {
        self.shape.amp.clone()
    }

    /// Sample the gradient strength `[Hz/m]` at a block-relative time `t`.
    /// Returns 0 outside `[delay, delay + shape_duration]`.
    fn sample(&self, t: f64) -> f64 {
        let rel = t - self.delay;
        if rel < 0.0 || rel > self.shape.duration {
            return 0.0;
        }
        self.amp * self.shape.interpolate(rel)
    }

    /// Integrate the gradient moment over `[t0, t1]` (block-relative seconds).
    /// Returns moment in `[Hz·s / m] = [cycles / m]`, matching what
    /// `pydisseqt.parser.integrate(...).gradient.{x,y,z}` returns.
    fn integrate(&self, t0: f64, t1: f64) -> f64 {
        // Shift to shape-relative time.
        let s0 = t0 - self.delay;
        let s1 = t1 - self.delay;
        self.amp * integrate_real_shape(self.shape.as_ref(), s0, s1)
    }

    fn __repr__(&self) -> String {
        format!(
            "PyGradient(amp={} Hz/m, delay={} s, dur={} s, n_samples={})",
            self.amp,
            self.delay,
            self.shape_duration,
            self.shape.amp.len()
        )
    }
}

#[pyclass(module = "_prepass")]
pub struct PyAdc {
    #[pyo3(get)]
    num: u32,
    #[pyo3(get)]
    dwell: f64,
    #[pyo3(get)]
    delay: f64,
    #[pyo3(get)]
    freq: f64,
    #[pyo3(get)]
    phase: f64,
    phase_shape: Option<Arc<int::Shape<f64>>>,
    labels: int::Labels,
}

#[pymethods]
impl PyAdc {
    /// Block-relative sample times `[s]`: `delay + (n + 0.5) * dwell`.
    fn sample_times(&self) -> Vec<f64> {
        (0..self.num)
            .map(|n| self.delay + (n as f64 + 0.5) * self.dwell)
            .collect()
    }

    /// Per-sample phases `[rad]`: base `phase` plus the optional in-shape
    /// modulation evaluated at each sample's relative time.
    fn sample_phases(&self) -> Vec<f64> {
        let n = self.num as usize;
        let mut out = vec![self.phase; n];
        if let Some(ps) = self.phase_shape.as_ref() {
            for (i, val) in out.iter_mut().enumerate() {
                let t = (i as f64 + 0.5) * self.dwell;
                *val += ps.interpolate(t);
            }
        }
        out
    }

    /// Snapshot of the pulseq label state at the time this ADC fires.
    /// Counters (slc, seg, rep, …) come back as `i32`; boolean flags
    /// (nav, rev, …) are returned as `0` / `1` so callers can pack them
    /// into a single tensor type.
    fn labels(&self) -> HashMap<&'static str, i32> {
        let l = self.labels;
        let mut m = HashMap::with_capacity(17);
        m.insert("slc", l.slc);
        m.insert("seg", l.seg);
        m.insert("rep", l.rep);
        m.insert("avg", l.avg);
        m.insert("set", l.set);
        m.insert("eco", l.eco);
        m.insert("phs", l.phs);
        m.insert("lin", l.lin);
        m.insert("par", l.par);
        m.insert("acq", l.acq);
        m.insert("nav", l.nav as i32);
        m.insert("rev", l.rev as i32);
        m.insert("sms", l.sms as i32);
        m.insert("ref", l.ref_ as i32);
        m.insert("ima", l.ima as i32);
        m.insert("off", l.off as i32);
        m.insert("noise", l.noise as i32);
        m
    }

    fn __repr__(&self) -> String {
        format!(
            "PyAdc(num={}, dwell={} s, delay={} s, freq={} Hz, phase={} rad)",
            self.num, self.dwell, self.delay, self.freq, self.phase
        )
    }
}

#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (
    path,
    *,
    larmor_hz = None,
    fov_scale = None,
    fov_pos = None,
    fov_rot = None,
    soft_delays = None,
))]
pub fn load_pulseq_rs(
    py: Python,
    path: &str,
    larmor_hz: Option<f64>,
    fov_scale: Option<f64>,
    fov_pos: Option<[f64; 3]>,
    fov_rot: Option<[f64; 4]>,
    soft_delays: Option<HashMap<String, f64>>,
) -> PyResult<Py<PyInterpSeq>> {
    let seq = seq::Sequence::from_file(path)
        .map_err(|e| PyValueError::new_err(format!("pulseq-rs parse error: {e}")))?;

    let transform = Transform {
        scale: fov_scale.unwrap_or(1.0),
        rotation: Quaternion(fov_rot.unwrap_or([1.0, 0.0, 0.0, 0.0])),
        position: fov_pos.unwrap_or([0.0, 0.0, 0.0]),
    };

    let (int_seq, warnings) = int::Sequence::from_seq(
        &seq,
        transform,
        larmor_hz.unwrap_or(DEFAULT_LARMOR_HZ),
        soft_delays.unwrap_or_default(),
    )
    .map_err(|e| PyValueError::new_err(format!("pulseq-rs interpreter error: {e}")))?;

    let user_warning = py.get_type_bound::<pyo3::exceptions::PyUserWarning>();
    for w in warnings {
        PyErr::warn_bound(py, &user_warning, &w.to_string(), 0)?;
    }

    let blocks = pyo3::types::PyList::empty_bound(py);
    let mut t_start = 0.0_f64;
    let mut total = 0.0_f64;

    for block in &int_seq.blocks {
        let rf = block.rf.as_ref().map(|x| rf_to_py(py, x)).transpose()?;
        let gx = block.gx.as_ref().map(|x| grad_to_py(py, x)).transpose()?;
        let gy = block.gy.as_ref().map(|x| grad_to_py(py, x)).transpose()?;
        let gz = block.gz.as_ref().map(|x| grad_to_py(py, x)).transpose()?;
        let adc = block.adc.as_ref().map(|x| adc_to_py(py, x)).transpose()?;

        let py_block = PyBlock {
            start: t_start,
            duration: block.duration,
            rf,
            gx,
            gy,
            gz,
            adc,
        };
        blocks.append(Py::new(py, py_block)?)?;
        t_start += block.duration;
        total = t_start;
    }

    let seq_obj = PyInterpSeq {
        name: int_seq.name.clone(),
        fov: int_seq.fov,
        duration: total,
        blocks: blocks.into(),
    };
    Py::new(py, seq_obj)
        .map_err(|e| PyRuntimeError::new_err(format!("failed to wrap PyInterpSeq: {e}")))
}

fn rf_to_py(py: Python, rf: &int::Rf) -> PyResult<Py<PyRf>> {
    let rf_use = match rf.rf_use {
        RfUse::Excitation => "excitation",
        RfUse::Refocusing => "refocusing",
        RfUse::Inversion => "inversion",
        RfUse::Saturation => "saturation",
        RfUse::Preparation => "preparation",
        RfUse::Other => "other",
        RfUse::Undefined => "undefined",
    };
    let shims: Vec<(f64, f64)> = rf.shims.iter().map(|c| (c.norm(), c.arg())).collect();
    Py::new(
        py,
        PyRf {
            amp: rf.amp,
            phase: rf.phase,
            freq: rf.freq,
            delay: rf.delay,
            center: rf.center,
            rf_use,
            shape_duration: rf.shape.duration,
            shims,
            shape: rf.shape.clone(),
        },
    )
}

fn grad_to_py(py: Python, g: &int::Gradient) -> PyResult<Py<PyGradient>> {
    Py::new(
        py,
        PyGradient {
            amp: g.amp,
            delay: g.delay,
            shape_duration: g.shape.duration,
            shape: g.shape.clone(),
        },
    )
}

fn adc_to_py(py: Python, adc: &int::Adc) -> PyResult<Py<PyAdc>> {
    Py::new(
        py,
        PyAdc {
            num: adc.num,
            dwell: adc.dwell,
            delay: adc.delay,
            freq: adc.freq,
            phase: adc.phase,
            phase_shape: adc.phase_shape.clone(),
            labels: adc.labels,
        },
    )
}

/// Trapezoidal integration of a sparse piecewise-linear real shape over
/// `[t0, t1]` (shape-relative seconds). Values outside the shape's time
/// support are treated as `shape.amp[0]` and `*shape.amp.last()` (matching
/// `Shape::interpolate`).
fn integrate_real_shape(shape: &int::Shape<f64>, t0: f64, t1: f64) -> f64 {
    if t1 <= t0 {
        return 0.0;
    }
    let t0 = t0.clamp(0.0, shape.duration);
    let t1 = t1.clamp(0.0, shape.duration);
    if t1 <= t0 {
        return 0.0;
    }
    let times = &shape.time;
    let amps = &shape.amp;
    let n = times.len();
    if n == 0 {
        return 0.0;
    }
    if n == 1 {
        return amps[0] * (t1 - t0);
    }

    let v0 = shape.interpolate(t0);
    let v1 = shape.interpolate(t1);

    let mut acc = 0.0;
    let mut prev_t = t0;
    let mut prev_v = v0;
    for i in 0..n {
        let ti = times[i];
        if ti <= prev_t {
            continue;
        }
        if ti >= t1 {
            break;
        }
        let vi = amps[i];
        acc += 0.5 * (prev_v + vi) * (ti - prev_t);
        prev_t = ti;
        prev_v = vi;
    }
    acc += 0.5 * (prev_v + v1) * (t1 - prev_t);
    acc
}

/// Same as `integrate_real_shape`, but for complex shapes. Used to integrate
/// the RF pulse to a complex moment whose magnitude → flip and arg → phase.
fn integrate_complex_shape(shape: &int::Shape<Complex64>, t0: f64, t1: f64) -> (f64, f64) {
    if t1 <= t0 {
        return (0.0, 0.0);
    }
    let t0 = t0.clamp(0.0, shape.duration);
    let t1 = t1.clamp(0.0, shape.duration);
    if t1 <= t0 {
        return (0.0, 0.0);
    }
    let times = &shape.time;
    let amps = &shape.amp;
    let n = times.len();
    if n == 0 {
        return (0.0, 0.0);
    }
    if n == 1 {
        let c = amps[0] * (t1 - t0);
        return (c.re, c.im);
    }

    let v0 = shape.interpolate(t0);
    let v1 = shape.interpolate(t1);

    let mut acc = Complex64::new(0.0, 0.0);
    let mut prev_t = t0;
    let mut prev_v = v0;
    for i in 0..n {
        let ti = times[i];
        if ti <= prev_t {
            continue;
        }
        if ti >= t1 {
            break;
        }
        let vi = amps[i];
        acc += (prev_v + vi) * (0.5 * (ti - prev_t));
        prev_t = ti;
        prev_v = vi;
    }
    acc += (prev_v + v1) * (0.5 * (t1 - prev_t));
    (acc.re, acc.im)
}
