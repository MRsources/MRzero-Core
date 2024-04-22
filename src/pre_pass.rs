use num_complex::{Complex32, ComplexFloat};
use std::cell::RefCell;
use std::collections::HashMap;
use std::iter;
use std::rc::Rc;

pub type RcDist = Rc<RefCell<Distribution>>;
type DistVec = Vec<RcDist>;
type DistMap = HashMap<[i32; 4], RcDist>;

#[derive(PartialEq, Eq, Copy, Clone, Default)]
pub enum DistType {
    P,
    Z,
    #[default]
    Z0,
}

#[derive(PartialEq, Eq, Hash, Copy, Clone)]
pub enum DistRelation {
    PP,
    PZ,
    ZP,
    ZZ,
    MP,
    MZ,
}

pub static DIST_TYPE_STR: [&str; 3] = ["+", "z", "z0"];
pub static DIST_RELATION_STR: [&str; 6] = ["++", "+z", "z+", "zz", "-+", "-z"];

pub struct Edge {
    pub relation: DistRelation,
    pub rot_mat_factor: Complex32,
    pub dist: RcDist,
}

#[derive(Default)]
pub struct Distribution {
    pub mag: Complex32,
    pub regrown_mag: f32, // Don't propagate to ancestors if its the own mag
    pub signal: f32,
    pub emitted_signal: f32, // relative to the strongest dist in the repetition
    pub kt_vec: [f64; 4],
    pub dist_type: DistType,
    pub ancestors: Vec<Edge>,
    /// latent_signal metric: if you want to measure states with a minimum signal of x,
    /// you should simulate states with a minimum latent_signal of <= x
    pub latent_signal: f32,
}

impl Distribution {
    fn measure(&mut self, t2dash: f32, nyquist: [f32; 3]) {
        let sigmoid = |x: f32| 1.0 / (1.0 + (-x).exp());
        let tmp = |nyq: f32, k: f64| sigmoid((nyq - k.abs() as f32 + 0.5) * 100.0);
        let dephasing = tmp(nyquist[0], self.kt_vec[0])
            * tmp(nyquist[1], self.kt_vec[1])
            * tmp(nyquist[2], self.kt_vec[2]);

        let norm = |x: f64, y: f64, z: f64| (x * x + y * y + z * z).sqrt() as f32;
        let k_len = norm(self.kt_vec[0], self.kt_vec[1], self.kt_vec[2]);

        // Empirical signal drop-off is roughly estimated by x^(-2.2)
        self.signal += self.mag.norm()
            * (-(self.kt_vec[3] as f32 / t2dash).abs()).exp()
            * (1.0 + k_len).powf(-2.2)
            * dephasing;
    }
}

pub struct Repetition {
    pub pulse_angle: f32,
    pub pulse_phase: f32,
    pub event_time: Vec<f32>,
    pub gradm_event: Vec<[f32; 3]>,
    pub adc_mask: Vec<bool>,
}

#[allow(clippy::too_many_arguments)]
pub fn comp_graph(
    seq: &[Repetition],
    t1: f32,
    t2: f32,
    t2dash: f32,
    d: f32, // expected to be defined in m²/s
    max_dist_count: usize,
    min_dist_mag: f32,
    nyquist: [f32; 3],
    grad_scale: [f32; 3], // scale gradients to SI if they are normalized
    avg_b1_trig: &[[f32; 3]],
) -> Vec<Vec<RcDist>> {
    // Make precision of state merging dependent on the units used in the seq.
    // For this, we use a fraction of the smallest used step.
    let min_kt_step = [
        seq.iter().flat_map(|r| r.gradm_event.iter().map(|g| g[0].abs() as f64)).filter(|x| *x > 1e-3).min_by(f64::total_cmp),
        seq.iter().flat_map(|r| r.gradm_event.iter().map(|g| g[1].abs() as f64)).filter(|x| *x > 1e-3).min_by(f64::total_cmp),
        seq.iter().flat_map(|r| r.gradm_event.iter().map(|g| g[2].abs() as f64)).filter(|x| *x > 1e-3).min_by(f64::total_cmp),
        seq.iter().flat_map(|r| r.event_time.iter().map(|t| *t as f64)).filter(|x| *x > 1e-6).min_by(f64::total_cmp)
    ];
    let inv_kt_grid = [
        1.0 / (0.1 * min_kt_step[0].unwrap_or(1.0)).clamp(1e-6, 1.0),
        1.0 / (0.1 * min_kt_step[1].unwrap_or(1.0)).clamp(1e-6, 1.0),
        1.0 / (0.1 * min_kt_step[2].unwrap_or(1.0)).clamp(1e-6, 1.0),
        1.0 / (0.1 * min_kt_step[3].unwrap_or(1e-3)).clamp(1e-9, 1e-3),
    ];
    let mut graph: Vec<Vec<RcDist>> = Vec::new();

    let mut dists_p = DistVec::new();
    let mut dists_z = DistVec::new();
    let mut dist_z0 = Rc::new(RefCell::new(Distribution {
        mag: Complex32::new(1.0, 0.0),
        dist_type: DistType::Z0,
        ..Default::default()
    }));

    graph.push(vec![dist_z0.clone()]);

    for rep in seq {
        {
            let (_dists_p, _dists_z, _dist_z0) = apply_pulse(
                &dists_p,
                &dists_z,
                &dist_z0,
                rep,
                max_dist_count,
                min_dist_mag,
                avg_b1_trig,
                inv_kt_grid
            );
            dists_p = _dists_p;
            dists_z = _dists_z;
            dist_z0 = _dist_z0;
        }
        graph.push(
            iter::once(&dist_z0)
                .chain(&dists_p)
                .chain(&dists_z)
                .cloned()
                .collect(),
        );

        // Simulate and measure + states
        let r2_vec: Vec<f32> = rep.event_time.iter().map(|dt| (-dt / t2).exp()).collect();

        for mut dist in dists_p.iter().map(|d| d.borrow_mut()) {
            for (((r2, gradm), dt), adc) in r2_vec
                .iter()
                .zip(&rep.gradm_event)
                .zip(&rep.event_time)
                .zip(&rep.adc_mask)
            {
                let k1 = dist.kt_vec;
                dist.kt_vec[0] += gradm[0] as f64 * grad_scale[0] as f64;
                dist.kt_vec[1] += gradm[1] as f64 * grad_scale[1] as f64;
                dist.kt_vec[2] += gradm[2] as f64 * grad_scale[2] as f64;
                dist.kt_vec[3] += *dt as f64;
                let k2 = dist.kt_vec;

                // Integrating (dt omitted) over k²(t) = ((1-x)*k1 + x*k2)^2
                // gives 1/3 * (k1^2 + k1*k2 + k2^2)
                let b = 1.0 / 3.0
                    * dt
                    * ((k1[0] * k1[0] + k1[0] * k2[0] + k2[0] * k2[0])
                        + (k1[1] * k1[1] + k1[1] * k2[1] + k2[1] * k2[1])
                        + (k1[2] * k1[2] + k1[2] * k2[2] + k2[2] * k2[2]))
                        as f32;

                dist.mag *= r2 * (-b * d).exp();
                if *adc {
                    dist.measure(t2dash, nyquist);
                }
            }
        }

        // Apply relaxation to z states
        let rep_time: f32 = rep.event_time.iter().sum();
        let r1 = (-rep_time / t1).exp();

        for mut dist in dists_z.iter().map(|d| d.borrow_mut()) {
            let sqr = |x| x * x;
            let k2 = sqr(dist.kt_vec[0]) + sqr(dist.kt_vec[1]) + sqr(dist.kt_vec[2]);

            dist.mag *= r1 * (-d * rep_time * k2 as f32).exp();
        }
        // Z0 has no diffusion because it's not dephased
        dist_z0.borrow_mut().mag *= r1;
        dist_z0.borrow_mut().mag += 1.0 - r1;
        dist_z0.borrow_mut().regrown_mag += 1.0 - r1;
    }
    graph
}

fn apply_pulse(
    dists_p: &[RcDist],
    dists_z: &[RcDist],
    dist_z0: &RcDist,
    rep: &Repetition,
    max_dist_count: usize,
    min_dist_mag: f32,
    avg_b1_trig: &[[f32; 3]],
    inv_kt_grid: [f64; 4],
) -> (DistVec, DistVec, RcDist) {
    let rot_mat = RotMat::new(rep.pulse_angle, rep.pulse_phase, avg_b1_trig);

    let mut dist_dict_p: DistMap = HashMap::new();
    let mut dist_dict_z: DistMap = HashMap::new();

    let mut add_dist = |kt_vec: [f64; 4],
                        mag: Complex32,
                        rot_mat_factor: Complex32,
                        relation: DistRelation,
                        ancestor: &RcDist| {
        let key = [
            (kt_vec[0] * inv_kt_grid[0]).round() as i32,
            (kt_vec[1] * inv_kt_grid[1]).round() as i32,
            (kt_vec[2] * inv_kt_grid[2]).round() as i32,
            (kt_vec[3] * inv_kt_grid[3]).round() as i32,
        ];
        let dist_type = match relation {
            DistRelation::PP | DistRelation::MP | DistRelation::ZP => DistType::P,
            DistRelation::PZ | DistRelation::MZ | DistRelation::ZZ => DistType::Z,
        };
        let dist_dict = if dist_type == DistType::P {
            &mut dist_dict_p
        } else {
            &mut dist_dict_z
        };
        let mag = mag * rot_mat_factor;

        match dist_dict.get(&key) {
            Some(dist) => {
                dist.borrow_mut().mag += mag;
                dist.borrow_mut().ancestors.push(Edge {
                    relation,
                    rot_mat_factor,
                    dist: ancestor.clone(),
                });
            }
            None => {
                let dist = Distribution {
                    mag,
                    kt_vec,
                    dist_type,
                    ancestors: vec![Edge {
                        relation,
                        rot_mat_factor,
                        dist: ancestor.clone(),
                    }],
                    ..Default::default()
                };
                dist_dict.insert(key, Rc::new(RefCell::new(dist)));
            }
        };
    };

    for dist in iter::once(dist_z0).chain(dists_z.iter()) {
        // zz, z+
        let mag = dist.borrow().mag;
        let kt_vec = dist.borrow().kt_vec;
        add_dist(kt_vec, mag, rot_mat.zz, DistRelation::ZZ, dist);
        add_dist(kt_vec, mag, rot_mat.zp, DistRelation::ZP, dist);
    }

    for dist in dists_p.iter() {
        // ++, +z
        let mag = dist.borrow().mag;
        let kt_vec = dist.borrow().kt_vec;
        add_dist(kt_vec, mag, rot_mat.pp, DistRelation::PP, dist);
        add_dist(kt_vec, mag, rot_mat.pz, DistRelation::PZ, dist);
        // -+, -z
        let mag = mag.conj();
        let kt_vec = [-kt_vec[0], -kt_vec[1], -kt_vec[2], -kt_vec[3]];
        add_dist(kt_vec, mag, rot_mat.mp, DistRelation::MP, dist);
        add_dist(kt_vec, mag, rot_mat.mz, DistRelation::MZ, dist);
    }

    // Separate z0 from rest of z states. We could treat z0 the same as the
    // other z states, but we would still have to keep track of it so that the
    // main pass knows where to regrow mag, so it wouldn't be any simpler.
    let dist_z0 = dist_dict_z.remove(&[0, 0, 0, 0]).expect("Z0 dist vanished");
    dist_z0.borrow_mut().dist_type = DistType::Z0;

    let truncate = |dists: &mut DistVec| {
        let mag2 = min_dist_mag * min_dist_mag;
        let len = dists.iter().position(|d| d.borrow().mag.norm_sqr() < mag2);
        dists.truncate(len.map_or(max_dist_count, |l| max_dist_count.min(l)));
    };

    let mut dists_p: DistVec = dist_dict_p.into_values().collect();
    let mut dists_z: DistVec = dist_dict_z.into_values().collect();
    let mag2 = |dist: &RcDist| dist.borrow().mag.norm_sqr();

    // sort distributions from largest mag to smallest
    dists_p.sort_unstable_by(|a, b| {
        mag2(b)
            .partial_cmp(&mag2(a))
            .expect("Encountered a NaN while sorting + dists by mag")
    });
    dists_z.sort_unstable_by(|a, b| {
        mag2(b)
            .partial_cmp(&mag2(a))
            .expect("Encountered a NaN while sorting z dists by mag")
    });
    // Truncates based on min_dist_mag and max_dist_count
    truncate(&mut dists_p);
    truncate(&mut dists_z);

    (dists_p, dists_z, dist_z0)
}

pub fn analyze_graph(graph: &mut Vec<Vec<RcDist>>) {
    // Tell each dist how large its signal is relative to the strongest dist
    for rep in graph.iter() {
        let max_signal = rep
            .iter()
            .map(|d| d.borrow().signal)
            .reduce(f32::max)
            .expect("Tried to find maximum signal but repetition is empty");

        for mut dist in rep.iter().map(|d| d.borrow_mut()) {
            dist.emitted_signal = if max_signal < 1e-9 {
                0.0
            } else {
                dist.signal / max_signal
            };
        }
    }

    // Calculate the latent_signal metric that is used to determine which states to simulate:
    // The latent_signal is the maximum of the own signal of the state or one of the propagated
    // latent_signals of its children. Propagation is relattive to the state with the largest
    // contribution: This ancestor is propagated the whole latent_signal.
    for rep in graph.iter().rev() {
        for mut dist in rep.iter().map(|d| d.borrow_mut()) {
            // Own signal is latent_signal if larger than what children produce
            dist.latent_signal = f32::max(dist.latent_signal, dist.emitted_signal);

            let max_contrib = std::iter::once(dist.regrown_mag)
                .chain(
                    dist.ancestors
                        .iter()
                        .map(|e| (e.rot_mat_factor * e.dist.borrow().mag).norm()),
                )
                .max_by(|x, y| x.partial_cmp(y).unwrap())
                .unwrap();

            // max_contrib could be the regrown mag, in this case no ancestor gets the full latent_signal
            for anc in dist.ancestors.iter() {
                let contrib = (anc.rot_mat_factor * anc.dist.borrow().mag).norm() / max_contrib;
                let tmp = anc.dist.borrow().latent_signal;
                anc.dist.borrow_mut().latent_signal = f32::max(tmp, dist.latent_signal * contrib);
            }
        }
    }
}

struct RotMat {
    pp: Complex32,
    pz: Complex32,
    mp: Complex32,
    mz: Complex32,
    zp: Complex32,
    zz: Complex32,
}

impl RotMat {
    fn new(angle: f32, phase: f32, avg_b1_trig: &[[f32; 3]]) -> Self {
        // Make angle positive and convert to degrees
        let phase = if angle >= 0.0 { phase } else { -phase };
        let angle = angle.abs() * 180.0 / std::f32::consts::PI;

        let index = angle.floor() as usize;
        let t = angle.fract();

        let v1 = &avg_b1_trig[index];
        let v2 = &avg_b1_trig[index + 1];

        let sin = v1[0] * (1.0 - t) + v2[0] * t;
        let cos = v1[1] * (1.0 - t) + v2[1] * t;
        let sin2 = v1[2] * (1.0 - t) + v2[2] * t;

        let i = Complex32::i();
        let f = i * std::f32::consts::FRAC_1_SQRT_2;

        Self {
            pp: Complex32::from(1.0 - sin2),
            pz: -f * sin * (-i * phase).exp(),
            mp: Complex32::from(sin2) * (2.0 * i * phase).exp(),
            mz: f * sin * (i * phase).exp(),
            zp: -f * sin * (i * phase).exp(),
            zz: Complex32::from(cos),
        }
    }
}
