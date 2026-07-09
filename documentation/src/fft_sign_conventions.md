# MR-zero sign conventions
(from version 1.0.0 on)

## 1. Scope

This note defines the sign convention used in MR-zero for simulated MR signals, k-space encoding, FID formation, spectral reconstruction, and phase interpretation.

The central convention is:

$$
M_{xy}(t) = M_x(t) + iM_y(t)
$$

For a proton with positive gyromagnetic ratio in a positive \\(B_0\\) field, the transverse magnetization rotates clockwise when viewed along \\(+B_0\\). In this convention, free precession is written as:

$$
M_{xy}(t)=M_{xy}(0)e^{-i\omega_0 t},
\qquad
\omega_0=\gamma B_0 .
$$

Thus, MR-zero uses the physical Bloch-equation sign convention:

$$
\boxed{
\text{physical MR signal evolution uses a negative complex exponent}
}
$$

This does **not** mean that every reconstruction step should numerically call `fft`.

It means that the **signal formation / encoding kernel** is FFT-like, while the **reconstruction from raw samples back to object space** is iFFT-like.

---

## 2. FFT terminology

Using the usual NumPy/MATLAB convention, the forward discrete Fourier transform is:

$$
A_k=\sum_{m=0}^{N-1}a_m
\exp\left(-2\pi i\,\frac{mk}{N}\right).
$$

The inverse discrete Fourier transform is:

$$
a_m=\frac{1}{N}\sum_{k=0}^{N-1}A_k
\exp\left(+2\pi i\,\frac{mk}{N}\right).
$$

In this note:

- **FFT-type kernel** means an \\(e^{-i(\cdot)}\\) kernel.
- **iFFT-type kernel** means an \\(e^{+i(\cdot)}\\) kernel.
- Scaling factors such as \\(1/N\\), \\(2\pi\\), `fftshift`, and `ifftshift` are treated separately.

Therefore, with the MR-zero physical convention,

$$
s(t)\sim e^{-i\omega t},
$$

a spectrum with a peak at \\(+\omega\\) is obtained using:

$$
\rho(\omega)=\int s(t)e^{+i\omega t}\,dt .
$$

In NumPy/MATLAB-style code, this corresponds to a scaled `ifft`, not `fft`.

---

## 3. Imaging and spectroscopy have the same sign structure

The same physical sign governs both spatial encoding and spectral/FID evolution.

### 3.1 Imaging

Gradient-induced dephasing gives:

$$
s(\mathbf k) = \int \rho(\mathbf r) \exp(-i\,\mathbf k\cdot\mathbf r)\,d\mathbf r .
$$

Equivalently, if using cycles rather than radians:

$$ s(\mathbf k) = \int \rho(\mathbf r) \exp(-i2\pi\,\mathbf k\cdot\mathbf r) \,d\mathbf r .
$$

This is a forward Fourier encoding of the image into k-space.

Therefore image reconstruction is the inverse transform:

$$
\rho(\mathbf r) = \int s(\mathbf k) \exp(+i\,\mathbf k\cdot\mathbf r)\,d\mathbf k .
$$

So, under this convention:

$$
\boxed{
\text{k-space}\rightarrow\text{image uses an iFFT-type reconstruction}
}
$$

### 3.2 Spectroscopy

For spectroscopy, a spin ensemble with frequency distribution \\(\rho(\omega)\\) gives an FID:

$$ 
s(t) = \int \rho(\omega) \exp(-i\omega t)\,d\omega .
$$

This is the same structure as imaging:

$$
s(\mathbf k) = \int \rho(\mathbf r) \exp(-i\,\mathbf k\cdot\mathbf r)\,d\mathbf r .
$$

The correspondence is:

$$
(\mathbf k,\mathbf r)\leftrightarrow(t,\omega).
$$

Thus the reconstruction is also the inverse-sign transform:

$$
\rho(\omega) = \int s(t)\exp(+i\omega t)\,dt .
$$

So, under the MR-zero convention:

$$
\boxed{
\text{FID}\rightarrow\text{spectrum uses an iFFT-type reconstruction}
}
$$

In code:

```python
spec = np.fft.fftshift(np.fft.ifft(fid, axis=time_axis), axes=time_axis)
spec *= fid.shape[time_axis]   # optional: remove NumPy's 1/N ifft scaling
```

---

## 4. Fat, water, and \\(B_0\\) phase signs

Assume water is the reference frequency.

Water rotates clockwise at the reference frequency. Fat is more shielded and therefore precesses more slowly than water. Relative to water, fat has a negative frequency offset:

$$
\Delta f_{\text{fat}} < 0 .
$$

Under the MR-zero convention,

$$
s(t)=e^{-i2\pi\Delta f t}.
$$

Therefore, for fat,

$$
\Delta f_{\text{fat}}<0 \quad\Rightarrow\quad s_{\text{fat}}(t) = e^{+i2\pi|\Delta f_{\text{fat}}|t}.
$$

So fat accumulates **positive phase** relative to water.

For a general \\(B_0\\) perturbation,

$$
\Delta f(\mathbf r)=\frac{\gamma}{2\pi}\Delta B_0(\mathbf r),
$$

and the phase evolves as:

$$
\phi(\mathbf r,t) = -2\pi\Delta f(\mathbf r)t = -\gamma \Delta B_0(\mathbf r)t .
$$

Therefore, under the MR-zero physical convention:

$$
\boxed{
\phi \text{ and } \Delta B_0 \text{ are anticorrelated}
}
$$

Positive \\(\Delta B_0\\) gives negative phase accumulation.

Negative frequency offsets, such as fat relative to water, give positive phase accumulation.

---

## 5. Relation to textbook and MRS-processing conventions

### 5.1 Keeler: physical precession versus processing convention

There is an apparent sign change in common NMR teaching material.

In the vector-model description, the physical Larmor precession is commonly written as:

$$
\omega_0=-\gamma B_0 .
$$

For nuclei with positive gyromagnetic ratio, such as protons, this means the Larmor frequency is negative under that coordinate convention. The transverse magnetization rotates clockwise when viewed along \\(+B_0\\).

With:

$$
M_{xy}=M_x+iM_y,
$$

this corresponds to:

$$
M_{xy}(t)=M_{xy}(0)e^{-i\omega t}
$$

for a positive physical precession magnitude \\(\omega>0\\).

However, in Fourier-processing discussions of NMR spectroscopy, the complex FID is often written differently. The two receiver channels may be defined as:

$$
S_x(t)=S_0\cos\Omega t,
\qquad
S_y(t)=S_0\sin\Omega t.
$$

Then the complex signal is combined as:

$$
S(t)=S_x(t)+iS_y(t).
$$

This gives:

$$
S(t) = S_0\cos\Omega t+iS_0\sin\Omega t = S_0e^{+i\Omega t}.
$$

In MR-zero terminology, the distinction is:

$$
\boxed{
\text{physical magnetization convention: } e^{-i\omega t}
}
$$

but:

$$
\boxed{
\text{receiver/processing convention: } e^{+i\Omega t}
}
$$

MR-zero intentionally follows the first convention, not the second.

---

## 6. Why MR-zero uses the physical sign

MR-zero simulates the MR signal directly in the rotating frame.

It does **not** simulate the complete analog/digital receive chain, including:

- coil voltage induction,
- mixer phase,
- receiver reference phase,
- quadrature-channel ordering,
- vendor-specific channel combination,
- storage-format convention.

Therefore MR-zero should not silently assume a particular receiver convention such as:

$$
S(t)=I(t)+iQ(t)
$$

versus:

$$
S(t)=I(t)-iQ(t).
$$

Those two choices differ by complex conjugation:

$$
S^*(t)
\quad\Longleftrightarrow\quad
e^{-i\omega t}\leftrightarrow e^{+i\omega t}.
$$

The MR-zero convention is instead:

$$
\boxed{
\text{simulate the physical complex transverse magnetization } M_x+iM_y
}
$$

Thus, for a positive frequency offset,

$$
s(t)=e^{-i\omega t}.
$$

This gives one consistent sign rule for both imaging and spectroscopy.

---

## 7. Relation to MRS literature saying “FFT”

Some MRS literature states that conversion of the discrete time-domain FID signal into a spectrum is performed using a discrete Fourier transformation such as the FFT, and that conversion from the spectral domain back to the time domain uses the inverse Fourier transform.

MR-zero deliberately differs from that numerical statement.

In MR-zero:

$$
s(t)=e^{-i\omega t}.
$$

Therefore, a peak at \\(+\omega\\) is obtained using:

$$
\rho(\omega) = \int s(t)e^{+i\omega t}dt,
$$

which is an iFFT-type operation.

So the MR-zero documentation should state explicitly:

$$
\boxed{
\text{MR-zero differs from the common MRS “FID}\rightarrow\text{spectrum = FFT” wording}
}
$$

This is not a disagreement about the physics of MR.

It is a disagreement about which complex signal is called “the FID.”

The MRS-processing convention usually assumes that the stored complex FID already follows a receiver/storage convention like:

$$
S_{\text{stored}}(t)\sim e^{+i\Omega t},
$$

so that a standard forward FFT places positive offsets at positive spectral frequencies.

MR-zero instead simulates:

$$
S_{\text{MR-zero}}(t)\sim e^{-i\Omega t}.
$$

Thus, one possible conversion between the two conventions is:

$$
S_{\text{stored}}(t) = S_{\text{MR-zero}}^*(t).
$$

---

## 8. Receiver and file-format conventions can still change the sign

Scanner data can differ from the MR-zero physical convention.

A receiver chain, reconstruction pipeline, or file converter may store either:

$$
s(t)
$$

or:

$$
s^*(t).
$$

Conjugation flips the exponent sign:

$$
s(t)=e^{-i\omega t}
\quad\Rightarrow\quad
s^*(t)=e^{+i\omega t}.
$$

This changes the apparent reconstruction rule.

A truly iFFT-type reconstruction can look like an FFT-type reconstruction if the stored data have already been conjugated.

Thus conjugation both mirrors the frequency/spatial axis and inverts phase.

---

## 9. Explicit conversion knob

Users who want the common MRS receiver/storage convention can apply:

```python
sig_receiver = np.conj(sig_mrzero)
```

The important design decision is that MR-zero does **not** build the receiver convention into the physical simulation. Instead, the conversion is explicit:

```python
signal = np.conj(signal)
```

---

## 10. Imaging consequences of conjugated k-space

Let the physically expected k-space be:

$$
K(\mathbf k) =\mathcal F\{\rho(\mathbf r)\}.
$$

Then the physically consistent reconstruction is:

$$
\rho(\mathbf r) =\mathcal F^{-1}\{K(\mathbf k)\}.
$$

If the stored scanner data are instead:

$$
K_{\text{stored}}(\mathbf k)=K^*(\mathbf k),
$$

then:

$$
\mathcal F^{-1}\{K_{\text{stored}}\}
$$

gives a flipped and conjugated image, while using the opposite transform can fix the apparent orientation but leaves the phase convention changed.

In practical terms:

```text
ifft(kspace)       -> correct reconstruction if kspace follows the physical convention
fft(kspace)        -> mirrored version of ifft(kspace), same phase after mirroring
ifft(conj(kspace)) -> mirrored image with inverted phase
fft(conj(kspace))  -> orientation may look fixed, but phase is conjugated
```

So if a scanner stores conjugated k-space, using `fft` instead of `ifft` can make the image orientation look correct, while the phase remains opposite to the MR-zero physical phase convention.

This should be treated as a vendor-, sequence-, and file-format-dependent issue rather than a universal law.

---

## 11. Spectroscopy consequence

For an MR-zero FID simulation with water at \\(0\\) Hz and fat at, for example,

$$
\Delta f_{\text{fat}}=-420\ \mathrm{Hz},
$$

the simulated FID should be:

$$
s(t) = A_{\text{water}} + A_{\text{fat}}e^{-i2\pi(-420)t}.
$$

The correct MR-zero spectral reconstruction is then:

```python
spec = np.fft.fftshift(np.fft.ifft(fid))
spec *= len(fid)
```

This gives fat at \\(-420\\) Hz relative to water.

Using `fft` directly would mirror the spectral axis and place fat at \\(+420\\) Hz unless the frequency axis is also reversed or the FID is conjugated.

Equivalent options are:

```python
# MR-zero physical convention
spec = np.fft.fftshift(np.fft.ifft(fid)) * len(fid)

# Equivalent, but using conjugated data
spec = np.fft.fftshift(np.fft.fft(np.conj(fid)))

# Equivalent if one also flips the frequency axis
spec = np.fft.fftshift(np.fft.fft(fid))
freq = -np.fft.fftshift(np.fft.fftfreq(len(fid), dt))
```

Thus:

$$
\boxed{
s(t)=e^{-i2\pi ft} \quad\Rightarrow\quad \text{use iFFT if } f \text{ should appear at } +f}
$$

---

## 12. Practical MR-zero rule

For MR-zero simulations, use the physical convention:

$$
M_{xy}=M_x+iM_y,
\qquad
M_{xy}(t)\sim e^{-i\omega t}.
$$

Then the consistent reconstruction rules are:

| Data | Signal formation | Reconstruction |
|---|---|---|
| Imaging | \\(s(\mathbf k)=\int\rho(\mathbf r)e^{-i\mathbf k\cdot\mathbf r}d\mathbf r\\) | iFFT |
| Spectroscopy | \\(s(t)=\int\rho(\omega)e^{-i\omega t}d\omega\\) | iFFT |
| Positive \\(B_0\\) offset | \\(e^{-i\gamma\Delta B_0t}\\) | negative phase |
| Fat relative to water | \\(\Delta f_{\text{fat}}<0\\) | positive phase |
| MRS receiver/software convention | optional conjugation | `sig = conj(sig)` |
| Standard MRS “FFT” convention | storage/receiver convention | not MR-zero physics |

The core conclusion is:

$$
\boxed{
\text{In MR-zero, both k-space and FID formation use the same }e^{-i(\cdot)}
\text{ physics}
}
$$

Therefore:

$$
\boxed{
\text{Both image reconstruction and FID-to-spectrum reconstruction are iFFT-type operations}
}
$$

If scanner data require `fft` to look correct, this is evidence of a stored-data convention change, most commonly conjugation and/or axis reversal, not a different physical sign law.

---

## 13. Suggested validation experiments

The following experiments can reveal which convention is being used by a simulator, scanner export, or reconstruction package.

### 13.1 Spectroscopy FID test

Simulate:

$$
s(t)=1+ae^{-i2\pi(-420)t}.
$$

Expected MR-zero result:

- `ifft(fid)` gives fat at \\(-420\\) Hz.
- `fft(fid)` gives fat at \\(+420\\) Hz unless the frequency axis is reversed.
- `fft(conj(fid))` gives the same frequency placement as `ifft(fid)` up to scaling and shift conventions.

### 13.2 \\(B_0\\) phase test

Simulate a positive \\(B_0\\) offset:

$$
\Delta B_0>0.
$$

Expected MR-zero phase:

$$
\phi(t)=-\gamma\Delta B_0t.
$$

So positive \\(\Delta B_0\\) should produce negative phase.

### 13.3 Fat/water GRE test

For a GRE acquisition with water and fat:

$$
\Delta f_{\text{fat}}<0.
$$

Expected MR-zero phase:

$$
\phi_{\text{fat-water}}(TE) =-2\pi\Delta f_{\text{fat}}TE>0.
$$

Thus fat should accumulate positive phase relative to water.

### 13.4 Imaging k-space conjugation test

Compare:

```python
img_ifft      = ifftn(kspace)
img_fft       = fftn(kspace)
img_ifft_conj = ifftn(np.conj(kspace))
img_fft_conj  = fftn(np.conj(kspace))
```

Expected behavior:

- `fft(kspace)` mirrors the `ifft(kspace)` image.
- Conjugating k-space inverts phase.
- `fft(conj(kspace))` can appear to fix orientation while preserving the conjugated phase convention.

These tests separate three effects that are often confused:

$$
\boxed{
\text{transform sign}
\quad
\text{vs.}
\quad
\text{axis reversal}
\quad
\text{vs.}
\quad
\text{complex conjugation}
}
$$

---

## 14. Final takeaway

The MR-zero sign change is consistent and should be kept.

$$
\boxed{
\text{MR-zero simulates the physical } M_x+iM_y \text{ signal directly}
}
$$

Therefore:

$$
\boxed{
s(t)=e^{-i\omega t}
}
$$

and:

$$
\boxed{
\text{FID}\rightarrow\text{spectrum uses an iFFT-type transform}
}
$$

This is the same sign logic as spatial MRI encoding:

$$
s(\mathbf k)=\int\rho(\mathbf r)e^{-i\mathbf k\cdot\mathbf r}d\mathbf r
\quad\Rightarrow\quad
\rho(\mathbf r)=\int s(\mathbf k)e^{+i\mathbf k\cdot\mathbf r}d\mathbf k .
$$

In one sentence:

$$
\boxed{
\text{MR-zero keeps the physics sign; receiver, vendor, and MRS-software signs are explicit conversions}
}
$$

---

## References

- Keeler, *Understanding NMR Spectroscopy*:  
  <https://faculty.washington.edu/seattle/physics541/%202010-reading/nmr-spectroscopy.pdf>

- Near et al., MRS preprocessing recommendations, *NMR in Biomedicine*:  
  <https://analyticalsciencejournals.onlinelibrary.wiley.com/doi/10.1002/nbm.4257>

- NumPy FFT conventions:  
  <https://numpy.org/doc/stable/reference/routines.fft.html>

- MRI signal equation notes:  
  <https://larsonlab.github.io/MRI-education-resources/MRI%20Signal%20Equation.html>

- NIfTI-MRS specification:  
  <https://github.com/wtclarke/mrs_nifti_standard/blob/master/specification.MD>
