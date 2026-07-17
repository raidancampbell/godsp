package dsp

import "math"

// DCBlocker is a one-pole/one-zero IIR high-pass that removes the DC bias (or
// slowly varying carrier offset) from a real signal, leaving everything above a
// low corner essentially untouched. It is the classic "DC blocker":
//
//	H(z) = (1 - z^-1) / (1 - R*z^-1)
//	y[n] = x[n] - x[n-1] + R*y[n-1]
//
// POLE/ZERO PLACEMENT AND STABILITY (why R must live in (0,1)):
//   - There is a zero fixed at z = 1, i.e. at DC (w = 0). H(1) = 0/(1-R) = 0, so
//     DC is rejected exactly regardless of R.
//   - There is a pole at z = R on the positive real axis. For 0 < R < 1 the pole
//     is strictly inside the unit circle, so the filter is BIBO stable and its
//     transient decays as R^n (time constant ~ 1/(1-R) samples).
//   - As R -> 1 the pole creeps toward the zero at z = 1: the DC notch gets
//     narrower (lower corner) and the transient settles more slowly.
//   - At R = 1 the pole lands exactly ON the unit circle, coinciding with the
//     zero. That is only MARGINALLY stable: it behaves like an integrator whose
//     state never decays, so in finite precision rounding error accumulates and
//     DC is no longer reliably removed. R = 1 is therefore forbidden.
//   - R <= 0 is not a DC-blocker corner in the intended sense (R = 0 degenerates
//     to a pure first-difference differentiator y = x - x[-1]), so it is rejected
//     too.
//
// Constructors clamp any R outside (0,1) to a safe default rather than panic,
// matching the constructor style used elsewhere in this package (see NewIQAGC).
type DCBlocker struct {
	x1, y1 float32 // previous input and output samples (filter state)
	r      float32 // pole radius; must satisfy 0 < r < 1
}

// defaultDCBlockerR is the fallback pole radius used when a caller supplies an
// out-of-range R. 0.995 puts the corner at roughly 0.08% of the sample rate
// (about 38 Hz at 48 kHz), a common general-purpose DC-blocker setting.
const defaultDCBlockerR = 0.995

// NewDCBlocker returns a DC blocker with pole radius r. r must satisfy
// 0 < r < 1; any value at or outside those bounds (in particular the
// marginally stable r >= 1) is clamped to defaultDCBlockerR. Larger r (closer
// to 1) gives a lower corner frequency but a longer settling transient.
func NewDCBlocker(r float32) *DCBlocker {
	if r <= 0 || r >= 1 {
		r = defaultDCBlockerR
	}
	return &DCBlocker{r: r}
}

// NewDCBlockerHz returns a DC blocker whose -3 dB corner sits near cutoffHz,
// using the small-angle mapping
//
//	R = 1 - 2*pi*cutoffHz/sampleRate.
//
// This follows from the one-pole high-pass corner w_c ~ (1 - R) radians/sample
// (so f_c ~ (1-R)*fs/(2*pi) Hz) and is only accurate for cutoffHz << sampleRate,
// where the linearization 1 - R ~ 2*pi*fc/fs holds. For larger fractions of the
// sample rate the true corner drifts from cutoffHz; use a Butterworth HPF if you
// need an accurate high corner. A cutoffHz that drives R out of (0,1) (for
// example a negative or near-Nyquist cutoff) is clamped by NewDCBlocker.
func NewDCBlockerHz(cutoffHz, sampleRate float64) *DCBlocker {
	// Coefficient math is done in float64 for precision, then narrowed once.
	r := 1.0 - 2.0*math.Pi*cutoffHz/sampleRate
	return NewDCBlocker(float32(r))
}

// Process filters a single sample: y[n] = x[n] - x[n-1] + R*y[n-1].
func (b *DCBlocker) Process(x float32) float32 {
	y := x - b.x1 + b.r*b.y1
	b.x1 = x
	b.y1 = y
	return y
}

// ProcessBlock filters buf in place. It is bit-for-bit identical to calling
// Process on each element in turn -- the same y = x - x1 + r*y1 arithmetic in
// the same evaluation order -- but hoists the coefficient and the two state
// words into locals so they stay in registers across the whole block instead of
// being reloaded from the struct every sample. The recursion (each output feeds
// the next) is inherently sequential, so this stays scalar; there is no FIR-style
// dot product to vectorize here.
func (b *DCBlocker) ProcessBlock(buf []float32) {
	r := b.r
	x1, y1 := b.x1, b.y1
	for i, x := range buf {
		y := x - x1 + r*y1
		x1 = x
		y1 = y
		buf[i] = y
	}
	b.x1, b.y1 = x1, y1
}

// Reset zeroes the filter state (previous input and output).
func (b *DCBlocker) Reset() {
	b.x1, b.y1 = 0, 0
}

// IQDCBlocker removes a DC offset from a complex baseband stream by running one
// independent DCBlocker on the in-phase (I) rail and another on the quadrature
// (Q) rail. This is the direct-conversion (zero-IF) SDR use case: LO leakage and
// ADC bias put a fixed spike at 0 Hz (the center of the complex spectrum), and
// blocking DC on each rail strips that spike while leaving off-center signals
// intact. The two rails share the same pole radius R but keep separate state, so
// an asymmetric I/Q bias is handled correctly.
type IQDCBlocker struct {
	i, q DCBlocker
}

// NewIQDCBlocker returns a complex DC blocker with pole radius r applied
// identically to both rails. r is validated/clamped by NewDCBlocker, so the same
// 0 < r < 1 rule (and the same default for out-of-range input) applies here.
func NewIQDCBlocker(r float32) *IQDCBlocker {
	proto := NewDCBlocker(r)
	return &IQDCBlocker{i: *proto, q: *proto}
}

// ProcessInPlace filters samples in place, running the one-pole DC blocker
// independently on the real (I) and imaginary (Q) parts of each sample. State
// for both rails is hoisted into locals for the duration of the call, matching
// DCBlocker.ProcessBlock; per rail the arithmetic is bit-identical to Process.
func (d *IQDCBlocker) ProcessInPlace(samples []complex64) {
	r := d.i.r
	ix1, iy1 := d.i.x1, d.i.y1
	qx1, qy1 := d.q.x1, d.q.y1
	for n, s := range samples {
		xi, xq := real(s), imag(s)
		yi := xi - ix1 + r*iy1
		ix1, iy1 = xi, yi
		yq := xq - qx1 + r*qy1
		qx1, qy1 = xq, yq
		samples[n] = complex(yi, yq)
	}
	d.i.x1, d.i.y1 = ix1, iy1
	d.q.x1, d.q.y1 = qx1, qy1
}

// Reset zeroes the state of both rails.
func (d *IQDCBlocker) Reset() {
	d.i.Reset()
	d.q.Reset()
}
