package dsp

import "math"

// ComplexFIRFilter applies a non-decimating FIR filter with COMPLEX taps to a
// COMPLEX input stream. Unlike DecimatingFilter (which convolves real taps
// against complex data and therefore can only realize a filter whose response
// is conjugate-symmetric about DC), complex taps let the passband sit at an
// arbitrary center frequency with no mirror-image image band. That is exactly
// what DesignComplexBandpass produces: a single-sided band-pass.
//
// State is stored struct-of-arrays (SoA): real and imaginary components live in
// separate float32 slices. This is deliberate. The SIMD-accelerated kernels
// (complexFIRDot on amd64 AVX2 / arm64 NEON) consume two plain float32 windows
// and never deinterleave complex64, so composing two real-tap dot products over
// the SoA window gets full-width vector throughput for free with no new
// assembly. See the math note on process() for the composition.
type ComplexFIRFilter struct {
	// tapsR/tapsI hold the complex taps in SoA form (real, imag separately).
	tapsR []float32
	tapsI []float32
	// overlapR/overlapI hold the last (nTaps-1) complex input samples (SoA),
	// the overlap-save history carried between successive Process calls so that
	// streaming in arbitrary chunk sizes yields the same output as one block.
	overlapR []float32
	overlapI []float32
	// bufR/bufI are reusable scratch buffers laid out [overlap | input] (SoA),
	// grown as needed. Staging the input here once lets the inner loop take
	// contiguous windows without re-reading the overlap slices.
	bufR []float32
	bufI []float32
	// outBuf is a reusable output buffer for ProcessReuse. Its contents are only
	// valid until the next call to Process or ProcessReuse.
	outBuf []complex64
}

// NewComplexFIRFilter creates a non-decimating complex-tap FIR filter. The taps
// are copied into SoA storage, so the caller may reuse or mutate the input
// slice afterward.
func NewComplexFIRFilter(taps []complex64) *ComplexFIRFilter {
	// Guard up front: with zero taps the overlap (nTaps-1) would be -1, and
	// process() would later slice buf[need-overlap:] = buf[:+1] over a -1-length
	// window, producing a cryptic out-of-range panic deep in the hot path. Fail
	// here with a clear message instead, matching the DesignLPF/DesignNotch
	// panic-on-bad-args style.
	if len(taps) == 0 {
		panic("NewComplexFIRFilter: taps must be non-empty")
	}
	n := len(taps)
	tapsR := make([]float32, n)
	tapsI := make([]float32, n)
	for i, t := range taps {
		tapsR[i] = real(t)
		tapsI[i] = imag(t)
	}
	overlap := n - 1
	return &ComplexFIRFilter{
		tapsR:    tapsR,
		tapsI:    tapsI,
		overlapR: make([]float32, overlap),
		overlapI: make([]float32, overlap),
	}
}

// process stages [overlap | input] into the SoA scratch, runs the complex FIR
// over every window position, and updates the overlap history. Non-decimating:
// exactly len(input) outputs are produced, output[i] being the filter response
// when input[i] is the newest sample in the window.
//
// The complex convolution y = sum_k t[k]*x[k] with complex tap t=(tr,ti) and
// complex sample x=(xr,xi) expands (via t*x = (tr*xr - ti*xi) + j(tr*xi + ti*xr))
// to:
//
//	Re(y) = sum(tr*xr) - sum(ti*xi)
//	Im(y) = sum(tr*xi) + sum(ti*xr)
//
// The reusable kernel complexFIRDot(taps, winR, winI) returns the pair
// (sum(taps*winR), sum(taps*winI)). Feeding it the real taps then the imag taps
// over the SAME complex window gives:
//
//	(aR, aI) = complexFIRDot(tapsR, winR, winI) = (sum(tr*xr), sum(tr*xi))
//	(bR, bI) = complexFIRDot(tapsI, winR, winI) = (sum(ti*xr), sum(ti*xi))
//
// so Re(y) = aR - bI and Im(y) = aI + bR. Two real-tap dot products per output,
// each SIMD-accelerated, no complex deinterleaving.
func (f *ComplexFIRFilter) process(input []complex64, reuseOutput bool) []complex64 {
	nTaps := len(f.tapsR)
	overlap := nTaps - 1
	need := overlap + len(input)
	if cap(f.bufR) < need {
		f.bufR = make([]float32, need)
		f.bufI = make([]float32, need)
	}
	f.bufR = f.bufR[:need]
	f.bufI = f.bufI[:need]
	copy(f.bufR, f.overlapR)
	copy(f.bufI, f.overlapI)
	inR := f.bufR[overlap:]
	inI := f.bufI[overlap:]
	for i, c := range input {
		inR[i] = real(c)
		inI[i] = imag(c)
	}

	n := len(input)
	var out []complex64
	if reuseOutput {
		if cap(f.outBuf) < n {
			f.outBuf = make([]complex64, n)
		}
		out = f.outBuf[:n]
	} else {
		out = make([]complex64, n)
	}

	bufR := f.bufR
	bufI := f.bufI
	tapsR := f.tapsR
	tapsI := f.tapsI
	// Window i spans buf[i : i+nTaps]; at i == n-1 that is buf[n-1 : need],
	// exactly nTaps elements, so no window ever runs past the staged buffer.
	for i := 0; i < n; i++ {
		winR := bufR[i : i+nTaps]
		winI := bufI[i : i+nTaps]
		aR, aI := complexFIRDot(tapsR, winR, winI)
		bR, bI := complexFIRDot(tapsI, winR, winI)
		out[i] = complex(aR-bI, aI+bR)
	}

	// Save the last (nTaps-1) input samples as the overlap for the next call.
	copy(f.overlapR, bufR[need-overlap:])
	copy(f.overlapI, bufI[need-overlap:])

	if reuseOutput {
		f.outBuf = out
	}
	return out
}

// Process applies the complex FIR filter to the input samples and returns a
// freshly allocated output slice of the same length.
func (f *ComplexFIRFilter) Process(input []complex64) []complex64 {
	return f.process(input, false)
}

// ProcessReuse applies the complex FIR filter and returns a slice backed by an
// internal buffer. The returned slice is only valid until the next call to
// Process or ProcessReuse on this filter. Use this for intermediate results
// (e.g. band-select output fed straight into a demodulator) to avoid a
// per-call allocation on hot streaming paths.
func (f *ComplexFIRFilter) ProcessReuse(input []complex64) []complex64 {
	return f.process(input, true)
}

// Reset clears the overlap-save history so the next Process call behaves as if
// on a freshly constructed filter.
func (f *ComplexFIRFilter) Reset() {
	for i := range f.overlapR {
		f.overlapR[i] = 0
		f.overlapI[i] = 0
	}
}

// DesignComplexBandpass designs complex FIR taps for a band-pass spanning
// [lowHz, highHz] on a stream sampled at sampleRate. It uses the standard
// modulated-prototype construction:
//
//  1. Design a real, unity-DC-gain low-pass prototype whose cutoff is the band's
//     half-bandwidth (highHz-lowHz)/2 via DesignLPF. That prototype passes
//     [-halfBW, +halfBW] around DC.
//  2. Frequency-shift the prototype up to the band center fc = (lowHz+highHz)/2
//     by multiplying tap k by exp(-j*2*pi*fc*k/sampleRate). Modulating a real
//     filter by a complex exponential slides its passband from DC to fc, so the
//     passband becomes [fc-halfBW, fc+halfBW] = [lowHz, highHz].
//
// Note the NEGATIVE modulation sign. ComplexFIRFilter convolves the taps against
// a forward-indexed window (out[i] = sum_k taps[k]*win[k], so taps[nTaps-1]
// multiplies the NEWEST sample), which reverses the effective impulse-response
// ordering relative to the textbook direct form where h[0] hits the newest
// sample. That reversal negates the frequency at which the modulated passband
// lands, so exp(-j...) here (not the +j of the usual upconversion formula) is
// what places the single surviving image at +fc. exp(+j...) would put it at the
// mirror -fc instead. For a real (symmetric) prototype this sign is invisible;
// it matters only once the taps become complex.
//
// Because the taps are complex (no conjugate symmetry), only the +fc image
// survives: there is no mirror band at -fc, which is the whole point of using
// complex taps instead of a real band-pass. The peak gain is ~unity (the
// prototype's DC gain) at the band center. Using k (not a centered index) only
// sets a global phase origin and does not change the magnitude response.
//
// numTaps is forwarded to DesignLPF, which forces it odd for a symmetric
// type-I prototype; the returned slice therefore has an odd length. Panics on
// an invalid band, matching the other designers: lowHz must be >= 0, highHz must
// not exceed Nyquist (sampleRate/2), and lowHz must be strictly below highHz.
func DesignComplexBandpass(lowHz, highHz, sampleRate float64, numTaps int) []complex64 {
	if lowHz < 0 || highHz > sampleRate/2 || lowHz >= highHz {
		panic("DesignComplexBandpass: require 0 <= lowHz < highHz <= sampleRate/2")
	}

	halfBW := (highHz - lowHz) / 2.0
	fc := (lowHz + highHz) / 2.0

	// halfBW is in (0, sampleRate/4] given the validated band, always strictly
	// below Nyquist, so DesignLPF's own cutoff precondition is never violated.
	proto := DesignLPF(halfBW, sampleRate, numTaps)

	taps := make([]complex64, len(proto))
	for k := range proto {
		// Negative modulation sign; see the doc comment for why the forward-
		// indexed convolution requires exp(-j...) to land the band at +fc.
		theta := -2.0 * math.Pi * fc * float64(k) / sampleRate
		h := float64(proto[k])
		taps[k] = complex64(complex(h*math.Cos(theta), h*math.Sin(theta)))
	}
	return taps
}
