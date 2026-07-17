package dsp

import "math"

// Goertzel evaluates the DFT at a SINGLE target frequency in O(N) time and O(1)
// state, without computing a full FFT. It is the detector counterpart to
// DesignNotch: where the notch REMOVES a CTCSS/PL sub-audible tone from demod
// audio, a Goertzel MEASURES the energy at that tone so you can decide whether
// it is present. For one bin it is far cheaper than an FFT (no O(N log N)
// butterflies, no bit-reversal, no N-length work buffer), which is exactly the
// regime tone detection lives in: you care about one frequency, not the whole
// spectrum.
//
// Bin snapping (the design choice made here). The recurrence coefficient is
//
//	coeff = 2*cos(2*pi*k/N)   with   k = round(targetHz/fs * N)
//
// i.e. the target is SNAPPED to the nearest integer DFT bin k. This is the
// classic (integer-k) Goertzel, chosen over the "generalized" (real-k) variant
// for two reasons: (1) at an integer k the algorithm computes EXACTLY the same
// complex value as FFT bin k of the same N samples, so Complex()/Power() are
// directly cross-checkable against an FFT and behave with textbook selectivity;
// (2) it needs no post-hoc phase-correction factor. The tradeoff is frequency
// resolution: the effective analysis frequency is the bin center k*fs/N, not
// targetHz. If targetHz falls between bins the tone's energy is split across
// neighbours (spectral leakage) and measured power drops. For best selectivity
// choose the block length N so that targetHz*N/fs lands on (or very near) an
// integer -- BinFreq reports the frequency actually analyzed. The generalized
// real-k variant would hit targetHz exactly but trades that against needing a
// complex correction term and no longer aligning with an FFT bin; for
// fixed-tone detection (CTCSS, DTMF, pilot tones) snapping is the better fit.
//
// Streaming contract. The classic usage is block-at-a-time: feed EXACTLY
// blockLen samples (via Process or Update), read Power()/Complex(), then Reset()
// before the next block. The state is a marginally stable resonator (poles on
// the unit circle), so it is only meaningful to interpret the accumulators after
// the intended N samples; letting it run unbounded lets rounding error grow.
// Reset() returns it to a clean slate for the next detection window.
type Goertzel struct {
	// coeff = 2*cos(w) is the single multiply of the resonator recurrence.
	// cosw/sinw are cached separately so Complex() can form exp(+jw) without a
	// second trig call per readout.
	coeff float64
	cosw  float64
	sinw  float64

	// s1 holds s[n-1], s2 holds s[n-2] of the recurrence. They are float64 (not
	// float32) on purpose: the resonator's poles sit ON the unit circle, so it
	// is only marginally stable and rounding error is NOT damped out -- it
	// accumulates across the block. Over a long N (thousands of samples) a
	// float32 accumulator would visibly lose the low-order bits of the running
	// sum and bias the measured power. float64 keeps ~15 significant digits, so
	// the O(N) accumulation stays accurate. Input samples are still float32
	// (the library's working type); only the accumulator is widened.
	s1, s2 float64

	k  int     // snapped bin index
	n  int     // blockLen the coefficient was designed for
	fs float64 // sample rate, for BinFreq
}

// NewGoertzel builds a detector for targetHz at sampleRate, tuned for blocks of
// blockLen samples. It panics if targetHz is not strictly inside (0, fs/2) --
// DC and frequencies at or above Nyquist are not representable tones -- or if
// blockLen < 1. It also panics if targetHz or sampleRate is NaN/Inf (those slip
// past ordered comparisons, since every comparison against NaN is false), or if
// the SNAPPED bin k lands on DC (k=0) or Nyquist and above (k >= blockLen/2) --
// which happens when blockLen is too small to resolve targetHz, and would
// otherwise silently turn the detector into a DC/Nyquist energy accumulator.
// targetHz is snapped to the nearest DFT bin k = round(targetHz * blockLen /
// fs); see the type doc for the resolution tradeoff and BinFreq for the
// frequency actually analyzed.
func NewGoertzel(targetHz, sampleRate float64, blockLen int) *Goertzel {
	// Reject non-finite inputs FIRST: NaN fails every ordered comparison (so the
	// (0, fs/2) and > 0 guards below would let a NaN through), and an Inf would
	// propagate through the bin math into a garbage coefficient.
	if math.IsNaN(targetHz) || math.IsInf(targetHz, 0) {
		panic("NewGoertzel: targetHz must be finite")
	}
	if math.IsNaN(sampleRate) || math.IsInf(sampleRate, 0) {
		panic("NewGoertzel: sampleRate must be finite")
	}
	if sampleRate <= 0 {
		panic("NewGoertzel: sampleRate must be > 0")
	}
	if targetHz <= 0 || targetHz >= sampleRate/2 {
		panic("NewGoertzel: targetHz must satisfy 0 < targetHz < sampleRate/2")
	}
	if blockLen < 1 {
		panic("NewGoertzel: blockLen must be >= 1")
	}
	k := int(math.Round(targetHz * float64(blockLen) / sampleRate))
	// The (0, fs/2) guard above rejects DC and Nyquist in CONTINUOUS terms, but
	// after snapping to the nearest integer bin k can still collapse onto k=0 (DC)
	// or k>=N/2 (Nyquist and above) when blockLen is too small to separate the
	// tone from those edges. Bins 1..N/2-1 are the only representable single
	// tones, so honor the documented contract here instead of degrading into a
	// DC/Nyquist energy accumulator.
	if k < 1 || k >= blockLen/2 {
		panic("NewGoertzel: targetHz snaps to DC or Nyquist bin; blockLen too small to resolve this tone")
	}
	// w is the bin's angular frequency in radians/sample. Using the snapped
	// integer k (not targetHz directly) is what makes this bin-exact vs an FFT.
	w := 2.0 * math.Pi * float64(k) / float64(blockLen)
	return &Goertzel{
		coeff: 2.0 * math.Cos(w),
		cosw:  math.Cos(w),
		sinw:  math.Sin(w),
		k:     k,
		n:     blockLen,
		fs:    sampleRate,
	}
}

// Process feeds one real sample into the resonator recurrence
//
//	s0 = coeff*s1 - s2 + x ;  s2 = s1 ;  s1 = s0
//
// which is the second-order IIR at the heart of the algorithm. Feed blockLen
// samples, then read Power()/Complex(). For a whole block prefer Update, which
// keeps the state in registers across the loop.
func (g *Goertzel) Process(x float32) {
	s0 := g.coeff*g.s1 - g.s2 + float64(x)
	g.s2 = g.s1
	g.s1 = s0
}

// Update runs the recurrence over an entire block. It is bit-for-bit identical
// to calling Process on each element in turn -- same float64 arithmetic, same
// order -- but hoists coeff and the two state words into locals so they live in
// registers for the whole loop instead of being reloaded from the struct per
// sample (the same optimization Biquad.ProcessBlock uses). The recurrence has a
// serial dependency (each s0 needs the previous two states), so unlike the FIR
// dot-product kernels it cannot be vectorized into a firDotReal-style reduction;
// register hoisting is the applicable win.
//
// Neither Update nor Process validates that exactly blockLen samples are fed --
// the recurrence happily runs over any count. Feeding the wrong number silently
// changes the accumulated energy (Power scales with the samples seen), so the
// "feed exactly blockLen, then read, then Reset" contract is the caller's to
// enforce.
func (g *Goertzel) Update(buf []float32) {
	coeff := g.coeff
	s1, s2 := g.s1, g.s2
	for _, x := range buf {
		s0 := coeff*s1 - s2 + float64(x)
		s2 = s1
		s1 = s0
	}
	g.s1, g.s2 = s1, s2
}

// Power returns the magnitude-squared of the DFT at the target bin,
//
//	|X[k]|^2 = s1^2 + s2^2 - coeff*s1*s2
//
// This closed form drops out of |exp(jw)*s1 - s2|^2 (see Complex): the cross
// term is -2*cos(w)*s1*s2 = -coeff*s1*s2, and the sin^2+cos^2 on s1 collapses to
// s1^2. It avoids the final complex multiply, so if you only need energy (the
// usual tone-present/absent decision) this is all you compute. The value is
// non-negative in exact arithmetic; callers comparing tiny leakage floors may
// clamp at zero to guard against a slightly negative rounding result.
func (g *Goertzel) Power() float64 {
	return g.s1*g.s1 + g.s2*g.s2 - g.coeff*g.s1*g.s2
}

// Complex returns the full complex DFT value X[k] for the block fed so far,
// using the same forward convention as FFT (X[k] = sum x[n]*exp(-j*2*pi*k*n/N)):
//
//	X[k] = exp(+jw)*s1 - s2 = (cos(w)*s1 - s2) + j*(sin(w)*s1)
//
// The +jw (not -jw) is correct for the -j forward transform: the final FIR
// stage of the Goertzel filter contributes exp(+jw) once the recurrence's own
// exp(-jw) pole is accounted for. Use this when you need phase (e.g. coherent
// combining or DTMF twist checks); use Power when magnitude-squared suffices.
func (g *Goertzel) Complex() complex128 {
	return complex(g.cosw*g.s1-g.s2, g.sinw*g.s1)
}

// Bin returns the integer DFT bin index k the detector was snapped to.
func (g *Goertzel) Bin() int { return g.k }

// BinFreq returns the frequency actually analyzed, the snapped bin center
// k*fs/N in Hz. This can differ from the requested targetHz by up to half a bin
// (fs/(2N)); it is the true center of the detector's response.
func (g *Goertzel) BinFreq() float64 {
	return float64(g.k) * g.fs / float64(g.n)
}

// Reset zeroes the resonator state so the next block starts clean. Call it
// between detection windows (the coefficient, bin, and rate are preserved).
func (g *Goertzel) Reset() {
	g.s1, g.s2 = 0, 0
}
