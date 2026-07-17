package dsp

import "math"

// HilbertTransformer converts a REAL input signal into its analytic (complex)
// representation: the real part is the original signal delayed to match the
// filter's group delay, and the imaginary part is the same signal phase-shifted
// by -90 degrees (the Hilbert transform). The analytic signal x + j*H{x} has
// only positive-frequency content, which is exactly what you want for
// instantaneous amplitude (magnitude), instantaneous phase/frequency, and
// single-sideband (SSB) generation.
//
// The imaginary path is a windowed, odd-length, Type III linear-phase FIR
// Hilbert kernel. A Type III FIR (odd length, ANTISYMMETRIC taps with a zero
// center tap) has a purely imaginary frequency response that is zero at both DC
// and Nyquist and approximates -j*sgn(f) across the passband -- i.e. it delays
// every passband component by 90 degrees. For a passband cosine cos(w n) the
// output imaginary part is therefore sin(w n): together with the delayed real
// part cos(w n) this traces the unit circle e^{jwn}, so |out| is a flat
// envelope. That flat-envelope property is the defining behavior of a good
// Hilbert transformer and the thing the tests key on.
//
// State is a single float32 overlap buffer (overlap-save), identical in shape to
// FIRFilterReal in filter.go -- the input is real, so no struct-of-arrays split
// is needed. Group delay is (numTaps-1)/2 samples; both the real (pure-delay)
// and imaginary (FIR) paths incur exactly that delay, so they stay aligned.
type HilbertTransformer struct {
	// taps holds the antisymmetric Type III Hilbert kernel used for the
	// imaginary (90-degree) path. The center tap is exactly zero.
	taps []float32
	// centerIndex is the group-delay tap position, (numTaps-1)/2. The real
	// output is a pure passthrough of the sample sitting at this position in the
	// sliding window, so the real path is delayed by exactly the FIR group delay
	// and needs no multiply-accumulate of its own.
	centerIndex int
	// overlap holds the last (numTaps-1) input samples from the previous block
	// so the FIR window can span the block boundary (overlap-save).
	overlap []float32
	// buf is a reusable [overlap | input] scratch buffer, grown as needed.
	buf []float32
	// outBuf is a reusable output buffer for ProcessReuse. Its contents are only
	// valid until the next call to Process or ProcessReuse.
	outBuf []complex64
}

// NewHilbertTransformer builds a Type III FIR Hilbert transformer with numTaps
// taps. numTaps is forced ODD (incremented if even), exactly like DesignLPF,
// because a Type III linear-phase kernel with a zero center tap and integer
// group delay (numTaps-1)/2 only exists for odd lengths.
//
// The ideal (unwindowed) Hilbert impulse response, indexed by offset m from the
// center tap, is:
//
//	h[m] = 0            if m is even (including the center, m = 0)
//	h[m] = 2/(pi*m)     if m is odd
//
// This is antisymmetric (h[-m] = -h[m]) and decays only as 1/m, so truncating it
// to a finite length rings badly (Gibbs) unless it is tapered. We multiply by a
// symmetric window; a symmetric window preserves the antisymmetry (and thus the
// zero DC/Nyquist response and exact 90-degree phase), it only shapes the
// passband ripple and transition width.
//
// The window defaults to HammingWindow: for a Hilbert transformer the dominant
// concern is passband amplitude flatness (ripple shows up directly as envelope
// wobble in |out|), and Hamming gives a good flatness-vs-length tradeoff at
// modest tap counts. Pass BlackmanWindow (or any func(int) []float64) as the
// optional argument to trade a wider transition band near DC/Nyquist for lower
// ripple. Only the first supplied window function is used.
func NewHilbertTransformer(numTaps int, window ...func(int) []float64) *HilbertTransformer {
	if numTaps < 3 {
		numTaps = 3
	}
	if numTaps%2 == 0 {
		numTaps++
	}

	win := HammingWindow
	if len(window) > 0 && window[0] != nil {
		win = window[0]
	}
	w := win(numTaps)

	center := (numTaps - 1) / 2
	taps := make([]float32, numTaps)
	for i := 0; i < numTaps; i++ {
		m := i - center // signed offset from the center tap
		// Even offsets (including the center) are exactly zero in the ideal
		// Hilbert response; leaving them at zero also keeps the kernel perfectly
		// antisymmetric so the DC and Nyquist responses stay at zero.
		if m%2 == 0 {
			continue
		}
		// The ideal Hilbert response is 2/(pi*m), but firDotReal (like the rest of
		// this package's FIR path) computes a CORRELATION sum(taps[j]*win[j]), not a
		// convolution. For an antisymmetric kernel correlation equals -convolution,
		// so applying the textbook 2/(pi*m) directly would deliver -H{x} = -sin for a
		// passband cosine (the lower-sideband conjugate e^{-jwn}). Negating the taps
		// here makes the correlation reproduce the textbook convolution, so the
		// imaginary path is +H{x} = +sin and out = x + j*H{x} carries only
		// positive-frequency content, exactly as the struct doc promises.
		h := -2.0 / (math.Pi * float64(m))
		taps[i] = float32(h * w[i])
	}

	return &HilbertTransformer{
		taps:        taps,
		centerIndex: center,
		overlap:     make([]float32, numTaps-1),
	}
}

// Process transforms input into its analytic signal, returning a freshly
// allocated slice of the same length. out[i] = complex(realDelayed, imag) where
// realDelayed is input delayed by the group delay and imag is its 90-degree
// phase shift. See processInto for the transient/alignment details.
func (h *HilbertTransformer) Process(input []float32) []complex64 {
	out := make([]complex64, len(input))
	h.processInto(input, out)
	return out
}

// ProcessReuse transforms input and returns a slice backed by an internal
// buffer that is overwritten on the next call to Process or ProcessReuse on this
// transformer. Use it for intermediate results consumed before the next call to
// avoid one len(input)-sized allocation per block on hot paths.
func (h *HilbertTransformer) ProcessReuse(input []float32) []complex64 {
	if cap(h.outBuf) < len(input) {
		h.outBuf = make([]complex64, len(input))
	}
	out := h.outBuf[:len(input)]
	h.processInto(input, out)
	return out
}

// processInto stages [overlap | input] into the scratch buffer and, for every
// output i, produces the analytic sample:
//
//	imag        = firDotReal(taps, win)   // the SIMD-accelerated 90-degree path
//	realDelayed = win[centerIndex]        // pure passthrough at the group-delay tap
//	out[i]      = complex(realDelayed, imag)
//
// The real path is a PURE DELAY: win[centerIndex] is exactly the input sample
// that entered (numTaps-1)/2 samples ago, which is the same group delay the FIR
// imposes on the imaginary path. Reading it directly (instead of convolving with
// a delta) keeps the two paths perfectly group-delay aligned and costs nothing.
//
// Transient: the FIR group delay is (numTaps-1)/2 samples. Because overlap-save
// seeds the history with zeros, the sliding window still straddles that initial
// zero state for the first (numTaps-1) outputs of a fresh (or just-Reset)
// transformer; only from index numTaps-1 onward is |out| a fully settled, flat
// envelope. Callers measuring instantaneous amplitude/phase should discard that
// leading transient.
func (h *HilbertTransformer) processInto(input []float32, out []complex64) {
	nTaps := len(h.taps)
	overlap := nTaps - 1
	need := overlap + len(input)

	if cap(h.buf) < need {
		h.buf = make([]float32, need)
	}
	buf := h.buf[:need]
	copy(buf, h.overlap)
	copy(buf[overlap:], input)

	for i := range input {
		win := buf[i : i+nTaps]
		imag := firDotReal(h.taps, win)
		out[i] = complex(win[h.centerIndex], imag)
	}

	copy(h.overlap, buf[len(buf)-overlap:])
}

// Reset zeroes the overlap history so the next block starts from a clean state.
func (h *HilbertTransformer) Reset() {
	for i := range h.overlap {
		h.overlap[i] = 0
	}
}
