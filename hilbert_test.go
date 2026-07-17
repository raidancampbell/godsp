package dsp

import (
	"math"
	"testing"
)

// The FIR group delay is (numTaps-1)/2 samples. Because overlap-save seeds the
// history with zeros, the sliding window straddles that initial zero state for
// the first (numTaps-1) outputs, so the analytic signal is only fully settled
// from index numTaps-1 onward. Every test below discards a leading transient of
// at least numTaps-1 samples before asserting.

// TestHilbertFlatEnvelope is the defining Hilbert test: for a pure cosine well
// inside the passband, the analytic signal magnitude |out| must be a flat
// envelope. The real path is a pure passthrough of the (delayed) cosine and the
// imaginary path is the 90-degree-shifted sine of the same amplitude, so
// real^2 + imag^2 traces a constant. Ripple in |out| is exactly the deviation of
// the windowed Hilbert passband gain from unity; at f=0.1 (normalized) with 65
// Hamming-windowed taps that deviation is a couple percent.
func TestHilbertFlatEnvelope(t *testing.T) {
	const (
		numTaps = 65
		f0      = 0.1 // cycles/sample, well inside (0, 0.5) and away from band edges
		n       = 4096
	)
	w := 2 * math.Pi * f0

	in := make([]float32, n)
	for i := range in {
		in[i] = float32(math.Cos(w * float64(i)))
	}

	h := NewHilbertTransformer(numTaps)
	out := h.Process(in)

	transient := numTaps - 1 // group delay is (numTaps-1)/2; be generous, discard 2x

	// Mean magnitude over the settled region.
	var sum float64
	count := 0
	for i := transient; i < n; i++ {
		sum += math.Hypot(float64(real(out[i])), float64(imag(out[i])))
		count++
	}
	mean := sum / float64(count)
	if mean < 0.5 {
		t.Fatalf("mean magnitude %g unexpectedly small; passband gain collapsed", mean)
	}

	// Max fractional deviation from the mean must stay within a few percent.
	const tol = 0.03
	var maxDev float64
	for i := transient; i < n; i++ {
		mag := math.Hypot(float64(real(out[i])), float64(imag(out[i])))
		dev := math.Abs(mag-mean) / mean
		if dev > maxDev {
			maxDev = dev
		}
	}
	if maxDev > tol {
		t.Fatalf("envelope not flat: max fractional deviation %g exceeds %g (mean %g)", maxDev, tol, mean)
	}
}

// TestHilbertQuadrature verifies the 90-degree phase relationship directly: the
// real output tracks the (delayed) input cosine, and the imaginary output tracks
// the same-frequency sine (in quadrature). This subsumes the flat-envelope
// property but pins the phase explicitly.
//
// This test HARD-CODES the sign of the imaginary path to +sin(theta) rather than
// sniffing it from the output. That pins the primitive's documented convention:
// the analytic signal is x + j*H{x} = cos + j*sin = e^{+j*theta}, carrying only
// positive-frequency content. The tap negation in NewHilbertTransformer exists
// precisely so the correlation-based FIR path delivers +sin here; a regression to
// the raw 2/(pi*m) kernel would flip the sideband to -sin and this assert would
// catch it (a sign-sniffing test would not).
func TestHilbertQuadrature(t *testing.T) {
	const (
		numTaps = 65
		f0      = 0.1
		n       = 4096
	)
	w := 2 * math.Pi * f0
	center := (numTaps - 1) / 2 // group delay in samples

	in := make([]float32, n)
	for i := range in {
		in[i] = float32(math.Cos(w * float64(i)))
	}

	h := NewHilbertTransformer(numTaps)
	out := h.Process(in)

	transient := numTaps - 1

	const tol = 0.05
	for i := transient; i < n; i++ {
		theta := w * float64(i-center)
		expReal := math.Cos(theta)
		// Positive-frequency analytic signal: imaginary part is +sin(theta), not
		// -sin. See the struct doc (out = x + j*H{x} = e^{+j*theta}).
		expImag := math.Sin(theta)
		gotReal := float64(real(out[i]))
		gotImag := float64(imag(out[i]))
		// Real path is a pure delayed passthrough, so it should match the
		// delayed cosine to near float32 precision.
		if math.Abs(gotReal-expReal) > 1e-4 {
			t.Fatalf("i=%d real=%g want delayed cos=%g", i, gotReal, expReal)
		}
		// Imaginary path is the +90-degree analytic sine within window ripple.
		if math.Abs(gotImag-expImag) > tol {
			t.Fatalf("i=%d imag=%g want +sin=%g (quadrature)", i, gotImag, expImag)
		}
	}
}

// TestHilbertKillsDC verifies that a constant (DC) input produces ~zero
// imaginary output. The kernel is antisymmetric, so once the sliding window is
// fully inside the constant region the FIR dot product is the constant times
// sum(taps), and sum(taps) is exactly zero by antisymmetry -- the Hilbert
// transform has a null at DC.
func TestHilbertKillsDC(t *testing.T) {
	const (
		numTaps = 65
		n       = 512
	)
	in := make([]float32, n)
	for i := range in {
		in[i] = 1.0
	}

	h := NewHilbertTransformer(numTaps)
	out := h.Process(in)

	// Once i >= numTaps-1 the window holds only the constant, so imag must
	// vanish to rounding. The real path still passes the constant through.
	for i := numTaps - 1; i < n; i++ {
		if math.Abs(float64(imag(out[i]))) > 1e-4 {
			t.Fatalf("i=%d imag=%g, expected ~0 (Hilbert nulls DC)", i, imag(out[i]))
		}
		if math.Abs(float64(real(out[i]))-1.0) > 1e-4 {
			t.Fatalf("i=%d real=%g, expected delayed DC passthrough 1.0", i, real(out[i]))
		}
	}
}

// TestHilbertVeryLowFreqSuppressed checks that a very-low-frequency input is
// strongly attenuated on the imaginary (Hilbert) path, since the Hilbert
// passband gain rolls off toward zero near DC. The real passthrough retains full
// amplitude, so imaginary energy should be a small fraction of real energy.
func TestHilbertVeryLowFreqSuppressed(t *testing.T) {
	const (
		numTaps = 65
		f0      = 0.002 // very close to DC
		n       = 8192
	)
	w := 2 * math.Pi * f0

	in := make([]float32, n)
	for i := range in {
		in[i] = float32(math.Cos(w * float64(i)))
	}

	h := NewHilbertTransformer(numTaps)
	out := h.Process(in)

	transient := numTaps - 1
	var sumImag2, sumReal2 float64
	for i := transient; i < n; i++ {
		re := float64(real(out[i]))
		im := float64(imag(out[i]))
		sumReal2 += re * re
		sumImag2 += im * im
	}
	ratio := math.Sqrt(sumImag2 / sumReal2)
	if ratio > 0.2 {
		t.Fatalf("low-freq imaginary energy ratio %g too high; Hilbert should suppress near-DC", ratio)
	}
}

// TestHilbertReset verifies Reset clears overlap state so a fresh block after
// Reset is bit-identical to processing that block on a brand-new transformer.
func TestHilbertReset(t *testing.T) {
	const numTaps = 65
	w := 2 * math.Pi * 0.1

	block := make([]float32, 256)
	for i := range block {
		block[i] = float32(math.Cos(w * float64(i)))
	}

	h := NewHilbertTransformer(numTaps)
	// Dirty the overlap state with an unrelated first block.
	noise := make([]float32, 256)
	for i := range noise {
		noise[i] = float32(math.Sin(0.37 * float64(i)))
	}
	h.Process(noise)
	h.Reset()
	got := h.Process(block)

	fresh := NewHilbertTransformer(numTaps)
	want := fresh.Process(block)

	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("i=%d after Reset got %v, fresh got %v", i, got[i], want[i])
		}
	}
}

// TestHilbertProcessReuseMatchesProcess verifies ProcessReuse produces the same
// values as Process for identical input streams (only the backing storage
// differs), and that numTaps is forced odd.
func TestHilbertProcessReuseMatchesProcess(t *testing.T) {
	// Even tap count must be bumped to odd so the Type III kernel has an
	// integer group delay and a real center tap.
	h := NewHilbertTransformer(64)
	if len(h.taps)%2 == 0 {
		t.Fatalf("numTaps not forced odd: got %d taps", len(h.taps))
	}
	if h.taps[h.centerIndex] != 0 {
		t.Fatalf("center tap must be zero for Type III kernel, got %g", h.taps[h.centerIndex])
	}

	w := 2 * math.Pi * 0.15
	in := make([]float32, 300)
	for i := range in {
		in[i] = float32(math.Cos(w * float64(i)))
	}

	a := NewHilbertTransformer(65)
	b := NewHilbertTransformer(65)
	want := a.Process(in)
	got := b.ProcessReuse(in)
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("i=%d ProcessReuse %v != Process %v", i, got[i], want[i])
		}
	}
}

// TestHilbertStreamingContinuity verifies that overlap-save carries state across
// block boundaries correctly: feeding a signal as one big block must produce the
// same settled output as feeding the identical signal in several UNEVEN blocks.
// TestHilbertReset does not exercise this -- both of its paths start with a fresh
// (zero) overlap, so it can pass even if the overlap were mishandled after the
// first block. Here the chunked transformer must correctly seed each block's
// window from the tail of the previous block.
func TestHilbertStreamingContinuity(t *testing.T) {
	const numTaps = 65
	w := 2 * math.Pi * 0.1

	// Uneven chunk sizes that sum to n; deliberately include a chunk smaller than
	// numTaps and a size-1 chunk to stress the boundary seeding.
	chunks := []int{7, 1, numTaps - 2, 200, 3, 137, 96}
	n := 0
	for _, c := range chunks {
		n += c
	}

	in := make([]float32, n)
	for i := range in {
		in[i] = float32(math.Cos(w * float64(i)))
	}

	whole := NewHilbertTransformer(numTaps)
	want := whole.Process(in)

	streamed := NewHilbertTransformer(numTaps)
	got := make([]complex64, 0, n)
	off := 0
	for _, c := range chunks {
		got = append(got, streamed.Process(in[off:off+c])...)
		off += c
	}

	// Overlap-save is exact, so the settled region must match bit-for-bit
	// regardless of how the stream was chunked.
	for i := numTaps - 1; i < n; i++ {
		if got[i] != want[i] {
			t.Fatalf("i=%d streamed %v != whole-block %v", i, got[i], want[i])
		}
	}
}

// TestHilbertKillsNyquist mirrors TestHilbertKillsDC at the other spectral null.
// The alternating sequence (-1)^i is a cosine at Nyquist (f=0.5). A Type III FIR
// (odd length, antisymmetric taps) has an exact zero at BOTH DC and Nyquist, so
// once the window holds only the alternating sequence the imaginary (Hilbert)
// output must vanish to rounding. The real passthrough still carries the +/-1
// alternation.
func TestHilbertKillsNyquist(t *testing.T) {
	const (
		numTaps = 65
		n       = 512
	)
	in := make([]float32, n)
	for i := range in {
		in[i] = float32(1 - 2*(i&1)) // (-1)^i: +1, -1, +1, -1, ...
	}

	h := NewHilbertTransformer(numTaps)
	out := h.Process(in)

	for i := numTaps - 1; i < n; i++ {
		if math.Abs(float64(imag(out[i]))) > 1e-4 {
			t.Fatalf("i=%d imag=%g, expected ~0 (Hilbert nulls Nyquist)", i, imag(out[i]))
		}
		want := float64(1 - 2*(i&1))
		if math.Abs(float64(real(out[i]))-want) > 1e-4 {
			t.Fatalf("i=%d real=%g, expected delayed alternation %g", i, real(out[i]), want)
		}
	}
}
