package dsp

import (
	"fmt"
	"math"
	"math/cmplx"
	"testing"
)

// genComplexTaps builds n deterministic complex64 taps with distinct real and
// imaginary patterns so the two complexFIRDot compositions (one over the real
// taps, one over the imag taps) exercise independent coefficient sets.
func genComplexTaps(n int) []complex64 {
	taps := make([]complex64, n)
	for i := range taps {
		re := float32((i*17)%29-14) / 31
		im := float32((i*23)%41-20) / 23
		taps[i] = complex(re, im)
	}
	return taps
}

// genComplexInput builds a deterministic complex64 stream distinct from the tap
// pattern so real*real, real*imag, imag*real, imag*imag cross terms are all
// non-trivial.
func genComplexInput(n int) []complex64 {
	in := make([]complex64, n)
	for i := range in {
		re := float32((i*11)%37-18) / 19
		im := float32((i*43)%53-26) / 27
		in[i] = complex(re, im)
	}
	return in
}

// oracleComplexFIR is a naive direct complex128 convolution with the same
// overlap-save semantics as ComplexFIRFilter over the whole stream at once:
// output[i] = sum_{k} taps[k] * padded[i+k] where padded is (nTaps-1) leading
// zeros followed by the input. This is the independent reference; it computes
// in complex128 (double precision) so it does not share the filter's float32
// accumulation rounding.
func oracleComplexFIR(taps, input []complex64) []complex64 {
	nTaps := len(taps)
	overlap := nTaps - 1
	padded := make([]complex128, overlap+len(input))
	for i, c := range input {
		padded[overlap+i] = complex(float64(real(c)), float64(imag(c)))
	}
	out := make([]complex64, len(input))
	for i := range input {
		var acc complex128
		for k := 0; k < nTaps; k++ {
			t := complex(float64(real(taps[k])), float64(imag(taps[k])))
			acc += t * padded[i+k]
		}
		out[i] = complex(float32(real(acc)), float32(imag(acc)))
	}
	return out
}

// assertComplexClose fails if got and want differ by more than the per-sample
// relative tolerance used by TestComplexFIRDotDispatchMatchesScalar:
// tol = 2e-5 * (max(|wantR|,|wantI|) + 1), checked on each component.
func assertComplexClose(t *testing.T, got, want []complex64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("length mismatch: got %d, want %d", len(got), len(want))
	}
	for i := range want {
		wantR, wantI := real(want[i]), imag(want[i])
		gotR, gotI := real(got[i]), imag(got[i])
		scale := abs(wantR)
		if abs(wantI) > scale {
			scale = abs(wantI)
		}
		tol := float32(2e-5) * (scale + 1)
		if abs(gotR-wantR) > tol || abs(gotI-wantI) > tol {
			t.Fatalf("sample %d: got (%g,%g), oracle (%g,%g), tolerance %g", i, gotR, gotI, wantR, wantI, tol)
		}
	}
}

// TestComplexFIRMatchesOracle pins the whole-block output against the
// independent complex128 direct-convolution oracle across tap counts that hit
// both SIMD-width tails: 7 (amd64 7-tap tail, arm64 3-tap tail), 8 (no tail on
// either), 9 (amd64 1-tap tail, arm64 1-tap tail), 32 (no tail), 33 (both
// arches 1-tap tail), plus the degenerate 1-tap case.
func TestComplexFIRMatchesOracle(t *testing.T) {
	for _, nTaps := range []int{1, 7, 8, 9, 32, 33} {
		t.Run(fmt.Sprintf("taps_%d", nTaps), func(t *testing.T) {
			taps := genComplexTaps(nTaps)
			input := genComplexInput(1000)
			got := NewComplexFIRFilter(taps).Process(input)
			want := oracleComplexFIR(taps, input)
			if len(got) != len(input) {
				t.Fatalf("non-decimating output length %d, want %d", len(got), len(input))
			}
			assertComplexClose(t, got, want)
		})
	}
}

// TestComplexFIRSplitMatchesSingle confirms that streaming the input in uneven
// chunks yields the same output as a single block, exercising the overlap-save
// carry between calls. Uses tap counts with non-trivial SIMD tails.
func TestComplexFIRSplitMatchesSingle(t *testing.T) {
	input := genComplexInput(4097)
	for _, nTaps := range []int{9, 33} {
		t.Run(fmt.Sprintf("taps_%d", nTaps), func(t *testing.T) {
			taps := genComplexTaps(nTaps)
			whole := NewComplexFIRFilter(taps).Process(input)

			split := make([]complex64, 0, len(input))
			f := NewComplexFIRFilter(taps)
			pos := 0
			for pos < len(input) {
				size := 1
				if pos >= 10 {
					size = []int{7, 13, 29, 64, 3, 101}[pos%6]
				}
				if size > len(input)-pos {
					size = len(input) - pos
				}
				split = append(split, f.Process(input[pos:pos+size])...)
				pos += size
			}

			if len(split) != len(whole) {
				t.Fatalf("split length %d, single-block length %d", len(split), len(whole))
			}
			assertComplexClose(t, split, whole)
		})
	}
}

// measureComplexGain returns the steady-state magnitude gain |H(f)| of the
// complex FIR filter at frequency freqHz. It drives the filter with a
// unit-magnitude complex exponential exp(j*2*pi*f*n/fs), discards the first
// quarter (filter transient), then takes the RMS of the complex output
// magnitude over the remainder. For a unit-magnitude complex tone through an
// LTI filter the steady-state output is H(f)*input, so RMS(|out|) = |H(f)|.
func measureComplexGain(taps []complex64, freqHz, sampleRate float64, n int) float64 {
	in := make([]complex64, n)
	for i := range in {
		theta := 2.0 * math.Pi * freqHz * float64(i) / sampleRate
		in[i] = complex64(cmplx.Exp(complex(0, theta)))
	}
	out := NewComplexFIRFilter(taps).Process(in)
	start := n / 4
	var sumSq float64
	for i := start; i < n; i++ {
		mag := float64(real(out[i]))*float64(real(out[i])) + float64(imag(out[i]))*float64(imag(out[i]))
		sumSq += mag
	}
	return math.Sqrt(sumSq / float64(n-start))
}

// TestDesignComplexBandpassResponse spot-checks the modulated-prototype
// band-pass: a tone at the band center passes at ~unity gain while tones well
// outside the band (three half-bandwidths below and above center, deep in the
// prototype's Blackman stopband) are strongly attenuated. Confirms the complex
// taps place a single passband at +fc with no surviving mirror band.
func TestDesignComplexBandpassResponse(t *testing.T) {
	const fs = 48000.0
	// Band [8000,12000]: fc=10000, halfBW=2000 (cutoff fraction ~0.042 of fs).
	taps := DesignComplexBandpass(8000, 12000, fs, 201)

	// In-band: exact center should pass at ~unity (prototype DC gain is 1).
	centerGain := measureComplexGain(taps, 10000, fs, 8192)
	if centerGain < 0.9 || centerGain > 1.1 {
		t.Errorf("band-center gain = %.4f, want ~1.0", centerGain)
	}

	// Band edges: lowHz and highHz sit at the prototype's windowed-sinc cutoff,
	// where a Blackman-windowed low-pass is ~-6 dB (|H| ~ 0.5). Pinning both
	// edges near 0.5 constrains the passband WIDTH and PLACEMENT, not just the
	// center: a band mis-sized by up to ~halfBW would still pass a center-only
	// check but would push one edge far from 0.5. Allow a generous window
	// [0.3,0.7] to absorb window-transition and float32 measurement slop.
	for _, edge := range []float64{8000, 12000} {
		g := measureComplexGain(taps, edge, fs, 8192)
		if g < 0.3 || g > 0.7 {
			t.Errorf("band-edge gain at %.0f Hz = %.4f, want ~0.5 (in [0.3,0.7])", edge, g)
		}
	}

	// Just outside the band: lowHz-halfBW/2 (7000) and highHz+halfBW/2 (13000)
	// are half a half-bandwidth past each edge, into the transition/stopband, so
	// they must be clearly below unity. Together with the ~0.5 edges above this
	// forbids a passband that is too wide (which would keep these near unity).
	for _, freq := range []float64{7000, 13000} {
		g := measureComplexGain(taps, freq, fs, 8192)
		if g >= 0.5 {
			t.Errorf("just-out-of-band gain at %.0f Hz = %.4f, want < 0.5", freq, g)
		}
	}

	// Out-of-band: 4000 Hz (6000 below center) and 16000 Hz (6000 above center)
	// are ~3x the cutoff into the stopband and must be strongly attenuated.
	for _, freq := range []float64{4000, 16000} {
		g := measureComplexGain(taps, freq, fs, 8192)
		if g > 0.05 {
			t.Errorf("out-of-band gain at %.0f Hz = %.4f, want < 0.05", freq, g)
		}
	}

	// The negative-frequency mirror of the band center (-10000 Hz, i.e. the
	// image a real band-pass would also pass) must be rejected: this is the
	// defining property of complex taps.
	mirrorGain := measureComplexGain(taps, -10000, fs, 8192)
	if mirrorGain > 0.05 {
		t.Errorf("mirror-band gain at -10000 Hz = %.4f, want < 0.05", mirrorGain)
	}
}

// TestDesignComplexBandpassPanicsOnInvalidBand verifies the band precondition
// guard: negative low edge, high edge above Nyquist, or low >= high are all
// invalid and must panic, matching the other designers.
func TestDesignComplexBandpassPanicsOnInvalidBand(t *testing.T) {
	cases := []struct {
		name          string
		low, high, fs float64
		numTaps       int
	}{
		{"negative_low", -100, 5000, 48000, 101},
		{"high_above_nyquist", 8000, 30000, 48000, 101},
		{"low_equals_high", 8000, 8000, 48000, 101},
		{"low_above_high", 12000, 8000, 48000, 101},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			defer func() {
				if r := recover(); r == nil {
					t.Errorf("expected panic on band [%v,%v] fs=%v", c.low, c.high, c.fs)
				}
			}()
			DesignComplexBandpass(c.low, c.high, c.fs, c.numTaps)
		})
	}
}

// TestNewComplexFIRFilterPanicsOnEmptyTaps verifies the constructor rejects nil
// and empty tap slices up front. Without the guard, overlap (nTaps-1) would be
// -1 and the first Process would panic cryptically ("slice bounds out of range
// [-1:]") deep inside the hot path; the constructor must fail clearly instead.
func TestNewComplexFIRFilterPanicsOnEmptyTaps(t *testing.T) {
	for _, c := range []struct {
		name string
		taps []complex64
	}{
		{"nil", nil},
		{"empty", []complex64{}},
	} {
		t.Run(c.name, func(t *testing.T) {
			defer func() {
				if r := recover(); r == nil {
					t.Errorf("expected panic on %s taps", c.name)
				}
			}()
			NewComplexFIRFilter(c.taps)
		})
	}
}

// TestComplexFIRProcessReuseZeroAlloc proves the steady-state per-call
// allocation count of ProcessReuse is zero once internal buffers are warm,
// mirroring TestDecimatingFilterProcessReuseZeroAlloc.
func TestComplexFIRProcessReuseZeroAlloc(t *testing.T) {
	taps := genComplexTaps(333)
	f := NewComplexFIRFilter(taps)
	input := genComplexInput(2731)
	f.ProcessReuse(input) // warm bufR/bufI/outBuf
	allocs := testing.AllocsPerRun(100, func() {
		_ = f.ProcessReuse(input)
	})
	if allocs != 0 {
		t.Fatalf("ProcessReuse allocs/op = %.2f, want 0", allocs)
	}
}

// TestComplexFIRReset verifies Reset clears the overlap-save history so a reused
// filter behaves identically to a freshly constructed one. It processes a first
// block whose tail is nonzero in BOTH real and imag (so overlapR and overlapI
// both get populated), calls Reset, then processes a second block; the result
// must match a brand-new filter over that same second block. Comparing against a
// fresh filter catches a Reset that clears only one of the two overlap arrays: a
// stale overlapR or overlapI would perturb the first (nTaps-1) outputs.
func TestComplexFIRReset(t *testing.T) {
	taps := genComplexTaps(9) // complex taps: both overlap arrays are load-bearing
	first := genComplexInput(64)
	second := genComplexInput(50)

	f := NewComplexFIRFilter(taps)
	f.Process(first) // leaves nonzero overlapR/overlapI from first's tail
	f.Reset()
	gotAfterReset := f.Process(second)

	fresh := NewComplexFIRFilter(taps).Process(second)
	assertComplexClose(t, gotAfterReset, fresh)
}

// TestComplexFIRImpulseOrdering independently locks the "newest sample multiplies
// taps[nTaps-1]" window convention that the shared complex128 oracle assumes but
// cannot itself prove (the oracle encodes the same convention, so a matching bug
// in both would be invisible). A unit impulse followed by zeros through a fresh
// filter yields the impulse response, which for out[i] = sum_k taps[k]*win[k]
// (win spanning [overlap|input], impulse at the overlap boundary) emerges in
// REVERSED tap order: out[0] = taps[nTaps-1] (the newest slot), out[1] =
// taps[nTaps-2], ..., out[nTaps-1] = taps[0], then zeros once the impulse leaves
// the window. Asymmetric taps make the ordering unambiguous. Each output is a
// single tap times exactly 1 (all other window terms are 0), so the values are
// exact in float32 and asserted with ==.
func TestComplexFIRImpulseOrdering(t *testing.T) {
	// Short asymmetric complex taps: reversed != forward, and every real/imag
	// component is distinct so a swapped index would change the answer.
	taps := []complex64{complex(1, 2), complex(3, -1), complex(-2, 4)}
	nTaps := len(taps)

	input := make([]complex64, nTaps+2) // two trailing samples verify the tail is 0
	input[0] = complex(1, 0)            // unit impulse, rest already zero

	got := NewComplexFIRFilter(taps).Process(input)

	want := []complex64{
		taps[2], // out[0]: newest sample slot == taps[nTaps-1]
		taps[1], // out[1]
		taps[0], // out[2]: oldest tap
		complex(0, 0),
		complex(0, 0),
	}
	if len(got) != len(want) {
		t.Fatalf("output length %d, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("sample %d: got %v, want %v (impulse response = taps reversed)", i, got[i], want[i])
		}
	}
}
