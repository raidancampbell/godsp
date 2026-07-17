package dsp

import (
	"math"
	"math/rand"
	"testing"
)

// barker13 is the length-13 Barker code as +/-1 real symbols carried on a
// complex64 stream (imag = 0). Barker codes have the ideal autocorrelation
// property: the off-peak (sidelobe) magnitudes are at most 1 while the aligned
// peak is the full length (13 here), giving a 13:1 peak-to-sidelobe ratio. That
// makes it a clean sync-word stand-in for a P25-style +/-1 frame sync, so the
// aligned correlation peak is unambiguous versus neighbouring lags.
func barker13() []complex64 {
	bits := []float32{1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1}
	p := make([]complex64, len(bits))
	for i, b := range bits {
		p[i] = complex(b, 0)
	}
	return p
}

// complexPattern is an ASYMMETRIC, non-palindromic complex sequence with
// nonzero, non-symmetric imaginary parts. It exists to prove the matched-filter
// CONJUGATION, which the real-valued barker13 cannot: for a real pattern
// conj(p) == p, so a dropped conjugate (or an imaginary-sign flip) would still
// peak at the pattern energy and every Barker-based test would pass. For this
// complex pattern conj(p) != p, so the two tap choices diverge sharply:
//
//	correct taps conj(p): peak = sum |p[k]|^2 = E (=37 here), real and maximal
//	wrong taps p:         peak = |sum p[k]^2| ~= 16.3, far below 0.9*E = 33.3
//
// so TestCorrelatorConjugationComplexPattern fires exactly one detection on the
// correct impl and ZERO on a conjugate-dropped one. Sum of squares E = 37.
func complexPattern() []complex64 {
	return []complex64{
		complex(1, 0),
		complex(0, 1),
		complex(2, -1),
		complex(-1, 3),
		complex(1, 1),
		complex(3, -2),
		complex(-2, -1),
	}
}

// embedPattern returns a length-total zero buffer with pattern copied in so its
// first sample lands at absolute index offset. This is the ground truth the
// tests recover: Detect must report Index == offset.
func embedPattern(total, offset int, pattern []complex64) []complex64 {
	buf := make([]complex64, total)
	copy(buf[offset:], pattern)
	return buf
}

// TestCorrelatorConjugationComplexPattern is the anti-toothless test: it uses an
// asymmetric COMPLEX pattern (see complexPattern) to prove the taps are the
// conjugate conj(p[k]) and not the raw p[k]. Embedded at a known offset in a
// zero buffer, the correctly conjugated matched filter produces exactly one peak
// of magnitude E at that offset. If the conjugate were dropped (or the imaginary
// sign flipped) the aligned output would collapse to |sum p^2| ~= 16.3, below
// the 0.9*E threshold, yielding ZERO detections and failing this test -- which
// is exactly the bug class barker13 (conj(p) == p) cannot catch.
func TestCorrelatorConjugationComplexPattern(t *testing.T) {
	pattern := complexPattern()
	const offset = 500
	buf := embedPattern(4096, offset, pattern)

	c := NewCorrelator(pattern)
	// 0.9*E clears only the true conjugated peak (== E); the correct impl's
	// worst sidelobe here is ~13.9, and a conjugate-dropped impl peaks at ~16.3,
	// both well under 0.9*E, so this threshold isolates the conjugation.
	dets := c.Detect(buf, c.NormalizedThreshold(0.9))

	if len(dets) != 1 {
		t.Fatalf("expected exactly one detection, got %d: %+v (matched-filter conjugation likely wrong)", len(dets), dets)
	}
	if dets[0].Index != offset {
		t.Fatalf("detection Index = %d, want %d", dets[0].Index, offset)
	}
	// Perfect alignment of the conjugated filter -> magnitude == pattern energy.
	// This equality is what a dropped conjugate breaks (it would read ~16.3).
	wantMag := c.PatternEnergy()
	if math.Abs(float64(dets[0].Magnitude-wantMag)) > 1e-3 {
		t.Fatalf("peak magnitude = %g, want pattern energy %g (conjugation broken)", dets[0].Magnitude, wantMag)
	}
}

// TestCorrelatorPerfectMatchAtUnityThreshold pins finding #2: a clean, unity-
// scaled match sits at exactly the pattern energy E, so NormalizedThreshold(1.0)
// (threshold == E) must still report it. That only holds because Detect compares
// inclusively (m >= threshold); a strict m > threshold would silently drop the
// perfect peak. Exactly one detection is expected -- sidelobes stay strictly
// below E, so the inclusive bound introduces no duplicates.
func TestCorrelatorPerfectMatchAtUnityThreshold(t *testing.T) {
	pattern := barker13()
	const offset = 400
	buf := embedPattern(4096, offset, pattern)

	c := NewCorrelator(pattern)
	dets := c.Detect(buf, c.NormalizedThreshold(1.0))

	if len(dets) != 1 {
		t.Fatalf("perfect match at NormalizedThreshold(1.0) gave %d detections, want 1: %+v (Detect must use m >= threshold)", len(dets), dets)
	}
	if dets[0].Index != offset {
		t.Fatalf("detection Index = %d, want %d", dets[0].Index, offset)
	}
	if math.Abs(float64(dets[0].Magnitude-c.PatternEnergy())) > 1e-3 {
		t.Fatalf("peak magnitude = %g, want pattern energy %g", dets[0].Magnitude, c.PatternEnergy())
	}
}

// TestCorrelatorDetectsAtKnownOffset is THE core test: the group-delay offset
// math. A pattern is embedded at a known absolute offset in an otherwise-zero
// buffer; Detect must report exactly one peak whose Index equals that offset
// (the pattern START), with the FIR group delay of L-1 already removed. Also
// checks the peak magnitude equals the pattern energy (13 unit symbols -> E=13).
func TestCorrelatorDetectsAtKnownOffset(t *testing.T) {
	pattern := barker13()
	const offset = 500
	buf := embedPattern(4096, offset, pattern)

	c := NewCorrelator(pattern)
	// Threshold at 0.9*E: only the near-perfect aligned peak (magnitude == E)
	// clears it; the Barker sidelobes (magnitude <= 1) fall far below.
	dets := c.Detect(buf, c.NormalizedThreshold(0.9))

	if len(dets) != 1 {
		t.Fatalf("expected exactly one detection, got %d: %+v", len(dets), dets)
	}
	if dets[0].Index != offset {
		t.Fatalf("detection Index = %d, want %d (group-delay offset math is wrong)", dets[0].Index, offset)
	}
	// Perfect alignment on a noise-free unit pattern -> magnitude == energy.
	wantMag := c.PatternEnergy()
	if math.Abs(float64(dets[0].Magnitude-wantMag)) > 1e-3 {
		t.Fatalf("peak magnitude = %g, want pattern energy %g", dets[0].Magnitude, wantMag)
	}
}

// TestCorrelatorDetectsAcrossBlockBoundary proves the streaming seam: the same
// pattern is embedded so it STRADDLES a block boundary, and the stream is fed in
// two Process/Detect calls split mid-pattern. The absolute reported Index must
// still equal the true embedded offset, which only works if (a) the inner
// filter's overlap-save carries the convolution across the seam and (b) the
// global sample counter offsets the second call's local indices correctly.
func TestCorrelatorDetectsAcrossBlockBoundary(t *testing.T) {
	pattern := barker13()
	const offset = 200
	buf := embedPattern(1024, offset, pattern)

	// Split at offset+5, i.e. 5 pattern symbols land in the first call and the
	// remaining 8 in the second, so the aligned window spans the seam.
	const split = offset + 5

	c := NewCorrelator(pattern)
	thr := c.NormalizedThreshold(0.9)

	var got []Detection
	got = append(got, c.Detect(buf[:split], thr)...)
	got = append(got, c.Detect(buf[split:], thr)...)

	if len(got) != 1 {
		t.Fatalf("expected exactly one detection across the seam, got %d: %+v", len(got), got)
	}
	if got[0].Index != offset {
		t.Fatalf("straddling detection Index = %d, want %d (streaming seam broken)", got[0].Index, offset)
	}

	// Cross-check: feeding the whole buffer in ONE call must find the identical
	// absolute index, confirming the split did not shift anything.
	whole := NewCorrelator(pattern).Detect(buf, thr)
	if len(whole) != 1 || whole[0].Index != got[0].Index {
		t.Fatalf("single-block detection %+v disagrees with split detection %+v", whole, got)
	}
}

// TestCorrelatorPeakIsClearMaximum checks that the magnitude at the true
// alignment is a strict maximum versus its neighbouring lags. It embeds the
// pattern, runs Process (raw per-sample magnitudes), and confirms the peak-lag
// sample dominates the samples immediately around it by a wide margin (Barker
// gives >=13:1). This is the property that makes threshold detection robust.
func TestCorrelatorPeakIsClearMaximum(t *testing.T) {
	pattern := barker13()
	const offset = 300
	buf := embedPattern(1024, offset, pattern)
	L := len(pattern)

	c := NewCorrelator(pattern)
	mags := c.Process(buf)

	// Process reports peak-lag-aligned magnitudes: the pattern starting at
	// offset peaks at local index offset+(L-1) (the pattern's last sample).
	peakIdx := offset + (L - 1)
	peak := mags[peakIdx]

	if math.Abs(float64(peak-c.PatternEnergy())) > 1e-3 {
		t.Fatalf("peak magnitude at lag %d = %g, want energy %g", peakIdx, peak, c.PatternEnergy())
	}

	// Every neighbour within +/-4 lags (excluding the peak itself) must be a
	// small sidelobe. Barker-13 sidelobes are <= 1, well under half the peak.
	for d := -4; d <= 4; d++ {
		if d == 0 {
			continue
		}
		nb := mags[peakIdx+d]
		if nb >= peak {
			t.Fatalf("neighbour lag %+d magnitude %g >= peak %g (not a clear maximum)", d, nb, peak)
		}
		if nb > 0.5*peak {
			t.Fatalf("neighbour lag %+d magnitude %g exceeds half the peak %g (sidelobe too high)", d, nb, peak)
		}
	}
}

// TestCorrelatorNoFalseDetectionInNoise feeds a pure-noise buffer (no embedded
// pattern) and asserts Detect fires nothing at a sane threshold.
//
// Noise math (and why the SNR must be positive). At a random lag the output is
// out = sum_k conj(p[k]) * x[k], an INCOHERENT sum of L independent products
// (|conj(p[k])| = 1 just rotates each noise sample). Its expected energy is
// E[|out|^2] = L * noisePower, so the noise magnitude has RMS sqrt(L*noisePower)
// -- a factor sqrt(L) above the per-sample noise amplitude, the matched filter's
// processing gain. The coherent peak of a genuine match is E = L (unit symbols),
// a factor sqrt(L/noisePower) above that RMS.
//
// Crucially, |out|^2 is roughly exponential, so the MAXIMUM over n independent
// lags grows like noisePower*L*ln(n): at 0 dB SNR (noisePower == symbol energy)
// the max over thousands of samples creeps up toward the peak and a
// fraction-of-E threshold false-alarms. That is a real property, not a bug, so
// this test uses a realistic POSITIVE SNR. Per-component stddev sigma = 0.25
// gives noisePower = 2*sigma^2 = 0.125 (about 9 dB per-symbol SNR); noise
// magnitude RMS is sqrt(13*0.125) ~ 1.27 and its extreme-value max over n
// samples stays near 4-5, well under the 0.7*E = 9.1 threshold.
func TestCorrelatorNoFalseDetectionInNoise(t *testing.T) {
	pattern := barker13()
	c := NewCorrelator(pattern)

	// Deterministic RNG so the test is reproducible.
	rng := rand.New(rand.NewSource(0xC0FFEE))
	const n = 8192
	noise := make([]complex64, n)
	const sigma = 0.25
	for i := range noise {
		noise[i] = complex(float32(rng.NormFloat64()*sigma), float32(rng.NormFloat64()*sigma))
	}

	// 0.7*E is the low end of a typical sync threshold and sits ~7x the noise
	// magnitude RMS, so nothing should fire.
	dets := c.Detect(noise, c.NormalizedThreshold(0.7))
	if len(dets) != 0 {
		// Report the worst offender to aid debugging if this ever trips.
		var worst float32
		for _, d := range dets {
			if d.Magnitude > worst {
				worst = d.Magnitude
			}
		}
		t.Fatalf("got %d false detections in pure noise (max magnitude %g, threshold %g)",
			len(dets), worst, c.NormalizedThreshold(0.7))
	}
}

// TestCorrelatorResetRewindsIndex confirms Reset returns the absolute sample
// counter to zero: after processing some samples then Reset, a pattern embedded
// at offset in a fresh buffer is again reported at Index == offset (not shifted
// by the previously consumed sample count).
func TestCorrelatorResetRewindsIndex(t *testing.T) {
	pattern := barker13()
	c := NewCorrelator(pattern)

	// Consume a chunk to advance the global index well past zero.
	c.Process(make([]complex64, 777))

	c.Reset()

	const offset = 123
	buf := embedPattern(1024, offset, pattern)
	dets := c.Detect(buf, c.NormalizedThreshold(0.9))
	if len(dets) != 1 || dets[0].Index != offset {
		t.Fatalf("after Reset got %+v, want single detection at Index %d", dets, offset)
	}
}

// TestNewCorrelatorPanicsOnEmptyPattern verifies the fail-fast precondition:
// an empty pattern has nothing to correlate against and must panic, matching
// the other constructors' guard style.
func TestNewCorrelatorPanicsOnEmptyPattern(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic on empty pattern")
		}
	}()
	NewCorrelator(nil)
}
