package dsp

import (
	"math"
	"math/cmplx"
	"testing"
)

// makeTone builds n samples of a complex exponential at normalized freq f
// (cycles/sample).
func makeTone(n int, f float64) []complex64 {
	out := make([]complex64, n)
	for i := 0; i < n; i++ {
		p := 2 * math.Pi * f * float64(i)
		out[i] = complex64(cmplx.Rect(1, p))
	}
	return out
}

func TestSpectralFlatnessToneVsNoise(t *testing.T) {
	tone := makeTone(4096, 0.1)
	pTone := WelchPSD(tone, 256, 0.5, HannWindow(256))
	fTone := SpectralFlatness(pTone)
	if fTone > 0.2 {
		t.Errorf("tone flatness = %.4f, want < 0.2", fTone)
	}

	// Deterministic pseudo-noise (LCG) so the test is reproducible without
	// touching disallowed RNG in the module under test.
	noise := make([]complex64, 4096)
	var s uint32 = 12345
	for i := range noise {
		s = s*1664525 + 1013904223
		re := float32(s)/float32(1<<32) - 0.5
		s = s*1664525 + 1013904223
		im := float32(s)/float32(1<<32) - 0.5
		noise[i] = complex(re, im)
	}
	pNoise := WelchPSD(noise, 256, 0.5, HannWindow(256))
	fNoise := SpectralFlatness(pNoise)
	if fNoise < 0.4 {
		t.Errorf("noise flatness = %.4f, want > 0.4", fNoise)
	}
}

func TestOccupiedBandwidthToneIsNarrow(t *testing.T) {
	tone := makeTone(8192, 0.0) // DC → energy in center bins
	psd := WelchPSD(tone, 256, 0.5, HannWindow(256))
	bw := OccupiedBandwidth(psd, 100.0, 0.99) // 100 Hz/bin
	full := 256 * 100.0
	if bw > 0.25*full {
		t.Errorf("tone occupied BW = %.0f Hz, want < %.0f (25%% of span)", bw, 0.25*full)
	}
}

func TestWelchPSDLength(t *testing.T) {
	p := WelchPSD(makeTone(1000, 0.1), 128, 0.5, HannWindow(128))
	if len(p) != 128 {
		t.Fatalf("len = %d, want 128", len(p))
	}
}

func TestWelchPSDRejectsWrongLengthWindow(t *testing.T) {
	// A window whose length != nfft must be rejected, not silently misapplied.
	if p := WelchPSD(makeTone(1000, 0.1), 128, 0.5, HannWindow(64)); p != nil {
		t.Errorf("wrong-length window: got len %d, want nil", len(p))
	}
}

func TestWelchPSDNilWindowRectangular(t *testing.T) {
	// The nil-window (rectangular) branch must produce a valid length-nfft PSD.
	p := WelchPSD(makeTone(1000, 0.1), 128, 0.5, nil)
	if len(p) != 128 {
		t.Fatalf("nil window: len = %d, want 128", len(p))
	}
}
