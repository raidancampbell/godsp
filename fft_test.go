package dsp

import (
	"math"
	"math/cmplx"
	"testing"
)

func TestFFTKnownValues(t *testing.T) {
	// 8-pt FFT of a real impulse at index 0 should be all-ones.
	x := make([]complex128, 8)
	x[0] = 1
	FFT(x)
	for i, v := range x {
		if math.Abs(real(v)-1) > 1e-9 || math.Abs(imag(v)) > 1e-9 {
			t.Errorf("bin %d: got %v, want 1+0i", i, v)
		}
	}
}

func TestFFTSinglePureTone(t *testing.T) {
	// 256-pt FFT of a complex exponential at bin 17 should peak at bin 17.
	n := 256
	bin := 17
	x := make([]complex128, n)
	for k := 0; k < n; k++ {
		x[k] = cmplx.Exp(complex(0, 2*math.Pi*float64(bin)*float64(k)/float64(n)))
	}
	FFT(x)
	var maxIdx int
	var maxMag float64
	for i, v := range x {
		m := cmplx.Abs(v)
		if m > maxMag {
			maxMag = m
			maxIdx = i
		}
	}
	if maxIdx != bin {
		t.Errorf("peak at bin %d, want %d", maxIdx, bin)
	}
}

func TestHannWindow(t *testing.T) {
	w := HannWindow(8)
	if len(w) != 8 {
		t.Fatalf("len(w)=%d, want 8", len(w))
	}
	if w[0] != 0 {
		t.Errorf("w[0]=%v, want 0", w[0])
	}
	// Symmetry: w[k] == w[n-1-k]
	for k := 0; k < len(w); k++ {
		if math.Abs(w[k]-w[len(w)-1-k]) > 1e-12 {
			t.Errorf("asymmetry at k=%d: w[k]=%v w[n-1-k]=%v", k, w[k], w[len(w)-1-k])
		}
	}
}

func TestIFFTRoundTrip(t *testing.T) {
	x := make([]complex128, 16)
	for i := range x {
		x[i] = complex(math.Sin(float64(i)), math.Cos(float64(i)*0.3))
	}
	orig := append([]complex128(nil), x...)
	FFT(x)
	IFFT(x)
	for i := range x {
		if math.Abs(real(x[i])-real(orig[i])) > 1e-9 || math.Abs(imag(x[i])-imag(orig[i])) > 1e-9 {
			t.Fatalf("round trip mismatch at %d: got %v want %v", i, x[i], orig[i])
		}
	}
}

