package dsp

import (
	"math"
	"testing"
)

func TestHammingWindowEndpoints(t *testing.T) {
	w := HammingWindow(8)
	if len(w) != 8 {
		t.Fatalf("len = %d, want 8", len(w))
	}
	// Hamming endpoints = 0.54 - 0.46 = 0.08.
	if math.Abs(w[0]-0.08) > 1e-9 || math.Abs(w[7]-0.08) > 1e-9 {
		t.Errorf("endpoints = %.6f,%.6f want 0.08", w[0], w[7])
	}
}

func TestBlackmanWindowSymmetric(t *testing.T) {
	w := BlackmanWindow(9)
	for i := 0; i < len(w)/2; i++ {
		if math.Abs(w[i]-w[len(w)-1-i]) > 1e-12 {
			t.Errorf("not symmetric at %d: %.9f vs %.9f", i, w[i], w[len(w)-1-i])
		}
	}
}

func TestKaiserWindowBetaZeroIsRect(t *testing.T) {
	// beta=0 → I0(0)/I0(0) = 1 for every tap (rectangular).
	w := KaiserWindow(6, 0)
	for i, v := range w {
		if math.Abs(v-1.0) > 1e-9 {
			t.Errorf("tap %d = %.9f, want 1.0", i, v)
		}
	}
}

func TestBesselI0Known(t *testing.T) {
	// I0(0) = 1; I0(1) ≈ 1.2660658778.
	if math.Abs(besselI0(0)-1.0) > 1e-12 {
		t.Errorf("I0(0) = %.12f", besselI0(0))
	}
	if math.Abs(besselI0(1)-1.2660658778) > 1e-6 {
		t.Errorf("I0(1) = %.10f, want ~1.2660658778", besselI0(1))
	}
}

func TestWindowN1(t *testing.T) {
	for _, w := range [][]float64{HammingWindow(1), BlackmanWindow(1), KaiserWindow(1, 5)} {
		if len(w) != 1 || w[0] != 1 {
			t.Errorf("n=1 window = %v, want [1]", w)
		}
	}
}
