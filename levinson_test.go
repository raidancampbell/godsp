package dsp

import (
	"math"
	"testing"
)

func TestAutocorrelation(t *testing.T) {
	// Sine wave at 200 Hz, 8 kHz sample rate
	x := make([]float32, 160)
	for i := range x {
		x[i] = float32(math.Sin(2 * math.Pi * 200 * float64(i) / 8000))
	}

	R := Autocorrelation(x, 10)

	// R[0] should be the energy (positive)
	if R[0] <= 0 {
		t.Fatal("R[0] should be positive")
	}

	// For a sine wave, |R[k]| <= R[0] for all k
	for k := 1; k <= 10; k++ {
		if math.Abs(R[k]) > R[0]*1.001 {
			t.Errorf("R[%d]=%f > R[0]=%f", k, R[k], R[0])
		}
	}
}

func TestExpandBandwidth(t *testing.T) {
	a := []float64{1.0, -0.5, 0.3, -0.1}
	expanded := ExpandBandwidth(a, 0.8)

	if expanded[0] != 1.0 {
		t.Errorf("expanded[0] = %f, want 1.0", expanded[0])
	}
	// a[1] * 0.8^1
	if math.Abs(expanded[1]-(-0.5*0.8)) > 1e-10 {
		t.Errorf("expanded[1] = %f, want %f", expanded[1], -0.5*0.8)
	}
	// a[2] * 0.8^2
	if math.Abs(expanded[2]-(0.3*0.64)) > 1e-10 {
		t.Errorf("expanded[2] = %f, want %f", expanded[2], 0.3*0.64)
	}
}

