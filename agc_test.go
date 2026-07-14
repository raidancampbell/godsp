package dsp

import (
	"math"
	"testing"
)

func TestIQAGCConvergesToTarget(t *testing.T) {
	a := NewIQAGC(1.0, 0.01, 100.0)
	// Constant-magnitude low-level input.
	in := make([]complex64, 20000)
	for i := range in {
		in[i] = complex(0.01, 0)
	}
	a.ProcessInPlace(in)
	// Tail RMS should approach the target (1.0) within 10%.
	var sum float64
	tail := in[len(in)-1000:]
	for _, s := range tail {
		sum += float64(real(s))*float64(real(s)) + float64(imag(s))*float64(imag(s))
	}
	rms := math.Sqrt(sum / float64(len(tail)))
	if rms < 0.9 || rms > 1.1 {
		t.Errorf("tail RMS = %.4f, want ~1.0", rms)
	}
}

func TestIQAGCRespectsMaxGain(t *testing.T) {
	a := NewIQAGC(1.0, 0.5, 4.0) // max gain 4x
	in := make([]complex64, 5000)
	for i := range in {
		in[i] = complex(0.001, 0) // would need 1000x to hit target
	}
	a.ProcessInPlace(in)
	if a.Gain() > 4.0+1e-6 {
		t.Errorf("gain = %.4f, want <= 4.0", a.Gain())
	}
}

func TestAudioAGCSilenceNoBlowup(t *testing.T) {
	a := NewAudioAGC(0.3, 0.01, 0.001, 50.0)
	in := make([]float32, 4000) // all zeros
	a.ProcessInPlace(in)
	for _, s := range in {
		if math.IsNaN(float64(s)) || math.IsInf(float64(s), 0) {
			t.Fatalf("silence produced non-finite sample %v", s)
		}
	}
}
