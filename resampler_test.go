package dsp

import (
	"math"
	"testing"
)

func TestRationalResampler_RateRatio(t *testing.T) {
	// 25 kSPS → 8 kHz: up by 8, down by 25
	r := NewRationalResampler(8, 25)

	// Feed 25000 samples (1 second at 25 kSPS)
	input := make([]float32, 25000)
	for i := range input {
		input[i] = 1.0 // DC
	}

	output := r.Process(input)

	// Should produce ~8000 samples (1 second at 8 kHz)
	// Allow small tolerance for filter edge effects
	expectedMin := 7900
	expectedMax := 8100
	if len(output) < expectedMin || len(output) > expectedMax {
		t.Errorf("expected ~8000 output samples, got %d", len(output))
	}
}

func TestRationalResampler_DCPreservation(t *testing.T) {
	r := NewRationalResampler(8, 25)

	// DC input should produce DC output (after settling)
	input := make([]float32, 5000)
	for i := range input {
		input[i] = 1.0
	}

	output := r.Process(input)

	// Check last portion of output for DC value ~1.0
	settled := output[len(output)/2:]
	var sum float64
	for _, s := range settled {
		sum += float64(s)
	}
	mean := sum / float64(len(settled))

	if math.Abs(mean-1.0) > 0.05 {
		t.Errorf("mean output = %f, expected ~1.0 for DC input", mean)
	}
}

// TestRationalResampler_ChunkedEqualsMonolithic guards the ring-buffer
// refactor: feeding the input in arbitrary chunks must produce bit-identical
// output to a single Process() call.
func TestRationalResampler_ChunkedEqualsMonolithic(t *testing.T) {
	input := make([]float32, 5000)
	for i := range input {
		input[i] = float32(math.Sin(2 * math.Pi * float64(i) * 1000 / 25000))
	}

	r1 := NewRationalResampler(8, 25)
	mono := r1.Process(input)

	for _, chunk := range []int{1, 7, 163, 250, 1000} {
		r2 := NewRationalResampler(8, 25)
		var got []float32
		for off := 0; off < len(input); off += chunk {
			end := min(off+chunk, len(input))
			got = append(got, r2.Process(input[off:end])...)
		}
		if len(got) != len(mono) {
			t.Errorf("chunk=%d: len=%d, want %d", chunk, len(got), len(mono))
			continue
		}
		for i := range got {
			if math.Abs(float64(got[i]-mono[i])) > 1e-6 {
				t.Errorf("chunk=%d: out[%d]=%f, want %f", chunk, i, got[i], mono[i])
				break
			}
		}
	}
}

func BenchmarkRationalResampler_Process(b *testing.B) {
	r := NewRationalResampler(8, 25)
	input := make([]float32, BlockSamples)
	for i := range input {
		input[i] = float32(math.Sin(float64(i) * 0.1))
	}
	b.ResetTimer()
	for b.Loop() {
		r.Process(input)
	}
}

func TestRationalResampler_ZeroInput(t *testing.T) {
	r := NewRationalResampler(8, 25)
	output := r.Process(nil)
	if len(output) != 0 {
		t.Errorf("expected 0 output for nil input, got %d", len(output))
	}

	output = r.Process([]float32{})
	if len(output) != 0 {
		t.Errorf("expected 0 output for empty input, got %d", len(output))
	}
}

func TestResamplerReset(t *testing.T) {
	r := NewRationalResampler(8, 25)
	in := make([]float32, 250)
	for i := range in {
		in[i] = float32(i%7) - 3
	}
	first := append([]float32(nil), r.Process(in)...)
	r.Reset()
	second := append([]float32(nil), r.Process(in)...)
	if len(first) != len(second) {
		t.Fatalf("len mismatch after reset: %d vs %d", len(first), len(second))
	}
	for i := range first {
		if first[i] != second[i] {
			t.Fatalf("reset did not restore initial state at %d: %v vs %v", i, first[i], second[i])
		}
	}
}

func TestResamplerNoAllocSteadyState(t *testing.T) {
	r := NewRationalResampler(8, 25)
	in := make([]float32, 250)
	r.Process(in) // warm up the internal buffer
	allocs := testing.AllocsPerRun(50, func() {
		r.Process(in)
	})
	if allocs > 0 {
		t.Errorf("Process allocated %.1f times per run, want 0", allocs)
	}
}
