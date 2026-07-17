package dsp

import (
	"fmt"
	"math"
	"testing"
)

func TestComplexFIRDotDispatchMatchesScalar(t *testing.T) {
	tapCounts := []int{1, 7, 8, 9, 31, 32, 33, 332, 333, 334}
	patterns := []struct {
		name string
		fill func(taps, winR, winI []float32)
	}{
		{
			name: "deterministic",
			fill: func(taps, winR, winI []float32) {
				for i := range taps {
					taps[i] = float32((i*17)%29-14) / 31
					winR[i] = float32((i*11)%37-18) / 19
					winI[i] = float32((i*23)%41-20) / 23
				}
			},
		},
		{
			name: "alternating_sign",
			fill: func(taps, winR, winI []float32) {
				for i := range taps {
					sign := float32(1)
					if i&1 != 0 {
						sign = -1
					}
					taps[i] = sign * float32(i+1) / 337
					winR[i] = -sign * float32((i%13)+1) / 7
					winI[i] = sign * float32((i%17)+1) / 9
				}
			},
		},
		{
			name: "maximum_finite_normal",
			fill: func(taps, winR, winI []float32) {
				const scaledTap = float32(0x1p-120)
				for i := range taps {
					sign := float32(1)
					if i&1 != 0 {
						sign = -1
					}
					taps[i] = sign * scaledTap
					winR[i] = math.MaxFloat32
					winI[i] = -sign * math.MaxFloat32
				}
			},
		},
	}

	for _, pattern := range patterns {
		for _, n := range tapCounts {
			t.Run(pattern.name+"/n="+fmt.Sprint(n), func(t *testing.T) {
				taps := make([]float32, n)
				winR := make([]float32, n)
				winI := make([]float32, n)
				pattern.fill(taps, winR, winI)
				wantR, wantI := complexFIRDotScalar(taps, winR, winI)
				gotR, gotI := complexFIRDot(taps, winR, winI)
				scale := abs(wantR)
				if abs(wantI) > scale {
					scale = abs(wantI)
				}
				scale++
				tol := float32(2e-5) * scale
				if abs(gotR-wantR) > tol || abs(gotI-wantI) > tol {
					t.Fatalf("got (%g,%g), scalar (%g,%g), tolerance %g", gotR, gotI, wantR, wantI, tol)
				}
			})
		}
	}
}

func abs(x float32) float32 {
	if x < 0 {
		return -x
	}
	return x
}

func TestDesignLPF_SymmetricTaps(t *testing.T) {
	taps := DesignLPF(1000, 10000, 51)
	if len(taps) != 51 {
		t.Fatalf("expected 51 taps, got %d", len(taps))
	}

	// FIR designed with a symmetric window should produce symmetric taps
	for i := 0; i < len(taps)/2; i++ {
		j := len(taps) - 1 - i
		if math.Abs(float64(taps[i]-taps[j])) > 1e-6 {
			t.Errorf("tap[%d]=%f != tap[%d]=%f (asymmetric)", i, taps[i], j, taps[j])
		}
	}
}

func TestDesignLPF_UnityDCGain(t *testing.T) {
	taps := DesignLPF(1000, 10000, 51)

	var sum float64
	for _, tap := range taps {
		sum += float64(tap)
	}

	if math.Abs(sum-1.0) > 1e-5 {
		t.Errorf("DC gain = %f, expected ~1.0", sum)
	}
}

func TestDesignLPF_EvenTapsMadeOdd(t *testing.T) {
	taps := DesignLPF(1000, 10000, 50)
	if len(taps)%2 != 1 {
		t.Errorf("expected odd number of taps, got %d", len(taps))
	}
}

func TestDecimatingFilter_OutputLength(t *testing.T) {
	taps := DesignLPF(125000, 10000000, 21)
	f := NewDecimatingFilter(taps, 10)

	input := make([]complex64, 1000)
	for i := range input {
		input[i] = complex(1.0, 0)
	}
	output := f.Process(input)

	// With 1000 input samples and decimation 10, expect ~100 outputs
	if len(output) != 100 {
		t.Errorf("expected 100 output samples, got %d", len(output))
	}
}

func TestDecimatingFilter_DCPassthrough(t *testing.T) {
	taps := DesignLPF(125000, 10000000, 21)
	f := NewDecimatingFilter(taps, 10)

	// Feed DC signal (all 1+0j) — after filter settles, output should approach 1+0j
	input := make([]complex64, 10000)
	for i := range input {
		input[i] = complex(1.0, 0)
	}
	output := f.Process(input)

	// Check last few samples (after settling)
	for i := len(output) - 10; i < len(output); i++ {
		re := real(output[i])
		if math.Abs(float64(re)-1.0) > 0.01 {
			t.Errorf("output[%d] real = %f, expected ~1.0", i, re)
		}
	}
}

func TestDecimatingFilterSplitMatchesSingle(t *testing.T) {
	input := make([]complex64, 4097)
	for i := range input {
		input[i] = complex(
			float32((i*17)%257-128)/127,
			float32((i*43)%263-131)/129,
		)
	}

	for _, cutoff := range []float64{6250, 12500} {
		t.Run(fmt.Sprintf("cutoff_%.0f", cutoff), func(t *testing.T) {
			taps := DesignLPF(cutoff, 250_000, 333)
			whole := NewDecimatingFilter(taps, 10).Process(input)
			splitFilter := NewDecimatingFilter(taps, 10)
			var split []complex64
			seenPhase := [10]bool{}
			pos := 0
			for pos < len(input) {
				seenPhase[splitFilter.phase] = true
				size := 1
				if pos >= 10 {
					size = []int{7, 13, 29, 64, 3, 101}[pos%6]
				}
				if size > len(input)-pos {
					size = len(input) - pos
				}
				split = append(split, splitFilter.Process(input[pos:pos+size])...)
				pos += size
			}
			for phase, seen := range seenPhase {
				if !seen {
					t.Fatalf("split stream did not exercise phase %d", phase)
				}
			}
			if len(split) != len(whole) {
				t.Fatalf("split output length %d, single-block length %d", len(split), len(whole))
			}
			for i := range whole {
				wantR, wantI := real(whole[i]), imag(whole[i])
				gotR, gotI := real(split[i]), imag(split[i])
				scale := abs(wantR)
				if abs(wantI) > scale {
					scale = abs(wantI)
				}
				tol := float32(2e-5) * (scale + 1)
				if abs(gotR-wantR) > tol || abs(gotI-wantI) > tol {
					t.Fatalf("sample %d: split (%g,%g), single (%g,%g), tolerance %g", i, gotR, gotI, wantR, wantI, tol)
				}
			}
		})
	}
}

func TestDecimatingFilterProcessReuseZeroAlloc(t *testing.T) {
	taps := DesignLPF(6250, 250_000, 333)
	f := NewDecimatingFilter(taps, 10)
	input := make([]complex64, 2731)
	for i := range input {
		input[i] = complex(float32(i%31)/31, -float32(i%37)/37)
	}
	f.ProcessReuse(input)
	allocs := testing.AllocsPerRun(100, func() {
		_ = f.ProcessReuse(input)
	})
	if allocs != 0 {
		t.Fatalf("ProcessReuse allocs/op = %.2f, want 0", allocs)
	}
}

func BenchmarkDecimatingFilter(b *testing.B) {
	b.Run("legacy_10M", func(b *testing.B) {
		taps := DesignLPF(125000, 10_000_000, 801)
		f := NewDecimatingFilter(taps, 40)
		input := benchmarkComplexInput(65536)
		b.ReportAllocs()
		b.SetBytes(int64(len(input) * 8))
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = f.Process(input)
		}
	})

	b.Run("stage2_333tap_decim10", func(b *testing.B) {
		hop, err := Stage1DecimationFor(48_000_000)
		if err != nil {
			b.Fatal(err)
		}
		input := benchmarkComplexInput(ReadBlockSize / hop)
		f := NewDecimatingFilter(DesignLPF(6250, 250_000, 333), Stage2Decimation)
		f.ProcessReuse(input)
		b.ReportAllocs()
		b.SetBytes(int64(len(input) * 8))
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = f.ProcessReuse(input)
		}
	})
}

func benchmarkComplexInput(n int) []complex64 {
	input := make([]complex64, n)
	for i := range input {
		input[i] = complex(float32(i%256)/128-1, float32((i+64)%256)/128-1)
	}
	return input
}

func TestDecimatingFilterReal_OutputLength(t *testing.T) {
	taps := DesignLPF(3000, 25000, 21)
	f := NewDecimatingFilterReal(taps, 3)

	input := make([]float32, 300)
	for i := range input {
		input[i] = 1.0
	}
	output := f.Process(input)

	if len(output) != 100 {
		t.Errorf("expected 100 output samples, got %d", len(output))
	}
}

// TestFIRFilterReal_ProcessReuseMatchesProcess confirms ProcessReuse produces
// the same output as Process for the same input stream (separate filter
// instances, since both mutate internal overlap state).
func TestFIRFilterReal_ProcessReuseMatchesProcess(t *testing.T) {
	taps := DesignLPF(3000, 25000, 51)
	a := NewFIRFilterReal(taps)
	b := NewFIRFilterReal(taps)

	for block := 0; block < 4; block++ {
		input := make([]float32, 250)
		for i := range input {
			input[i] = float32((block*7 + i) % 11) // arbitrary deterministic stream
		}
		want := a.Process(input)
		got := b.ProcessReuse(input)
		if len(got) != len(want) {
			t.Fatalf("block %d: len mismatch got %d want %d", block, len(got), len(want))
		}
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("block %d sample %d: got %v want %v", block, i, got[i], want[i])
			}
		}
	}
}

// TestFIRFilterReal_ProcessReuseZeroAlloc proves the steady-state per-call
// allocation count is zero. The previous Process path made one fresh
// len(input)-sized slice every call (~2.77 GB / 30 s in the captured profile).
func TestFIRFilterReal_ProcessReuseZeroAlloc(t *testing.T) {
	taps := DesignLPF(3000, 25000, 51)
	f := NewFIRFilterReal(taps)
	input := make([]float32, 250)
	for i := range input {
		input[i] = float32(i % 7)
	}
	// Warm internal buffers so the steady-state allocs measure zero.
	f.ProcessReuse(input)
	allocs := testing.AllocsPerRun(100, func() {
		_ = f.ProcessReuse(input)
	})
	if allocs > 0 {
		t.Fatalf("ProcessReuse allocs/op = %.2f, want 0", allocs)
	}
}

// TestDesignRRC_PanicsOnNonPositiveRolloff verifies the rolloff precondition
// guard: rolloff <= 0 would divide by zero in the tap formulas and yield NaN
// taps, so the designer must panic instead.
func TestDesignRRC_PanicsOnNonPositiveRolloff(t *testing.T) {
	for _, rolloff := range []float64{0.0, -0.2} {
		func() {
			defer func() {
				if r := recover(); r == nil {
					t.Errorf("expected panic on rolloff=%v", rolloff)
				}
			}()
			DesignRRC(4800, 25000, rolloff, 8)
		}()
	}
}

// TestDesignRRC_ValidRolloffNoNaN confirms the guard is a no-op for valid
// rolloffs and that the resulting taps contain no NaN/Inf values.
func TestDesignRRC_ValidRolloffNoNaN(t *testing.T) {
	taps := DesignRRC(4800, 25000, 0.2, 8)
	for i, c := range taps {
		if math.IsNaN(float64(c)) || math.IsInf(float64(c), 0) {
			t.Fatalf("tap %d is non-finite: %v", i, c)
		}
	}
}

// TestDesignLPF_PanicsAtOrAboveNyquist verifies the cutoff precondition guard:
// a cutoff at or above Nyquist (sampleRate/2) produces an unstable/aliased
// window-sinc, so the designer must panic.
func TestDesignLPF_PanicsAtOrAboveNyquist(t *testing.T) {
	for _, cutoff := range []float64{12500.0 /* == Nyquist */, 20000.0 /* > Nyquist */, 0.0, -1.0} {
		func() {
			defer func() {
				if r := recover(); r == nil {
					t.Errorf("expected panic on cutoff=%v", cutoff)
				}
			}()
			DesignLPF(cutoff, 25000, 63)
		}()
	}
}

// TestDesignLPF_ValidCutoffNoPanic confirms the guard is a no-op just below
// Nyquist and that the taps are finite.
func TestDesignLPF_ValidCutoffNoPanic(t *testing.T) {
	taps := DesignLPF(12499.0, 25000, 63)
	for i, c := range taps {
		if math.IsNaN(float64(c)) || math.IsInf(float64(c), 0) {
			t.Fatalf("tap %d is non-finite: %v", i, c)
		}
	}
}

// BenchmarkPackUnpack isolates the arm64 batch transpose at production sizes.
func BenchmarkPackUnpack(b *testing.B) {
	for _, n := range []int{288, 512, 800} {
		src := make([]complex64, fftcBatchWidth*n)
		dst := make([]complex64, fftcBatchWidth*n)
		for i := range src {
			src[i] = complex(float32(i), float32(-i))
		}
		b.Run(fmt.Sprintf("pack/n=%d", n), func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				packBatch4(dst, src, n)
			}
		})
		b.Run(fmt.Sprintf("unpack/n=%d", n), func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				unpackBatch4(dst, src, n)
			}
		})
	}
}
