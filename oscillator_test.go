package dsp

import (
	"math"
	"testing"
)

func TestOscillator_DCAtZeroOffset(t *testing.T) {
	osc := NewOscillator(0, 10000)
	samples := make([]complex64, 100)
	for i := range samples {
		samples[i] = complex(1.0, 0)
	}
	osc.MixInPlace(samples)

	// Zero offset should not change samples
	for i, s := range samples {
		if math.Abs(float64(real(s))-1.0) > 1e-5 || math.Abs(float64(imag(s))) > 1e-5 {
			t.Errorf("sample[%d] = (%f, %f), expected (1, 0)", i, real(s), imag(s))
			break
		}
	}
}

func TestOscillator_UnitMagnitude(t *testing.T) {
	osc := NewOscillator(1000, 10000)
	samples := make([]complex64, 10000)
	for i := range samples {
		samples[i] = complex(1.0, 0)
	}
	osc.MixInPlace(samples)

	// Mixing with a unit complex exponential should preserve magnitude
	for i, s := range samples {
		mag := math.Sqrt(float64(real(s)*real(s) + imag(s)*imag(s)))
		if math.Abs(mag-1.0) > 1e-4 {
			t.Errorf("sample[%d] magnitude = %f, expected ~1.0", i, mag)
			break
		}
	}
}

func TestOscillator_MixMatchesCopyMixInPlace(t *testing.T) {
	const n = 256
	const fs = 48000.0
	const fOff = 1234.5

	in := make([]complex64, n)
	for i := range in {
		ph := 2 * math.Pi * 440.0 * float64(i) / fs
		in[i] = complex(float32(math.Cos(ph)), float32(math.Sin(ph)))
	}
	inSnap := make([]complex64, n)
	copy(inSnap, in)

	// Reference: copy then MixInPlace.
	want := make([]complex64, n)
	copy(want, in)
	NewOscillator(fOff, fs).MixInPlace(want)

	// Under test: out-of-place Mix from a fresh oscillator with the same params.
	got := make([]complex64, n)
	NewOscillator(fOff, fs).Mix(in, got)

	for i := range got {
		if dr, di := float64(real(got[i])-real(want[i])), float64(imag(got[i])-imag(want[i])); math.Abs(dr) > 1e-6 || math.Abs(di) > 1e-6 {
			t.Fatalf("sample[%d] = (%g,%g), want (%g,%g)", i, real(got[i]), imag(got[i]), real(want[i]), imag(want[i]))
		}
	}

	// Mix must not mutate its input.
	for i := range in {
		if in[i] != inSnap[i] {
			t.Fatalf("Mix mutated in[%d]: got (%g,%g), was (%g,%g)", i, real(in[i]), imag(in[i]), real(inSnap[i]), imag(inSnap[i]))
		}
	}
}

func TestOscillator_PeriodTableMatchesReference(t *testing.T) {
	const fs = 10_000_000.0
	const off = -3_412_500.0 // 460.4125 − 457.0 MHz; on the 12.5 kHz grid
	osc := NewOscillator(off, fs)
	if osc.period == 0 {
		t.Fatal("expected a precomputed period table for an integer offset, got fallback")
	}
	if osc.period != 800 { // fs / gcd(fs,|off|) = 1e7 / 12500
		t.Errorf("period = %d, want 800", osc.period)
	}
	inc := -2.0 * math.Pi * off / fs
	in := make([]complex64, 1000)
	for i := range in {
		in[i] = complex(1, 0)
	}
	out := make([]complex64, len(in))
	osc.Mix(in, out)
	for i := range out {
		ph := inc * float64(i)
		wr, wi := float32(math.Cos(ph)), float32(math.Sin(ph))
		if dr, di := float64(real(out[i])-wr), float64(imag(out[i])-wi); math.Abs(dr) > 1e-5 || math.Abs(di) > 1e-5 {
			t.Fatalf("sample[%d]=(%g,%g) want (%g,%g)", i, real(out[i]), imag(out[i]), wr, wi)
		}
	}
}

func TestOscillator_PeriodTableCrossBlock(t *testing.T) {
	const fs = 10_000_000.0
	const off = 1_000_000.0
	in := make([]complex64, 777)
	for i := range in {
		ph := 2 * math.Pi * 12345.0 * float64(i) / fs
		in[i] = complex(float32(math.Cos(ph)), float32(math.Sin(ph)))
	}
	one := make([]complex64, len(in))
	NewOscillator(off, fs).Mix(in, one)

	osc := NewOscillator(off, fs)
	two := make([]complex64, len(in))
	const split = 300
	osc.Mix(in[:split], two[:split])
	osc.Mix(in[split:], two[split:])
	for i := range one {
		if dr, di := float64(real(one[i])-real(two[i])), float64(imag(one[i])-imag(two[i])); math.Abs(dr) > 1e-6 || math.Abs(di) > 1e-6 {
			t.Fatalf("split-block mismatch at %d: (%g,%g) vs (%g,%g)", i, real(two[i]), imag(two[i]), real(one[i]), imag(one[i]))
		}
	}
}

func TestOscillator_FallbackNonIntegerAndLongPeriod(t *testing.T) {
	if osc := NewOscillator(1234.5, 48000); osc.period != 0 {
		t.Errorf("non-integer offset: period = %d, want 0 (recurrence fallback)", osc.period)
	}
	if osc := NewOscillator(1, 10_000_000); osc.period != 0 { // q = 1e7 > oscMaxPeriod
		t.Errorf("1 Hz offset: period = %d, want 0 (recurrence fallback)", osc.period)
	}
}

func TestOscillator_PhaseWrapping(t *testing.T) {
	// Run the oscillator for a very long time and verify it doesn't drift
	osc := NewOscillator(1234567.89, 10000000)
	samples := make([]complex64, 100)

	// Process many blocks
	for block := 0; block < 100000; block++ {
		for i := range samples {
			samples[i] = complex(1.0, 0)
		}
		osc.MixInPlace(samples)
	}

	// Verify output still has unit magnitude after 10M samples
	for i := range samples {
		samples[i] = complex(1.0, 0)
	}
	osc.MixInPlace(samples)

	for i, s := range samples {
		mag := math.Sqrt(float64(real(s)*real(s) + imag(s)*imag(s)))
		if math.Abs(mag-1.0) > 1e-4 {
			t.Errorf("after 10M samples: sample[%d] magnitude = %f, expected ~1.0", i, mag)
			break
		}
	}
}
