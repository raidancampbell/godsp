package dsp

import (
	"math"
	"testing"
)

// measureGain returns the steady-state amplitude gain of f at frequency freqHz.
// It drives the filter with a unit-amplitude sinusoid, discards the transient
// (first 1/4 of the buffer), then takes the RMS of the rest and scales back to
// peak. For a linear time-invariant filter at a single tone this is equivalent
// to |H(e^{j2π freqHz/sr})|.
func measureGain(c *BiquadCascade, freqHz, sampleRate float64, n int) float64 {
	in := make([]float32, n)
	for i := range in {
		in[i] = float32(math.Sin(2.0 * math.Pi * freqHz * float64(i) / sampleRate))
	}
	c.Reset()
	out := make([]float32, n)
	for i, x := range in {
		out[i] = c.Process(x)
	}
	start := n / 4
	var sumSq float64
	for i := start; i < n; i++ {
		sumSq += float64(out[i]) * float64(out[i])
	}
	rms := math.Sqrt(sumSq / float64(n-start))
	return rms * math.Sqrt2 // RMS → peak for a sinusoid
}

func TestButterworthHPF_CutoffMinus3dB(t *testing.T) {
	sr := 25000.0
	fc := 300.0
	hpf := DesignButterworthHPF(fc, sr, 4)

	gain := measureGain(hpf, fc, sr, 20000)
	gainDB := 20 * math.Log10(gain)
	// Butterworth cutoff is the -3 dB point. Allow 0.5 dB tolerance.
	if math.Abs(gainDB-(-3.0)) > 0.5 {
		t.Errorf("gain at fc=%.0f Hz = %.2f dB, expected ~-3 dB", fc, gainDB)
	}
}

func TestButterworthHPF_PassbandFlat(t *testing.T) {
	sr := 25000.0
	hpf := DesignButterworthHPF(300, sr, 4)

	for _, f := range []float64{1000, 2000, 3000} {
		gain := measureGain(hpf, f, sr, 20000)
		gainDB := 20 * math.Log10(gain)
		if gainDB < -1.0 || gainDB > 0.5 {
			t.Errorf("passband gain at %.0f Hz = %.2f dB, expected ~0 dB", f, gainDB)
		}
	}
}

func TestButterworthHPF_StopbandAttenuation(t *testing.T) {
	sr := 25000.0
	hpf := DesignButterworthHPF(300, sr, 4)

	// 4th-order Butterworth = 24 dB/octave. At 233.6 Hz we expect ~-7 dB,
	// at 100 Hz ~-38 dB, at 60 Hz ~-56 dB.
	cases := []struct {
		freq    float64
		maxGain float64 // dB; gain must be below this
	}{
		{233.6, -5.0},  // CTCSS PL 4Z — weakest expected attenuation
		{150.0, -18.0}, // mid-CTCSS band
		{100.0, -30.0}, // strong attenuation
		{60.0, -45.0},  // mains hum
	}
	for _, c := range cases {
		gain := measureGain(hpf, c.freq, sr, 30000)
		gainDB := 20 * math.Log10(gain)
		if gainDB > c.maxGain {
			t.Errorf("stopband gain at %.1f Hz = %.2f dB, expected <%.1f dB",
				c.freq, gainDB, c.maxGain)
		}
	}
}

func TestButterworthHPF_DCBlocked(t *testing.T) {
	hpf := DesignButterworthHPF(300, 25000, 4)
	// Constant input → steady-state output must converge to 0.
	last := float32(0)
	for i := 0; i < 50000; i++ {
		last = hpf.Process(1.0)
	}
	if math.Abs(float64(last)) > 1e-3 {
		t.Errorf("DC output not blocked: steady-state = %g", last)
	}
}

func TestButterworthHPF_PanicsOnOddOrder(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("expected panic on odd order")
		}
	}()
	DesignButterworthHPF(300, 25000, 3)
}

func TestDesignButterworthLPFPassesLowBand(t *testing.T) {
	lpf := DesignButterworthLPF(1000, 8000, 6)
	gain := measureGain(lpf, 100, 8000, 4000)
	ratio := gain // input is unit amplitude
	if ratio < 0.89 || ratio > 1.1 {
		t.Errorf("100 Hz passband: gain = %.3f, want ~1.0", ratio)
	}
}

func TestDesignButterworthLPFAttenuatesHighBand(t *testing.T) {
	lpf := DesignButterworthLPF(1000, 8000, 6)
	gain := measureGain(lpf, 3000, 8000, 4000)
	if gain > 0.05 {
		t.Errorf("3 kHz stopband: gain = %.4f, want < 0.05", gain)
	}
}

func TestDesignNotch_RejectsCenterFreq(t *testing.T) {
	sr := 25000.0
	notch := DesignNotch(233.6, 10.0, sr)
	// Wrap in a cascade for measureGain
	cascade := &BiquadCascade{sections: []Biquad{*notch}}

	// At center frequency: expect deep notch (≥30 dB rejection)
	gain := measureGain(cascade, 233.6, sr, 50000)
	gainDB := 20 * math.Log10(gain)
	if gainDB > -30.0 {
		t.Errorf("notch gain at 233.6 Hz = %.1f dB, expected <-30 dB", gainDB)
	}
}

func TestDesignNotch_PassesVoiceBand(t *testing.T) {
	sr := 25000.0
	notch := DesignNotch(233.6, 10.0, sr)
	cascade := &BiquadCascade{sections: []Biquad{*notch}}

	// Voice band (300–3000 Hz) should be essentially unaffected
	for _, f := range []float64{300, 500, 1000, 2000, 3000} {
		gain := measureGain(cascade, f, sr, 50000)
		gainDB := 20 * math.Log10(gain)
		if gainDB < -1.0 || gainDB > 0.5 {
			t.Errorf("voice band gain at %.0f Hz = %.2f dB, expected ~0 dB", f, gainDB)
		}
	}
}

func TestDesignButterworthLPFPanicsOnOddOrder(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("expected panic on odd order")
		}
	}()
	DesignButterworthLPF(1000, 8000, 3)
}

// TestDesignNotch_PanicsOnZeroQ verifies the q precondition guard: q == 0
// divides by zero in alpha and yields Inf/NaN coefficients, so the designer
// must panic.
func TestDesignNotch_PanicsOnZeroQ(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("expected panic on q=0")
		}
	}()
	DesignNotch(233.6, 0.0, 25000)
}

// TestDesignNotch_PanicsAtOrAboveNyquist verifies the center-frequency guard:
// a center at or above Nyquist places the notch outside the representable band.
func TestDesignNotch_PanicsAtOrAboveNyquist(t *testing.T) {
	for _, cf := range []float64{12500.0 /* == Nyquist */, 20000.0 /* > Nyquist */} {
		func() {
			defer func() {
				if r := recover(); r == nil {
					t.Errorf("expected panic on centerFreq=%v", cf)
				}
			}()
			DesignNotch(cf, 10.0, 25000)
		}()
	}
}

// TestDesignNotch_ValidArgsNoNaN confirms the guards are no-ops for valid args
// and that the resulting coefficients are finite.
func TestDesignNotch_ValidArgsNoNaN(t *testing.T) {
	b := DesignNotch(233.6, 10.0, 25000)
	for name, c := range map[string]float32{"b0": b.b0, "b1": b.b1, "b2": b.b2, "a1": b.a1, "a2": b.a2} {
		if math.IsNaN(float64(c)) || math.IsInf(float64(c), 0) {
			t.Fatalf("coefficient %s is non-finite: %v", name, c)
		}
	}
}

// TestDesignButterworthLPF_PanicsAtOrAboveNyquist verifies the cutoff guard:
// a cutoff at or above Nyquist warps to an out-of-band pole, yielding an
// unstable section.
func TestDesignButterworthLPF_PanicsAtOrAboveNyquist(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("expected panic on cutoff at Nyquist")
		}
	}()
	DesignButterworthLPF(4000, 8000, 6)
}

// TestDesignButterworthHPF_PanicsAtOrAboveNyquist verifies the same cutoff
// guard for the high-pass designer.
func TestDesignButterworthHPF_PanicsAtOrAboveNyquist(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("expected panic on cutoff above Nyquist")
		}
	}()
	DesignButterworthHPF(5000, 8000, 6)
}
