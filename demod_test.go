package dsp

import (
	"math"
	"testing"
)

func TestFMDemodulator_DCInput(t *testing.T) {
	demod := NewFMDemodulator(25000)

	// A DC signal (constant complex value) has zero frequency deviation
	samples := make([]complex64, 1000)
	for i := range samples {
		samples[i] = complex(1.0, 0)
	}

	raw, _ := demod.Demodulate(samples)

	// After the first sample (which has no previous), all should be ~0 Hz
	for i := 2; i < len(raw); i++ {
		if math.Abs(float64(raw[i])) > 0.1 {
			t.Errorf("raw[%d] = %f Hz, expected ~0 for DC input", i, raw[i])
			break
		}
	}
}

func TestFMDemodulator_KnownFrequency(t *testing.T) {
	sampleRate := 25000.0
	demod := NewFMDemodulator(sampleRate)

	// Generate a complex sinusoid at 1000 Hz — this should demodulate to +1000 Hz
	freqHz := 1000.0
	samples := make([]complex64, 2000)
	for i := range samples {
		phase := 2.0 * math.Pi * freqHz * float64(i) / sampleRate
		samples[i] = complex(float32(math.Cos(phase)), float32(math.Sin(phase)))
	}

	raw, _ := demod.Demodulate(samples)

	// Check that the steady-state discriminator output is ~1000 Hz
	// Skip the first few samples for settling
	var sum float64
	count := 0
	for i := 100; i < len(raw); i++ {
		sum += float64(raw[i])
		count++
	}
	mean := sum / float64(count)

	if math.Abs(mean-freqHz) > 5.0 {
		t.Errorf("mean discriminator output = %.1f Hz, expected ~%.1f Hz", mean, freqHz)
	}
}

func TestFMDemodulator_FreqOffset(t *testing.T) {
	sampleRate := 25000.0
	demod := NewFMDemodulator(sampleRate)

	// A sinusoid at +500 Hz should yield a freq offset of ~500
	freqHz := 500.0
	samples := make([]complex64, 5000)
	for i := range samples {
		phase := 2.0 * math.Pi * freqHz * float64(i) / sampleRate
		samples[i] = complex(float32(math.Cos(phase)), float32(math.Sin(phase)))
	}

	demod.Demodulate(samples)
	offset := demod.FreqOffset()

	if math.Abs(offset-freqHz) > 5.0 {
		t.Errorf("FreqOffset() = %.1f, expected ~%.1f", offset, freqHz)
	}
}

func TestFMDemodulator_PLNotchRemovesCTCSS(t *testing.T) {
	sampleRate := 25000.0
	plToneHz := 233.6

	// Generate FM signal: carrier at 0 Hz + 233.6 Hz PL tone at ±700 Hz deviation
	// The PL tone appears as a 233.6 Hz sinusoid in the discriminator output.
	// We bypass the FM modulator and feed the discriminator output directly:
	// create IQ samples whose instantaneous frequency is plToneHz * sin(2π*t*plToneHz)
	n := int(sampleRate) * 2 // 2 seconds
	samples := make([]complex64, n)
	var phase float64
	plDeviation := 700.0 // Hz peak deviation of the PL tone
	for i := 0; i < n; i++ {
		t := float64(i) / sampleRate
		instFreq := plDeviation * math.Sin(2.0*math.Pi*plToneHz*t)
		phase += 2.0 * math.Pi * instFreq / sampleRate
		samples[i] = complex(float32(math.Cos(phase)), float32(math.Sin(phase)))
	}

	// Demodulate WITHOUT notch
	demodNoNotch := NewFMDemodulator(sampleRate)
	_, audioNoNotch := demodNoNotch.Demodulate(samples)

	// Demodulate WITH notch
	demodNotch := NewFMDemodulator(sampleRate)
	demodNotch.SetPLNotch(plToneHz)
	_, audioNotch := demodNotch.Demodulate(samples)

	// Measure 233.6 Hz energy via Goertzel in the second half (after settling)
	half := n / 2
	ampNoNotch := goertzelAmp(audioNoNotch[half:], plToneHz, sampleRate)
	ampNotch := goertzelAmp(audioNotch[half:], plToneHz, sampleRate)

	if ampNoNotch < 1.0 {
		t.Fatalf("PL tone not detected without notch: amp=%.2f", ampNoNotch)
	}

	rejectionDB := 20.0 * math.Log10(ampNoNotch/math.Max(ampNotch, 1e-10))
	t.Logf("PL tone amp: without_notch=%.2f, with_notch=%.4f, rejection=%.1f dB",
		ampNoNotch, ampNotch, rejectionDB)

	if rejectionDB < 30.0 {
		t.Errorf("PL notch rejection = %.1f dB, want ≥30 dB", rejectionDB)
	}
}

func TestFMDemodulator_DeviationLimit(t *testing.T) {
	sampleRate := 25000.0

	// Create a signal with a large frequency spike: 500 Hz steady, then
	// a single-sample 12 kHz spike, then 500 Hz again.
	n := 1000
	samples := make([]complex64, n)
	var phase float64
	for i := 0; i < n; i++ {
		instFreq := 500.0
		if i == 500 {
			instFreq = 12000.0 // spike
		}
		phase += 2.0 * math.Pi * instFreq / sampleRate
		samples[i] = complex(float32(math.Cos(phase)), float32(math.Sin(phase)))
	}

	// Demodulate with default limit (5000 Hz)
	demod := NewFMDemodulator(sampleRate)
	raw, audio := demod.Demodulate(samples)

	// Raw discriminator should show the spike
	rawSpike := float64(raw[500])
	if rawSpike < 10000 {
		t.Fatalf("raw spike = %.0f Hz, expected >10000 Hz", rawSpike)
	}

	// Audio around the spike should be clamped — the de-emphasis output
	// after a 5 kHz clamp should be much smaller than after a 12 kHz spike.
	// Check that audio near the spike doesn't exceed ~1000 Hz (de-emphasis
	// attenuates a 5 kHz step to ~625 Hz; the IIR tail decays from there).
	audioAtSpike := math.Abs(float64(audio[501]))
	if audioAtSpike > 1500.0 {
		t.Errorf("audio after spike = %.0f Hz, expected <1500 Hz (deviation limited)", audioAtSpike)
	}

	// Compare with unlimited demod
	demodUnlim := NewFMDemodulator(sampleRate)
	demodUnlim.SetDeviationLimit(12500) // effectively disabled
	_, audioUnlim := demodUnlim.Demodulate(samples)

	unlimSpike := math.Abs(float64(audioUnlim[501]))
	t.Logf("audio after spike: limited=%.1f Hz, unlimited=%.1f Hz", audioAtSpike, unlimSpike)

	if unlimSpike <= audioAtSpike {
		t.Errorf("unlimited audio (%.0f) should exceed limited (%.0f)", unlimSpike, audioAtSpike)
	}
}

// goertzelAmp computes the Goertzel single-frequency magnitude.
func goertzelAmp(samples []float32, freq, sampleRate float64) float64 {
	N := len(samples)
	k := int(math.Round(freq * float64(N) / sampleRate))
	w := 2.0 * math.Pi * float64(k) / float64(N)
	coeff := 2.0 * math.Cos(w)
	var s0, s1, s2 float64
	for _, x := range samples {
		s0 = float64(x) + coeff*s1 - s2
		s2, s1 = s1, s0
	}
	power := s1*s1 + s2*s2 - coeff*s1*s2
	return math.Sqrt(math.Max(0, power)) / float64(N)
}

func TestFMDemodulator_ResetAccumulators(t *testing.T) {
	demod := NewFMDemodulator(25000)

	samples := make([]complex64, 1000)
	for i := range samples {
		phase := 2.0 * math.Pi * 1000.0 * float64(i) / 25000.0
		samples[i] = complex(float32(math.Cos(phase)), float32(math.Sin(phase)))
	}
	demod.Demodulate(samples)

	if demod.FreqOffset() == 0 {
		t.Error("FreqOffset should be non-zero before reset")
	}

	demod.ResetAccumulators()

	if demod.FreqOffset() != 0 {
		t.Errorf("FreqOffset should be 0 after reset, got %f", demod.FreqOffset())
	}
}
