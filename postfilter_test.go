package dsp

import (
	"math"
	"testing"
)

func TestLevinsonDurbin_KnownSignal(t *testing.T) {
	// AR(2) process: x(n) = 0.5*x(n-1) - 0.3*x(n-2) + noise
	// Expected LPC coefficients: a[1] ≈ -0.5, a[2] ≈ 0.3 (sign convention)
	// Generate a long signal to get stable autocorrelation
	const N = 10000
	x := make([]float32, N)
	x[0], x[1] = 1.0, 0.5
	for n := 2; n < N; n++ {
		x[n] = 0.5*x[n-1] - 0.3*x[n-2]
	}

	R := Autocorrelation(x, 2)
	a, E := LevinsonDurbin(R, 2)

	if E < 0 {
		t.Fatalf("negative prediction error")
	}
	// a[1] should be close to -0.5, a[2] close to 0.3
	if math.Abs(a[1]-(-0.5)) > 0.01 {
		t.Errorf("a[1] = %f, want ≈ -0.5", a[1])
	}
	if math.Abs(a[2]-0.3) > 0.01 {
		t.Errorf("a[2] = %f, want ≈ 0.3", a[2])
	}
}

func TestPitchPostFilter_SineWave(t *testing.T) {
	// 200 Hz sine at 8 kHz → T=40 samples
	const T = 40
	pf := NewPerceptualPostFilter(PostFilterConfig{
		PitchGain:    0.5,
		PitchEnabled: true,
		GainSmooth:   0.9,
	})

	// Prime the buffer with one frame
	prime := make([]float32, 160)
	for i := range prime {
		prime[i] = float32(math.Sin(2 * math.Pi * 200 * float64(i) / 8000))
	}
	pf.pitchPostFilter(prime, T, true)

	// Process a second frame
	input := make([]float32, 160)
	for i := range input {
		input[i] = float32(math.Sin(2 * math.Pi * 200 * float64(i+160) / 8000))
	}
	inputCopy := make([]float32, 160)
	copy(inputCopy, input)

	pf.pitchPostFilter(input, T, true)

	// Output should have more energy (pitch filter reinforces periodicity)
	var eIn, eOut float64
	for i := range input {
		eIn += float64(inputCopy[i]) * float64(inputCopy[i])
		eOut += float64(input[i]) * float64(input[i])
	}
	if eOut <= eIn {
		t.Errorf("pitch post-filter did not boost periodic signal: eIn=%f eOut=%f", eIn, eOut)
	}
}

func TestPitchPostFilter_Unvoiced(t *testing.T) {
	pf := NewPerceptualPostFilter(PostFilterConfig{
		PitchGain:    0.5,
		PitchEnabled: true,
		GainSmooth:   0.9,
	})

	// White noise-like input, unvoiced
	input := make([]float32, 160)
	for i := range input {
		input[i] = float32(i%7) - 3.0
	}
	inputCopy := make([]float32, 160)
	copy(inputCopy, input)

	pf.pitchPostFilter(input, 40, false) // voiced=false → bypass

	for i := range input {
		if input[i] != inputCopy[i] {
			t.Fatalf("unvoiced frame was modified at sample %d", i)
		}
	}
}

func TestFormantPostFilter_FlatSpectrum(t *testing.T) {
	// White noise → LPC coefficients are small → filter is near identity
	pf := NewPerceptualPostFilter(PostFilterConfig{
		FormantGammaN:  0.5,
		FormantGammaD:  0.8,
		FormantEnabled: true,
		GainSmooth:     0.9,
	})

	// Use pseudo-random signal (deterministic) that's spectrally flat-ish
	input := make([]float32, 160)
	state := uint32(12345)
	for i := range input {
		state = state*1103515245 + 12345
		input[i] = float32(int32(state>>16)&0xFFFF-0x8000) / 100.0
	}

	var eIn float64
	for _, s := range input {
		eIn += float64(s) * float64(s)
	}

	pf.formantPostFilter(input)

	var eOut float64
	for _, s := range input {
		eOut += float64(s) * float64(s)
	}

	// Energy should be within ±6 dB for a flat-spectrum signal
	ratio := 10 * math.Log10(eOut/eIn)
	if math.Abs(ratio) > 6.0 {
		t.Errorf("flat-spectrum energy ratio %.1f dB exceeds ±6 dB", ratio)
	}
}

func TestAdaptiveGain_EnergyPreservation(t *testing.T) {
	pf := NewPerceptualPostFilter(DefaultPostFilterConfig())

	// Process many frames to let gain fully stabilize
	for frame := 0; frame < 50; frame++ {
		buf := make([]float32, 160)
		for i := range buf {
			buf[i] = float32(math.Sin(2*math.Pi*200*float64(i+frame*160)/8000)) * 500
		}
		pf.Process(buf, 40, true)
	}

	// Now measure on a stable frame
	input := make([]float32, 160)
	for i := range input {
		input[i] = float32(math.Sin(2*math.Pi*200*float64(i+1600)/8000)) * 500
	}

	var eIn float64
	for _, s := range input {
		eIn += float64(s) * float64(s)
	}

	pf.Process(input, 40, true)

	var eOut float64
	for _, s := range input {
		eOut += float64(s) * float64(s)
	}

	// Energy should be preserved within ±3 dB after stabilization
	ratio := 10 * math.Log10(eOut/eIn)
	if math.Abs(ratio) > 3.0 {
		t.Errorf("energy ratio %.1f dB exceeds ±3 dB tolerance", ratio)
	}
}

func TestPostFilter_Reset(t *testing.T) {
	pf := NewPerceptualPostFilter(DefaultPostFilterConfig())

	// Process some data
	input := make([]float32, 160)
	for i := range input {
		input[i] = float32(math.Sin(2*math.Pi*150*float64(i)/8000)) * 300
	}
	pf.Process(input, 53, true)

	// Reset
	pf.Reset()

	// Verify state is cleared
	if pf.pitchBufIdx != 0 {
		t.Error("pitchBufIdx not cleared")
	}
	if pf.prevGain != 1.0 {
		t.Errorf("prevGain = %f, want 1.0", pf.prevGain)
	}
	if pf.tiltPrev != 0 {
		t.Error("tiltPrev not cleared")
	}
}




