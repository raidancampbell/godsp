package dsp

import (
	"math"
	"math/rand"
	"testing"
)

// noiseFloorDB returns 10*log10(mean power) of the quietest 20% of 10ms frames.
func quietPowerDB(x []float32, sr int) float64 {
	fr := sr / 100 // 10ms
	if fr < 1 {
		fr = 1
	}
	var powers []float64
	for i := 0; i+fr <= len(x); i += fr {
		var s float64
		for _, v := range x[i : i+fr] {
			s += float64(v) * float64(v)
		}
		powers = append(powers, s/float64(fr))
	}
	// nearest-rank 20th percentile
	for i := 1; i < len(powers); i++ {
		for j := i; j > 0 && powers[j] < powers[j-1]; j-- {
			powers[j], powers[j-1] = powers[j-1], powers[j]
		}
	}
	p := powers[len(powers)/5]
	return 10 * math.Log10(p+1e-12)
}

func TestDenoiseReducesNoiseFloorPreservesSignal(t *testing.T) {
	sr := 16000
	n := sr * 3
	clean := make([]float32, n)
	for i := range clean {
		t := float64(i) / float64(sr)
		// voiced-ish: gate the tone on/off so there are quiet frames
		env := 0.0
		if math.Mod(t, 1.0) < 0.6 {
			env = 1.0
		}
		clean[i] = float32(env * 0.5 * (math.Sin(2*math.Pi*500*t) + math.Sin(2*math.Pi*1200*t)))
	}
	rng := rand.New(rand.NewSource(1))
	noisy := make([]float32, n)
	for i := range noisy {
		noisy[i] = clean[i] + float32(rng.NormFloat64()*0.05)
	}

	d := NewSpectralDenoiser(DefaultDenoiseConfig())
	out := d.Process(noisy)

	if len(out) != len(noisy) {
		t.Fatalf("length changed: got %d want %d", len(out), len(noisy))
	}
	before := quietPowerDB(noisy, sr)
	after := quietPowerDB(out, sr)
	if before-after < 8 {
		t.Fatalf("noise floor reduction too small: before=%.1f after=%.1f (%.1f dB)", before, after, before-after)
	}
	// signal preserved: correlation with clean over loud frames stays high
	var sxy, sxx, syy float64
	for i := range clean {
		if math.Abs(float64(clean[i])) < 0.05 {
			continue
		}
		a, b := float64(clean[i]), float64(out[i])
		sxy += a * b
		sxx += a * a
		syy += b * b
	}
	corr := sxy / (math.Sqrt(sxx*syy) + 1e-12)
	if corr < 0.95 {
		t.Fatalf("signal correlation too low: %.3f", corr)
	}
}

func TestDenoiseShortClipPassthrough(t *testing.T) {
	d := NewSpectralDenoiser(DefaultDenoiseConfig())
	in := make([]float32, 200) // < NFFT
	for i := range in {
		in[i] = float32(i)
	}
	out := d.Process(in)
	if len(out) != len(in) {
		t.Fatalf("length changed: %d vs %d", len(out), len(in))
	}
	for i := range in {
		if out[i] != in[i] {
			t.Fatalf("short clip modified at %d: %v vs %v", i, out[i], in[i])
		}
	}
}

func TestDenoiseRobustness(t *testing.T) {
	d := NewSpectralDenoiser(DefaultDenoiseConfig())
	cases := map[string][]float32{
		"silence":   make([]float32, 16000),
		"fullscale": nil,
	}
	fs := make([]float32, 16000)
	for i := range fs {
		fs[i] = 1.0
		if i%2 == 0 {
			fs[i] = -1.0
		}
	}
	cases["fullscale"] = fs
	for name, in := range cases {
		out := d.Process(in)
		for i, v := range out {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				t.Fatalf("%s: non-finite output at %d: %v", name, i, v)
			}
			if math.Abs(float64(v)) > 1.5 {
				t.Fatalf("%s: output magnitude exceeds input scale at %d: %v", name, i, v)
			}
		}
	}
}
