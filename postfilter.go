package dsp

import "math"

const lpcOrder = 10

// PostFilterConfig holds tunable parameters for the perceptual post-filter.
type PostFilterConfig struct {
	PitchGain      float32 // max pitch reinforcement gain (default 0.5)
	PitchEnabled   bool    // enable pitch post-filter
	FormantGammaN  float64 // numerator bandwidth expansion (default 0.5)
	FormantGammaD  float64 // denominator bandwidth expansion (default 0.8)
	FormantEnabled bool    // enable formant post-filter
	TiltMu         float32 // tilt compensation factor (default 0.35)
	GainSmooth     float32 // gain smoothing factor (default 0.9)
}

// DefaultPostFilterConfig returns recommended defaults for IMBE/AMBE+2 output.
func DefaultPostFilterConfig() PostFilterConfig {
	return PostFilterConfig{
		PitchGain:      0.5,
		PitchEnabled:   true,
		FormantGammaN:  0.5,
		FormantGammaD:  0.8,
		FormantEnabled: true,
		TiltMu:         0.35,
		GainSmooth:     0.9,
	}
}

// PerceptualPostFilter applies pitch + formant + gain post-filtering to
// vocoder output to improve speech clarity. Stateful; one instance per transmission.
type PerceptualPostFilter struct {
	cfg PostFilterConfig

	// Stage 1: Pitch post-filter
	pitchBuf    [160]float32
	pitchBufIdx int
	prevT       [3]int
	prevTIdx    int

	// Stage 2: Formant post-filter
	firState [lpcOrder]float32
	iirState [lpcOrder]float32

	// Stage 3: Tilt + gain
	tiltPrev float32
	prevGain float32
}

// NewPerceptualPostFilter creates a new post-filter with the given config.
// Zero-valued fields are replaced with defaults.
func NewPerceptualPostFilter(cfg PostFilterConfig) *PerceptualPostFilter {
	defaults := DefaultPostFilterConfig()
	if cfg.PitchGain == 0 {
		cfg.PitchGain = defaults.PitchGain
	}
	if cfg.FormantGammaN == 0 {
		cfg.FormantGammaN = defaults.FormantGammaN
	}
	if cfg.FormantGammaD == 0 {
		cfg.FormantGammaD = defaults.FormantGammaD
	}
	if cfg.TiltMu == 0 {
		cfg.TiltMu = defaults.TiltMu
	}
	if cfg.GainSmooth == 0 {
		cfg.GainSmooth = defaults.GainSmooth
	}
	return &PerceptualPostFilter{
		cfg:      cfg,
		prevGain: 1.0,
	}
}

// Process applies the three-stage perceptual post-filter to a 160-sample PCM block.
// pitchPeriod is from the vocoder (in samples at 8 kHz); voiced indicates voicing state.
func (p *PerceptualPostFilter) Process(pcm []float32, pitchPeriod int, voiced bool) {
	if len(pcm) == 0 {
		return
	}

	// Save input energy for gain normalization
	var eIn float64
	for _, s := range pcm {
		eIn += float64(s) * float64(s)
	}

	// Stage 1: Pitch post-filter
	if p.cfg.PitchEnabled {
		p.pitchPostFilter(pcm, pitchPeriod, voiced)
	}

	// Stage 2: Formant post-filter (voiced only)
	var k1 float64
	if p.cfg.FormantEnabled && voiced {
		k1 = p.formantPostFilter(pcm)
	}

	// Stage 3: Tilt compensation (only if formant filter was applied)
	if p.cfg.FormantEnabled && voiced && p.cfg.TiltMu > 0 {
		p.tiltCompensate(pcm, float32(k1))
	}

	// Stage 3b: Adaptive gain normalization
	var eOut float64
	for _, s := range pcm {
		eOut += float64(s) * float64(s)
	}
	if eOut > 1e-10 && eIn > 1e-10 {
		g := float32(math.Sqrt(eIn / eOut))
		// Smooth gain
		g = p.cfg.GainSmooth*p.prevGain + (1-p.cfg.GainSmooth)*g
		p.prevGain = g
		for i := range pcm {
			pcm[i] *= g
		}
	}
}

func (p *PerceptualPostFilter) pitchPostFilter(pcm []float32, pitchPeriod int, voiced bool) {
	if !voiced || pitchPeriod <= 0 || pitchPeriod > 160 {
		// Update buffer with input, no modification
		for i, s := range pcm {
			p.pitchBuf[(p.pitchBufIdx+i)%160] = s
		}
		p.pitchBufIdx = (p.pitchBufIdx + len(pcm)) % 160
		return
	}

	T := p.smoothT(pitchPeriod)

	// Compute adaptive gain β from normalized autocorrelation
	var r0, rT float64
	for i, s := range pcm {
		r0 += float64(s) * float64(s)
		delayIdx := (p.pitchBufIdx + 160 - T + i) % 160
		delayed := p.pitchBuf[delayIdx]
		rT += float64(s) * float64(delayed)
	}

	var beta float32
	if r0 > 0 {
		beta = p.cfg.PitchGain * float32(rT/r0)
		if beta > 0.5 {
			beta = 0.5
		}
		if beta < 0 {
			beta = 0
		}
	}

	// Apply: y(n) = x(n) + β·x(n-T), store original in buffer
	for i, s := range pcm {
		delayIdx := (p.pitchBufIdx + 160 - T + i) % 160
		delayed := p.pitchBuf[delayIdx]
		pcm[i] = s + beta*delayed
		p.pitchBuf[(p.pitchBufIdx+i)%160] = s // store pre-filter
	}
	p.pitchBufIdx = (p.pitchBufIdx + len(pcm)) % 160
}

func (p *PerceptualPostFilter) smoothT(newT int) int {
	// Store and get median of last 3
	p.prevT[p.prevTIdx%3] = newT
	p.prevTIdx++

	a, b, c := p.prevT[0], p.prevT[1], p.prevT[2]
	// Median of three
	if a > b {
		a, b = b, a
	}
	if b > c {
		b = c
	}
	if a > b {
		b = a
	}
	if b <= 0 {
		return newT
	}
	return b
}

// formantPostFilter applies LPC-based spectral shaping. Returns k1 (first reflection coeff).
func (p *PerceptualPostFilter) formantPostFilter(pcm []float32) float64 {
	// Compute LPC
	R := Autocorrelation(pcm, lpcOrder)
	if R[0] <= 0 {
		return 0
	}

	a, _ := LevinsonDurbin(R, lpcOrder)
	k1 := a[1] // first reflection coefficient (approximation)

	// Expand bandwidths
	aN := ExpandBandwidth(a, p.cfg.FormantGammaN)
	aD := ExpandBandwidth(a, p.cfg.FormantGammaD)

	// Apply FIR (numerator) then IIR (denominator)
	for i := range pcm {
		x := pcm[i]

		// FIR: y1(n) = x(n) + aN[1]*x(n-1) + aN[2]*x(n-2) + ...
		y1 := float64(x)
		for k := 1; k <= lpcOrder; k++ {
			y1 += aN[k] * float64(p.firState[k-1])
		}
		// Shift FIR state
		copy(p.firState[1:], p.firState[:lpcOrder-1])
		p.firState[0] = x

		// IIR: y(n) = y1(n) - aD[1]*y(n-1) - aD[2]*y(n-2) - ...
		y := y1
		for k := 1; k <= lpcOrder; k++ {
			y -= aD[k] * float64(p.iirState[k-1])
		}
		// Shift IIR state
		copy(p.iirState[1:], p.iirState[:lpcOrder-1])
		p.iirState[0] = float32(y)

		pcm[i] = float32(y)
	}

	return k1
}

func (p *PerceptualPostFilter) tiltCompensate(pcm []float32, k1 float32) {
	coeff := p.cfg.TiltMu * k1
	for i, s := range pcm {
		out := s - coeff*p.tiltPrev
		p.tiltPrev = out
		pcm[i] = out
	}
}

// Reset clears all filter state (call at transmission boundaries).
func (p *PerceptualPostFilter) Reset() {
	p.pitchBuf = [160]float32{}
	p.pitchBufIdx = 0
	p.prevT = [3]int{}
	p.prevTIdx = 0
	p.firState = [lpcOrder]float32{}
	p.iirState = [lpcOrder]float32{}
	p.tiltPrev = 0
	p.prevGain = 1.0
}

