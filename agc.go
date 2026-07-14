package dsp

import "math"

// IQAGC is a feed-forward automatic gain control for complex baseband. It
// tracks the input power with an exponential moving average and applies a
// single scalar gain to drive the RMS magnitude toward targetRMS, clamped to
// maxGain. Gain is applied per sample after updating the estimate, so it
// adapts within a buffer.
type IQAGC struct {
	targetPow float64 // targetRMS^2
	attack    float64 // EMA coefficient in (0,1]; larger = faster
	maxGain   float64
	powEMA    float64
	seeded    bool
}

// NewIQAGC creates an IQ AGC. targetRMS is the desired output RMS magnitude,
// attack in (0,1] sets adaptation speed, maxGain caps amplification.
func NewIQAGC(targetRMS, attack, maxGain float64) *IQAGC {
	if attack <= 0 || attack > 1 {
		attack = 0.01
	}
	if maxGain <= 0 {
		maxGain = 1
	}
	return &IQAGC{targetPow: targetRMS * targetRMS, attack: attack, maxGain: maxGain}
}

// ProcessInPlace scales samples toward the target level.
func (a *IQAGC) ProcessInPlace(samples []complex64) {
	for i, s := range samples {
		re, im := float64(real(s)), float64(imag(s))
		pow := re*re + im*im
		if !a.seeded {
			a.powEMA = pow
			a.seeded = true
		} else {
			a.powEMA += a.attack * (pow - a.powEMA)
		}
		g := a.gainFor(a.powEMA)
		samples[i] = complex(float32(re*g), float32(im*g))
	}
}

func (a *IQAGC) gainFor(powEMA float64) float64 {
	if powEMA <= 1e-30 {
		return a.maxGain
	}
	g := math.Sqrt(a.targetPow / powEMA)
	if g > a.maxGain {
		g = a.maxGain
	}
	return g
}

// Gain returns the most recent applied gain (based on the current estimate).
func (a *IQAGC) Gain() float64 {
	if !a.seeded {
		return 1
	}
	return a.gainFor(a.powEMA)
}

// AudioAGC is an envelope-follower AGC for real audio. It tracks the signal
// envelope with asymmetric attack/release smoothing and drives it toward
// targetLevel, clamped to maxGain.
type AudioAGC struct {
	target  float64
	attack  float64
	release float64
	maxGain float64
	env     float64
}

// NewAudioAGC creates an audio AGC. attack applies when the envelope rises,
// release when it falls; both in (0,1].
func NewAudioAGC(targetLevel, attack, release, maxGain float64) *AudioAGC {
	if attack <= 0 || attack > 1 {
		attack = 0.01
	}
	if release <= 0 || release > 1 {
		release = 0.001
	}
	if maxGain <= 0 {
		maxGain = 1
	}
	return &AudioAGC{target: targetLevel, attack: attack, release: release, maxGain: maxGain}
}

// ProcessInPlace scales audio toward the target level.
func (a *AudioAGC) ProcessInPlace(samples []float32) {
	for i, s := range samples {
		mag := math.Abs(float64(s))
		if mag > a.env {
			a.env += a.attack * (mag - a.env)
		} else {
			a.env += a.release * (mag - a.env)
		}
		g := a.maxGain
		if a.env > 1e-9 {
			g = a.target / a.env
			if g > a.maxGain {
				g = a.maxGain
			}
		}
		samples[i] = float32(float64(s) * g)
	}
}
