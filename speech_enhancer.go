package dsp

import "math"

// SpeechEnhancer applies real-time audio enhancement to vocoder output
// to improve intelligibility. It chains: high-pass filter → pre-emphasis →
// AGC → soft limiter. All state is maintained per-transmission.
type SpeechEnhancer struct {
	hp       *BiquadCascade // high-pass to remove vocoder rumble
	preAlpha float32        // pre-emphasis coefficient (0 = disabled)
	prevSamp float32        // pre-emphasis state

	// AGC state
	agcTargetRMS float32 // target RMS level (in mbelib float space)
	agcGain      float32 // current smoothed gain
	agcAttack    float32 // attack smoothing coefficient (fast, for loud)
	agcRelease   float32 // release smoothing coefficient (slow, for quiet)
	agcMaxGain   float32 // maximum gain to avoid amplifying noise

	// Limiter
	limitThresh float32 // soft-knee threshold
}

// SpeechEnhancerConfig holds tunable parameters for the enhancer.
type SpeechEnhancerConfig struct {
	HighpassHz   float64 // high-pass cutoff (0 = disabled), default 250
	PreEmphasis  float32 // pre-emphasis alpha (0 = disabled), default 0.5
	AGCTargetRMS float32 // target RMS in vocoder output scale, default 180
	AGCMaxGain   float32 // max gain multiplier, default 10
	LimitThresh  float32 // soft limiter threshold, default 800
}

// DefaultSpeechEnhancerConfig returns reasonable defaults for P25 IMBE output.
// mbelib outputs speech with peaks typically ±500–2000 (floattoshort divisor=7
// would map ±32767 → ±4681, but real speech is well below that).
func DefaultSpeechEnhancerConfig() SpeechEnhancerConfig {
	return SpeechEnhancerConfig{
		HighpassHz:   250,
		PreEmphasis:  0.5,
		AGCTargetRMS: 180, // in mbelib float space; with p25Gain=28 → ~5040 int16 RMS
		AGCMaxGain:   10,
		LimitThresh:  800, // with p25Gain=28 → ~22400 int16, leaves headroom
	}
}

// NewSpeechEnhancer creates a new enhancer with the given config.
// sampleRate should be AudioRate (8000).
func NewSpeechEnhancer(cfg SpeechEnhancerConfig, sampleRate float64) *SpeechEnhancer {
	e := &SpeechEnhancer{
		preAlpha:     cfg.PreEmphasis,
		agcTargetRMS: cfg.AGCTargetRMS,
		agcGain:      1.0,
		agcMaxGain:   cfg.AGCMaxGain,
		limitThresh:  cfg.LimitThresh,
	}

	// HP filter
	if cfg.HighpassHz > 0 {
		e.hp = DesignButterworthHPF(cfg.HighpassHz, sampleRate, 4)
	}

	// AGC smoothing: attack ~5ms, release ~100ms at 8kHz
	// Per-codeword (160 samples = 20ms), so use per-block smoothing
	e.agcAttack = float32(1.0 - math.Exp(-1.0/(sampleRate*0.005)))
	e.agcRelease = float32(1.0 - math.Exp(-1.0/(sampleRate*0.100)))

	return e
}

// Process enhances a block of PCM samples in-place (typically 160 samples = 1 IMBE codeword).
// Samples are in mbelib float output scale (NOT int16).
func (e *SpeechEnhancer) Process(pcm []float32) {
	if len(pcm) == 0 {
		return
	}

	// 1. High-pass filter
	if e.hp != nil {
		for i, s := range pcm {
			pcm[i] = e.hp.Process(s)
		}
	}

	// 2. Pre-emphasis
	if e.preAlpha > 0 {
		for i, s := range pcm {
			out := s - e.preAlpha*e.prevSamp
			e.prevSamp = s
			pcm[i] = out
		}
	}

	// 3. AGC — compute block RMS, update gain
	var sumSq float64
	for _, s := range pcm {
		sumSq += float64(s) * float64(s)
	}
	rms := float32(math.Sqrt(sumSq / float64(len(pcm))))

	if rms > 1.0 { // don't adjust for silence
		desiredGain := e.agcTargetRMS / rms
		if desiredGain > e.agcMaxGain {
			desiredGain = e.agcMaxGain
		}
		if desiredGain < 0.1 {
			desiredGain = 0.1
		}
		// Smooth gain: fast attack (getting quieter = reducing gain), slow release (increasing gain)
		var alpha float32
		if desiredGain < e.agcGain {
			alpha = e.agcAttack
		} else {
			alpha = e.agcRelease
		}
		// Per-block smoothing (not per-sample), scale alpha by block size
		blockAlpha := float32(1.0 - math.Pow(float64(1.0-alpha), float64(len(pcm))))
		e.agcGain += blockAlpha * (desiredGain - e.agcGain)
	}

	// Apply gain
	for i := range pcm {
		pcm[i] *= e.agcGain
	}

	// 4. Soft limiter (tanh-style)
	if e.limitThresh > 0 {
		for i, s := range pcm {
			if s > e.limitThresh || s < -e.limitThresh {
				// Soft clip: beyond threshold, compress with tanh
				sign := float32(1.0)
				if s < 0 {
					sign = -1.0
					s = -s
				}
				excess := (s - e.limitThresh) / e.limitThresh
				compressed := e.limitThresh + e.limitThresh*float32(math.Tanh(float64(excess)))
				pcm[i] = sign * compressed
			}
		}
	}
}

// Reset clears all filter and AGC state (call at transmission boundaries).
func (e *SpeechEnhancer) Reset() {
	if e.hp != nil {
		e.hp.Reset()
	}
	e.prevSamp = 0
	e.agcGain = 1.0
}

