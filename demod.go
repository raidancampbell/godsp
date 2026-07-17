package dsp

import "math"

// FMDemodulator performs FM demodulation using the phase-difference
// (frequency discriminator) method, and applies de-emphasis.
type FMDemodulator struct {
	prev complex64 // previous sample for phase difference

	// De-emphasis IIR state (single-pole low-pass)
	deemphPrev  float32
	deemphAlpha float32

	// audioHPF strips sub-audible content (CTCSS PL tones 67–254 Hz, DCS,
	// DC drift) from the recorded/played audio. Applied only to the audio
	// output; the raw discriminator stream is left untouched so P25 symbol
	// recovery and fingerprinting see the full bandwidth.
	audioHPF *BiquadCascade

	// plNotch is an optional second-order notch filter tuned to the channel's
	// configured CTCSS PL tone frequency. The 300 Hz HPF alone provides only
	// ~13 dB rejection at 233.6 Hz (the highest standard tones are close to
	// the cutoff); the notch adds ≥40 dB at the exact tone frequency.
	// nil when the channel has no PL tone (CSQ, DPL, P25).
	plNotch *Biquad

	// devLimit clamps the discriminator output (in Hz) before de-emphasis,
	// preventing over-deviation spikes (e.g. from adjacent-channel interference
	// or loud transmitters exceeding the NFM deviation limit) from creating
	// audible clicks in the audio. Only the audio path is clamped; the raw
	// discriminator output is left untouched for P25/fingerprinting.
	// Default: 5000 Hz (standard NFM maximum deviation).
	devLimit float32

	// Discriminator output scaling:
	// Raw atan2 output is in radians/sample. To convert to Hz:
	//   freq_hz = dphi * sampleRate / (2*pi)
	// We scale so output is in Hz of deviation.
	freqScale float32

	sampleRate float64

	// Reusable output buffers to avoid per-call allocation.
	// Valid only until the next call to Demodulate.
	rawBuf   []float32
	audioBuf []float32

	// Reusable scratch for the vectorizable discriminator pass: prodRe/prodIm
	// = re/im of s·conj(prev). Valid only until the next call to Demodulate.
	prodReBuf []float32
	prodImBuf []float32

	// Fingerprinting accumulators (raw discriminator, before de-emphasis)
	freqOffsetSum float64 // running sum for mean
	sampleCount   int64
	devHistogram  [100]int64 // histogram bins: 0-8000 Hz, 80 Hz per bin
}

// NewFMDemodulator creates a demodulator for the given narrowband channel sample rate.
// De-emphasis uses a 300 µs time constant. The audio output is high-passed at
// 300 Hz (6th-order Butterworth) to strip CTCSS/DCS sub-audible content.
func NewFMDemodulator(sampleRate float64) *FMDemodulator {
	// De-emphasis: α = 1 - exp(-1 / (τ * Fs))
	tau := 300e-6 // 300 µs
	alpha := float32(1.0 - math.Exp(-1.0/(tau*sampleRate)))

	return &FMDemodulator{
		deemphAlpha: alpha,
		freqScale:   float32(sampleRate / (2.0 * math.Pi)),
		sampleRate:  sampleRate,
		devLimit:    fmDefaultDevLimit,
		audioHPF:    DesignButterworthHPF(audioHPFCutoff, sampleRate, audioHPFOrder),
	}
}

// SetPLNotch installs a narrow notch filter at the given CTCSS PL tone
// frequency. The notch uses Q=10, giving ~23 Hz bandwidth at 233.6 Hz —
// wide enough to cover the ±1.5 % EIA frequency tolerance while leaving
// the 300–3000 Hz voice band intact.
func (d *FMDemodulator) SetPLNotch(toneHz float64) {
	d.plNotch = DesignNotch(toneHz, plNotchQ, d.sampleRate)
}

// SetDeviationLimit overrides the default ±5 kHz FM deviation clamp.
func (d *FMDemodulator) SetDeviationLimit(hz float32) {
	d.devLimit = hz
}

// CTCSS PL tones run 67–254 Hz; 300 Hz @ 6th order gives ≥13 dB attenuation
// at the highest standard tone (233.6 Hz) and ≥38 dB below 150 Hz, while
// leaving the 300–3000 Hz comm voice band intact.
const (
	audioHPFCutoff = 300.0
	audioHPFOrder  = 6

	// fmDefaultDevLimit is the hard clamp applied to the FM discriminator
	// output (in Hz) before de-emphasis. Over-deviation spikes (e.g. ±12 kHz
	// from adjacent-channel bleed or loud transmitters) create audible clicks
	// after de-emphasis. Standard NFM maximum deviation is ±5 kHz.
	fmDefaultDevLimit float32 = 5000.0

	// plNotchQ is the quality factor for the CTCSS PL tone notch filter.
	// Q=10 at 233.6 Hz gives a 3 dB bandwidth of ~23 Hz, covering the
	// ±1.5 % EIA tolerance while staying well below the 300 Hz voice band.
	plNotchQ = 10.0
)

// Demodulate takes complex IQ samples and returns demodulated audio (float32).
// Returns two slices: raw discriminator output (Hz, for fingerprinting)
// and de-emphasized audio output (Hz, for recording/playback).
// The returned slices are backed by internal buffers and are only valid
// until the next call to Demodulate.
func (d *FMDemodulator) Demodulate(samples []complex64) (raw []float32, audio []float32) {
	n := len(samples)
	if cap(d.rawBuf) < n {
		d.rawBuf = make([]float32, n)
		d.audioBuf = make([]float32, n)
		d.prodReBuf = make([]float32, n)
		d.prodImBuf = make([]float32, n)
	}
	raw = d.rawBuf[:n]
	audio = d.audioBuf[:n]

	// Vectorizable pass: raw[i] = arg(s·conj(prev)) · freqScale. Per-sample
	// independent (only prev chains, at the block seam), so the atan2 is
	// batched and NEON-accelerated on arm64.
	d.discriminate(samples, raw)

	// Serial pass: fingerprint accumulate + deviation clamp + de-emphasis IIR
	// + optional PL notch + audio HPF. Loop-carried; unchanged from the
	// original per-sample logic.
	devLim := d.devLimit
	for i := 0; i < n; i++ {
		freqHz := raw[i]

		// Update fingerprinting accumulators.
		d.freqOffsetSum += float64(freqHz)
		d.sampleCount++
		absHz := math.Abs(float64(freqHz))
		bin := int(absHz / 80.0)
		if bin >= 100 {
			bin = 99
		}
		d.devHistogram[bin]++

		// Clamp deviation for the audio path only. Over-deviation spikes
		// (from adjacent-channel interference or loud transmitters) would
		// otherwise ring through the de-emphasis IIR and create audible
		// clicks in the output.
		audioHz := freqHz
		if audioHz > devLim {
			audioHz = devLim
		} else if audioHz < -devLim {
			audioHz = -devLim
		}

		// De-emphasis IIR: y[n] = α*x[n] + (1-α)*y[n-1]
		d.deemphPrev = d.deemphAlpha*audioHz + (1.0-d.deemphAlpha)*d.deemphPrev

		// Optional PL tone notch (before HPF for additional rejection)
		deemph := d.deemphPrev
		if d.plNotch != nil {
			deemph = d.plNotch.Process(deemph)
		}
		audio[i] = d.audioHPF.Process(deemph)
	}

	return raw, audio
}

// discriminate fills raw[i] = arg(samples[i]·conj(samples[i-1])) · freqScale.
// It builds prodRe/prodIm (per-sample independent; prev chains only at the
// seam), then batches the atan2 via atan2Block (NEON on arm64, poly elsewhere).
func (d *FMDemodulator) discriminate(samples []complex64, raw []float32) {
	n := len(samples)
	pr := d.prodReBuf[:n]
	pi := d.prodImBuf[:n]
	prev := d.prev
	for i, s := range samples {
		// prod = s · conj(prev); conj(prev) = (re, -im)
		pr[i] = real(s)*real(prev) + imag(s)*imag(prev)
		pi[i] = imag(s)*real(prev) - real(s)*imag(prev)
		prev = s
	}
	d.prev = prev
	// raw[i] = atan2(pi[i], pr[i]) · freqScale
	atan2Block(raw, pi, pr, d.freqScale)
}

// FreqOffset returns the mean carrier frequency offset in Hz.
// Call at end of transmission.
func (d *FMDemodulator) FreqOffset() float64 {
	if d.sampleCount == 0 {
		return 0
	}
	return d.freqOffsetSum / float64(d.sampleCount)
}

// Deviation returns the 95th percentile of |discriminator output| in Hz.
// Call at end of transmission.
func (d *FMDemodulator) Deviation() float64 {
	if d.sampleCount == 0 {
		return 0
	}

	threshold := int64(float64(d.sampleCount) * 0.95)
	var cumulative int64
	for i, count := range d.devHistogram {
		cumulative += count
		if cumulative >= threshold {
			return float64(i)*80.0 + 40.0 // bin center
		}
	}
	return 8000.0
}

// ResetAccumulators clears the fingerprinting accumulators for a new transmission.
func (d *FMDemodulator) ResetAccumulators() {
	d.freqOffsetSum = 0
	d.sampleCount = 0
	d.devHistogram = [100]int64{}
}
