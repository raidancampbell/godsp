package dsp

import (
	"math/cmplx"
)

// DenoiseConfig tunes the spectral-subtraction denoiser. Parameters are
// expressed in bins/fractions so the same config works at any sample rate.
type DenoiseConfig struct {
	NFFT            int     // FFT/frame size, power of two (default 512)
	Overlap         float64 // STFT overlap fraction, 0..1 (default 0.75)
	NoisePercentile float64 // per-bin noise estimate percentile, 0..100 (default 15)
	Oversubtract    float64 // noise over-subtraction factor (default 2.0)
	SpectralFloor   float64 // residual floor as a fraction of |X| (default 0.05)
}

// DefaultDenoiseConfig returns the validated defaults from the design spec.
func DefaultDenoiseConfig() DenoiseConfig {
	return DenoiseConfig{
		NFFT:            512,
		Overlap:         0.75,
		NoisePercentile: 15,
		Oversubtract:    2.0,
		SpectralFloor:   0.05,
	}
}

// minDenoiseFrames is the smallest number of STFT frames for a usable per-bin
// noise percentile. Shorter clips pass through unchanged.
const minDenoiseFrames = 8

// SpectralDenoiser removes a stationary broadband noise floor from a whole
// audio buffer via STFT spectral subtraction. One instance is reusable across
// buffers; Process holds no cross-call state.
type SpectralDenoiser struct {
	cfg  DenoiseConfig
	nfft int
	hop  int
	win  []float64
}

// NewSpectralDenoiser builds a denoiser, sanitizing the config to a usable
// power-of-two NFFT and a hop of at least one sample.
func NewSpectralDenoiser(cfg DenoiseConfig) *SpectralDenoiser {
	nfft := cfg.NFFT
	if nfft < 64 || nfft&(nfft-1) != 0 {
		nfft = 512
	}
	ov := cfg.Overlap
	if ov < 0 || ov >= 1 {
		ov = 0.75
	}
	hop := int(float64(nfft) * (1 - ov))
	if hop < 1 {
		hop = 1
	}
	return &SpectralDenoiser{cfg: cfg, nfft: nfft, hop: hop, win: HannWindow(nfft)}
}

// Process returns a denoised copy of pcm (same length). Clips shorter than one
// FFT frame (< NFFT samples) are returned unchanged.
func (d *SpectralDenoiser) Process(pcm []float32) []float32 {
	n := len(pcm)
	if n < d.nfft {
		// Short-clip passthrough: return a copy to uphold the "new slice" contract.
		out := make([]float32, len(pcm))
		copy(out, pcm)
		return out
	}

	// Reflect-pad the input by nfft on each side to ensure edge samples get
	// full window coverage under COLA.
	pad := d.nfft
	ext := make([]float32, n+2*pad)
	copy(ext[pad:pad+n], pcm)
	for i := 0; i < pad; i++ {
		ext[pad-1-i] = pcm[min(i+1, n-1)]
		ext[pad+n+i] = pcm[max(n-2-i, 0)]
	}

	nframes := (len(ext)-d.nfft)/d.hop + 1
	// Defensive guard: effectively unreachable for n >= d.nfft after padding, but
	// kept as a backstop in case future changes alter the reflect-pad logic.
	if nframes < minDenoiseFrames {
		out := make([]float32, len(pcm))
		copy(out, pcm)
		return out
	}
	bins := d.nfft/2 + 1

	// Forward STFT: keep each frame's full complex spectrum.
	frames := make([][]complex128, nframes)
	for f := 0; f < nframes; f++ {
		buf := make([]complex128, d.nfft)
		start := f * d.hop
		for i := 0; i < d.nfft; i++ {
			buf[i] = complex(float64(ext[start+i])*d.win[i], 0)
		}
		FFT(buf)
		frames[f] = buf
	}

	// Per-bin noise floor = NoisePercentile of |X| across all frames.
	noise := make([]float64, bins)
	col := make([]float64, nframes)
	for b := 0; b < bins; b++ {
		for f := 0; f < nframes; f++ {
			col[f] = cmplx.Abs(frames[f][b])
		}
		noise[b] = percentileFloat(col, d.cfg.NoisePercentile)
	}

	// Subtract, clamp to spectral floor, restore Hermitian symmetry, iFFT,
	// weighted overlap-add.
	out := make([]float64, len(ext))
	wsum := make([]float64, len(ext))
	for f := 0; f < nframes; f++ {
		buf := frames[f]
		for b := 0; b <= d.nfft/2; b++ {
			mag := cmplx.Abs(buf[b])
			newMag := mag - d.cfg.Oversubtract*noise[b]
			if floor := d.cfg.SpectralFloor * mag; newMag < floor {
				newMag = floor
			}
			scale := 0.0
			if mag > 1e-12 {
				scale = newMag / mag
			}
			buf[b] *= complex(scale, 0)
			if b > 0 && b < d.nfft/2 {
				buf[d.nfft-b] = cmplx.Conj(buf[b])
			}
		}
		IFFT(buf)
		start := f * d.hop
		for i := 0; i < d.nfft; i++ {
			out[start+i] += real(buf[i]) * d.win[i]
			wsum[start+i] += d.win[i] * d.win[i]
		}
	}

	// Crop the padded output back to original length.
	res := make([]float32, n)
	for i := 0; i < n; i++ {
		v := out[pad+i]
		if wsum[pad+i] > 1e-12 {
			v /= wsum[pad+i]
		}
		res[i] = float32(v)
	}
	return res
}

// percentileFloat returns the p-th percentile (0..100) of vals by nearest rank.
// It sorts a copy; vals is not mutated.
func percentileFloat(vals []float64, p float64) float64 {
	if len(vals) == 0 {
		return 0
	}
	s := append([]float64(nil), vals...)
	insertionSortFloat(s)
	idx := int(p / 100 * float64(len(s)-1))
	if idx < 0 {
		idx = 0
	}
	if idx >= len(s) {
		idx = len(s) - 1
	}
	return s[idx]
}

func insertionSortFloat(s []float64) {
	for i := 1; i < len(s); i++ {
		for j := i; j > 0 && s[j] < s[j-1]; j-- {
			s[j], s[j-1] = s[j-1], s[j]
		}
	}
}
