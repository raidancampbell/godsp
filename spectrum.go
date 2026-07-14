package dsp

import "math"

// WelchPSD estimates the power spectral density of samples via Welch's method:
// overlapping windowed segments, averaged periodograms. The result has length
// nfft, in linear power, and is fftshifted so index nfft/2 is DC. overlap is a
// fraction in [0,1); window must have length nfft (nil → rectangular).
func WelchPSD(samples []complex64, nfft int, overlap float64, window []float64) []float64 {
	if nfft <= 0 {
		return nil
	}
	if window != nil && len(window) != nfft {
		// A wrong-length window would index out of bounds or silently apply
		// the wrong coefficients; reject rather than corrupt the estimate.
		return nil
	}
	if overlap < 0 || overlap >= 1 {
		overlap = 0
	}
	step := int(float64(nfft) * (1 - overlap))
	if step < 1 {
		step = 1
	}
	var winPow float64
	if window == nil {
		winPow = float64(nfft)
	} else {
		for _, w := range window {
			winPow += w * w
		}
	}

	acc := make([]float64, nfft)
	buf := make([]complex128, nfft)
	nSeg := 0
	for start := 0; start+nfft <= len(samples); start += step {
		for i := 0; i < nfft; i++ {
			s := samples[start+i]
			if window != nil {
				buf[i] = complex(float64(real(s))*window[i], float64(imag(s))*window[i])
			} else {
				buf[i] = complex(float64(real(s)), float64(imag(s)))
			}
		}
		FFT(buf)
		for i := 0; i < nfft; i++ {
			re, im := real(buf[i]), imag(buf[i])
			acc[i] += (re*re + im*im) / winPow
		}
		nSeg++
	}
	if nSeg == 0 {
		// Fewer than one full segment: single zero-padded segment.
		for i := 0; i < nfft && i < len(samples); i++ {
			s := samples[i]
			if window != nil {
				buf[i] = complex(float64(real(s))*window[i], float64(imag(s))*window[i])
			} else {
				buf[i] = complex(float64(real(s)), float64(imag(s)))
			}
		}
		for i := len(samples); i < nfft; i++ {
			buf[i] = 0
		}
		FFT(buf)
		for i := 0; i < nfft; i++ {
			re, im := real(buf[i]), imag(buf[i])
			acc[i] = (re*re + im*im) / winPow
		}
		nSeg = 1
	}
	for i := range acc {
		acc[i] /= float64(nSeg)
	}
	return fftshift(acc)
}

// fftshift moves the zero-frequency bin to the center (index n/2).
func fftshift(a []float64) []float64 {
	n := len(a)
	out := make([]float64, n)
	half := (n + 1) / 2
	copy(out[n-half:], a[:half])
	copy(out[:n-half], a[half:])
	return out
}

// SpectralFlatness returns the Wiener entropy of a linear-power PSD:
// geometric mean / arithmetic mean of the bins, in [0,1]. Near 1 means
// noise-like (flat); near 0 means tonal/structured. Non-positive bins are
// floored to a tiny epsilon so the log is finite.
func SpectralFlatness(psd []float64) float64 {
	if len(psd) == 0 {
		return 0
	}
	const eps = 1e-20
	var logSum, sum float64
	for _, p := range psd {
		if p < eps {
			p = eps
		}
		logSum += math.Log(p)
		sum += p
	}
	n := float64(len(psd))
	geo := math.Exp(logSum / n)
	arith := sum / n
	if arith <= 0 {
		return 0
	}
	return geo / arith
}

// OccupiedBandwidth returns the width (in Hz) of the smallest central band
// around the peak bin that contains `fraction` of the total PSD power.
// binHz is the frequency spacing per bin.
func OccupiedBandwidth(psd []float64, binHz, fraction float64) float64 {
	n := len(psd)
	if n == 0 {
		return 0
	}
	var total float64
	peak := 0
	for i, p := range psd {
		total += p
		if p > psd[peak] {
			peak = i
		}
	}
	if total <= 0 {
		return 0
	}
	target := total * fraction
	acc := psd[peak]
	lo, hi := peak, peak
	for acc < target && (lo > 0 || hi < n-1) {
		// Grow toward whichever neighbor has more power.
		var left, right float64
		if lo > 0 {
			left = psd[lo-1]
		}
		if hi < n-1 {
			right = psd[hi+1]
		}
		if lo > 0 && (left >= right || hi >= n-1) {
			lo--
			acc += left
		} else if hi < n-1 {
			hi++
			acc += right
		}
	}
	return float64(hi-lo+1) * binHz
}
