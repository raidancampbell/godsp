package dsp

import "math"

// HammingWindow returns a length-n Hamming window:
//
//	w[k] = 0.54 - 0.46*cos(2π k / (n-1))
func HammingWindow(n int) []float64 {
	w := make([]float64, n)
	if n == 1 {
		w[0] = 1
		return w
	}
	for k := 0; k < n; k++ {
		w[k] = 0.54 - 0.46*math.Cos(2*math.Pi*float64(k)/float64(n-1))
	}
	return w
}

// BlackmanWindow returns a length-n Blackman window (classic coefficients):
//
//	w[k] = 0.42 - 0.5*cos(2π k/(n-1)) + 0.08*cos(4π k/(n-1))
func BlackmanWindow(n int) []float64 {
	w := make([]float64, n)
	if n == 1 {
		w[0] = 1
		return w
	}
	d := float64(n - 1)
	for k := 0; k < n; k++ {
		x := 2 * math.Pi * float64(k) / d
		w[k] = 0.42 - 0.5*math.Cos(x) + 0.08*math.Cos(2*x)
	}
	return w
}

// KaiserWindow returns a length-n Kaiser window with shape parameter beta.
// beta=0 is rectangular; larger beta trades main-lobe width for sidelobe
// suppression.
func KaiserWindow(n int, beta float64) []float64 {
	w := make([]float64, n)
	if n == 1 {
		w[0] = 1
		return w
	}
	d := float64(n - 1)
	denom := besselI0(beta)
	for k := 0; k < n; k++ {
		r := 2*float64(k)/d - 1 // in [-1, 1]
		w[k] = besselI0(beta*math.Sqrt(1-r*r)) / denom
	}
	return w
}

// besselI0 is the zeroth-order modified Bessel function of the first kind,
// evaluated by its power series (converges quickly for the beta range used
// in window design).
func besselI0(x float64) float64 {
	sum := 1.0
	term := 1.0
	xh := x / 2
	for k := 1; k < 50; k++ {
		term *= (xh / float64(k)) * (xh / float64(k))
		sum += term
		if term < 1e-12*sum {
			break
		}
	}
	return sum
}
