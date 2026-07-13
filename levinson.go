package dsp

// LevinsonDurbin computes LPC coefficients from an autocorrelation sequence
// using the Levinson-Durbin recursion. Returns coefficients a[0..order] where
// a[0]=1.0, and the prediction error energy E.
//
// The convention is A(z) = 1 + a[1]*z^-1 + a[2]*z^-2 + ... (note: a[k] will
// be negative at formant frequencies for typical speech signals).
func LevinsonDurbin(R []float64, order int) (a []float64, E float64) {
	a = make([]float64, order+1)
	a[0] = 1.0
	E = R[0]
	if E <= 0 {
		return a, 0
	}

	for i := 1; i <= order; i++ {
		// Compute reflection coefficient
		sum := 0.0
		for j := 1; j < i; j++ {
			sum += a[j] * R[i-j]
		}
		k := -(R[i] + sum) / E

		// Update coefficients
		aPrev := make([]float64, i+1)
		copy(aPrev, a[:i+1])
		for j := 1; j < i; j++ {
			a[j] = aPrev[j] + k*aPrev[i-j]
		}
		a[i] = k

		E *= (1 - k*k)
		if E <= 0 {
			break
		}
	}

	return a, E
}

// Autocorrelation computes the autocorrelation R[0..order] of a signal.
func Autocorrelation(x []float32, order int) []float64 {
	R := make([]float64, order+1)
	n := len(x)
	for k := 0; k <= order; k++ {
		for i := k; i < n; i++ {
			R[k] += float64(x[i]) * float64(x[i-k])
		}
	}
	return R
}

// ExpandBandwidth applies bandwidth expansion to LPC coefficients:
// a_expanded[k] = a[k] * gamma^k
func ExpandBandwidth(a []float64, gamma float64) []float64 {
	expanded := make([]float64, len(a))
	expanded[0] = 1.0
	g := gamma
	for i := 1; i < len(a); i++ {
		expanded[i] = a[i] * g
		g *= gamma
	}
	return expanded
}

