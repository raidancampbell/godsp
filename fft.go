package dsp

import (
	"math"
	"math/cmplx"
)

// FFT performs an in-place radix-2 Cooley-Tukey FFT. len(a) must be a power
// of two. Bin k corresponds to frequency k*Fs/N (or, for k > N/2, the
// negative frequency (k-N)*Fs/N — caller's responsibility to fftshift if
// they want a centered spectrum).
func FFT(a []complex128) {
	n := len(a)
	if n&(n-1) != 0 || n == 0 {
		panic("FFT: length must be a power of two")
	}
	// Bit-reverse permutation.
	for i, j := 1, 0; i < n; i++ {
		bit := n >> 1
		for ; j&bit != 0; bit >>= 1 {
			j ^= bit
		}
		j ^= bit
		if i < j {
			a[i], a[j] = a[j], a[i]
		}
	}
	// Butterflies.
	for size := 2; size <= n; size <<= 1 {
		half := size >> 1
		theta := -2 * math.Pi / float64(size)
		wn := complex(math.Cos(theta), math.Sin(theta))
		for start := 0; start < n; start += size {
			w := complex(1, 0)
			for k := 0; k < half; k++ {
				t := w * a[start+k+half]
				a[start+k+half] = a[start+k] - t
				a[start+k] = a[start+k] + t
				w *= wn
			}
		}
	}
}

// HannWindow returns a length-n Hann (Hanning) window:
//
//	w[k] = 0.5 * (1 - cos(2π k / (n-1)))
func HannWindow(n int) []float64 {
	w := make([]float64, n)
	if n == 1 {
		w[0] = 1
		return w
	}
	for k := 0; k < n; k++ {
		w[k] = 0.5 * (1 - math.Cos(2*math.Pi*float64(k)/float64(n-1)))
	}
	return w
}

// IFFT performs an in-place inverse FFT via the conjugation identity
// ifft(x) = conj(fft(conj(x))) / N. len(a) must be a power of two.
func IFFT(a []complex128) {
	n := len(a)
	for i := range a {
		a[i] = cmplx.Conj(a[i])
	}
	FFT(a)
	inv := complex(1/float64(n), 0)
	for i := range a {
		a[i] = cmplx.Conj(a[i]) * inv
	}
}

