//go:build amd64

package dsp

// bfly3Dispatch runs the AVX2 radix-3 kernel over the largest multiple-of-4
// block of k, then finishes the [mv,m) tail in pure Go. The tail must loop on k
// (not call bfly3Kernel on a sub-slice) because radix blocks are strided by m.
func bfly3Dispatch(out []complex64, tw []complex64, m int) {
	mv := m &^ 3
	if mv > 0 {
		bfly3AVX2(out, tw, mv, m)
	}
	const s = 0.8660254037844386
	for k := mv; k < m; k++ {
		t0 := out[k]
		t1 := out[k+m] * tw[k]
		t2 := out[k+2*m] * tw[m+k]
		sum := t1 + t2
		dif := t1 - t2
		out[k] = t0 + sum
		a := complex(real(t0)-0.5*real(sum), imag(t0)-0.5*imag(sum))
		w := complex(s*real(dif), s*imag(dif))
		out[k+m] = complex(real(a)+imag(w), imag(a)-real(w))
		out[k+2*m] = complex(real(a)-imag(w), imag(a)+real(w))
	}
}

// bfly4Dispatch runs the AVX2 radix-4 kernel over the largest multiple-of-4
// block of k, then finishes the [mv,m) tail in pure Go. The tail must loop on k
// (not call bfly4Kernel on a sub-slice) because radix blocks are strided by m.
func bfly4Dispatch(out []complex64, tw []complex64, m int) {
	mv := m &^ 3
	if mv > 0 {
		bfly4AVX2(out, tw, mv, m)
	}
	for k := mv; k < m; k++ {
		s0 := out[k+m] * tw[k]
		s1 := out[k+2*m] * tw[m+k]
		s2 := out[k+3*m] * tw[2*m+k]
		d0 := out[k] - s1
		a0 := out[k] + s1
		a1 := s0 + s2
		d1 := s0 - s2
		out[k] = a0 + a1
		out[k+2*m] = a0 - a1
		out[k+m] = complex(real(d0)+imag(d1), imag(d0)-real(d1))
		out[k+3*m] = complex(real(d0)-imag(d1), imag(d0)+real(d1))
	}
}
