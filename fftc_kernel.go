package dsp

func bfly2Kernel(out []complex64, tw []complex64, m int) {
	for k := 0; k < m; k++ {
		t := out[m+k] * tw[k]
		out[m+k] = out[k] - t
		out[k] += t
	}
}

// bfly4Kernel is the radix-4 DIT butterfly.
func bfly4Kernel(out []complex64, tw []complex64, m int) {
	for k := 0; k < m; k++ {
		s0 := out[k+m] * tw[k]
		s1 := out[k+2*m] * tw[m+k]
		s2 := out[k+3*m] * tw[2*m+k]
		d0 := out[k] - s1
		a0 := out[k] + s1
		a1 := s0 + s2
		d1 := s0 - s2
		out[k] = a0 + a1
		out[k+2*m] = a0 - a1
		// Forward transform: ±j rotation of d1 against d0.
		out[k+m] = complex(real(d0)+imag(d1), imag(d0)-real(d1))
		out[k+3*m] = complex(real(d0)-imag(d1), imag(d0)+real(d1))
	}
}

// bfly3Kernel is the radix-3 DIT butterfly. After the per-sub twiddles it is a 3-point
// DFT: the two nontrivial roots exp(∓2πi/3) are conjugates, so the combine
// collapses to one real ×(−½) scale and one ±i rotation by sin(2π/3).
func bfly3Kernel(out []complex64, tw []complex64, m int) {
	const s = 0.8660254037844386 // sin(2π/3) = √3/2
	for k := 0; k < m; k++ {
		t0 := out[k]
		t1 := out[k+m] * tw[k]
		t2 := out[k+2*m] * tw[m+k]
		sum := t1 + t2
		dif := t1 - t2
		out[k] = t0 + sum
		// a = t0 − ½·sum; the other two outputs are a ∓ i·(s·dif).
		a := complex(real(t0)-0.5*real(sum), imag(t0)-0.5*imag(sum))
		w := complex(s*real(dif), s*imag(dif))
		out[k+m] = complex(real(a)+imag(w), imag(a)-real(w))
		out[k+2*m] = complex(real(a)-imag(w), imag(a)+real(w))
	}
}

// bfly5Kernel is the radix-5 DIT butterfly (Singleton form). After the per-sub
// twiddles, the 5-point DFT factors into two real cosine combinations and two
// ±i sine rotations by exploiting the conjugate symmetry of the 5th roots.
func bfly5Kernel(out []complex64, tw []complex64, m int) {
	const (
		c1 = 0.30901699437494745 // cos(2π/5)
		c2 = -0.8090169943749475 // cos(4π/5)
		s1 = 0.9510565162951535  // sin(2π/5)
		s2 = 0.5877852522924731  // sin(4π/5)
	)
	for k := 0; k < m; k++ {
		t0 := out[k]
		t1 := out[k+m] * tw[k]
		t2 := out[k+2*m] * tw[m+k]
		t3 := out[k+3*m] * tw[2*m+k]
		t4 := out[k+4*m] * tw[3*m+k]

		a1 := t1 + t4 // outer pair sum / diff
		d1 := t1 - t4
		a2 := t2 + t3 // inner pair sum / diff
		d2 := t2 - t3

		out[k] = t0 + a1 + a2

		m1 := complex(real(t0)+c1*real(a1)+c2*real(a2), imag(t0)+c1*imag(a1)+c2*imag(a2))
		m2 := complex(real(t0)+c2*real(a1)+c1*real(a2), imag(t0)+c2*imag(a1)+c1*imag(a2))
		r1 := complex(s1*real(d1)+s2*real(d2), s1*imag(d1)+s2*imag(d2))
		r2 := complex(s2*real(d1)-s1*real(d2), s2*imag(d1)-s1*imag(d2))

		out[k+m] = complex(real(m1)+imag(r1), imag(m1)-real(r1))   // m1 − i·r1
		out[k+4*m] = complex(real(m1)-imag(r1), imag(m1)+real(r1)) // m1 + i·r1
		out[k+2*m] = complex(real(m2)+imag(r2), imag(m2)-real(r2)) // m2 − i·r2
		out[k+3*m] = complex(real(m2)-imag(r2), imag(m2)+real(r2)) // m2 + i·r2
	}
}
