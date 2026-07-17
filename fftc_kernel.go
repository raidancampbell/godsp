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

// bfly8Kernel is the radix-8 DIT butterfly. After the 7 per-leg twiddles it is
// an 8-point DFT, computed as two radix-4 DFTs (even legs 0,2,4,6 and odd legs
// 1,3,5,7) recombined with the eighth roots. out holds 8 sub-transforms of
// length m at offsets k+j*m (j=0..7); tw holds 7 twiddle rows of length m.
func bfly8Kernel(out []complex64, tw []complex64, m int) {
	const r = 0.7071067811865476 // √2/2 = |W8 real/imag|
	for k := 0; k < m; k++ {
		// Load + per-leg twiddle (leg 0 has no twiddle).
		x0 := out[k]
		x1 := out[k+m] * tw[k]
		x2 := out[k+2*m] * tw[m+k]
		x3 := out[k+3*m] * tw[2*m+k]
		x4 := out[k+4*m] * tw[3*m+k]
		x5 := out[k+5*m] * tw[4*m+k]
		x6 := out[k+6*m] * tw[5*m+k]
		x7 := out[k+7*m] * tw[6*m+k]

		// Radix-4 DFT of the even legs (indices 0,2,4,6).
		e0 := x0 + x4
		e1 := x0 - x4
		e2 := x2 + x6
		e3 := x2 - x6
		// even outputs E0..E3 (radix-4 combine, forward -j on the odd diff):
		E0 := e0 + e2
		E2 := e0 - e2
		E1 := complex(real(e1)+imag(e3), imag(e1)-real(e3)) // e1 - j*e3
		E3 := complex(real(e1)-imag(e3), imag(e1)+real(e3)) // e1 + j*e3

		// Radix-4 DFT of the odd legs (indices 1,3,5,7).
		o0 := x1 + x5
		o1 := x1 - x5
		o2 := x3 + x7
		o3 := x3 - x7
		O0 := o0 + o2
		O2 := o0 - o2
		O1 := complex(real(o1)+imag(o3), imag(o1)-real(o3)) // o1 - j*o3
		O3 := complex(real(o1)-imag(o3), imag(o1)+real(o3)) // o1 + j*o3

		// Recombine with eighth roots W8^j:
		//   W8^0 = 1
		//   W8^1 = r*(1 - i)
		//   W8^2 = -i
		//   W8^3 = r*(-1 - i)
		w1 := complex(r*(real(O1)+imag(O1)), r*(imag(O1)-real(O1)))  // O1 * r(1-i)
		w2 := complex(imag(O2), -real(O2))                            // O2 * (-i)
		w3 := complex(r*(imag(O3)-real(O3)), -r*(real(O3)+imag(O3)))  // O3 * r(-1-i)

		out[k] = E0 + O0
		out[k+m] = E1 + w1
		out[k+2*m] = E2 + w2
		out[k+3*m] = E3 + w3
		out[k+4*m] = E0 - O0
		out[k+5*m] = E1 - w1
		out[k+6*m] = E2 - w2
		out[k+7*m] = E3 - w3
	}
}
