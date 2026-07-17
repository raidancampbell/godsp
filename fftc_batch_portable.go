package dsp

// This file holds the architecture-neutral pure-Go batched Stockham stage
// functions. Each computes one autosort (Stockham) FFT stage across the four
// AoSoA lanes. Buffers use logicalIndex*4+lane layout so a logical sample's
// four transform lanes are contiguous. Signatures match the amd64 AVX2 stubs.
//
// Role: these are NOT a runtime path in the shipped configuration. On arm64
// the dispatch (fftc_batch_arm64.go) uses the NEON kernels, and on other
// non-amd64 arches transformBatch4 uses the sequential recursive fallback
// (fftc_batch_generic.go). They serve as (1) the on-arch parity oracle the
// NEON kernels are tested against and (2) scaffolding for any future SIMD
// port. Keep them behaviourally identical to the scalar oracle.
//
// Twiddle layout: tw[(q-1)*butterflies+p] is the twiddle for input leg q
// (q>=1) at butterfly p, matching buildStockhamTwiddles.

func stockhamRadix2Portable(dst, src, tw []complex64, butterflies, sections, n int) {
	half := n / 2
	w := fftcBatchWidth
	for s := 0; s < sections; s++ {
		for p := 0; p < butterflies; p++ {
			twp := tw[p] // (q-1=0)*butterflies + p
			inBase0 := (s*butterflies + p) * w
			inBase1 := (s*butterflies + p + half) * w
			outBase0 := (s*butterflies*2 + p) * w
			outBase1 := (s*butterflies*2 + p + butterflies) * w
			for lane := 0; lane < w; lane++ {
				a := src[inBase0+lane]
				b := src[inBase1+lane] * twp
				dst[outBase0+lane] = a + b
				dst[outBase1+lane] = a - b
			}
		}
	}
}

func stockhamRadix4Portable(dst, src, tw []complex64, butterflies, sections, n int) {
	quarter := n / 4
	w := fftcBatchWidth
	for s := 0; s < sections; s++ {
		for p := 0; p < butterflies; p++ {
			tw1 := tw[0*butterflies+p]
			tw2 := tw[1*butterflies+p]
			tw3 := tw[2*butterflies+p]
			in0 := (s*butterflies + p) * w
			in1 := (s*butterflies + p + quarter) * w
			in2 := (s*butterflies + p + 2*quarter) * w
			in3 := (s*butterflies + p + 3*quarter) * w
			o0 := (s*butterflies*4 + p) * w
			o1 := (s*butterflies*4 + p + butterflies) * w
			o2 := (s*butterflies*4 + p + 2*butterflies) * w
			o3 := (s*butterflies*4 + p + 3*butterflies) * w
			for lane := 0; lane < w; lane++ {
				x0 := src[in0+lane]
				s0 := src[in1+lane] * tw1
				s1 := src[in2+lane] * tw2
				s2 := src[in3+lane] * tw3
				d0 := x0 - s1
				a0 := x0 + s1
				a1 := s0 + s2
				d1 := s0 - s2
				dst[o0+lane] = a0 + a1
				dst[o2+lane] = a0 - a1
				// Forward transform: -j rotation of d1 against d0.
				dst[o1+lane] = complex(real(d0)+imag(d1), imag(d0)-real(d1))
				dst[o3+lane] = complex(real(d0)-imag(d1), imag(d0)+real(d1))
			}
		}
	}
}

func stockhamRadix3Portable(dst, src, tw []complex64, butterflies, sections, n int) {
	const s = 0.8660254037844386 // sin(2π/3)
	third := n / 3
	w := fftcBatchWidth
	for sec := 0; sec < sections; sec++ {
		for p := 0; p < butterflies; p++ {
			tw1 := tw[0*butterflies+p]
			tw2 := tw[1*butterflies+p]
			in0 := (sec*butterflies + p) * w
			in1 := (sec*butterflies + p + third) * w
			in2 := (sec*butterflies + p + 2*third) * w
			o0 := (sec*butterflies*3 + p) * w
			o1 := (sec*butterflies*3 + p + butterflies) * w
			o2 := (sec*butterflies*3 + p + 2*butterflies) * w
			for lane := 0; lane < w; lane++ {
				t0 := src[in0+lane]
				t1 := src[in1+lane] * tw1
				t2 := src[in2+lane] * tw2
				sum := t1 + t2
				dif := t1 - t2
				dst[o0+lane] = t0 + sum
				a := complex(real(t0)-0.5*real(sum), imag(t0)-0.5*imag(sum))
				ww := complex(s*real(dif), s*imag(dif))
				dst[o1+lane] = complex(real(a)+imag(ww), imag(a)-real(ww))
				dst[o2+lane] = complex(real(a)-imag(ww), imag(a)+real(ww))
			}
		}
	}
}

func stockhamRadix5Portable(dst, src, tw []complex64, butterflies, sections, n int) {
	const (
		c1 = 0.30901699437494745
		c2 = -0.8090169943749475
		s1 = 0.9510565162951535
		s2 = 0.5877852522924731
	)
	fifth := n / 5
	w := fftcBatchWidth
	for sec := 0; sec < sections; sec++ {
		for p := 0; p < butterflies; p++ {
			tw1 := tw[0*butterflies+p]
			tw2 := tw[1*butterflies+p]
			tw3 := tw[2*butterflies+p]
			tw4 := tw[3*butterflies+p]
			in0 := (sec*butterflies + p) * w
			in1 := (sec*butterflies + p + fifth) * w
			in2 := (sec*butterflies + p + 2*fifth) * w
			in3 := (sec*butterflies + p + 3*fifth) * w
			in4 := (sec*butterflies + p + 4*fifth) * w
			o0 := (sec*butterflies*5 + p) * w
			o1 := (sec*butterflies*5 + p + butterflies) * w
			o2 := (sec*butterflies*5 + p + 2*butterflies) * w
			o3 := (sec*butterflies*5 + p + 3*butterflies) * w
			o4 := (sec*butterflies*5 + p + 4*butterflies) * w
			for lane := 0; lane < w; lane++ {
				t0 := src[in0+lane]
				t1 := src[in1+lane] * tw1
				t2 := src[in2+lane] * tw2
				t3 := src[in3+lane] * tw3
				t4 := src[in4+lane] * tw4
				a1 := t1 + t4
				d1 := t1 - t4
				a2 := t2 + t3
				d2 := t2 - t3
				dst[o0+lane] = t0 + a1 + a2
				m1 := complex(real(t0)+c1*real(a1)+c2*real(a2), imag(t0)+c1*imag(a1)+c2*imag(a2))
				m2 := complex(real(t0)+c2*real(a1)+c1*real(a2), imag(t0)+c2*imag(a1)+c1*imag(a2))
				r1 := complex(s1*real(d1)+s2*real(d2), s1*imag(d1)+s2*imag(d2))
				r2 := complex(s2*real(d1)-s1*real(d2), s2*imag(d1)-s1*imag(d2))
				dst[o1+lane] = complex(real(m1)+imag(r1), imag(m1)-real(r1))
				dst[o4+lane] = complex(real(m1)-imag(r1), imag(m1)+real(r1))
				dst[o2+lane] = complex(real(m2)+imag(r2), imag(m2)-real(r2))
				dst[o3+lane] = complex(real(m2)-imag(r2), imag(m2)+real(r2))
			}
		}
	}
}

func stockhamRadix8Portable(dst, src, tw []complex64, butterflies, sections, n int) {
	const r = 0.7071067811865476 // √2/2
	eighth := n / 8
	w := fftcBatchWidth
	for sec := 0; sec < sections; sec++ {
		for p := 0; p < butterflies; p++ {
			tw1 := tw[0*butterflies+p]
			tw2 := tw[1*butterflies+p]
			tw3 := tw[2*butterflies+p]
			tw4 := tw[3*butterflies+p]
			tw5 := tw[4*butterflies+p]
			tw6 := tw[5*butterflies+p]
			tw7 := tw[6*butterflies+p]
			ib := sec*butterflies + p
			in0 := ib * w
			in1 := (ib + eighth) * w
			in2 := (ib + 2*eighth) * w
			in3 := (ib + 3*eighth) * w
			in4 := (ib + 4*eighth) * w
			in5 := (ib + 5*eighth) * w
			in6 := (ib + 6*eighth) * w
			in7 := (ib + 7*eighth) * w
			ob := sec*butterflies*8 + p
			o0 := ob * w
			o1 := (ob + butterflies) * w
			o2 := (ob + 2*butterflies) * w
			o3 := (ob + 3*butterflies) * w
			o4 := (ob + 4*butterflies) * w
			o5 := (ob + 5*butterflies) * w
			o6 := (ob + 6*butterflies) * w
			o7 := (ob + 7*butterflies) * w
			for lane := 0; lane < w; lane++ {
				x0 := src[in0+lane]
				x1 := src[in1+lane] * tw1
				x2 := src[in2+lane] * tw2
				x3 := src[in3+lane] * tw3
				x4 := src[in4+lane] * tw4
				x5 := src[in5+lane] * tw5
				x6 := src[in6+lane] * tw6
				x7 := src[in7+lane] * tw7
				// even legs 0,2,4,6
				e0 := x0 + x4
				e1 := x0 - x4
				e2 := x2 + x6
				e3 := x2 - x6
				E0 := e0 + e2
				E2 := e0 - e2
				E1 := complex(real(e1)+imag(e3), imag(e1)-real(e3))
				E3 := complex(real(e1)-imag(e3), imag(e1)+real(e3))
				// odd legs 1,3,5,7
				oo0 := x1 + x5
				oo1 := x1 - x5
				oo2 := x3 + x7
				oo3 := x3 - x7
				O0 := oo0 + oo2
				O2 := oo0 - oo2
				O1 := complex(real(oo1)+imag(oo3), imag(oo1)-real(oo3))
				O3 := complex(real(oo1)-imag(oo3), imag(oo1)+real(oo3))
				// recombine with eighth roots
				w1 := complex(r*(real(O1)+imag(O1)), r*(imag(O1)-real(O1)))
				w2 := complex(imag(O2), -real(O2))
				w3 := complex(r*(imag(O3)-real(O3)), -r*(real(O3)+imag(O3)))
				dst[o0+lane] = E0 + O0
				dst[o1+lane] = E1 + w1
				dst[o2+lane] = E2 + w2
				dst[o3+lane] = E3 + w3
				dst[o4+lane] = E0 - O0
				dst[o5+lane] = E1 - w1
				dst[o6+lane] = E2 - w2
				dst[o7+lane] = E3 - w3
			}
		}
	}
}
