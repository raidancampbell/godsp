package dsp

import "unsafe"

const fftcBatchWidth = 4

type fftcBatchWorkspace struct {
	a []complex64 // AoSoA, length 4*n
	b []complex64 // AoSoA, length 4*n
}

func newFFTCBatchWorkspace(n int) *fftcBatchWorkspace {
	return &fftcBatchWorkspace{
		a: make([]complex64, fftcBatchWidth*n),
		b: make([]complex64, fftcBatchWidth*n),
	}
}

func packBatch4(dst, src []complex64, n int) {
	for i := 0; i < n; i++ {
		for lane := 0; lane < fftcBatchWidth; lane++ {
			dst[i*fftcBatchWidth+lane] = src[lane*n+i]
		}
	}
}

func unpackBatch4(dst, src []complex64, n int) {
	for i := 0; i < n; i++ {
		for lane := 0; lane < fftcBatchWidth; lane++ {
			dst[lane*n+i] = src[i*fftcBatchWidth+lane]
		}
	}
}

// TransformBatch4 computes four consecutive row-major transforms of size f.n.
// Exact in-place operation is supported; partial overlap is not.
func (f *FFTC) TransformBatch4(dst, src []complex64, work *fftcBatchWorkspace) {
	need := fftcBatchWidth * f.n
	if len(dst) != need || len(src) != need {
		panic("FFTC.TransformBatch4: slice length != four times planned size")
	}
	if work == nil || len(work.a) < need || len(work.b) < need {
		panic("FFTC.TransformBatch4: workspace too small")
	}
	if &dst[0] != &src[0] && complex64SlicesOverlap(dst, src) {
		panic("FFTC.TransformBatch4: partial overlap")
	}
	transformBatch4(f, dst, src, work)
}

func complex64SlicesOverlap(a, b []complex64) bool {
	const elementSize = unsafe.Sizeof(complex64(0))
	aStart := uintptr(unsafe.Pointer(&a[0]))
	aEnd := aStart + uintptr(len(a))*elementSize
	bStart := uintptr(unsafe.Pointer(&b[0]))
	bEnd := bStart + uintptr(len(b))*elementSize
	return aStart < bEnd && bStart < aEnd
}

// stockhamBatch4Scalar is the allocation-free scalar oracle for the batched
// Stockham autosort transform. Its work buffers use logicalIndex*4+lane AoSoA
// layout so each logical sample holds all four transform lanes contiguously.
func stockhamBatch4Scalar(dst, src []complex64, f *FFTC, work *fftcBatchWorkspace) {
	need := fftcBatchWidth * f.n
	in := work.a[:need]
	out := work.b[:need]
	packBatch4(in, src, f.n)

	butterflies := 1
	for level, r := range f.radices {
		sections := f.n / (butterflies * r)
		var inputs, outputs [8][fftcBatchWidth]complex64
		for s := 0; s < sections; s++ {
			for p := 0; p < butterflies; p++ {
				for q := 0; q < r; q++ {
					inputIndex := s*butterflies + p + q*(f.n/r)
					for lane := 0; lane < fftcBatchWidth; lane++ {
						inputs[q][lane] = in[inputIndex*fftcBatchWidth+lane]
					}
					if q > 0 {
						twiddle := f.stockhamTw[level][(q-1)*butterflies+p]
						for lane := 0; lane < fftcBatchWidth; lane++ {
							inputs[q][lane] *= twiddle
						}
					}
				}

				for outQ := 0; outQ < r; outQ++ {
					for lane := 0; lane < fftcBatchWidth; lane++ {
						var sum complex64
						for inQ := 0; inQ < r; inQ++ {
							root := f.tw[(outQ*inQ*(f.n/r))%f.n]
							sum += inputs[inQ][lane] * root
						}
						outputs[outQ][lane] = sum
					}
				}

				for q := 0; q < r; q++ {
					outputIndex := s*butterflies*r + p + q*butterflies
					for lane := 0; lane < fftcBatchWidth; lane++ {
						out[outputIndex*fftcBatchWidth+lane] = outputs[q][lane]
					}
				}
			}
		}
		in, out = out, in
		butterflies *= r
	}

	unpackBatch4(dst, in, f.n)
}
