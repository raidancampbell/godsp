//go:build amd64

package dsp

import (
	"fmt"
	"math"
	"testing"
)

func TestFFTCBatch4AVX2MatchesScalar(t *testing.T) {
	type inputCase struct {
		name string
		fill func([]complex64, int)
	}
	cases := []inputCase{
		{name: "impulse", fill: func(src []complex64, n int) {
			for lane := 0; lane < fftcBatchWidth; lane++ {
				src[lane*n+(lane*17)%n] = complex(float32(lane+1), float32(-lane))
			}
		}},
		{name: "dc", fill: func(src []complex64, n int) {
			for lane := 0; lane < fftcBatchWidth; lane++ {
				for i := 0; i < n; i++ {
					src[lane*n+i] = complex(float32(lane+1)/4, float32(lane-2)/7)
				}
			}
		}},
		{name: "one-bin-tone", fill: func(src []complex64, n int) {
			for lane := 0; lane < fftcBatchWidth; lane++ {
				bin := lane*7 + 1
				for i := 0; i < n; i++ {
					phase := 2 * math.Pi * float64(bin*i) / float64(n)
					src[lane*n+i] = complex(float32(math.Cos(phase)), float32(math.Sin(phase)))
				}
			}
		}},
		{name: "alternating-large-finite", fill: func(src []complex64, n int) {
			for lane := 0; lane < fftcBatchWidth; lane++ {
				for i := 0; i < n; i++ {
					sign := float32(1)
					if (i+lane)&1 != 0 {
						sign = -1
					}
					src[lane*n+i] = complex(sign*float32(lane+1)*1e15, -sign*float32(lane+1)*5e14)
				}
			}
		}},
		{name: "random", fill: func(src []complex64, _ int) {
			seed := uint32(0x5eed1234)
			for i := range src {
				src[i] = complex(lcg(&seed), lcg(&seed))
			}
		}},
	}

	for _, n := range []int{120, 192, 240, 288, 384, 400, 480} {
		f, err := NewFFTC(n)
		if err != nil {
			t.Fatal(err)
		}
		for _, tc := range cases {
			t.Run(fmt.Sprintf("n=%d/%s", n, tc.name), func(t *testing.T) {
				src := make([]complex64, fftcBatchWidth*n)
				tc.fill(src, n)
				work := newFFTCBatchWorkspace(n)
				want := make([]complex64, len(src))
				stockhamBatch4Scalar(want, src, f, work)
				got := make([]complex64, len(src))
				transformBatch4(f, got, src, work)
				tol := 1e-5 * (maxComplexMagnitude(want) + 1)
				assertComplex64SlicesClose(t, got, want, tol)
			})
		}
	}
}

// standaloneBatch4 runs the unfused pack -> FFT stages -> unpack sequence,
// mirroring the non-gated branch of transformBatch4. It is the bit-identity
// oracle for the fused edge kernels: the fusion only relocates the stage-0 load
// source and the last-stage store destination, so a gated plan must produce
// byte-for-byte identical bins.
func standaloneBatch4(f *FFTC, dst, src []complex64, work *fftcBatchWorkspace) {
	need := fftcBatchWidth * f.n
	in := work.a[:need]
	out := work.b[:need]
	packBatch4AVX2(in, src, f.n)
	butterflies := 1
	for level, radix := range f.radices {
		sections := f.n / (butterflies * radix)
		tw := f.stockhamTw[level]
		switch radix {
		case 2:
			stockhamRadix2AVX2(out, in, tw, butterflies, sections, f.n)
		case 3:
			stockhamRadix3AVX2(out, in, tw, butterflies, sections, f.n)
		case 4:
			stockhamRadix4AVX2(out, in, tw, butterflies, sections, f.n)
		case 5:
			stockhamRadix5AVX2(out, in, tw, butterflies, sections, f.n)
		}
		in, out = out, in
		butterflies *= radix
	}
	unpackBatch4AVX2(dst, in, f.n)
}

// TestFFTCBatch4FusedBitIdentical proves the pack/unpack-fused edge kernels
// produce byte-for-byte identical output to the standalone kernel sequence for
// the gated plan shapes (first radix 4, last radix 3): 192=[4,4,4,3],
// 288=[4,4,2,3,3], 384=[4,4,4,2,3]. 240=[4,4,3,5] is a non-gated control that
// must also match (it takes the same standalone path in both).
func TestFFTCBatch4FusedBitIdentical(t *testing.T) {
	sizes := []int{192, 288, 384, 240}
	for _, n := range sizes {
		f, err := NewFFTC(n)
		if err != nil {
			t.Fatal(err)
		}
		gated := f.radices[0] == 4 && f.radices[len(f.radices)-1] == 3
		t.Run(fmt.Sprintf("n=%d/gated=%v", n, gated), func(t *testing.T) {
			seed := uint32(0xf00d0000 + n)
			src := make([]complex64, fftcBatchWidth*n)
			for i := range src {
				src[i] = complex(lcg(&seed), lcg(&seed))
			}
			work := newFFTCBatchWorkspace(n)

			want := make([]complex64, len(src))
			standaloneBatch4(f, want, src, work)

			// Out-of-place through the production entry (fused for gated n).
			got := make([]complex64, len(src))
			transformBatch4(f, got, src, work)
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("out-of-place bin %d: fused %v != standalone %v", i, got[i], want[i])
				}
			}

			// In-place (dst==src) must be bit-identical too: stage 0 consumes all
			// of src into a work buffer before the last stage writes dst.
			inplace := make([]complex64, len(src))
			copy(inplace, src)
			transformBatch4(f, inplace, inplace, work)
			for i := range want {
				if inplace[i] != want[i] {
					t.Fatalf("in-place bin %d: fused %v != standalone %v", i, inplace[i], want[i])
				}
			}
		})
	}
}

func TestStockhamAVX2StagesMatchScalar(t *testing.T) {
	for _, radix := range []int{2, 3, 4, 5} {
		t.Run(fmt.Sprintf("radix=%d", radix), func(t *testing.T) {
			const butterflies = 3
			const sections = 2
			n := radix * butterflies * sections
			src := make([]complex64, fftcBatchWidth*n)
			seed := uint32(0xabc000 + radix)
			for i := range src {
				src[i] = complex(lcg(&seed), lcg(&seed))
			}
			tw := make([]complex64, (radix-1)*butterflies)
			for q := 1; q < radix; q++ {
				for p := 0; p < butterflies; p++ {
					phase := -2 * math.Pi * float64(q*p*sections) / float64(n)
					tw[(q-1)*butterflies+p] = complex(float32(math.Cos(phase)), float32(math.Sin(phase)))
				}
			}

			want := stockhamStageScalarTest(src, tw, radix, butterflies, sections, n)
			got := make([]complex64, len(src))
			switch radix {
			case 2:
				stockhamRadix2AVX2(got, src, tw, butterflies, sections, n)
			case 3:
				stockhamRadix3AVX2(got, src, tw, butterflies, sections, n)
			case 4:
				stockhamRadix4AVX2(got, src, tw, butterflies, sections, n)
			case 5:
				stockhamRadix5AVX2(got, src, tw, butterflies, sections, n)
			}
			assertComplex64SlicesClose(t, got, want, 2e-5)
		})
	}
}

func stockhamStageScalarTest(src, tw []complex64, radix, butterflies, sections, n int) []complex64 {
	dst := make([]complex64, len(src))
	for section := 0; section < sections; section++ {
		for p := 0; p < butterflies; p++ {
			for outQ := 0; outQ < radix; outQ++ {
				for lane := 0; lane < fftcBatchWidth; lane++ {
					var sum complex64
					for inQ := 0; inQ < radix; inQ++ {
						i := section*butterflies + p + inQ*(n/radix)
						v := src[i*fftcBatchWidth+lane]
						if inQ > 0 {
							v *= tw[(inQ-1)*butterflies+p]
						}
						phase := -2 * math.Pi * float64(outQ*inQ) / float64(radix)
						root := complex(float32(math.Cos(phase)), float32(math.Sin(phase)))
						sum += v * root
					}
					o := section*butterflies*radix + p + outQ*butterflies
					dst[o*fftcBatchWidth+lane] = sum
				}
			}
		}
	}
	return dst
}
