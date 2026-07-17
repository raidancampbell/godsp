//go:build arm64

package dsp

import (
	"fmt"
	"math"
	"testing"
)

// runPortableStage dispatches one portable batched Stockham stage by radix.
func runPortableStage(radix int, dst, src, tw []complex64, butterflies, sections, n int) {
	switch radix {
	case 2:
		stockhamRadix2Portable(dst, src, tw, butterflies, sections, n)
	case 3:
		stockhamRadix3Portable(dst, src, tw, butterflies, sections, n)
	case 4:
		stockhamRadix4Portable(dst, src, tw, butterflies, sections, n)
	case 5:
		stockhamRadix5Portable(dst, src, tw, butterflies, sections, n)
	case 8:
		stockhamRadix8Portable(dst, src, tw, butterflies, sections, n)
	default:
		panic(fmt.Sprintf("no portable stage for radix %d", radix))
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

func TestPortableStagesMatchScalar(t *testing.T) {
	for _, radix := range []int{2, 3, 4, 5, 8} {
		t.Run(fmt.Sprintf("radix=%d", radix), func(t *testing.T) {
			const butterflies = 3
			const sections = 2
			n := radix * butterflies * sections
			src := make([]complex64, fftcBatchWidth*n)
			seed := uint32(0xabc000 + uint32(radix))
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
			runPortableStage(radix, got, src, tw, butterflies, sections, n)
			assertComplex64SlicesClose(t, got, want, 2e-5)
		})
	}
}

func TestStockhamNEONStagesMatchScalar(t *testing.T) {
	for _, radix := range []int{2, 3, 4, 5, 8} {
		t.Run(fmt.Sprintf("radix=%d", radix), func(t *testing.T) {
			const butterflies = 3
			const sections = 2
			n := radix * butterflies * sections
			src := make([]complex64, fftcBatchWidth*n)
			seed := uint32(0xbee000 + uint32(radix))
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
				stockhamRadix2NEON(got, src, tw, butterflies, sections, n)
			case 3:
				stockhamRadix3NEON(got, src, tw, butterflies, sections, n)
			case 4:
				stockhamRadix4NEON(got, src, tw, butterflies, sections, n)
			case 5:
				stockhamRadix5NEON(got, src, tw, butterflies, sections, n)
			case 8:
				stockhamRadix8NEON(got, src, tw, butterflies, sections, n)
			}
			assertComplex64SlicesClose(t, got, want, 2e-5)
		})
	}
}

// TestFFTCBatch4NEONMatchesScalarRandom exercises the arm64 NEON dispatch
// (transformBatch4 routes every planner radix to a NEON kernel) against the
// scalar oracle over a random input at each production size. The broader
// TestFFTCBatch4NEONMatchesScalar covers the same sizes across five input
// patterns; this retains a distinct random seed as a cheap extra sample.
func TestFFTCBatch4NEONMatchesScalarRandom(t *testing.T) {
	for _, n := range []int{120, 192, 240, 288, 384, 400, 480, 512, 600, 800} {
		f, err := NewFFTC(n)
		if err != nil {
			t.Fatal(err)
		}
		t.Run(fmt.Sprintf("n=%d", n), func(t *testing.T) {
			seed := uint32(0x5eed0000 + uint32(n))
			src := make([]complex64, fftcBatchWidth*n)
			for i := range src {
				src[i] = complex(lcg(&seed), lcg(&seed))
			}
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

func TestFFTCBatch4NEONInPlace(t *testing.T) {
	for _, n := range []int{288, 384, 480, 512, 600, 800} {
		f, err := NewFFTC(n)
		if err != nil {
			t.Fatal(err)
		}
		t.Run(fmt.Sprintf("n=%d", n), func(t *testing.T) {
			seed := uint32(0x10000 + uint32(n))
			src := make([]complex64, fftcBatchWidth*n)
			for i := range src {
				src[i] = complex(lcg(&seed), lcg(&seed))
			}
			work := newFFTCBatchWorkspace(n)
			outOfPlace := make([]complex64, len(src))
			transformBatch4(f, outOfPlace, src, work)
			inPlace := make([]complex64, len(src))
			copy(inPlace, src)
			transformBatch4(f, inPlace, inPlace, work)
			tol := 1e-5 * (maxComplexMagnitude(outOfPlace) + 1)
			assertComplex64SlicesClose(t, inPlace, outOfPlace, tol)
		})
	}
}

func TestFFTCBatch4NEONMatchesScalar(t *testing.T) {
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

	for _, n := range []int{120, 192, 240, 288, 384, 400, 480, 512, 600, 800} {
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
