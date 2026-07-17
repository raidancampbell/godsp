package dsp

import (
	"fmt"
	"math"
	"testing"
)

// dftRef is the O(N²) reference DFT in float64.
func dftRef(in []complex64) []complex128 {
	n := len(in)
	out := make([]complex128, n)
	for k := 0; k < n; k++ {
		var acc complex128
		for t := 0; t < n; t++ {
			ph := -2 * math.Pi * float64(k) * float64(t) / float64(n)
			acc += complex128(in[t]) * complex(math.Cos(ph), math.Sin(ph))
		}
		out[k] = acc
	}
	return out
}

// lcg is a deterministic pseudo-random float32 in [-1, 1).
func lcg(seed *uint32) float32 {
	*seed = *seed*1664525 + 1013904223
	return float32(int32(*seed)) / float32(math.MaxInt32+1)
}

func TestFFTC_MatchesDFT(t *testing.T) {
	sizes := []int{
		2, 3, 4, 5, 6, 8, 10, 12, 15, 16, 20, 25, 30, 48, 60, 80,
		120, 192, 200, 240, 288, 384, 400, 480,
	}
	seed := uint32(12345)
	for _, n := range sizes {
		f, err := NewFFTC(n)
		if err != nil {
			t.Fatalf("NewFFTC(%d): %v", n, err)
		}
		src := make([]complex64, n)
		for i := range src {
			src[i] = complex(lcg(&seed), lcg(&seed))
		}
		dst := make([]complex64, n)
		f.Transform(dst, src)

		want := dftRef(src)
		var maxMag float64
		for _, w := range want {
			if m := cmplxAbs128(w); m > maxMag {
				maxMag = m
			}
		}
		tol := 1e-4 * (maxMag + 1)
		for k := range want {
			if d := cmplxAbs128(complex128(dst[k]) - want[k]); d > tol {
				t.Fatalf("n=%d bin %d: |got-want| = %g > %g (got %v want %v)", n, k, d, tol, dst[k], want[k])
			}
		}

		// Parseval: Σ|x|² == Σ|X|²/n.
		var inP, outP float64
		for i := range src {
			inP += float64(real(src[i]))*float64(real(src[i])) + float64(imag(src[i]))*float64(imag(src[i]))
		}
		for i := range dst {
			outP += float64(real(dst[i]))*float64(real(dst[i])) + float64(imag(dst[i]))*float64(imag(dst[i]))
		}
		outP /= float64(n)
		if math.Abs(inP-outP) > 1e-3*(inP+1) {
			t.Errorf("n=%d Parseval: in %g vs out %g", n, inP, outP)
		}
	}
}

func cmplxAbs128(c complex128) float64 {
	return math.Hypot(real(c), imag(c))
}

func assertComplex64SlicesClose(t testing.TB, got, want []complex64, tol float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("len(got)=%d len(want)=%d", len(got), len(want))
	}
	for i := range got {
		if d := cmplxAbs128(complex128(got[i]) - complex128(want[i])); d > tol {
			t.Fatalf("index %d: |got-want|=%g > %g (got %v want %v)", i, d, tol, got[i], want[i])
		}
	}
}

func maxComplexMagnitude(values []complex64) float64 {
	var max float64
	for _, v := range values {
		if mag := math.Hypot(float64(real(v)), float64(imag(v))); mag > max {
			max = mag
		}
	}
	return max
}

func TestFFTC_MatchesRadix2FFT(t *testing.T) {
	const n = 256
	seed := uint32(99)
	src := make([]complex64, n)
	ref := make([]complex128, n)
	for i := range src {
		src[i] = complex(lcg(&seed), lcg(&seed))
		ref[i] = complex128(src[i])
	}
	FFT(ref) // existing in-place complex128 radix-2
	f, err := NewFFTC(n)
	if err != nil {
		t.Fatal(err)
	}
	dst := make([]complex64, n)
	f.Transform(dst, src)
	for k := range ref {
		if d := cmplxAbs128(complex128(dst[k]) - ref[k]); d > 1e-3 {
			t.Fatalf("bin %d: FFTC %v vs FFT %v (|d|=%g)", k, dst[k], ref[k], d)
		}
	}
}

func TestFFTPlan_Radix8(t *testing.T) {
	cases := []struct {
		n       int
		radices []int
	}{
		{8, []int{8}},
		{16, []int{8, 2}},
		{64, []int{8, 8}},
		{288, []int{8, 4, 3, 3}},
		{384, []int{8, 8, 2, 3}},
		{480, []int{8, 4, 3, 5}},
		{512, []int{8, 8, 8}},
		{600, []int{8, 3, 5, 5}},
		{800, []int{8, 4, 5, 5}},
		{432, []int{8, 2, 3, 3, 3}}, // 2^4·3^3: one 8 leaves a single 2 (documented wash)
	}
	for _, tc := range cases {
		radices, subs, ok := fftPlan(tc.n)
		if !ok {
			t.Fatalf("n=%d: fftPlan not ok", tc.n)
		}
		if len(radices) != len(tc.radices) {
			t.Fatalf("n=%d: radices=%v, want %v", tc.n, radices, tc.radices)
		}
		for i := range radices {
			if radices[i] != tc.radices[i] {
				t.Fatalf("n=%d: radices=%v, want %v", tc.n, radices, tc.radices)
			}
		}
		// subs[i] must equal n / product(radices[0..i]).
		prod := 1
		for i, r := range radices {
			prod *= r
			if subs[i] != tc.n/prod {
				t.Fatalf("n=%d level %d: subs=%d, want %d", tc.n, i, subs[i], tc.n/prod)
			}
		}
	}
}

func TestFFTC_RejectsBadSizes(t *testing.T) {
	for _, n := range []int{-4, 0, 1, 7, 14, 142} {
		if _, err := NewFFTC(n); err == nil {
			t.Errorf("NewFFTC(%d): expected error, got nil", n)
		}
	}
}

func TestFFTC_LevelTwiddlesMatchStridedMasterTable(t *testing.T) {
	for _, n := range []int{192, 240, 288, 384, 480} {
		f, err := NewFFTC(n)
		if err != nil {
			t.Fatal(err)
		}
		stride := 1
		for level, r := range f.radices {
			m := f.subs[level]
			tw := f.levelTw[level]
			if len(tw) != (r-1)*m {
				t.Fatalf("n=%d level=%d: len(levelTw)=%d want %d", n, level, len(tw), (r-1)*m)
			}
			for j := 1; j < r; j++ {
				for k := 0; k < m; k++ {
					got := tw[(j-1)*m+k]
					want := f.tw[(j*k*stride)%n]
					if got != want {
						t.Fatalf("n=%d level=%d j=%d k=%d: got %v want %v", n, level, j, k, got, want)
					}
				}
			}
			stride *= r
		}
	}
}

func TestFFTC_StockhamTwiddles(t *testing.T) {
	for _, n := range []int{120, 192, 240, 288, 384, 400, 480} {
		f, err := NewFFTC(n)
		if err != nil {
			t.Fatal(err)
		}
		butterflies := 1
		for level, r := range f.radices {
			sections := n / (butterflies * r)
			got := f.stockhamTw[level]
			if len(got) != (r-1)*butterflies {
				t.Fatalf("n=%d level=%d len=%d want=%d", n, level, len(got), (r-1)*butterflies)
			}
			for q := 1; q < r; q++ {
				for p := 0; p < butterflies; p++ {
					want := f.tw[(p*q*sections)%n]
					if v := got[(q-1)*butterflies+p]; v != want {
						t.Fatalf("n=%d level=%d q=%d p=%d got=%v want=%v", n, level, q, p, v, want)
					}
				}
			}
			butterflies *= r
		}
	}
}

func TestFFTCBatch4MatchesTransform(t *testing.T) {
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
				scale := float32(lane + 1)
				for i := 0; i < n; i++ {
					sign := float32(1)
					if (i+lane)%2 != 0 {
						sign = -1
					}
					src[lane*n+i] = complex(sign*scale*1e15, -sign*scale*5e14)
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
				want := make([]complex64, len(src))
				for lane := 0; lane < fftcBatchWidth; lane++ {
					lo := lane * n
					f.Transform(want[lo:lo+n], src[lo:lo+n])
				}
				tol := 1e-4 * (maxComplexMagnitude(want) + 1)

				work := newFFTCBatchWorkspace(n)
				scalar := make([]complex64, len(src))
				stockhamBatch4Scalar(scalar, src, f, work)
				assertComplex64SlicesClose(t, scalar, want, tol)

				got := make([]complex64, len(src))
				f.TransformBatch4(got, src, work)
				assertComplex64SlicesClose(t, got, want, tol)

				inPlace := append([]complex64(nil), src...)
				f.TransformBatch4(inPlace, inPlace, work)
				assertComplex64SlicesClose(t, inPlace, want, tol)
			})
		}
	}
}

func TestFFTCBatch4RejectsInvalidBuffers(t *testing.T) {
	f, err := NewFFTC(120)
	if err != nil {
		t.Fatal(err)
	}
	valid := make([]complex64, fftcBatchWidth*f.n)
	work := newFFTCBatchWorkspace(f.n)
	tests := []struct {
		name string
		dst  []complex64
		src  []complex64
		work *fftcBatchWorkspace
	}{
		{name: "short destination", dst: valid[:len(valid)-1], src: valid, work: work},
		{name: "short source", dst: valid, src: valid[:len(valid)-1], work: work},
		{name: "short workspace a", dst: valid, src: valid, work: &fftcBatchWorkspace{a: work.a[:len(work.a)-1], b: work.b}},
		{name: "short workspace b", dst: valid, src: valid, work: &fftcBatchWorkspace{a: work.a, b: work.b[:len(work.b)-1]}},
	}
	backing := make([]complex64, len(valid)+1)
	tests = append(tests, struct {
		name string
		dst  []complex64
		src  []complex64
		work *fftcBatchWorkspace
	}{name: "partial overlap", dst: backing[1:], src: backing[:len(valid)], work: work})

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			defer func() {
				if recover() == nil {
					t.Fatal("TransformBatch4 did not panic")
				}
			}()
			f.TransformBatch4(tc.dst, tc.src, tc.work)
		})
	}
}

func TestFFTCBatch4ZeroAlloc(t *testing.T) {
	f, err := NewFFTC(384)
	if err != nil {
		t.Fatal(err)
	}
	src := make([]complex64, fftcBatchWidth*f.n)
	dst := make([]complex64, len(src))
	work := newFFTCBatchWorkspace(f.n)
	if allocs := testing.AllocsPerRun(100, func() {
		f.TransformBatch4(dst, src, work)
	}); allocs != 0 {
		t.Fatalf("TransformBatch4 allocations = %g, want 0", allocs)
	}
}

func TestFFTCButterflyDispatchMatchesPureGo(t *testing.T) {
	cases := []struct {
		name  string
		radix int
		m     int
	}{
		{name: "bfly3_m1", radix: 3, m: 1},
		{name: "bfly3_m3", radix: 3, m: 3},
		{name: "bfly3_m50", radix: 3, m: 50}, // head (mv=48) + scalar tail [48,50)
		{name: "bfly3_m96", radix: 3, m: 96},
		{name: "bfly4_m1", radix: 4, m: 1},
		{name: "bfly4_m48", radix: 4, m: 48},
		{name: "bfly4_m50", radix: 4, m: 50}, // head (mv=48) + scalar tail [48,50)
		{name: "bfly4_m96", radix: 4, m: 96},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			seed := uint32(9001)
			outA := make([]complex64, tc.radix*tc.m)
			outB := make([]complex64, tc.radix*tc.m)
			for i := range outA {
				outA[i] = complex(lcg(&seed), lcg(&seed))
				outB[i] = outA[i]
			}
			tw := make([]complex64, (tc.radix-1)*tc.m)
			for i := range tw {
				tw[i] = complex(lcg(&seed), lcg(&seed))
			}
			switch tc.radix {
			case 3:
				bfly3Kernel(outA, tw, tc.m)
				bfly3Dispatch(outB, tw, tc.m)
			case 4:
				bfly4Kernel(outA, tw, tc.m)
				bfly4Dispatch(outB, tw, tc.m)
			}
			assertComplex64SlicesClose(t, outB, outA, 1e-5)
		})
	}
}

func BenchmarkFFTCButterflies(b *testing.B) {
	cases := []struct {
		name   string
		radix  int
		m      int
		stride int
	}{
		{name: "bfly4_m48_stride1", radix: 4, m: 48, stride: 1},
		{name: "bfly4_m96_stride1", radix: 4, m: 96, stride: 1},
		{name: "bfly3_m1_stride64", radix: 3, m: 1, stride: 64},
		{name: "bfly3_m3_stride32", radix: 3, m: 3, stride: 32},
		{name: "bfly5_m1_stride96", radix: 5, m: 1, stride: 96},
	}
	for _, tc := range cases {
		f, err := NewFFTC(tc.radix * tc.m * tc.stride)
		if err != nil {
			b.Fatal(err)
		}
		out := make([]complex64, tc.radix*tc.m)
		seed := uint32(77)
		for i := range out {
			out[i] = complex(lcg(&seed), lcg(&seed))
		}
		tw := make([]complex64, (tc.radix-1)*tc.m)
		for j := 1; j < tc.radix; j++ {
			for k := 0; k < tc.m; k++ {
				tw[(j-1)*tc.m+k] = f.tw[(j*k*tc.stride)%len(f.tw)]
			}
		}
		b.Run(tc.name, func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				switch tc.radix {
				case 3:
					bfly3Kernel(out, tw, tc.m)
				case 4:
					bfly4Kernel(out, tw, tc.m)
				case 5:
					bfly5Kernel(out, tw, tc.m)
				}
			}
		})
	}
}

// BenchmarkFFTC measures a forward transform at the production FFT sizes
// spanning the rate ladder {120,192,240,288,384,400,480}.
func BenchmarkFFTC(b *testing.B) {
	for _, n := range []int{120, 192, 240, 288, 384, 400, 480} {
		f, err := NewFFTC(n)
		if err != nil {
			b.Fatal(err)
		}
		seed := uint32(1)
		src := make([]complex64, n)
		for i := range src {
			src[i] = complex(lcg(&seed), lcg(&seed))
		}
		dst := make([]complex64, n)
		b.Run(fmt.Sprintf("n=%d", n), func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				f.Transform(dst, src)
			}
		})
	}
}

func BenchmarkFFTCBatch4(b *testing.B) {
	for _, n := range []int{120, 192, 240, 288, 384, 400, 480, 512, 600, 800} {
		f, err := NewFFTC(n)
		if err != nil {
			b.Fatal(err)
		}
		seed := uint32(1)
		src := make([]complex64, fftcBatchWidth*n)
		for i := range src {
			src[i] = complex(lcg(&seed), lcg(&seed))
		}
		dst := make([]complex64, len(src))
		work := newFFTCBatchWorkspace(n)

		b.Run(fmt.Sprintf("K=%d", n), func(b *testing.B) {
			b.Run("recursive4", func(b *testing.B) {
				b.ReportAllocs()
				b.SetBytes(int64(fftcBatchWidth * n * 8))
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					for lane := 0; lane < fftcBatchWidth; lane++ {
						lo := lane * n
						f.Transform(dst[lo:lo+n], src[lo:lo+n])
					}
				}
			})
			b.Run("batch4", func(b *testing.B) {
				b.ReportAllocs()
				b.SetBytes(int64(fftcBatchWidth * n * 8))
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					f.TransformBatch4(dst, src, work)
				}
			})
		})
	}
}

func BenchmarkTransformBatch4(b *testing.B) {
	for _, n := range []int{288, 384, 480, 512, 600, 800} {
		f, _ := NewFFTC(n)
		work := newFFTCBatchWorkspace(n)
		src := make([]complex64, fftcBatchWidth*n)
		dst := make([]complex64, fftcBatchWidth*n)
		seed := uint32(0xb0000 + uint32(n))
		for i := range src {
			src[i] = complex(lcg(&seed), lcg(&seed))
		}
		b.Run(fmt.Sprintf("n=%d", n), func(b *testing.B) {
			b.SetBytes(int64(fftcBatchWidth * n * 8))
			for i := 0; i < b.N; i++ {
				f.TransformBatch4(dst, src, work)
			}
		})
	}
}
