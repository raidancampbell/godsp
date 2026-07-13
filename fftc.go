package dsp

import (
	"fmt"
	"math"
)

// FFTC is a forward complex64 FFT of fixed size n, where n must factor into
// 2, 3, and 5 (pairs of 2s use a radix-4 kernel). Twiddles are precomputed at
// construction; the object is read-only afterwards, so concurrent Transform
// calls are safe provided each caller passes a distinct dst.
type FFTC struct {
	n          int
	radices    []int         // per-level radix, e.g. [4 4 5 5] for 400
	subs       []int         // per-level sub-transform length: subs[i] = n / Π radices[0..i]
	tw         []complex64   // tw[k] = exp(-2πi·k/n)
	levelTw    [][]complex64 // dense per-level twiddles in butterfly access order
	stockhamTw [][]complex64 // dense per-stage twiddles in Stockham access order
}

// NewFFTC plans a forward FFT of size n (n ≥ 2, 2/3/5-smooth).
func NewFFTC(n int) (*FFTC, error) {
	if n < 2 {
		return nil, fmt.Errorf("FFTC: size must be >= 2, got %d", n)
	}
	radices, subs, ok := fftPlan(n)
	if !ok {
		return nil, fmt.Errorf("FFTC: size %d must factor into 2/3/5", n)
	}
	tw := make([]complex64, n)
	for k := range tw {
		ph := -2 * math.Pi * float64(k) / float64(n)
		tw[k] = complex(float32(math.Cos(ph)), float32(math.Sin(ph)))
	}
	levelTw := buildFFTCLevelTwiddles(n, radices, subs, tw)
	stockhamTw := buildStockhamTwiddles(n, radices, tw)
	return &FFTC{n: n, radices: radices, subs: subs, tw: tw, levelTw: levelTw, stockhamTw: stockhamTw}, nil
}

func buildStockhamTwiddles(n int, radices []int, tw []complex64) [][]complex64 {
	out := make([][]complex64, len(radices))
	butterflies := 1
	for level, r := range radices {
		sections := n / (butterflies * r)
		stage := make([]complex64, (r-1)*butterflies)
		for q := 1; q < r; q++ {
			for p := 0; p < butterflies; p++ {
				stage[(q-1)*butterflies+p] = tw[(p*q*sections)%n]
			}
		}
		out[level] = stage
		butterflies *= r
	}
	return out
}

func buildFFTCLevelTwiddles(n int, radices, subs []int, tw []complex64) [][]complex64 {
	levelTw := make([][]complex64, len(radices))
	stride := 1
	for level, r := range radices {
		m := subs[level]
		dense := make([]complex64, (r-1)*m)
		for j := 1; j < r; j++ {
			for k := 0; k < m; k++ {
				dense[(j-1)*m+k] = tw[(j*k*stride)%n]
			}
		}
		levelTw[level] = dense
		stride *= r
	}
	return levelTw
}

// fftPlan factors n into the radix sequence (greedy 4s, then at most one 2,
// then 3s, then 5s) and the sub-length at each level.
func fftPlan(n int) (radices, subs []int, ok bool) {
	rem := n
	take := func(r int) {
		rem /= r
		radices = append(radices, r)
		subs = append(subs, rem)
	}
	for rem%4 == 0 {
		take(4)
	}
	if rem%2 == 0 {
		take(2)
	}
	for rem%3 == 0 {
		take(3)
	}
	for rem%5 == 0 {
		take(5)
	}
	return radices, subs, rem == 1
}

// Transform computes the forward DFT of src into dst. Both must have length n;
// they must not overlap (dst doubles as the recursion work area; src is never
// written). Bin k of dst is the component at +k·Fs/n (k > n/2 wraps negative).
func (f *FFTC) Transform(dst, src []complex64) {
	if len(dst) != f.n || len(src) != f.n {
		panic("FFTC.Transform: slice length != planned size")
	}
	f.work(dst, src, 0, 1)
}

// work runs one decimation-in-time level: gather the radix-r strided
// sub-transforms into contiguous chunks of dst, then combine with twiddled
// butterflies. stride is the cumulative input stride above this level.
func (f *FFTC) work(dst, src []complex64, level, stride int) {
	r := f.radices[level]
	m := f.subs[level]
	if m == 1 {
		for i := 0; i < r; i++ {
			dst[i] = src[i*stride]
		}
	} else {
		for i := 0; i < r; i++ {
			f.work(dst[i*m:(i+1)*m], src[i*stride:], level+1, stride*r)
		}
	}
	switch r {
	case 2:
		f.bfly2(dst, f.levelTw[level], m)
	case 3:
		f.bfly3(dst, f.levelTw[level], m)
	case 4:
		f.bfly4(dst, f.levelTw[level], m)
	case 5:
		f.bfly5(dst, f.levelTw[level], m)
	default:
		panic(fmt.Sprintf("FFTC: unsupported radix %d (planner emits only 2/3/4/5)", r))
	}
}

func (f *FFTC) bfly2(out []complex64, tw []complex64, m int) {
	bfly2Kernel(out, tw, m)
}

func (f *FFTC) bfly4(out []complex64, tw []complex64, m int) {
	bfly4Dispatch(out, tw, m)
}

func (f *FFTC) bfly3(out []complex64, tw []complex64, m int) {
	bfly3Dispatch(out, tw, m)
}

func (f *FFTC) bfly5(out []complex64, tw []complex64, m int) {
	bfly5Kernel(out, tw, m)
}
