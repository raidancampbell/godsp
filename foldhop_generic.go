//go:build !amd64

package dsp

// foldHopKernel (pure Go) — the reference implementation, used on non-amd64
// builds and as the oracle the float64 reference test validates. See the amd64
// file for the AVX2 path.
func foldHopKernel(k int, win []complex64, proto []float32, fold []complex64) {
	foldHopScalarRange(0, k, win, proto, fold)
}

// foldHop4Kernel (pure Go) folds four overlapping hop-windows via four
// independent scalar folds. win is lane 0's window base; lane L's window
// starts r complex64 later and runs length len(proto). fold holds the four
// contiguous k-length output rows. See the amd64 file for the AVX2 path.
func foldHop4Kernel(k, r int, win []complex64, proto []float32, fold []complex64) {
	n := len(proto)
	for lane := 0; lane < 4; lane++ {
		foldHopScalarRange(0, k, win[lane*r:lane*r+n], proto, fold[lane*k:(lane+1)*k])
	}
}
