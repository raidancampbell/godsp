//go:build amd64

package dsp

// foldHopKernel (amd64) runs the AVX2 8-bin-block head, then the shared scalar
// range helper for the remaining k%8 bins. The vector head covers bins [0,kv);
// foldHopAVX2 reads win/proto with tap stride k and accumulates over p taps per
// bin. Reads only its args (writes only fold), so distinct-win/fold calls are
// concurrency-safe like the generic path.
func foldHopKernel(k int, win []complex64, proto []float32, fold []complex64) {
	p := len(win) / k
	kv := k &^ 7 // largest multiple of 8 <= k
	if kv > 0 {
		foldHopAVX2(win, proto, fold, kv, k, p)
	}
	if kv == k {
		return
	}
	foldHopScalarRange(kv, k, win, proto, fold)
}

// foldHop4Kernel (amd64) folds four overlapping hop-windows in a single proto
// pass via foldHop4AVX2 (each tap's proto load+permute shared across all four
// lanes), then runs the shared scalar range helper per lane for the k%8 tail.
// win is lane 0's window base; lane L's window starts r complex64 later and
// runs length len(proto)=k·p. fold holds the four contiguous k-length output
// rows. The per-lane result is byte-identical to four foldHopKernel calls:
// only the proto load/permute is hoisted, no lane's arithmetic order changes.
// Reads only win/proto and writes only fold, so disjoint hop ranges stay safe.
func foldHop4Kernel(k, r int, win []complex64, proto []float32, fold []complex64) {
	n := len(proto)
	p := n / k
	kv := k &^ 7
	if kv > 0 {
		foldHop4AVX2(win, proto, fold, kv, k, p, r)
	}
	if kv == k {
		return
	}
	for lane := 0; lane < 4; lane++ {
		foldHopScalarRange(kv, k, win[lane*r:lane*r+n], proto, fold[lane*k:(lane+1)*k])
	}
}
