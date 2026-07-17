//go:build arm64

package dsp

// foldHopKernel (arm64) runs the NEON 4-bin-block head, then the shared scalar
// range helper for the remaining k%4 bins. Reads only its args (writes only
// fold), so distinct-win/fold calls are concurrency-safe like the generic path.
func foldHopKernel(k int, win []complex64, proto []float32, fold []complex64) {
	p := len(win) / k
	kv := k &^ 3 // largest multiple of 4 <= k
	if kv > 0 {
		foldHopNEON(win, proto, fold, kv, k, p)
	}
	if kv == k {
		return
	}
	foldHopScalarRange(kv, k, win, proto, fold)
}

// foldHop4Kernel (arm64) folds four overlapping hop-windows in one proto pass
// via foldHop4NEON (each tap's proto load+zip shared across all four lanes),
// then runs the shared scalar range helper per lane for the k%4 tail. win is
// lane 0's window base; lane L's window starts r complex64 later and runs
// length len(proto)=k·p. fold holds the four contiguous k-length output rows.
// The per-lane result is byte-identical to four foldHopKernel calls: only the
// proto load/zip is hoisted, no lane's arithmetic order changes.
func foldHop4Kernel(k, r int, win []complex64, proto []float32, fold []complex64) {
	n := len(proto)
	p := n / k
	kv := k &^ 3
	if kv > 0 {
		foldHop4NEON(win, proto, fold, kv, k, p, r)
	}
	if kv == k {
		return
	}
	for lane := 0; lane < 4; lane++ {
		foldHopScalarRange(kv, k, win[lane*r:lane*r+n], proto, fold[lane*k:(lane+1)*k])
	}
}
