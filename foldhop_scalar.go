package dsp

// foldHopScalarRange computes the portable polyphase alias-sum for output bins
// [start,k): fold[i] = Σ_q win[q·k+i]·proto[q·k+i]. win and proto are length
// k·P; fold[start:k] is overwritten. Reads only its arguments, so concurrent
// calls with distinct win/fold are safe. Shared by the !amd64 kernel (start=0)
// and the amd64 AVX2 dispatch's scalar tail (start=kv).
func foldHopScalarRange(start, k int, win []complex64, proto []float32, fold []complex64) {
	n := len(win)
	for i := start; i < k; i++ {
		var re, im float32
		for si := i; si < n; si += k {
			re += real(win[si]) * proto[si]
			im += imag(win[si]) * proto[si]
		}
		fold[i] = complex(re, im)
	}
}
