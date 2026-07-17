//go:build arm64

package dsp

func complexFIRDot(taps, winR, winI []float32) (float32, float32) {
	if len(taps) != len(winR) || len(taps) != len(winI) {
		panic("complexFIRDot: taps and windows must have equal lengths")
	}

	nv := len(taps) &^ 3
	if nv == 0 {
		return complexFIRDotScalar(taps, winR, winI)
	}

	var out [2]float32
	complexFIRDotNEON(taps, winR, winI, nv, &out)
	for i := nv; i < len(taps); i++ {
		t := taps[i]
		out[0] += winR[i] * t
		out[1] += winI[i] * t
	}
	return out[0], out[1]
}

func firDotReal(taps, win []float32) float32 {
	if len(taps) != len(win) {
		panic("firDotReal: taps and window must have equal lengths")
	}

	nv := len(taps) &^ 3
	if nv == 0 {
		return firDotRealScalar(taps, win)
	}

	var out float32
	firDotRealNEON(taps, win, nv, &out)
	for i := nv; i < len(taps); i++ {
		out += win[i] * taps[i]
	}
	return out
}

func complexFIRDot4(taps, winR, winI []float32, stride int, out *[8]float32) {
	n := len(taps)
	nv := n &^ 3
	if nv == 0 {
		complexFIRDot4Scalar(taps, winR, winI, stride, out)
		return
	}

	complexFIRDot4NEON(taps, winR, winI, stride, nv, out)
	for l := 0; l < 4; l++ {
		base := l * stride
		for i := nv; i < n; i++ {
			t := taps[i]
			out[2*l] += winR[base+i] * t
			out[2*l+1] += winI[base+i] * t
		}
	}
}
