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
