//go:build !amd64 && !arm64

package dsp

func complexFIRDot(taps, winR, winI []float32) (float32, float32) {
	return complexFIRDotScalar(taps, winR, winI)
}

func firDotReal(taps, win []float32) float32 {
	return firDotRealScalar(taps, win)
}

func complexFIRDot4(taps, winR, winI []float32, stride int, out *[8]float32) {
	complexFIRDot4Scalar(taps, winR, winI, stride, out)
}
