//go:build !amd64 && !arm64

package dsp

func complexFIRDot(taps, winR, winI []float32) (float32, float32) {
	return complexFIRDotScalar(taps, winR, winI)
}
