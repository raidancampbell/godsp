//go:build !arm64

package dsp

// atan2Block (amd64 + generic) — the portable polynomial IS the fast path on
// these arches; there is no SIMD kernel here.
func atan2Block(dst, y, x []float32, scale float32) {
	atan2BlockScalar(dst, y, x, scale)
}
