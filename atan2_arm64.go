//go:build arm64

package dsp

// atan2Block (arm64): NEON 4-lane head + scalar poly tail for the n%4 remainder.
func atan2Block(dst, y, x []float32, scale float32) {
	n := len(dst)
	nv := n &^ 3 // largest multiple of 4 <= n
	if nv > 0 {
		atan2BlockNEON(dst[:nv], y[:nv], x[:nv], nv, scale)
	}
	if nv < n {
		atan2BlockScalar(dst[nv:], y[nv:], x[nv:], scale)
	}
}
