//go:build arm64

package dsp

// atan2BlockNEON computes dst[i] = atan2Approx(y[i], x[i]) * scale for i in
// [0,n), n%4==0, four lanes at a time. The Go dispatch (atan2_arm64.go) runs
// atan2BlockScalar for the n%4 tail. Hand-written (avo is x86-only) — see
// atan2_neon_arm64.s. Uses true FDIV so each lane matches the scalar oracle.
func atan2BlockNEON(dst, y, x []float32, n int, scale float32)
