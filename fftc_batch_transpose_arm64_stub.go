//go:build arm64

package dsp

// packBatch4NEON transposes four length-n row-major complex64 lanes into AoSoA
// layout (dst[i*4+lane] = src[lane*n+i]); the Go-free tail is handled inline.
// Hand-written (avo is x86-only).
//
//go:noescape
func packBatch4NEON(dst, src []complex64, n int)

// unpackBatch4NEON is the inverse transpose (dst[lane*n+i] = src[i*4+lane]).
//
//go:noescape
func unpackBatch4NEON(dst, src []complex64, n int)
