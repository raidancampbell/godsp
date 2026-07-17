//go:build arm64

package dsp

// bfly3NEON computes radix-3 FFTC butterflies for k in [0,mv) (mv even, 2
// complex64/iter); the Go dispatch handles the m&1 scalar tail. Hand-written.
//
//go:noescape
func bfly3NEON(out, tw []complex64, mv, m int)

// bfly4NEON computes radix-4 FFTC butterflies for k in [0,mv) (mv even, 2
// complex64/iter); the Go dispatch handles the m&1 scalar tail. Hand-written.
//
//go:noescape
func bfly4NEON(out, tw []complex64, mv, m int)
