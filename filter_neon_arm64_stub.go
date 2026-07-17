//go:build arm64

package dsp

// complexFIRDotNEON accumulates the multiple-of-four FIR prefix with NEON FMA
// into out (out[0]=Σ winR·taps, out[1]=Σ winI·taps over [0,n)); the Go dispatch
// handles the n%4 scalar tail. Hand-written (avo is x86-only).
//
//go:noescape
func complexFIRDotNEON(taps, winR, winI []float32, n int, out *[2]float32)
