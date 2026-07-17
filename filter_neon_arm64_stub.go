//go:build arm64

package dsp

// complexFIRDotNEON accumulates the multiple-of-four FIR prefix with NEON FMA
// into out (out[0]=Σ winR·taps, out[1]=Σ winI·taps over [0,n)); the Go dispatch
// handles the n%4 scalar tail. Hand-written (avo is x86-only).
//
//go:noescape
func complexFIRDotNEON(taps, winR, winI []float32, n int, out *[2]float32)

// firDotRealNEON accumulates the multiple-of-four real FIR prefix with NEON FMA
// into out (out = Σ win·taps over [0,n)); the Go dispatch handles the n%4 scalar
// tail. Hand-written (avo is x86-only).
//
//go:noescape
func firDotRealNEON(taps, win []float32, n int, out *float32)

// complexFIRDot4NEON accumulates FOUR shared-tap complex FIR prefixes (multiple
// of four) in one tap pass — lane L's window starts stride float32 elements into
// winR/winI, and out holds {r0,i0,r1,i1,r2,i2,r3,i3}. The Go dispatch handles
// the nv%4 scalar tail per lane. Hand-written (avo is x86-only).
//
//go:noescape
func complexFIRDot4NEON(taps, winR, winI []float32, stride int, nv int, out *[8]float32)
