//go:build arm64

package dsp

// foldHopNEON is the NEON 4-bin-block polyphase fold; the Go dispatch handles
// the k%4 scalar tail. Hand-written (avo is x86-only) — see foldhop_neon_arm64.s.
func foldHopNEON(win []complex64, proto []float32, fold []complex64, kv int, k int, p int)

// foldHop4NEON folds four overlapping hop-windows in one proto pass (4-bin
// blocks); the Go dispatch handles the k%4 scalar tail per lane. Hand-written.
func foldHop4NEON(win []complex64, proto []float32, fold []complex64, kv int, k int, p int, r int)
