//go:build arm64

package dsp

// These kernels are hand-written Plan9 arm64 assembly in fftc_batch_neon_arm64.s.
// They are NOT generated (avo is x86-only); see doc.go. Signatures match the
// amd64 AVX2 stubs so the dispatch in fftc_batch_arm64.go can swap radices
// between the portable and NEON implementations freely.

func stockhamRadix2NEON(dst, src, tw []complex64, butterflies, sections, n int)

func stockhamRadix3NEON(dst, src, tw []complex64, butterflies, sections, n int)

func stockhamRadix4NEON(dst, src, tw []complex64, butterflies, sections, n int)

func stockhamRadix5NEON(dst, src, tw []complex64, butterflies, sections, n int)

func stockhamRadix8NEON(dst, src, tw []complex64, butterflies, sections, n int)
