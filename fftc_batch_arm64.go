//go:build arm64

package dsp

import "fmt"

// transformBatch4 runs the batched-four Stockham transform using the portable
// pure-Go stage kernels. Stage 2 swaps individual radices for NEON kernels.
func transformBatch4(f *FFTC, dst, src []complex64, work *fftcBatchWorkspace) {
	need := fftcBatchWidth * f.n
	in := work.a[:need]
	out := work.b[:need]
	packBatch4NEON(in, src, f.n)
	butterflies := 1
	for level, radix := range f.radices {
		sections := f.n / (butterflies * radix)
		tw := f.stockhamTw[level]
		switch radix {
		case 2:
			stockhamRadix2NEON(out, in, tw, butterflies, sections, f.n)
		case 3:
			stockhamRadix3NEON(out, in, tw, butterflies, sections, f.n)
		case 4:
			stockhamRadix4NEON(out, in, tw, butterflies, sections, f.n)
		case 5:
			stockhamRadix5NEON(out, in, tw, butterflies, sections, f.n)
		case 8:
			stockhamRadix8NEON(out, in, tw, butterflies, sections, f.n)
		default:
			panic(fmt.Sprintf("stockham batch: unsupported radix %d", radix))
		}
		in, out = out, in
		butterflies *= radix
	}
	unpackBatch4NEON(dst, in, f.n)
}
