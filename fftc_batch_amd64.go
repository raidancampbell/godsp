//go:build amd64

package dsp

import "fmt"

func transformBatch4(f *FFTC, dst, src []complex64, work *fftcBatchWorkspace) {
	need := fftcBatchWidth * f.n
	in := work.a[:need]
	out := work.b[:need]
	radices := f.radices
	last := len(radices) - 1

	// Fuse the AoSoA transpose into the FFT edge stages when the plan opens with
	// a radix-4 stage and closes with a radix-3 one (the PFB K=384 shape and its
	// neighbours: [4,4,4,2,3], [4,4,4,3], [4,4,2,3,3]). The fused stage-0 kernel
	// gathers row-major src straight into its butterfly (folding packBatch4) and
	// the fused last stage scatters straight to row-major dst (folding
	// unpackBatch4), removing one L1 round-trip through the work buffers per
	// transform. Every arithmetic op is emitted by the same body as the
	// standalone kernels, so the output is bit-identical. In-place (dst==src) is
	// safe: stage 0 reads all of src into a work buffer before the last stage
	// writes dst. Other plan shapes fall back to the standalone pack/FFT/unpack.
	// NOTE: under greedy-radix-8, production sizes (K=288/384/480/800) now open
	// with radix-8, so they take the standalone path and use the standalone pack
	// + radix-8 kernel. Forgoing stage-0 fusion for these plans is deliberate (a
	// fused radix-8 pack kernel is a future optimization).
	if radices[0] != 4 || radices[last] != 3 {
		packBatch4AVX2(in, src, f.n)
		butterflies := 1
		for level, radix := range radices {
			sections := f.n / (butterflies * radix)
			tw := f.stockhamTw[level]
			switch radix {
			case 2:
				stockhamRadix2AVX2(out, in, tw, butterflies, sections, f.n)
			case 3:
				stockhamRadix3AVX2(out, in, tw, butterflies, sections, f.n)
			case 4:
				stockhamRadix4AVX2(out, in, tw, butterflies, sections, f.n)
			case 5:
				stockhamRadix5AVX2(out, in, tw, butterflies, sections, f.n)
			case 8:
				stockhamRadix8AVX2(out, in, tw, butterflies, sections, f.n)
			default:
				panic(fmt.Sprintf("stockham batch: unsupported radix %d", radix))
			}
			in, out = out, in
			butterflies *= radix
		}
		unpackBatch4AVX2(dst, in, f.n)
		return
	}

	butterflies := 1
	for level, radix := range radices {
		sections := f.n / (butterflies * radix)
		tw := f.stockhamTw[level]
		switch {
		case level == 0:
			// radix 4 (gated); gather row-major src, write AoSoA out.
			stockhamRadix4PackAVX2(out, src, tw, butterflies, sections, f.n)
		case level == last:
			// radix 3 (gated); read AoSoA in, scatter row-major dst.
			stockhamRadix3UnpackAVX2(dst, in, tw, butterflies, sections, f.n)
		default:
			switch radix {
			case 2:
				stockhamRadix2AVX2(out, in, tw, butterflies, sections, f.n)
			case 3:
				stockhamRadix3AVX2(out, in, tw, butterflies, sections, f.n)
			case 4:
				stockhamRadix4AVX2(out, in, tw, butterflies, sections, f.n)
			case 5:
				stockhamRadix5AVX2(out, in, tw, butterflies, sections, f.n)
			case 8:
				stockhamRadix8AVX2(out, in, tw, butterflies, sections, f.n)
			default:
				panic(fmt.Sprintf("stockham batch: unsupported radix %d", radix))
			}
		}
		in, out = out, in
		butterflies *= radix
	}
}
