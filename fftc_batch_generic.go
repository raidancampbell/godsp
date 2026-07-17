//go:build !amd64 && !arm64

package dsp

func transformBatch4(f *FFTC, dst, src []complex64, work *fftcBatchWorkspace) {
	// Preserve exact in-place input before Transform starts writing rows.
	copy(work.a, src)
	for lane := 0; lane < fftcBatchWidth; lane++ {
		lo := lane * f.n
		f.Transform(dst[lo:lo+f.n], work.a[lo:lo+f.n])
	}
}
