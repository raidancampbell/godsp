//go:build !amd64 && !arm64

package dsp

func bfly3Dispatch(out []complex64, tw []complex64, m int) {
	bfly3Kernel(out, tw, m)
}

func bfly4Dispatch(out []complex64, tw []complex64, m int) {
	bfly4Kernel(out, tw, m)
}
