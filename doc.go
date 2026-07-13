// Package dsp provides original, dependency-free digital-signal-processing
// primitives: FIR/IIR filtering, FM demodulation, FFTs, polyphase
// filterbanks and rational resampling, oscillators, noise-floor and SNR
// estimation, and speech-enhancement building blocks.
//
// The package has no runtime dependencies. SIMD kernels for amd64 are
// generated with avo (see the go:generate directives in pfb.go); the committed
// *_amd64.s and *_amd64_stub.go files are the source of truth, so consumers
// never need to run the generators.
package dsp
