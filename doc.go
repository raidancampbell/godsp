// Package dsp provides original, dependency-free digital-signal-processing
// primitives: FIR/IIR filtering, FM demodulation, FFTs, polyphase
// filterbanks and rational resampling, oscillators, noise-floor and SNR
// estimation, and speech-enhancement building blocks.
//
// The package has no runtime dependencies. SIMD kernels for amd64 are
// generated with avo (see the go:generate directives in pfb.go); the committed
// *_amd64.s and *_amd64_stub.go files are the source of truth, so consumers
// never need to run the generators.
//
// The arm64 NEON kernels (fftc_batch_neon_arm64.s, foldhop_neon_arm64.s,
// filter_neon_arm64.s, fftc_batch_transpose_arm64.s, fftc_bfly_neon_arm64.s,
// atan2_neon_arm64.s) are hand-written, not generated: avo has no arm64
// backend. Go's arm64 assembler exposes only VFMLA/VFMLS as float-vector
// mnemonics, so where vector FADD/FSUB/FMUL/FNEG are needed (the FFT radix and
// bfly kernels) they are emitted as raw WORD encodings; the
// FIR reduce likewise WORD-encodes FADDP. The atan2 discriminator kernel
// WORD-encodes vector FABS/FMAX/FMIN/FDIV/FMUL/FSUB/FNEG/FCMGT/FCMEQ (using
// real VBIT/VEOR/VDUP/VFMLA for the branchless octant blends). The foldHop,
// FIR-accumulate, and batch-transpose kernels are real·complex FMA or pure data
// movement and use only VLD1/VZIP/VTRN/VFMLA/VST1. Edit the .s files directly;
// no generator.
package dsp
