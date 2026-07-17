package dsp

// complexFIRDotScalar computes one complex FIR output from SoA real and
// imaginary windows. It is the portable implementation and SIMD oracle.
func complexFIRDotScalar(taps, winR, winI []float32) (float32, float32) {
	n := len(taps)
	_, _ = winR[n-1], winI[n-1]

	var r0, r1, r2, r3 float32
	var im0, im1, im2, im3 float32
	j := 0
	for ; j+3 < n; j += 4 {
		t0, t1, t2, t3 := taps[j], taps[j+1], taps[j+2], taps[j+3]
		r0 += winR[j] * t0
		r1 += winR[j+1] * t1
		r2 += winR[j+2] * t2
		r3 += winR[j+3] * t3
		im0 += winI[j] * t0
		im1 += winI[j+1] * t1
		im2 += winI[j+2] * t2
		im3 += winI[j+3] * t3
	}
	accR := (r0 + r1) + (r2 + r3)
	accI := (im0 + im1) + (im2 + im3)
	for ; j < n; j++ {
		t := taps[j]
		accR += winR[j] * t
		accI += winI[j] * t
	}
	return accR, accI
}

// firDotRealScalar computes one real FIR output from a real window. It is the
// portable implementation and SIMD oracle, using the same 4-way
// ((r0+r1)+(r2+r3)) accumulation shape as complexFIRDotScalar but with a single
// accumulator set (no imaginary component).
func firDotRealScalar(taps, win []float32) float32 {
	n := len(taps)
	_ = win[n-1]

	var r0, r1, r2, r3 float32
	j := 0
	for ; j+3 < n; j += 4 {
		r0 += win[j] * taps[j]
		r1 += win[j+1] * taps[j+1]
		r2 += win[j+2] * taps[j+2]
		r3 += win[j+3] * taps[j+3]
	}
	acc := (r0 + r1) + (r2 + r3)
	for ; j < n; j++ {
		acc += win[j] * taps[j]
	}
	return acc
}

// complexFIRDot4Scalar computes FOUR complex FIR outputs that share the same
// taps, from windows offset 0, stride, 2*stride, 3*stride elements into the SoA
// winR/winI buffers. It is the portable implementation and SIMD oracle for the
// batched kernel: because it is literally four independent complexFIRDotScalar
// calls, each lane is byte-identical to the single-window scalar path. Results
// are written as out = {r0,i0, r1,i1, r2,i2, r3,i3}.
func complexFIRDot4Scalar(taps, winR, winI []float32, stride int, out *[8]float32) {
	n := len(taps)
	for l := 0; l < 4; l++ {
		base := l * stride
		accR, accI := complexFIRDotScalar(taps, winR[base:base+n], winI[base:base+n])
		out[2*l] = accR
		out[2*l+1] = accI
	}
}
