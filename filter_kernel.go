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
