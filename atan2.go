package dsp

import "math"

// atan2 approximation: octant reduction + an 8-term Horner odd polynomial.
// Validated (float32) to max abs error 2.70e-07 rad over the unit circle and
// 2.73e-07 rad over r ∈ [0.001, 3] — below float32 epsilon. See
// TestAtan2Approx_Accuracy. Do NOT alter the coefficients without re-validating.
//
// This is the source of truth for the discriminator's atan2 on all arches. On
// arm64 a hand-written NEON kernel (atan2_neon_arm64.s) computes the identical
// arithmetic 4 lanes at a time; atan2BlockScalar is its per-lane oracle and its
// n%4 tail.
var atan2Poly = [8]float32{
	0.0028662257, -0.0161657367, 0.0429096138, -0.0752896400,
	0.1065626393, -0.1420889944, 0.1999355085, -0.3333314528,
}

const (
	atan2PiOver2 = float32(math.Pi / 2)
	atan2Pi      = float32(math.Pi)
)

// atan2Approx approximates math.Atan2(y, x) to ≤3e-7 rad.
func atan2Approx(y, x float32) float32 {
	ax := x
	if ax < 0 {
		ax = -ax
	}
	ay := y
	if ay < 0 {
		ay = -ay
	}
	// num = min(|x|,|y|), den = max(|x|,|y|). ratioSwap is true when |y|>|x|
	// (the octant where the raw poly gives the co-angle, fixed with π/2 − a).
	num, den := ay, ax
	ratioSwap := false
	if ay > ax {
		num, den = ax, ay
		ratioSwap = true
	}
	if den == 0 {
		return 0 // x == y == 0
	}
	t := num / den
	t2 := t * t
	// P(t²) via Horner, then atan(t) = t + t·t²·P(t²).
	p := atan2Poly[0]
	for i := 1; i < 8; i++ {
		p = p*t2 + atan2Poly[i]
	}
	a := t + t*t2*p
	if ratioSwap {
		a = atan2PiOver2 - a
	}
	if x < 0 {
		a = atan2Pi - a
	}
	if y < 0 {
		a = -a
	}
	return a
}

// atan2BlockScalar computes dst[i] = atan2Approx(y[i], x[i]) * scale for all i.
// It is the portable path, the NEON oracle, and the NEON n%4 tail helper.
func atan2BlockScalar(dst, y, x []float32, scale float32) {
	for i := range dst {
		dst[i] = atan2Approx(y[i], x[i]) * scale
	}
}
