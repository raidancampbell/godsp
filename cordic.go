package dsp

import "math"

// CORDIC (COordinate Rotation DIgital Computer) computes cos/sin and
// magnitude/phase using only shifts, adds, and a small table of precomputed
// angle constants -- no multiplies in the inner loop and, critically, no
// runtime trig or a big sin/cos lookup table. Two modes are provided:
//
//   - ROTATION mode (CordicRotate): given an angle, produce (cos, sin). This is
//     a table-free NCO/oscillator alternative. Unlike the phasor recurrence in
//     Oscillator (phasor *= step each sample), every sample here is computed
//     from the absolute angle, so there is no multiplicative magnitude drift to
//     renormalise -- see CordicNCO.
//   - VECTORING mode (CordicMagPhase): given (x, y), produce (magnitude, phase)
//     where phase == atan2(y, x). This is a hypot+atan2 without a sqrt or a
//     divide in the core loop.
//
// Numerical design (documented per the task):
//
//   - cordicN = 24 iterations. The i-th iteration removes an angle of
//     atan(2^-i); the smallest step is atan(2^-23) ~= 1.19e-7 rad, which is at
//     the float32 epsilon floor (2^-23 ~= 1.19e-7). Fewer iterations leave a
//     larger residual angle: N=20 leaves ~atan(2^-19) ~= 1.9e-6 rad. With N=24
//     and the working recurrence carried in float32, the measured worst-case
//     error over [-pi, pi] is well under 1e-5 for both cos/sin and phase (see
//     TestCordic_*). This matches the classic rule of thumb "float32 precision
//     needs ~24 CORDIC iterations".
//
//   - atanTable[i] = atan(2^-i). These are the only precomputed constants; they
//     are computed once at package init in float64 and stored as float32.
//
//   - The CORDIC pseudo-rotations each scale the vector length by
//     sqrt(1 + 2^-2i), so after N iterations the length is multiplied by the
//     aggregate gain A = product(sqrt(1 + 2^-2i)) ~= 1.64676025 (cordicKinv).
//     Rotation mode pre-scales the seed vector by its reciprocal
//     K = product(1/sqrt(1 + 2^-2i)) ~= 0.60725293 (cordicK) so the result comes
//     out unit-length; vectoring mode multiplies the final x by K to recover the
//     true magnitude.
//
// cordicN is the iteration count. See the numerical design note above for why 24.
const cordicN = 24

// cordicPi is math.Pi in float32, used for the pi pre-rotation in the quadrant
// reduction of both modes.
const cordicPi = float32(math.Pi)

var (
	// atanTable[i] = atan(2^-i), the per-iteration elementary rotation angles.
	atanTable [cordicN]float32
	// twoPow[i] = 2^-i, the per-iteration shift factor. A power of two is exact
	// in float32, so multiplying by it is a lossless "shift" that keeps the
	// inner loop free of rounding beyond the add itself.
	twoPow [cordicN]float32
	// cordicK = product(1/sqrt(1 + 2^-2i)) ~= 0.60725293, the gain-compensation
	// factor. Rotation mode seeds x with this so the answer is unit-length;
	// vectoring mode multiplies the final x by it to get true magnitude.
	cordicK float32
	// cordicKinv = 1/cordicK = product(sqrt(1 + 2^-2i)) ~= 1.64676025, the raw
	// CORDIC processing gain. Kept for documentation and callers that want the
	// uncompensated length.
	cordicKinv float32
)

// init precomputes the CORDIC angle table, the shift table, and the aggregate
// gain. The coefficient math is done in float64 for accuracy and stored as
// float32 to match the float32 working type of the recurrence.
func init() {
	k := 1.0
	for i := 0; i < cordicN; i++ {
		p := math.Ldexp(1.0, -i) // 2^-i, computed exactly
		twoPow[i] = float32(p)
		atanTable[i] = float32(math.Atan(p))
		k *= 1.0 / math.Sqrt(1.0+p*p)
	}
	cordicK = float32(k)
	cordicKinv = float32(1.0 / k)
}

// CordicRotate returns cos(angle) and sin(angle) computed by rotation-mode
// CORDIC. It is a table-free oscillator sample: no sin/cos lookup, no runtime
// math.Sin/Cos, just shifts and adds over a 24-entry angle table.
//
// Quadrant handling (the classic CORDIC footgun): rotation mode converges only
// for |angle| <= ~1.7433 rad, so an arbitrary angle must first be brought into
// [-pi/2, pi/2]. We do this in two steps, both in float64 to preserve low bits:
//
//  1. Wrap to [-pi, pi] via a - 2*pi*round(a / (2*pi)).
//  2. If the wrapped angle is still outside [-pi/2, pi/2], pre-rotate by pi
//     (subtract pi from a positive angle, add pi to a negative one). Because
//     cos(a +/- pi) = -cos(a) and sin(a +/- pi) = -sin(a), a pi shift negates
//     BOTH outputs -- so we solve for the reduced angle and flip the sign of the
//     final cos and sin together. Forgetting this flip is the usual source of
//     wrong-quadrant CORDIC output.
func CordicRotate(angle float32) (cos, sin float32) {
	// Reduce in float64 so the 2*pi-multiple subtraction does not lose bits of
	// the (already float32-quantized) input. Note the input arrives quantized to
	// float32 before this function sees it, so for |angle| > 2^24 the low bits
	// are already gone by construction and the reduction is inherently lossy.
	a := float64(angle)
	a -= 2.0 * math.Pi * math.Round(a/(2.0*math.Pi)) // now in [-pi, pi]

	flip := float32(1)
	if a > math.Pi/2 {
		a -= math.Pi
		flip = -1
	} else if a < -math.Pi/2 {
		a += math.Pi
		flip = -1
	}

	// Seed x with the gain-compensation factor so the converged vector is
	// unit-length: (x, y) -> (cos z0, sin z0). y starts at 0, z holds the
	// residual angle still to be rotated away.
	x := cordicK
	y := float32(0)
	z := float32(a)
	for i := 0; i < cordicN; i++ {
		// Both candidate deltas use the pre-update x and y (simultaneous
		// rotation); only the sign differs by the direction decision.
		dx := y * twoPow[i]
		dy := x * twoPow[i]
		if z >= 0 {
			// Rotate by +atan(2^-i) toward z == 0.
			x -= dx
			y += dy
			z -= atanTable[i]
		} else {
			// Rotate by -atan(2^-i).
			x += dx
			y -= dy
			z += atanTable[i]
		}
	}
	return flip * x, flip * y
}

// CordicMagPhase returns the magnitude sqrt(x^2 + y^2) and the phase
// atan2(y, x) of the vector (x, y) using vectoring-mode CORDIC. It computes a
// hypot and an atan2 together without a sqrt or an inner-loop divide.
//
// Quadrant handling: vectoring mode drives y to zero and only converges when
// the vector starts in the right half-plane (x > 0, i.e. |phase| < pi/2). For
// x < 0 we pre-rotate the input by pi (negate x and y), which is invariant for
// magnitude, then correct the phase by +/-pi: atan2(y, x) = atan2(-y, -x) + pi
// when y >= 0 and atan2(-y, -x) - pi when y < 0. The (0, 0) input is special-
// cased to (0, 0) because with a zero vector the direction decision would
// otherwise accumulate the whole angle table into a bogus phase.
func CordicMagPhase(x, y float32) (mag, phase float32) {
	if x == 0 && y == 0 {
		return 0, 0 // matches math.Atan2(0, 0) == 0
	}

	addPhase := float32(0)
	vx, vy := x, y
	if x < 0 {
		// Bring the vector into the right half-plane; magnitude is unchanged.
		vx, vy = -x, -y
		if y >= 0 {
			addPhase = cordicPi
		} else {
			addPhase = -cordicPi
		}
	}

	z := float32(0)
	for i := 0; i < cordicN; i++ {
		dx := vy * twoPow[i]
		dy := vx * twoPow[i]
		if vy < 0 {
			// vy below the x-axis: rotate counter-clockwise (+atan) to lift it
			// toward zero, and subtract that angle from the accumulator.
			vx -= dx
			vy += dy
			z -= atanTable[i]
		} else {
			// vy on or above the x-axis: rotate clockwise (-atan) toward zero.
			vx += dx
			vy -= dy
			z += atanTable[i]
		}
	}
	// After convergence vx == cordicKinv * hypot(x, y); multiply by cordicK to
	// undo the processing gain and recover the true magnitude.
	return vx * cordicK, z + addPhase
}

// CordicNCO is a table-free numerically controlled oscillator: a phase
// accumulator plus rotation-mode CORDIC. Each Step advances the phase by a
// fixed increment and returns cos/sin of the ABSOLUTE accumulated phase.
//
// Unlike the phasor recurrence in Oscillator -- which multiplies a running
// complex phasor by a fixed step each sample and slowly loses (or gains)
// magnitude to float rounding, requiring periodic renormalisation -- a CordicNCO
// cannot drift in magnitude: every sample is CordicRotate of the accumulated
// phase, so its length is always the CORDIC unit length regardless of how many
// steps have elapsed. The phase accumulator is carried in float64 and wrapped
// to [-pi, pi] each step so it stays bounded over arbitrarily long runs.
type CordicNCO struct {
	phase float64 // accumulated phase in radians, wrapped to [-pi, pi]
	inc   float64 // per-sample phase increment in radians
}

// NewCordicNCO creates an oscillator producing cos(2*pi*freq*t) and
// sin(2*pi*freq*t) sampled at sampleRate. A negative freq runs the phase
// backwards. Phase starts at 0, so the first Step returns (1, 0).
func NewCordicNCO(freq, sampleRate float64) *CordicNCO {
	// Reject non-positive sampleRate to match the codebase convention
	// (NewGoertzel/DesignLPF panic on bad args) rather than silently degrading
	// to a DC oscillator (inc=0) or accepting a negative rate that runs time
	// backwards.
	if sampleRate <= 0 {
		panic("NewCordicNCO: sampleRate must be > 0")
	}
	inc := 2.0 * math.Pi * freq / sampleRate
	// Pre-wrap the increment into [-pi, pi] so accumulation stays well-behaved
	// even for offsets above Nyquist.
	inc -= 2.0 * math.Pi * math.Round(inc/(2.0*math.Pi))
	return &CordicNCO{inc: inc}
}

// Step returns cos/sin of the current phase, then advances the phase by one
// increment (wrapped to [-pi, pi]). The returned pair is always CORDIC
// unit-length, so magnitude never drifts no matter how many Steps are taken.
func (n *CordicNCO) Step() (cos, sin float32) {
	cos, sin = CordicRotate(float32(n.phase))
	n.phase += n.inc
	// Keep the accumulator bounded; wrapping in float64 loses no precision that
	// matters for the subsequent float32 rotation.
	if n.phase > math.Pi {
		n.phase -= 2.0 * math.Pi
	} else if n.phase < -math.Pi {
		n.phase += 2.0 * math.Pi
	}
	return cos, sin
}

// Reset returns the phase accumulator to zero without changing the frequency.
func (n *CordicNCO) Reset() {
	n.phase = 0
}
