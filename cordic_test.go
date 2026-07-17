package dsp

import (
	"math"
	"testing"
)

// cordicRotTol is the accepted worst-case cos/sin error for CordicRotate at
// cordicN == 24 iterations. The task allows ~1e-3 for N=20; with N=24 the
// measured worst case over [-pi, pi] (and beyond, to exercise range reduction)
// is well under 1e-5, so we assert a much tighter 5e-5 bound. The test logs the
// actual maximum so a regression in iteration count or the gain constant shows
// up immediately.
const cordicRotTol = 5e-5

// TestCordicRotate_VsMath sweeps a fine grid over [-4*pi, 4*pi] -- deliberately
// beyond [-pi, pi] so the 2*pi wrap and the pi pre-rotation (all four quadrants)
// are exercised -- and compares cos/sin against math.Cos/math.Sin.
func TestCordicRotate_VsMath(t *testing.T) {
	const steps = 200000
	const lo, hi = -4.0 * math.Pi, 4.0 * math.Pi

	var maxCos, maxSin float64
	for i := 0; i <= steps; i++ {
		a := lo + (hi-lo)*float64(i)/float64(steps)
		cos, sin := CordicRotate(float32(a))
		ec := math.Abs(float64(cos) - math.Cos(a))
		es := math.Abs(float64(sin) - math.Sin(a))
		if ec > maxCos {
			maxCos = ec
		}
		if es > maxSin {
			maxSin = es
		}
	}
	t.Logf("CordicRotate over [-4pi,4pi]: max cos err = %.3e, max sin err = %.3e (N=%d)", maxCos, maxSin, cordicN)
	if maxCos > cordicRotTol || maxSin > cordicRotTol {
		t.Errorf("CordicRotate error exceeds tol %.1e: cos=%.3e sin=%.3e", cordicRotTol, maxCos, maxSin)
	}
}

// TestCordicRotate_Edges checks the exact-boundary angles that stress the
// quadrant reduction: 0, +/-pi/2 (the convergence-range boundary) and +/-pi
// (which wrap to the opposite quadrant and require the cos/sin sign flip).
func TestCordicRotate_Edges(t *testing.T) {
	cases := []struct {
		name  string
		angle float32
	}{
		{"zero", 0},
		{"+pi/2", float32(math.Pi / 2)},
		{"-pi/2", float32(-math.Pi / 2)},
		{"+pi", float32(math.Pi)},
		{"-pi", float32(-math.Pi)},
	}
	for _, tc := range cases {
		cos, sin := CordicRotate(tc.angle)
		wantCos := math.Cos(float64(tc.angle))
		wantSin := math.Sin(float64(tc.angle))
		if math.Abs(float64(cos)-wantCos) > cordicRotTol || math.Abs(float64(sin)-wantSin) > cordicRotTol {
			t.Errorf("%s: got (cos=%.6f, sin=%.6f), want (cos=%.6f, sin=%.6f)",
				tc.name, cos, sin, wantCos, wantSin)
		}
	}
}

// TestCordicRotate_UnitMagnitude confirms the gain-compensation constant
// (cordicK) yields unit-length output: sqrt(cos^2 + sin^2) ~= 1 for every angle.
func TestCordicRotate_UnitMagnitude(t *testing.T) {
	const steps = 100000
	var maxDev float64
	for i := 0; i <= steps; i++ {
		a := -math.Pi + 2.0*math.Pi*float64(i)/float64(steps)
		cos, sin := CordicRotate(float32(a))
		mag := math.Sqrt(float64(cos)*float64(cos) + float64(sin)*float64(sin))
		if d := math.Abs(mag - 1.0); d > maxDev {
			maxDev = d
		}
	}
	t.Logf("CordicRotate unit-magnitude: max |mag-1| = %.3e", maxDev)
	if maxDev > cordicRotTol {
		t.Errorf("CordicRotate magnitude deviates by %.3e, want <= %.1e", maxDev, cordicRotTol)
	}
}

// cordicVecTol is the accepted worst-case error for CordicMagPhase: relative
// error on magnitude and absolute error (radians) on phase, both at N=24.
const cordicVecTol = 5e-5

// TestCordicMagPhase_VsMath compares magnitude and phase against math.Hypot and
// math.Atan2 across all four quadrants (both signs of x and y) plus a range of
// magnitudes, verifying the x<0 phase correction (+/-pi) and the gain-corrected
// hypot.
func TestCordicMagPhase_VsMath(t *testing.T) {
	// A grid of angles across all four quadrants and several radii. Radii vary
	// widely to confirm the magnitude scales linearly (CORDIC gain is
	// magnitude-independent) and phase is radius-independent.
	radii := []float64{0.01, 0.5, 1.0, 7.5, 1000.0}
	const angSteps = 720 // half-degree resolution over the full circle

	var maxRelMag, maxPhase float64
	for _, r := range radii {
		for k := 0; k < angSteps; k++ {
			ang := -math.Pi + 2.0*math.Pi*float64(k)/float64(angSteps)
			x := float32(r * math.Cos(ang))
			y := float32(r * math.Sin(ang))

			mag, phase := CordicMagPhase(x, y)
			wantMag := math.Hypot(float64(x), float64(y))
			wantPhase := math.Atan2(float64(y), float64(x))

			if wantMag > 0 {
				if e := math.Abs(float64(mag)-wantMag) / wantMag; e > maxRelMag {
					maxRelMag = e
				}
			}
			// The seam-unwrap below folds a 2*pi error down to ~0, so on its own
			// it would PASS a quadrant-correction sign flip (which produces a
			// +/-2*pi phase error and pushes the raw phase to ~+/-4.07, outside
			// atan2's range). Guard that by also asserting the RAW returned phase
			// lies within atan2's documented [-pi, pi] range; the legitimate
			// -pi-vs-+pi seam stays inside this bound and is still tolerated.
			if float64(phase) < -math.Pi-cordicVecTol || float64(phase) > math.Pi+cordicVecTol {
				t.Errorf("r=%.3g ang=%.6f: phase %.6f outside atan2 range [-pi, pi]", r, ang, phase)
			}
			// Phase error, unwrapped across the +/-pi seam so a result of -pi
			// vs +pi is not counted as a 2*pi miss.
			pe := math.Abs(float64(phase) - wantPhase)
			if pe > math.Pi {
				pe = 2.0*math.Pi - pe
			}
			if pe > maxPhase {
				maxPhase = pe
			}
		}
	}
	t.Logf("CordicMagPhase: max rel mag err = %.3e, max phase err = %.3e rad (N=%d)", maxRelMag, maxPhase, cordicN)
	if maxRelMag > cordicVecTol {
		t.Errorf("magnitude relative error %.3e exceeds tol %.1e", maxRelMag, cordicVecTol)
	}
	if maxPhase > cordicVecTol {
		t.Errorf("phase error %.3e rad exceeds tol %.1e", maxPhase, cordicVecTol)
	}
}

// TestCordicMagPhase_Edges covers the axis-aligned and zero inputs where a
// naive vectoring loop misbehaves: x or y == 0 on each axis, and the (0,0)
// vector which must return (0, 0) rather than an accumulated garbage phase.
func TestCordicMagPhase_Edges(t *testing.T) {
	cases := []struct {
		name string
		x, y float32
	}{
		{"origin", 0, 0},
		{"+x axis", 3, 0},
		{"-x axis", -3, 0},
		{"+y axis", 0, 4},
		{"-y axis", 0, -4},
		{"Q1", 3, 4},
		{"Q2", -3, 4},
		{"Q3", -3, -4},
		{"Q4", 3, -4},
	}
	for _, tc := range cases {
		mag, phase := CordicMagPhase(tc.x, tc.y)
		wantMag := math.Hypot(float64(tc.x), float64(tc.y))
		wantPhase := math.Atan2(float64(tc.y), float64(tc.x))

		magErr := math.Abs(float64(mag) - wantMag)
		if wantMag > 0 {
			magErr /= wantMag
		}
		// Assert the RAW phase is within atan2's [-pi, pi] range before the
		// seam-unwrap; a quadrant-correction sign flip escapes to ~+/-4.07 and is
		// caught here even though the unwrap below would fold it back to ~0.
		if float64(phase) < -math.Pi-cordicVecTol || float64(phase) > math.Pi+cordicVecTol {
			t.Errorf("%s: phase %.6f outside atan2 range [-pi, pi]", tc.name, phase)
		}
		pe := math.Abs(float64(phase) - wantPhase)
		if pe > math.Pi {
			pe = 2.0*math.Pi - pe
		}
		if magErr > cordicVecTol || pe > cordicVecTol {
			t.Errorf("%s: got (mag=%.6f, phase=%.6f), want (mag=%.6f, phase=%.6f); magRelErr=%.3e phaseErr=%.3e",
				tc.name, mag, phase, wantMag, wantPhase, magErr, pe)
		}
	}
}

// TestCordicNCO_Correctness verifies the NCO tracks cos/sin of the accumulated
// phase for the first few thousand samples, before phase-accumulator rounding
// has meaningfully accumulated.
func TestCordicNCO_Correctness(t *testing.T) {
	const freq, fs = 1000.0, 48000.0
	const n = 4000
	nco := NewCordicNCO(freq, fs)
	inc := 2.0 * math.Pi * freq / fs

	var maxErr float64
	for i := 0; i < n; i++ {
		cos, sin := nco.Step()
		ph := inc * float64(i)
		ec := math.Abs(float64(cos) - math.Cos(ph))
		es := math.Abs(float64(sin) - math.Sin(ph))
		if ec > maxErr {
			maxErr = ec
		}
		if es > maxErr {
			maxErr = es
		}
	}
	t.Logf("CordicNCO correctness over %d samples: max err = %.3e", n, maxErr)
	if maxErr > 1e-4 {
		t.Errorf("CordicNCO cos/sin error %.3e exceeds 1e-4", maxErr)
	}

	// Reset must return the phase to zero: the first post-reset sample is (1,0).
	nco.Reset()
	cos, sin := nco.Step()
	if math.Abs(float64(cos)-1) > cordicRotTol || math.Abs(float64(sin)) > cordicRotTol {
		t.Errorf("after Reset, first sample = (%.6f, %.6f), want (1, 0)", cos, sin)
	}
}

// TestCordicNCO_NoDrift is the headline property: over a very long run the
// per-sample magnitude stays ~1 and, crucially, does NOT grow with sample
// index. This is the contrast with the phasor-recurrence Oscillator, whose
// running phasor loses/gains magnitude to float rounding and must be
// renormalised. Here each sample is computed from the absolute accumulated
// phase, so the late-run magnitude spread is no larger than the early-run one.
func TestCordicNCO_NoDrift(t *testing.T) {
	const freq, fs = 1234.5, 48000.0
	const n = 1_000_000
	nco := NewCordicNCO(freq, fs)

	var maxEarly, maxLate, maxAll float64
	for i := 0; i < n; i++ {
		cos, sin := nco.Step()
		dev := math.Abs(math.Sqrt(float64(cos)*float64(cos)+float64(sin)*float64(sin)) - 1.0)
		if dev > maxAll {
			maxAll = dev
		}
		if i < 1000 {
			if dev > maxEarly {
				maxEarly = dev
			}
		}
		if i >= n-1000 {
			if dev > maxLate {
				maxLate = dev
			}
		}
	}
	t.Logf("CordicNCO no-drift over %d samples: max |mag-1| all=%.3e early=%.3e late=%.3e",
		n, maxAll, maxEarly, maxLate)

	if maxAll > cordicRotTol {
		t.Errorf("magnitude deviates by %.3e over the run, want <= %.1e", maxAll, cordicRotTol)
	}
	// No drift: the worst late-run deviation must not exceed the worst
	// early-run deviation (allowing a tiny slack for grid alignment). A drifting
	// recurrence would fail this even while passing the absolute bound early on.
	if maxLate > maxEarly+1e-6 {
		t.Errorf("magnitude drifted: late max %.3e > early max %.3e", maxLate, maxEarly)
	}
}

// TestCordicNCO_BadSampleRate asserts NewCordicNCO panics on a non-positive
// sampleRate rather than silently degrading to a DC oscillator (inc=0) or
// running time backwards, matching the panic-on-bad-args convention used by
// NewGoertzel/DesignLPF elsewhere in the package.
func TestCordicNCO_BadSampleRate(t *testing.T) {
	cases := []struct {
		name       string
		sampleRate float64
	}{
		{"zero", 0},
		{"negative", -48000},
	}
	for _, tc := range cases {
		func() {
			defer func() {
				if r := recover(); r == nil {
					t.Errorf("%s: NewCordicNCO(1000, %g) did not panic", tc.name, tc.sampleRate)
				}
			}()
			NewCordicNCO(1000, tc.sampleRate)
		}()
	}
}

// TestCordicConstants sanity-checks the precomputed gain constants against their
// documented values, so a bad edit to the init loop is caught directly rather
// than only via downstream accuracy failures.
func TestCordicConstants(t *testing.T) {
	if math.Abs(float64(cordicK)-0.60725293) > 1e-6 {
		t.Errorf("cordicK = %.8f, want ~0.60725293", cordicK)
	}
	if math.Abs(float64(cordicKinv)-1.64676025) > 1e-5 {
		t.Errorf("cordicKinv = %.8f, want ~1.64676025", cordicKinv)
	}
	if math.Abs(float64(cordicK*cordicKinv)-1.0) > 1e-6 {
		t.Errorf("cordicK * cordicKinv = %.8f, want ~1.0", cordicK*cordicKinv)
	}
	// atanTable[0] = atan(1) = pi/4.
	if math.Abs(float64(atanTable[0])-math.Pi/4) > 1e-6 {
		t.Errorf("atanTable[0] = %.8f, want pi/4 = %.8f", atanTable[0], math.Pi/4)
	}
}
