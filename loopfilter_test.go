package dsp

import (
	"math"
	"testing"
)

// TestLoopFilterGainGolden (L1) pins the Rice closed-form gain algebra. It
// recomputes theta_n, D, k1 and k2 independently in float64 from the same
// (bnT, zeta, kd, k0) the constructor receives and asserts the constructor's
// Gains() match to 1e-12. Because the expected values are derived here from the
// documented formula (not read back from the implementation), a transcription
// error in loopfilter.go's algebra -- a dropped factor of 4, a wrong D term, a
// k1/k2 swap -- would move the constructor's output away from this hand form
// and fail. The chosen point (0.01, 0.707, 1, 1) is the carrier/timing default.
func TestLoopFilterGainGolden(t *testing.T) {
	const (
		bnT  = 0.01
		zeta = 0.707
		kd   = 1.0
		k0   = 1.0
	)

	thetaN := bnT / (zeta + 1.0/(4.0*zeta))
	d := 1.0 + 2.0*zeta*thetaN + thetaN*thetaN
	wantK1 := 4.0 * zeta * thetaN / (d * kd * k0)
	wantK2 := 4.0 * thetaN * thetaN / (d * kd * k0)

	lf := NewLoopFilter(bnT, zeta, kd, k0)
	gotK1, gotK2 := lf.Gains()

	const tol = 1e-12
	if math.Abs(gotK1-wantK1) > tol {
		t.Errorf("k1 = %.15g, want %.15g (diff %.3g)", gotK1, wantK1, gotK1-wantK1)
	}
	if math.Abs(gotK2-wantK2) > tol {
		t.Errorf("k2 = %.15g, want %.15g (diff %.3g)", gotK2, wantK2, gotK2-wantK2)
	}
	t.Logf("k1=%.15g k2=%.15g thetaN=%.15g D=%.15g", gotK1, gotK2, thetaN, d)
}

// TestLoopFilterIntegratorRamp (L2) feeds a constant error for n Advance calls
// with no clamp and asserts the integrator ramps to exactly k2*e*n and the
// final return value is exactly k1*e + k2*e*n. This pins the update ORDER
// (integrator stepped before v is formed) and the integral-branch gain: after n
// steps integ must equal k2*e*n, and v on the n-th call must include that fully
// updated integrator. The constant-error case is intentionally the one a
// per-sample-vs-per-symbol bnT bug does NOT hide -- the arithmetic is exact.
func TestLoopFilterIntegratorRamp(t *testing.T) {
	lf := NewLoopFilter(0.02, 0.707, 1, 1)
	k1, k2 := lf.Gains()

	const e = 0.25
	const n = 1000

	var lastV float64
	for i := 0; i < n; i++ {
		lastV = lf.Advance(e)
	}

	wantInteg := k2 * e * float64(n)
	wantV := k1*e + wantInteg

	// The integrator is a straight sum of n identical increments; float64
	// summation of a constant is effectively exact here, so a tight tolerance
	// still catches an off-by-one in the step count or a mis-ordered update.
	const tol = 1e-12
	if math.Abs(lf.Integrator()-wantInteg) > tol {
		t.Errorf("integ = %.15g, want k2*e*n = %.15g", lf.Integrator(), wantInteg)
	}
	if math.Abs(lastV-wantV) > tol {
		t.Errorf("v = %.15g, want k1*e + k2*e*n = %.15g", lastV, wantV)
	}
	t.Logf("integ=%.15g v=%.15g (k1=%.6g k2=%.6g)", lf.Integrator(), lastV, k1, k2)
}

// TestLoopFilterSetClampSaturates (L3) enables an anti-windup clamp then feeds a
// large one-signed error long enough that the unclamped integrator would blow
// well past the limit. It asserts the integrator saturates exactly at +limit
// for positive error and at -limit for negative error, and never exceeds the
// bound on any intermediate step. Without the clamp branch the integrator would
// grow without bound; this proves the guard actually fires and is symmetric.
func TestLoopFilterSetClampSaturates(t *testing.T) {
	const limit = 0.5

	// Positive error saturates at +limit.
	lf := NewLoopFilter(0.05, 0.707, 1, 1)
	lf.SetClamp(limit)
	for i := 0; i < 10000; i++ {
		lf.Advance(1.0)
		if lf.Integrator() > limit+1e-12 {
			t.Fatalf("integ %.15g exceeded +limit %.15g on step %d", lf.Integrator(), limit, i)
		}
	}
	if math.Abs(lf.Integrator()-limit) > 1e-12 {
		t.Errorf("positive saturation: integ = %.15g, want %.15g", lf.Integrator(), limit)
	}

	// Negative error saturates at -limit.
	lf.Reset()
	for i := 0; i < 10000; i++ {
		lf.Advance(-1.0)
		if lf.Integrator() < -limit-1e-12 {
			t.Fatalf("integ %.15g exceeded -limit %.15g on step %d", lf.Integrator(), -limit, i)
		}
	}
	if math.Abs(lf.Integrator()+limit) > 1e-12 {
		t.Errorf("negative saturation: integ = %.15g, want %.15g", lf.Integrator(), -limit)
	}

	// limit <= 0 disables the clamp: the integrator is free to grow past the
	// former bound again.
	lf.Reset()
	lf.SetClamp(0)
	for i := 0; i < 10000; i++ {
		lf.Advance(1.0)
	}
	if lf.Integrator() <= limit {
		t.Errorf("clamp not disabled: integ = %.15g, want > %.15g", lf.Integrator(), limit)
	}
}

// TestLoopFilterReset (L4) drives the integrator away from zero, calls Reset,
// and asserts the integrator returns to zero while the gains are preserved
// (gains are configuration, not carried state). A Reset that also recomputed or
// zeroed the gains -- or that failed to zero the integrator -- would fail here.
func TestLoopFilterReset(t *testing.T) {
	lf := NewLoopFilter(0.03, 0.707, 1, 1)
	wantK1, wantK2 := lf.Gains()

	for i := 0; i < 100; i++ {
		lf.Advance(0.7)
	}
	if lf.Integrator() == 0 {
		t.Fatalf("integrator did not move away from zero before Reset")
	}

	lf.Reset()

	if lf.Integrator() != 0 {
		t.Errorf("after Reset integ = %.15g, want 0", lf.Integrator())
	}
	gotK1, gotK2 := lf.Gains()
	if gotK1 != wantK1 || gotK2 != wantK2 {
		t.Errorf("Reset altered gains: got (%.15g, %.15g), want (%.15g, %.15g)", gotK1, gotK2, wantK1, wantK2)
	}
}

// TestLoopFilterPanics (L5) verifies NewLoopFilter rejects every argument that
// makes the gain algebra undefined or the loop nonsensical: bnT <= 0, zeta <= 0,
// kd == 0, k0 == 0. Each case uses the recover() idiom to assert a panic occurs.
func TestLoopFilterPanics(t *testing.T) {
	cases := []struct {
		name              string
		bnT, zeta, kd, k0 float64
	}{
		{"bnT_zero", 0, 0.707, 1, 1},
		{"bnT_negative", -0.01, 0.707, 1, 1},
		{"zeta_zero", 0.01, 0, 1, 1},
		{"zeta_negative", 0.01, -0.707, 1, 1},
		{"kd_zero", 0.01, 0.707, 0, 1},
		{"k0_zero", 0.01, 0.707, 1, 0},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			defer func() {
				if r := recover(); r == nil {
					t.Errorf("expected panic for %s (bnT=%v zeta=%v kd=%v k0=%v)", c.name, c.bnT, c.zeta, c.kd, c.k0)
				}
			}()
			NewLoopFilter(c.bnT, c.zeta, c.kd, c.k0)
		})
	}
}
