package dsp

// LoopFilter is a type-2 proportional-plus-integral (PI) loop filter, the
// linear controller at the heart of both a phase-locked loop (carrier recovery)
// and a symbol-timing-recovery loop. It converts a per-sample (or per-symbol)
// error estimate e into a correction v that a downstream numerically-controlled
// oscillator or interpolator counter integrates into its phase / rate. Because
// the filter itself contains one integrator, the closed loop is order two: it
// drives the steady-state error to zero for a CONSTANT input offset (a fixed
// frequency or clock-rate error), not merely a constant phase.
//
// The control law, evaluated once per Advance call, is:
//
//	integ <- integ + k2*e      (the integral branch; the running frequency/rate estimate)
//	v      = k1*e + integ      (proportional branch plus the updated integrator)
//
// k1 is the proportional gain and k2 the integral gain. The integrator alone
// IS the loop's estimate of the constant offset (residual carrier frequency in
// rad/sample, or clock-rate deviation), which is why it is exposed via
// Integrator / SetIntegrator: coarse acquisition (an FFT or an FLL) can seed it
// through SetIntegrator so the fine loop starts already near lock.
//
// Gains are computed once from a small-signal design (Rice, "Digital
// Communications: A Discrete-Time Approach", the same closed form GNU Radio's
// control_loop uses):
//
//	theta_n = bnT / (zeta + 1/(4*zeta))
//	D       = 1 + 2*zeta*theta_n + theta_n*theta_n
//	k1      = 4*zeta*theta_n     / (D * kd * k0)
//	k2      = 4*theta_n*theta_n  / (D * kd * k0)
//
// where:
//   - bnT is the loop bandwidth times the update period, i.e. the NORMALIZED
//     one-sided noise bandwidth Bn*T. THE UNITS ARE THE CALLER'S: LoopFilter is
//     deliberately unit-agnostic. A carrier loop updating once per sample passes
//     bnT = Bn/fs (per-sample); a symbol-timing loop updating once per symbol
//     passes bnT = Bn/symbolRate (per-symbol). Passing a per-sample bnT to a
//     per-symbol loop (or vice versa) mistunes the loop by exactly the
//     samples-per-symbol factor -- an easy and silent error, guarded downstream
//     by settling-window tests rather than here.
//   - zeta is the damping ratio. Callers use zeta = 0.707 (= 1/sqrt(2)), which
//     gives the maximally-flat (Butterworth) closed-loop response: the fastest
//     acquisition with no peaking / overshoot in the error response. Smaller
//     zeta acquires faster but rings; larger zeta is sluggish. This is the
//     classic acquisition-speed-versus-jitter tradeoff.
//   - kd is the phase/timing detector gain (slope of the S-curve through the
//     origin, in error-units per radian or per sample) and k0 is the
//     oscillator/interpolator gain. Folding both into k1/k2 makes the loop's
//     small-signal open-loop gain equal to the intended design regardless of
//     detector scaling. Costas and SymbolSync normalize their detectors to
//     kd = k0 = 1 and let an upstream AGC hold kd roughly constant, so they pass
//     kd = k0 = 1 and rely purely on bnT and zeta.
//
// The gains are computed in float64 on purpose: they are tiny (k2 is
// O(bnT^2), often ~1e-4 or smaller), and the integrator accumulates them across
// potentially millions of samples, so the coefficient algebra and the running
// sum must not lose precision to float32 rounding. The surrounding DSP works in
// complex64 / float32, but every loop accumulator in this package is float64
// for exactly this reason.
//
// clamp is an optional anti-windup bound on the integrator magnitude. During
// acquisition, or when the true offset exceeds the loop's pull-in range, e can
// stay one-signed for a long run and drive the integrator arbitrarily large;
// clamping |integ| <= clamp bounds the maximum trackable offset and prevents a
// runaway that would need an equally long run of opposite-sign error to unwind.
// clamp == 0 disables the bound (unbounded integrator).
//
// Easy-to-get-wrong, stated precisely: Advance updates the integrator BEFORE
// forming the return value, so v already reflects the current sample's integral
// step (v = k1*e + integ_new, not k1*e + integ_old). The clamp is applied to
// the integrator immediately after its update and therefore also bounds the
// integrator term inside the very v it is returning. A constant error e held for
// n calls (with no clamping) makes integ ramp to exactly k2*e*n and returns
// v = k1*e + k2*e*n on the n-th call.
//
// LoopFilter carries exactly one piece of mutable state, the integrator.
// Reset zeroes it and preserves the configured gains and clamp. It is not safe
// for concurrent use.
type LoopFilter struct {
	k1, k2 float64 // proportional, integral gains (derived once at construction)
	integ  float64 // the integrator == the frequency/rate estimate; ONLY carried state
	clamp  float64 // |integ| anti-windup limit; 0 means unbounded
}

// NewLoopFilter builds a type-2 PI loop filter from the normalized loop
// bandwidth-time product bnT, damping ratio zeta, detector gain kd, and
// oscillator gain k0, using the Rice closed form documented on LoopFilter. The
// anti-windup clamp starts disabled; call SetClamp to enable it.
//
// It panics on arguments that make the gain algebra undefined or the loop
// nonsensical: bnT <= 0 (no bandwidth), zeta <= 0 (no damping; also a divide by
// the zeta term), and kd == 0 or k0 == 0 (division by zero in k1/k2). Negative
// kd/k0 are permitted -- they simply fold a known detector/oscillator sign
// inversion into the gains -- but the callers in this package pass kd = k0 = 1.
func NewLoopFilter(bnT, zeta, kd, k0 float64) *LoopFilter {
	if bnT <= 0 {
		panic("NewLoopFilter: bnT must be positive")
	}
	if zeta <= 0 {
		panic("NewLoopFilter: zeta must be positive")
	}
	if kd == 0 {
		panic("NewLoopFilter: kd must be non-zero")
	}
	if k0 == 0 {
		panic("NewLoopFilter: k0 must be non-zero")
	}

	thetaN := bnT / (zeta + 1.0/(4.0*zeta))
	d := 1.0 + 2.0*zeta*thetaN + thetaN*thetaN
	k1 := 4.0 * zeta * thetaN / (d * kd * k0)
	k2 := 4.0 * thetaN * thetaN / (d * kd * k0)

	return &LoopFilter{k1: k1, k2: k2}
}

// Advance runs one PI update on the error estimate e and returns the loop
// correction v. It first integrates (integ += k2*e), applies the anti-windup
// clamp if one is configured, then returns v = k1*e + integ. See the
// LoopFilter doc for the exact ordering guarantee.
func (l *LoopFilter) Advance(e float64) float64 {
	l.integ += l.k2 * e
	if l.clamp > 0 {
		if l.integ > l.clamp {
			l.integ = l.clamp
		} else if l.integ < -l.clamp {
			l.integ = -l.clamp
		}
	}
	return l.k1*e + l.integ
}

// Integrator returns the current integrator value, which is the loop's estimate
// of the constant offset it is tracking (residual carrier frequency in
// rad/sample for a carrier loop, or clock-rate deviation for a timing loop).
func (l *LoopFilter) Integrator() float64 {
	return l.integ
}

// SetIntegrator seeds the integrator, typically to hand off a coarse-acquisition
// estimate (from an FFT or FLL) so the fine loop starts near lock. The value is
// NOT clamped here; the next Advance applies the clamp if one is configured.
func (l *LoopFilter) SetIntegrator(v float64) {
	l.integ = v
}

// SetClamp sets the anti-windup bound on the integrator magnitude: subsequent
// Advance calls hold |integ| <= limit. A limit <= 0 disables clamping (sets the
// bound to 0, meaning unbounded). Note this does not immediately re-clamp the
// current integrator; the bound takes effect on the next Advance.
func (l *LoopFilter) SetClamp(limit float64) {
	if limit <= 0 {
		l.clamp = 0
		return
	}
	l.clamp = limit
}

// Gains returns the derived proportional gain k1 and integral gain k2.
func (l *LoopFilter) Gains() (k1, k2 float64) {
	return l.k1, l.k2
}

// Reset zeroes the integrator, returning the loop to its just-constructed state.
// The gains and the anti-windup clamp are configuration, not carried state, and
// are preserved.
func (l *LoopFilter) Reset() {
	l.integ = 0
}
