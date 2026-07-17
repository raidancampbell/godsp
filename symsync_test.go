package dsp

import (
	"math"
	"math/cmplx"
	"testing"
)

// ===========================================================================
// Oracle signal machinery (INDEPENDENT of the DUT's interpolator).
//
// The single most important property of this test file: the timing-error
// injector -- the thing that produces receiver samples at a controlled timing
// offset / clock rate -- must NOT be the same interpolator the device under test
// uses. If it were, a bug in the DUT's Farrow kernel would cancel against the
// same bug in the injector and the S-curve / acquisition tests would pass while
// the primitive is broken. So the injector is a Blackman-windowed sinc evaluated
// over a densely-oversampled (denseSps = 32) raised-cosine reference. At that
// oversampling the RC signal is band-limited to ~2% of the dense Nyquist, so the
// sinc oracle is orders of magnitude more accurate than the DUT's 4-tap Farrow
// running at ~4 samples/symbol -- a DUT interpolation error cannot hide behind it.
// ===========================================================================

const (
	symDenseSps = 32   // dense reference oversampling (symbol periods)
	symRolloff  = 0.35 // raised-cosine excess bandwidth
	symSpan     = 8    // pulse span in symbols each side
	symSincL    = 16   // sinc-oracle half-width in dense samples
)

// symSinc is the normalized sinc, sinc(0) = 1.
func symSinc(x float64) float64 {
	if x == 0 {
		return 1
	}
	px := math.Pi * x
	return math.Sin(px) / px
}

// symDesignRC builds a raised-cosine pulse (NOT root-raised) normalized to unit
// peak h(0) = 1 and exact zeros at nonzero integer symbol offsets, so convolving
// an impulse train with it gives a zero-ISI waveform whose value at each symbol
// center is exactly that symbol. Using the full RC (matched-filter output),
// rather than an RRC, lets the timing tests operate on an already-matched-filtered
// signal -- the loop's actual input in a real chain.
func symDesignRC(denseSps int, beta float64, span int) []float32 {
	n := 2*span*denseSps + 1
	mid := (n - 1) / 2
	taps := make([]float32, n)
	for i := range taps {
		t := float64(i-mid) / float64(denseSps) // symbol periods
		var h float64
		switch {
		case t == 0:
			h = 1
		case beta > 0 && math.Abs(math.Abs(2*beta*t)-1) < 1e-9:
			h = (math.Pi / 4) * symSinc(1.0/(2*beta))
		default:
			h = symSinc(t) * math.Cos(math.Pi*beta*t) / (1 - (2*beta*t)*(2*beta*t))
		}
		taps[i] = float32(h)
	}
	return taps
}

// symDenseRC pulse-shapes symbols into a dense complex reference at denseSps
// samples/symbol. It returns the dense stream and mid, the dense index of symbol
// 0's center (symbol i center is at i*denseSps + mid). Zero-ISI: dense[i*denseSps
// + mid] == symbols[i].
func symDenseRC(symbols []complex64, denseSps int, beta float64, span int) (dense []complex64, mid int) {
	taps := symDesignRC(denseSps, beta, span)
	mid = (len(taps) - 1) / 2
	up := make([]complex64, len(symbols)*denseSps)
	for i, s := range symbols {
		up[i*denseSps] = s
	}
	dense = make([]complex64, len(up)+len(taps)-1)
	for n := range up {
		s := up[n]
		if s == 0 {
			continue
		}
		sr, si := real(s), imag(s)
		for j, tp := range taps {
			d := &dense[n+j]
			*d = complex(real(*d)+sr*tp, imag(*d)+si*tp)
		}
	}
	return dense, mid
}

// symSincInterp evaluates dense at fractional index pos using a Blackman-windowed
// sinc of half-width symSincL. This is the ORACLE interpolator: deliberately a
// different, higher-order kernel than the DUT's Farrow (see the file header).
func symSincInterp(dense []complex64, pos float64) complex64 {
	i0 := int(math.Floor(pos))
	var accR, accI float64
	for k := i0 - symSincL + 1; k <= i0+symSincL; k++ {
		if k < 0 || k >= len(dense) {
			continue
		}
		d := pos - float64(k)
		// Blackman window over |d| <= symSincL.
		var w float64
		if math.Abs(d) <= float64(symSincL) {
			x := math.Pi * d / float64(symSincL)
			w = 0.42 + 0.5*math.Cos(x) + 0.08*math.Cos(2*x)
		}
		s := symSinc(d) * w
		accR += float64(real(dense[k])) * s
		accI += float64(imag(dense[k])) * s
	}
	return complex(float32(accR), float32(accI))
}

// symSymbols returns n unit-modulus symbols for the given modulation drawn from
// the seeded LCG (shared lcg from fftc_test.go): BPSK +/-1, QPSK on the
// 45-degree grid.
func symSymbols(n int, mod Modulation, seed *uint32) []complex64 {
	out := make([]complex64, n)
	for i := 0; i < n; i++ {
		if mod == ModQPSK {
			r := symSign(lcg(seed))
			im := symSign(lcg(seed))
			// Normalize to unit modulus (45-degree grid).
			out[i] = complex(r/float32(math.Sqrt2), im/float32(math.Sqrt2))
		} else {
			out[i] = complex(symSign(lcg(seed)), 0)
		}
	}
	return out
}

// symAltBPSK returns n strictly alternating +1/-1 symbols -- the maximum-transition
// pattern that produces the cleanest Gardner S-curve.
func symAltBPSK(n int) []complex64 {
	out := make([]complex64, n)
	for i := range out {
		if i%2 == 0 {
			out[i] = complex(1, 0)
		} else {
			out[i] = complex(-1, 0)
		}
	}
	return out
}

// symReceiver resamples the dense reference into a receiver stream at ~spsRx
// samples/symbol, starting at dense position startPos and stepping by
// (denseSps/spsRx)*clkRatio dense samples per output sample. clkRatio > 1 spaces
// the samples wider (fewer true samples/symbol => negative TimingPPM). It stops
// leaving a symSincL margin at the end so the oracle never reads past the array.
func symReceiver(dense []complex64, startPos, spsRx, clkRatio float64) []complex64 {
	step := (float64(symDenseSps) / spsRx) * clkRatio
	var out []complex64
	limit := float64(len(dense) - symSincL - 1)
	for pos := startPos; pos < limit; pos += step {
		out = append(out, symSincInterp(dense, pos))
	}
	return out
}

// symBestLagEVM aligns got against want over an integer lag in [-maxLag, maxLag],
// skipping the first `settle` want-symbols, and returns the lag and normalized
// RMS EVM at the best alignment. The lag sweep is TWO-SIDED because the recovered
// stream generally LEADS the source: the loop's 4-tap warm-up and basepoint
// indexing mean recovered symbol k corresponds to source symbol k + (a small
// constant), so the alignment lag is negative (got[i+lag] with lag < 0 compares an
// EARLIER recovered symbol to a later source symbol). A one-sided [0,maxLag] sweep
// silently misses this and reports the sqrt(2) uncorrelated noise floor even when
// the loop is perfectly locked. Timing recovery introduces no phase rotation, so
// no constellation de-rotation is needed here (unit-modulus symbols => the EVM
// denominator is 1).
func symBestLagEVM(got, want []complex64, settle, maxLag int) (bestLag int, bestEVM float64) {
	bestEVM = math.Inf(1)
	for lag := -maxLag; lag <= maxLag; lag++ {
		var sumSq float64
		cnt := 0
		for i := settle; i < len(want); i++ {
			j := i + lag
			if j < 0 || j >= len(got) {
				continue
			}
			d := complex128(got[j]) - complex128(want[i])
			sumSq += real(d)*real(d) + imag(d)*imag(d)
			cnt++
		}
		if cnt < 16 {
			continue
		}
		evm := math.Sqrt(sumSq / float64(cnt))
		if evm < bestEVM {
			bestEVM = evm
			bestLag = lag
		}
	}
	return bestLag, bestEVM
}

var symAllInterps = []Interpolator{InterpLinear, InterpFarrow}

// ===========================================================================
// S1: linear interpolator endpoints. out(0) == x0, out(1) == x1 exactly.
// ===========================================================================
func TestSymLinearEndpoints(t *testing.T) {
	x0 := complex64(complex(0.3, -0.7))
	x1 := complex64(complex(-0.2, 0.9))
	// mu = 0 is bit-exact (r0 + 0*(r1-r0)); the loop only ever asks for mu in
	// [0,1) (mu = eta/w with eta < w), so mu = 1 is not a reachable input -- we
	// require it only to a float32 rounding tolerance.
	if got := linearInterp(x0, x1, 0); got != x0 {
		t.Errorf("linear mu=0: got %v want %v", got, x0)
	}
	if got := linearInterp(x0, x1, 1); cmplxAbs64(got-x1) > 1e-6 {
		t.Errorf("linear mu~1: got %v want %v", got, x1)
	}
}

// ===========================================================================
// S2: linear interpolator reproduces a straight line exactly. On samples lying
// on a complex ramp x[k] = a + b*k, the interpolant at any mu equals a + b*(k+mu).
// ===========================================================================
func TestSymLinearRampExact(t *testing.T) {
	a := complex128(complex(1.0, -2.0))
	b := complex128(complex(0.5, 0.25))
	x0 := complex64(a)
	x1 := complex64(a + b)
	for _, mu := range []float32{0.1, 0.25, 0.5, 0.75, 0.9} {
		want := complex64(a + b*complex(float64(mu), 0))
		got := linearInterp(x0, x1, mu)
		if cmplxAbs64(got-want) > 1e-6 {
			t.Errorf("linear ramp mu=%.2f: got %v want %v", mu, got, want)
		}
	}
}

// ===========================================================================
// S3: Farrow parabolic endpoints + LINEAR-exactness. NOTE: the alpha=0.5
// piecewise-parabolic Farrow does NOT reproduce an arbitrary quadratic (that is a
// cubic-Lagrange property); it reproduces up to a linear ramp exactly. This test
// pins the two real, provable facts: out(0)=x0, out(1)=x1 (endpoint interpolation)
// and exact reproduction of a linear ramp. A separate golden test (S4) pins the
// coefficients, and a negative control (verification gate) confirms teeth.
// ===========================================================================
func TestSymFarrowEndpointsAndLinear(t *testing.T) {
	// Endpoints: for any four taps, mu=0 -> x0 (=xm1's neighbor, the basepoint)
	// and mu=1 -> x1.
	xm1 := complex64(complex(0.2, 1.1))
	x0 := complex64(complex(-0.5, 0.3))
	x1 := complex64(complex(0.8, -0.4))
	x2 := complex64(complex(-0.1, -0.9))
	if got := farrowParabolic(xm1, x0, x1, x2, 0); cmplxAbs64(got-x0) > 1e-6 {
		t.Errorf("farrow mu=0: got %v want x0=%v", got, x0)
	}
	if got := farrowParabolic(xm1, x0, x1, x2, 1); cmplxAbs64(got-x1) > 1e-6 {
		t.Errorf("farrow mu=1: got %v want x1=%v", got, x1)
	}

	// Linear ramp x[k] = a + b*k sampled at k = -1,0,1,2 -> exact at any mu.
	a := complex128(complex(2.0, -1.0))
	b := complex128(complex(-0.3, 0.6))
	ramp := func(k float64) complex64 { return complex64(a + b*complex(k, 0)) }
	rm1, r0, r1, r2 := ramp(-1), ramp(0), ramp(1), ramp(2)
	for _, mu := range []float32{0.1, 0.3, 0.5, 0.7, 0.95} {
		want := complex64(a + b*complex(float64(mu), 0))
		got := farrowParabolic(rm1, r0, r1, r2, mu)
		if cmplxAbs64(got-want) > 1e-5 {
			t.Errorf("farrow linear-ramp mu=%.2f: got %v want %v", mu, got, want)
		}
	}
}

// ===========================================================================
// S4: Farrow coefficient golden. Two independent checks:
//
//	(a) An EXTERNAL hand-computed value that does not reuse the docstring algebra,
//	    pinning the coefficients to an outside reference. For the unit-impulse taps
//	    (xm1,x0,x1,x2) = (0,0,1,0) at mu=0.5 with alpha=0.5: the x1-tap response is
//	    out(mu) = -0.5*mu^2 + 1.5*mu, so out(0.5) = -0.125 + 0.75 = 0.625 EXACTLY.
//	    A wrong alpha or tap order moves this off 0.625.
//	(b) A bit-tight match against the closed-form Horner coefficients over several
//	    mu, which additionally guards against a transcription typo between the doc
//	    formula and the code.
//
// ===========================================================================
func TestSymFarrowCoeffGolden(t *testing.T) {
	// (a) External golden: unit impulse on the x1 tap, mu=0.5 -> 0.625.
	if got := farrowParabolic(0, 0, complex64(1), 0, 0.5); math.Abs(float64(real(got))-0.625) > 1e-6 || math.Abs(float64(imag(got))) > 1e-6 {
		t.Errorf("farrow external golden: got %v want (0.625+0i)", got)
	}
	// Symmetric impulse on the x0 tap, mu=0.5 -> also 0.625 by the pulse symmetry
	// of the alpha=0.5 kernel (out(mu) on x0 equals out(1-mu) on x1).
	if got := farrowParabolic(0, complex64(1), 0, 0, 0.5); math.Abs(float64(real(got))-0.625) > 1e-6 {
		t.Errorf("farrow external golden (x0 tap): got %v want 0.625 real", got)
	}

	// (b) Match the closed-form coefficients over several mu.
	const a = 0.5
	taps := [4]complex128{
		complex(0.11, -0.22), complex(-0.33, 0.44),
		complex(0.55, -0.66), complex(-0.77, 0.88),
	}
	xm1, x0, x1, x2 := taps[0], taps[1], taps[2], taps[3]
	for _, mu64 := range []float64{0, 0.2, 0.5, 0.8, 0.999} {
		mu := complex(mu64, 0)
		v2 := a * (x2 - x1 - x0 + xm1)
		v1 := -a*x2 + (1+a)*x1 + (a-1)*x0 - a*xm1
		v0 := x0
		want := (v2*mu+v1)*mu + v0
		got := farrowParabolic(complex64(xm1), complex64(x0), complex64(x1), complex64(x2), float32(mu64))
		if cmplxAbs64(got-complex64(want)) > 1e-5 {
			t.Errorf("farrow golden mu=%.3f: got %v want %v", mu64, got, want)
		}
	}
}

// ===========================================================================
// S5: GARDNER open-loop S-curve (the TED sign gate). Sweep a static timing error
// tau through the detector, sampling the three Gardner points (y[k-1], y[k-1/2],
// y[k]) from the ORACLE sinc interpolator over an alternating-BPSK RC reference,
// and assert e(0) ~ 0 with a strictly POSITIVE slope through the origin. The
// midpoint is INTERPOLATED at the true half-symbol instant. The positive sign is
// hard-coded: flipping gardnerError's sign flips the slope and fails here.
//
// It also asserts a "midpoint must be the half-symbol instant" negative control:
// substituting the ON-TIME sample y[k] for the midpoint (the mistake of skipping
// the half-symbol interpolation and reusing a symbol-instant sample) destroys the
// zero crossing -- e(0) becomes large. Note we compare against y[k], NOT the
// "nearest raw dense sample" (at 32x oversampling the nearest dense sample IS
// essentially the true midpoint, so that would be an inert control proving
// nothing); reusing a full symbol-spaced sample is the genuinely wrong choice.
// ===========================================================================
func TestSymGardnerSCurve(t *testing.T) {
	syms := symAltBPSK(200)
	dense, mid := symDenseRC(syms, symDenseSps, symRolloff, symSpan)
	ds := float64(symDenseSps)

	// midMode: 0 = true half-symbol interpolation, 1 = wrongly reuse on-time y[k].
	scurve := func(tau float64, wrongMid bool) float64 {
		var sum float64
		cnt := 0
		for k := symSpan + 4; k < len(syms)-symSpan-4; k++ {
			cen := float64(k)*ds + float64(mid) + tau*ds
			prevCen := float64(k-1)*ds + float64(mid) + tau*ds
			yk := symSincInterp(dense, cen)
			yprev := symSincInterp(dense, prevCen)
			ymid := symSincInterp(dense, (cen+prevCen)/2)
			if wrongMid {
				// Wrong: reuse the on-time symbol sample as the "midpoint",
				// skipping the half-symbol interpolation entirely.
				ymid = yk
			}
			sum += gardnerError(yprev, ymid, yk)
			cnt++
		}
		return sum / float64(cnt)
	}

	// Zero crossing and positive slope over a monotone range.
	rng := 0.2
	const steps = 21
	var prev float64
	for i := 0; i < steps; i++ {
		tau := -rng + 2*rng*float64(i)/float64(steps-1)
		e := scurve(tau, false)
		if math.Abs(tau) < 1e-9 && math.Abs(e) > 1e-3 {
			t.Errorf("Gardner e(0) = %.4g, want ~0", e)
		}
		if i > 0 && e <= prev {
			t.Errorf("Gardner S-curve not increasing at tau=%.3f: e=%.4g <= prev=%.4g", tau, e, prev)
		}
		if tau > 0.02 && e <= 0 {
			t.Errorf("Gardner e(%.3f)=%.4g, want > 0 (late => positive)", tau, e)
		}
		if tau < -0.02 && e >= 0 {
			t.Errorf("Gardner e(%.3f)=%.4g, want < 0 (early => negative)", tau, e)
		}
		prev = e
	}
	t.Logf("Gardner S-curve monotone through origin over +/-%.2f symbol", rng)

	// Negative control: the wrong (on-time) midpoint destroys the e(0) ~ 0
	// property. This has teeth -- it must be substantially biased, not ~0.
	wrongBias := scurve(0, true)
	t.Logf("wrong-midpoint (reuse on-time) e(0) = %.4g (true-midpoint is ~0)", wrongBias)
	if math.Abs(wrongBias) < 0.05 {
		t.Errorf("negative control inert: wrong-midpoint e(0) = %.4g, expected large bias", wrongBias)
	}
}

// ===========================================================================
// S6: MUELLER & MULLER open-loop S-curve. Same idea for the decision-directed
// detector: with correct decisions (no carrier offset), sweep static tau and
// assert e(0) ~ 0 with positive slope through the origin. Uses random BPSK so
// decisions are well-defined.
// ===========================================================================
func TestSymMMSCurve(t *testing.T) {
	seed := uint32(31)
	syms := symSymbols(400, ModBPSK, &seed)
	dense, mid := symDenseRC(syms, symDenseSps, symRolloff, symSpan)
	ds := float64(symDenseSps)

	scurve := func(tau float64) float64 {
		var sum float64
		cnt := 0
		for k := symSpan + 4; k < len(syms)-symSpan-4; k++ {
			cen := float64(k)*ds + float64(mid) + tau*ds
			prevCen := float64(k-1)*ds + float64(mid) + tau*ds
			xk := symSincInterp(dense, cen)
			xprev := symSincInterp(dense, prevCen)
			ak := complex(symSign(real(xk)), 0)
			aprev := complex(symSign(real(xprev)), 0)
			sum += mmError(aprev, xprev, ak, xk)
			cnt++
		}
		return sum / float64(cnt)
	}

	rng := 0.15
	const steps = 21
	var prev float64
	for i := 0; i < steps; i++ {
		tau := -rng + 2*rng*float64(i)/float64(steps-1)
		e := scurve(tau)
		if math.Abs(tau) < 1e-9 && math.Abs(e) > 1e-2 {
			t.Errorf("M&M e(0) = %.4g, want ~0", e)
		}
		if i > 0 && e <= prev {
			t.Errorf("M&M S-curve not increasing at tau=%.3f: e=%.4g <= prev=%.4g", tau, e, prev)
		}
		if tau > 0.02 && e <= 0 {
			t.Errorf("M&M e(%.3f)=%.4g, want > 0", tau, e)
		}
		prev = e
	}
	t.Logf("M&M S-curve monotone through origin over +/-%.2f symbol", rng)
}

// ===========================================================================
// S7: acquisition with a clock-rate offset (ppm). Inject a known clock ratio and
// assert the loop (a) recovers the symbols (small EVM after settling) and (b)
// reports EffectiveSps / TimingPPM matching the injected rate in magnitude AND
// sign. The settling-window bound is what catches a per-sample-vs-per-symbol bnT
// mistake -- a mistuned loop still eventually converges but misses the window.
// ===========================================================================
func TestSymAcquisitionPPM(t *testing.T) {
	const (
		symbolRate = 4800.0
		spsRx      = 4.0
		sampleRate = symbolRate * spsRx
		// A modest bandwidth so the smoothed control-word rate estimate settles to
		// within a fraction of the injected offset (higher bandwidth trades rate
		// bias for faster response; see EffectiveSps's doc). Long run so wAvg fully
		// converges before the tail read.
		loopBW = symbolRate * 0.002
		nSym   = 20000
		// The injected offset is 300 ppm; require the reported estimate within
		// 80 ppm and the correct SIGN. This is ~4x tighter than the offset, so a
		// loop that fails to track (estimate ~ 0 ppm) fails the test -- the
		// assertion is not vacuous. 80 ppm = 0.00032 sps on a 4.0-sps signal.
		wantTolPPM = 80.0
		wantTolSps = 4.0 * wantTolPPM * 1e-6
	)
	// clkRatio > 1 => wider spacing => fewer true samples/symbol => negative ppm.
	for _, clkRatio := range []float64{1.0 + 300e-6, 1.0 - 300e-6} {
		seed := uint32(101)
		syms := symSymbols(nSym, ModQPSK, &seed)
		dense, mid := symDenseRC(syms, symDenseSps, symRolloff, symSpan)
		startPos := float64(mid) + 2*symDenseSps // skip the first couple symbols' transient
		rx := symReceiver(dense, startPos, spsRx, clkRatio)

		s := NewGardnerSync(symbolRate, loopBW, sampleRate, InterpFarrow)
		got := s.Process(rx)

		settleSym := len(got) / 2
		lag, evm := symBestLagEVM(got, syms, settleSym, 8)

		wantSps := spsRx / clkRatio
		wantPPM := (1.0/clkRatio - 1.0) * 1e6
		t.Logf("clkRatio=%.6f: %d symbols, lag=%d, tail EVM=%.4g, EffectiveSps=%.6f (want %.6f), TimingPPM=%.1f (want %.1f), Lock=%.3f",
			clkRatio, len(got), lag, evm, s.EffectiveSps(), wantSps, s.TimingPPM(), wantPPM, s.LockQuality())

		if evm > 0.05 {
			t.Errorf("clkRatio=%.6f: tail EVM %.4g > 0.05", clkRatio, evm)
		}
		if math.Abs(s.EffectiveSps()-wantSps) > wantTolSps {
			t.Errorf("clkRatio=%.6f: EffectiveSps %.6f, want %.6f +/- %.6f", clkRatio, s.EffectiveSps(), wantSps, wantTolSps)
		}
		if math.Abs(s.TimingPPM()-wantPPM) > wantTolPPM {
			t.Errorf("clkRatio=%.6f: TimingPPM %.1f, want %.1f +/- %.1f ppm", clkRatio, s.TimingPPM(), wantPPM, wantTolPPM)
		}
		// Sign must match: positive clkRatio-1 => negative ppm and vice versa.
		if (wantPPM > 0) != (s.TimingPPM() > 0) {
			t.Errorf("clkRatio=%.6f: TimingPPM sign wrong: got %.1f want sign of %.1f", clkRatio, s.TimingPPM(), wantPPM)
		}
		if !s.Locked() {
			t.Errorf("clkRatio=%.6f: not Locked (LockQuality=%.3f)", clkRatio, s.LockQuality())
		}
	}
}

// ===========================================================================
// S8: static fractional-delay convergence. With no clock offset but a fixed
// fractional sampling-phase offset, the loop must converge (Locked, small tail
// EVM) and hold a STABLE effective sps -- the rate estimate must not keep
// wandering. Raw Mu() is deliberately NOT used as the stability metric: Gardner
// alternates on-time and midpoint strobes whose mu values differ by ~0.5, and mu
// wraps [0,1), so a raw Mu() spread is bimodal-plus-wrapping and says nothing
// about lock. EffectiveSps (the smoothed control word) is the honest steady-state
// indicator.
// ===========================================================================
func TestSymStaticDelayMuConverges(t *testing.T) {
	const (
		symbolRate = 4800.0
		spsRx      = 4.0
		sampleRate = symbolRate * spsRx
		loopBW     = symbolRate * 0.005
		nSym       = 6000
	)
	seed := uint32(202)
	syms := symSymbols(nSym, ModQPSK, &seed)
	dense, mid := symDenseRC(syms, symDenseSps, symRolloff, symSpan)
	// A fixed fractional-sample phase offset (0.37 of a receiver sample).
	startPos := float64(mid) + 2*symDenseSps + 0.37*(symDenseSps/spsRx)
	rx := symReceiver(dense, startPos, spsRx, 1.0)

	s := NewGardnerSync(symbolRate, loopBW, sampleRate, InterpFarrow)

	// Track EffectiveSps over the settled tail; with no clock offset it must sit
	// at the nominal sps with only small residual wander.
	const chunk = 128
	var effs []float64
	for off := 0; off < len(rx); off += chunk {
		end := off + chunk
		if end > len(rx) {
			end = len(rx)
		}
		s.ProcessReuse(rx[off:end])
		effs = append(effs, s.EffectiveSps())
	}
	tail := effs[len(effs)*3/4:]
	var lo, hi = math.Inf(1), math.Inf(-1)
	for _, m := range tail {
		if m < lo {
			lo = m
		}
		if m > hi {
			hi = m
		}
	}
	t.Logf("static delay: settled EffectiveSps range over tail = [%.5f, %.5f] spread %.4g (nominal %.1f), Lock=%.3f", lo, hi, hi-lo, spsRx, s.LockQuality())
	if hi-lo > 0.02 {
		t.Errorf("EffectiveSps not settled: tail spread %.4g > 0.02", hi-lo)
	}
	if math.Abs((lo+hi)/2-spsRx) > 0.02 {
		t.Errorf("EffectiveSps settled at %.5f, want ~%.1f (no clock offset)", (lo+hi)/2, spsRx)
	}
	if !s.Locked() {
		t.Errorf("not Locked after static-delay run (LockQuality=%.3f)", s.LockQuality())
	}
}

// ===========================================================================
// S9: post-settle EVM -> small, BOTH detectors and BOTH interpolators. With a
// static delay the recovered constellation must collapse onto the source symbols.
// ===========================================================================
func TestSymEVMConverges(t *testing.T) {
	const (
		symbolRate = 4800.0
		spsRx      = 4.0
		sampleRate = symbolRate * spsRx
		loopBW     = symbolRate * 0.01
		nSym       = 5000
	)
	seed0 := uint32(303)
	syms := symSymbols(nSym, ModQPSK, &seed0)
	dense, mid := symDenseRC(syms, symDenseSps, symRolloff, symSpan)
	startPos := float64(mid) + 2*symDenseSps + 0.4*(symDenseSps/spsRx)
	rx := symReceiver(dense, startPos, spsRx, 1.0)

	for _, interp := range symAllInterps {
		t.Run("Gardner_"+interp.String(), func(t *testing.T) {
			s := NewGardnerSync(symbolRate, loopBW, sampleRate, interp)
			got := s.Process(rx)
			_, evm := symBestLagEVM(got, syms, len(got)/2, 8)
			t.Logf("Gardner/%s: %d symbols, tail EVM=%.4g, Lock=%.3f", interp, len(got), evm, s.LockQuality())
			// Linear at 4 sps carries more ISI than Farrow; allow a looser bound.
			tol := 0.06
			if interp == InterpLinear {
				tol = 0.12
			}
			if evm > tol {
				t.Errorf("Gardner/%s tail EVM %.4g > %.4g", interp, evm, tol)
			}
		})
		t.Run("MM_"+interp.String(), func(t *testing.T) {
			s := NewMuellerMullerSync(symbolRate, loopBW, sampleRate, ModQPSK, interp)
			got := s.Process(rx)
			_, evm := symBestLagEVM(got, syms, len(got)/2, 8)
			t.Logf("M&M/%s: %d symbols, tail EVM=%.4g, Lock=%.3f", interp, len(got), evm, s.LockQuality())
			tol := 0.06
			if interp == InterpLinear {
				tol = 0.12
			}
			if evm > tol {
				t.Errorf("M&M/%s tail EVM %.4g > %.4g", interp, evm, tol)
			}
		})
	}
}

// ===========================================================================
// S10: negative-feedback step. Start the loop deliberately off the correct
// sampling phase (seed the integrator) and assert the timing error's magnitude,
// measured as EVM in a sliding window, DECREASES from an early window to a late
// window -- the closed-loop statement of net negative feedback. A flipped control
// sign (W = w0 - v) makes the error grow instead; this is the runaway catcher.
// ===========================================================================
func TestSymNegativeFeedbackStep(t *testing.T) {
	const (
		symbolRate = 4800.0
		spsRx      = 4.0
		sampleRate = symbolRate * spsRx
		loopBW     = symbolRate * 0.015
		nSym       = 3000
	)
	seed := uint32(404)
	syms := symSymbols(nSym, ModQPSK, &seed)
	dense, mid := symDenseRC(syms, symDenseSps, symRolloff, symSpan)
	// Inject a sizable static phase error so there is a real step to pull in.
	startPos := float64(mid) + 2*symDenseSps + 0.45*(symDenseSps/spsRx)
	rx := symReceiver(dense, startPos, spsRx, 1.0)

	s := NewGardnerSync(symbolRate, loopBW, sampleRate, InterpFarrow)
	got := s.Process(rx)

	// EVM in an early window vs a late window (aligned to source by best lag).
	lag, _ := symBestLagEVM(got, syms, len(got)/2, 8)
	winEVM := func(start, n int) float64 {
		var sumSq float64
		cnt := 0
		for i := start; i < start+n && i+lag < len(got) && i < len(syms); i++ {
			d := complex128(got[i+lag]) - complex128(syms[i])
			sumSq += real(d)*real(d) + imag(d)*imag(d)
			cnt++
		}
		return math.Sqrt(sumSq / float64(cnt))
	}
	early := winEVM(20, 200)
	late := winEVM(len(got)-400, 200)
	t.Logf("negative-feedback: early-window EVM=%.4g, late-window EVM=%.4g", early, late)
	if late >= early {
		t.Errorf("timing error did not shrink: early=%.4g late=%.4g (feedback sign?)", early, late)
	}
}

// ===========================================================================
// S11: split-matches-single (variable length). The same input processed as one
// block and as uneven chunks must yield BIT-IDENTICAL concatenated symbols and
// identical final state -- proving loop state carries across Process boundaries
// with no dependence on chunk size despite the variable output length.
// ===========================================================================
func TestSymSplitMatchesSingle(t *testing.T) {
	const (
		symbolRate = 4800.0
		spsRx      = 4.0
		sampleRate = symbolRate * spsRx
		loopBW     = symbolRate * 0.01
	)
	seed := uint32(505)
	syms := symSymbols(2000, ModQPSK, &seed)
	dense, mid := symDenseRC(syms, symDenseSps, symRolloff, symSpan)
	rx := symReceiver(dense, float64(mid)+2*symDenseSps, spsRx, 1.0)

	chunks := []int{7, 13, 29, 64, 3, 101}
	cases := []struct {
		name string
		mk   func() *SymbolSync
	}{
		{"Gardner", func() *SymbolSync { return NewGardnerSync(symbolRate, loopBW, sampleRate, InterpFarrow) }},
		{"MM", func() *SymbolSync { return NewMuellerMullerSync(symbolRate, loopBW, sampleRate, ModQPSK, InterpFarrow) }},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			single := tc.mk()
			want := single.Process(rx)

			split := tc.mk()
			var got []complex64
			off := 0
			ci := 0
			for off < len(rx) {
				n := chunks[ci%len(chunks)]
				ci++
				if off+n > len(rx) {
					n = len(rx) - off
				}
				got = append(got, split.Process(rx[off:off+n])...)
				off += n
			}
			if len(got) != len(want) {
				t.Fatalf("%s: split produced %d symbols, single %d", tc.name, len(got), len(want))
			}
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("%s: symbol %d differs: got %v want %v", tc.name, i, got[i], want[i])
				}
			}
			if !symStateEqual(single, split) {
				t.Errorf("%s: final state differs after split processing", tc.name)
			}
		})
	}
}

// symStateEqual reports whether two loops carry bit-identical carried state.
func symStateEqual(a, b *SymbolSync) bool {
	return a.eta == b.eta && a.w == b.w &&
		a.d0 == b.d0 && a.d1 == b.d1 && a.d2 == b.d2 && a.d3 == b.d3 &&
		a.nhist == b.nhist && a.strobeOnTime == b.strobeOnTime &&
		a.prevOnTime == b.prevOnTime && a.havePrevOn == b.havePrevOn &&
		a.mid == b.mid && a.haveMid == b.haveMid &&
		a.prevSample == b.prevSample && a.prevDecision == b.prevDecision && a.havePrevMM == b.havePrevMM &&
		a.lf.Integrator() == b.lf.Integrator()
}

// ===========================================================================
// S12: reset-equals-fresh. A loop that processed data, then Reset, must behave
// bit-identically to a fresh loop on the next input.
// ===========================================================================
func TestSymResetEqualsFresh(t *testing.T) {
	const (
		symbolRate = 4800.0
		spsRx      = 4.0
		sampleRate = symbolRate * spsRx
		loopBW     = symbolRate * 0.01
	)
	seedW := uint32(606)
	warm := symSymbols(1500, ModQPSK, &seedW)
	denseW, midW := symDenseRC(warm, symDenseSps, symRolloff, symSpan)
	rxWarm := symReceiver(denseW, float64(midW)+2*symDenseSps, spsRx, 1.0+200e-6)

	seedI := uint32(707)
	in := symSymbols(1500, ModQPSK, &seedI)
	denseI, midI := symDenseRC(in, symDenseSps, symRolloff, symSpan)
	rxIn := symReceiver(denseI, float64(midI)+2*symDenseSps, spsRx, 1.0)

	for _, tc := range []struct {
		name string
		mk   func() *SymbolSync
	}{
		{"Gardner", func() *SymbolSync { return NewGardnerSync(symbolRate, loopBW, sampleRate, InterpFarrow) }},
		{"MM", func() *SymbolSync { return NewMuellerMullerSync(symbolRate, loopBW, sampleRate, ModQPSK, InterpFarrow) }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			reused := tc.mk()
			reused.Process(rxWarm)
			reused.Reset()
			got := reused.Process(rxIn)

			fresh := tc.mk()
			want := fresh.Process(rxIn)

			if len(got) != len(want) {
				t.Fatalf("%s: reused %d symbols, fresh %d", tc.name, len(got), len(want))
			}
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("%s: symbol %d after Reset differs: got %v want %v", tc.name, i, got[i], want[i])
				}
			}
			if !symStateEqual(reused, fresh) {
				t.Errorf("%s: state after Reset+Process differs from fresh", tc.name)
			}
		})
	}
}

// ===========================================================================
// S13: zero-alloc ProcessReuse. After a warm-up sizes the internal buffer,
// ProcessReuse must allocate nothing per call even though output length varies.
// ===========================================================================
func TestSymProcessReuseZeroAlloc(t *testing.T) {
	const (
		symbolRate = 4800.0
		spsRx      = 4.0
		sampleRate = symbolRate * spsRx
		loopBW     = symbolRate * 0.01
	)
	seed := uint32(808)
	syms := symSymbols(3000, ModQPSK, &seed)
	dense, mid := symDenseRC(syms, symDenseSps, symRolloff, symSpan)
	rx := symReceiver(dense, float64(mid)+2*symDenseSps, spsRx, 1.0)

	s := NewGardnerSync(symbolRate, loopBW, sampleRate, InterpFarrow)
	s.ProcessReuse(rx) // warm the outBuf

	allocs := testing.AllocsPerRun(50, func() {
		s.ProcessReuse(rx)
	})
	t.Logf("ProcessReuse allocs/op = %.1f", allocs)
	if allocs != 0 {
		t.Errorf("ProcessReuse allocated %.1f times, want 0", allocs)
	}
}

// ===========================================================================
// S14: constructor panics. Non-positive rates/bandwidth, unknown interpolator,
// unknown modulation, and too-few-samples-per-symbol for each detector.
// ===========================================================================
func TestSymPanics(t *testing.T) {
	mustPanic := func(t *testing.T, name string, f func()) {
		defer func() {
			if r := recover(); r == nil {
				t.Errorf("expected panic for %s", name)
			}
		}()
		f()
	}
	// Gardner cases.
	mustPanic(t, "gardner_fs<=0", func() { NewGardnerSync(4800, 48, 0, InterpFarrow) })
	mustPanic(t, "gardner_symrate<=0", func() { NewGardnerSync(0, 48, 19200, InterpFarrow) })
	mustPanic(t, "gardner_bw<=0", func() { NewGardnerSync(4800, 0, 19200, InterpFarrow) })
	mustPanic(t, "gardner_bad_interp", func() { NewGardnerSync(4800, 48, 19200, Interpolator(99)) })
	mustPanic(t, "gardner_sps<2", func() { NewGardnerSync(4800, 48, 4800*1.5, InterpFarrow) })
	// M&M cases.
	mustPanic(t, "mm_bad_mod", func() { NewMuellerMullerSync(4800, 48, 19200, Modulation(99), InterpFarrow) })
	mustPanic(t, "mm_sps<1", func() { NewMuellerMullerSync(4800, 48, 4800*0.5, ModQPSK, InterpFarrow) })
	mustPanic(t, "mm_bad_interp", func() { NewMuellerMullerSync(4800, 48, 19200, ModQPSK, Interpolator(99)) })
}

// ===========================================================================
// S15: M&M REQUIRES a recovered carrier -- measured on DECISIONS, not the lock
// flag, and demonstrating WHY M&M must come after carrier recovery (not before).
//
// Two subtleties pinned by the debugging that built this test:
//  1. LockQuality is the constant-modulus ratio, which is carrier-BLIND for a PSK
//     constellation (a rotation does not change |y|). So an uncorrected carrier
//     does NOT lower M&M's LockQuality -- the assertion must be on SYMBOL ERRORS.
//  2. M&M is decision-directed, so a spinning constellation corrupts its OWN
//     timing error, not merely the final decisions -- it cannot simply run first
//     and be cleaned up by a downstream Costas (that is Gardner's privilege; see
//     S16). Hence the correct chain is carrier-recovery-THEN-M&M. The "carrier
//     recovered upstream" ideal is modelled here by the carrier-FREE stream (what
//     a converged Costas delivers): M&M on it decodes with zero symbol errors,
//     while M&M on the carrier-spinning stream produces a high symbol-error rate.
//
// ===========================================================================
func TestSymMMNeedsCarrierLock(t *testing.T) {
	const (
		symbolRate = 4800.0
		spsRx      = 4.0
		sampleRate = symbolRate * spsRx
		loopBW     = symbolRate * 0.005
		dfHz       = 60.0
		nSym       = 8000
	)
	seed := uint32(909)
	syms := symSymbols(nSym, ModQPSK, &seed)
	dense, mid := symDenseRC(syms, symDenseSps, symRolloff, symSpan)
	rxClean := symReceiver(dense, float64(mid)+2*symDenseSps, spsRx, 1.0)
	rxCarrier := costasApplyCarrier(rxClean, dfHz, 0.3, sampleRate)

	// symErrRate returns the minimum QPSK symbol-error fraction over the tail,
	// across all pi/2 rotations and the best two-sided lag.
	symErrRate := func(got []complex64) float64 {
		settle := len(got) / 2
		best := 1.0
		for r := 0; r < 4; r++ {
			rot := cmplx.Rect(1, float64(r)*math.Pi/2)
			lag, _ := symBestLagEVMRot(got, syms, settle, 8, rot)
			errs, cnt := 0, 0
			for i := settle; i < len(syms); i++ {
				j := i + lag
				if j < 0 || j >= len(got) {
					continue
				}
				g := complex128(got[j]) * rot
				if symSign(float32(real(g))) != symSign(float32(real(syms[i]))) ||
					symSign(float32(imag(g))) != symSign(float32(imag(syms[i]))) {
					errs++
				}
				cnt++
			}
			if cnt > 0 {
				if f := float64(errs) / float64(cnt); f < best {
					best = f
				}
			}
		}
		return best
	}

	// (a) M&M on the carrier-corrupted stream, NO carrier recovery: the decisions
	// (and thus the loop itself) are driven by a spinning constellation -> high
	// symbol-error rate.
	sBad := NewMuellerMullerSync(symbolRate, loopBW, sampleRate, ModQPSK, InterpFarrow)
	serBad := symErrRate(sBad.Process(rxCarrier))

	// (b) M&M on a carrier-free stream (the "carrier recovered upstream" ideal) ->
	// zero symbol errors.
	sGood := NewMuellerMullerSync(symbolRate, loopBW, sampleRate, ModQPSK, InterpFarrow)
	serGood := symErrRate(sGood.Process(rxClean))

	t.Logf("M&M symbol-error rate: with residual carrier=%.4g ; carrier-free=%.4g (Lock bad=%.3f good=%.3f)",
		serBad, serGood, sBad.LockQuality(), sGood.LockQuality())

	if serBad < 0.1 {
		t.Errorf("M&M on uncorrected carrier should have high symbol-error rate, got %.4g", serBad)
	}
	if serGood != 0 {
		t.Errorf("M&M on carrier-free stream should yield zero symbol errors, got rate %.4g", serGood)
	}
}

// ===========================================================================
// S16: end-to-end QPSK. Timing offset + clock ppm + a residual carrier -> Gardner
// THEN Costas. Order matters and is the crux of this test: Gardner is
// carrier-independent, so it recovers symbol timing first and decimates the
// oversampled stream to one constellation point per symbol; Costas then cleans the
// residual carrier on those near-constant-modulus points. (The reverse order --
// Costas on the 4-sps pulse-shaped stream -- fails, because between symbol centers
// the RC pulse sweeps through amplitude/phase excursions and zero crossings that
// the per-sample decision-directed PED cannot track.) The carrier offset is kept
// modest (within the loop's pull-in range at this bandwidth -- see the Costas doc:
// coarse acquisition first, small residual only), so no SetFreq seeding is needed.
// The recovered constellation must decode to the source symbols with ZERO symbol
// errors after resolving the QPSK pi/2 ambiguity.
// ===========================================================================
func TestSymEndToEndQPSK(t *testing.T) {
	const (
		symbolRate = 4800.0
		spsRx      = 4.0
		sampleRate = symbolRate * spsRx
		timingBW   = symbolRate * 0.005
		carrierBW  = symbolRate * 0.02 // relative to the SYMBOL rate (1 sps into Costas)
		dfHz       = 60.0              // residual carrier, within Costas pull-in
		nSym       = 8000
	)
	seed := uint32(1234)
	syms := symSymbols(nSym, ModQPSK, &seed)
	dense, mid := symDenseRC(syms, symDenseSps, symRolloff, symSpan)
	startPos := float64(mid) + 2*symDenseSps + 0.3*(symDenseSps/spsRx)
	rx := symReceiver(dense, startPos, spsRx, 1.0+150e-6)
	rx = costasApplyCarrier(rx, dfHz, 0.5, sampleRate)

	// Symbol timing first (carrier-independent), then carrier recovery on the
	// 1-sample-per-symbol constellation points.
	sync := NewGardnerSync(symbolRate, timingBW, sampleRate, InterpFarrow)
	sym1 := sync.Process(rx)
	cos := NewCostas(carrierBW, symbolRate, CostasQPSK)
	got := cos.Process(sym1)

	// Resolve the QPSK pi/2 ambiguity: pick the rotation and lag minimizing errors.
	settle := len(got) / 2
	bestErr := math.MaxInt32
	bestRot := 0
	for r := 0; r < 4; r++ {
		rot := cmplx.Rect(1, float64(r)*math.Pi/2)
		// Count hard symbol errors at the best (two-sided) lag for this rotation.
		lag, evm := symBestLagEVMRot(got, syms, settle, 8, rot)
		errs := 0
		for i := settle; i < len(syms); i++ {
			j := i + lag
			if j < 0 || j >= len(got) {
				continue
			}
			g := complex128(got[j]) * rot
			if symSign(float32(real(g))) != symSign(float32(real(syms[i]))) ||
				symSign(float32(imag(g))) != symSign(float32(imag(syms[i]))) {
				errs++
			}
		}
		if errs < bestErr {
			bestErr = errs
			bestRot = r
		}
		t.Logf("rotation %d*pi/2: EVM=%.4g, symErrors=%d", r, evm, errs)
	}
	t.Logf("end-to-end: %d symbols, best rotation=%d*pi/2, symbol errors=%d, carrier FreqOffset=%.1f Hz, timing Lock=%.3f",
		len(got), bestRot, bestErr, cos.FreqOffset(), sync.LockQuality())
	if bestErr != 0 {
		t.Errorf("end-to-end QPSK: %d symbol errors after ambiguity resolution, want 0", bestErr)
	}
}

// symBestLagEVMRot is symBestLagEVM with a fixed constellation rotation applied to
// got before comparison (used to resolve the QPSK pi/2 carrier ambiguity in S16).
// The lag sweep is two-sided for the same reason as symBestLagEVM.
func symBestLagEVMRot(got, want []complex64, settle, maxLag int, rot complex128) (bestLag int, bestEVM float64) {
	bestEVM = math.Inf(1)
	for lag := -maxLag; lag <= maxLag; lag++ {
		var sumSq float64
		cnt := 0
		for i := settle; i < len(want); i++ {
			j := i + lag
			if j < 0 || j >= len(got) {
				continue
			}
			g := complex128(got[j]) * rot
			d := g - complex128(want[i])
			sumSq += real(d)*real(d) + imag(d)*imag(d)
			cnt++
		}
		if cnt < 16 {
			continue
		}
		evm := math.Sqrt(sumSq / float64(cnt))
		if evm < bestEVM {
			bestEVM = evm
			bestLag = lag
		}
	}
	return bestLag, bestEVM
}

// cmplxAbs64 is the magnitude of a complex64 difference, for interpolator tests.
func cmplxAbs64(z complex64) float64 {
	return math.Hypot(float64(real(z)), float64(imag(z)))
}
