package dsp

import "math"

// Interpolator selects the fractional-delay interpolator a SymbolSync uses to
// evaluate the input stream at the loop-controlled instant between two input
// samples. It is orthogonal to the timing-error detector (Gardner vs M&M): any
// detector can drive any interpolator, so the two are separate parameters rather
// than a combined 2x2 constructor set.
//
//   - InterpLinear: two-tap linear interpolation, out = x0 + mu*(x1-x0). Cheap,
//     but it is only exact for a straight line; on a real (band-limited) signal
//     it introduces amplitude/phase error that the loop sees as timing jitter.
//     Adequate at heavy oversampling; a strict accuracy downgrade otherwise.
//   - InterpFarrow: the piecewise-parabolic Farrow interpolator (Erup-Gardner-
//     Harris) with the standard free parameter alpha = 0.5, evaluated over four
//     taps (x[m-1], x[m], x[m+1], x[m+2]). It is a degree-2 polynomial in mu and
//     reproduces polynomials up to degree ONE exactly (a constant and a ramp);
//     it does NOT reproduce an arbitrary quadratic exactly -- that is a property
//     of a cubic Lagrange kernel, not of the parabolic Farrow. alpha = 0.5 gives
//     the maximally-flat magnitude response and is the canonical pairing with a
//     Gardner timing loop.
type Interpolator int

const (
	// InterpLinear: two-tap linear interpolation.
	InterpLinear Interpolator = iota
	// InterpFarrow: four-tap piecewise-parabolic Farrow, alpha = 0.5.
	InterpFarrow
)

// String renders the interpolator name for readable test logs and errors.
func (i Interpolator) String() string {
	switch i {
	case InterpLinear:
		return "Linear"
	case InterpFarrow:
		return "Farrow"
	default:
		return "Interpolator(?)"
	}
}

// Modulation selects the decision slicer the decision-directed Mueller & Muller
// timing-error detector uses to form its symbol decisions. It is meaningful only
// for a M&M loop; a Gardner loop is non-data-aided and ignores it. The two
// supported constellations mirror the Costas CarrierPED set so a M&M SymbolSync
// pairs naturally with a Costas of the same modulation.
//
//   - ModBPSK: decision = sign(Re(x)) on the real axis (imag decided as 0).
//   - ModQPSK: decision = sign(Re(x)) + j*sign(Im(x)), the four 45-degree points.
//
// sign(0) is taken as +1 (see symSign), matching the slicer convention used by
// the Costas detectors.
type Modulation int

const (
	// ModBPSK: real-axis binary slicer.
	ModBPSK Modulation = iota
	// ModQPSK: quadrant slicer on the 45-degree grid.
	ModQPSK
)

// String renders the modulation name for readable test logs and errors.
func (m Modulation) String() string {
	switch m {
	case ModBPSK:
		return "BPSK"
	case ModQPSK:
		return "QPSK"
	default:
		return "Modulation(?)"
	}
}

// timingTED is the internal selector for which timing-error detector a
// SymbolSync runs. It is unexported because the two detectors are chosen through
// distinct constructors (NewGardnerSync / NewMuellerMullerSync) whose required
// parameters genuinely differ -- Gardner takes no modulation, M&M requires one.
type timingTED int

const (
	tedGardner timingTED = iota // 2 strobes/symbol, non-data-aided
	tedMM                       // 1 strobe/symbol, decision-directed
)

// symLockEps guards the constant-modulus lock ratio E[|y|]^2 / E[|y|^2] against a
// near-zero-power transient before the EMAs have spun up.
const symLockEps = 1e-12

// SymbolSync is an interpolating symbol-timing-recovery loop: a fractional-delay
// interpolator whose sampling instant is steered every symbol by a type-2 PI
// LoopFilter driven by a timing-error detector (TED). It consumes an
// asynchronously-sampled complex baseband stream at roughly sps = sampleRate /
// symbolRate samples per symbol and emits ONE recovered symbol per symbol period,
// sampled at the interpolated optimum instant. Because the symbol clock is not
// locked to the sample clock, the output length is VARIABLE (about len(in)/sps),
// which is why Process returns a freshly-sized slice rather than a 1:1 mapping.
//
// # Interpolator control (mod-1 decrementing counter)
//
// The loop follows the Gardner-Erup-Harris interpolator-control structure. A
// modulo-1 counter eta is decremented by a control word W once per input sample;
// when it underflows (eta < W) that input sample is a strobe instant. The
// fractional interval mu = eta / W in [0,1) locates the interpolation point
// between two input samples, and the interpolator evaluates the stream there. W
// is the number of strobes per input sample: W = outsPerSym / sps, so the counter
// underflows outsPerSym times per symbol period (twice for Gardner, once for
// M&M). The LoopFilter nudges W each symbol to track the true symbol clock.
//
// A four-tap complex history d0..d3 (d3 = newest) feeds the interpolator with the
// basepoint fixed at d1 = x[m] (interpolating within [d1, d2] using d0 and d3 as
// the outer taps). Because d3 = x[m+2] must already be in hand, the interpolator
// carries a fixed two-sample look-ahead latency; the first strobe therefore
// cannot fire until four input samples have been buffered.
//
// # Timing-error detectors
//
//   - Gardner (NewGardnerSync): non-data-aided, needs 2 samples/symbol, and is
//     INDEPENDENT of carrier recovery -- it can run before or without a Costas
//     loop. It strobes twice per symbol: at the on-time instant y[k] and at the
//     midpoint y[k-1/2] halfway to the previous symbol. The error is
//     e = Re{ (y[k] - y[k-1]) * conj(y[k-1/2]) }. The midpoint MUST be an
//     interpolated sample, never a raw input sample -- using a raw sample biases
//     the S-curve (guarded by the tau = 0.5 S-curve test).
//   - Mueller & Muller (NewMuellerMullerSync): decision-directed, needs only 1
//     sample/symbol, but REQUIRES the carrier to be recovered first (it slices to
//     a constellation decision, which is meaningless while the constellation
//     spins). The error is e = Re{ conj(ahat[k])*x[k-1] - conj(ahat[k-1])*x[k] }
//     where ahat is the sliced decision for the selected Modulation (see mmError
//     for the sign derivation; the mirror-image ordering has the opposite slope).
//
// # Sign convention (the number-one timing-loop bug)
//
// Both detectors are built so that near lock e ~ +Kd*tau, where tau > 0 means the
// strobes are LATE (sampling after the symbol center). The control update W =
// w0 + v with v ~ +e then advances the strobe (a larger W underflows the counter
// sooner), pulling the sampling instant earlier to close the error -- net NEGATIVE
// feedback. The closed-loop sign also depends on the interpolation-index / mu
// direction, which is the classic footgun; it is PINNED BY THE S-CURVE TEST
// (open-loop, against an independent oracle interpolator), not by trusting this
// comment. If the S-curve slope came out negative the fix is the documented
// one-liner W = w0 - v.
//
// # Lock metric (constant-modulus)
//
// LockQuality is the constant-modulus ratio LQ = E[|y|]^2 / E[|y|^2] of the
// recovered symbols, formed from two per-symbol EMAs (the mean magnitude and the
// mean power). For a zero-ISI pulse (RRC/RC matched-filter output) sampled AT the
// symbol center the magnitude is constant, so the ratio -> 1; when the strobe is
// off-center, intersymbol interference makes |y| vary and the ratio falls below 1.
// By the Cauchy-Schwarz inequality LQ lies in [0, 1] with no tuning constant, and
// it is amplitude-independent (a global gain cancels top and bottom). Locked
// reports LockQuality > 0.9.
//
// This reflects TIMING lock ONLY. It is deliberately blind to a residual carrier:
// for a constant-modulus constellation (PSK) a carrier rotation does not change
// |y|, so a timing-locked Gardner loop reads LQ ~ 1 even while the carrier spins
// (that is Costas's job). A M&M loop can likewise report timing lock on a
// carrier-uncorrected stream while producing garbage DECISIONS -- see the M&M
// precondition note above; the caller must ensure carrier recovery for M&M.
//
// # State and precision
//
// The counter eta, the control word W, the LoopFilter integrator, and the lock
// EMAs are all carried in float64: the integrator sums tiny per-symbol increments
// over long runs and the counter must not lose low bits before forming mu. Only
// the interpolated OUTPUT symbols are complex64. Loop state carries across
// Process calls, so streaming in chunks yields the same result as one block.
//
// SymbolSync is not safe for concurrent use.
type SymbolSync struct {
	ted        timingTED    // which detector
	interp     Interpolator // fractional interpolator
	mod        Modulation   // M&M decision constellation (ignored by Gardner)
	sps        float64      // nominal samples per symbol = sampleRate/symbolRate
	symbolRate float64      // symbols/second, for PPM reporting
	outsPerSym int          // strobes per symbol: 2 (Gardner) or 1 (M&M)
	w0         float64      // nominal control word = outsPerSym/sps
	wLo, wHi   float64      // hard clamp on the control word (counter safety)
	lf         *LoopFilter  // type-2 PI loop filter; integ is the rate estimate

	// Carried per-sample state.
	eta            float64   // mod-1 counter in [0,1)
	w              float64   // current control word (w0 + loop correction)
	d0, d1, d2, d3 complex64 // 4-tap history, d3 = newest
	nhist          int       // buffered input samples (warm-up gate; caps at 4)
	mu             float64   // fractional interval of the last strobe, [0,1)

	// Carried Gardner strobe state. strobeOnTime toggles each strobe: true means
	// the next strobe is an on-time instant y[k], false a midpoint y[k-1/2].
	strobeOnTime bool
	prevOnTime   complex64 // y[k-1]
	havePrevOn   bool
	mid          complex64 // y[k-1/2]
	haveMid      bool

	// Carried M&M strobe state.
	prevSample   complex64 // x[k-1] on-time interpolant
	prevDecision complex64 // ahat[k-1]
	havePrevMM   bool

	// Lock metric state: constant-modulus EMAs LQ = magAvg^2 / powAvg.
	alpha  float64 // EMA smoothing factor in [1e-4, 0.5]
	magAvg float64 // mean |y| of recovered symbols
	powAvg float64 // mean |y|^2 of recovered symbols

	// wAvg is a smoothed estimate of the control word, used by EffectiveSps /
	// TimingPPM. It is NOT the loop-filter integrator: at steady state the DC rate
	// correction is carried substantially by the loop filter's PROPORTIONAL branch
	// (k2 is O(bnT^2), so the integrator settles very slowly on TED self-noise),
	// which means w0 + integrator alone is a biased -- and often wrong-signed --
	// rate estimate. The control word w = w0 + k1*e + integ that the loop actually
	// applies each symbol equals outsPerSym/trueSps at steady state, so its EMA is
	// the honest tracked-rate estimate. Seeded to w0 so a fresh loop reports the
	// nominal sps.
	wAvg float64

	// outBuf is a reusable output buffer for ProcessReuse; valid only until the
	// next Process/ProcessReuse call on this loop.
	outBuf []complex64
}

// newSymbolSync is the shared constructor body. loopBandwidthHz is converted to
// the PER-SYMBOL normalized bandwidth bnT = loopBandwidthHz/symbolRate (the
// symbol-timing convention; contrast Costas's per-sample bnT). The loop filter
// uses damping zeta = 0.707 and detector/oscillator gains kd = k0 = 1.
func newSymbolSync(ted timingTED, mod Modulation, interp Interpolator, symbolRate, loopBandwidthHz, sampleRate float64, outsPerSym int) *SymbolSync {
	if sampleRate <= 0 {
		panic("SymbolSync: sampleRate must be positive")
	}
	if symbolRate <= 0 {
		panic("SymbolSync: symbolRate must be positive")
	}
	if loopBandwidthHz <= 0 {
		panic("SymbolSync: loopBandwidthHz must be positive")
	}
	if interp != InterpLinear && interp != InterpFarrow {
		panic("SymbolSync: unknown Interpolator")
	}
	sps := sampleRate / symbolRate
	// Gardner needs at least 2 samples/symbol (it strobes the midpoint); M&M
	// needs at least 1. Below these the counter cannot produce the required
	// strobes per symbol.
	if ted == tedGardner && sps < 2 {
		panic("SymbolSync: Gardner requires sampleRate/symbolRate >= 2")
	}
	if ted == tedMM && sps < 1 {
		panic("SymbolSync: Mueller-Muller requires sampleRate/symbolRate >= 1")
	}

	bnT := loopBandwidthHz / symbolRate
	lf := NewLoopFilter(bnT, 0.707, 1, 1)
	// Anti-windup: bound the rate estimate to +/-50% of the nominal control
	// word, so the tracked clock deviation cannot run away during acquisition.
	lf.SetClamp(0.5 * float64(outsPerSym) / sps)

	w0 := float64(outsPerSym) / sps

	// Hard clamp on the control word, applied when it is formed each symbol. This
	// absorbs the transient proportional term and, together with the wHi <= 1
	// ceiling, guarantees at most one strobe per input sample (so the variable
	// output length is bounded by len(in)).
	wLo := 0.25 * w0
	wHi := 1.75 * w0
	if wHi > 1.0 {
		wHi = 1.0
	}

	alpha := bnT
	if alpha < 1e-4 {
		alpha = 1e-4
	} else if alpha > 0.5 {
		alpha = 0.5
	}

	return &SymbolSync{
		ted:          ted,
		interp:       interp,
		mod:          mod,
		sps:          sps,
		symbolRate:   symbolRate,
		outsPerSym:   outsPerSym,
		w0:           w0,
		wLo:          wLo,
		wHi:          wHi,
		lf:           lf,
		w:            w0,
		wAvg:         w0,
		strobeOnTime: true, // first Gardner strobe is treated as on-time
		alpha:        alpha,
	}
}

// NewGardnerSync builds a non-data-aided Gardner timing-recovery loop for a
// stream at sampleRate carrying symbolRate symbols/second, with one-sided loop
// noise bandwidth loopBandwidthHz and the given fractional interpolator. Gardner
// is carrier-independent (it may run before Costas) but requires at least two
// samples per symbol.
//
// It panics on sampleRate <= 0, symbolRate <= 0, loopBandwidthHz <= 0, an
// unknown interpolator, or sampleRate/symbolRate < 2.
func NewGardnerSync(symbolRate, loopBandwidthHz, sampleRate float64, interp Interpolator) *SymbolSync {
	return newSymbolSync(tedGardner, ModBPSK, interp, symbolRate, loopBandwidthHz, sampleRate, 2)
}

// NewMuellerMullerSync builds a decision-directed Mueller & Muller timing-recovery
// loop for the given constellation mod. M&M runs at one sample per symbol but
// REQUIRES a recovered carrier upstream (it slices to a decision); feeding it a
// stream with a residual carrier offset prevents lock. Use it downstream of a
// Costas loop of the same modulation.
//
// It panics on sampleRate <= 0, symbolRate <= 0, loopBandwidthHz <= 0, an
// unknown interpolator, an unknown modulation, or sampleRate/symbolRate < 1.
func NewMuellerMullerSync(symbolRate, loopBandwidthHz, sampleRate float64, mod Modulation, interp Interpolator) *SymbolSync {
	if mod != ModBPSK && mod != ModQPSK {
		panic("SymbolSync: unknown Modulation")
	}
	return newSymbolSync(tedMM, mod, interp, symbolRate, loopBandwidthHz, sampleRate, 1)
}

// linearInterp evaluates the two-tap linear interpolant between x0 (mu=0) and x1
// (mu=1) at fractional interval mu, computed per rail.
func linearInterp(x0, x1 complex64, mu float32) complex64 {
	r0, i0 := real(x0), imag(x0)
	r1, i1 := real(x1), imag(x1)
	return complex(r0+mu*(r1-r0), i0+mu*(i1-i0))
}

// farrowParabolic evaluates the piecewise-parabolic Farrow interpolant (alpha =
// 0.5) over the four taps xm1 = x[m-1], x0 = x[m], x1 = x[m+1], x2 = x[m+2] at
// fractional interval mu in [0,1) between x0 and x1. It is computed per rail as a
// Horner-form quadratic in mu:
//
//	v2 = a*(x2 - x1 - x0 + xm1)
//	v1 = -a*x2 + (1+a)*x1 + (a-1)*x0 - a*xm1
//	v0 = x0
//	out = (v2*mu + v1)*mu + v0     with a = 0.5
//
// By construction out(0) = x0 and out(1) = x1, and the kernel reproduces a
// constant and a linear ramp exactly (but NOT an arbitrary quadratic).
func farrowParabolic(xm1, x0, x1, x2 complex64, mu float32) complex64 {
	const a = float32(0.5)
	fr := func(m1, z0, z1, z2 float32) float32 {
		v2 := a * (z2 - z1 - z0 + m1)
		v1 := -a*z2 + (1+a)*z1 + (a-1)*z0 - a*m1
		v0 := z0
		return (v2*mu+v1)*mu + v0
	}
	re := fr(real(xm1), real(x0), real(x1), real(x2))
	im := fr(imag(xm1), imag(x0), imag(x1), imag(x2))
	return complex(re, im)
}

// interpolate evaluates the configured interpolator at fractional interval mu
// over the current four-tap history, basepoint d1. Linear uses only d1,d2; Farrow
// uses all four with d1 as x[m].
func (s *SymbolSync) interpolate(mu float32) complex64 {
	if s.interp == InterpLinear {
		return linearInterp(s.d1, s.d2, mu)
	}
	return farrowParabolic(s.d0, s.d1, s.d2, s.d3, mu)
}

// symSign returns +1 for x >= 0 and -1 for x < 0 (sign(0) = +1), matching the
// slicer convention used by the Costas detectors.
func symSign(x float32) float32 {
	if x < 0 {
		return -1
	}
	return 1
}

// gardnerError is the Gardner timing-error detector, factored out as a free
// function so the open-loop S-curve test can drive it directly with samples from
// an independent oracle interpolator (the same pattern as carrierPED in costas.go).
// Given the current on-time symbol onTime = y[k], the previous on-time
// prevOnTime = y[k-1], and the interpolated midpoint mid = y[k-1/2], it returns
//
//	e = Re{ (y[k] - y[k-1]) * conj(y[k-1/2]) }.
//
// For an alternating (half-symbol-rate) BPSK tone of amplitude A sampled with a
// timing error tau (tau > 0 = sampling late), this evaluates to
// e = A^2 * sin(2*pi*tau) -- zero at tau = 0 with POSITIVE slope through the
// origin. That positive sign is what pairs with the W = w0 + v control update to
// give net negative feedback; it is asserted by the S-curve test.
func gardnerError(prevOnTime, mid, onTime complex64) float64 {
	dR := real(onTime) - real(prevOnTime)
	dI := imag(onTime) - imag(prevOnTime)
	return float64(dR*real(mid) + dI*imag(mid))
}

// mmError is the Mueller & Muller decision-directed timing-error detector,
// factored out for the same open-loop-test reason as gardnerError. Given the
// current on-time sample sample = x[k] with decision decision = ahat[k], and the
// previous prevSample = x[k-1] with prevDecision = ahat[k-1], it returns
//
//	e = Re{ conj(ahat[k])*x[k-1] - conj(ahat[k-1])*x[k] }.
//
// The sign convention is pinned to match gardnerError by the S-curve test:
// e ~ +Kd*tau near lock (tau > 0 late), pairing with W = w0 + v for net negative
// feedback. (The mirror-image ordering e = Re{ahat[k-1]* x[k] - ahat[k]* x[k-1]}
// gives the opposite slope; the S-curve test rejects it.) M&M is decision-directed,
// so a correct decision -- which requires a recovered carrier -- is a precondition
// for the error to be meaningful.
func mmError(prevDecision, prevSample, decision, sample complex64) float64 {
	t1 := real(decision)*real(prevSample) + imag(decision)*imag(prevSample)
	t2 := real(prevDecision)*real(sample) + imag(prevDecision)*imag(sample)
	return float64(t1 - t2)
}

// slice returns the constellation decision ahat for on-time sample x under the
// configured M&M modulation: BPSK on the real axis, QPSK on the 45-degree grid.
func (s *SymbolSync) slice(x complex64) complex64 {
	if s.mod == ModQPSK {
		return complex(symSign(real(x)), symSign(imag(x)))
	}
	return complex(symSign(real(x)), 0)
}

// controlWord forms the next control word from the loop correction v and applies
// the hard [wLo, wHi] clamp that keeps the counter well-behaved.
func (s *SymbolSync) controlWord(v float64) float64 {
	w := s.w0 + v
	if w < s.wLo {
		w = s.wLo
	} else if w > s.wHi {
		w = s.wHi
	}
	return w
}

// updateLock folds one recovered on-time symbol y (sampled under control word w)
// into the per-symbol EMAs: the constant-modulus magnitude/power pair behind
// LockQuality, and the smoothed control word behind EffectiveSps. Only ON-TIME
// symbols (the ones actually emitted) update these; the Gardner midpoint strobe
// does not.
func (s *SymbolSync) updateLock(y complex64, w float64) {
	mag := math.Hypot(float64(real(y)), float64(imag(y)))
	s.magAvg += s.alpha * (mag - s.magAvg)
	s.powAvg += s.alpha * (mag*mag - s.powAvg)
	s.wAvg += s.alpha * (w - s.wAvg)
}

// process runs the timing loop over in, appending one recovered symbol per symbol
// period to dst (which must have capacity for the worst-case count) and returning
// the grown slice. Per-sample hot state is hoisted into locals and written back
// once at the end so a warmed ProcessReuse allocates nothing.
func (s *SymbolSync) process(in, dst []complex64) []complex64 {
	eta := s.eta
	w := s.w
	d0, d1, d2, d3 := s.d0, s.d1, s.d2, s.d3
	nhist := s.nhist

	for _, x := range in {
		// Push the new sample into the 4-tap history (d3 newest).
		d0, d1, d2, d3 = d1, d2, d3, x
		if nhist < 4 {
			// Warm-up: the interpolator needs four valid taps before it can
			// strobe. The counter idles until then (a fixed startup latency).
			nhist++
			continue
		}

		if eta < w {
			// Underflow: this input sample is a strobe instant.
			mu := eta / w
			s.d0, s.d1, s.d2, s.d3 = d0, d1, d2, d3 // interpolate() reads fields
			y := s.interpolate(float32(mu))
			s.mu = mu

			if s.ted == tedGardner {
				if s.strobeOnTime {
					// On-time instant y[k]. Form the Gardner error from the
					// previous on-time y[k-1] and the midpoint y[k-1/2].
					if s.havePrevOn && s.haveMid {
						e := gardnerError(s.prevOnTime, s.mid, y)
						v := s.lf.Advance(e)
						w = s.controlWord(v)
					}
					s.updateLock(y, w)
					dst = append(dst, y)
					s.prevOnTime = y
					s.havePrevOn = true
					s.haveMid = false
					s.strobeOnTime = false // next strobe is the midpoint
				} else {
					// Midpoint instant y[k-1/2]; interpolated, never a raw sample.
					s.mid = y
					s.haveMid = true
					s.strobeOnTime = true // next strobe is on-time
				}
			} else {
				// Mueller & Muller: one strobe per symbol, decision-directed.
				ahat := s.slice(y)
				if s.havePrevMM {
					e := mmError(s.prevDecision, s.prevSample, ahat, y)
					v := s.lf.Advance(e)
					w = s.controlWord(v)
				}
				s.updateLock(y, w)
				dst = append(dst, y)
				s.prevSample = y
				s.prevDecision = ahat
				s.havePrevMM = true
			}

			eta += 1 // reload the mod-1 counter
		}
		eta -= w
	}

	s.eta = eta
	s.w = w
	s.d0, s.d1, s.d2, s.d3 = d0, d1, d2, d3
	s.nhist = nhist
	return dst
}

// maxOut returns a hard upper bound on the number of symbols process can emit for
// an input of length n. Because the control word is clamped to wHi <= 1, the
// counter underflows at most once per input sample, so total strobes <= n; the
// emitted-symbol count never exceeds that (Gardner emits only half the strobes).
func (s *SymbolSync) maxOut(n int) int {
	return int(float64(n)*s.wHi) + 2
}

// Process runs the timing loop over the input stream and returns a freshly
// allocated slice of the recovered symbols (about len(in)/sps of them). Loop
// state carries across calls, so streaming in chunks yields the same symbols as
// one block.
func (s *SymbolSync) Process(in []complex64) []complex64 {
	dst := make([]complex64, 0, s.maxOut(len(in)))
	return s.process(in, dst)
}

// ProcessReuse runs the timing loop and returns the recovered symbols in a slice
// backed by an internal buffer, valid only until the next Process/ProcessReuse
// call on this loop. Sized once to the worst-case count, the buffer does not grow
// on repeated calls with the same input length, so the hot path allocates nothing.
func (s *SymbolSync) ProcessReuse(in []complex64) []complex64 {
	need := s.maxOut(len(in))
	if cap(s.outBuf) < need {
		s.outBuf = make([]complex64, 0, need)
	}
	return s.process(in, s.outBuf[:0])
}

// Mu returns the fractional interval mu in [0,1) of the most recent strobe, the
// interpolation point between two input samples. A settled loop tracking a
// constant timing offset holds Mu roughly constant.
func (s *SymbolSync) Mu() float64 {
	return s.mu
}

// EffectiveSps returns the loop's current estimate of the true samples-per-symbol,
// which follows the transmit-clock rate (including any sampling-frequency offset)
// rather than the nominal sampleRate/symbolRate. It is outsPerSym divided by the
// SMOOTHED control word (wAvg), which at steady state equals outsPerSym/trueSps.
//
// It deliberately does NOT use w0 + integrator: because the integral gain k2 is
// O(bnT^2), the loop filter's integrator settles very slowly and, for a small
// clock offset, the steady-state DC rate correction is carried substantially by
// the PROPORTIONAL branch (k1*e) instead -- the Gardner/M&M TEDs have a nonzero
// mean error at lock of the same order as a modest ppm offset. Reading only the
// integrator therefore gives a biased, frequently wrong-signed estimate; the
// applied control word w (proportional + integral) is the honest one. Before the
// loop has processed any symbols wAvg = w0, so a fresh loop reports the nominal
// sps. Expect a residual jitter of a few hundred ppm at typical bandwidths.
func (s *SymbolSync) EffectiveSps() float64 {
	if s.wAvg <= 0 {
		return float64(s.outsPerSym) / s.w0
	}
	return float64(s.outsPerSym) / s.wAvg
}

// TimingPPM returns the tracked sampling-clock deviation in parts per million,
// (EffectiveSps/nominalSps - 1)*1e6. A positive value means the true clock puts
// more samples per symbol than nominal.
func (s *SymbolSync) TimingPPM() float64 {
	return (s.EffectiveSps()/s.sps - 1) * 1e6
}

// LockQuality returns the timing-lock indicator in [0,1]: the constant-modulus
// ratio E[|y|]^2 / E[|y|^2] of the recovered symbols. It approaches 1 when the
// strobe sits at the zero-ISI symbol centers (constant magnitude) and falls when
// the sampling instant drifts off-center (ISI makes |y| vary). It reflects TIMING
// lock only -- for a PSK constellation a residual carrier does not change |y|, so
// a Gardner loop reads ~1 even while the carrier is still unrecovered.
func (s *SymbolSync) LockQuality() float64 {
	if s.powAvg <= symLockEps {
		return 0
	}
	q := s.magAvg * s.magAvg / s.powAvg
	if q > 1 {
		q = 1
	}
	return q
}

// Locked reports whether the timing loop is locked, using the documented
// heuristic LockQuality() > 0.9.
func (s *SymbolSync) Locked() bool {
	return s.LockQuality() > 0.9
}

// Reset returns the loop to its just-constructed state: counter, control word,
// history, strobe state, loop-filter integrator, and lock EMAs all cleared.
// Configuration (detector, interpolator, modulation, gains, clamps) is preserved.
func (s *SymbolSync) Reset() {
	s.eta = 0
	s.w = s.w0
	s.d0, s.d1, s.d2, s.d3 = 0, 0, 0, 0
	s.nhist = 0
	s.mu = 0
	s.strobeOnTime = true
	s.prevOnTime = 0
	s.havePrevOn = false
	s.mid = 0
	s.haveMid = false
	s.prevSample = 0
	s.prevDecision = 0
	s.havePrevMM = false
	s.magAvg = 0
	s.powAvg = 0
	s.wAvg = s.w0
	s.lf.Reset()
}
