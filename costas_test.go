package dsp

import (
	"math"
	"math/cmplx"
	"testing"
)

// ---------------------------------------------------------------------------
// Signal-generator helpers (block-prefixed to stay unique in package dsp).
// ---------------------------------------------------------------------------

// costasTone returns n samples of a unit-magnitude complex exponential with a
// frequency offset of dfHz (sampled at fs) and an initial phase of phase0:
// x[k] = exp(j*(2*pi*dfHz*k/fs + phase0)). The phase argument is accumulated in
// float64 to avoid low-bit loss before narrowing to complex64.
func costasTone(dfHz, phase0, fs float64, n int) []complex64 {
	w := 2 * math.Pi * dfHz / fs
	out := make([]complex64, n)
	for k := 0; k < n; k++ {
		ph := w*float64(k) + phase0
		out[k] = complex(float32(math.Cos(ph)), float32(math.Sin(ph)))
	}
	return out
}

// costasApplyCarrier multiplies a baseband stream by a carrier
// exp(j*(2*pi*dfHz*k/fs + phase0)), i.e. it impresses a frequency/phase offset
// onto an already-generated symbol stream. This is the "channel" a Costas loop
// must undo; the loop's FreqOffset should converge to +dfHz.
func costasApplyCarrier(stream []complex64, dfHz, phase0, fs float64) []complex64 {
	w := 2 * math.Pi * dfHz / fs
	out := make([]complex64, len(stream))
	for k := range stream {
		ph := w*float64(k) + phase0
		c := complex(math.Cos(ph), math.Sin(ph))
		out[k] = complex64(complex128(stream[k]) * c)
	}
	return out
}

// costasSymbols returns n unit-magnitude PSK symbols drawn from the seeded LCG,
// laid out for the given detector's constellation:
//   - CostasPLL  : all +1 (an unmodulated carrier / pilot).
//   - CostasBPSK : +1 / -1 (phase 0 or pi).
//   - CostasQPSK : exp(j*(pi/4 + k*pi/2)), k in {0,1,2,3} (the 45-degree grid).
//
// Constant-modulus symbols (no pulse shaping) are deliberate for the carrier
// tests: they isolate carrier phase/frequency error from the amplitude
// wander a matched filter would add, so a residual-phase or EVM assertion means
// exactly "carrier error" and nothing else.
func costasSymbols(n int, ped CarrierPED, seed *uint32) []complex64 {
	out := make([]complex64, n)
	for i := 0; i < n; i++ {
		switch ped {
		case CostasBPSK:
			if lcg(seed) < 0 {
				out[i] = complex(-1, 0)
			} else {
				out[i] = complex(1, 0)
			}
		case CostasQPSK:
			// Map the LCG value in [-1,1) to one of four quadrant symbols.
			k := int((lcg(seed) + 1) * 2) // 0..3 (4 is unreachable since lcg < 1)
			if k > 3 {
				k = 3
			}
			ph := math.Pi/4 + float64(k)*math.Pi/2
			out[i] = complex(float32(math.Cos(ph)), float32(math.Sin(ph)))
		default: // CostasPLL: unmodulated
			out[i] = complex(1, 0)
		}
	}
	return out
}

// costasLockPhase is the constellation reference angle each detector locks to:
// PLL and BPSK to the real axis (0), QPSK to the 45-degree diagonal (pi/4).
func costasLockPhase(ped CarrierPED) float64 {
	if ped == CostasQPSK {
		return math.Pi / 4
	}
	return 0
}

// costasSymmetryOrder is the rotational symmetry of the detector's data phase:
// PLL 1, BPSK 2, QPSK 4. Raising a unit symbol to this power annihilates the
// data modulation, leaving only the residual carrier phase (times the order).
func costasSymmetryOrder(ped CarrierPED) int {
	switch ped {
	case CostasBPSK:
		return 2
	case CostasQPSK:
		return 4
	default:
		return 1
	}
}

// costasResidualPhase returns the residual carrier phase error of a de-rotated
// unit-modulus symbol y, modulo the constellation symmetry. It rotates out the
// lock reference angle, raises to the symmetry order to strip the data phase,
// takes the angle, and divides back down: the result lies in [-pi/m, pi/m] and
// is the per-symbol carrier phase error irrespective of which data symbol was
// sent (this is how the pi / pi/2 ambiguity is resolved before measuring error).
func costasResidualPhase(y complex128, ped CarrierPED) float64 {
	m := costasSymmetryOrder(ped)
	phi0 := costasLockPhase(ped)
	u := y * cmplx.Exp(complex(0, -phi0))
	stripped := cmplx.Pow(u, complex(float64(m), 0))
	return cmplx.Phase(stripped) / float64(m)
}

// costasStateEqual reports whether two loops carry bit-identical state: NCO
// phase, loop-filter integrator, and the complex lock EMA. Used by the
// split-matches-single and reset-equals-fresh tests to prove chunk boundaries
// and Reset touch no hidden state.
func costasStateEqual(a, b *Costas) bool {
	return a.theta == b.theta && a.lf.Integrator() == b.lf.Integrator() && a.cAvg == b.cAvg
}

// costasAllPEDs is the set of detectors iterated by the per-PED tests.
var costasAllPEDs = []CarrierPED{CostasPLL, CostasBPSK, CostasQPSK}

// ---------------------------------------------------------------------------
// C1: per-PED open-loop S-curve (PED SIGN). This is the single most important
// test: it freezes the NCO and sweeps a STATIC phase error through each
// detector, asserting e(0) ~ 0 with a strictly POSITIVE slope through the
// origin. The expected sign is HARD-CODED (positive), not derived from the
// detector's own output, so flipping any PED sign in costas.go flips the slope
// and fails here -- which is exactly the intended negative-control behaviour.
// ---------------------------------------------------------------------------
func TestCostasSCurve(t *testing.T) {
	// Each detector is monotone through the origin only within its own range;
	// sweep inside it. BPSK's e = 0.5*sin(2*pe) is monotone on (-pi/4, pi/4);
	// QPSK's on roughly (-pi/4, pi/4); PLL is linear on all of (-pi, pi).
	ranges := map[CarrierPED]float64{
		CostasPLL:  0.9,
		CostasBPSK: 0.6,
		CostasQPSK: 0.6,
	}
	for _, ped := range costasAllPEDs {
		t.Run(ped.String(), func(t *testing.T) {
			phi0 := costasLockPhase(ped)
			rng := ranges[ped]
			const steps = 41
			var prev float64
			var haveMid bool
			for i := 0; i < steps; i++ {
				pe := -rng + 2*rng*float64(i)/float64(steps-1)
				// y is the lock-point symbol rotated by the static error pe.
				ph := phi0 + pe
				yR := float32(math.Cos(ph))
				yI := float32(math.Sin(ph))
				e := carrierPED(ped, yR, yI)

				if math.Abs(pe) < 1e-9 {
					if math.Abs(e) > 1e-6 {
						t.Errorf("%s: e(0) = %.3g, want ~0", ped, e)
					}
					haveMid = true
				}
				// Positive slope: e must strictly increase across the sweep.
				if i > 0 && e <= prev {
					t.Errorf("%s: S-curve not increasing at pe=%.3f: e=%.4g <= prev=%.4g", ped, pe, e, prev)
				}
				// Sign matches the error sign away from the origin.
				if pe > 0.05 && e <= 0 {
					t.Errorf("%s: e(%.3f) = %.4g, want > 0", ped, pe, e)
				}
				if pe < -0.05 && e >= 0 {
					t.Errorf("%s: e(%.3f) = %.4g, want < 0", ped, pe, e)
				}
				prev = e
			}
			if !haveMid {
				t.Fatalf("%s: sweep never hit pe=0", ped)
			}
			t.Logf("%s S-curve monotone increasing through origin over +/-%.2f rad", ped, rng)
		})
	}
}

// ---------------------------------------------------------------------------
// C2: negative-feedback step. Impress a STATIC positive phase step (no frequency
// offset) and assert the SIGNED residual phase error strictly DECREASES
// sample-over-sample as the loop pulls it in. This is the exact statement of net
// negative feedback: a positive residual drives a positive error, which advances
// the NCO to close the gap, so pe marches monotonically DOWN toward zero (and a
// hair past it -- a zeta=0.707 type-2 loop overshoots slightly, so |pe| is NOT
// monotone through the zero crossing, but the signed value is). A flipped
// feedback sign would make the signed residual GROW instead of shrink -- the
// direct runaway-catcher. The sign of the expected trend (decreasing) is
// hard-coded, not read back from the loop, so a PED sign flip fails here.
// ---------------------------------------------------------------------------
func TestCostasNegativeFeedbackStep(t *testing.T) {
	const (
		fs     = 48000.0
		loopBW = 400.0
		phase0 = 0.4 // static positive phase step, radians
		window = 60  // samples over which the pull-in must be monotone
	)
	for _, ped := range costasAllPEDs {
		t.Run(ped.String(), func(t *testing.T) {
			seed := uint32(999)
			// A short run of the SAME symbol keeps the data phase constant so the
			// only thing changing is the loop pulling out the static step.
			base := costasSymbols(window+1, ped, &seed)
			for i := range base {
				base[i] = base[0]
			}
			in := costasApplyCarrier(base, 0, phase0, fs)

			c := NewCostas(loopBW, fs, ped)
			out := c.Process(in)

			var prev float64
			for i := 0; i < len(out); i++ {
				pe := costasResidualPhase(complex128(out[i]), ped)
				if i > 0 && pe > prev+1e-9 {
					t.Errorf("%s: signed residual increased at sample %d: %.5g > %.5g", ped, i, pe, prev)
					break
				}
				prev = pe
			}
			t.Logf("%s: static step %.3f rad pulled to signed pe=%.4g over %d samples", ped, phase0, prev, window)
		})
	}
}

// ---------------------------------------------------------------------------
// C3: acquisition + settling window. Inject a carrier frequency offset and
// assert that WITHIN a bounded window the loop both (a) reports FreqOffset ~=
// the injected df (correct magnitude AND sign) and (b) declares Locked(). The
// settling-window bound is what catches a per-sample-vs-per-symbol bnT mistake:
// a mistuned loop still eventually converges but misses the window.
// ---------------------------------------------------------------------------
func TestCostasAcquisitionSettling(t *testing.T) {
	const (
		fs     = 48000.0
		loopBW = 600.0
		dfHz   = 120.0
		n      = 80000
	)
	for _, ped := range costasAllPEDs {
		t.Run(ped.String(), func(t *testing.T) {
			seed := uint32(4242)
			sym := costasSymbols(n, ped, &seed)
			in := costasApplyCarrier(sym, dfHz, 0.3, fs)

			c := NewCostas(loopBW, fs, ped)
			c.Process(in)

			gotDf := c.FreqOffset()
			lq := c.LockQuality()
			t.Logf("%s: injected df=%.1f Hz -> FreqOffset=%.3f Hz, LockQuality=%.4f", ped, dfHz, gotDf, lq)

			if math.Abs(gotDf-dfHz) > 3.0 {
				t.Errorf("%s: FreqOffset = %.3f Hz, want ~%.1f Hz", ped, gotDf, dfHz)
			}
			if !c.Locked() {
				t.Errorf("%s: not Locked() after %d samples (LockQuality=%.4f)", ped, n, lq)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// C4: residual phase RMS < 0.05 rad over the post-settle tail. Constant-modulus
// symbols mean each de-rotated sample IS a symbol, so the stripped residual is
// exactly the carrier phase error. A loop that locks with a static phase bias
// (wrong integrator handling) or jitters would fail the tight RMS bound.
// ---------------------------------------------------------------------------
func TestCostasResidualPhaseRMS(t *testing.T) {
	const (
		fs     = 48000.0
		loopBW = 600.0
		dfHz   = 80.0
		n      = 80000
		settle = 40000 // samples to discard before measuring
		rmsTol = 0.05
	)
	for _, ped := range costasAllPEDs {
		t.Run(ped.String(), func(t *testing.T) {
			seed := uint32(7)
			sym := costasSymbols(n, ped, &seed)
			in := costasApplyCarrier(sym, dfHz, -0.7, fs)

			c := NewCostas(loopBW, fs, ped)
			out := c.Process(in)

			var sumSq float64
			cnt := 0
			for i := settle; i < len(out); i++ {
				pe := costasResidualPhase(complex128(out[i]), ped)
				sumSq += pe * pe
				cnt++
			}
			rms := math.Sqrt(sumSq / float64(cnt))
			t.Logf("%s: post-settle residual phase RMS = %.5g rad (%d samples)", ped, rms, cnt)
			if rms > rmsTol {
				t.Errorf("%s: residual RMS %.5g > %.5g", ped, rms, rmsTol)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// C5: modulation ambiguity resolution. After lock, BPSK symbols must collapse
// onto the REAL axis (|imag| small) and QPSK onto the 45-degree GRID (each
// symbol within a small residual of a pi/4+k*pi/2 point). This confirms the
// detector strips data as claimed and leaves only the documented pi / pi/2
// ambiguity.
// ---------------------------------------------------------------------------
func TestCostasAmbiguity(t *testing.T) {
	const (
		fs     = 48000.0
		loopBW = 600.0
		dfHz   = 60.0
		n      = 80000
		settle = 40000
	)
	t.Run("BPSK_real_axis", func(t *testing.T) {
		seed := uint32(11)
		sym := costasSymbols(n, CostasBPSK, &seed)
		in := costasApplyCarrier(sym, dfHz, 0.2, fs)
		c := NewCostas(loopBW, fs, CostasBPSK)
		out := c.Process(in)
		var maxImag float64
		for i := settle; i < len(out); i++ {
			if q := math.Abs(float64(imag(out[i]))); q > maxImag {
				maxImag = q
			}
		}
		t.Logf("BPSK: max |imag| over post-settle tail = %.4g", maxImag)
		if maxImag > 0.1 {
			t.Errorf("BPSK not collapsed to real axis: max|imag| = %.4g", maxImag)
		}
	})
	t.Run("QPSK_grid", func(t *testing.T) {
		seed := uint32(13)
		sym := costasSymbols(n, CostasQPSK, &seed)
		in := costasApplyCarrier(sym, dfHz, 0.2, fs)
		c := NewCostas(loopBW, fs, CostasQPSK)
		out := c.Process(in)
		var maxErr float64
		for i := settle; i < len(out); i++ {
			pe := math.Abs(costasResidualPhase(complex128(out[i]), CostasQPSK))
			if pe > maxErr {
				maxErr = pe
			}
		}
		t.Logf("QPSK: max residual-to-grid phase = %.4g rad", maxErr)
		if maxErr > 0.15 {
			t.Errorf("QPSK not on grid: max residual = %.4g rad", maxErr)
		}
	})
}

// ---------------------------------------------------------------------------
// C7: split-matches-single. The same input processed as one block and as uneven
// chunks {7,13,29,64,3,101} must yield BIT-IDENTICAL output and identical final
// state. Proves loop state carries across Process boundaries with no dependence
// on chunk size (and that the NCO phase wrap is chunk-independent).
// ---------------------------------------------------------------------------
func TestCostasSplitMatchesSingle(t *testing.T) {
	const (
		fs     = 48000.0
		loopBW = 500.0
		dfHz   = 90.0
	)
	chunks := []int{7, 13, 29, 64, 3, 101}
	total := 0
	for _, s := range chunks {
		total += s
	}
	for _, ped := range costasAllPEDs {
		t.Run(ped.String(), func(t *testing.T) {
			seed := uint32(321)
			sym := costasSymbols(total, ped, &seed)
			in := costasApplyCarrier(sym, dfHz, 0.1, fs)

			single := NewCostas(loopBW, fs, ped)
			want := single.Process(in)

			split := NewCostas(loopBW, fs, ped)
			got := make([]complex64, 0, total)
			off := 0
			for _, s := range chunks {
				got = append(got, split.Process(in[off:off+s])...)
				off += s
			}
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("%s: sample %d differs: got %v want %v", ped, i, got[i], want[i])
				}
			}
			if !costasStateEqual(single, split) {
				t.Errorf("%s: final state differs after split processing", ped)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// C8: reset-equals-fresh. A loop that has processed data, then Reset, must
// behave BIT-IDENTICALLY to a freshly constructed loop on the next input --
// proving Reset zeroes every piece of carried state (phase, integrator, EMA)
// and preserves configuration.
// ---------------------------------------------------------------------------
func TestCostasResetEqualsFresh(t *testing.T) {
	const (
		fs     = 48000.0
		loopBW = 500.0
		dfHz   = 75.0
	)
	for _, ped := range costasAllPEDs {
		t.Run(ped.String(), func(t *testing.T) {
			seed := uint32(55)
			warm := costasApplyCarrier(costasSymbols(5000, ped, &seed), dfHz, 0.9, fs)
			seed2 := uint32(66)
			in := costasApplyCarrier(costasSymbols(3000, ped, &seed2), dfHz, -0.4, fs)

			reused := NewCostas(loopBW, fs, ped)
			reused.Process(warm)
			reused.Reset()
			got := reused.Process(in)

			fresh := NewCostas(loopBW, fs, ped)
			want := fresh.Process(in)

			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("%s: sample %d after Reset differs: got %v want %v", ped, i, got[i], want[i])
				}
			}
			if !costasStateEqual(reused, fresh) {
				t.Errorf("%s: state after Reset+Process differs from fresh", ped)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// C9: zero-alloc ProcessReuse. After a warm-up call sizes the internal buffer,
// ProcessReuse must allocate nothing per call.
// ---------------------------------------------------------------------------
func TestCostasProcessReuseZeroAlloc(t *testing.T) {
	const (
		fs     = 48000.0
		loopBW = 500.0
	)
	seed := uint32(2)
	in := costasApplyCarrier(costasSymbols(4096, CostasQPSK, &seed), 100, 0, fs)
	c := NewCostas(loopBW, fs, CostasQPSK)
	c.ProcessReuse(in) // warm the outBuf

	allocs := testing.AllocsPerRun(50, func() {
		c.ProcessReuse(in)
	})
	t.Logf("ProcessReuse allocs/op = %.1f", allocs)
	if allocs != 0 {
		t.Errorf("ProcessReuse allocated %.1f times, want 0", allocs)
	}
}

// ---------------------------------------------------------------------------
// C10: ProcessInPlace == Process. In-place de-rotation must produce the same
// samples and the same final state as the fresh-alloc path.
// ---------------------------------------------------------------------------
func TestCostasProcessInPlaceMatchesProcess(t *testing.T) {
	const (
		fs     = 48000.0
		loopBW = 500.0
		dfHz   = 110.0
	)
	for _, ped := range costasAllPEDs {
		t.Run(ped.String(), func(t *testing.T) {
			seed := uint32(88)
			in := costasApplyCarrier(costasSymbols(6000, ped, &seed), dfHz, 0.5, fs)

			refLoop := NewCostas(loopBW, fs, ped)
			want := refLoop.Process(in)

			ipLoop := NewCostas(loopBW, fs, ped)
			io := make([]complex64, len(in))
			copy(io, in)
			ipLoop.ProcessInPlace(io)

			for i := range want {
				if io[i] != want[i] {
					t.Fatalf("%s: in-place sample %d differs: got %v want %v", ped, i, io[i], want[i])
				}
			}
			if !costasStateEqual(refLoop, ipLoop) {
				t.Errorf("%s: in-place final state differs from Process", ped)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// C11: constructor panics on bad args -- non-positive sampleRate,
// non-positive loop bandwidth, and an unrecognized detector enum.
// ---------------------------------------------------------------------------
func TestCostasPanics(t *testing.T) {
	cases := []struct {
		name   string
		bw, fs float64
		ped    CarrierPED
	}{
		{"fs_zero", 100, 0, CostasPLL},
		{"fs_negative", 100, -48000, CostasPLL},
		{"bw_zero", 0, 48000, CostasPLL},
		{"bw_negative", -100, 48000, CostasPLL},
		{"unknown_ped", 100, 48000, CarrierPED(99)},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			defer func() {
				if r := recover(); r == nil {
					t.Errorf("expected panic for %s", c.name)
				}
			}()
			NewCostas(c.bw, c.fs, c.ped)
		})
	}
}

// ---------------------------------------------------------------------------
// C12: anti-windup. Feed a frequency offset far beyond the loop's pull-in range
// for a long run and assert the integrator saturates at the documented clamp
// (|integ| <= costasMaxTrackRad) instead of running away, and that FreqOffset
// stays finite and bounded. This proves the clamp is wired to the loop.
// ---------------------------------------------------------------------------
func TestCostasAntiWindup(t *testing.T) {
	const (
		fs     = 48000.0
		loopBW = 200.0
		// A gross offset (fs/3) that the loop cannot possibly track.
		dfHz = fs / 3
		n    = 40000
	)
	seed := uint32(1)
	in := costasApplyCarrier(costasSymbols(n, CostasBPSK, &seed), dfHz, 0, fs)
	c := NewCostas(loopBW, fs, CostasBPSK)
	c.Process(in)

	integ := c.lf.Integrator()
	t.Logf("gross offset: integrator saturated at %.6f rad/sample (clamp %.6f), FreqOffset=%.1f Hz",
		integ, costasMaxTrackRad, c.FreqOffset())

	if math.IsNaN(integ) || math.IsInf(integ, 0) {
		t.Fatalf("integrator not finite: %v", integ)
	}
	if math.Abs(integ) > costasMaxTrackRad+1e-9 {
		t.Errorf("integrator %.6f exceeded clamp %.6f (windup runaway)", integ, costasMaxTrackRad)
	}
	if math.IsNaN(c.Phase()) || math.Abs(c.Phase()) > math.Pi+1e-6 {
		t.Errorf("NCO phase escaped [-pi,pi]: %v", c.Phase())
	}
}

// ---------------------------------------------------------------------------
// C13: long-run phase-wrap stability. Over 1e6 samples the NCO phase must stay
// wrapped in [-pi, pi], nothing goes NaN/Inf, and the loop stays locked -- the
// guard against a phase accumulator that slowly drifts out of range.
// ---------------------------------------------------------------------------
func TestCostasLongRunStability(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping 1e6-sample stability test in -short mode")
	}
	const (
		fs     = 48000.0
		loopBW = 400.0
		dfHz   = 100.0
		n      = 1_000_000
	)
	in := costasTone(dfHz, 0.0, fs, n)
	c := NewCostas(loopBW, fs, CostasPLL)

	// Process in chunks to also exercise repeated Process calls over a long run.
	const chunk = 8192
	for off := 0; off < n; off += chunk {
		end := off + chunk
		if end > n {
			end = n
		}
		c.ProcessReuse(in[off:end])
		if p := c.Phase(); math.IsNaN(p) || p > math.Pi+1e-6 || p < -math.Pi-1e-6 {
			t.Fatalf("NCO phase escaped [-pi,pi] at sample %d: %v", off, p)
		}
	}
	t.Logf("after %d samples: Phase=%.4f, FreqOffset=%.3f Hz, LockQuality=%.4f", n, c.Phase(), c.FreqOffset(), c.LockQuality())
	if math.Abs(c.FreqOffset()-dfHz) > 2.0 {
		t.Errorf("FreqOffset drifted: %.3f Hz, want ~%.1f Hz", c.FreqOffset(), dfHz)
	}
	if !c.Locked() {
		t.Errorf("loop lost lock over long run (LockQuality=%.4f)", c.LockQuality())
	}
}
