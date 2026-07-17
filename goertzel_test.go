package dsp

import (
	"math"
	"testing"
)

// N is chosen a power of two so the same buffer can be cross-checked against the
// package FFT, and targetHz is placed exactly on a bin center so the snapped
// integer k analyzes precisely the tone we inject.
const (
	gzN  = 1024
	gzFs = 8000.0
	gzK  = 128 // target bin -> targetHz = gzK*gzFs/gzN = 1000 Hz, an exact bin center
)

func gzTargetHz() float64 { return float64(gzK) * gzFs / float64(gzN) }

// realTone builds n samples of a real cosine at freqHz with the given amplitude.
func realTone(n int, freqHz, amp float64) []float32 {
	x := make([]float32, n)
	for i := 0; i < n; i++ {
		x[i] = float32(amp * math.Cos(2*math.Pi*freqHz*float64(i)/gzFs))
	}
	return x
}

// phaseTone builds n samples of a real cosine at freqHz with a phase offset. A
// zero-phase cosine has a purely REAL DFT, so it cannot constrain the imaginary
// part of Complex(); a phase-offset cosine (phase not a multiple of pi) has a
// substantial imaginary DFT component, which is what pins the -jw phase-sign
// convention.
func phaseTone(n int, freqHz, amp, phase float64) []float32 {
	x := make([]float32, n)
	for i := 0; i < n; i++ {
		x[i] = float32(amp * math.Cos(2*math.Pi*freqHz*float64(i)/gzFs+phase))
	}
	return x
}

// TestGoertzelMatchesFFTBin verifies that a pure tone at the target frequency
// yields Goertzel power equal to the corresponding FFT bin power. For an
// integer-k Goertzel over N real samples, Power() is exactly |X[k]|^2 of the
// length-N DFT, so agreement should be tight (well under a percent); we allow a
// few percent to be safe against float32 input quantization.
func TestGoertzelMatchesFFTBin(t *testing.T) {
	targetHz := gzTargetHz()
	x := realTone(gzN, targetHz, 1.0)

	g := NewGoertzel(targetHz, gzFs, gzN)
	g.Update(x)
	gzPower := g.Power()

	// Reference: full FFT of the same real samples, power at bin gzK.
	buf := make([]complex128, gzN)
	for i, v := range x {
		buf[i] = complex(float64(v), 0)
	}
	FFT(buf)
	re, im := real(buf[gzK]), imag(buf[gzK])
	fftPower := re*re + im*im

	relErr := math.Abs(gzPower-fftPower) / fftPower
	if relErr > 0.03 {
		t.Errorf("Goertzel power %.6g vs FFT bin power %.6g: rel err %.4f exceeds 3%%",
			gzPower, fftPower, relErr)
	}

	// Complex() must also match FFT bin k (same forward -j convention), which is
	// a stronger check than power alone since it pins the phase too. The
	// zero-phase cosine above has a purely REAL DFT (im ~ 0), so it CANNOT catch a
	// phase-sign flip (a conjugated Complex() would still pass). Re-run with a
	// phase-offset cosine whose bin k has a large imaginary part, and compare BOTH
	// real and imag against the FFT -- this is what constrains the -jw convention.
	xp := phaseTone(gzN, targetHz, 1.0, 0.7)
	gp := NewGoertzel(targetHz, gzFs, gzN)
	gp.Update(xp)
	c := gp.Complex()

	pbuf := make([]complex128, gzN)
	for i, v := range xp {
		pbuf[i] = complex(float64(v), 0)
	}
	FFT(pbuf)
	pre, pim := real(pbuf[gzK]), imag(pbuf[gzK])

	// Guard against a toothless test: the phase offset must make the imaginary
	// part a real fraction of the magnitude, otherwise the phase check is vacuous.
	if math.Abs(pim) < 0.2*math.Hypot(pre, pim) {
		t.Fatalf("phase-offset tone has near-real DFT (re=%.6g im=%.6g); phase check would be vacuous", pre, pim)
	}

	cErr := math.Hypot(real(c)-pre, imag(c)-pim) / math.Hypot(pre, pim)
	if cErr > 0.03 {
		t.Errorf("Goertzel Complex() %v vs FFT bin %v: rel err %.4f exceeds 3%%",
			c, pbuf[gzK], cErr)
	}

	// A conjugated Complex() (the wrong +j/-j convention) must FAIL this same
	// tolerance -- otherwise the check above proves nothing about phase sign.
	conj := complex(real(c), -imag(c))
	conjErr := math.Hypot(real(conj)-pre, imag(conj)-pim) / math.Hypot(pre, pim)
	if conjErr <= 0.03 {
		t.Errorf("conjugated Complex() %v also matched FFT bin %v (err %.4f); phase-sign check is toothless",
			conj, pbuf[gzK], conjErr)
	}
}

// TestGoertzelOffTargetRejects checks selectivity: a tone placed on a distant
// bin produces vastly less power at the detector's bin than an on-target tone
// of the same amplitude. Both off-target tones sit exactly on bins so there is
// no leakage confound; the ratio should be enormous (many orders of magnitude),
// so a 1e4 floor is conservative.
func TestGoertzelOffTargetRejects(t *testing.T) {
	targetHz := gzTargetHz()
	onTarget := realTone(gzN, targetHz, 1.0)
	offHz := float64(gzK+40) * gzFs / float64(gzN) // 40 bins away, still a bin center
	offTarget := realTone(gzN, offHz, 1.0)

	g := NewGoertzel(targetHz, gzFs, gzN)
	g.Update(onTarget)
	onPow := g.Power()

	g.Reset()
	g.Update(offTarget)
	offPow := g.Power()

	if offPow <= 0 {
		// Exactly zero is fine (perfect rejection); guard the division below.
		offPow = math.SmallestNonzeroFloat64
	}
	ratio := onPow / offPow
	if ratio < 1e4 {
		t.Errorf("selectivity too weak: on-target %.6g / off-target %.6g = %.3g, want >= 1e4",
			onPow, offPow, ratio)
	}
}

// TestGoertzelQuadraticScaling verifies power is quadratic in input amplitude:
// doubling the amplitude quadruples the power. Goertzel is linear in its input
// (a linear recurrence) and Power() is magnitude-squared, so 2x amplitude ->
// 4x power to numerical precision.
func TestGoertzelQuadraticScaling(t *testing.T) {
	targetHz := gzTargetHz()

	g := NewGoertzel(targetHz, gzFs, gzN)
	g.Update(realTone(gzN, targetHz, 1.0))
	p1 := g.Power()

	g.Reset()
	g.Update(realTone(gzN, targetHz, 2.0))
	p2 := g.Power()

	ratio := p2 / p1
	if math.Abs(ratio-4.0) > 0.02 {
		t.Errorf("doubling amplitude gave power ratio %.5f, want 4.0 (+/-0.02)", ratio)
	}
}

// TestGoertzelProcessEqualsUpdate pins Update as bit-for-bit identical to a
// per-sample Process loop -- same float64 arithmetic in the same order, just
// register-hoisted. Any divergence would signal the hoist changed the
// computation.
func TestGoertzelProcessEqualsUpdate(t *testing.T) {
	targetHz := gzTargetHz()
	x := realTone(gzN, targetHz, 0.7)

	gu := NewGoertzel(targetHz, gzFs, gzN)
	gu.Update(x)

	gp := NewGoertzel(targetHz, gzFs, gzN)
	for _, s := range x {
		gp.Process(s)
	}

	if gu.Power() != gp.Power() {
		t.Errorf("Update power %v != Process power %v (must be bit-identical)", gu.Power(), gp.Power())
	}
	if gu.Complex() != gp.Complex() {
		t.Errorf("Update complex %v != Process complex %v (must be bit-identical)", gu.Complex(), gp.Complex())
	}
}

// TestGoertzelReset confirms Reset zeroes state so a subsequent block is
// independent of the prior one.
func TestGoertzelReset(t *testing.T) {
	targetHz := gzTargetHz()
	g := NewGoertzel(targetHz, gzFs, gzN)

	g.Update(realTone(gzN, targetHz, 1.0))
	if g.Power() == 0 {
		t.Fatal("expected nonzero power after feeding an on-target tone")
	}
	g.Reset()
	if g.Power() != 0 {
		t.Errorf("Power after Reset = %v, want 0", g.Power())
	}

	// A fresh block after Reset must equal a brand-new detector's result.
	g.Update(realTone(gzN, targetHz, 1.0))
	fresh := NewGoertzel(targetHz, gzFs, gzN)
	fresh.Update(realTone(gzN, targetHz, 1.0))
	if g.Power() != fresh.Power() {
		t.Errorf("post-Reset power %v != fresh detector power %v", g.Power(), fresh.Power())
	}
}

// TestGoertzelPartialFeedDiffersFromFull documents that the "feed exactly
// blockLen samples" contract is the caller's to enforce: neither Update nor
// Process validates the sample count, so a short block (blockLen-1 samples)
// accumulates different energy than a full block and yields a different Power().
// This pins the normalization to sample count so a future refactor cannot
// silently change how many samples the readout depends on.
func TestGoertzelPartialFeedDiffersFromFull(t *testing.T) {
	targetHz := gzTargetHz()
	full := realTone(gzN, targetHz, 1.0)

	gFull := NewGoertzel(targetHz, gzFs, gzN)
	gFull.Update(full)

	gPartial := NewGoertzel(targetHz, gzFs, gzN)
	gPartial.Update(full[:gzN-1]) // one sample short

	if gPartial.Power() == gFull.Power() {
		t.Errorf("partial feed (%d samples) gave same Power() %v as full feed (%d samples); "+
			"sample count must affect the readout", gzN-1, gFull.Power(), gzN)
	}
}

// TestGoertzelBinSnapping documents and checks the snapping behavior: a target
// between bins snaps to the nearest integer bin, and BinFreq reports that bin
// center.
func TestGoertzelBinSnapping(t *testing.T) {
	// 1003 Hz at N=1024, fs=8000 -> k = round(1003*1024/8000) = round(128.384) = 128.
	g := NewGoertzel(1003.0, gzFs, gzN)
	if g.Bin() != 128 {
		t.Errorf("Bin() = %d, want 128", g.Bin())
	}
	wantFreq := 128.0 * gzFs / float64(gzN) // 1000 Hz
	if math.Abs(g.BinFreq()-wantFreq) > 1e-9 {
		t.Errorf("BinFreq() = %v, want %v", g.BinFreq(), wantFreq)
	}
}

// TestGoertzelValidation checks the constructor's precondition panics.
func TestGoertzelValidation(t *testing.T) {
	inf := math.Inf(1)
	cases := []struct {
		name         string
		targetHz, fs float64
		blockLen     int
	}{
		{"zero target", 0, gzFs, gzN},
		{"negative target", -100, gzFs, gzN},
		{"target at nyquist", gzFs / 2, gzFs, gzN},
		{"target above nyquist", gzFs, gzFs, gzN},
		{"zero sample rate", 100, 0, gzN},
		{"zero blockLen", 100, gzFs, 0},
		// Non-finite inputs slip past every ordered comparison (NaN) or corrupt the
		// bin math (Inf), so they must be rejected explicitly.
		{"nan target", math.NaN(), gzFs, gzN},
		{"inf target", inf, gzFs, gzN},
		{"neg inf target", math.Inf(-1), gzFs, gzN},
		{"nan sample rate", 100, math.NaN(), gzN},
		{"inf sample rate", 100, inf, gzN},
		// These pass the continuous (0, fs/2) guard but SNAP to a forbidden bin:
		// 1 Hz at N=1024,fs=8000 -> k=round(0.128)=0 (DC); 3999 Hz -> k=round(511.87)
		// =512=N/2 (Nyquist); blockLen=2 leaves no representable bin (1..N/2-1 empty).
		{"snaps to DC bin", 1.0, gzFs, gzN},
		{"snaps to nyquist bin", 3999.0, gzFs, gzN},
		{"tiny blockLen", 1000, gzFs, 2},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			defer func() {
				if recover() == nil {
					t.Errorf("expected panic for %s", tc.name)
				}
			}()
			NewGoertzel(tc.targetHz, tc.fs, tc.blockLen)
		})
	}
}
