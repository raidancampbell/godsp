package dsp

import "math"

// CarrierPED selects the phase-error detector (PED) a Costas loop uses to turn a
// de-rotated complex sample into a scalar phase-error estimate. The three
// detectors correspond to three signal structures; choosing the wrong one for
// the modulation on the wire produces an S-curve with the wrong shape (or the
// wrong sign) and the loop either fails to lock or locks to a false point.
//
//   - CostasPLL uses e = atan2(Q, I). This is a plain phase-locked loop: it
//     assumes there is NO data modulation, only a residual carrier (an
//     unmodulated tone or a transmitted pilot). atan2 is a wide, linear-range
//     detector -- its S-curve is a straight line e = pe over the whole interval
//     [-pi, pi] -- so it has the largest pull-in range of the three, but it
//     cannot be used on a modulated signal because the data itself would look
//     like phase error.
//   - CostasBPSK uses e = I*Q. Squaring the in-phase and quadrature rails
//     removes a binary (0 / pi) data phase, so the detector is blind to a BPSK
//     bit flip and tracks only the carrier. Its S-curve is e = (A^2/2)*sin(2*pe),
//     which is linear near the origin and periodic with period pi -- hence the
//     residual pi phase ambiguity (see the type doc callout).
//   - CostasQPSK uses e = sign(I)*Q - sign(Q)*I, the classic decision-directed
//     QPSK Costas detector. It removes a quaternary (k*pi/2) data phase and
//     therefore leaves a pi/2 ambiguity. Its S-curve passes through zero with
//     positive slope when the reference constellation point sits on the 45-degree
//     diagonal (I == Q), which is where an ideal QPSK symbol lives.
type CarrierPED int

const (
	// CostasPLL: e = atan2(Q, I) -- unmodulated carrier / pilot tone.
	CostasPLL CarrierPED = iota
	// CostasBPSK: e = I*Q -- pi phase ambiguity.
	CostasBPSK
	// CostasQPSK: e = sign(I)*Q - sign(Q)*I -- pi/2 phase ambiguity.
	CostasQPSK
)

// String renders the detector name for readable test logs and error messages.
func (p CarrierPED) String() string {
	switch p {
	case CostasPLL:
		return "PLL"
	case CostasBPSK:
		return "BPSK"
	case CostasQPSK:
		return "QPSK"
	default:
		return "CarrierPED(?)"
	}
}

// costasLockEps guards the lock-metric normalization u = y/|y| against a
// near-zero sample (e.g. a pulse-shaped stream passing through a zero crossing),
// where dividing by |y| would produce a garbage unit phasor. Samples below this
// magnitude simply do not update the lock EMA.
const costasLockEps = 1e-12

// costasMaxTrackRad is the anti-windup clamp on the loop-filter integrator, in
// radians/sample. The integrator IS the loop's estimate of the residual carrier
// frequency (in rad/sample), so clamping |integ| <= pi/2 bounds the maximum
// trackable |FreqOffset| to (pi/2)*fs/(2*pi) = fs/4 Hz and prevents a runaway
// during acquisition of an offset beyond the loop's pull-in range. pi/2
// rad/sample is a quarter of the sample rate -- far larger than any residual a
// Costas loop is meant to track -- so in normal operation the clamp never fires;
// it exists only to keep a pathological input from winding the integrator to
// infinity (see the anti-windup test).
const costasMaxTrackRad = math.Pi / 2

// Costas is a decision-directed carrier phase/frequency recovery loop: a
// numerically-controlled oscillator (NCO) whose phase is steered every sample by
// a type-2 PI LoopFilter driven by a phase-error detector. It removes a residual
// carrier phase and frequency offset from a complex baseband stream, de-rotating
// each input sample onto the constellation.
//
// The per-sample recurrence (each output feeds the next NCO step, so this is an
// inherently serial loop -- a tight scalar loop, never SIMD) is:
//
//	(cos, sin) = CordicRotate(theta)          // table-free NCO sample at phase theta
//	yR = rR*cos + rI*sin                       // de-rotate: y = r * exp(-j*theta)
//	yI = rI*cos - rR*sin
//	e  = PED(yR, yI)                           // phase-error estimate (see CarrierPED)
//	v  = loopFilter.Advance(e)                 // PI correction (rad/sample)
//	out = y                                    // de-rotated sample
//	theta += w0 + v                            // advance the NCO
//	wrap theta into [-pi, pi]
//
// De-rotation multiplies the input r by exp(-j*theta): expanding
// r*exp(-j*theta) = (rR + j*rI)(cos - j*sin) gives
// Re = rR*cos + rI*sin and Im = rI*cos - rR*sin, which is the yR/yI above.
//
// Sign convention (the number-one loop bug -- pinned by the S-curve test, not by
// trusting this comment): near lock every PED returns e ~ +Kd*(phi - theta),
// i.e. a POSITIVE error when the input phase phi leads the NCO phase theta. The
// NCO update theta += w0 + v with v ~ +e then advances theta toward phi, closing
// the gap -- net NEGATIVE feedback. If the PED sign were flipped the loop would
// run away instead of locking; the open-loop S-curve test asserts e(0) ~ 0 with
// POSITIVE slope through the origin precisely so that a sign error fails loudly.
//
// Lock metric (uniform, amplitude-independent): the de-rotated sample is
// normalized to a unit phasor u = y/|y|, its data modulation is stripped by
// raising it to the appropriate power (PLL z = u, BPSK z = u^2, QPSK z = u^4),
// and z is smoothed into a complex EMA cAvg. When the loop is locked the stripped
// phasor is a constant direction so |cAvg| -> 1; when it is unlocked (rotating)
// or noise-dominated the stripped phasor spins and averages toward 0. LockQuality
// returns |cAvg| in [0, 1] and Locked reports |cAvg| > 0.9.
//
// The NCO phase accumulator, the loop-filter integrator, and the lock EMA are all
// carried in float64/complex128 on purpose: the integrator sums tiny per-sample
// increments (k2 is O(bnT^2)) over potentially millions of samples, and the phase
// accumulator must not lose low bits to float32 rounding before it is handed to
// CordicRotate. Only the de-rotated OUTPUT is narrowed back to complex64.
//
// Easy-to-get-wrong, stated precisely: a Costas loop tracks only a SMALL residual
// carrier offset. It is not an acquisition device. You MUST do coarse frequency
// acquisition first (an FFT peak or an FLL), hand the estimate to SetFreq, and
// only then let Costas clean up the remainder; feeding it a large raw offset just
// saturates the anti-windup clamp. Two more caller responsibilities:
//   - Phase ambiguity is NOT resolved here. BPSK leaves a pi ambiguity and QPSK a
//     pi/2 ambiguity (the detector is deliberately blind to the data phase). The
//     caller must resolve it downstream -- differential decoding, or a Correlator
//     against a known unique word / preamble.
//   - Put an IQAGC upstream. The BPSK/QPSK detector gains scale with input
//     amplitude (Kd ~ A^2), so a varying envelope retunes the loop bandwidth;
//     holding the amplitude roughly constant keeps Kd ~ 1 as designed.
//
// Costas is not safe for concurrent use.
type Costas struct {
	theta float64     // NCO phase in radians, wrapped to [-pi, pi]; carried state
	w0    float64     // nominal center frequency in rad/sample (0 for Costas)
	lf    *LoopFilter // type-2 PI loop filter; its integrator is the freq estimate
	ped   CarrierPED  // which phase-error detector to run
	fs    float64     // sample rate in Hz, for FreqOffset unit conversion
	alpha float64     // lock-metric EMA smoothing factor in [1e-4, 0.5]
	cAvg  complex128  // stripped-modulation phasor EMA; |cAvg| is LockQuality

	// outBuf is a reusable output buffer for ProcessReuse. Its contents are only
	// valid until the next call to Process or ProcessReuse on this loop.
	outBuf []complex64
}

// NewCostas builds a carrier-recovery loop with the given one-sided loop noise
// bandwidth loopBandwidthHz on a stream sampled at sampleRate, using the phase-
// error detector selected by ped. The loop filter is a type-2 PI stage with
// damping zeta = 0.707 (maximally flat) and detector/oscillator gains kd = k0 = 1
// (the detectors are normalized to unit small-signal slope and an upstream AGC is
// assumed to hold that gain); the normalized bandwidth handed to the loop filter
// is bnT = loopBandwidthHz/sampleRate, the PER-SAMPLE convention. An anti-windup
// clamp of pi/2 rad/sample is installed on the integrator (see costasMaxTrackRad).
//
// It panics on arguments that make the loop nonsensical: sampleRate <= 0,
// loopBandwidthHz <= 0, or an unrecognized ped.
func NewCostas(loopBandwidthHz, sampleRate float64, ped CarrierPED) *Costas {
	if sampleRate <= 0 {
		panic("NewCostas: sampleRate must be positive")
	}
	if loopBandwidthHz <= 0 {
		panic("NewCostas: loopBandwidthHz must be positive")
	}
	if ped != CostasPLL && ped != CostasBPSK && ped != CostasQPSK {
		panic("NewCostas: unknown CarrierPED")
	}

	bnT := loopBandwidthHz / sampleRate
	lf := NewLoopFilter(bnT, 0.707, 1, 1)
	lf.SetClamp(costasMaxTrackRad)

	// The lock-metric EMA tracks the loop bandwidth: fast enough to reflect the
	// loop's own settling, clamped so a very small or very large bandwidth does
	// not make the metric useless (frozen or all-noise).
	alpha := bnT
	if alpha < 1e-4 {
		alpha = 1e-4
	} else if alpha > 0.5 {
		alpha = 0.5
	}

	return &Costas{
		lf:    lf,
		ped:   ped,
		fs:    sampleRate,
		alpha: alpha,
	}
}

// carrierPED evaluates the selected phase-error detector on one de-rotated
// sample (yR = I, yI = Q). It is a free function so the open-loop S-curve test
// can exercise each detector directly. All three are normalized so that near the
// lock point e ~ +Kd*(phi - theta) with Kd ~ 1, giving positive S-curve slope
// through the origin (the shared sign convention documented on Costas):
//
//	PLL : e = atan2(Q, I)                 -> e = pe, slope 1 over [-pi, pi]
//	BPSK: e = I*Q                          -> e = (A^2/2) sin(2*pe), slope 1 at pe=0
//	QPSK: e = sign(I)*Q - sign(Q)*I        -> slope > 0 at an I==Q lock point
//
// sign(0) is taken as +1 (see costasSign).
func carrierPED(ped CarrierPED, yR, yI float32) float64 {
	switch ped {
	case CostasBPSK:
		return float64(yR) * float64(yI)
	case CostasQPSK:
		return costasSign(yR)*float64(yI) - costasSign(yI)*float64(yR)
	default: // CostasPLL
		return float64(atan2Approx(yI, yR))
	}
}

// costasSign returns +1 for x >= 0 and -1 for x < 0. Taking sign(0) = +1 (rather
// than 0) keeps the QPSK detector well-defined on the axes and matches the
// slicer convention used elsewhere.
func costasSign(x float32) float64 {
	if x < 0 {
		return -1
	}
	return 1
}

// process runs the loop over in, writing the de-rotated samples to out. out may
// alias in (ProcessInPlace passes io for both): output[i] depends only on
// input[i] and the loop state carried into that sample, so writing out[i] after
// reading in[i] is safe under aliasing. len(out) must equal len(in).
//
// State is hoisted into locals for the duration of the loop (the biquad/Goertzel
// pattern) and written back once at the end, so this allocates nothing.
func (c *Costas) process(in, out []complex64) {
	theta := c.theta
	w0 := c.w0
	lf := c.lf
	ped := c.ped
	alpha := c.alpha
	cR := real(c.cAvg)
	cI := imag(c.cAvg)

	for i := range in {
		r := in[i]
		rR := real(r)
		rI := imag(r)

		cos, sin := CordicRotate(float32(theta))
		// De-rotate: y = r * exp(-j*theta).
		yR := rR*cos + rI*sin
		yI := rI*cos - rR*sin

		e := carrierPED(ped, yR, yI)
		v := lf.Advance(e)

		out[i] = complex(yR, yI)

		// Lock metric: normalize to a unit phasor, strip the data modulation, and
		// smooth into the complex EMA. Skip the update near a zero crossing where
		// the unit phasor would be meaningless.
		mag := math.Hypot(float64(yR), float64(yI))
		if mag > costasLockEps {
			uR := float64(yR) / mag
			uI := float64(yI) / mag
			var zR, zI float64
			switch ped {
			case CostasBPSK:
				// z = u^2.
				zR = uR*uR - uI*uI
				zI = 2 * uR * uI
			case CostasQPSK:
				// z = u^4 = (u^2)^2.
				sR := uR*uR - uI*uI
				sI := 2 * uR * uI
				zR = sR*sR - sI*sI
				zI = 2 * sR * sI
			default: // CostasPLL
				zR = uR
				zI = uI
			}
			cR += alpha * (zR - cR)
			cI += alpha * (zI - cI)
		}

		// Advance the NCO and wrap. The per-sample increment |w0 + v| is bounded
		// well below pi (v is bounded by the k1*e term plus the clamped
		// integrator), so a single add/subtract keeps theta in [-pi, pi].
		theta += w0 + v
		if theta > math.Pi {
			theta -= 2 * math.Pi
		} else if theta < -math.Pi {
			theta += 2 * math.Pi
		}
	}

	c.theta = theta
	c.cAvg = complex(cR, cI)
}

// Process de-rotates the input stream and returns a freshly allocated slice of
// the same length. Loop state (NCO phase, integrator, lock EMA) carries across
// calls, so streaming in chunks yields the same result as one block.
func (c *Costas) Process(in []complex64) []complex64 {
	out := make([]complex64, len(in))
	c.process(in, out)
	return out
}

// ProcessReuse de-rotates the input stream and returns a slice backed by an
// internal buffer. The returned slice is valid only until the next call to
// Process, ProcessReuse, or ProcessInPlace on this loop. Use it on hot streaming
// paths to avoid a per-call allocation.
func (c *Costas) ProcessReuse(in []complex64) []complex64 {
	if cap(c.outBuf) < len(in) {
		c.outBuf = make([]complex64, len(in))
	}
	out := c.outBuf[:len(in)]
	c.process(in, out)
	return out
}

// ProcessInPlace de-rotates the stream in place, overwriting each element with
// its de-rotated value. De-rotation is 1:1 and depends only on the current
// sample and loop state, so in-place operation is exact.
func (c *Costas) ProcessInPlace(io []complex64) {
	c.process(io, io)
}

// SetFreq seeds the loop's frequency estimate to radPerSample (radians/sample),
// typically to hand off a coarse-acquisition result (an FFT peak or an FLL) so
// the fine loop starts already near lock. It sets the loop-filter integrator
// directly, which is the estimate the loop maintains; the value is not clamped
// here but the next Process applies the anti-windup bound.
func (c *Costas) SetFreq(radPerSample float64) {
	c.lf.SetIntegrator(radPerSample)
}

// FreqOffset returns the loop's current estimate of the carrier frequency offset
// in Hz. The estimate is the NCO center plus the loop-filter integrator (both in
// rad/sample) converted to Hz: (w0 + integ) * fs / (2*pi).
func (c *Costas) FreqOffset() float64 {
	return (c.w0 + c.lf.Integrator()) * c.fs / (2 * math.Pi)
}

// Phase returns the current NCO phase in radians, wrapped to [-pi, pi].
func (c *Costas) Phase() float64 {
	return c.theta
}

// LockQuality returns the loop lock indicator in [0, 1]: the magnitude of the
// smoothed, modulation-stripped unit phasor. It approaches 1 when the loop is
// phase-locked (the stripped phasor holds a constant direction) and falls toward
// 0 when the loop is unlocked or noise-dominated. See the Costas doc for how the
// stripping power depends on the detector.
func (c *Costas) LockQuality() float64 {
	q := cmplxAbs128Costas(c.cAvg)
	if q > 1 {
		q = 1
	}
	return q
}

// Locked reports whether the loop is phase-locked, using the documented
// heuristic LockQuality() > 0.9.
func (c *Costas) Locked() bool {
	return c.LockQuality() > 0.9
}

// Reset returns the loop to its just-constructed state: NCO phase zeroed, loop-
// filter integrator zeroed, lock EMA cleared. Configuration (detector, gains,
// clamp, sample rate, smoothing factor) is preserved.
func (c *Costas) Reset() {
	c.theta = 0
	c.cAvg = 0
	c.lf.Reset()
}

// cmplxAbs128Costas is the float64 magnitude of a complex128, used by the lock
// metric. It is block-prefixed to avoid colliding with other package helpers.
func cmplxAbs128Costas(z complex128) float64 {
	return math.Hypot(real(z), imag(z))
}
