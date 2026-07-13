package dsp

import "math"

// oscMaxPeriod caps the precomputed period table; offsets whose period exceeds
// this fall back to the running phasor recurrence. At 50 MSPS on the 6.25 kHz
// channel grid the worst-case period is 50e6/6250 = 8000, well under this.
const oscMaxPeriod = 1 << 16

// Oscillator generates a complex exponential for frequency shifting.
type Oscillator struct {
	// Fallback recurrence state (used when no period table was built).
	phasor complex64 // current e^(j·phase), kept on the unit circle
	step   complex64 // e^(j·phaseIncrement), multiplied each sample

	// Precomputed one-period tables (fast path). period == 0 means "use recurrence".
	periodCos []float32
	periodSin []float32
	period    int // == len(periodCos)
	idx       int // running position within the period, advanced per block

	// Scratch tables for the recurrence-path Mix apply pass. The period-table
	// fast path reads periodCos/periodSin directly and does not use these.
	cosTable []float32
	sinTable []float32
}

// NewOscillator creates an oscillator producing exp(-j·2π·freqOffset·t).
func NewOscillator(freqOffset float64, sampleRate float64) *Oscillator {
	inc := -2.0 * math.Pi * freqOffset / sampleRate
	o := &Oscillator{
		phasor: complex(float32(1), float32(0)),
		step:   complex(float32(math.Cos(inc)), float32(math.Sin(inc))),
	}
	o.buildPeriodTable(freqOffset, sampleRate)
	return o
}

func gcdInt64(a, b int64) int64 {
	if a < 0 {
		a = -a
	}
	if b < 0 {
		b = -b
	}
	for b != 0 {
		a, b = b, a%b
	}
	return a
}

// buildPeriodTable precomputes one full period of cos/sin when freqOffset and
// sampleRate are integer Hz with a tractable period q = sampleRate/gcd(sampleRate,
// |freqOffset|). After q samples the phasor returns exactly to 1, so the sequence
// repeats with period q. Leaves period == 0 (recurrence fallback) for non-integer
// offsets or q > oscMaxPeriod.
func (o *Oscillator) buildPeriodTable(freqOffset, sampleRate float64) {
	off := math.Round(freqOffset)
	rate := math.Round(sampleRate)
	if rate <= 0 || math.Abs(off-freqOffset) > 1e-6 || math.Abs(rate-sampleRate) > 1e-6 {
		return
	}
	g := gcdInt64(int64(rate), int64(off)) // gcd(rate,0) == rate
	if g == 0 {
		return
	}
	q := int64(rate) / g
	if q <= 0 || q > oscMaxPeriod {
		return
	}
	inc := -2.0 * math.Pi * freqOffset / sampleRate
	o.periodCos = make([]float32, q)
	o.periodSin = make([]float32, q)
	for k := int64(0); k < q; k++ {
		ph := inc * float64(k)
		o.periodCos[k] = float32(math.Cos(ph))
		o.periodSin[k] = float32(math.Sin(ph))
	}
	o.period = int(q)
}

// fillTables produces cos/sin for the next n sample positions into the reusable
// scratch tables by advancing the phasor recurrence (loop-carried dependency,
// cannot vectorize) and renormalising once per block. Only the recurrence path
// (non-integer or long-period offsets) uses this; the period-table fast path
// reads periodCos/periodSin directly in Mix.
func (o *Oscillator) fillTables(n int) (cosT, sinT []float32) {
	if cap(o.cosTable) < n {
		o.cosTable = make([]float32, n)
		o.sinTable = make([]float32, n)
	}
	cosT = o.cosTable[:n]
	sinT = o.sinTable[:n]

	// Running phasor recurrence (non-integer or long-period offsets).
	p := o.phasor
	for i := range cosT {
		cosT[i] = real(p)
		sinT[i] = imag(p)
		p *= o.step
	}
	// Renormalise once per block to prevent phasor magnitude drift.
	invMag := float32(1.0 / math.Sqrt(float64(real(p)*real(p)+imag(p)*imag(p))))
	o.phasor = complex(real(p)*invMag, imag(p)*invMag)
	return cosT, sinT
}

// Mix multiplies each sample of in by the oscillator's complex exponential
// and writes the result to out, shifting freqOffset to baseband. in is not
// modified. out must have len(out) >= len(in); only the first len(in)
// elements of out are written. in and out may be the same slice.
//
// The period-table fast path reads the precomputed one-period cos/sin tables
// directly with wraparound. The recurrence fallback (non-integer or long-period
// offsets) records cos/sin into scratch tables via fillTables, then applies the
// shift — a loop-carried phasor dependency that cannot be vectorized in one pass.
func (o *Oscillator) Mix(in, out []complex64) {
	n := len(in)
	if n == 0 {
		return
	}
	_ = out[n-1] // bounds check: len(out) >= len(in)

	if o.period > 0 {
		// Fast path: read the precomputed one-period table directly with
		// wraparound, advancing idx exactly as the copy path did. s is a copy
		// of in[i], so this stays correct when in and out alias (MixInPlace).
		p := o.period
		idx := o.idx
		for i, s := range in {
			re := real(s)
			im := imag(s)
			cos := o.periodCos[idx]
			sin := o.periodSin[idx]
			out[i] = complex(re*cos-im*sin, re*sin+im*cos)
			idx++
			if idx == p {
				idx = 0
			}
		}
		o.idx = idx
		return
	}

	cosT, sinT := o.fillTables(n)

	// Apply the frequency shift. Each output depends only on in[i], cosT[i],
	// sinT[i] — no loop-carried dependencies — so the compiler can vectorize.
	for i, s := range in {
		re := real(s)
		im := imag(s)
		out[i] = complex(re*cosT[i]-im*sinT[i], re*sinT[i]+im*cosT[i])
	}
}

// MixInPlace multiplies each sample by the oscillator's complex exponential,
// shifting the frequency in-place. This moves freqOffset to baseband.
func (o *Oscillator) MixInPlace(samples []complex64) {
	o.Mix(samples, samples)
}
