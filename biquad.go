package dsp

import "math"

// Biquad is a single second-order IIR section in Direct Form I.
type Biquad struct {
	b0, b1, b2 float32
	a1, a2     float32
	x1, x2     float32
	y1, y2     float32
}

func (b *Biquad) Process(x float32) float32 {
	y := b.b0*x + b.b1*b.x1 + b.b2*b.x2 - b.a1*b.y1 - b.a2*b.y2
	b.x2, b.x1 = b.x1, x
	b.y2, b.y1 = b.y1, y
	return y
}

func (b *Biquad) Reset() {
	b.x1, b.x2, b.y1, b.y2 = 0, 0, 0, 0
}

// BiquadCascade chains second-order sections in series.
type BiquadCascade struct {
	sections []Biquad
}

func (c *BiquadCascade) Process(x float32) float32 {
	y := x
	for i := range c.sections {
		y = c.sections[i].Process(y)
	}
	return y
}

func (c *BiquadCascade) Reset() {
	for i := range c.sections {
		c.sections[i].Reset()
	}
}

// DesignButterworthHPF returns an Nth-order Butterworth high-pass as
// order/2 cascaded biquads. order must be a positive even integer.
//
// Coefficients use the RBJ Audio EQ Cookbook biquad form. Each section's
// Q is set to the analog Butterworth pole Q for that section so the
// cascaded response is maximally flat in the passband.
func DesignButterworthHPF(cutoff, sampleRate float64, order int) *BiquadCascade {
	if order < 2 || order%2 != 0 {
		panic("DesignButterworthHPF: order must be a positive even integer")
	}
	// Precondition: 0 < cutoff < sampleRate/2 (Nyquist). A cutoff at or above
	// Nyquist warps to an out-of-band pole and yields an unstable section.
	if cutoff <= 0 || cutoff >= sampleRate/2 {
		panic("DesignButterworthHPF: cutoff must satisfy 0 < cutoff < sampleRate/2")
	}
	nSections := order / 2
	sections := make([]Biquad, nSections)
	omega := 2.0 * math.Pi * cutoff / sampleRate
	cosW := math.Cos(omega)
	sinW := math.Sin(omega)
	cosP1 := 1.0 + cosW
	for k := 0; k < nSections; k++ {
		// Pole Q for the k-th section of an Nth-order Butterworth cascade.
		kk := float64(k) + 1.0
		q := 1.0 / (2.0 * math.Sin(math.Pi*(2.0*kk-1.0)/(2.0*float64(order))))
		alpha := sinW / (2.0 * q)
		b0 := cosP1 / 2.0
		b1 := -cosP1
		b2 := cosP1 / 2.0
		a0 := 1.0 + alpha
		a1 := -2.0 * cosW
		a2 := 1.0 - alpha
		sections[k] = Biquad{
			b0: float32(b0 / a0),
			b1: float32(b1 / a0),
			b2: float32(b2 / a0),
			a1: float32(a1 / a0),
			a2: float32(a2 / a0),
		}
	}
	return &BiquadCascade{sections: sections}
}

// DesignNotch returns a second-order IIR notch (band-reject) filter centered
// on centerFreq with the given Q factor. Higher Q gives a narrower notch.
// Q=10 at 233.6 Hz gives a 23 Hz bandwidth — wide enough to cover the ±1.5%
// CTCSS tolerance while leaving the voice band untouched.
//
// Coefficients follow the RBJ Audio EQ Cookbook "notch" form.
func DesignNotch(centerFreq, q, sampleRate float64) *Biquad {
	// Precondition: q > 0 and 0 < centerFreq < sampleRate/2 (Nyquist).
	// q == 0 divides by zero in alpha, yielding Inf/NaN coefficients; a center
	// frequency at or above Nyquist places the notch outside the representable
	// band and produces an unstable/aliased response.
	if q <= 0 {
		panic("DesignNotch: q must be > 0")
	}
	if centerFreq <= 0 || centerFreq >= sampleRate/2 {
		panic("DesignNotch: centerFreq must satisfy 0 < centerFreq < sampleRate/2")
	}
	omega := 2.0 * math.Pi * centerFreq / sampleRate
	alpha := math.Sin(omega) / (2.0 * q)
	cosW := math.Cos(omega)
	a0 := 1.0 + alpha
	return &Biquad{
		b0: float32(1.0 / a0),
		b1: float32(-2.0 * cosW / a0),
		b2: float32(1.0 / a0),
		a1: float32(-2.0 * cosW / a0),
		a2: float32((1.0 - alpha) / a0),
	}
}

// DesignButterworthLPF returns an Nth-order Butterworth low-pass as
// order/2 cascaded biquads. order must be a positive even integer.
func DesignButterworthLPF(cutoff, sampleRate float64, order int) *BiquadCascade {
	if order < 2 || order%2 != 0 {
		panic("DesignButterworthLPF: order must be a positive even integer")
	}
	// Precondition: 0 < cutoff < sampleRate/2 (Nyquist). A cutoff at or above
	// Nyquist warps to an out-of-band pole and yields an unstable section.
	if cutoff <= 0 || cutoff >= sampleRate/2 {
		panic("DesignButterworthLPF: cutoff must satisfy 0 < cutoff < sampleRate/2")
	}
	nSections := order / 2
	sections := make([]Biquad, nSections)
	omega := 2.0 * math.Pi * cutoff / sampleRate
	cosW := math.Cos(omega)
	sinW := math.Sin(omega)
	cosM1 := 1.0 - cosW
	for k := 0; k < nSections; k++ {
		kk := float64(k) + 1.0
		q := 1.0 / (2.0 * math.Sin(math.Pi*(2.0*kk-1.0)/(2.0*float64(order))))
		alpha := sinW / (2.0 * q)
		b0 := cosM1 / 2.0
		b1 := cosM1
		b2 := cosM1 / 2.0
		a0 := 1.0 + alpha
		a1 := -2.0 * cosW
		a2 := 1.0 - alpha
		sections[k] = Biquad{
			b0: float32(b0 / a0),
			b1: float32(b1 / a0),
			b2: float32(b2 / a0),
			a1: float32(a1 / a0),
			a2: float32(a2 / a0),
		}
	}
	return &BiquadCascade{sections: sections}
}
