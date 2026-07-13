package dsp

import "math"

const (
	// snrTau is the EWM smoothing time constant (seconds) for the SNR
	// estimator's moments. 2 s keeps the reported value stable but responsive.
	snrTau = 2.0

	// snrMaxDB caps the reported SNR so a (near-)noiseless block doesn't
	// divide by zero.
	snrMaxDB = 60.0
)

// CMSNREstimator estimates the SNR of a constant-modulus signal (e.g. P25
// C4FM) buried in complex Gaussian noise, using the M2M4 moment method. C4FM
// is constant-envelope, so additive noise shows up as amplitude jitter on the
// envelope. For a constant-modulus signal (kurtosis ka=1) plus complex
// Gaussian noise (kw=2):
//
//	M2 = E[|r|^2] = S + N
//	M4 = E[|r|^4] = S^2 + 4*S*N + 2*N^2
//	S  = sqrt(2*M2^2 - M4)   (exact: 2*M2^2 - M4 == S^2)
//	N  = M2 - S
//
// Both moments are EWM-smoothed across blocks.
type CMSNREstimator struct {
	m2, m4      float64
	alpha       float64
	initialized bool
}

// NewCMSNREstimator returns an estimator smoothed with snrTau at the channel
// block rate (ChannelIQRate / BlockSamples blocks per second).
func NewCMSNREstimator() *CMSNREstimator {
	blockRate := float64(ChannelIQRate) / float64(BlockSamples)
	return &CMSNREstimator{
		alpha: 1.0 - math.Exp(-1.0/(snrTau*blockRate)),
	}
}

// Update folds one block's raw 2nd and 4th envelope moments into the smoothed
// estimate.
func (e *CMSNREstimator) Update(samples []complex64) {
	if len(samples) == 0 {
		return
	}
	var s2, s4 float64
	for _, s := range samples {
		re := float64(real(s))
		im := float64(imag(s))
		p := re*re + im*im // |r|^2
		s2 += p
		s4 += p * p
	}
	n := float64(len(samples))
	m2 := s2 / n
	m4 := s4 / n
	if !e.initialized {
		e.m2, e.m4 = m2, m4
		e.initialized = true
		return
	}
	e.m2 = e.alpha*m2 + (1.0-e.alpha)*e.m2
	e.m4 = e.alpha*m4 + (1.0-e.alpha)*e.m4
}

// PowerDB is the total in-channel power 10*log10(M2). This matches a
// conventional RMS power measurement, since 20*log10(rms) = 10*log10(M2).
func (e *CMSNREstimator) PowerDB() float64 {
	if e.m2 <= 0 {
		return -200.0
	}
	return 10.0 * math.Log10(e.m2)
}

// NoiseDB is the estimated noise power 10*log10(N).
func (e *CMSNREstimator) NoiseDB() float64 {
	n := e.noisePower()
	if n <= 0 {
		return -200.0
	}
	return 10.0 * math.Log10(n)
}

// SNRdB is PowerDB - NoiseDB: the (S+N)/N ratio in dB.
func (e *CMSNREstimator) SNRdB() float64 {
	return e.PowerDB() - e.NoiseDB()
}

// noisePower returns N, clamped so SNR never exceeds snrMaxDB.
func (e *CMSNREstimator) noisePower() float64 {
	s2 := 2.0*e.m2*e.m2 - e.m4
	if s2 < 0 {
		s2 = 0
	}
	sig := math.Sqrt(s2)
	n := e.m2 - sig
	floor := e.m2 * math.Pow(10, -snrMaxDB/10.0)
	if n < floor {
		n = floor
	}
	return n
}
