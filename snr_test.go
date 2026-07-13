package dsp

import (
	"math"
	"math/cmplx"
	"math/rand"
	"testing"
)

// feedCMSignal pushes ~2 s of unit-modulus (signal power = 1) samples plus
// complex Gaussian noise of the given per-component sigma through the estimator,
// in 250-sample blocks, using a fixed seed for determinism.
func feedCMSignal(e *CMSNREstimator, sigma float64) {
	r := rand.New(rand.NewSource(1))
	const blocks = 400 // 4 s — well past the 2 s tau
	block := make([]complex64, BlockSamples)
	for b := 0; b < blocks; b++ {
		for i := range block {
			phase := 2 * math.Pi * r.Float64()
			s := cmplx.Exp(complex(0, phase)) // |s| = 1
			n := complex(r.NormFloat64()*sigma, r.NormFloat64()*sigma)
			block[i] = complex64(s + n)
		}
		e.Update(block)
	}
}

func TestCMSNREstimator_RecoversKnownSNR(t *testing.T) {
	// noise power N = 2*sigma^2 (two real components). Displayed SNR uses the
	// (S+N)/N convention (matches the voice rows: total power minus noise floor).
	cases := []float64{5, 15, 30}
	for _, wantDB := range cases {
		// want = 10log10((1+N)/N)  =>  solve for N, then sigma.
		ratio := math.Pow(10, wantDB/10) // (1+N)/N
		N := 1.0 / (ratio - 1.0)
		sigma := math.Sqrt(N / 2.0)
		e := NewCMSNREstimator()
		feedCMSignal(e, sigma)
		got := e.SNRdB()
		if math.Abs(got-wantDB) > 1.5 {
			t.Errorf("SNR %.0f dB: got %.2f dB (want within 1.5)", wantDB, got)
		}
	}
}

func TestCMSNREstimator_PureNoiseIsZeroDB(t *testing.T) {
	e := NewCMSNREstimator()
	r := rand.New(rand.NewSource(2))
	block := make([]complex64, BlockSamples)
	for b := 0; b < 400; b++ {
		for i := range block {
			block[i] = complex64(complex(r.NormFloat64(), r.NormFloat64()))
		}
		e.Update(block)
	}
	if got := e.SNRdB(); got > 2.0 {
		t.Errorf("pure noise SNR: got %.2f dB (want ~0)", got)
	}
}

func TestCMSNREstimator_PureSignalIsClamped(t *testing.T) {
	e := NewCMSNREstimator()
	feedCMSignal(e, 0) // no noise
	if got := e.SNRdB(); got < 50 {
		t.Errorf("pure signal SNR: got %.2f dB (want clamped high)", got)
	}
}
