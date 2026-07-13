package dsp

// RationalResampler implements a polyphase rational resampler.
// Converts from one sample rate to another using integer up/down factors.
// For 25 kSPS → 8 kHz: upsample by 8, downsample by 25.
type RationalResampler struct {
	upFactor   int
	downFactor int
	filter     []float32 // prototype low-pass filter
	filterLen  int
	history    []float32
	histPos    int // ring write index into history
	phase      int // current phase in [0, downFactor)
}

// NewRationalResampler creates a resampler for the given up/down factors.
// The prototype filter is designed as a low-pass at min(1/up, 1/down) * Nyquist.
func NewRationalResampler(upFactor, downFactor int) *RationalResampler {
	// Design the prototype anti-aliasing filter at the upsampled rate.
	// Cutoff is at the lower of the two Nyquist rates.
	// For up=8, down=25: cutoff = 1/(2*25) of the upsampled rate = 0.02
	// But we need it in terms of the upsampled sample rate.
	// Upsampled rate = inputRate * upFactor.
	// Output Nyquist = outputRate/2 = inputRate * upFactor / (2 * downFactor)
	// Cutoff fraction = 1 / (2 * max(upFactor, downFactor))
	//
	// For 25000*8 = 200000 intermediate rate:
	//   output rate = 200000/25 = 8000
	//   cutoff = 4000 Hz at 200000 Hz rate = 0.02
	//   We design at normalized frequency.
	cutoff := 1.0 / (2.0 * float64(max(upFactor, downFactor)))

	// Filter length: ~4 * max(up,down) * upFactor taps for decent quality
	filterLen := 4 * max(upFactor, downFactor) * upFactor
	if filterLen < 64 {
		filterLen = 64
	}
	if filterLen%2 == 0 {
		filterLen++
	}

	// Design using window-sinc at the upsampled rate
	// Cutoff in terms of normalized frequency (0.5 = Nyquist of upsampled rate)
	filter := DesignLPF(cutoff, 1.0, filterLen)

	// Scale by upFactor to compensate for zero-insertion gain loss
	for i := range filter {
		filter[i] *= float32(upFactor)
	}

	return &RationalResampler{
		upFactor:   upFactor,
		downFactor: downFactor,
		filter:     filter,
		filterLen:  filterLen,
		history:    make([]float32, (filterLen+upFactor-1)/upFactor+1),
		phase:      0,
	}
}

// Process resamples the input and returns the output at the new rate.
func (r *RationalResampler) Process(input []float32) []float32 {
	out := make([]float32, 0, len(input)*r.upFactor/r.downFactor+1)

	histLen := len(r.history)
	hist := r.history
	filt := r.filter
	flen := r.filterLen
	up := r.upFactor

	for _, s := range input {
		// Ring-buffer push: overwrite the oldest sample. r.histPos always
		// points at the slot that will receive the NEXT input; equivalently,
		// the slot just behind it (histPos-1 mod histLen) is the newest.
		hist[r.histPos] = s
		newest := r.histPos
		r.histPos++
		if r.histPos == histLen {
			r.histPos = 0
		}

		for r.phase < up {
			// Polyphase MAC: walk backward from the newest sample, stepping
			// the filter index by upFactor each tap. j wraps modulo histLen.
			var acc float32
			j := newest
			for sub := r.phase; sub < flen; sub += up {
				acc += hist[j] * filt[sub]
				if j == 0 {
					j = histLen
				}
				j--
			}
			out = append(out, acc)
			r.phase += r.downFactor
		}
		r.phase -= up
	}

	return out
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
