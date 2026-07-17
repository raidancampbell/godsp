package dsp

import "math"

// DesignLPF generates a low-pass FIR filter using the window-sinc method
// with a Blackman window for ~60 dB stopband attenuation.
//   - cutoff: cutoff frequency in Hz
//   - sampleRate: sample rate in Hz
//   - numTaps: number of filter taps (must be odd for symmetric type I FIR)
func DesignLPF(cutoff, sampleRate float64, numTaps int) []float32 {
	// Precondition: 0 < cutoff < sampleRate/2 (Nyquist). A cutoff at or above
	// Nyquist exceeds the representable band; the resulting window-sinc has no
	// stopband and behaves as an unstable/aliased filter.
	if cutoff <= 0 || cutoff >= sampleRate/2 {
		panic("DesignLPF: cutoff must satisfy 0 < cutoff < sampleRate/2")
	}
	if numTaps%2 == 0 {
		numTaps++
	}

	fc := cutoff / sampleRate
	m := numTaps - 1
	taps := make([]float32, numTaps)
	sum := 0.0

	for i := 0; i < numTaps; i++ {
		n := float64(i) - float64(m)/2.0

		// Sinc function
		var h float64
		if n == 0 {
			h = 2.0 * math.Pi * fc
		} else {
			h = math.Sin(2.0*math.Pi*fc*n) / n
		}

		// Blackman window
		w := 0.42 - 0.5*math.Cos(2.0*math.Pi*float64(i)/float64(m)) +
			0.08*math.Cos(4.0*math.Pi*float64(i)/float64(m))

		taps[i] = float32(h * w)
		sum += h * w
	}

	// Normalize for unity gain at DC
	for i := range taps {
		taps[i] /= float32(sum)
	}

	return taps
}

// DecimatingFilter applies an FIR filter with decimation.
// Uses a linear overlap-save buffer with a struct-of-arrays (SoA) layout:
// real and imaginary components are stored in separate float32 slices so that
// the FIR kernel receives two plain float32 dot products. amd64 dispatches these
// to explicit generated SIMD assembly without interleaved complex64 deinterleaving.
type DecimatingFilter struct {
	taps       []float32
	decimation int
	// phase tracks how many input samples have been consumed since the last
	// output sample, mod decimation.
	phase int
	// overlapR/overlapI hold the last (nTaps-1) input samples (SoA).
	overlapR []float32
	overlapI []float32
	// bufR/bufI are reusable scratch buffers [overlap | input] (SoA), grown as needed.
	bufR []float32
	bufI []float32
	// outBuf is a reusable output buffer for ProcessReuse. Its contents are only
	// valid until the next call to Process or ProcessReuse.
	outBuf []complex64
}

func NewDecimatingFilter(taps []float32, decimation int) *DecimatingFilter {
	overlap := len(taps) - 1
	return &DecimatingFilter{
		taps:       taps,
		decimation: decimation,
		overlapR:   make([]float32, overlap),
		overlapI:   make([]float32, overlap),
	}
}

// process deinterleaves the input into the SoA scratch then filters.
func (f *DecimatingFilter) process(input []complex64, reuseOutput bool) []complex64 {
	nTaps := len(f.taps)
	overlap := nTaps - 1
	need := overlap + len(input)
	if cap(f.bufR) < need {
		f.bufR = make([]float32, need)
		f.bufI = make([]float32, need)
	}
	f.bufR = f.bufR[:need]
	f.bufI = f.bufI[:need]
	copy(f.bufR, f.overlapR)
	copy(f.bufI, f.overlapI)
	inR := f.bufR[overlap:]
	inI := f.bufI[overlap:]
	for i, c := range input {
		inR[i] = real(c)
		inI[i] = imag(c)
	}
	return f.filterStaged(len(input), reuseOutput)
}

// filterStaged runs the strided dot product over the SoA f.bufR/f.bufI kernel
// inputs, which already hold [overlap | n input samples], then updates phase
// and overlap. amd64 uses an explicit generated SIMD dot-product kernel.
func (f *DecimatingFilter) filterStaged(n int, reuseOutput bool) []complex64 {
	nTaps := len(f.taps)
	overlap := nTaps - 1
	need := overlap + n
	bufR := f.bufR[:need]
	bufI := f.bufI[:need]

	// Compute the index of the first output sample within input[].
	// phase = number of samples already consumed since last output (0..D-1).
	// The next output fires after (D - phase) more samples, i.e. at i = D-phase-1.
	start := (f.decimation - f.phase - 1) % f.decimation

	maxOut := (n - start + f.decimation - 1) / f.decimation
	if maxOut < 0 {
		maxOut = 0
	}

	// Pre-allocate the output to the exact required length so the inner loop
	// can write by index (no append growth check, no slice header mutation).
	var out []complex64
	if reuseOutput {
		if cap(f.outBuf) < maxOut {
			f.outBuf = make([]complex64, maxOut)
		}
		out = f.outBuf[:maxOut]
	} else {
		out = make([]complex64, maxOut)
	}

	taps := f.taps
	outIdx := 0
	D := f.decimation

	// Stride the outer loop by the decimation factor — only visit output positions,
	// skipping the (D-1) non-output samples between them entirely. When four output
	// windows still fit (the 4th's lane-3 window ends within the buffer), batch them
	// through complexFIRDot4 so the shared taps are loaded once per tap block; fall
	// back to the single-window kernel for the remainder.
	i := start
	for ; i < n; i += D {
		if outIdx+4 <= maxOut && i+3*D+nTaps <= need {
			var acc [8]float32
			complexFIRDot4(taps, bufR[i:], bufI[i:], D, &acc)
			out[outIdx] = complex(acc[0], acc[1])
			out[outIdx+1] = complex(acc[2], acc[3])
			out[outIdx+2] = complex(acc[4], acc[5])
			out[outIdx+3] = complex(acc[6], acc[7])
			outIdx += 4
			i += 3 * D
			continue
		}
		winR := bufR[i : i+nTaps]
		winI := bufI[i : i+nTaps]

		accR, accI := complexFIRDot(taps, winR, winI)
		out[outIdx] = complex(accR, accI)
		outIdx++
	}
	out = out[:outIdx]

	// Update phase: total input samples consumed mod decimation.
	f.phase = (f.phase + n) % f.decimation

	// Save the last (nTaps-1) samples as the overlap for the next call.
	copy(f.overlapR, bufR[need-overlap:])
	copy(f.overlapI, bufI[need-overlap:])

	if reuseOutput {
		f.outBuf = out
	}
	return out
}

// Process applies the decimating FIR filter to the input samples.
// Returns a freshly allocated decimated output slice.
func (f *DecimatingFilter) Process(input []complex64) []complex64 {
	return f.process(input, false)
}

// ProcessReuse applies the decimating FIR filter and returns a slice backed by
// an internal buffer. The returned slice is only valid until the next call to
// Process or ProcessReuse on this filter. Use this for intermediate results
// (e.g. stage-1 output fed directly into stage 2) to avoid per-call allocation.
func (f *DecimatingFilter) ProcessReuse(input []complex64) []complex64 {
	return f.process(input, true)
}

// DecimatingFilterReal applies an FIR filter with decimation to real-valued data.
// Uses a linear overlap-save buffer for cache-friendly sequential access.
type DecimatingFilterReal struct {
	taps       []float32
	decimation int
	overlap    []float32
	phase      int
	// buf is a reusable scratch buffer, grown as needed.
	buf []float32
	// outBuf is a reusable output buffer for ProcessReuse. Its contents are only
	// valid until the next call to Process or ProcessReuse.
	outBuf []float32
}

func NewDecimatingFilterReal(taps []float32, decimation int) *DecimatingFilterReal {
	return &DecimatingFilterReal{
		taps:       taps,
		decimation: decimation,
		overlap:    make([]float32, len(taps)-1),
		phase:      0,
	}
}

// DesignRRC generates a root-raised-cosine matched filter for symbol recovery.
//   - symbolRate: symbol rate in symbols/second (e.g. 4800 for P25)
//   - sampleRate: sample rate of the input signal in Hz
//   - rolloff: roll-off factor β (0 < β ≤ 1, typically 0.2 for P25 C4FM)
//   - spanSymbols: filter span in symbol periods (each side, typical 4-8)
//
// The filter is normalized so that the convolution of two identical RRC filters
// produces a raised-cosine pulse with zero ISI at symbol centers.
func DesignRRC(symbolRate, sampleRate, rolloff float64, spanSymbols int) []float32 {
	// Precondition: 0 < rolloff <= 1. A non-positive rolloff divides by zero in
	// the special-case and general-case tap formulas below, producing NaN/Inf
	// taps; a rolloff > 1 is outside the RRC definition.
	if rolloff <= 0 || rolloff > 1 {
		panic("DesignRRC: rolloff must satisfy 0 < rolloff <= 1")
	}
	sampPerSym := sampleRate / symbolRate
	numTaps := 2*spanSymbols*int(sampPerSym+0.5) + 1
	if numTaps%2 == 0 {
		numTaps++
	}

	taps := make([]float32, numTaps)
	mid := (numTaps - 1) / 2
	sumSq := 0.0

	for i := 0; i < numTaps; i++ {
		t := float64(i-mid) / sampPerSym // time in symbol periods

		var h float64
		if t == 0.0 {
			h = 1.0 - rolloff + 4.0*rolloff/math.Pi
		} else if math.Abs(math.Abs(t)-1.0/(4.0*rolloff)) < 1e-8 {
			h = rolloff / math.Sqrt2 * ((1.0+2.0/math.Pi)*math.Sin(math.Pi/(4.0*rolloff)) +
				(1.0-2.0/math.Pi)*math.Cos(math.Pi/(4.0*rolloff)))
		} else {
			num := math.Sin(math.Pi*t*(1-rolloff)) + 4*rolloff*t*math.Cos(math.Pi*t*(1+rolloff))
			den := math.Pi * t * (1 - math.Pow(4*rolloff*t, 2))
			h = num / den
		}

		taps[i] = float32(h)
		sumSq += h * h
	}

	// Normalize for unit energy
	norm := float32(math.Sqrt(sumSq))
	for i := range taps {
		taps[i] /= norm
	}

	return taps
}

// FIRFilterReal applies a non-decimating FIR filter to real-valued samples.
type FIRFilterReal struct {
	taps    []float32
	overlap []float32
	// buf is a reusable scratch buffer, grown as needed.
	buf []float32
	// outBuf is a reusable output buffer for ProcessReuse. Its contents are
	// only valid until the next call to Process or ProcessReuse.
	outBuf []float32
}

// NewFIRFilterReal creates a non-decimating real-valued FIR filter.
func NewFIRFilterReal(taps []float32) *FIRFilterReal {
	return &FIRFilterReal{
		taps:    taps,
		overlap: make([]float32, len(taps)-1),
	}
}

// Process applies the FIR filter to the input samples and returns filtered output
// of the same length.
func (f *FIRFilterReal) Process(input []float32) []float32 {
	out := make([]float32, len(input))
	f.processInto(input, out)
	return out
}

// ProcessReuse applies the filter and returns a slice backed by an internal
// buffer that is overwritten on the next call to Process or ProcessReuse on
// this filter. Use this for intermediate results that are consumed before the
// next call. Avoids one len(input)-sized allocation per block on hot paths
// (e.g. P25Decoder.Process at 25 kSPS) where the previous Process allocator
// dominated heap churn.
func (f *FIRFilterReal) ProcessReuse(input []float32) []float32 {
	if cap(f.outBuf) < len(input) {
		f.outBuf = make([]float32, len(input))
	}
	out := f.outBuf[:len(input)]
	f.processInto(input, out)
	return out
}

func (f *FIRFilterReal) processInto(input, out []float32) {
	nTaps := len(f.taps)
	overlap := nTaps - 1
	need := overlap + len(input)

	if cap(f.buf) < need {
		f.buf = make([]float32, need)
	}
	buf := f.buf[:need]
	copy(buf, f.overlap)
	copy(buf[overlap:], input)

	for i := range input {
		out[i] = firDotReal(f.taps, buf[i:i+nTaps])
	}

	copy(f.overlap, buf[len(buf)-overlap:])
}

// ResetFIR clears the filter state.
func (f *FIRFilterReal) ResetFIR() {
	for i := range f.overlap {
		f.overlap[i] = 0
	}
}

// Process applies the decimating FIR filter to the real input samples.
// Returns a freshly allocated decimated output slice.
func (f *DecimatingFilterReal) Process(input []float32) []float32 {
	maxOut := (len(input) + f.decimation - 1) / f.decimation
	out := make([]float32, maxOut)
	return f.processInto(input, out)
}

// ProcessReuse applies the filter and returns a slice backed by an internal
// buffer that is overwritten on the next call to Process or ProcessReuse on
// this filter. Use this for intermediate results consumed before the next call
// to avoid one output allocation per block on hot paths.
func (f *DecimatingFilterReal) ProcessReuse(input []float32) []float32 {
	maxOut := (len(input) + f.decimation - 1) / f.decimation
	if cap(f.outBuf) < maxOut {
		f.outBuf = make([]float32, maxOut)
	}
	out := f.processInto(input, f.outBuf[:maxOut])
	f.outBuf = out
	return out
}

// processInto stages [overlap | input] into the scratch buffer, strides the
// decimated dot product, and returns the used prefix of out. out must have
// length >= ceil(len(input)/decimation), the exact upper bound on emitted
// outputs given any starting phase.
func (f *DecimatingFilterReal) processInto(input, out []float32) []float32 {
	nTaps := len(f.taps)
	overlap := nTaps - 1
	need := overlap + len(input)

	if cap(f.buf) < need {
		f.buf = make([]float32, need)
	}
	buf := f.buf[:need]
	copy(buf, f.overlap)
	copy(buf[overlap:], input)

	outIdx := 0
	for i := 0; i < len(input); i++ {
		f.phase++
		if f.phase >= f.decimation {
			f.phase = 0
			out[outIdx] = firDotReal(f.taps, buf[i:i+nTaps])
			outIdx++
		}
	}

	copy(f.overlap, buf[len(buf)-overlap:])
	return out[:outIdx]
}
