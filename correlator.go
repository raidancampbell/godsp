package dsp

import "math"

// Correlator is a streaming cross-correlator / matched filter that detects a
// known complex pattern (a preamble, sync word, or frame-sync symbol sequence
// such as the P25 frame sync) inside a complex baseband stream. It wraps a
// ComplexFIRFilter whose taps are the matched-filter taps for the pattern, and
// adds the streaming bookkeeping needed to report detected peaks at an ABSOLUTE
// sample index that stays correct across many Process calls (including patterns
// that straddle a block boundary -- the inner filter's overlap-save already
// carries the convolution seam, so all this type adds on top is an output-index
// offset).
//
// Matched-filter tap construction (the part that is easy to get backwards, so
// it is spelled out). For a length-L pattern p, the matched filter maximizes
// output magnitude at perfect alignment. The TEXTBOOK statement is that the
// matched filter's impulse response is the time-reversed conjugate of the
// pattern, tap[k] = conj(p[L-1-k]); that form is correct for a filter that
// convolves, out[i] = sum_k h[k]*x[i-k]. But ComplexFIRFilter does NOT convolve
// in that sense: its inner loop computes
//
//	out[i] = sum_k taps[k] * window[k]
//
// where window[k] runs FORWARD over the input (taps[L-1] multiplies the NEWEST
// sample). Both indices increase together, so this is a CORRELATION form, not a
// flipped convolution. The forward indexing already supplies the time reversal,
// so the reversal must NOT be applied a second time in the taps. The correct
// taps for this filter are therefore the plain (un-reversed) conjugate:
//
//	tap[k] = conj(p[k])
//
// This is the exact same forward-indexing subtlety that makes DesignComplexBandpass
// use exp(-j...) instead of the textbook exp(+j...); see its doc comment. To see
// why conj(p) is right here, embed p at input index m (input[m+j] = p[j]). The
// output whose window exactly covers the pattern is i = m + L - 1, and at that i
// the window element k equals p[k], so
//
//	out[m+L-1] = sum_k conj(p[k]) * p[k] = sum_k |p[k]|^2 = E
//
// i.e. the peak output equals the pattern ENERGY E (a real, positive number),
// the largest magnitude any unit-scaled input window can produce (Cauchy-Schwarz).
// Correlation magnitude therefore peaks -- at the value E -- when the pattern
// aligns. Reset() returns the streaming state to construction time.
type Correlator struct {
	// fir holds the matched-filter taps conj(p[k]) and all the overlap-save
	// state. All the real convolution work (and its SIMD acceleration) lives
	// there; Correlator only adds magnitude extraction and index bookkeeping.
	fir *ComplexFIRFilter

	// patternLen is L, the number of pattern symbols. The FIR group delay is
	// L-1 samples: a correlation peak observed at stream index P corresponds to
	// a pattern that STARTS at index P-(L-1) (see Detect).
	patternLen int

	// energy is E = sum_k |p[k]|^2, the correlation magnitude at perfect
	// alignment. Held as float32 (the library working type) but accumulated in
	// float64 in the constructor. Thresholds are naturally expressed as a
	// fraction of E; see NormalizedThreshold.
	energy float32

	// globalIndex is the absolute stream index of the NEXT input sample not yet
	// processed, i.e. the running count of samples fed since construction/Reset.
	// It is the peak-lag base for the next Process/Detect call: the magnitude at
	// local output position lo corresponds to peak-lag stream index
	// globalIndex+lo. int64 so long-running streams (hours of high-rate IQ) do
	// not overflow a 32-bit counter.
	globalIndex int64

	// magBuf is a reusable output buffer for Process, grown as needed. Its
	// contents are only valid until the next Process/Detect call, mirroring the
	// ProcessReuse contract on ComplexFIRFilter and the other stateful types.
	magBuf []float32
}

// NewCorrelator builds a streaming matched filter for the given complex pattern
// (preamble / sync word). The pattern is conjugated into the inner FIR taps, so
// the caller may reuse or mutate the input slice afterward. Panics on an empty
// pattern -- there is nothing to correlate against -- matching the fail-fast
// precondition style of the other constructors.
//
// Pick a pattern with a sharp autocorrelation (a large peak-to-sidelobe ratio),
// e.g. a P25-style +/-1 sync sequence or a Barker code, so the aligned peak
// stands clearly above off-alignment lags.
func NewCorrelator(pattern []complex64) *Correlator {
	n := len(pattern)
	if n == 0 {
		panic("NewCorrelator: pattern must be non-empty")
	}

	// Matched-filter taps for this correlation-form FIR are the plain conjugate
	// (NOT time-reversed); see the type doc for the derivation. Accumulate the
	// energy in float64 so a long pattern's sum of squares does not lose
	// low-order bits before it is stored back as float32.
	taps := make([]complex64, n)
	var energy float64
	for k, p := range pattern {
		re, im := real(p), imag(p)
		taps[k] = complex(re, -im) // conj(p[k])
		energy += float64(re)*float64(re) + float64(im)*float64(im)
	}

	return &Correlator{
		fir:        NewComplexFIRFilter(taps),
		patternLen: n,
		energy:     float32(energy),
	}
}

// process runs the matched filter over input and writes the per-sample
// correlation MAGNITUDE (|out[i]|, not magnitude-squared) into the reusable
// magBuf. Magnitude (not magnitude-squared) is the reported quantity because
// its peak value is exactly the pattern energy E, so a threshold reads on the
// same linear scale as the taps and NormalizedThreshold(fraction) = fraction*E
// is directly meaningful. The returned slice aliases magBuf and is valid only
// until the next call. globalIndex is advanced by len(input) so successive
// calls report absolute peak-lag positions.
func (c *Correlator) process(input []complex64) []float32 {
	// ProcessReuse avoids a per-call allocation on the hot path; its result is
	// only valid until our next FIR call, but we consume it immediately below
	// (into magBuf) and never retain it.
	out := c.fir.ProcessReuse(input)

	n := len(out)
	if cap(c.magBuf) < n {
		c.magBuf = make([]float32, n)
	}
	c.magBuf = c.magBuf[:n]
	for i, y := range out {
		// Compute the magnitude in float64 to avoid intermediate float32
		// overflow/precision loss in the sum of squares, then narrow to the
		// library working type. This is the peak-lag-aligned magnitude:
		// magBuf[i] is the correlation when input[i] is the NEWEST sample of the
		// aligned window.
		r := float64(real(y))
		im := float64(imag(y))
		c.magBuf[i] = float32(math.Sqrt(r*r + im*im))
	}

	c.globalIndex += int64(n)
	return c.magBuf
}

// Process applies the matched filter and returns the per-sample correlation
// magnitudes aligned to the input samples: result[i] is |correlation| when
// input[i] is the newest sample of the aligned window (the "peak-lag" position).
// The returned slice is backed by an internal buffer and is only valid until the
// next call to Process or Detect on this Correlator (same reuse contract as
// ComplexFIRFilter.ProcessReuse). At perfect pattern alignment the magnitude at
// the corresponding peak-lag sample equals the pattern energy E (PatternEnergy).
//
// Note the group delay: the magnitude peak sits at the pattern's LAST sample,
// L-1 samples after its start. Use Detect when you want peaks reported at the
// pattern's absolute START index with that delay already removed.
func (c *Correlator) Process(input []complex64) []float32 {
	return c.process(input)
}

// Detection is one above-threshold correlation peak. Index is the ABSOLUTE
// stream sample index (counting from construction / last Reset) at which the
// matched pattern STARTS -- the FIR group delay of L-1 has already been removed,
// so Index points at the pattern's first sample, not the correlation peak lag.
// Magnitude is the correlation magnitude at that peak (~= pattern energy E at a
// clean alignment).
type Detection struct {
	Index     int64
	Magnitude float32
}

// Detect runs the matched filter over input and reports every position whose
// correlation magnitude is at least threshold (m >= threshold). threshold is an
// ABSOLUTE magnitude on the same scale as the correlation output; because the
// peak magnitude at perfect alignment is the pattern energy E, a natural choice
// is a fraction of E via NormalizedThreshold (e.g. 0.7*E). The comparison is
// inclusive so that NormalizedThreshold(1.0) (threshold == E) still reports a
// mathematically perfect match, whose magnitude is exactly E.
//
// Detect reports EVERY sample at or above threshold; it does NOT peak-pick or
// deduplicate. For a clean impulse-like autocorrelation (e.g. a Barker or P25
// sync sequence, one sample wide at threshold) that is one detection per
// arrival. But with a broad correlation lobe -- pulse-shaped symbols, heavy
// noise, or a low threshold -- a single arrival can exceed threshold at several
// ADJACENT lags and produce a cluster of detections. Callers that need one
// detection per arrival should cluster the returned Detections (e.g. keep the
// max-magnitude sample within each run of consecutive indices).
//
// Offset math (the easy-to-get-wrong part, stated precisely). Let base be the
// absolute stream index of the first sample of THIS call (the value of
// globalIndex on entry). The magnitude at local output position lo corresponds
// to the window whose NEWEST sample is stream index base+lo -- that is the
// correlation PEAK LAG, which lands on the pattern's LAST sample. The pattern is
// L samples long, so its FIRST sample is L-1 earlier:
//
//	peakLag   = base + lo
//	patternStart (reported Index) = peakLag - (patternLen - 1)
//
// Concretely, if a pattern is embedded starting at absolute stream index m
// (input[m+j] = p[j]), the peak appears at local lo giving peakLag = m+L-1, and
// the reported Index = (m+L-1) - (L-1) = m. The reported index therefore equals
// the embedded start offset exactly, and stays correct across Process calls (a
// pattern split across two calls peaks in the second call at the lo that makes
// base+lo-(L-1) land back on its true absolute start).
//
// Because the FIR is fed (L-1) leading zeros at the very start of the stream,
// the first few outputs of the first call use a window that is only partially
// filled with real data; a genuine full-pattern match cannot occur there, so
// those transients do not cross a sane threshold. In principle a spurious early
// crossing could yield a negative reported Index (a pattern notionally starting
// before the stream began); callers can ignore negative indices, but a
// well-chosen threshold makes them not arise.
func (c *Correlator) Detect(input []complex64, threshold float32) []Detection {
	base := c.globalIndex
	mags := c.process(input)

	var dets []Detection
	delay := int64(c.patternLen - 1)
	for lo, m := range mags {
		// >= (not >) so a mathematically perfect match, whose magnitude equals
		// exactly the pattern energy E, is reported when threshold is set to E
		// (NormalizedThreshold(1.0)); a strict > would silently drop it. For an
		// ideal one-sample-wide peak this admits no duplicates.
		if m >= threshold {
			dets = append(dets, Detection{
				Index:     base + int64(lo) - delay,
				Magnitude: m,
			})
		}
	}
	return dets
}

// PatternLen returns L, the number of pattern symbols (the correlator's group
// delay is L-1 samples).
func (c *Correlator) PatternLen() int { return c.patternLen }

// PatternEnergy returns E = sum_k |p[k]|^2, the correlation magnitude produced
// at perfect alignment. Use it to set data-independent thresholds (e.g. accept
// a detection only above 0.7*E); NormalizedThreshold is the convenience wrapper.
func (c *Correlator) PatternEnergy() float32 { return c.energy }

// NormalizedThreshold converts a fraction in [0,1] of the perfect-alignment
// peak into the absolute-magnitude threshold Detect expects: it returns
// fraction*PatternEnergy(). fraction=1 yields threshold == E, which (given
// Detect's inclusive m >= threshold comparison) accepts a mathematically
// perfect (noise-free, unity-scaled) match sitting exactly at E, and nothing
// less. Any real match is dragged slightly below E by noise and amplitude
// offset, so callers typically pass fraction < 1 (~0.6-0.8) to tolerate that
// while still rejecting autocorrelation sidelobes.
func (c *Correlator) NormalizedThreshold(fraction float32) float32 {
	return fraction * c.energy
}

// Reset clears the inner filter's overlap-save history and rewinds the absolute
// sample counter to zero, so the next Process/Detect call behaves as on a
// freshly constructed Correlator (subsequent detections index from that new
// origin). The taps, pattern length, and energy are preserved.
func (c *Correlator) Reset() {
	c.fir.Reset()
	c.globalIndex = 0
}
