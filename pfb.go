package dsp

import "fmt"

// Regenerate the AVX2 foldHop kernel. The generator is `//go:build ignore`, so
// the file form (.../main.go) is required — the package form `go run ./asm/foldhop`
// fails with "build constraints exclude all Go files". Do not "simplify" it back.
//
// -pkg dsp is required: avo names the generated stub package after the working
// directory (filepath.Base(cwd)), which is the module root "godsp" here — but
// this Go package is "dsp". Without -pkg the stubs would be emitted as
// `package godsp` and fail to compile against the rest of the package.
//go:generate go run ./asm/foldhop/main.go -out foldhop_amd64.s -stubs foldhop_amd64_stub.go -pkg dsp
//go:generate go run ./asm/fftc/main.go -out fftc_amd64.s -stubs fftc_amd64_stub.go -pkg dsp
//go:generate go run ./asm/fftc_batch/main.go -out fftc_batch_amd64.s -stubs fftc_batch_amd64_stub.go -pkg dsp
//go:generate go run ./asm/filter/main.go -out filter_amd64.s -stubs filter_amd64_stub.go -pkg dsp

// PFB is a 2×-oversampled polyphase analysis filterbank (weighted overlap-add
// structure). It consumes a complex64 wideband stream and produces K bins
// spaced inputRate/K apart; the hop is R = K/2 input samples, so every bin
// outputs at inputRate/R = 2·inputRate/K — twice its spacing. That 2×
// oversampling keeps content out to a full bin spacing from each center clear
// of decimation aliasing, so a consumer can pick the bin nearest any frequency
// (≤ spacing/2 away) without edge penalty.
//
// Bin b (signed, −K/2 ≤ b < K/2; FFT index (b+K) mod K) is centered at
// +b·(inputRate/K) relative to the input center; its output is that band
// down-converted to baseband and lowpassed by the prototype.
//
// Per-block usage (single producer goroutine):
//
//	hops := p.Stage(input)        // buffer input, size the output matrix
//	p.ComputeHops(0, hops, work)  // any partition of [0,hops) may run
//	                              // concurrently, one workspace each
//	out := p.Bins()               // row-major [hops][K], valid until next Stage
//	p.Finish()                    // retire consumed input, advance parity
//
// Process wraps the sequence serially for tests and tools.
type PFB struct {
	k, r, p int       // bins, hop (k/2), taps per phase
	proto   []float32 // prototype lowpass, len k·p, read-only
	fft     *FFTC
	buf     []complex64   // [retained history | staged input]
	hop     int64         // global hop counter (parity persists across blocks)
	nHops   int           // hops staged by the last Stage call
	out     []complex64   // row-major [nHops][k] bin matrix, reused
	work    *PFBWorkspace // scratch for the serial Process wrapper
}

// PFBWorkspace is per-caller scratch for concurrent hop-range computation.
type PFBWorkspace struct {
	tail []complex64 // row-major scratch for a padded four-hop tail batch
	fft  *fftcBatchWorkspace
}

// NewPFB builds a K-bin filterbank from a prototype lowpass of length K·P.
// k must be even (the hop is k/2) and 2/3/5-smooth (the FFT).
func NewPFB(proto []float32, k int) (*PFB, error) {
	if k <= 0 || k%2 != 0 {
		return nil, fmt.Errorf("PFB: bin count %d must be even", k)
	}
	if len(proto) == 0 || len(proto)%k != 0 {
		return nil, fmt.Errorf("PFB: prototype length %d must be a non-zero multiple of %d", len(proto), k)
	}
	fft, err := NewFFTC(k)
	if err != nil {
		return nil, fmt.Errorf("PFB: %w", err)
	}
	b := &PFB{k: k, r: k / 2, p: len(proto) / k, proto: proto, fft: fft}
	// Prime the history with L−R zeros so the first hop fires after R real
	// samples — the same zero warm-up convention as DecimatingFilter.
	b.buf = make([]complex64, len(proto)-b.r)
	return b, nil
}

// K returns the number of bins.
func (b *PFB) K() int { return b.k }

// NewWorkspace allocates reusable scratch for one ComputeHops caller.
func (b *PFB) NewWorkspace() *PFBWorkspace {
	return &PFBWorkspace{
		tail: make([]complex64, fftcBatchWidth*b.k),
		fft:  newFFTCBatchWorkspace(b.k),
	}
}

// Stage appends input to the internal buffer and returns the number of hops
// now computable. It sizes (but does not fill) the output matrix.
func (b *PFB) Stage(input []complex64) int {
	b.buf = append(b.buf, input...)
	l := b.k * b.p
	n := (len(b.buf) - (l - b.r)) / b.r
	if n < 0 {
		n = 0
	}
	b.nHops = n
	if cap(b.out) < n*b.k {
		b.out = make([]complex64, n*b.k)
	}
	b.out = b.out[:n*b.k]
	return n
}

// ComputeHops fills output rows [h0, h1): for each hop, fold the trailing K·P
// buffered samples through the prototype (polyphase alias-sum to length K),
// FFT, and apply the hop-parity rotation — with hop K/2, bin k accrues
// e^(−jπk) per hop, so odd bins are negated on odd global hops. Disjoint hop
// ranges may run concurrently: the buffer is only read and rows don't overlap.
// work is a distinct per-caller workspace sized by NewWorkspace.
func (b *PFB) ComputeHops(h0, h1 int, work *PFBWorkspace) {
	k, r, l := b.k, b.r, b.k*b.p
	h := h0
	for ; h+4 <= h1; h += 4 {
		rows := b.out[h*k : (h+4)*k]
		// Fold all four overlapping windows in one proto pass. Lane L's window is
		// b.buf[(h+L)*r : (h+L)*r+l]; the four rows are contiguous in `rows`.
		b.foldHop4(b.buf[h*r:(h+3)*r+l], rows)
		b.transformBatch4InPlace(rows, work.fft)
		for lane := 0; lane < fftcBatchWidth; lane++ {
			if (b.hop+int64(h+lane))&1 == 1 {
				row := rows[lane*k : (lane+1)*k]
				for i := 1; i < k; i += 2 {
					row[i] = -row[i]
				}
			}
		}
	}
	if h < h1 {
		n := h1 - h
		tail := work.tail[:fftcBatchWidth*k]
		for lane := 0; lane < n; lane++ {
			b.foldHop(b.buf[(h+lane)*r:(h+lane)*r+l], tail[lane*k:(lane+1)*k])
		}
		// TransformBatch4 operates on four independent FFT lanes: padding the
		// unused rows with zeroes cannot influence any actual row. Clearing them
		// also makes reuse deterministic without allocating per range.
		clear(tail[n*k:])
		b.transformBatch4InPlace(tail, work.fft)
		copy(b.out[h*k:h1*k], tail[:n*k])
		for lane := 0; lane < n; lane++ {
			row := b.out[(h+lane)*k : (h+lane+1)*k]
			if (b.hop+int64(h+lane))&1 == 0 {
				continue
			}
			for i := 1; i < k; i += 2 {
				row[i] = -row[i]
			}
		}
	}
}

// transformBatch4InPlace is safe because TransformBatch4 packs all four
// row-major inputs into its AoSoA workspace before any output row is written.
func (b *PFB) transformBatch4InPlace(rows []complex64, work *fftcBatchWorkspace) {
	b.fft.TransformBatch4(rows, rows, work)
}

// foldHop computes the length-K polyphase alias-sum of one K·P window through
// the prototype: fold[i] = Σ_q win[q·K+i]·proto[q·K+i] (the coefficient is
// real, so each term is two float32 multiplies). win must be length K·P and
// fold length K; fold is fully overwritten. Reads only b.proto, so concurrent
// calls with distinct win/fold are safe.
func (b *PFB) foldHop(win, fold []complex64) {
	k, n := b.k, b.k*b.p
	foldHopKernel(k, win[:n], b.proto[:n], fold)
}

// foldHop4 folds four overlapping hop-windows in one prototype pass. win is
// lane 0's window base (b.buf[h·R:…]); lane L's window begins L·R samples later
// and runs K·P, so win must cover through lane 3 (length ≥ 3·R + K·P). fold is
// the four contiguous K-length output rows. The per-row result is identical to
// four foldHop calls — the batched kernel only shares the proto load/permute.
func (b *PFB) foldHop4(win, fold []complex64) {
	k, n := b.k, b.k*b.p
	foldHop4Kernel(k, b.r, win, b.proto[:n], fold)
}

// Bins returns the staged row-major [hops][K] bin matrix. Valid until the
// next Stage call.
func (b *PFB) Bins() []complex64 { return b.out }

// Finish retires the input consumed by the staged hops and advances the
// global hop counter (parity).
func (b *PFB) Finish() {
	rem := copy(b.buf, b.buf[b.nHops*b.r:])
	b.buf = b.buf[:rem]
	b.hop += int64(b.nHops)
	b.nHops = 0
}

// Process stages input, computes every hop serially, finishes, and returns
// the bin matrix and hop count. The matrix is valid until the next call.
func (b *PFB) Process(input []complex64) ([]complex64, int) {
	n := b.Stage(input)
	if b.work == nil {
		b.work = b.NewWorkspace()
	}
	b.ComputeHops(0, n, b.work)
	out := b.out[:n*b.k]
	b.Finish()
	return out, n
}
