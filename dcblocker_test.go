package dsp

import (
	"math"
	"testing"
)

// TestDCBlocker_BlocksDC verifies a constant (pure DC) input converges to ~0 at
// steady state, mirroring TestButterworthHPF_DCBlocked. For x[n]=1 the recursion
// collapses to y[n] = R*y[n-1] after the first sample, so y[n] = R^n -> 0.
func TestDCBlocker_BlocksDC(t *testing.T) {
	b := NewDCBlocker(0.995)
	last := float32(0)
	for i := 0; i < 50000; i++ {
		last = b.Process(1.0)
	}
	if math.Abs(float64(last)) > 1e-3 {
		t.Errorf("DC output not blocked: steady-state = %g", last)
	}
}

// TestDCBlocker_PassesAC verifies a pure tone well above the corner passes with
// ~unity gain. The corner for R=0.995 at 48 kHz is ~38 Hz, so a 1 kHz tone is
// far into the passband where |H| ~ 1. We compare output RMS to input RMS after
// discarding the transient.
func TestDCBlocker_PassesAC(t *testing.T) {
	const (
		n  = 20000
		fs = 48000.0
		f  = 1000.0
	)
	b := NewDCBlocker(0.995)
	in := make([]float32, n)
	for i := range in {
		in[i] = float32(math.Sin(2 * math.Pi * f * float64(i) / fs))
	}
	out := make([]float32, n)
	for i, x := range in {
		out[i] = b.Process(x)
	}
	// Discard the first quarter so only steady-state is measured; R^5000 is
	// astronomically small, so the transient is long gone by then.
	start := n / 4
	var sumIn, sumOut float64
	for i := start; i < n; i++ {
		sumIn += float64(in[i]) * float64(in[i])
		sumOut += float64(out[i]) * float64(out[i])
	}
	ratio := math.Sqrt(sumOut / sumIn)
	if ratio < 0.99 || ratio > 1.01 {
		t.Errorf("AC RMS gain ratio at %.0f Hz = %.4f, want ~1.0", f, ratio)
	}
}

// TestDCBlockerProcessBlockBitExact verifies ProcessBlock produces bit-for-bit
// identical output AND identical final state to per-sample Process, across
// arbitrary block splits. This must be exact (no tolerance) because ProcessBlock
// runs the same arithmetic in the same order as Process. Mirrors
// TestBiquadProcessBlockBitExact.
func TestDCBlockerProcessBlockBitExact(t *testing.T) {
	proto := NewDCBlocker(0.997)

	in := make([]float32, 1000)
	for i := range in {
		in[i] = float32(math.Sin(2*math.Pi*233.6*float64(i)/25000) + 0.3*math.Sin(2*math.Pi*1000*float64(i)/25000))
	}

	// Reference: single-sample Process over the whole stream.
	ref := *proto
	want := make([]float32, len(in))
	for i, x := range in {
		want[i] = ref.Process(x)
	}

	// Block path: ProcessBlock over uneven chunks.
	blk := *proto
	got := append([]float32(nil), in...)
	for pos := 0; pos < len(got); {
		size := []int{1, 7, 64, 3, 128, 33}[pos%6]
		if size > len(got)-pos {
			size = len(got) - pos
		}
		blk.ProcessBlock(got[pos : pos+size])
		pos += size
	}

	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("sample %d: block %v != per-sample %v (must be bit-exact)", i, got[i], want[i])
		}
	}
	if blk.x1 != ref.x1 || blk.y1 != ref.y1 {
		t.Fatalf("final state diverged: block {%v %v} vs per-sample {%v %v}",
			blk.x1, blk.y1, ref.x1, ref.y1)
	}
}

// TestIQDCBlocker_RailIndependence verifies each rail is filtered independently
// and bit-exactly against a standalone DCBlocker. A constant-DC input CANNOT
// prove this: with DC on both rails every rail decays as R^n regardless of which
// state it reads, so a broken impl where Q reads I's history still converges to
// zero and passes. Instead we drive I and Q with DISTINCT tones (fA != fB) on
// DISTINCT biases (biasI != biasQ) and compare, with NO tolerance, against two
// independent reference DCBlockers fed the untouched per-rail inputs. This one
// test proves several things at once:
//   - independence: a rail swap would put fA on the Q output, breaking equality;
//   - no cross-talk: any leakage between rails perturbs a rail off its reference;
//   - the per-rail bit-exactness claim in ProcessInPlace's doc comment;
//   - the passband, since both tones sit well above the ~38 Hz corner.
func TestIQDCBlocker_RailIndependence(t *testing.T) {
	const (
		n     = 20000
		fs    = 48000.0
		fA    = 1000.0 // I-rail tone
		fB    = 1700.0 // Q-rail tone, distinct from fA to catch a rail swap
		biasI = 0.7
		biasQ = -0.3 // distinct from biasI to catch cross-rail bias leakage
	)
	realIn := make([]float32, n)
	imagIn := make([]float32, n)
	samples := make([]complex64, n)
	for i := 0; i < n; i++ {
		ri := float32(biasI + math.Sin(2*math.Pi*fA*float64(i)/fs))
		qi := float32(biasQ + math.Sin(2*math.Pi*fB*float64(i)/fs))
		realIn[i], imagIn[i] = ri, qi
		samples[i] = complex(ri, qi)
	}

	NewIQDCBlocker(0.995).ProcessInPlace(samples)

	// Two fully independent references: if the IQ blocker shares or swaps state
	// between rails, one of these comparisons must diverge.
	iRef := NewDCBlocker(0.995)
	qRef := NewDCBlocker(0.995)
	for i := 0; i < n; i++ {
		wantI := iRef.Process(realIn[i])
		wantQ := qRef.Process(imagIn[i])
		if real(samples[i]) != wantI {
			t.Fatalf("sample %d: I rail = %v, independent ref = %v (rail swap or cross-talk?)", i, real(samples[i]), wantI)
		}
		if imag(samples[i]) != wantQ {
			t.Fatalf("sample %d: Q rail = %v, independent ref = %v (rail swap or cross-talk?)", i, imag(samples[i]), wantQ)
		}
	}
}

// TestNewDCBlocker_ClampsInvalidR verifies the constructor guard: R at or
// outside (0,1) -- including the marginally stable R>=1 -- is clamped into the
// stable range, while a valid R passes through unchanged.
func TestNewDCBlocker_ClampsInvalidR(t *testing.T) {
	for _, r := range []float32{0, -0.5, 1, 1.5, 2} {
		b := NewDCBlocker(r)
		if b.r <= 0 || b.r >= 1 {
			t.Errorf("NewDCBlocker(%v): r=%v not clamped into (0,1)", r, b.r)
		}
	}
	if b := NewDCBlocker(0.99); b.r != 0.99 {
		t.Errorf("NewDCBlocker(0.99): r=%v, want 0.99 (valid R must pass through)", b.r)
	}
}

// TestNewDCBlockerHz verifies the R = 1 - 2*pi*fc/fs mapping and that a cutoff
// large enough to drive R out of (0,1) is clamped.
func TestNewDCBlockerHz(t *testing.T) {
	b := NewDCBlockerHz(20, 48000)
	want := float32(1.0 - 2.0*math.Pi*20.0/48000.0)
	if b.r != want {
		t.Errorf("NewDCBlockerHz(20,48000): r=%v, want %v", b.r, want)
	}
	// 2*pi*20000/48000 ~ 2.618, so R ~ -1.618, which must be clamped.
	hi := NewDCBlockerHz(20000, 48000)
	if hi.r <= 0 || hi.r >= 1 {
		t.Errorf("NewDCBlockerHz(20000,48000): r=%v not clamped into (0,1)", hi.r)
	}
}

// TestDCBlocker_Reset verifies Reset returns the filter to its as-constructed
// state: after running samples to make the state non-zero, Reset() must make the
// next Process output bit-identical to that of a fresh blocker fed the same
// sample. If Reset left either state word dirty, the recursion would carry it
// into the first post-reset output and the equality would fail.
func TestDCBlocker_Reset(t *testing.T) {
	b := NewDCBlocker(0.995)
	for i := 0; i < 100; i++ {
		b.Process(float32(math.Sin(float64(i))) + 0.5)
	}
	if b.x1 == 0 && b.y1 == 0 {
		t.Fatal("precondition failed: state still zero after priming, Reset test proves nothing")
	}
	b.Reset()

	fresh := NewDCBlocker(0.995)
	const probe = 0.42
	if got, want := b.Process(probe), fresh.Process(probe); got != want {
		t.Errorf("after Reset: Process(%v)=%v, fresh blocker=%v", probe, got, want)
	}
}

// TestIQDCBlocker_Reset verifies Reset clears BOTH rails. Distinct I/Q inputs
// make both rails' state non-zero; after Reset the next sample's real and
// imaginary outputs must each match a fresh IQ blocker. A Reset that cleared
// only one rail would leave residual state on the other and diverge there.
func TestIQDCBlocker_Reset(t *testing.T) {
	b := NewIQDCBlocker(0.995)
	prime := make([]complex64, 100)
	for i := range prime {
		prime[i] = complex(float32(math.Sin(float64(i)))+0.7, float32(math.Cos(float64(i)))-0.3)
	}
	b.ProcessInPlace(prime)
	if b.i.x1 == 0 && b.i.y1 == 0 && b.q.x1 == 0 && b.q.y1 == 0 {
		t.Fatal("precondition failed: rail state still zero after priming, Reset test proves nothing")
	}
	b.Reset()

	fresh := NewIQDCBlocker(0.995)
	probe := []complex64{complex(0.42, -0.17)}
	freshProbe := append([]complex64(nil), probe...)
	b.ProcessInPlace(probe)
	fresh.ProcessInPlace(freshProbe)
	if real(probe[0]) != real(freshProbe[0]) {
		t.Errorf("after Reset: I rail=%v, fresh=%v", real(probe[0]), real(freshProbe[0]))
	}
	if imag(probe[0]) != imag(freshProbe[0]) {
		t.Errorf("after Reset: Q rail=%v, fresh=%v", imag(probe[0]), imag(freshProbe[0]))
	}
}
