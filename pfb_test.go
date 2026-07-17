package dsp

import (
	"fmt"
	"math"
	"testing"
)

// pfbTestProto mirrors a typical polyphase-channelizer prototype design:
// cutoff = bin spacing, 10 taps per phase, odd-designed then zero-padded to K·P.
func pfbTestProto(rate float64, k int) []float32 {
	taps := DesignLPF(rate/float64(k), rate, k*10-1)
	return append(taps, 0)
}

// pfbBinStream feeds n samples of exp(j2πf·t/rate) through a fresh PFB and
// returns bin fftIdx's stream with the first `skip` warm-up samples dropped.
func pfbBinStream(t *testing.T, rate float64, k int, freq float64, fftIdx, n, skip int) []complex64 {
	t.Helper()
	p, err := NewPFB(pfbTestProto(rate, k), k)
	if err != nil {
		t.Fatal(err)
	}
	in := make([]complex64, n)
	for i := range in {
		ph := 2 * math.Pi * freq * float64(i) / rate
		in[i] = complex(float32(math.Cos(ph)), float32(math.Sin(ph)))
	}
	bins, hops := p.Process(in)
	if hops <= skip {
		t.Fatalf("only %d hops for %d samples", hops, n)
	}
	out := make([]complex64, 0, hops-skip)
	for h := skip; h < hops; h++ {
		out = append(out, bins[h*k+fftIdx])
	}
	return out
}

func pfbPowerDB(s []complex64) float64 {
	var sum float64
	for _, v := range s {
		sum += float64(real(v))*float64(real(v)) + float64(imag(v))*float64(imag(v))
	}
	if len(s) == 0 || sum == 0 {
		return math.Inf(-1)
	}
	return 10 * math.Log10(sum/float64(len(s)))
}

// pfbMeanFreq estimates the dominant baseband frequency (Hz at output rate
// outRate) from the mean phase increment.
func pfbMeanFreq(s []complex64, outRate float64) float64 {
	var acc complex128
	for i := 1; i < len(s); i++ {
		acc += complex128(s[i]) * complex128(complex(real(s[i-1]), -imag(s[i-1])))
	}
	return math.Atan2(imag(acc), real(acc)) * outRate / (2 * math.Pi)
}

func TestPFB_TonePlacementAndGain(t *testing.T) {
	const rate = 10_000_000.0
	const k = 80 // 125 kHz bins, output rate 250 kSPS
	// Tone 20 kHz above bin 13's center (1.625 MHz → 1.645 MHz).
	const freq = 13*125_000.0 + 20_000.0
	s := pfbBinStream(t, rate, k, freq, 13, 200_000, 50)
	if db := pfbPowerDB(s); db < -0.5 || db > 0.5 {
		t.Errorf("bin 13 power %.2f dB, want ~0 dB", db)
	}
	if got := pfbMeanFreq(s, 250_000); math.Abs(got-20_000) > 200 {
		t.Errorf("bin 13 baseband freq %.0f Hz, want 20000 Hz", got)
	}
	// A bin far away sees only stopband leakage.
	far := pfbBinStream(t, rate, k, freq, 40, 200_000, 50)
	if db := pfbPowerDB(far); db > -50 {
		t.Errorf("bin 40 leakage %.2f dB, want <= -50 dB", db)
	}
}

func TestPFB_BinEdgeToneCleanInBothBins(t *testing.T) {
	const rate = 10_000_000.0
	const k = 80
	// Exactly between bins 5 and 6: +62.5 kHz from bin 5, −62.5 kHz from bin 6.
	const freq = 5*125_000.0 + 62_500.0
	lo := pfbBinStream(t, rate, k, freq, 5, 200_000, 50)
	hi := pfbBinStream(t, rate, k, freq, 6, 200_000, 50)
	if db := pfbPowerDB(lo); db < -1 || db > 1 {
		t.Errorf("bin 5 edge power %.2f dB, want ~0 dB", db)
	}
	if db := pfbPowerDB(hi); db < -1 || db > 1 {
		t.Errorf("bin 6 edge power %.2f dB, want ~0 dB", db)
	}
	if got := pfbMeanFreq(lo, 250_000); math.Abs(got-62_500) > 200 {
		t.Errorf("bin 5 freq %.0f Hz, want +62500", got)
	}
	if got := pfbMeanFreq(hi, 250_000); math.Abs(got+62_500) > 200 {
		t.Errorf("bin 6 freq %.0f Hz, want -62500", got)
	}
}

func TestPFB_AliasRejection(t *testing.T) {
	const rate = 10_000_000.0
	const k = 80
	// 200 kHz above bin 13's center: after ÷R it would fold to −50 kHz inside
	// bin 13's output unless the prototype stopband kills it first.
	const freq = 13*125_000.0 + 200_000.0
	s := pfbBinStream(t, rate, k, freq, 13, 200_000, 50)
	if db := pfbPowerDB(s); db > -50 {
		t.Errorf("bin 13 alias power %.2f dB, want <= -50 dB", db)
	}
}

func TestPFB_SplitBlockBitIdentical(t *testing.T) {
	const k = 80
	const total = 50_000
	proto := pfbTestProto(10_000_000, k)
	in := make([]complex64, total)
	for i := range in {
		ph := 2 * math.Pi * 0.0137 * float64(i)
		in[i] = complex(float32(math.Cos(ph)), float32(math.Sin(3*ph)))
	}

	one, hops1 := NewPFBMust(t, proto, k).Process(in)
	if want := total / (k / 2); hops1 != want {
		t.Fatalf("hops = %d, want %d", hops1, want)
	}
	oneCopy := append([]complex64(nil), one...)

	two := make([]complex64, 0, len(oneCopy))
	p2 := NewPFBMust(t, proto, k)
	for off := 0; off < total; off += 977 {
		end := min(off+977, total)
		bins, hops := p2.Process(in[off:end])
		two = append(two, bins[:hops*k]...)
	}
	if len(two) != len(oneCopy) {
		t.Fatalf("split outputs %d samples, single %d", len(two), len(oneCopy))
	}
	for i := range oneCopy {
		if two[i] != oneCopy[i] {
			t.Fatalf("sample %d: split %v != single %v", i, two[i], oneCopy[i])
		}
	}
}

func TestPFBComputeHopsBatchAndTailsMatchProcess(t *testing.T) {
	const k = 120
	proto := pfbTestProto(15_000_000, k)
	seed := uint32(0x4b1d)
	prefix := make([]complex64, k/2) // advance global hop parity to odd
	input := make([]complex64, 10*k/2)
	for i := range prefix {
		prefix[i] = complex(lcg(&seed), lcg(&seed))
	}
	for i := range input {
		input[i] = complex(lcg(&seed), lcg(&seed))
	}

	for split := 1; split <= 9; split++ {
		t.Run(fmt.Sprintf("split=%d", split), func(t *testing.T) {
			serial := NewPFBMust(t, proto, k)
			serial.Process(prefix)
			want, wantHops := serial.Process(input)
			want = append([]complex64(nil), want...)

			partitioned := NewPFBMust(t, proto, k)
			partitioned.Process(prefix)
			hops := partitioned.Stage(input)
			if hops != wantHops {
				t.Fatalf("hops = %d, want %d", hops, wantHops)
			}
			partitioned.ComputeHops(0, split, partitioned.NewWorkspace())
			partitioned.ComputeHops(split, hops, partitioned.NewWorkspace())
			got := partitioned.Bins()
			assertComplex64SlicesClose(t, got, want, 2e-4)
			partitioned.Finish()
		})
	}
}

func TestPFBComputeHopsTailUsesBatchScratch(t *testing.T) {
	const k = 120
	for n := 1; n < fftcBatchWidth; n++ {
		t.Run(fmt.Sprintf("hops=%d", n), func(t *testing.T) {
			p := NewPFBMust(t, pfbTestProto(15_000_000, k), k)
			input := make([]complex64, n*k/2)
			seed := uint32(0x17a9)
			for i := range input {
				input[i] = complex(lcg(&seed), lcg(&seed))
			}
			work := p.NewWorkspace()
			if len(work.tail) != fftcBatchWidth*k {
				t.Fatalf("tail scratch length = %d, want %d", len(work.tail), fftcBatchWidth*k)
			}

			hops := p.Stage(input)
			if hops != n {
				t.Fatalf("hops = %d, want %d", hops, n)
			}
			p.ComputeHops(0, hops, work)
			for h := 0; h < hops; h++ {
				for i := 0; i < k; i++ {
					want := work.tail[h*k+i]
					if (int64(h)&1) == 1 && i&1 == 1 {
						want = -want
					}
					if got := p.Bins()[h*k+i]; got != want {
						t.Fatalf("hop %d bin %d = %v, tail scratch gives %v", h, i, got, want)
					}
				}
			}
			for i, got := range work.tail[hops*k:] {
				if got != 0 {
					t.Fatalf("padding sample %d = %v, want zero", i, got)
				}
			}
		})
	}
}

func TestPFBComputeHopsZeroAlloc(t *testing.T) {
	const k = 120
	proto := pfbTestProto(15_000_000, k)
	for _, wantHops := range []int{1, 2, 3, 5, 6, 7} {
		t.Run(fmt.Sprintf("hops=%d", wantHops), func(t *testing.T) {
			p := NewPFBMust(t, proto, k)
			input := make([]complex64, wantHops*k/2)
			seed := uint32(77 + wantHops)
			for i := range input {
				input[i] = complex(lcg(&seed), lcg(&seed))
			}
			work := p.NewWorkspace()

			hops := p.Stage(input)
			if hops != wantHops {
				t.Fatalf("warm-up hops = %d, want %d", hops, wantHops)
			}
			p.ComputeHops(0, hops, work)
			p.Finish()
			hops = p.Stage(input)
			if hops != wantHops {
				t.Fatalf("measured hops = %d, want %d", hops, wantHops)
			}
			if allocs := testing.AllocsPerRun(100, func() {
				p.ComputeHops(0, hops, work)
			}); allocs != 0 {
				t.Fatalf("ComputeHops(%d hops) allocations = %g, want 0", wantHops, allocs)
			}
		})
	}
}

func NewPFBMust(t *testing.T, proto []float32, k int) *PFB {
	t.Helper()
	p, err := NewPFB(proto, k)
	if err != nil {
		t.Fatal(err)
	}
	return p
}

// TestPFBFold_MatchesReference checks the polyphase alias-sum against an
// independent float64 reference for random input (guards the register-
// accumulating fold rewrite).
func TestPFBFold_MatchesReference(t *testing.T) {
	// Production stepping-stone bin counts. 120 is legacy; 192 is the current live
	// K; 240/288/384/480 are rate-ladder targets — all multiples of 8, exercising
	// the pure AVX2 vector head on amd64. 180 (= 2²·3²·5, not a multiple of 8)
	// additionally drives the amd64 scalar tail (bins 176–179). The float64 oracle
	// gates whichever kernel the current arch compiles.
	for _, k := range []int{90, 120, 180, 192, 240, 288, 384, 480} {
		t.Run(fmt.Sprintf("K=%d", k), func(t *testing.T) {
			p, err := NewPFB(pfbTestProto(15_000_000, k), k)
			if err != nil {
				t.Fatal(err)
			}
			win := make([]complex64, k*p.p)
			seed := uint32(2024)
			for i := range win {
				win[i] = complex(lcg(&seed), lcg(&seed))
			}
			fold := make([]complex64, k)
			p.foldHop(win, fold)
			for i := 0; i < k; i++ {
				var re, im float64
				for q := 0; q < p.p; q++ {
					h := float64(p.proto[q*k+i])
					re += float64(real(win[q*k+i])) * h
					im += float64(imag(win[q*k+i])) * h
				}
				if math.Abs(float64(real(fold[i]))-re) > 1e-3 || math.Abs(float64(imag(fold[i]))-im) > 1e-3 {
					t.Fatalf("K=%d bin %d: got (%g,%g) want (%g,%g)", k, i, real(fold[i]), imag(fold[i]), re, im)
				}
			}
		})
	}
}

// TestPFBFold4_BitIdenticalToFold checks the batched four-window kernel against
// four independent single-window folds BYTE-FOR-BYTE (exact complex64 equality,
// not epsilon). The batched kernel only hoists the shared proto load/permute, so
// every lane's accumulation must match its foldHop result bit-for-bit. K values
// span multiples of 8 (pure vector head) and 180 (drives the k%8 scalar tail).
func TestPFBFold4_BitIdenticalToFold(t *testing.T) {
	for _, k := range []int{90, 120, 180, 192, 240, 288, 384, 480} {
		t.Run(fmt.Sprintf("K=%d", k), func(t *testing.T) {
			p, err := NewPFB(pfbTestProto(15_000_000, k), k)
			if err != nil {
				t.Fatal(err)
			}
			r, n := p.r, k*p.p
			// One buffer covering all four overlapping windows: lane L starts at
			// L*r and runs n, so the last lane needs 3*r + n samples.
			buf := make([]complex64, 3*r+n)
			seed := uint32(1301)
			for i := range buf {
				buf[i] = complex(lcg(&seed), lcg(&seed))
			}

			want := make([]complex64, 4*k)
			for lane := 0; lane < 4; lane++ {
				p.foldHop(buf[lane*r:lane*r+n], want[lane*k:(lane+1)*k])
			}

			got := make([]complex64, 4*k)
			p.foldHop4(buf, got)

			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("K=%d idx %d (lane %d bin %d): batched %v != single %v",
						k, i, i/k, i%k, got[i], want[i])
				}
			}
		})
	}
}

// BenchmarkPFBFold measures one polyphase fold at each production geometry,
// isolated from the FFT. The fold is the channelizer's dominant CPU cost.
func BenchmarkPFBFold(b *testing.B) {
	for _, k := range []int{120, 180, 192, 240, 288, 384, 480} {
		b.Run(fmt.Sprintf("K=%d", k), func(b *testing.B) {
			p, err := NewPFB(pfbTestProto(15_000_000, k), k)
			if err != nil {
				b.Fatal(err)
			}
			win := make([]complex64, k*p.p)
			seed := uint32(7)
			for i := range win {
				win[i] = complex(lcg(&seed), lcg(&seed))
			}
			fold := make([]complex64, k)
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				p.foldHop(win, fold)
			}
		})
	}
}
