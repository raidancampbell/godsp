package dsp

import "testing"

func TestStage1DecimationFor(t *testing.T) {
	cases := []struct {
		rate uint32
		want int
	}{
		{10_000_000, 40},
		{12_000_000, 48}, // 12 MSPS is a valid multiple of 250k (it failed before only
		{20_000_000, 80}, // because the OLD decimation was fixed at 400 -> 30 kSPS).
		{30_000_000, 120},
		{40_000_000, 160},
		{50_000_000, 200},
	}
	for _, c := range cases {
		got, err := Stage1DecimationFor(c.rate)
		if err != nil {
			t.Errorf("Stage1DecimationFor(%d) unexpected error: %v", c.rate, err)
			continue
		}
		if got != c.want {
			t.Errorf("Stage1DecimationFor(%d) = %d, want %d", c.rate, got, c.want)
		}
		// The channel contract must hold for every valid rate.
		if c.rate/uint32(got)/Stage2Decimation != ChannelIQRate {
			t.Errorf("rate %d: derived channel rate != %d", c.rate, ChannelIQRate)
		}
	}
}

func TestStage1DecimationFor_RejectsNonMultiples(t *testing.T) {
	for _, bad := range []uint32{0, 10_700_000, 13_333_333, 50_000_001} {
		if _, err := Stage1DecimationFor(bad); err == nil {
			t.Errorf("Stage1DecimationFor(%d) = nil error, want error", bad)
		}
	}
}

func TestChannelIQRateIsFixed(t *testing.T) {
	if ChannelIQRate != 25_000 {
		t.Errorf("ChannelIQRate = %d, want fixed 25000", ChannelIQRate)
	}
}
