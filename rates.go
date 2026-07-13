package dsp

import "fmt"

// Canonical DSP rate constants. All packages should reference these
// instead of redeclaring their own copies.
const (
	// ChannelIQRate is the narrowband channel rate (25 kSPS). This is the FIXED
	// rate the downstream demod, gating, recording, and audio resampling all
	// depend on; it must NOT move when the wideband capture rate changes. Stage-1
	// decimation is derived from the configured rate to land here (Stage1DecimationFor).
	ChannelIQRate = 25_000

	// Stage2Decimation from the 250 kSPS intermediate rate to ChannelIQRate.
	Stage2Decimation = 10

	// WidebandRate is the nominal/default IQ capture rate (10 MSPS). The live path
	// reads the real rate from config; this is only the default for tools and tests.
	WidebandRate = 10_000_000

	// AudioRate is the final audio playback and recording sample rate (8 kHz).
	AudioRate = 8000

	// ResampleUp is the rational resampler interpolation factor (ChannelIQRate → AudioRate).
	ResampleUp = 8

	// ResampleDown is the rational resampler decimation factor (ChannelIQRate → AudioRate).
	ResampleDown = 25

	// BlockSamples is the number of narrowband IQ samples per processing block.
	// At ChannelIQRate=25000, this is 10 ms per block.
	BlockSamples = 250

	// ReadBlockSize is the number of complex samples per wideband IQ read.
	// 512K halves the per-second fork-join and decode-pump wakeup count versus
	// 256K. One block is 14.56 ms at 36 MSPS / 10.92 ms at 48 MSPS, which is
	// negligible against the 2 s pre-roll. The wideband ring and source block
	// pool derive their depth from this value so each stays near 512 MiB.
	ReadBlockSize = 524288
)

// stage1OutputRate is the intermediate rate between stage 1 and stage 2:
// ChannelIQRate * Stage2Decimation = 250 kSPS, held constant across capture rates.
const stage1OutputRate = ChannelIQRate * Stage2Decimation

// Stage1DecimationFor returns the stage-1 decimation factor that lands the given
// wideband sampleRate at the fixed 250 kSPS intermediate rate (hence 25 kSPS
// channel rate). It errors unless sampleRate is a positive multiple of 250000 Hz,
// because a fractional decimation would move the channel IQ off the rate the
// downstream demod and audio resampler assume (the original fractional-rate
// failure that produced no decodes).
func Stage1DecimationFor(sampleRate uint32) (int, error) {
	if sampleRate == 0 || sampleRate%stage1OutputRate != 0 {
		return 0, fmt.Errorf("sample rate %d Hz must be a positive multiple of %d Hz (Stage2Decimation*ChannelIQRate)", sampleRate, stage1OutputRate)
	}
	return int(sampleRate / stage1OutputRate), nil
}
