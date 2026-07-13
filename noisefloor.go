package dsp

import "math"

const (
	// noiseFloorInitDB is the initial floor estimate in dBFS. A -60 dBFS
	// starting point lets a signal opening in the first few blocks report a
	// plausible SNR before the estimate converges.
	noiseFloorInitDB = -60.0

	// noiseFloorDecayTau is the slow-decay time constant (seconds) for the
	// floor's rise when channel power stays elevated. 5 s keeps the floor
	// stable while a signal is present.
	noiseFloorDecayTau = 5.0
)

// NoiseFloorTracker estimates a channel's noise floor in dBFS from per-block
// power measurements, using a fast-attack / slow-decay envelope but WITHOUT a
// gating input: it runs continuously.
//
// It exists for demodulation paths that have no power-gated squelch to
// converge a floor from — their lifecycle is driven by decoded frames, not a
// power gate. With a zero floor, SNR = signalDB - noiseDB would make a clean
// -20 dBFS signal report as -20 dB SNR. This tracker gives those paths a real
// floor: it latches onto the quiet inter-burst / inter-slot blocks (fast
// attack) while resisting brief signal power (slow decay), so
// signalDB - NoiseFloorDB yields a positive SNR for a strong signal.
type NoiseFloorTracker struct {
	floorDB float64
	alpha   float64 // slow-decay EWM coefficient
}

// NewNoiseFloorTracker returns a tracker smoothed with noiseFloorDecayTau at the
// channel block rate (ChannelIQRate / BlockSamples blocks per second).
func NewNoiseFloorTracker() *NoiseFloorTracker {
	blockRate := float64(ChannelIQRate) / float64(BlockSamples)
	return &NoiseFloorTracker{
		floorDB: noiseFloorInitDB,
		alpha:   1.0 - math.Exp(-1.0/(noiseFloorDecayTau*blockRate)),
	}
}

// Update folds one block's power (dBFS) into the floor estimate. A block below
// the current floor pulls it down immediately (fast attack); a block above it
// raises the floor only slowly via the EWM (slow decay).
func (n *NoiseFloorTracker) Update(powerDB float64) {
	if powerDB < n.floorDB {
		n.floorDB = powerDB
		return
	}
	n.floorDB = n.alpha*powerDB + (1.0-n.alpha)*n.floorDB
}

// NoiseFloorDB returns the current noise floor estimate in dBFS.
func (n *NoiseFloorTracker) NoiseFloorDB() float64 {
	return n.floorDB
}
