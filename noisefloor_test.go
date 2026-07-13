package dsp

import (
	"math"
	"testing"
)

// TestNoiseFloorTracker_FastAttack verifies a single low-power block immediately
// pulls the floor down (fast attack), so the tracker latches onto quiet blocks
// even when they are rare.
func TestNoiseFloorTracker_FastAttack(t *testing.T) {
	nf := NewNoiseFloorTracker()
	// Warm up high, then feed one low block.
	for i := 0; i < 100; i++ {
		nf.Update(-20.0)
	}
	nf.Update(-70.0)
	if got := nf.NoiseFloorDB(); got > -69.9 {
		t.Errorf("after low block, floor=%.2f dB, want ~-70 (fast attack)", got)
	}
}

// TestNoiseFloorTracker_ConvergesToQuietFloor verifies that on a channel with
// strong voice blocks interspersed with quiet blocks, the floor converges to the
// quiet level (not the signal level), so PowerDB-NoiseDB yields a positive SNR.
func TestNoiseFloorTracker_ConvergesToQuietFloor(t *testing.T) {
	nf := NewNoiseFloorTracker()
	const signalDB = -18.0
	const noiseFloorDB = -60.0
	// Alternate strong and quiet blocks for a few seconds at the block rate.
	for i := 0; i < 1000; i++ {
		if i%4 == 0 {
			nf.Update(noiseFloorDB)
		} else {
			nf.Update(signalDB)
		}
	}
	floor := nf.NoiseFloorDB()
	if math.Abs(floor-noiseFloorDB) > 3.0 {
		t.Errorf("floor=%.2f dB, want within 3 dB of %.1f", floor, noiseFloorDB)
	}
	// The reported SNR (signal - floor) must be strongly positive.
	if snr := signalDB - floor; snr < 30.0 {
		t.Errorf("SNR=%.2f dB, want >=30 for a clean strong signal", snr)
	}
}

// TestNoiseFloorTracker_SlowDecay verifies the floor rises only slowly when the
// channel power stays elevated (EWM decay), so a brief burst of quiet followed
// by sustained signal does not immediately inflate the floor to the signal level.
func TestNoiseFloorTracker_SlowDecay(t *testing.T) {
	nf := NewNoiseFloorTracker()
	nf.Update(-60.0) // establish a low floor
	// One block of sustained -20 signal must not jump the floor up to -20.
	nf.Update(-20.0)
	if got := nf.NoiseFloorDB(); got > -55.0 {
		t.Errorf("after one high block, floor=%.2f dB, want still near -60 (slow decay)", got)
	}
}
