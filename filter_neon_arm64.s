#include "textflag.h"

// Hand-written NEON complex-FIR dot (avo has no arm64 backend).
// out[0] = Σ_j winR[j]·taps[j], out[1] = Σ_j winI[j]·taps[j], j in [0,n), n%4==0.
// taps is REAL float32; winR/winI are the split real/imag windows. Per 4-tap
// block: load taps/winR/winI (VLD1 .S4), VFMLA accR += winR·taps and
// accI += winI·taps. After the loop, horizontally reduce each 4-lane
// accumulator to a scalar via two pairwise adds (FADDP, WORD-encoded because
// Go's arm64 assembler does not expose it) and store to out.
//
// WORD encodings (decode-verified):
//   FADDP V0.4S, V0.4S, V0.4S  -> 0x6e20d400   (pairwise: {l0+l1,l2+l3,l0+l1,l2+l3})
//   FADDP S0, V0.2S            -> 0x7e30d800   (final scalar (l0+l1)+(l2+l3) in F0)
//   FADDP V1.4S, V1.4S, V1.4S  -> 0x6e21d421
//   FADDP S1, V1.2S            -> 0x7e30d821

// func complexFIRDotNEON(taps, winR, winI []float32, n int, out *[2]float32)
TEXT ·complexFIRDotNEON(SB), NOSPLIT, $0-88
	MOVD taps_base+0(FP), R0
	MOVD winR_base+24(FP), R1
	MOVD winI_base+48(FP), R2
	MOVD n+72(FP), R3
	MOVD out+80(FP), R4

	VEOR V0.B16, V0.B16, V0.B16   // accR
	VEOR V1.B16, V1.B16, V1.B16   // accI

	MOVD $0, R5                   // j
loop:
	CMP  R3, R5
	BGE  reduce
	VLD1.P 16(R0), [V2.S4]        // taps[j..j+3], post-inc R0 by 16
	VLD1.P 16(R1), [V3.S4]        // winR
	VLD1.P 16(R2), [V4.S4]        // winI
	VFMLA V2.S4, V3.S4, V0.S4     // accR += winR·taps
	VFMLA V2.S4, V4.S4, V1.S4     // accI += winI·taps
	ADD  $4, R5, R5
	B    loop
reduce:
	WORD $0x6e20d400              // FADDP V0.4S,V0.4S,V0.4S
	WORD $0x7e30d800              // FADDP S0,V0.2S    -> accR in F0
	WORD $0x6e21d421              // FADDP V1.4S,V1.4S,V1.4S
	WORD $0x7e30d821              // FADDP S1,V1.2S    -> accI in F1
	FMOVS F0, (R4)                // out[0]
	FMOVS F1, 4(R4)               // out[1]
	RET

// Hand-written NEON real-FIR dot (avo has no arm64 backend).
// out = Σ_j win[j]·taps[j], j in [0,n), n%4==0. Per 4-tap block: load
// taps/win (VLD1 .S4), VFMLA acc += win·taps. After the loop, horizontally
// reduce the single 4-lane accumulator to a scalar via two pairwise adds
// (FADDP, WORD-encoded — same reduction the complex kernel uses) and store.
//
// WORD encodings (decode-verified, reused from complexFIRDotNEON):
//   FADDP V0.4S, V0.4S, V0.4S  -> 0x6e20d400   (pairwise: {l0+l1,l2+l3,l0+l1,l2+l3})
//   FADDP S0, V0.2S            -> 0x7e30d800   (final scalar (l0+l1)+(l2+l3) in F0)

// func firDotRealNEON(taps, win []float32, n int, out *float32)
TEXT ·firDotRealNEON(SB), NOSPLIT, $0-64
	MOVD taps_base+0(FP), R0
	MOVD win_base+24(FP), R1
	MOVD n+48(FP), R2
	MOVD out+56(FP), R3

	VEOR V0.B16, V0.B16, V0.B16   // acc

	MOVD $0, R4                   // j
firloop:
	CMP  R2, R4
	BGE  firreduce
	VLD1.P 16(R0), [V1.S4]        // taps[j..j+3], post-inc R0 by 16
	VLD1.P 16(R1), [V2.S4]        // win
	VFMLA V1.S4, V2.S4, V0.S4     // acc += win·taps
	ADD  $4, R4, R4
	B    firloop
firreduce:
	WORD $0x6e20d400              // FADDP V0.4S,V0.4S,V0.4S
	WORD $0x7e30d800              // FADDP S0,V0.2S    -> acc in F0
	FMOVS F0, (R3)                // out
	RET

// Hand-written NEON batched complex-FIR dot: FOUR complex outputs sharing the
// SAME taps (the transpose of foldHop4 — there four windows share the loaded
// proto, here four windows share the loaded taps). Lane L's window starts
// stride float32 ELEMENTS after lane L-1 in the SoA winR/winI buffers:
//   out[2L]   = Σ_j winR[L*stride+j]·taps[j]
//   out[2L+1] = Σ_j winI[L*stride+j]·taps[j]   j in [0,nv), nv%4==0.
// Eight accumulators (accR/accI per lane, V16..V23) keep four complex FMA pairs
// in flight; the tap block (V2) is loaded ONCE per 4-tap block and reused by all
// four lanes. Each lane's arithmetic is byte-identical to complexFIRDotNEON over
// that window — only the tap load is hoisted. The Go dispatch handles the nv%4
// scalar tail per lane.
//
// Reduction reuses ONLY the decode-verified FADDP encodings from
// complexFIRDotNEON above: each lane's accR is moved into V0 and accI into V1
// (VMOV .B16, a real mnemonic — no new encoding), then the same 0x6e20d400 /
// 0x7e30d800 (V0) and 0x6e21d421 / 0x7e30d821 (V1) pairwise adds collapse them
// to scalars. NO new WORD encodings are introduced.

// func complexFIRDot4NEON(taps, winR, winI []float32, stride int, nv int, out *[8]float32)
TEXT ·complexFIRDot4NEON(SB), NOSPLIT, $0-96
	MOVD taps_base+0(FP), R0
	MOVD winR_base+24(FP), R1     // lane0 winR
	MOVD winI_base+48(FP), R2     // lane0 winI
	MOVD stride+72(FP), R3
	MOVD nv+80(FP), R4
	MOVD out+88(FP), R5

	LSL  $2, R3, R6               // R6 = stride*4 bytes = per-lane byte stride
	ADD  R6, R1, R7              // lane1 winR = winR0 + stride
	ADD  R6, R7, R8              // lane2 winR
	ADD  R6, R8, R9              // lane3 winR
	ADD  R6, R2, R10             // lane1 winI = winI0 + stride
	ADD  R6, R10, R11            // lane2 winI
	ADD  R6, R11, R12            // lane3 winI

	VEOR V16.B16, V16.B16, V16.B16   // lane0 accR
	VEOR V17.B16, V17.B16, V17.B16   // lane0 accI
	VEOR V18.B16, V18.B16, V18.B16   // lane1 accR
	VEOR V19.B16, V19.B16, V19.B16   // lane1 accI
	VEOR V20.B16, V20.B16, V20.B16   // lane2 accR
	VEOR V21.B16, V21.B16, V21.B16   // lane2 accI
	VEOR V22.B16, V22.B16, V22.B16   // lane3 accR
	VEOR V23.B16, V23.B16, V23.B16   // lane3 accI

	MOVD $0, R13                  // j
loop4:
	CMP  R4, R13
	BGE  reduce4
	VLD1.P 16(R0), [V2.S4]        // taps[j..j+3], shared across lanes
	VLD1.P 16(R1), [V3.S4]        // lane0 winR
	VLD1.P 16(R2), [V4.S4]        // lane0 winI
	VFMLA V2.S4, V3.S4, V16.S4    // accR0 += winR·taps
	VFMLA V2.S4, V4.S4, V17.S4    // accI0 += winI·taps
	VLD1.P 16(R7), [V3.S4]        // lane1 winR
	VLD1.P 16(R10), [V4.S4]       // lane1 winI
	VFMLA V2.S4, V3.S4, V18.S4
	VFMLA V2.S4, V4.S4, V19.S4
	VLD1.P 16(R8), [V3.S4]        // lane2 winR
	VLD1.P 16(R11), [V4.S4]       // lane2 winI
	VFMLA V2.S4, V3.S4, V20.S4
	VFMLA V2.S4, V4.S4, V21.S4
	VLD1.P 16(R9), [V3.S4]        // lane3 winR
	VLD1.P 16(R12), [V4.S4]       // lane3 winI
	VFMLA V2.S4, V3.S4, V22.S4
	VFMLA V2.S4, V4.S4, V23.S4
	ADD  $4, R13, R13
	B    loop4
reduce4:
	// Lane 0.
	VMOV V16.B16, V0.B16
	VMOV V17.B16, V1.B16
	WORD $0x6e20d400              // FADDP V0.4S,V0.4S,V0.4S
	WORD $0x7e30d800              // FADDP S0,V0.2S    -> r0 in F0
	WORD $0x6e21d421              // FADDP V1.4S,V1.4S,V1.4S
	WORD $0x7e30d821              // FADDP S1,V1.2S    -> i0 in F1
	FMOVS F0, (R5)                // out[0]
	FMOVS F1, 4(R5)               // out[1]
	// Lane 1.
	VMOV V18.B16, V0.B16
	VMOV V19.B16, V1.B16
	WORD $0x6e20d400
	WORD $0x7e30d800
	WORD $0x6e21d421
	WORD $0x7e30d821
	FMOVS F0, 8(R5)               // out[2]
	FMOVS F1, 12(R5)              // out[3]
	// Lane 2.
	VMOV V20.B16, V0.B16
	VMOV V21.B16, V1.B16
	WORD $0x6e20d400
	WORD $0x7e30d800
	WORD $0x6e21d421
	WORD $0x7e30d821
	FMOVS F0, 16(R5)              // out[4]
	FMOVS F1, 20(R5)              // out[5]
	// Lane 3.
	VMOV V22.B16, V0.B16
	VMOV V23.B16, V1.B16
	WORD $0x6e20d400
	WORD $0x7e30d800
	WORD $0x6e21d421
	WORD $0x7e30d821
	FMOVS F0, 24(R5)              // out[6]
	FMOVS F1, 28(R5)              // out[7]
	RET
