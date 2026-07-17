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
