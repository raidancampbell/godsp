#include "textflag.h"

// Hand-written NEON polyphase-fold kernels (avo has no arm64 backend).
// The fold is fold[i] = Σ_q win[q·k+i]·proto[q·k+i]. proto is REAL float32
// scaling a complex64 win sample, so every op is a real·complex FMA — no
// complex multiply, no twiddle, NO WORD-encoded float ops. All mnemonics are
// real: VLD1, VZIP1/VZIP2 (dup each proto real into its {re,im} pair), VFMLA
// (Vacc += Vn·Vd), VST1, VEOR (zero an accumulator).
//
// Layout: 4 bins = 8 float32 = 2 Q-regs. Per tap q, load 4 proto floats into
// V4, VZIP1 V4,V4 -> {p0,p0,p1,p1}=V5, VZIP2 V4,V4 -> {p2,p2,p3,p3}=V6, load 8
// win floats into V2 (bins 0,1) and V3 (bins 2,3), then acc0 += V2·V5,
// acc1 += V3·V6.

// func foldHopNEON(win []complex64, proto []float32, fold []complex64, kv, k, p int)
TEXT ·foldHopNEON(SB), NOSPLIT, $0-96
	MOVD win_base+0(FP), R0     // win base
	MOVD proto_base+24(FP), R1  // proto base
	MOVD fold_base+48(FP), R2   // fold base
	MOVD kv+72(FP), R3          // kv (multiple of 4)
	MOVD k+80(FP), R4           // k
	MOVD p+88(FP), R5           // p (taps)

	LSL  $3, R4, R6             // R6 = k*8  bytes = complex64 tap stride in win/fold
	LSL  $2, R4, R7             // R7 = k*4  bytes = float32 tap stride in proto

	MOVD $0, R8                 // i = bin index, steps by 4
binloop:
	CMP  R3, R8
	BGE  done

	// Zero the two accumulators for this 4-bin block.
	VEOR V16.B16, V16.B16, V16.B16   // acc0 (bins i, i+1)
	VEOR V17.B16, V17.B16, V17.B16   // acc1 (bins i+2, i+3)

	// Running tap pointers: win + i*8, proto + i*4.
	LSL  $3, R8, R9
	ADD  R0, R9, R9             // R9 = win + i*8
	LSL  $2, R8, R10
	ADD  R1, R10, R10           // R10 = proto + i*4

	MOVD $0, R11                // q = 0
taploop:
	CMP  R5, R11
	BGE  tapdone

	VLD1 (R10), [V4.S4]                // 4 proto reals
	VZIP1 V4.S4, V4.S4, V5.S4          // {p0,p0,p1,p1}
	VZIP2 V4.S4, V4.S4, V6.S4          // {p2,p2,p3,p3}
	VLD1 (R9), [V2.S4, V3.S4]          // 8 win floats: V2=bins i,i+1  V3=bins i+2,i+3
	VFMLA V5.S4, V2.S4, V16.S4         // acc0 += win·dup(proto_lo)
	VFMLA V6.S4, V3.S4, V17.S4         // acc1 += win·dup(proto_hi)

	ADD  R6, R9, R9            // advance win by k complex64
	ADD  R7, R10, R10          // advance proto by k float32
	ADD  $1, R11, R11
	B    taploop
tapdone:
	// Store 4 complex64 results at fold + i*8.
	LSL  $3, R8, R12
	ADD  R2, R12, R12
	VST1 [V16.S4, V17.S4], (R12)

	ADD  $4, R8, R8
	B    binloop
done:
	RET

// func foldHop4NEON(win []complex64, proto []float32, fold []complex64, kv, k, p, r int)
TEXT ·foldHop4NEON(SB), NOSPLIT, $0-104
	MOVD win_base+0(FP), R0     // lane 0 win base
	MOVD proto_base+24(FP), R1
	MOVD fold_base+48(FP), R2   // lane 0 fold base
	MOVD kv+72(FP), R3
	MOVD k+80(FP), R4
	MOVD p+88(FP), R5
	MOVD r+96(FP), R14          // r = lane stride in win (complex64)

	LSL  $3, R4, R6             // R6 = k*8 = complex64 tap stride (win) & lane stride (fold)
	LSL  $2, R4, R7             // R7 = k*4 = float32 tap stride (proto)
	LSL  $3, R14, R15           // R15 = r*8 = lane stride in win (bytes)

	MOVD $0, R8                 // i = bin index, steps by 4
binloop4:
	CMP  R3, R8
	BGE  done4

	VEOR V16.B16, V16.B16, V16.B16
	VEOR V17.B16, V17.B16, V17.B16
	VEOR V18.B16, V18.B16, V18.B16
	VEOR V19.B16, V19.B16, V19.B16
	VEOR V20.B16, V20.B16, V20.B16
	VEOR V21.B16, V21.B16, V21.B16
	VEOR V22.B16, V22.B16, V22.B16
	VEOR V23.B16, V23.B16, V23.B16

	// Per-lane win tap pointers at this bin block: win + i*8 + L*r*8.
	LSL  $3, R8, R9
	ADD  R0, R9, R9            // R9  = lane0 win + i*8
	ADD  R9, R15, R10          // R10 = lane1
	ADD  R10, R15, R11         // R11 = lane2
	ADD  R11, R15, R12         // R12 = lane3

	// proto tap pointer: proto + i*4.
	LSL  $2, R8, R13
	ADD  R1, R13, R13

	MOVD $0, R19               // q = 0
taploop4:
	CMP  R5, R19
	BGE  tapdone4

	// Shared proto load + zip (once per tap, reused by all four lanes).
	VLD1 (R13), [V4.S4]
	VZIP1 V4.S4, V4.S4, V5.S4
	VZIP2 V4.S4, V4.S4, V6.S4

	// Lane 0.
	VLD1 (R9), [V2.S4, V3.S4]
	VFMLA V5.S4, V2.S4, V16.S4
	VFMLA V6.S4, V3.S4, V17.S4
	// Lane 1.
	VLD1 (R10), [V2.S4, V3.S4]
	VFMLA V5.S4, V2.S4, V18.S4
	VFMLA V6.S4, V3.S4, V19.S4
	// Lane 2.
	VLD1 (R11), [V2.S4, V3.S4]
	VFMLA V5.S4, V2.S4, V20.S4
	VFMLA V6.S4, V3.S4, V21.S4
	// Lane 3.
	VLD1 (R12), [V2.S4, V3.S4]
	VFMLA V5.S4, V2.S4, V22.S4
	VFMLA V6.S4, V3.S4, V23.S4

	ADD  R6, R9, R9
	ADD  R6, R10, R10
	ADD  R6, R11, R11
	ADD  R6, R12, R12
	ADD  R7, R13, R13
	ADD  $1, R19, R19
	B    taploop4
tapdone4:
	// Store each lane at fold + L*k*8, bin offset i*8.
	LSL  $3, R8, R20
	ADD  R2, R20, R20          // lane0 fold + i*8
	VST1 [V16.S4, V17.S4], (R20)
	ADD  R20, R6, R21          // lane1
	VST1 [V18.S4, V19.S4], (R21)
	ADD  R21, R6, R22          // lane2
	VST1 [V20.S4, V21.S4], (R22)
	ADD  R22, R6, R23          // lane3
	VST1 [V22.S4, V23.S4], (R23)

	ADD  $4, R8, R8
	B    binloop4
done4:
	RET
