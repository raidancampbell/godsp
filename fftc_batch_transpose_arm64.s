#include "textflag.h"

// Hand-written NEON 4x4 complex64 transpose (avo has no arm64 backend).
// complex64 = one 64-bit lane. NO WORD encodings — pure data movement
// (VLD1/VTRN1/VTRN2/VST1 .D2). VTRN1 Vb.D2,Va.D2,Vd.D2 -> {Va[0],Vb[0]};
// VTRN2 -> {Va[1],Vb[1]}. Register lists in VLD1/VST1 are consecutive.
//
// The 4-block transpose of a 4x4 complex64 tile:
//   rows r0..r3 each hold 4 complex64 (bins i..i+3) as {lo=.D2 bins i,i+1; hi=.D2 bins i+2,i+3}
//   want columns c0..c3: c_bin holds {r0[bin],r1[bin],r2[bin],r3[bin]}
// Do it as two independent 2x2 .D2 transposes per half.

// func packBatch4NEON(dst, src []complex64, n int)
// src lane L row starts at src + L*n*8 bytes; within a row, bin i at +i*8.
// dst AoSoA: bin i's 4 lanes contiguous at dst + i*4*8 = dst + i*32.
TEXT ·packBatch4NEON(SB), NOSPLIT, $0-56
	MOVD dst_base+0(FP), R0
	MOVD src_base+24(FP), R1
	MOVD n+48(FP), R2

	LSL  $3, R2, R3          // R3 = n*8 = lane row stride in src (bytes)
	AND  $~3, R2, R4         // R4 = nv = n &^ 3 (process 4 bins/iter)

	MOVD $0, R5              // i
ploop:
	CMP  R4, R5
	BGE  ptail
	// Per-lane src pointers at bin i: src + L*R3 + i*8.
	LSL  $3, R5, R6
	ADD  R1, R6, R7          // lane0 addr
	ADD  R7, R3, R8          // lane1
	ADD  R8, R3, R9          // lane2
	ADD  R9, R3, R10         // lane3
	// Load 4 complex64 per lane (2 .D2 each): V0/V1=lane0, V2/V3=lane1, V4/V5=lane2, V6/V7=lane3.
	VLD1 (R7), [V0.D2, V1.D2]
	VLD1 (R8), [V2.D2, V3.D2]
	VLD1 (R9), [V4.D2, V5.D2]
	VLD1 (R10), [V6.D2, V7.D2]
	// bin i   = {l0.lo[0], l1.lo[0], l2.lo[0], l3.lo[0]}
	// bin i+1 = {*.lo[1]}, bin i+2 = {*.hi[0]}, bin i+3 = {*.hi[1]}
	VTRN1 V2.D2, V0.D2, V16.D2   // {l0.lo[0], l1.lo[0]}  -> bin i, lanes 0,1
	VTRN1 V6.D2, V4.D2, V17.D2   // {l2.lo[0], l3.lo[0]}  -> bin i, lanes 2,3
	VTRN2 V2.D2, V0.D2, V18.D2   // {l0.lo[1], l1.lo[1]}  -> bin i+1, lanes 0,1
	VTRN2 V6.D2, V4.D2, V19.D2   // {l2.lo[1], l3.lo[1]}  -> bin i+1, lanes 2,3
	VTRN1 V3.D2, V1.D2, V20.D2   // {l0.hi[0], l1.hi[0]}  -> bin i+2, lanes 0,1
	VTRN1 V7.D2, V5.D2, V21.D2   // {l2.hi[0], l3.hi[0]}  -> bin i+2, lanes 2,3
	VTRN2 V3.D2, V1.D2, V22.D2   // {l0.hi[1], l1.hi[1]}  -> bin i+3, lanes 0,1
	VTRN2 V7.D2, V5.D2, V23.D2   // {l2.hi[1], l3.hi[1]}  -> bin i+3, lanes 2,3
	// dst bin i at dst + i*32: each bin is 4 complex64 = 32 bytes = 2 .D2.
	LSL  $5, R5, R11
	ADD  R0, R11, R11
	VST1 [V16.D2, V17.D2], (R11)          // bin i   (lanes 0..3)
	ADD  $32, R11, R11
	VST1 [V18.D2, V19.D2], (R11)          // bin i+1
	ADD  $32, R11, R11
	VST1 [V20.D2, V21.D2], (R11)          // bin i+2
	ADD  $32, R11, R11
	VST1 [V22.D2, V23.D2], (R11)          // bin i+3
	ADD  $4, R5, R5
	B    ploop
ptail:
	// Scalar tail for i in [nv, n): dst[i*4+lane] = src[lane*n+i].
	CMP  R2, R5
	BGE  pdone
	LSL  $3, R5, R6
	ADD  R1, R6, R7          // src + i*8
	LSL  $5, R5, R11
	ADD  R0, R11, R11        // dst + i*32
	MOVD $0, R12             // lane
ptaillane:
	CMP  $4, R12
	BGE  ptailnext
	MOVD (R7), R13           // src[lane*n + i] (8 bytes = 1 complex64)
	MOVD R13, (R11)
	ADD  R3, R7, R7          // += n*8 (next lane row)
	ADD  $8, R11, R11        // next AoSoA slot
	ADD  $1, R12, R12
	B    ptaillane
ptailnext:
	ADD  $1, R5, R5
	B    ptail
pdone:
	RET

// func unpackBatch4NEON(dst, src []complex64, n int)
// Inverse: src is AoSoA (bin i's 4 lanes at src + i*32), dst lane L row at dst + L*n*8.
TEXT ·unpackBatch4NEON(SB), NOSPLIT, $0-56
	MOVD dst_base+0(FP), R0
	MOVD src_base+24(FP), R1
	MOVD n+48(FP), R2

	LSL  $3, R2, R3          // R3 = n*8 = lane row stride in dst
	AND  $~3, R2, R4         // nv

	MOVD $0, R5              // i
uloop:
	CMP  R4, R5
	BGE  utail
	// src bin block at src + i*32: 4 bins x 4 lanes.
	LSL  $5, R5, R6
	ADD  R1, R6, R7
	VLD1 (R7), [V0.D2, V1.D2]     // bin i  : lanes {0,1},{2,3}
	ADD  $32, R7, R7
	VLD1 (R7), [V2.D2, V3.D2]     // bin i+1
	ADD  $32, R7, R7
	VLD1 (R7), [V4.D2, V5.D2]     // bin i+2
	ADD  $32, R7, R7
	VLD1 (R7), [V6.D2, V7.D2]     // bin i+3
	// Regroup by lane: lane0 row = {bin_i, bin_{i+1}, bin_{i+2}, bin_{i+3}} lane-0 elements.
	VTRN1 V2.D2, V0.D2, V16.D2   // {bin_i.l0, bin_{i+1}.l0} -> lane0 .lo
	VTRN1 V6.D2, V4.D2, V17.D2   // {bin_{i+2}.l0, bin_{i+3}.l0} -> lane0 .hi
	VTRN2 V2.D2, V0.D2, V18.D2   // {bin_i.l1, bin_{i+1}.l1} -> lane1 .lo
	VTRN2 V6.D2, V4.D2, V19.D2   // lane1 .hi
	VTRN1 V3.D2, V1.D2, V20.D2   // {bin_i.l2, bin_{i+1}.l2} -> lane2 .lo
	VTRN1 V7.D2, V5.D2, V21.D2   // lane2 .hi
	VTRN2 V3.D2, V1.D2, V22.D2   // lane3 .lo
	VTRN2 V7.D2, V5.D2, V23.D2   // lane3 .hi
	// Store each lane row at dst + lane*R3 + i*8.
	LSL  $3, R5, R11
	ADD  R0, R11, R12        // lane0 dst + i*8
	VST1 [V16.D2, V17.D2], (R12)
	ADD  R12, R3, R12
	VST1 [V18.D2, V19.D2], (R12)  // lane1
	ADD  R12, R3, R12
	VST1 [V20.D2, V21.D2], (R12)  // lane2
	ADD  R12, R3, R12
	VST1 [V22.D2, V23.D2], (R12)  // lane3
	ADD  $4, R5, R5
	B    uloop
utail:
	CMP  R2, R5
	BGE  udone
	LSL  $5, R5, R6
	ADD  R1, R6, R7          // src + i*32
	LSL  $3, R5, R11
	ADD  R0, R11, R12        // dst + i*8
	MOVD $0, R14             // lane
utaillane:
	CMP  $4, R14
	BGE  utailnext
	MOVD (R7), R13           // src[i*4 + lane]
	MOVD R13, (R12)          // dst[lane*n + i]
	ADD  $8, R7, R7          // next AoSoA slot
	ADD  R3, R12, R12        // += n*8 (next lane row)
	ADD  $1, R14, R14
	B    utaillane
utailnext:
	ADD  $1, R5, R5
	B    utail
udone:
	RET
