#include "textflag.h"

// ---------------------------------------------------------------------------
// WORD-encoding recipe for NEON float-vector ops (same as fftc_batch_neon).
//
// Base encodings (.4S arrangement):
//   FADD Vd.4S, Vn.4S, Vm.4S  -> 0x4E20D400
//   FSUB Vd.4S, Vn.4S, Vm.4S  -> 0x4EA0D400
//   FMUL Vd.4S, Vn.4S, Vm.4S  -> 0x6E20DC00
//
// Result = base | (Rm << 16) | (Rn << 5) | Rd
// ---------------------------------------------------------------------------

// Sign masks for complex operations (distinct symbols from fftc_batch_neon).
DATA fftc_bfly_negre<>+0(SB)/4, $0x80000000
DATA fftc_bfly_negre<>+4(SB)/4, $0x00000000
DATA fftc_bfly_negre<>+8(SB)/4, $0x80000000
DATA fftc_bfly_negre<>+12(SB)/4, $0x00000000
GLOBL fftc_bfly_negre<>(SB), RODATA|NOPTR, $16

DATA fftc_bfly_negim<>+0(SB)/4, $0x00000000
DATA fftc_bfly_negim<>+4(SB)/4, $0x80000000
DATA fftc_bfly_negim<>+8(SB)/4, $0x00000000
DATA fftc_bfly_negim<>+12(SB)/4, $0x80000000
GLOBL fftc_bfly_negim<>(SB), RODATA|NOPTR, $16

// Radix-3 constants: -0.5 and sqrt(3)/2
DATA fftc_bfly_r3const<>+0(SB)/4, $0xbf000000  // -0.5
DATA fftc_bfly_r3const<>+4(SB)/4, $0x3f5db3d7  // sqrt(3)/2 = 0.8660254037844386
GLOBL fftc_bfly_r3const<>(SB), RODATA|NOPTR, $8

// func bfly3NEON(out, tw []complex64, mv, m int)
//
// Radix-3 DIT butterfly for NEON (2 complex64/iter).
// Frame: out@0(24 bytes), tw@24(24 bytes), mv@48(8 bytes), m@56(8 bytes) = $0-64.
//
// Per iteration k (step 2), load 2 complex from:
//   x0 = out + k*8
//   x1 = x0 + m*8
//   x2 = x1 + m*8
// Twiddles:
//   tw1 = tw + k*8
//   tw2 = tw1 + m*8
//
// Arithmetic (from bfly3Kernel):
//   t1 = x1 * tw1
//   t2 = x2 * tw2
//   sum = t1 + t2
//   dif = t1 - t2
//   y0 = x0 + sum
//   a = x0 - 0.5*sum
//   w = (sqrt(3)/2)*dif
//   rot = w * (-i)
//   y1 = a + rot
//   y2 = a - rot
TEXT ·bfly3NEON(SB), NOSPLIT, $0-64
	MOVD out_base+0(FP), R0
	MOVD tw_base+24(FP), R1
	MOVD mv+48(FP), R2
	MOVD m+56(FP), R3

	// R4 = m*8 (byte stride between radix blocks)
	LSL  $3, R3, R4

	// Load sign masks and constants
	MOVD $fftc_bfly_negre<>(SB), R13
	VLD1 (R13), [V30.S4]                        // V30 = negre
	MOVD $fftc_bfly_negim<>(SB), R13
	VLD1 (R13), [V31.S4]                        // V31 = negim
	MOVD $fftc_bfly_r3const<>(SB), R13
	VLD1 (R13), [V29.S2]
	VDUP V29.S[0], V28.S4                       // V28 = {-0.5,-0.5,-0.5,-0.5}
	VDUP V29.S[1], V29.S4                       // V29 = {sqrt(3)/2, ...}

	MOVD $0, R5                                 // k = 0

bfly3_loop:
	CMP  R2, R5
	BGE  bfly3_done

	// Compute element pointers
	// x0p = out + k*8
	ADD  R5<<3, R0, R6                          // R6 = x0p
	// x1p = x0p + m*8
	ADD  R4, R6, R7                             // R7 = x1p
	// x2p = x1p + m*8
	ADD  R4, R7, R8                             // R8 = x2p

	// tw1p = tw + k*8
	ADD  R5<<3, R1, R9                          // R9 = tw1p
	// tw2p = tw1p + m*8
	ADD  R4, R9, R10                            // R10 = tw2p

	// Load x0, x1, x2 (each is 2 complex64 in one .S4 register)
	VLD1 (R6), [V0.S4]                          // V0 = x0
	VLD1 (R7), [V1.S4]                          // V1 = x1
	VLD1 (R8), [V2.S4]                          // V2 = x2

	// Load tw1, tw2
	VLD1 (R9), [V3.S4]                          // V3 = tw1
	VLD1 (R10), [V4.S4]                         // V4 = tw2

	// --- t1 = x1 * tw1 (complex multiply) ---
	// Duplicate real and imag parts
	VTRN1 V1.S4, V1.S4, V5.S4                   // V5 = {x1r0,x1r0,x1r1,x1r1}
	VTRN2 V1.S4, V1.S4, V6.S4                   // V6 = {x1i0,x1i0,x1i1,x1i1}
	VREV64 V3.S4, V7.S4                         // V7 = tw1_swap
	WORD $0x6E23DCA8     // FMUL V8.4S, V5.4S, V3.4S   (xr*tw)
	WORD $0x6E27DCC9     // FMUL V9.4S, V6.4S, V7.4S   (xi*tw_swap)
	VEOR V30.B16, V9.B16, V9.B16                // negate real lanes
	WORD $0x4E29D50A     // FADD V10.4S, V8.4S, V9.4S  (t1)

	// --- t2 = x2 * tw2 (complex multiply) ---
	VTRN1 V2.S4, V2.S4, V5.S4                   // V5 = {x2r0,x2r0,x2r1,x2r1}
	VTRN2 V2.S4, V2.S4, V6.S4                   // V6 = {x2i0,x2i0,x2i1,x2i1}
	VREV64 V4.S4, V7.S4                         // V7 = tw2_swap
	WORD $0x6E24DCA8     // FMUL V8.4S, V5.4S, V4.4S   (xr*tw)
	WORD $0x6E27DCC9     // FMUL V9.4S, V6.4S, V7.4S   (xi*tw_swap)
	VEOR V30.B16, V9.B16, V9.B16                // negate real lanes
	WORD $0x4E29D50B     // FADD V11.4S, V8.4S, V9.4S  (t2)

	// --- sum = t1 + t2; dif = t1 - t2 ---
	WORD $0x4E2BD54C     // FADD V12.4S, V10.4S, V11.4S (sum)
	WORD $0x4EABD54D     // FSUB V13.4S, V10.4S, V11.4S (dif)

	// --- y0 = x0 + sum ---
	WORD $0x4E2CD40E     // FADD V14.4S, V0.4S, V12.4S  (y0)

	// --- a = x0 - 0.5*sum  (computed as x0 + (-0.5)*sum) ---
	WORD $0x6E3CDD8F     // FMUL V15.4S, V12.4S, V28.4S ((-0.5)*sum)
	WORD $0x4E2FD410     // FADD V16.4S, V0.4S, V15.4S  (a)

	// --- w = (sqrt(3)/2)*dif ---
	WORD $0x6E3DDDB1     // FMUL V17.4S, V13.4S, V29.4S (w)

	// --- rot = w * (-i): VREV64 to swap, then XOR negim to negate imag ---
	VREV64 V17.S4, V18.S4                       // V18 = {wi,wr,wi,wr}
	VEOR V31.B16, V18.B16, V18.B16              // V18 = {wi,-wr,wi,-wr} = rot

	// --- y1 = a + rot; y2 = a - rot ---
	WORD $0x4E32D613     // FADD V19.4S, V16.4S, V18.4S (y1)
	WORD $0x4EB2D614     // FSUB V20.4S, V16.4S, V18.4S (y2)

	// Store results
	VST1 [V14.S4], (R6)                         // store y0 -> x0
	VST1 [V19.S4], (R7)                         // store y1 -> x1
	VST1 [V20.S4], (R8)                         // store y2 -> x2

	ADD  $2, R5, R5                             // k += 2 (2 complex64/iter)
	JMP  bfly3_loop

bfly3_done:
	RET

// func bfly4NEON(out, tw []complex64, mv, m int)
//
// Radix-4 DIT butterfly for NEON (2 complex64/iter).
// Frame: out@0(24 bytes), tw@24(24 bytes), mv@48(8 bytes), m@56(8 bytes) = $0-64.
//
// Per iteration k (step 2), load 2 complex from:
//   x0 = out + k*8
//   x1 = x0 + m*8
//   x2 = x1 + m*8
//   x3 = x2 + m*8
// Twiddles:
//   tw1 = tw + k*8
//   tw2 = tw1 + m*8
//   tw3 = tw2 + m*8
//
// Arithmetic (from bfly4Kernel):
//   s0 = x1 * tw1
//   s1 = x2 * tw2
//   s2 = x3 * tw3
//   d0 = x0 - s1
//   a0 = x0 + s1
//   a1 = s0 + s2
//   d1 = s0 - s2
//   y0 = a0 + a1
//   y2 = a0 - a1
//   rot = d1 * (-i)
//   y1 = d0 + rot
//   y3 = d0 - rot
TEXT ·bfly4NEON(SB), NOSPLIT, $0-64
	MOVD out_base+0(FP), R0
	MOVD tw_base+24(FP), R1
	MOVD mv+48(FP), R2
	MOVD m+56(FP), R3

	// R4 = m*8 (byte stride between radix blocks)
	LSL  $3, R3, R4

	// Load sign masks
	MOVD $fftc_bfly_negre<>(SB), R13
	VLD1 (R13), [V30.S4]                        // V30 = negre
	MOVD $fftc_bfly_negim<>(SB), R13
	VLD1 (R13), [V31.S4]                        // V31 = negim

	MOVD $0, R5                                 // k = 0

bfly4_loop:
	CMP  R2, R5
	BGE  bfly4_done

	// Compute element pointers
	// x0p = out + k*8
	ADD  R5<<3, R0, R6                          // R6 = x0p
	// x1p = x0p + m*8
	ADD  R4, R6, R7                             // R7 = x1p
	// x2p = x1p + m*8
	ADD  R4, R7, R8                             // R8 = x2p
	// x3p = x2p + m*8
	ADD  R4, R8, R9                             // R9 = x3p

	// tw1p = tw + k*8
	ADD  R5<<3, R1, R10                         // R10 = tw1p
	// tw2p = tw1p + m*8
	ADD  R4, R10, R11                           // R11 = tw2p
	// tw3p = tw2p + m*8
	ADD  R4, R11, R12                           // R12 = tw3p

	// Load x0, x1, x2, x3
	VLD1 (R6), [V0.S4]                          // V0 = x0
	VLD1 (R7), [V1.S4]                          // V1 = x1
	VLD1 (R8), [V2.S4]                          // V2 = x2
	VLD1 (R9), [V3.S4]                          // V3 = x3

	// Load tw1, tw2, tw3
	VLD1 (R10), [V4.S4]                         // V4 = tw1
	VLD1 (R11), [V5.S4]                         // V5 = tw2
	VLD1 (R12), [V6.S4]                         // V6 = tw3

	// --- s0 = x1 * tw1 (complex multiply) ---
	VTRN1 V1.S4, V1.S4, V7.S4                   // V7 = {x1r0,x1r0,x1r1,x1r1}
	VTRN2 V1.S4, V1.S4, V8.S4                   // V8 = {x1i0,x1i0,x1i1,x1i1}
	VREV64 V4.S4, V9.S4                         // V9 = tw1_swap
	WORD $0x6E24DCEA     // FMUL V10.4S, V7.4S, V4.4S  (xr*tw)
	WORD $0x6E29DD0B     // FMUL V11.4S, V8.4S, V9.4S  (xi*tw_swap)
	VEOR V30.B16, V11.B16, V11.B16              // negate real lanes
	WORD $0x4E2BD54C     // FADD V12.4S, V10.4S, V11.4S (s0)

	// --- s1 = x2 * tw2 (complex multiply) ---
	VTRN1 V2.S4, V2.S4, V7.S4                   // V7 = {x2r0,x2r0,x2r1,x2r1}
	VTRN2 V2.S4, V2.S4, V8.S4                   // V8 = {x2i0,x2i0,x2i1,x2i1}
	VREV64 V5.S4, V9.S4                         // V9 = tw2_swap
	WORD $0x6E25DCEA     // FMUL V10.4S, V7.4S, V5.4S  (xr*tw)
	WORD $0x6E29DD0B     // FMUL V11.4S, V8.4S, V9.4S  (xi*tw_swap)
	VEOR V30.B16, V11.B16, V11.B16              // negate real lanes
	WORD $0x4E2BD54D     // FADD V13.4S, V10.4S, V11.4S (s1)

	// --- s2 = x3 * tw3 (complex multiply) ---
	VTRN1 V3.S4, V3.S4, V7.S4                   // V7 = {x3r0,x3r0,x3r1,x3r1}
	VTRN2 V3.S4, V3.S4, V8.S4                   // V8 = {x3i0,x3i0,x3i1,x3i1}
	VREV64 V6.S4, V9.S4                         // V9 = tw3_swap
	WORD $0x6E26DCEA     // FMUL V10.4S, V7.4S, V6.4S  (xr*tw)
	WORD $0x6E29DD0B     // FMUL V11.4S, V8.4S, V9.4S  (xi*tw_swap)
	VEOR V30.B16, V11.B16, V11.B16              // negate real lanes
	WORD $0x4E2BD54E     // FADD V14.4S, V10.4S, V11.4S (s2)

	// Now we have: V0=x0, V12=s0, V13=s1, V14=s2

	// --- d0 = x0 - s1; a0 = x0 + s1 ---
	WORD $0x4EADD40F     // FSUB V15.4S, V0.4S, V13.4S  (d0)
	WORD $0x4E2DD410     // FADD V16.4S, V0.4S, V13.4S  (a0)

	// --- a1 = s0 + s2; d1 = s0 - s2 ---
	WORD $0x4E2ED591     // FADD V17.4S, V12.4S, V14.4S (a1)
	WORD $0x4EAED592     // FSUB V18.4S, V12.4S, V14.4S (d1)

	// --- y0 = a0 + a1; y2 = a0 - a1 ---
	WORD $0x4E31D613     // FADD V19.4S, V16.4S, V17.4S (y0)
	WORD $0x4EB1D614     // FSUB V20.4S, V16.4S, V17.4S (y2)

	// --- rot = d1 * (-i): VREV64 to swap, then XOR negim ---
	VREV64 V18.S4, V21.S4                       // V21 = {d1i,d1r,d1i,d1r}
	VEOR V31.B16, V21.B16, V21.B16              // V21 = {d1i,-d1r,d1i,-d1r} = rot

	// --- y1 = d0 + rot; y3 = d0 - rot ---
	WORD $0x4E35D5F6     // FADD V22.4S, V15.4S, V21.4S (y1)
	WORD $0x4EB5D5F7     // FSUB V23.4S, V15.4S, V21.4S (y3)

	// Store results
	VST1 [V19.S4], (R6)                         // store y0 -> x0
	VST1 [V22.S4], (R7)                         // store y1 -> x1
	VST1 [V20.S4], (R8)                         // store y2 -> x2
	VST1 [V23.S4], (R9)                         // store y3 -> x3

	ADD  $2, R5, R5                             // k += 2 (2 complex64/iter)
	JMP  bfly4_loop

bfly4_done:
	RET
