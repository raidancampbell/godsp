#include "textflag.h"

// ---------------------------------------------------------------------------
// WORD-encoding recipe for NEON float-vector ops (avo is x86-only; see doc.go).
//
// Go's arm64 assembler only exposes the fused VFMLA/VFMLS float-vector
// mnemonics, so plain float add/sub/mul/neg on .4S vectors are emitted as raw
// WORD instructions. Each of the four ops has a base encoding for the ".4S"
// (four 32-bit float) arrangement; the operand register numbers are OR-ed in:
//
//   result = base | (Rm << 16) | (Rn << 5) | Rd     // Vd = Vn op Vm
//   result = base | (Rn << 5) | Rd                  // FNEG: Vd = -Vn (no Rm)
//
// Base encodings (.4S arrangement):
//   FADD Vd.4S, Vn.4S, Vm.4S  -> 0x4E20D400
//   FSUB Vd.4S, Vn.4S, Vm.4S  -> 0x4EA0D400
//   FMUL Vd.4S, Vn.4S, Vm.4S  -> 0x6E20DC00
//   FNEG Vd.4S, Vn.4S         -> 0x6EA0F800
//
// e.g. FADD V16.4S, V12.4S, V13.4S = 0x4E20D400 | (13<<16) | (12<<5) | 16
//    = 0x4E2DD590. Every WORD below carries a decoded comment matching its hex.
// ---------------------------------------------------------------------------

// Sign mask that negates the real (even) lanes of a .4S {re,im,re,im} vector.
// Used to turn brev*d into the -/+ addsub half of the complex product.
DATA fftc_batch_neon_negre<>+0(SB)/4, $0x80000000
DATA fftc_batch_neon_negre<>+4(SB)/4, $0x00000000
DATA fftc_batch_neon_negre<>+8(SB)/4, $0x80000000
DATA fftc_batch_neon_negre<>+12(SB)/4, $0x00000000
GLOBL fftc_batch_neon_negre<>(SB), RODATA|NOPTR, $16

// Sign mask that negates the imaginary (odd) lanes of a .4S {re,im,re,im}
// vector. After VREV64 turns d1={re,im} into {im,re}, XOR-ing this mask yields
// the -j rotation {im,-re}; matches fftc_batch_jrot_sign<> on amd64.
DATA fftc_batch_neon_negim<>+0(SB)/4, $0x00000000
DATA fftc_batch_neon_negim<>+4(SB)/4, $0x80000000
DATA fftc_batch_neon_negim<>+8(SB)/4, $0x00000000
DATA fftc_batch_neon_negim<>+12(SB)/4, $0x80000000
GLOBL fftc_batch_neon_negim<>(SB), RODATA|NOPTR, $16

// Radix-3 Singleton constant s = sin(2*pi/3) = 0.8660254037844386, plus the
// scalar 0.5 used for the -1/2*sum combine. Both are loaded and VDUP-broadcast
// to all four .4S lanes so a plain WORD-FMUL scales re and im identically.
DATA fftc_neon_r3s<>+0(SB)/4, $0.8660254037844386
DATA fftc_neon_r3s<>+4(SB)/4, $0.5
GLOBL fftc_neon_r3s<>(SB), RODATA|NOPTR, $8

// Radix-5 Singleton constants (real): c1, c2, s1, s2. Loaded once as a .4S
// vector then each lane VDUP-broadcast to a full {k,k,k,k} for WORD-FMUL.
DATA fftc_neon_r5<>+0(SB)/4, $0.30901699437494745  // c1 = cos(2*pi/5)
DATA fftc_neon_r5<>+4(SB)/4, $-0.8090169943749475  // c2 = cos(4*pi/5)
DATA fftc_neon_r5<>+8(SB)/4, $0.9510565162951535   // s1 = sin(2*pi/5)
DATA fftc_neon_r5<>+12(SB)/4, $0.5877852522924731  // s2 = sin(4*pi/5)
GLOBL fftc_neon_r5<>(SB), RODATA|NOPTR, $16

// func stockhamRadix2NEON(dst, src, tw []complex64, butterflies, sections, n int)
//
// AoSoA width 4: one logical sample = 4 complex64 = 32 bytes = two .4S regs
// ({r0,i0,r1,i1},{r2,i2,r3,i3}). Radix-2 per (section s, butterfly p):
//   a  = src[in0]                       in0 = (s*bf + p)*4
//   b  = src[in1] * tw[p]               in1 = (s*bf + p + n/2)*4
//   dst[out0] = a + b                   out0 = (s*bf*2 + p)*4
//   dst[out1] = a - b                   out1 = (s*bf*2 + p + bf)*4
//
// Complex multiply b*tw with b={r,i,r,i}, tw broadcast to {c,c,c,c} and
// {d,d,d,d}:  t = b*{c,c,c,c}; brev = rev64(b) = {i,r,i,r};
//   p = brev*{d,d,d,d}; p ^= {-,+,-,+}; bt = t + p
//   => bt = {r*c - i*d, i*c + r*d, ...}
TEXT ·stockhamRadix2NEON(SB), NOSPLIT, $0-96
	MOVD dst_base+0(FP), R0
	MOVD src_base+24(FP), R1
	MOVD tw_base+48(FP), R2
	MOVD butterflies+72(FP), R3
	MOVD sections+80(FP), R4
	MOVD n+88(FP), R5

	// R6 = half byte stride between legs = (n/2)*4*8 = n*16 bytes.
	LSR  $1, R5, R6      // R6 = n/2 (logical samples)
	LSL  $5, R6, R6      // R6 = (n/2)*32 bytes
	// R7 = output stride between out0 and out1 groups = butterflies*32 bytes.
	LSL  $5, R3, R7      // R7 = butterflies*32

	// V7 = sign mask negating real lanes.
	MOVD $fftc_batch_neon_negre<>(SB), R13
	VLD1 (R13), [V7.S4]

	MOVD $0, R9          // section counter

section_loop:
	CMP  R4, R9
	BGE  done
	MOVD R2, R8          // reset running tw pointer to base
	MOVD $0, R10         // butterfly counter

p_loop:
	CMP  R3, R10
	BGE  next_section

	// Load a (in0) into V0,V1 and b (in1 = in0 + half) into V2,V3.
	VLD1 (R1), [V0.S4, V1.S4]
	ADD  R6, R1, R11
	VLD1 (R11), [V2.S4, V3.S4]

	// Load twiddle {c,d} and broadcast: V5={c,c,c,c}, V6={d,d,d,d}.
	VLD1 (R8), [V4.S2]
	VDUP V4.S[0], V5.S4
	VDUP V4.S[1], V6.S4

	// Reverse re/im within each complex: V8=rev64(V2), V9=rev64(V3).
	VREV64 V2.S4, V8.S4
	VREV64 V3.S4, V9.S4

	// --- low half (lanes 0,1) ---
	WORD $0x6E25DC4A     // FMUL V10.4S, V2.4S, V5.4S   (t_lo = b_lo * c)
	WORD $0x6E26DD0B     // FMUL V11.4S, V8.4S, V6.4S   (p_lo = brev_lo * d)
	VEOR V7.B16, V11.B16, V11.B16                       // p_lo ^= negre
	WORD $0x4E2BD54C     // FADD V12.4S, V10.4S, V11.4S (bt_lo = t_lo + p_lo)
	WORD $0x4E2CD410     // FADD V16.4S, V0.4S, V12.4S  (out0_lo = a_lo + bt_lo)
	WORD $0x4EACD412     // FSUB V18.4S, V0.4S, V12.4S  (out1_lo = a_lo - bt_lo)

	// --- high half (lanes 2,3) ---
	WORD $0x6E25DC6D     // FMUL V13.4S, V3.4S, V5.4S   (t_hi = b_hi * c)
	WORD $0x6E26DD2E     // FMUL V14.4S, V9.4S, V6.4S   (p_hi = brev_hi * d)
	VEOR V7.B16, V14.B16, V14.B16                       // p_hi ^= negre
	WORD $0x4E2ED5AF     // FADD V15.4S, V13.4S, V14.4S (bt_hi = t_hi + p_hi)
	WORD $0x4E2FD431     // FADD V17.4S, V1.4S, V15.4S  (out0_hi = a_hi + bt_hi)
	WORD $0x4EAFD433     // FSUB V19.4S, V1.4S, V15.4S  (out1_hi = a_hi - bt_hi)

	// Store out0 at (R0) and out1 at (R0 + R7).
	VST1 [V16.S4, V17.S4], (R0)
	ADD  R7, R0, R12
	VST1 [V18.S4, V19.S4], (R12)

	ADD  $32, R1, R1     // advance src group
	ADD  $32, R0, R0     // advance dst out0 group
	ADD  $8, R8, R8      // advance tw one complex64
	ADD  $1, R10, R10
	JMP  p_loop

next_section:
	ADD  R7, R0, R0      // skip over the out1 block we already wrote
	ADD  $1, R9, R9
	JMP  section_loop

done:
	RET

// func stockhamRadix4NEON(dst, src, tw []complex64, butterflies, sections, n int)
//
// AoSoA width 4: one logical sample = 4 complex64 = 32 bytes = two .4S regs.
// Radix-4 per (section s, butterfly p), mirroring stockhamRadix4Portable:
//   x0 = src[in0]                       in0 = (s*bf + p)*4
//   s0 = src[in1] * tw1                 in1 = (s*bf + p +   n/4)*4   tw1 = tw[p]
//   s1 = src[in2] * tw2                 in2 = (s*bf + p + 2*n/4)*4   tw2 = tw[bf+p]
//   s2 = src[in3] * tw3                 in3 = (s*bf + p + 3*n/4)*4   tw3 = tw[2*bf+p]
//   d0 = x0 - s1;  a0 = x0 + s1;  a1 = s0 + s2;  d1 = s0 - s2
//   out0 = a0 + a1;  out2 = a0 - a1               // out{q} = (s*bf*4 + p + q*bf)*4
//   out1 = d0 + jrot(d1);  out3 = d0 - jrot(d1)
// where jrot(d1) is the -j rotation (re,im)->(im,-re): VREV64 (swap re/im per
// complex) then XOR negim (flip the imaginary/odd lane).
//
// Complex multiply leg*tw with leg={r,i,r,i}, tw broadcast to c={c,c,c,c} and
// d={d,d,d,d}:  t = leg*c; brev = rev64(leg) = {i,r,i,r};
//   pr = brev*d; pr ^= negre {-,+,-,+}; prod = t + pr
//   => prod = {r*c - i*d, i*c + r*d, ...}
TEXT ·stockhamRadix4NEON(SB), NOSPLIT, $0-96
	MOVD dst_base+0(FP), R0
	MOVD src_base+24(FP), R1
	MOVD tw_base+48(FP), R2
	MOVD butterflies+72(FP), R3
	MOVD sections+80(FP), R4
	MOVD n+88(FP), R5

	// R6 = quarter byte stride between legs = (n/4)*32 bytes.
	LSR  $2, R5, R6      // R6 = n/4 (logical samples)
	LSL  $5, R6, R6      // R6 = (n/4)*32 bytes
	// R7 = output stride between out groups = butterflies*32 bytes.
	LSL  $5, R3, R7      // R7 = butterflies*32
	// R14 = tw leg stride = butterflies*8 bytes (one complex64 * butterflies).
	LSL  $3, R3, R14     // R14 = butterflies*8
	// R15 = 3*R7, the dst skip past the out1..out3 blocks at section end.
	ADD  R7<<1, R7, R15  // R15 = R7 + 2*R7 = 3*butterflies*32

	// V31 = sign mask negating real lanes; V30 = sign mask negating imag lanes.
	MOVD $fftc_batch_neon_negre<>(SB), R13
	VLD1 (R13), [V31.S4]
	MOVD $fftc_batch_neon_negim<>(SB), R13
	VLD1 (R13), [V30.S4]

	MOVD $0, R9          // section counter

r4_section_loop:
	CMP  R4, R9
	BGE  r4_done
	MOVD R2, R8          // reset running tw pointer to base
	MOVD $0, R10         // butterfly counter

r4_p_loop:
	CMP  R3, R10
	BGE  r4_next_section

	// Load x0 (in0) into V0,V1.
	VLD1 (R1), [V0.S4, V1.S4]
	// leg1 (in1 = in0 + quarter) into V2,V3.
	ADD  R6, R1, R11
	VLD1 (R11), [V2.S4, V3.S4]
	// leg2 (in2 = in0 + 2*quarter) into V4,V5.
	ADD  R6, R11, R12
	VLD1 (R12), [V4.S4, V5.S4]
	// leg3 (in3 = in0 + 3*quarter) into V6,V7.
	ADD  R6, R12, R13
	VLD1 (R13), [V6.S4, V7.S4]

	// --- s0 = leg1 * tw1  (tw1 at R8) ---
	VLD1 (R8), [V14.S2]
	VDUP V14.S[0], V8.S4                                // c1
	VDUP V14.S[1], V9.S4                                // d1
	VREV64 V2.S4, V10.S4                                // brev1_lo
	VREV64 V3.S4, V11.S4                                // brev1_hi
	WORD $0x6E28DC4C     // FMUL V12.4S, V2.4S, V8.4S (t_lo = leg1_lo * c1)
	WORD $0x6E29DD4D     // FMUL V13.4S, V10.4S, V9.4S (p_lo = brev1_lo * d1)
	VEOR V31.B16, V13.B16, V13.B16                     // p_lo ^= negre
	WORD $0x4E2DD590     // FADD V16.4S, V12.4S, V13.4S (s0_lo)
	WORD $0x6E28DC6C     // FMUL V12.4S, V3.4S, V8.4S (t_hi = leg1_hi * c1)
	WORD $0x6E29DD6D     // FMUL V13.4S, V11.4S, V9.4S (p_hi = brev1_hi * d1)
	VEOR V31.B16, V13.B16, V13.B16                     // p_hi ^= negre
	WORD $0x4E2DD591     // FADD V17.4S, V12.4S, V13.4S (s0_hi)

	// --- s1 = leg2 * tw2  (tw2 at R8 + R14) ---
	ADD  R14, R8, R11
	VLD1 (R11), [V14.S2]
	VDUP V14.S[0], V8.S4                                // c2
	VDUP V14.S[1], V9.S4                                // d2
	VREV64 V4.S4, V10.S4                                // brev2_lo
	VREV64 V5.S4, V11.S4                                // brev2_hi
	WORD $0x6E28DC8C     // FMUL V12.4S, V4.4S, V8.4S (t_lo = leg2_lo * c2)
	WORD $0x6E29DD4D     // FMUL V13.4S, V10.4S, V9.4S (p_lo = brev2_lo * d2)
	VEOR V31.B16, V13.B16, V13.B16                     // p_lo ^= negre
	WORD $0x4E2DD592     // FADD V18.4S, V12.4S, V13.4S (s1_lo)
	WORD $0x6E28DCAC     // FMUL V12.4S, V5.4S, V8.4S (t_hi = leg2_hi * c2)
	WORD $0x6E29DD6D     // FMUL V13.4S, V11.4S, V9.4S (p_hi = brev2_hi * d2)
	VEOR V31.B16, V13.B16, V13.B16                     // p_hi ^= negre
	WORD $0x4E2DD593     // FADD V19.4S, V12.4S, V13.4S (s1_hi)

	// --- s2 = leg3 * tw3  (tw3 at R8 + 2*R14) ---
	ADD  R14, R11, R12
	VLD1 (R12), [V14.S2]
	VDUP V14.S[0], V8.S4                                // c3
	VDUP V14.S[1], V9.S4                                // d3
	VREV64 V6.S4, V10.S4                                // brev3_lo
	VREV64 V7.S4, V11.S4                                // brev3_hi
	WORD $0x6E28DCCC     // FMUL V12.4S, V6.4S, V8.4S (t_lo = leg3_lo * c3)
	WORD $0x6E29DD4D     // FMUL V13.4S, V10.4S, V9.4S (p_lo = brev3_lo * d3)
	VEOR V31.B16, V13.B16, V13.B16                     // p_lo ^= negre
	WORD $0x4E2DD594     // FADD V20.4S, V12.4S, V13.4S (s2_lo)
	WORD $0x6E28DCEC     // FMUL V12.4S, V7.4S, V8.4S (t_hi = leg3_hi * c3)
	WORD $0x6E29DD6D     // FMUL V13.4S, V11.4S, V9.4S (p_hi = brev3_hi * d3)
	VEOR V31.B16, V13.B16, V13.B16                     // p_hi ^= negre
	WORD $0x4E2DD595     // FADD V21.4S, V12.4S, V13.4S (s2_hi)

	// --- low half butterfly (lanes 0,1) ---
	WORD $0x4EB2D418     // FSUB V24.4S, V0.4S, V18.4S (d0_lo = x0_lo - s1_lo)
	WORD $0x4E32D419     // FADD V25.4S, V0.4S, V18.4S (a0_lo = x0_lo + s1_lo)
	WORD $0x4E34D61A     // FADD V26.4S, V16.4S, V20.4S (a1_lo = s0_lo + s2_lo)
	WORD $0x4EB4D61B     // FSUB V27.4S, V16.4S, V20.4S (d1_lo = s0_lo - s2_lo)
	VREV64 V27.S4, V28.S4                               // swap re/im of d1_lo
	VEOR V30.B16, V28.B16, V29.B16                     // jrot_lo = (im, -re)
	WORD $0x4E3AD730     // FADD V16.4S, V25.4S, V26.4S (out0_lo = a0_lo + a1_lo)
	WORD $0x4EBAD734     // FSUB V20.4S, V25.4S, V26.4S (out2_lo = a0_lo - a1_lo)
	WORD $0x4E3DD712     // FADD V18.4S, V24.4S, V29.4S (out1_lo = d0_lo + jrot_lo)
	WORD $0x4EBDD716     // FSUB V22.4S, V24.4S, V29.4S (out3_lo = d0_lo - jrot_lo)

	// --- high half butterfly (lanes 2,3) ---
	WORD $0x4EB3D438     // FSUB V24.4S, V1.4S, V19.4S (d0_hi = x0_hi - s1_hi)
	WORD $0x4E33D439     // FADD V25.4S, V1.4S, V19.4S (a0_hi = x0_hi + s1_hi)
	WORD $0x4E35D63A     // FADD V26.4S, V17.4S, V21.4S (a1_hi = s0_hi + s2_hi)
	WORD $0x4EB5D63B     // FSUB V27.4S, V17.4S, V21.4S (d1_hi = s0_hi - s2_hi)
	VREV64 V27.S4, V28.S4                               // swap re/im of d1_hi
	VEOR V30.B16, V28.B16, V29.B16                     // jrot_hi = (im, -re)
	WORD $0x4E3AD731     // FADD V17.4S, V25.4S, V26.4S (out0_hi = a0_hi + a1_hi)
	WORD $0x4EBAD735     // FSUB V21.4S, V25.4S, V26.4S (out2_hi = a0_hi - a1_hi)
	WORD $0x4E3DD713     // FADD V19.4S, V24.4S, V29.4S (out1_hi = d0_hi + jrot_hi)
	WORD $0x4EBDD717     // FSUB V23.4S, V24.4S, V29.4S (out3_hi = d0_hi - jrot_hi)

	// Store out0..out3 at R0, R0+R7, R0+2*R7, R0+3*R7.
	VST1 [V16.S4, V17.S4], (R0)
	ADD  R7, R0, R11
	VST1 [V18.S4, V19.S4], (R11)
	ADD  R7, R11, R12
	VST1 [V20.S4, V21.S4], (R12)
	ADD  R7, R12, R13
	VST1 [V22.S4, V23.S4], (R13)

	ADD  $32, R1, R1     // advance src group
	ADD  $32, R0, R0     // advance dst out0 group
	ADD  $8, R8, R8      // advance tw one complex64
	ADD  $1, R10, R10
	JMP  r4_p_loop

r4_next_section:
	ADD  R15, R0, R0     // skip over the out1..out3 blocks we already wrote
	ADD  $1, R9, R9
	JMP  r4_section_loop

r4_done:
	RET

// func stockhamRadix3NEON(dst, src, tw []complex64, butterflies, sections, n int)
//
// AoSoA width 4: one logical sample = 4 complex64 = 32 bytes = two .4S regs.
// Radix-3 per (section s, butterfly p), mirroring stockhamRadix3Portable:
//   t0 = src[in0]                       in0 = (s*bf + p)*4
//   t1 = src[in1] * tw1                 in1 = (s*bf + p +   n/3)*4   tw1 = tw[p]
//   t2 = src[in2] * tw2                 in2 = (s*bf + p + 2*n/3)*4   tw2 = tw[bf+p]
//   sum = t1 + t2;  dif = t1 - t2
//   out0 = t0 + sum                                // out{q} = (s*bf*3 + p + q*bf)*4
//   a  = t0 - 0.5*sum                              // real scalar 0.5, both lanes
//   ww = s*dif                                     // real scalar s, both lanes
//   out1 = (re(a)+im(ww), im(a)-re(ww)) = a + jrot(ww)
//   out2 = (re(a)-im(ww), im(a)+re(ww)) = a - jrot(ww)
// where jrot(v) is the -j rotation (re,im)->(im,-re): VREV64 then XOR negim.
//
// Complex multiply leg*tw uses the same t/brev/negre idiom as radix-2/4.
TEXT ·stockhamRadix3NEON(SB), NOSPLIT, $0-96
	MOVD dst_base+0(FP), R0
	MOVD src_base+24(FP), R1
	MOVD tw_base+48(FP), R2
	MOVD butterflies+72(FP), R3
	MOVD sections+80(FP), R4
	MOVD n+88(FP), R5

	// R6 = third byte stride between legs = (n/3)*32 bytes.
	MOVD $3, R11
	UDIV R11, R5, R6     // R6 = n/3 (logical samples)
	LSL  $5, R6, R6      // R6 = (n/3)*32 bytes
	// R7 = output stride between out groups = butterflies*32 bytes.
	LSL  $5, R3, R7      // R7 = butterflies*32
	// R14 = tw leg stride = butterflies*8 bytes.
	LSL  $3, R3, R14     // R14 = butterflies*8
	// R15 = 2*R7, the dst skip past the out1..out2 blocks at section end.
	ADD  R7<<1, ZR, R15  // R15 = 2*butterflies*32

	// V31 = negre (flip real lanes); V30 = negim (flip imag lanes).
	MOVD $fftc_batch_neon_negre<>(SB), R13
	VLD1 (R13), [V31.S4]
	MOVD $fftc_batch_neon_negim<>(SB), R13
	VLD1 (R13), [V30.S4]
	// V26 = {0.5,0.5,0.5,0.5}; V27 = {s,s,s,s}.
	MOVD $fftc_neon_r3s<>(SB), R13
	VLD1 (R13), [V25.S2]
	VDUP V25.S[0], V27.S4                              // s
	VDUP V25.S[1], V26.S4                              // 0.5

	MOVD $0, R9          // section counter

r3_section_loop:
	CMP  R4, R9
	BGE  r3_done
	MOVD R2, R8          // reset running tw pointer to base
	MOVD $0, R10         // butterfly counter

r3_p_loop:
	CMP  R3, R10
	BGE  r3_next_section

	// Load t0 (in0) into V0,V1.
	VLD1 (R1), [V0.S4, V1.S4]
	// leg1 (in1 = in0 + third) into V2,V3.
	ADD  R6, R1, R11
	VLD1 (R11), [V2.S4, V3.S4]
	// leg2 (in2 = in0 + 2*third) into V4,V5.
	ADD  R6, R11, R12
	VLD1 (R12), [V4.S4, V5.S4]

	// --- t1 = leg1 * tw1  (tw1 at R8) ---
	VLD1 (R8), [V14.S2]
	VDUP V14.S[0], V8.S4                                // c1
	VDUP V14.S[1], V9.S4                                // d1
	VREV64 V2.S4, V10.S4                                // brev1_lo
	VREV64 V3.S4, V11.S4                                // brev1_hi
	WORD $0x6E28DC4C     // FMUL V12.4S, V2.4S, V8.4S   (t_lo=leg1_lo*c1)
	WORD $0x6E29DD4D     // FMUL V13.4S, V10.4S, V9.4S  (p_lo=brev1_lo*d1)
	VEOR V31.B16, V13.B16, V13.B16                      // p_lo ^= negre
	WORD $0x4E2DD590     // FADD V16.4S, V12.4S, V13.4S (t1_lo)
	WORD $0x6E28DC6C     // FMUL V12.4S, V3.4S, V8.4S   (t_hi=leg1_hi*c1)
	WORD $0x6E29DD6D     // FMUL V13.4S, V11.4S, V9.4S  (p_hi=brev1_hi*d1)
	VEOR V31.B16, V13.B16, V13.B16                      // p_hi ^= negre
	WORD $0x4E2DD591     // FADD V17.4S, V12.4S, V13.4S (t1_hi)

	// --- t2 = leg2 * tw2  (tw2 at R8 + R14) ---
	ADD  R14, R8, R11
	VLD1 (R11), [V14.S2]
	VDUP V14.S[0], V8.S4                                // c2
	VDUP V14.S[1], V9.S4                                // d2
	VREV64 V4.S4, V10.S4                                // brev2_lo
	VREV64 V5.S4, V11.S4                                // brev2_hi
	WORD $0x6E28DC8C     // FMUL V12.4S, V4.4S, V8.4S   (t_lo=leg2_lo*c2)
	WORD $0x6E29DD4D     // FMUL V13.4S, V10.4S, V9.4S  (p_lo=brev2_lo*d2)
	VEOR V31.B16, V13.B16, V13.B16                      // p_lo ^= negre
	WORD $0x4E2DD592     // FADD V18.4S, V12.4S, V13.4S (t2_lo)
	WORD $0x6E28DCAC     // FMUL V12.4S, V5.4S, V8.4S   (t_hi=leg2_hi*c2)
	WORD $0x6E29DD6D     // FMUL V13.4S, V11.4S, V9.4S  (p_hi=brev2_hi*d2)
	VEOR V31.B16, V13.B16, V13.B16                      // p_hi ^= negre
	WORD $0x4E2DD593     // FADD V19.4S, V12.4S, V13.4S (t2_hi)

	// --- combine low half (t0=V0, t1=V16, t2=V18); out{q}_lo=V(2+2q) ---
	WORD $0x4E32D614     // FADD V20.4S, V16.4S, V18.4S (sum_lo=t1+t2)
	WORD $0x4EB2D615     // FSUB V21.4S, V16.4S, V18.4S (dif_lo=t1-t2)
	WORD $0x4E34D402     // FADD V2.4S, V0.4S, V20.4S   (out0_lo=t0+sum)
	WORD $0x6E3ADE96     // FMUL V22.4S, V20.4S, V26.4S (half_lo=0.5*sum)
	WORD $0x4EB6D417     // FSUB V23.4S, V0.4S, V22.4S  (a_lo=t0-0.5*sum)
	WORD $0x6E3BDEB8     // FMUL V24.4S, V21.4S, V27.4S (ww_lo=s*dif)
	VREV64 V24.S4, V28.S4                               // swap re/im of ww_lo
	VEOR V30.B16, V28.B16, V28.B16                      // jrot_lo=(im,-re)
	WORD $0x4E3CD6E4     // FADD V4.4S, V23.4S, V28.4S  (out1_lo=a+jrot)
	WORD $0x4EBCD6E6     // FSUB V6.4S, V23.4S, V28.4S  (out2_lo=a-jrot)

	// --- combine high half (t0=V1, t1=V17, t2=V19); out{q}_hi=V(3+2q) ---
	WORD $0x4E33D634     // FADD V20.4S, V17.4S, V19.4S (sum_hi=t1+t2)
	WORD $0x4EB3D635     // FSUB V21.4S, V17.4S, V19.4S (dif_hi=t1-t2)
	WORD $0x4E34D423     // FADD V3.4S, V1.4S, V20.4S   (out0_hi=t0+sum)
	WORD $0x6E3ADE96     // FMUL V22.4S, V20.4S, V26.4S (half_hi=0.5*sum)
	WORD $0x4EB6D437     // FSUB V23.4S, V1.4S, V22.4S  (a_hi=t0-0.5*sum)
	WORD $0x6E3BDEB8     // FMUL V24.4S, V21.4S, V27.4S (ww_hi=s*dif)
	VREV64 V24.S4, V28.S4                               // swap re/im of ww_hi
	VEOR V30.B16, V28.B16, V28.B16                      // jrot_hi=(im,-re)
	WORD $0x4E3CD6E5     // FADD V5.4S, V23.4S, V28.4S  (out1_hi=a+jrot)
	WORD $0x4EBCD6E7     // FSUB V7.4S, V23.4S, V28.4S  (out2_hi=a-jrot)

	// Store out0..out2 (each {lo,hi} pair) at R0, R0+R7, R0+2*R7.
	VST1 [V2.S4, V3.S4], (R0)
	ADD  R7, R0, R11
	VST1 [V4.S4, V5.S4], (R11)
	ADD  R7, R11, R12
	VST1 [V6.S4, V7.S4], (R12)

	ADD  $32, R1, R1     // advance src group
	ADD  $32, R0, R0     // advance dst out0 group
	ADD  $8, R8, R8      // advance tw one complex64
	ADD  $1, R10, R10
	JMP  r3_p_loop

r3_next_section:
	ADD  R15, R0, R0     // skip over the out1..out2 blocks we already wrote
	ADD  $1, R9, R9
	JMP  r3_section_loop

r3_done:
	RET

// func stockhamRadix5NEON(dst, src, tw []complex64, butterflies, sections, n int)
//
// AoSoA width 4: one logical sample = 4 complex64 = 32 bytes = two .4S regs.
// Radix-5 per (section s, butterfly p), mirroring stockhamRadix5Portable:
//   t0 = src[in0]; t{k} = src[in{k}] * tw{k}   in{k} = (s*bf + p + k*n/5)*4
//     tw{k} = tw[(k-1)*bf + p]
//   a1 = t1+t4; d1 = t1-t4; a2 = t2+t3; d2 = t2-t3
//   out0 = t0 + a1 + a2                             // out{q} = (s*bf*5 + p + q*bf)*4
//   m1 = t0 + c1*a1 + c2*a2;  m2 = t0 + c2*a1 + c1*a2
//   r1 = s1*d1 + s2*d2;       r2 = s2*d1 - s1*d2
//   out1 = m1 + jrot(r1);  out4 = m1 - jrot(r1)
//   out2 = m2 + jrot(r2);  out3 = m2 - jrot(r2)
// where jrot(v) is the -j rotation (re,im)->(im,-re): VREV64 then XOR negim.
// c1,c2,s1,s2 are real scalars broadcast to all four .4S lanes.
TEXT ·stockhamRadix5NEON(SB), NOSPLIT, $0-96
	MOVD dst_base+0(FP), R0
	MOVD src_base+24(FP), R1
	MOVD tw_base+48(FP), R2
	MOVD butterflies+72(FP), R3
	MOVD sections+80(FP), R4
	MOVD n+88(FP), R5

	// R6 = fifth byte stride between legs = (n/5)*32 bytes.
	MOVD $5, R11
	UDIV R11, R5, R6     // R6 = n/5 (logical samples)
	LSL  $5, R6, R6      // R6 = (n/5)*32 bytes
	// R7 = output stride between out groups = butterflies*32 bytes.
	LSL  $5, R3, R7      // R7 = butterflies*32
	// R14 = tw leg stride = butterflies*8 bytes.
	LSL  $3, R3, R14     // R14 = butterflies*8
	// R15 = 4*R7, the dst skip past the out1..out4 blocks at section end.
	ADD  R7<<2, ZR, R15  // R15 = 4*butterflies*32

	// V31 = negre; V30 = negim.
	MOVD $fftc_batch_neon_negre<>(SB), R13
	VLD1 (R13), [V31.S4]
	MOVD $fftc_batch_neon_negim<>(SB), R13
	VLD1 (R13), [V30.S4]
	// Load {c1,c2,s1,s2} and broadcast each: V24=c1,V25=c2,V26=s1,V27=s2.
	MOVD $fftc_neon_r5<>(SB), R13
	VLD1 (R13), [V23.S4]
	VDUP V23.S[0], V24.S4                               // c1
	VDUP V23.S[1], V25.S4                               // c2
	VDUP V23.S[2], V26.S4                               // s1
	VDUP V23.S[3], V27.S4                               // s2

	MOVD $0, R9          // section counter

r5_section_loop:
	CMP  R4, R9
	BGE  r5_done
	MOVD R2, R8          // reset running tw pointer to base
	MOVD $0, R10         // butterfly counter

r5_p_loop:
	CMP  R3, R10
	BGE  r5_next_section

	// Load t0 (in0) into V0,V1.
	VLD1 (R1), [V0.S4, V1.S4]

	// --- t1 = leg1 * tw1  (in1 = in0 + fifth; tw1 at R8) ---
	ADD  R6, R1, R11
	VLD1 (R11), [V2.S4, V3.S4]
	VLD1 (R8), [V14.S2]
	VDUP V14.S[0], V8.S4                                // c
	VDUP V14.S[1], V9.S4                                // d
	VREV64 V2.S4, V10.S4
	VREV64 V3.S4, V11.S4
	WORD $0x6E28DC4C     // FMUL V12.4S, V2.4S, V8.4S   (t_lo=leg1_lo*c)
	WORD $0x6E29DD4D     // FMUL V13.4S, V10.4S, V9.4S  (p_lo=brev1_lo*d)
	VEOR V31.B16, V13.B16, V13.B16                      // p_lo ^= negre
	WORD $0x4E2DD590     // FADD V16.4S, V12.4S, V13.4S (t1_lo)
	WORD $0x6E28DC6C     // FMUL V12.4S, V3.4S, V8.4S   (t_hi=leg1_hi*c)
	WORD $0x6E29DD6D     // FMUL V13.4S, V11.4S, V9.4S  (p_hi=brev1_hi*d)
	VEOR V31.B16, V13.B16, V13.B16                      // p_hi ^= negre
	WORD $0x4E2DD591     // FADD V17.4S, V12.4S, V13.4S (t1_hi)

	// --- t2 = leg2 * tw2  (in2 = in1 + fifth; tw2 at R8 + R14) ---
	ADD  R6, R11, R12
	VLD1 (R12), [V4.S4, V5.S4]
	ADD  R14, R8, R11
	VLD1 (R11), [V14.S2]
	VDUP V14.S[0], V8.S4
	VDUP V14.S[1], V9.S4
	VREV64 V4.S4, V10.S4
	VREV64 V5.S4, V11.S4
	WORD $0x6E28DC8C     // FMUL V12.4S, V4.4S, V8.4S   (t_lo=leg2_lo*c)
	WORD $0x6E29DD4D     // FMUL V13.4S, V10.4S, V9.4S  (p_lo=brev2_lo*d)
	VEOR V31.B16, V13.B16, V13.B16                      // p_lo ^= negre
	WORD $0x4E2DD592     // FADD V18.4S, V12.4S, V13.4S (t2_lo)
	WORD $0x6E28DCAC     // FMUL V12.4S, V5.4S, V8.4S   (t_hi=leg2_hi*c)
	WORD $0x6E29DD6D     // FMUL V13.4S, V11.4S, V9.4S  (p_hi=brev2_hi*d)
	VEOR V31.B16, V13.B16, V13.B16                      // p_hi ^= negre
	WORD $0x4E2DD593     // FADD V19.4S, V12.4S, V13.4S (t2_hi)

	// --- t3 = leg3 * tw3  (in3 = in2 + fifth; tw3 at R8 + 2*R14) ---
	ADD  R6, R12, R13
	VLD1 (R13), [V6.S4, V7.S4]
	ADD  R14, R11, R12
	VLD1 (R12), [V14.S2]
	VDUP V14.S[0], V8.S4
	VDUP V14.S[1], V9.S4
	VREV64 V6.S4, V10.S4
	VREV64 V7.S4, V11.S4
	WORD $0x6E28DCCC     // FMUL V12.4S, V6.4S, V8.4S   (t_lo=leg3_lo*c)
	WORD $0x6E29DD4D     // FMUL V13.4S, V10.4S, V9.4S  (p_lo=brev3_lo*d)
	VEOR V31.B16, V13.B16, V13.B16                      // p_lo ^= negre
	WORD $0x4E2DD594     // FADD V20.4S, V12.4S, V13.4S (t3_lo)
	WORD $0x6E28DCEC     // FMUL V12.4S, V7.4S, V8.4S   (t_hi=leg3_hi*c)
	WORD $0x6E29DD6D     // FMUL V13.4S, V11.4S, V9.4S  (p_hi=brev3_hi*d)
	VEOR V31.B16, V13.B16, V13.B16                      // p_hi ^= negre
	WORD $0x4E2DD595     // FADD V21.4S, V12.4S, V13.4S (t3_hi)

	// --- t4 = leg4 * tw4  (in4 = in3 + fifth; tw4 at R8 + 3*R14) ---
	ADD  R6, R13, R11
	VLD1 (R11), [V2.S4, V3.S4]
	ADD  R14, R12, R11
	VLD1 (R11), [V14.S2]
	VDUP V14.S[0], V8.S4
	VDUP V14.S[1], V9.S4
	VREV64 V2.S4, V10.S4
	VREV64 V3.S4, V11.S4
	WORD $0x6E28DC4C     // FMUL V12.4S, V2.4S, V8.4S   (t_lo=leg4_lo*c)
	WORD $0x6E29DD4D     // FMUL V13.4S, V10.4S, V9.4S  (p_lo=brev4_lo*d)
	VEOR V31.B16, V13.B16, V13.B16                      // p_lo ^= negre
	WORD $0x4E2DD596     // FADD V22.4S, V12.4S, V13.4S (t4_lo)
	WORD $0x6E28DC6C     // FMUL V12.4S, V3.4S, V8.4S   (t_hi=leg4_hi*c)
	WORD $0x6E29DD6D     // FMUL V13.4S, V11.4S, V9.4S  (p_hi=brev4_hi*d)
	VEOR V31.B16, V13.B16, V13.B16                      // p_hi ^= negre
	WORD $0x4E2DD597     // FADD V23.4S, V12.4S, V13.4S (t4_hi)

	// --- combine low half: t0=V0 t1=V16 t2=V18 t3=V20 t4=V22; out{q}_lo=V(2+2q) ---
	WORD $0x4E36D60C     // FADD V12.4S, V16.4S, V22.4S (a1_lo=t1+t4)
	WORD $0x4EB6D60D     // FSUB V13.4S, V16.4S, V22.4S (d1_lo=t1-t4)
	WORD $0x4E34D64E     // FADD V14.4S, V18.4S, V20.4S (a2_lo=t2+t3)
	WORD $0x4EB4D64F     // FSUB V15.4S, V18.4S, V20.4S (d2_lo=t2-t3)
	WORD $0x4E2CD41C     // FADD V28.4S, V0.4S, V12.4S  (t0+a1)
	WORD $0x4E2ED782     // FADD V2.4S, V28.4S, V14.4S  (out0_lo=t0+a1+a2)
	WORD $0x6E38DD9C     // FMUL V28.4S, V12.4S, V24.4S (c1*a1)
	WORD $0x4E3CD41D     // FADD V29.4S, V0.4S, V28.4S  (t0+c1*a1)
	WORD $0x6E39DDDC     // FMUL V28.4S, V14.4S, V25.4S (c2*a2)
	WORD $0x4E3CD7BD     // FADD V29.4S, V29.4S, V28.4S (m1_lo=+c2*a2)
	WORD $0x6E39DD9C     // FMUL V28.4S, V12.4S, V25.4S (c2*a1)
	WORD $0x4E3CD40C     // FADD V12.4S, V0.4S, V28.4S  (t0+c2*a1)
	WORD $0x6E38DDDC     // FMUL V28.4S, V14.4S, V24.4S (c1*a2)
	WORD $0x4E3CD58C     // FADD V12.4S, V12.4S, V28.4S (m2_lo=+c1*a2)
	WORD $0x6E3ADDBC     // FMUL V28.4S, V13.4S, V26.4S (s1*d1)
	WORD $0x6E3BDDEE     // FMUL V14.4S, V15.4S, V27.4S (s2*d2)
	WORD $0x4E2ED79C     // FADD V28.4S, V28.4S, V14.4S (r1_lo=s1*d1+s2*d2)
	WORD $0x6E3BDDAE     // FMUL V14.4S, V13.4S, V27.4S (s2*d1)
	WORD $0x6E3ADDED     // FMUL V13.4S, V15.4S, V26.4S (s1*d2)
	WORD $0x4EADD5CE     // FSUB V14.4S, V14.4S, V13.4S (r2_lo=s2*d1-s1*d2)
	VREV64 V28.S4, V13.S4                               // swap re/im of r1_lo
	VEOR V30.B16, V13.B16, V13.B16                      // jrot(r1)=(im,-re)
	WORD $0x4E2DD7A4     // FADD V4.4S, V29.4S, V13.4S  (out1_lo=m1+jrot(r1))
	WORD $0x4EADD7AA     // FSUB V10.4S, V29.4S, V13.4S (out4_lo=m1-jrot(r1))
	VREV64 V14.S4, V15.S4                               // swap re/im of r2_lo
	VEOR V30.B16, V15.B16, V15.B16                      // jrot(r2)=(im,-re)
	WORD $0x4E2FD586     // FADD V6.4S, V12.4S, V15.4S  (out2_lo=m2+jrot(r2))
	WORD $0x4EAFD588     // FSUB V8.4S, V12.4S, V15.4S  (out3_lo=m2-jrot(r2))

	// --- combine high half: t0=V1 t1=V17 t2=V19 t3=V21 t4=V23; out{q}_hi=V(3+2q) ---
	WORD $0x4E37D62C     // FADD V12.4S, V17.4S, V23.4S (a1_hi=t1+t4)
	WORD $0x4EB7D62D     // FSUB V13.4S, V17.4S, V23.4S (d1_hi=t1-t4)
	WORD $0x4E35D66E     // FADD V14.4S, V19.4S, V21.4S (a2_hi=t2+t3)
	WORD $0x4EB5D66F     // FSUB V15.4S, V19.4S, V21.4S (d2_hi=t2-t3)
	WORD $0x4E2CD43C     // FADD V28.4S, V1.4S, V12.4S  (t0+a1)
	WORD $0x4E2ED783     // FADD V3.4S, V28.4S, V14.4S  (out0_hi=t0+a1+a2)
	WORD $0x6E38DD9C     // FMUL V28.4S, V12.4S, V24.4S (c1*a1)
	WORD $0x4E3CD43D     // FADD V29.4S, V1.4S, V28.4S  (t0+c1*a1)
	WORD $0x6E39DDDC     // FMUL V28.4S, V14.4S, V25.4S (c2*a2)
	WORD $0x4E3CD7BD     // FADD V29.4S, V29.4S, V28.4S (m1_hi=+c2*a2)
	WORD $0x6E39DD9C     // FMUL V28.4S, V12.4S, V25.4S (c2*a1)
	WORD $0x4E3CD42C     // FADD V12.4S, V1.4S, V28.4S  (t0+c2*a1)
	WORD $0x6E38DDDC     // FMUL V28.4S, V14.4S, V24.4S (c1*a2)
	WORD $0x4E3CD58C     // FADD V12.4S, V12.4S, V28.4S (m2_hi=+c1*a2)
	WORD $0x6E3ADDBC     // FMUL V28.4S, V13.4S, V26.4S (s1*d1)
	WORD $0x6E3BDDEE     // FMUL V14.4S, V15.4S, V27.4S (s2*d2)
	WORD $0x4E2ED79C     // FADD V28.4S, V28.4S, V14.4S (r1_hi=s1*d1+s2*d2)
	WORD $0x6E3BDDAE     // FMUL V14.4S, V13.4S, V27.4S (s2*d1)
	WORD $0x6E3ADDED     // FMUL V13.4S, V15.4S, V26.4S (s1*d2)
	WORD $0x4EADD5CE     // FSUB V14.4S, V14.4S, V13.4S (r2_hi=s2*d1-s1*d2)
	VREV64 V28.S4, V13.S4                               // swap re/im of r1_hi
	VEOR V30.B16, V13.B16, V13.B16                      // jrot(r1)=(im,-re)
	WORD $0x4E2DD7A5     // FADD V5.4S, V29.4S, V13.4S  (out1_hi=m1+jrot(r1))
	WORD $0x4EADD7AB     // FSUB V11.4S, V29.4S, V13.4S (out4_hi=m1-jrot(r1))
	VREV64 V14.S4, V15.S4                               // swap re/im of r2_hi
	VEOR V30.B16, V15.B16, V15.B16                      // jrot(r2)=(im,-re)
	WORD $0x4E2FD587     // FADD V7.4S, V12.4S, V15.4S  (out2_hi=m2+jrot(r2))
	WORD $0x4EAFD589     // FSUB V9.4S, V12.4S, V15.4S  (out3_hi=m2-jrot(r2))

	// Store out0..out4 (each {lo,hi} pair) at R0, R0+R7, R0+2*R7, R0+3*R7, R0+4*R7.
	VST1 [V2.S4, V3.S4], (R0)
	ADD  R7, R0, R11
	VST1 [V4.S4, V5.S4], (R11)
	ADD  R7, R11, R12
	VST1 [V6.S4, V7.S4], (R12)
	ADD  R7, R12, R13
	VST1 [V8.S4, V9.S4], (R13)
	ADD  R7, R13, R11
	VST1 [V10.S4, V11.S4], (R11)

	ADD  $32, R1, R1     // advance src group
	ADD  $32, R0, R0     // advance dst out0 group
	ADD  $8, R8, R8      // advance tw one complex64
	ADD  $1, R10, R10
	JMP  r5_p_loop

r5_next_section:
	ADD  R15, R0, R0     // skip over the out1..out4 blocks we already wrote
	ADD  $1, R9, R9
	JMP  r5_section_loop

r5_done:
	RET


// Radix-8 eighth-root real scale r = sqrt(2)/2 = 0.7071067811865476, replicated
// to all four .4S lanes so a plain WORD-FMUL scales re and im identically. Used
// to build the w1/w3 eighth-root recombine terms in stockhamRadix8NEON.
DATA fftc_neon_r8<>+0(SB)/4, $0.7071067811865476
DATA fftc_neon_r8<>+4(SB)/4, $0.7071067811865476
DATA fftc_neon_r8<>+8(SB)/4, $0.7071067811865476
DATA fftc_neon_r8<>+12(SB)/4, $0.7071067811865476
GLOBL fftc_neon_r8<>(SB), RODATA|NOPTR, $16


// func stockhamRadix8NEON(dst, src, tw []complex64, butterflies, sections, n int)
//
// AoSoA width 4: one logical sample = 4 complex64 = 32 bytes = two .4S regs.
// Radix-8 per (section s, butterfly p), mirroring stockhamRadix8Portable, is
// two radix-4 DFTs (even legs 0,2,4,6; odd legs 1,3,5,7) recombined with the
// eighth roots:
//   x0 = src[in0]; x{k} = src[in{k}] * tw{k}  in{k}=(s*bf+p+k*n/8)*4
//     tw{k} = tw[(k-1)*bf + p]
//   even: e0=x0+x4; e1=x0-x4; e2=x2+x6; e3=x2-x6
//         E0=e0+e2; E2=e0-e2; E1=e1+jrot(e3); E3=e1-jrot(e3)
//   odd:  oo0=x1+x5; oo1=x1-x5; oo2=x3+x7; oo3=x3-x7
//         O0=oo0+oo2; O2=oo0-oo2; O1=oo1+jrot(oo3); O3=oo1-jrot(oo3)
//   w1 = r*(O1.re+O1.im, O1.im-O1.re)
//   w2 = (O2.im, -O2.re) = jrot(O2)
//   w3 = r*(O3.im-O3.re, -(O3.re+O3.im))
//   out0=E0+O0; out1=E1+w1; out2=E2+w2; out3=E3+w3       out{q}=(s*bf*8+p+q*bf)*4
//   out4=E0-O0; out5=E1-w1; out6=E2-w2; out7=E3-w3
// where jrot(v) is the -j rotation (re,im)->(im,-re): VREV64 then XOR negim.
//
// 8 legs x 2 halves = 16 .4S leg regs held live in V0..V15; V16..V28 scratch;
// V29={r,r,r,r}; V30=negim; V31=negre. Each output group is stored as two
// single-register .4S stores (lo at group base, hi at base+16) so no contiguous
// VST1 register list is needed.
//
// Complex multiply leg*tw uses the same t/brev/negre idiom as radix-2/3/4/5.
TEXT ·stockhamRadix8NEON(SB), NOSPLIT, $0-96
	MOVD dst_base+0(FP), R0
	MOVD src_base+24(FP), R1
	MOVD tw_base+48(FP), R2
	MOVD butterflies+72(FP), R3
	MOVD sections+80(FP), R4
	MOVD n+88(FP), R5

	// R6 = eighth byte stride between legs = (n/8)*32 bytes.
	LSR  $3, R5, R6      // R6 = n/8 (logical samples)
	LSL  $5, R6, R6      // R6 = (n/8)*32 bytes
	// R7 = output stride between out groups = butterflies*32 bytes.
	LSL  $5, R3, R7      // R7 = butterflies*32
	// R14 = tw leg stride = butterflies*8 bytes.
	LSL  $3, R3, R14     // R14 = butterflies*8
	// R15 = 7*R7, the dst skip past the out1..out7 blocks at section end.
	LSL  $3, R7, R15     // R15 = 8*R7
	SUB  R7, R15, R15    // R15 = 7*R7

	// V29 = {r,r,r,r}; V30 = negim; V31 = negre.
	MOVD $fftc_neon_r8<>(SB), R13
	VLD1 (R13), [V29.S4]
	MOVD $fftc_batch_neon_negim<>(SB), R13
	VLD1 (R13), [V30.S4]
	MOVD $fftc_batch_neon_negre<>(SB), R13
	VLD1 (R13), [V31.S4]

	MOVD $0, R9          // section counter

r8_section_loop:
	CMP  R4, R9
	BGE  r8_done
	MOVD R2, R8          // reset running tw pointer to base
	MOVD $0, R10         // butterfly counter

r8_p_loop:
	CMP  R3, R10
	BGE  r8_next_section

	// Load x0 (in0) into V0,V1.
	VLD1 (R1), [V0.S4, V1.S4]
	ADD  R6, R1, R11
	VLD1 (R11), [V2.S4, V3.S4]     // leg1 (in1)
	ADD  R6, R11, R11
	VLD1 (R11), [V4.S4, V5.S4]     // leg2 (in2)
	ADD  R6, R11, R11
	VLD1 (R11), [V6.S4, V7.S4]     // leg3 (in3)
	ADD  R6, R11, R11
	VLD1 (R11), [V8.S4, V9.S4]     // leg4 (in4)
	ADD  R6, R11, R11
	VLD1 (R11), [V10.S4, V11.S4]     // leg5 (in5)
	ADD  R6, R11, R11
	VLD1 (R11), [V12.S4, V13.S4]     // leg6 (in6)
	ADD  R6, R11, R11
	VLD1 (R11), [V14.S4, V15.S4]     // leg7 (in7)

	// x1 = leg1*tw1  (tw1 at R8)
	VLD1 (R8), [V28.S2]
	VDUP V28.S[0], V17.S4                                // c
	VDUP V28.S[1], V18.S4                                // d
	VREV64 V2.S4, V19.S4                              // brev_lo
	VREV64 V3.S4, V20.S4                              // brev_hi
	WORD $0x6E31DC55     // FMUL V21.4S, V2.4S, V17.4S (t_lo=leg_lo*c)
	WORD $0x6E32DE70     // FMUL V16.4S, V19.4S, V18.4S (p_lo=brev_lo*d)
	VEOR V31.B16, V16.B16, V16.B16                      // p_lo ^= negre
	WORD $0x4E30D6A2     // FADD V2.4S, V21.4S, V16.4S (x1_lo)
	WORD $0x6E31DC75     // FMUL V21.4S, V3.4S, V17.4S (t_hi=leg_hi*c)
	WORD $0x6E32DE90     // FMUL V16.4S, V20.4S, V18.4S (p_hi=brev_hi*d)
	VEOR V31.B16, V16.B16, V16.B16                      // p_hi ^= negre
	WORD $0x4E30D6A3     // FADD V3.4S, V21.4S, V16.4S (x1_hi)
	MOVD R8, R13

	ADD  R14, R13, R13
	// x2 = leg2*tw2  (tw2 at R13)
	VLD1 (R13), [V28.S2]
	VDUP V28.S[0], V17.S4                                // c
	VDUP V28.S[1], V18.S4                                // d
	VREV64 V4.S4, V19.S4                              // brev_lo
	VREV64 V5.S4, V20.S4                              // brev_hi
	WORD $0x6E31DC95     // FMUL V21.4S, V4.4S, V17.4S (t_lo=leg_lo*c)
	WORD $0x6E32DE70     // FMUL V16.4S, V19.4S, V18.4S (p_lo=brev_lo*d)
	VEOR V31.B16, V16.B16, V16.B16                      // p_lo ^= negre
	WORD $0x4E30D6A4     // FADD V4.4S, V21.4S, V16.4S (x2_lo)
	WORD $0x6E31DCB5     // FMUL V21.4S, V5.4S, V17.4S (t_hi=leg_hi*c)
	WORD $0x6E32DE90     // FMUL V16.4S, V20.4S, V18.4S (p_hi=brev_hi*d)
	VEOR V31.B16, V16.B16, V16.B16                      // p_hi ^= negre
	WORD $0x4E30D6A5     // FADD V5.4S, V21.4S, V16.4S (x2_hi)

	ADD  R14, R13, R13
	// x3 = leg3*tw3  (tw3 at R13)
	VLD1 (R13), [V28.S2]
	VDUP V28.S[0], V17.S4                                // c
	VDUP V28.S[1], V18.S4                                // d
	VREV64 V6.S4, V19.S4                              // brev_lo
	VREV64 V7.S4, V20.S4                              // brev_hi
	WORD $0x6E31DCD5     // FMUL V21.4S, V6.4S, V17.4S (t_lo=leg_lo*c)
	WORD $0x6E32DE70     // FMUL V16.4S, V19.4S, V18.4S (p_lo=brev_lo*d)
	VEOR V31.B16, V16.B16, V16.B16                      // p_lo ^= negre
	WORD $0x4E30D6A6     // FADD V6.4S, V21.4S, V16.4S (x3_lo)
	WORD $0x6E31DCF5     // FMUL V21.4S, V7.4S, V17.4S (t_hi=leg_hi*c)
	WORD $0x6E32DE90     // FMUL V16.4S, V20.4S, V18.4S (p_hi=brev_hi*d)
	VEOR V31.B16, V16.B16, V16.B16                      // p_hi ^= negre
	WORD $0x4E30D6A7     // FADD V7.4S, V21.4S, V16.4S (x3_hi)

	ADD  R14, R13, R13
	// x4 = leg4*tw4  (tw4 at R13)
	VLD1 (R13), [V28.S2]
	VDUP V28.S[0], V17.S4                                // c
	VDUP V28.S[1], V18.S4                                // d
	VREV64 V8.S4, V19.S4                              // brev_lo
	VREV64 V9.S4, V20.S4                              // brev_hi
	WORD $0x6E31DD15     // FMUL V21.4S, V8.4S, V17.4S (t_lo=leg_lo*c)
	WORD $0x6E32DE70     // FMUL V16.4S, V19.4S, V18.4S (p_lo=brev_lo*d)
	VEOR V31.B16, V16.B16, V16.B16                      // p_lo ^= negre
	WORD $0x4E30D6A8     // FADD V8.4S, V21.4S, V16.4S (x4_lo)
	WORD $0x6E31DD35     // FMUL V21.4S, V9.4S, V17.4S (t_hi=leg_hi*c)
	WORD $0x6E32DE90     // FMUL V16.4S, V20.4S, V18.4S (p_hi=brev_hi*d)
	VEOR V31.B16, V16.B16, V16.B16                      // p_hi ^= negre
	WORD $0x4E30D6A9     // FADD V9.4S, V21.4S, V16.4S (x4_hi)

	ADD  R14, R13, R13
	// x5 = leg5*tw5  (tw5 at R13)
	VLD1 (R13), [V28.S2]
	VDUP V28.S[0], V17.S4                                // c
	VDUP V28.S[1], V18.S4                                // d
	VREV64 V10.S4, V19.S4                              // brev_lo
	VREV64 V11.S4, V20.S4                              // brev_hi
	WORD $0x6E31DD55     // FMUL V21.4S, V10.4S, V17.4S (t_lo=leg_lo*c)
	WORD $0x6E32DE70     // FMUL V16.4S, V19.4S, V18.4S (p_lo=brev_lo*d)
	VEOR V31.B16, V16.B16, V16.B16                      // p_lo ^= negre
	WORD $0x4E30D6AA     // FADD V10.4S, V21.4S, V16.4S (x5_lo)
	WORD $0x6E31DD75     // FMUL V21.4S, V11.4S, V17.4S (t_hi=leg_hi*c)
	WORD $0x6E32DE90     // FMUL V16.4S, V20.4S, V18.4S (p_hi=brev_hi*d)
	VEOR V31.B16, V16.B16, V16.B16                      // p_hi ^= negre
	WORD $0x4E30D6AB     // FADD V11.4S, V21.4S, V16.4S (x5_hi)

	ADD  R14, R13, R13
	// x6 = leg6*tw6  (tw6 at R13)
	VLD1 (R13), [V28.S2]
	VDUP V28.S[0], V17.S4                                // c
	VDUP V28.S[1], V18.S4                                // d
	VREV64 V12.S4, V19.S4                              // brev_lo
	VREV64 V13.S4, V20.S4                              // brev_hi
	WORD $0x6E31DD95     // FMUL V21.4S, V12.4S, V17.4S (t_lo=leg_lo*c)
	WORD $0x6E32DE70     // FMUL V16.4S, V19.4S, V18.4S (p_lo=brev_lo*d)
	VEOR V31.B16, V16.B16, V16.B16                      // p_lo ^= negre
	WORD $0x4E30D6AC     // FADD V12.4S, V21.4S, V16.4S (x6_lo)
	WORD $0x6E31DDB5     // FMUL V21.4S, V13.4S, V17.4S (t_hi=leg_hi*c)
	WORD $0x6E32DE90     // FMUL V16.4S, V20.4S, V18.4S (p_hi=brev_hi*d)
	VEOR V31.B16, V16.B16, V16.B16                      // p_hi ^= negre
	WORD $0x4E30D6AD     // FADD V13.4S, V21.4S, V16.4S (x6_hi)

	ADD  R14, R13, R13
	// x7 = leg7*tw7  (tw7 at R13)
	VLD1 (R13), [V28.S2]
	VDUP V28.S[0], V17.S4                                // c
	VDUP V28.S[1], V18.S4                                // d
	VREV64 V14.S4, V19.S4                              // brev_lo
	VREV64 V15.S4, V20.S4                              // brev_hi
	WORD $0x6E31DDD5     // FMUL V21.4S, V14.4S, V17.4S (t_lo=leg_lo*c)
	WORD $0x6E32DE70     // FMUL V16.4S, V19.4S, V18.4S (p_lo=brev_lo*d)
	VEOR V31.B16, V16.B16, V16.B16                      // p_lo ^= negre
	WORD $0x4E30D6AE     // FADD V14.4S, V21.4S, V16.4S (x7_lo)
	WORD $0x6E31DDF5     // FMUL V21.4S, V15.4S, V17.4S (t_hi=leg_hi*c)
	WORD $0x6E32DE90     // FMUL V16.4S, V20.4S, V18.4S (p_hi=brev_hi*d)
	VEOR V31.B16, V16.B16, V16.B16                      // p_hi ^= negre
	WORD $0x4E30D6AF     // FADD V15.4S, V21.4S, V16.4S (x7_hi)

	// ===== lo half: even/odd radix-4 + eighth-root recombine =====
	WORD $0x4E28D410     // FADD V16.4S, V0.4S, V8.4S (e0=x0+x4)
	WORD $0x4EA8D411     // FSUB V17.4S, V0.4S, V8.4S (e1=x0-x4)
	WORD $0x4E2CD492     // FADD V18.4S, V4.4S, V12.4S (e2=x2+x6)
	WORD $0x4EACD493     // FSUB V19.4S, V4.4S, V12.4S (e3=x2-x6)
	WORD $0x4E32D614     // FADD V20.4S, V16.4S, V18.4S (E0=e0+e2)
	WORD $0x4EB2D615     // FSUB V21.4S, V16.4S, V18.4S (E2=e0-e2)
	VREV64 V19.S4, V22.S4                               // rev(e3)
	VEOR V30.B16, V22.B16, V22.B16                     // jrot(e3)=(im,-re)
	WORD $0x4E36D630     // FADD V16.4S, V17.4S, V22.4S (E1=e1+jrot(e3))
	WORD $0x4EB6D631     // FSUB V17.4S, V17.4S, V22.4S (E3=e1-jrot(e3))
	WORD $0x4E2AD452     // FADD V18.4S, V2.4S, V10.4S (oo0=x1+x5)
	WORD $0x4EAAD453     // FSUB V19.4S, V2.4S, V10.4S (oo1=x1-x5)
	WORD $0x4E2ED4D6     // FADD V22.4S, V6.4S, V14.4S (oo2=x3+x7)
	WORD $0x4EAED4D7     // FSUB V23.4S, V6.4S, V14.4S (oo3=x3-x7)
	WORD $0x4E36D658     // FADD V24.4S, V18.4S, V22.4S (O0=oo0+oo2)
	WORD $0x4EB6D659     // FSUB V25.4S, V18.4S, V22.4S (O2=oo0-oo2)
	VREV64 V23.S4, V26.S4                               // rev(oo3)
	VEOR V30.B16, V26.B16, V26.B16                     // jrot(oo3)=(im,-re)
	WORD $0x4E3AD672     // FADD V18.4S, V19.4S, V26.4S (O1=oo1+jrot(oo3))
	WORD $0x4EBAD673     // FSUB V19.4S, V19.4S, V26.4S (O3=oo1-jrot(oo3))
	VREV64 V18.S4, V22.S4                               // rev(O1)=(im,re)
	VEOR V31.B16, V22.B16, V22.B16                     // negre(rev(O1))=(-im,re)
	WORD $0x4EB6D656     // FSUB V22.4S, V18.4S, V22.4S ((re+im, im-re)=O1-negre(rev(O1)))
	WORD $0x6E3DDED6     // FMUL V22.4S, V22.4S, V29.4S (w1=r*(...))
	VREV64 V25.S4, V23.S4                               // rev(O2)=(im,re)
	VEOR V30.B16, V23.B16, V23.B16                     // w2=jrot(O2)=(im,-re)
	VREV64 V19.S4, V26.S4                               // rev(O3)=(im,re)
	VEOR V30.B16, V26.B16, V26.B16                     // negim(rev(O3))=(im,-re)
	WORD $0x4EB3D75A     // FSUB V26.4S, V26.4S, V19.4S ((im-re, -re-im)=negim(rev(O3))-O3)
	WORD $0x6E3DDF5A     // FMUL V26.4S, V26.4S, V29.4S (w3=r*(...))
	MOVD R0, R12          // running store pointer
	WORD $0x4E38D69B     // FADD V27.4S, V20.4S, V24.4S (out0=E0+O0)
	VST1 [V27.S4], (R12)
	ADD  R7, R12, R12
	WORD $0x4E36D61B     // FADD V27.4S, V16.4S, V22.4S (out1=E1+w1)
	VST1 [V27.S4], (R12)
	ADD  R7, R12, R12
	WORD $0x4E37D6BB     // FADD V27.4S, V21.4S, V23.4S (out2=E2+w2)
	VST1 [V27.S4], (R12)
	ADD  R7, R12, R12
	WORD $0x4E3AD63B     // FADD V27.4S, V17.4S, V26.4S (out3=E3+w3)
	VST1 [V27.S4], (R12)
	ADD  R7, R12, R12
	WORD $0x4EB8D69B     // FSUB V27.4S, V20.4S, V24.4S (out4=E0-O0)
	VST1 [V27.S4], (R12)
	ADD  R7, R12, R12
	WORD $0x4EB6D61B     // FSUB V27.4S, V16.4S, V22.4S (out5=E1-w1)
	VST1 [V27.S4], (R12)
	ADD  R7, R12, R12
	WORD $0x4EB7D6BB     // FSUB V27.4S, V21.4S, V23.4S (out6=E2-w2)
	VST1 [V27.S4], (R12)
	ADD  R7, R12, R12
	WORD $0x4EBAD63B     // FSUB V27.4S, V17.4S, V26.4S (out7=E3-w3)
	VST1 [V27.S4], (R12)

	// ===== hi half: even/odd radix-4 + eighth-root recombine =====
	WORD $0x4E29D430     // FADD V16.4S, V1.4S, V9.4S (e0=x0+x4)
	WORD $0x4EA9D431     // FSUB V17.4S, V1.4S, V9.4S (e1=x0-x4)
	WORD $0x4E2DD4B2     // FADD V18.4S, V5.4S, V13.4S (e2=x2+x6)
	WORD $0x4EADD4B3     // FSUB V19.4S, V5.4S, V13.4S (e3=x2-x6)
	WORD $0x4E32D614     // FADD V20.4S, V16.4S, V18.4S (E0=e0+e2)
	WORD $0x4EB2D615     // FSUB V21.4S, V16.4S, V18.4S (E2=e0-e2)
	VREV64 V19.S4, V22.S4                               // rev(e3)
	VEOR V30.B16, V22.B16, V22.B16                     // jrot(e3)=(im,-re)
	WORD $0x4E36D630     // FADD V16.4S, V17.4S, V22.4S (E1=e1+jrot(e3))
	WORD $0x4EB6D631     // FSUB V17.4S, V17.4S, V22.4S (E3=e1-jrot(e3))
	WORD $0x4E2BD472     // FADD V18.4S, V3.4S, V11.4S (oo0=x1+x5)
	WORD $0x4EABD473     // FSUB V19.4S, V3.4S, V11.4S (oo1=x1-x5)
	WORD $0x4E2FD4F6     // FADD V22.4S, V7.4S, V15.4S (oo2=x3+x7)
	WORD $0x4EAFD4F7     // FSUB V23.4S, V7.4S, V15.4S (oo3=x3-x7)
	WORD $0x4E36D658     // FADD V24.4S, V18.4S, V22.4S (O0=oo0+oo2)
	WORD $0x4EB6D659     // FSUB V25.4S, V18.4S, V22.4S (O2=oo0-oo2)
	VREV64 V23.S4, V26.S4                               // rev(oo3)
	VEOR V30.B16, V26.B16, V26.B16                     // jrot(oo3)=(im,-re)
	WORD $0x4E3AD672     // FADD V18.4S, V19.4S, V26.4S (O1=oo1+jrot(oo3))
	WORD $0x4EBAD673     // FSUB V19.4S, V19.4S, V26.4S (O3=oo1-jrot(oo3))
	VREV64 V18.S4, V22.S4                               // rev(O1)=(im,re)
	VEOR V31.B16, V22.B16, V22.B16                     // negre(rev(O1))=(-im,re)
	WORD $0x4EB6D656     // FSUB V22.4S, V18.4S, V22.4S ((re+im, im-re)=O1-negre(rev(O1)))
	WORD $0x6E3DDED6     // FMUL V22.4S, V22.4S, V29.4S (w1=r*(...))
	VREV64 V25.S4, V23.S4                               // rev(O2)=(im,re)
	VEOR V30.B16, V23.B16, V23.B16                     // w2=jrot(O2)=(im,-re)
	VREV64 V19.S4, V26.S4                               // rev(O3)=(im,re)
	VEOR V30.B16, V26.B16, V26.B16                     // negim(rev(O3))=(im,-re)
	WORD $0x4EB3D75A     // FSUB V26.4S, V26.4S, V19.4S ((im-re, -re-im)=negim(rev(O3))-O3)
	WORD $0x6E3DDF5A     // FMUL V26.4S, V26.4S, V29.4S (w3=r*(...))
	ADD  $16, R0, R12          // running store pointer
	WORD $0x4E38D69B     // FADD V27.4S, V20.4S, V24.4S (out0=E0+O0)
	VST1 [V27.S4], (R12)
	ADD  R7, R12, R12
	WORD $0x4E36D61B     // FADD V27.4S, V16.4S, V22.4S (out1=E1+w1)
	VST1 [V27.S4], (R12)
	ADD  R7, R12, R12
	WORD $0x4E37D6BB     // FADD V27.4S, V21.4S, V23.4S (out2=E2+w2)
	VST1 [V27.S4], (R12)
	ADD  R7, R12, R12
	WORD $0x4E3AD63B     // FADD V27.4S, V17.4S, V26.4S (out3=E3+w3)
	VST1 [V27.S4], (R12)
	ADD  R7, R12, R12
	WORD $0x4EB8D69B     // FSUB V27.4S, V20.4S, V24.4S (out4=E0-O0)
	VST1 [V27.S4], (R12)
	ADD  R7, R12, R12
	WORD $0x4EB6D61B     // FSUB V27.4S, V16.4S, V22.4S (out5=E1-w1)
	VST1 [V27.S4], (R12)
	ADD  R7, R12, R12
	WORD $0x4EB7D6BB     // FSUB V27.4S, V21.4S, V23.4S (out6=E2-w2)
	VST1 [V27.S4], (R12)
	ADD  R7, R12, R12
	WORD $0x4EBAD63B     // FSUB V27.4S, V17.4S, V26.4S (out7=E3-w3)
	VST1 [V27.S4], (R12)


	ADD  $32, R1, R1     // advance src group
	ADD  $32, R0, R0     // advance dst out0 group
	ADD  $8, R8, R8      // advance tw one complex64
	ADD  $1, R10, R10
	JMP  r8_p_loop

r8_next_section:
	ADD  R15, R0, R0     // skip over the out1..out7 blocks we already wrote
	ADD  $1, R9, R9
	JMP  r8_section_loop

r8_done:
	RET
