#include "textflag.h"

// Hand-written NEON atan2 kernel (avo has no arm64 backend). Mirrors the
// scalar atan2Approx lane-wise, 4 lanes at a time, using true FDIV so each
// lane is bit-close to the scalar oracle (gated by TestAtan2Block_MatchesScalar).
//
// dst[i] = atan2Approx(y[i], x[i]) * scale, i in [0,n), n%4==0.
//
// Go's arm64 assembler exposes only VFMLA/VFMLS as float-vector mnemonics, so
// every other float .4S op is WORD-encoded. Encodings decode-verified via
// objdump on the arm64 target (dest in bits [4:0], objdump prints dest last):
//   FABS  Vd.4S, Vn.4S        base 0x4EA0F800 | Rn<<5 | Rd
//   FNEG  Vd.4S, Vn.4S        base 0x6EA0F800 | Rn<<5 | Rd
//   FSUB  Vd.4S, Vn.4S, Vm.4S base 0x4EA0D400 | Rm<<16 | Rn<<5 | Rd
//   FMUL  Vd.4S, Vn.4S, Vm.4S base 0x6E20DC00 | Rm<<16 | Rn<<5 | Rd
//   FDIV  Vd.4S, Vn.4S, Vm.4S base 0x6E20FC00 | Rm<<16 | Rn<<5 | Rd
//   FMAX  Vd.4S, Vn.4S, Vm.4S base 0x4E20F400 | Rm<<16 | Rn<<5 | Rd
//   FMIN  Vd.4S, Vn.4S, Vm.4S base 0x4EA0F400 | Rm<<16 | Rn<<5 | Rd
//   FCMGT Vd.4S, Vn.4S, Vm.4S base 0x6EA0E400 | Rm<<16 | Rn<<5 | Rd
//   FCMEQ Vd.4S, Vn.4S, Vm.4S base 0x4E20E400 | Rm<<16 | Rn<<5 | Rd
// Real mnemonics: VLD1/VST1/VDUP/VEOR/VMOV/VBIT/VFMLA.
//   VFMLA Vm,Vn,Vd => Vd += Vn*Vm (dest is 3rd operand, read-modify).
//   VBIT  Vm,Vn,Vd => Vd = Vm ? Vn : Vd (mask is 1st, alt is 2nd, dest 3rd).
//
// Register map: V0=x V1=y V2=ax/mNegX V3=ay/mNegY V4=num V5=den V6=mSwap
//   V8=mZero V9=t V10=t2 V11=p V12=acc V13=tt2 V14=a V15=alt
//   V16..V23=coeff0..7 V24=pi/2 V25=pi V26=scaleVec V27=zero

// 4-wide broadcast constants (each 16 bytes = 4 float32 copies).
DATA atan2_coeff0<>+0(SB)/4, $0x3b3bd74a
DATA atan2_coeff0<>+4(SB)/4, $0x3b3bd74a
DATA atan2_coeff0<>+8(SB)/4, $0x3b3bd74a
DATA atan2_coeff0<>+12(SB)/4, $0x3b3bd74a
GLOBL atan2_coeff0<>(SB), RODATA|NOPTR, $16

DATA atan2_coeff1<>+0(SB)/4, $0xbc846e02
DATA atan2_coeff1<>+4(SB)/4, $0xbc846e02
DATA atan2_coeff1<>+8(SB)/4, $0xbc846e02
DATA atan2_coeff1<>+12(SB)/4, $0xbc846e02
GLOBL atan2_coeff1<>(SB), RODATA|NOPTR, $16

DATA atan2_coeff2<>+0(SB)/4, $0x3d2fc1fe
DATA atan2_coeff2<>+4(SB)/4, $0x3d2fc1fe
DATA atan2_coeff2<>+8(SB)/4, $0x3d2fc1fe
DATA atan2_coeff2<>+12(SB)/4, $0x3d2fc1fe
GLOBL atan2_coeff2<>(SB), RODATA|NOPTR, $16

DATA atan2_coeff3<>+0(SB)/4, $0xbd9a3174
DATA atan2_coeff3<>+4(SB)/4, $0xbd9a3174
DATA atan2_coeff3<>+8(SB)/4, $0xbd9a3174
DATA atan2_coeff3<>+12(SB)/4, $0xbd9a3174
GLOBL atan2_coeff3<>(SB), RODATA|NOPTR, $16

DATA atan2_coeff4<>+0(SB)/4, $0x3dda3d83
DATA atan2_coeff4<>+4(SB)/4, $0x3dda3d83
DATA atan2_coeff4<>+8(SB)/4, $0x3dda3d83
DATA atan2_coeff4<>+12(SB)/4, $0x3dda3d83
GLOBL atan2_coeff4<>(SB), RODATA|NOPTR, $16

DATA atan2_coeff5<>+0(SB)/4, $0xbe117fc7
DATA atan2_coeff5<>+4(SB)/4, $0xbe117fc7
DATA atan2_coeff5<>+8(SB)/4, $0xbe117fc7
DATA atan2_coeff5<>+12(SB)/4, $0xbe117fc7
GLOBL atan2_coeff5<>(SB), RODATA|NOPTR, $16

DATA atan2_coeff6<>+0(SB)/4, $0x3e4cbbe5
DATA atan2_coeff6<>+4(SB)/4, $0x3e4cbbe5
DATA atan2_coeff6<>+8(SB)/4, $0x3e4cbbe5
DATA atan2_coeff6<>+12(SB)/4, $0x3e4cbbe5
GLOBL atan2_coeff6<>(SB), RODATA|NOPTR, $16

DATA atan2_coeff7<>+0(SB)/4, $0xbeaaaa6c
DATA atan2_coeff7<>+4(SB)/4, $0xbeaaaa6c
DATA atan2_coeff7<>+8(SB)/4, $0xbeaaaa6c
DATA atan2_coeff7<>+12(SB)/4, $0xbeaaaa6c
GLOBL atan2_coeff7<>(SB), RODATA|NOPTR, $16

DATA atan2_pi2<>+0(SB)/4, $0x3fc90fdb
DATA atan2_pi2<>+4(SB)/4, $0x3fc90fdb
DATA atan2_pi2<>+8(SB)/4, $0x3fc90fdb
DATA atan2_pi2<>+12(SB)/4, $0x3fc90fdb
GLOBL atan2_pi2<>(SB), RODATA|NOPTR, $16

DATA atan2_pi<>+0(SB)/4, $0x40490fdb
DATA atan2_pi<>+4(SB)/4, $0x40490fdb
DATA atan2_pi<>+8(SB)/4, $0x40490fdb
DATA atan2_pi<>+12(SB)/4, $0x40490fdb
GLOBL atan2_pi<>(SB), RODATA|NOPTR, $16

// func atan2BlockNEON(dst, y, x []float32, n int, scale float32)
TEXT ·atan2BlockNEON(SB), NOSPLIT, $0-84
	MOVD dst_base+0(FP), R0
	MOVD y_base+24(FP), R1
	MOVD x_base+48(FP), R2
	MOVD n+72(FP), R3

	// zero = 0
	VEOR V27.B16, V27.B16, V27.B16

	// broadcast scale (float32 at +80) into all 4 lanes of V26
	FMOVS scale+80(FP), F26
	VDUP  V26.S[0], V26.S4

	// load broadcast constants
	MOVD $atan2_coeff0<>(SB), R6
	VLD1 (R6), [V16.S4]
	MOVD $atan2_coeff1<>(SB), R6
	VLD1 (R6), [V17.S4]
	MOVD $atan2_coeff2<>(SB), R6
	VLD1 (R6), [V18.S4]
	MOVD $atan2_coeff3<>(SB), R6
	VLD1 (R6), [V19.S4]
	MOVD $atan2_coeff4<>(SB), R6
	VLD1 (R6), [V20.S4]
	MOVD $atan2_coeff5<>(SB), R6
	VLD1 (R6), [V21.S4]
	MOVD $atan2_coeff6<>(SB), R6
	VLD1 (R6), [V22.S4]
	MOVD $atan2_coeff7<>(SB), R6
	VLD1 (R6), [V23.S4]
	MOVD $atan2_pi2<>(SB), R6
	VLD1 (R6), [V24.S4]
	MOVD $atan2_pi<>(SB), R6
	VLD1 (R6), [V25.S4]

	MOVD $0, R5 // i
loop:
	CMP  R3, R5
	BGE  done
	VLD1.P 16(R2), [V0.S4]        // x[i..i+3]
	VLD1.P 16(R1), [V1.S4]        // y[i..i+3]

	WORD $0x4EA0F802             // FABS  V2.4S, V0.4S      (ax)
	WORD $0x4EA0F823             // FABS  V3.4S, V1.4S      (ay)
	WORD $0x4EA3F444             // FMIN  V4.4S, V2.4S, V3.4S  (num)
	WORD $0x4E23F445             // FMAX  V5.4S, V2.4S, V3.4S  (den)
	WORD $0x6EA2E466             // FCMGT V6.4S, V3.4S, V2.4S  (mSwap = ay>ax)
	WORD $0x4E3BE4A8             // FCMEQ V8.4S, V5.4S, V27.4S (mZero = den==0)
	WORD $0x6E25FC89             // FDIV  V9.4S, V4.4S, V5.4S  (t = num/den)
	WORD $0x6E29DD2A             // FMUL  V10.4S, V9.4S, V9.4S (t2)

	// Horner: p = coeff0; for i=1..7: acc = coeff_i; acc += p*t2; p = acc.
	// VFMLA Vm,Vn,Vd => Vd += Vn*Vm, so VFMLA V10(t2),V11(p),V12(acc).
	VMOV  V16.B16, V11.B16       // p = coeff0
	VMOV  V17.B16, V12.B16       // acc = coeff1
	VFMLA V10.S4, V11.S4, V12.S4 // acc += p*t2
	VMOV  V12.B16, V11.B16       // p = acc
	VMOV  V18.B16, V12.B16       // acc = coeff2
	VFMLA V10.S4, V11.S4, V12.S4
	VMOV  V12.B16, V11.B16
	VMOV  V19.B16, V12.B16       // acc = coeff3
	VFMLA V10.S4, V11.S4, V12.S4
	VMOV  V12.B16, V11.B16
	VMOV  V20.B16, V12.B16       // acc = coeff4
	VFMLA V10.S4, V11.S4, V12.S4
	VMOV  V12.B16, V11.B16
	VMOV  V21.B16, V12.B16       // acc = coeff5
	VFMLA V10.S4, V11.S4, V12.S4
	VMOV  V12.B16, V11.B16
	VMOV  V22.B16, V12.B16       // acc = coeff6
	VFMLA V10.S4, V11.S4, V12.S4
	VMOV  V12.B16, V11.B16
	VMOV  V23.B16, V12.B16       // acc = coeff7
	VFMLA V10.S4, V11.S4, V12.S4
	VMOV  V12.B16, V11.B16       // p = P(t2)

	// a = t + t*t2*p:  tt2 = t*t2; a = t; a += tt2*p
	WORD  $0x6E2ADD2D             // FMUL V13.4S, V9.4S, V10.4S  (tt2 = t*t2)
	VMOV  V9.B16, V14.B16         // a = t
	VFMLA V13.S4, V11.S4, V14.S4  // a += p*tt2  => a = t + t*t2*p

	// mSwap: a = pi/2 - a on |y|>|x| lanes
	WORD $0x4EAED70F            // FSUB V15.4S, V24.4S, V14.4S (alt = pi/2 - a)
	VBIT V6.B16, V15.B16, V14.B16 // a = mSwap ? alt : a

	// x<0: a = pi - a
	WORD $0x4EAED72F            // FSUB V15.4S, V25.4S, V14.4S (alt = pi - a)
	WORD $0x6EA0E762           // FCMGT V2.4S, V27.4S, V0.4S  (mNegX = 0>x)
	VBIT V2.B16, V15.B16, V14.B16 // a = mNegX ? alt : a

	// y<0: a = -a
	WORD $0x6EA0F9CF           // FNEG  V15.4S, V14.4S        (alt = -a)
	WORD $0x6EA1E763           // FCMGT V3.4S, V27.4S, V1.4S  (mNegY = 0>y)
	VBIT V3.B16, V15.B16, V14.B16 // a = mNegY ? -a : a

	// x==y==0 lanes: force exactly 0 (discard 0/0 NaN)
	VBIT V8.B16, V27.B16, V14.B16 // a = mZero ? 0 : a

	// a *= scale; store
	WORD $0x6E3ADDCE           // FMUL V14.4S, V14.4S, V26.4S
	VST1.P [V14.S4], 16(R0)

	ADD  $4, R5, R5
	B    loop
done:
	RET
