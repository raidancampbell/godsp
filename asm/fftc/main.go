//go:build ignore

// Command fftc generates AVX2 kernels for FFTC butterflies.
// Run via: go generate ./...
package main

import (
	"github.com/mmcloughlin/avo/attr"
	. "github.com/mmcloughlin/avo/build"
	. "github.com/mmcloughlin/avo/operand"
	"github.com/mmcloughlin/avo/reg"
)

func main() {
	Package("github.com/raidancampbell/godsp")
	ConstraintExpr("amd64")

	// Sign mask for the ±i rotations. After VPERMILPS swaps (re,im)→(im,re) in
	// each complex64 lane, XOR with this mask negates the odd (imag-slot) lanes,
	// yielding (im,−re) — i.e. multiplication by −i. The even lanes stay 0.
	signMem := GLOBL("fftc_jrot_sign", attr.RODATA|attr.NOPTR)
	for lane, v := range []uint32{0, 0x80000000, 0, 0x80000000, 0, 0x80000000, 0, 0x80000000} {
		DATA(lane*4, U32(v))
	}

	genBfly4(signMem)
	genBfly3(signMem)

	Generate()
}

// broadcastFloat32Const declares a single-float32 RODATA symbol holding the
// given IEEE-754 bit pattern and broadcasts it across all eight lanes of a YMM.
func broadcastFloat32Const(name string, bits uint32) reg.VecVirtual {
	mem := GLOBL(name, attr.RODATA|attr.NOPTR)
	DATA(0, U32(bits))
	v := YMM()
	VBROADCASTSS(mem, v)
	return v
}

// jRotNeg returns v·(−i) for four interleaved complex64 lanes: VPERMILPS swaps
// (re,im)→(im,re) in each lane, then XOR with the sign mask negates the odd
// (imag-slot) lanes, yielding (im, −re).
func jRotNeg(sign, v reg.VecVirtual) reg.VecVirtual {
	rot := YMM()
	VPERMILPS(Imm(0xB1), v, rot)
	VXORPS(sign, rot, rot)
	return rot
}

func genBfly3(signMem Mem) {
	TEXT("bfly3AVX2", NOSPLIT, "func(out []complex64, tw []complex64, mv int, m int)")
	Doc("bfly3AVX2 computes radix-3 FFTC butterflies for k in [0,mv), four complex lanes per iteration; the Go dispatch handles the scalar tail.")

	outPtr := Load(Param("out").Base(), GP64())
	twPtr := Load(Param("tw").Base(), GP64())
	mv := Load(Param("mv"), GP64())
	m := Load(Param("m"), GP64())

	// Load the ±i sign mask and broadcast the radix-3 scalar constants once.
	sign := YMM()
	VMOVUPS(signMem, sign)
	half := broadcastFloat32Const("fftc_neg_half", 0xbf000000)    // −0.5
	sin := broadcastFloat32Const("fftc_sqrt3_over_2", 0x3f5db3d7) // sin(2π/3)=√3/2

	k := GP64()
	XORQ(k, k)

	Label("loop3")
	CMPQ(k, mv)
	JGE(LabelRef("done3"))

	// Element pointers. Each complex64 is 8 bytes; m is a count of complex64,
	// so the radix block stride is m*8 bytes (Index:m, Scale:8).
	x0p := GP64()
	LEAQ(Mem{Base: outPtr, Index: k, Scale: 8}, x0p)
	x1p := GP64()
	LEAQ(Mem{Base: x0p, Index: m, Scale: 8}, x1p)
	x2p := GP64()
	LEAQ(Mem{Base: x1p, Index: m, Scale: 8}, x2p)

	t1p := GP64()
	LEAQ(Mem{Base: twPtr, Index: k, Scale: 8}, t1p)
	t2p := GP64()
	LEAQ(Mem{Base: t1p, Index: m, Scale: 8}, t2p)

	t0 := YMM()
	x1 := YMM()
	x2 := YMM()
	w1 := YMM()
	w2 := YMM()
	VMOVUPS(Mem{Base: x0p}, t0)
	VMOVUPS(Mem{Base: x1p}, x1)
	VMOVUPS(Mem{Base: x2p}, x2)
	VMOVUPS(Mem{Base: t1p}, w1)
	VMOVUPS(Mem{Base: t2p}, w2)

	// t1 = x1·tw[k]; t2 = x2·tw[m+k].
	t1 := complexMul(x1, w1)
	t2 := complexMul(x2, w2)

	// sum = t1 + t2; dif = t1 − t2.
	sum := YMM()
	dif := YMM()
	VADDPS(t2, t1, sum)
	VSUBPS(t2, t1, dif)

	// y0 = t0 + sum.
	y0 := YMM()
	VADDPS(sum, t0, y0)

	// a = t0 − ½·sum (half holds −0.5, so a = t0 + half·sum).
	halfSum := YMM()
	VMULPS(half, sum, halfSum)
	a := YMM()
	VADDPS(halfSum, t0, a)

	// w = (√3/2)·dif; rot = w·(−i). y1 = a + rot; y2 = a − rot.
	w := YMM()
	VMULPS(sin, dif, w)
	rot := jRotNeg(sign, w)
	y1 := YMM()
	y2 := YMM()
	VADDPS(rot, a, y1)
	VSUBPS(rot, a, y2)

	VMOVUPS(y0, Mem{Base: x0p})
	VMOVUPS(y1, Mem{Base: x1p})
	VMOVUPS(y2, Mem{Base: x2p})

	ADDQ(Imm(4), k)
	JMP(LabelRef("loop3"))

	Label("done3")
	VZEROUPPER()
	RET()
}

func genBfly4(signMem Mem) {
	TEXT("bfly4AVX2", NOSPLIT, "func(out []complex64, tw []complex64, mv int, m int)")
	Doc("bfly4AVX2 computes radix-4 FFTC butterflies for k in [0,mv), four complex lanes per iteration; the Go dispatch handles the scalar tail.")

	outPtr := Load(Param("out").Base(), GP64())
	twPtr := Load(Param("tw").Base(), GP64())
	mv := Load(Param("mv"), GP64())
	m := Load(Param("m"), GP64())

	// Load the ±i sign mask once.
	sign := YMM()
	VMOVUPS(signMem, sign)

	k := GP64()
	XORQ(k, k)

	Label("loop")
	CMPQ(k, mv)
	JGE(LabelRef("done"))

	// Element pointers. Each complex64 is 8 bytes; m is a count of complex64,
	// so the radix block stride is m*8 bytes (Index:m, Scale:8).
	x0p := GP64()
	LEAQ(Mem{Base: outPtr, Index: k, Scale: 8}, x0p)
	x1p := GP64()
	LEAQ(Mem{Base: x0p, Index: m, Scale: 8}, x1p)
	x2p := GP64()
	LEAQ(Mem{Base: x1p, Index: m, Scale: 8}, x2p)
	x3p := GP64()
	LEAQ(Mem{Base: x2p, Index: m, Scale: 8}, x3p)

	t0p := GP64()
	LEAQ(Mem{Base: twPtr, Index: k, Scale: 8}, t0p)
	t1p := GP64()
	LEAQ(Mem{Base: t0p, Index: m, Scale: 8}, t1p)
	t2p := GP64()
	LEAQ(Mem{Base: t1p, Index: m, Scale: 8}, t2p)

	x0 := YMM()
	x1 := YMM()
	x2 := YMM()
	x3 := YMM()
	w0 := YMM()
	w1 := YMM()
	w2 := YMM()
	VMOVUPS(Mem{Base: x0p}, x0)
	VMOVUPS(Mem{Base: x1p}, x1)
	VMOVUPS(Mem{Base: x2p}, x2)
	VMOVUPS(Mem{Base: x3p}, x3)
	VMOVUPS(Mem{Base: t0p}, w0)
	VMOVUPS(Mem{Base: t1p}, w1)
	VMOVUPS(Mem{Base: t2p}, w2)

	s0 := complexMul(x1, w0)
	s1 := complexMul(x2, w1)
	s2 := complexMul(x3, w2)

	// d0 = x0 - s1; a0 = x0 + s1; a1 = s0 + s2; d1 = s0 - s2.
	d0 := YMM()
	a0 := YMM()
	a1 := YMM()
	d1 := YMM()
	VSUBPS(s1, x0, d0)
	VADDPS(s1, x0, a0)
	VADDPS(s2, s0, a1)
	VSUBPS(s2, s0, d1)

	// y0 = a0 + a1; y2 = a0 - a1.
	y0 := YMM()
	y2 := YMM()
	VADDPS(a1, a0, y0)
	VSUBPS(a1, a0, y2)

	// rot = (im(d1), −re(d1)) = d1 ·(−i). y1 = d0 + rot; y3 = d0 − rot.
	rot := YMM()
	VPERMILPS(Imm(0xB1), d1, rot) // swap re/im → (im, re)
	VXORPS(sign, rot, rot)        // negate odd lanes → (im, −re)
	y1 := YMM()
	y3 := YMM()
	VADDPS(rot, d0, y1)
	VSUBPS(rot, d0, y3)

	VMOVUPS(y0, Mem{Base: x0p})
	VMOVUPS(y1, Mem{Base: x1p})
	VMOVUPS(y2, Mem{Base: x2p})
	VMOVUPS(y3, Mem{Base: x3p})

	ADDQ(Imm(4), k)
	JMP(LabelRef("loop"))

	Label("done")
	VZEROUPPER()
	RET()
}

// complexMul returns a*b for four interleaved complex64 lanes:
// [ar0,ai0,ar1,ai1,...] · [br0,bi0,...]. Result re = ar*br − ai*bi,
// im = ai*br + ar*bi.
//
// Let t1 = a·dup(re_b) = [ar*br, ai*br] and t2 = swap(a)·dup(im_b) =
// [ai*bi, ar*bi]. The answer is ADDSUB(t1, t2): even (re) = t1−t2, odd (im) =
// t1+t2. VFMADDSUB231PS(SRC2,SRC3,DEST) gives even = SRC2*SRC3−DEST and odd =
// SRC2*SRC3+DEST, so DEST must hold t2 and the fused product must be t1 = a·br.
func complexMul(a, b reg.VecVirtual) reg.VecVirtual {
	br := YMM()
	bi := YMM()
	VSHUFPS(Imm(0xA0), b, b, br) // duplicate real parts  → [br,br,br',br']
	VSHUFPS(Imm(0xF5), b, b, bi) // duplicate imag parts  → [bi,bi,bi',bi']
	sw := YMM()
	VPERMILPS(Imm(0xB1), a, sw) // swap re/im → [ai,ar,...]
	t2 := YMM()
	VMULPS(bi, sw, t2)        // t2 = swap(a)·bi = [ai*bi, ar*bi]
	VFMADDSUB231PS(a, br, t2) // t2 = (a·br) ADDSUB t2 = [ar*br−ai*bi, ai*br+ar*bi]
	return t2
}
