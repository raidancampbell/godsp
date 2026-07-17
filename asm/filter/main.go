//go:build ignore

// Command filter generates the AVX2/FMA complex FIR dot-product kernel.
package main

import (
	. "github.com/mmcloughlin/avo/build"
	. "github.com/mmcloughlin/avo/operand"
	"github.com/mmcloughlin/avo/reg"
)

func main() {
	Package("github.com/raidancampbell/godsp")
	ConstraintExpr("amd64")

	TEXT("complexFIRDotAVX2", NOSPLIT, "func(taps, winR, winI []float32, n int, out *[2]float32)")
	Doc("complexFIRDotAVX2 accumulates the multiple-of-eight FIR prefix with AVX2/FMA.")
	Pragma("noescape")

	taps := Load(Param("taps").Base(), GP64())
	winR := Load(Param("winR").Base(), GP64())
	winI := Load(Param("winI").Base(), GP64())
	n := Load(Param("n"), GP64())
	out := Load(Param("out"), GP64())

	// Eight independent accumulators (four real, four imag) keep four complex
	// FMA pairs in flight so the loop is bound by FMA throughput rather than
	// the latency-4 dependency chain of a single accumulator per lane.
	accR := [4]reg.VecVirtual{YMM(), YMM(), YMM(), YMM()}
	accI := [4]reg.VecVirtual{YMM(), YMM(), YMM(), YMM()}
	for g := 0; g < 4; g++ {
		VXORPS(accR[g], accR[g], accR[g])
		VXORPS(accI[g], accI[g], accI[g])
	}

	i := GP64()
	XORQ(i, i)

	// Main loop: 4 groups of 8 taps (32 taps) per iteration, each group into an
	// independent accumulator pair. Bound at n-32 with a signed compare so the
	// loop is skipped entirely when n < 32.
	n32 := GP64()
	MOVQ(n, n32)
	SUBQ(Imm(32), n32)

	Label("loop32")
	CMPQ(i, n32)
	JG(LabelRef("tail"))
	for g := 0; g < 4; g++ {
		disp := g * 8
		t := YMM()
		r := YMM()
		im := YMM()
		VMOVUPS(Mem{Base: taps, Index: i, Scale: 4, Disp: disp * 4}, t)
		VMOVUPS(Mem{Base: winR, Index: i, Scale: 4, Disp: disp * 4}, r)
		VMOVUPS(Mem{Base: winI, Index: i, Scale: 4, Disp: disp * 4}, im)
		VFMADD231PS(t, r, accR[g])
		VFMADD231PS(t, im, accI[g])
	}
	ADDQ(Imm(32), i)
	JMP(LabelRef("loop32"))

	// Tail loop: fold the remaining (n%32)/8 groups of 8 into accumulator 0.
	// n is a multiple of 8, so this consumes exactly the leftover groups.
	Label("tail")
	n8 := GP64()
	MOVQ(n, n8)
	SUBQ(Imm(8), n8)

	Label("loop8")
	CMPQ(i, n8)
	JG(LabelRef("reduce"))
	t := YMM()
	r := YMM()
	im := YMM()
	VMOVUPS(Mem{Base: taps, Index: i, Scale: 4}, t)
	VMOVUPS(Mem{Base: winR, Index: i, Scale: 4}, r)
	VMOVUPS(Mem{Base: winI, Index: i, Scale: 4}, im)
	VFMADD231PS(t, r, accR[0])
	VFMADD231PS(t, im, accI[0])
	ADDQ(Imm(8), i)
	JMP(LabelRef("loop8"))

	// Combine the four partial accumulators with a fixed summation tree, then
	// reduce each vector lane-wise. Reassociating the partials shifts float32
	// rounding versus a single serial chain, but stays within the scalar-oracle
	// tolerance (the oracle itself uses 4-way ((r0+r1)+(r2+r3)) accumulation).
	Label("reduce")
	VADDPS(accR[1], accR[0], accR[0])
	VADDPS(accR[3], accR[2], accR[2])
	VADDPS(accR[2], accR[0], accR[0])
	VADDPS(accI[1], accI[0], accI[0])
	VADDPS(accI[3], accI[2], accI[2])
	VADDPS(accI[2], accI[0], accI[0])

	sumR := horizontalSum(accR[0])
	sumI := horizontalSum(accI[0])
	VMOVSS(sumR, Mem{Base: out})
	VMOVSS(sumI, Mem{Base: out, Disp: 4})
	VZEROUPPER()
	RET()

	genFirDotReal()

	genComplexFIRDot4()

	Generate()
}

// genFirDotReal emits firDotRealAVX2, the real-valued FIR dot product. It is the
// real half of complexFIRDotAVX2: four independent accumulators keep four FMA
// chains in flight, a 32-tap main loop feeds one group of 8 taps into each, an
// 8-tap tail folds the leftover groups into accumulator 0, and a fixed summation
// tree plus horizontal reduce collapses to one float32. The Go dispatch handles
// the n%8 scalar tail.
func genFirDotReal() {
	TEXT("firDotRealAVX2", NOSPLIT, "func(taps, win []float32, n int, out *float32)")
	Doc("firDotRealAVX2 accumulates the multiple-of-eight real FIR prefix with AVX2/FMA.")
	Pragma("noescape")

	taps := Load(Param("taps").Base(), GP64())
	win := Load(Param("win").Base(), GP64())
	n := Load(Param("n"), GP64())
	out := Load(Param("out"), GP64())

	// Four independent accumulators keep four FMA chains in flight so the loop is
	// bound by FMA throughput rather than the latency-4 single-accumulator chain.
	acc := [4]reg.VecVirtual{YMM(), YMM(), YMM(), YMM()}
	for g := 0; g < 4; g++ {
		VXORPS(acc[g], acc[g], acc[g])
	}

	i := GP64()
	XORQ(i, i)

	// Main loop: 4 groups of 8 taps (32 taps) per iteration into independent
	// accumulators. Bound at n-32 with a signed compare so the loop is skipped
	// entirely when n < 32.
	n32 := GP64()
	MOVQ(n, n32)
	SUBQ(Imm(32), n32)

	Label("loop32")
	CMPQ(i, n32)
	JG(LabelRef("tail"))
	for g := 0; g < 4; g++ {
		disp := g * 8
		t := YMM()
		w := YMM()
		VMOVUPS(Mem{Base: taps, Index: i, Scale: 4, Disp: disp * 4}, t)
		VMOVUPS(Mem{Base: win, Index: i, Scale: 4, Disp: disp * 4}, w)
		VFMADD231PS(t, w, acc[g])
	}
	ADDQ(Imm(32), i)
	JMP(LabelRef("loop32"))

	// Tail loop: fold the remaining (n%32)/8 groups of 8 into accumulator 0.
	// n is a multiple of 8, so this consumes exactly the leftover groups.
	Label("tail")
	n8 := GP64()
	MOVQ(n, n8)
	SUBQ(Imm(8), n8)

	Label("loop8")
	CMPQ(i, n8)
	JG(LabelRef("reduce"))
	t := YMM()
	w := YMM()
	VMOVUPS(Mem{Base: taps, Index: i, Scale: 4}, t)
	VMOVUPS(Mem{Base: win, Index: i, Scale: 4}, w)
	VFMADD231PS(t, w, acc[0])
	ADDQ(Imm(8), i)
	JMP(LabelRef("loop8"))

	// Combine the four partial accumulators with a fixed summation tree, then
	// reduce lane-wise. The reassociation matches the scalar oracle's 4-way
	// ((r0+r1)+(r2+r3)) accumulation and stays within its tolerance.
	Label("reduce")
	VADDPS(acc[1], acc[0], acc[0])
	VADDPS(acc[3], acc[2], acc[2])
	VADDPS(acc[2], acc[0], acc[0])

	sum := horizontalSum(acc[0])
	VMOVSS(sum, Mem{Base: out})
	VZEROUPPER()
	RET()
}

// genComplexFIRDot4 emits complexFIRDot4AVX2, which computes FOUR complex FIR
// outputs that share the SAME taps in a single pass over the taps so each tap
// block is loaded ONCE (the transpose of foldHop4AVX2, where four windows share
// the loaded proto; here four windows share the loaded taps). Lane L's window
// starts stride float32 elements after lane L-1 in the SoA winR/winI buffers.
// Eight YMM accumulators (accR/accI per lane) keep four complex FMA pairs in
// flight so the loop is FMA-bound rather than latency-bound.
//
// Each lane's arithmetic is byte-identical to complexFIRDotAVX2 over that
// window's [0,n) prefix — same tap order, same single-accumulator-per-component
// reduction — only the tap load is hoisted and shared. The Go dispatch handles
// the n%8 scalar tail per lane. Results are written as
// out = {r0,i0, r1,i1, r2,i2, r3,i3}.
func genComplexFIRDot4() {
	TEXT("complexFIRDot4AVX2", NOSPLIT, "func(taps, winR, winI []float32, stride int, nv int, out *[8]float32)")
	Doc("complexFIRDot4AVX2 accumulates four shared-tap complex FIR prefixes (multiple of eight) in one tap pass; the Go dispatch handles the per-lane scalar tail.")
	Pragma("noescape")

	taps := Load(Param("taps").Base(), GP64())
	winR := Load(Param("winR").Base(), GP64())
	winI := Load(Param("winI").Base(), GP64())
	stride := Load(Param("stride"), GP64())
	nv := Load(Param("nv"), GP64())
	out := Load(Param("out"), GP64())

	// Lane byte stride: stride float32 elements = stride*4 bytes.
	strideB := GP64()
	MOVQ(stride, strideB)
	SHLQ(Imm(2), strideB)

	// Per-lane window base pointers: lane L starts L*strideB bytes into winR/winI.
	wr := [4]reg.Register{winR, GP64(), GP64(), GP64()}
	wi := [4]reg.Register{winI, GP64(), GP64(), GP64()}
	for l := 1; l < 4; l++ {
		LEAQ(Mem{Base: wr[l-1], Index: strideB, Scale: 1}, wr[l])
		LEAQ(Mem{Base: wi[l-1], Index: strideB, Scale: 1}, wi[l])
	}

	// One accR and one accI accumulator per lane (8 YMM total).
	accR := [4]reg.VecVirtual{YMM(), YMM(), YMM(), YMM()}
	accI := [4]reg.VecVirtual{YMM(), YMM(), YMM(), YMM()}
	for l := 0; l < 4; l++ {
		VXORPS(accR[l], accR[l], accR[l])
		VXORPS(accI[l], accI[l], accI[l])
	}

	i := GP64()
	XORQ(i, i)

	// Main loop: one block of 8 taps per iteration. The tap block is loaded once
	// and reused across all four lanes; each lane's winR/winI block is loaded and
	// fused into that lane's accumulators. nv is a multiple of 8.
	Label("loop8")
	CMPQ(i, nv)
	JGE(LabelRef("reduce"))
	t := YMM()
	VMOVUPS(Mem{Base: taps, Index: i, Scale: 4}, t)
	r := YMM()
	im := YMM()
	for l := 0; l < 4; l++ {
		VMOVUPS(Mem{Base: wr[l], Index: i, Scale: 4}, r)
		VMOVUPS(Mem{Base: wi[l], Index: i, Scale: 4}, im)
		VFMADD231PS(t, r, accR[l])
		VFMADD231PS(t, im, accI[l])
	}
	ADDQ(Imm(8), i)
	JMP(LabelRef("loop8"))

	// Reduce each lane's two accumulators to scalars and store in interleaved
	// {r,i} order. Each horizontalSum matches the single-window kernel's reduce.
	Label("reduce")
	for l := 0; l < 4; l++ {
		sumR := horizontalSum(accR[l])
		sumI := horizontalSum(accI[l])
		VMOVSS(sumR, Mem{Base: out, Disp: (2 * l) * 4})
		VMOVSS(sumI, Mem{Base: out, Disp: (2*l + 1) * 4})
	}
	VZEROUPPER()
	RET()
}

func horizontalSum(acc reg.VecVirtual) reg.VecVirtual {
	lo := XMM()
	hi := XMM()
	VEXTRACTF128(U8(0), acc, lo)
	VEXTRACTF128(U8(1), acc, hi)
	sum := XMM()
	VADDPS(hi, lo, sum)
	VHADDPS(sum, sum, sum)
	VHADDPS(sum, sum, sum)
	return sum
}
