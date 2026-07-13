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
	Generate()
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
