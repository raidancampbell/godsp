//go:build ignore

// Command foldhop generates the AVX2 kernel for PFB.foldHop.
// Run via: go generate ./...
package main

import (
	"github.com/mmcloughlin/avo/attr"
	. "github.com/mmcloughlin/avo/build"
	. "github.com/mmcloughlin/avo/operand"
)

func main() {
	Package("github.com/raidancampbell/godsp")
	ConstraintExpr("amd64")

	// Static VPERMPS index vectors. Each duplicates four float32 proto reals into
	// the re/im lanes of four complex64 win samples:
	//   d0 = h[{0,0,1,1,2,2,3,3}]  (pairs with win lanes 0..3)
	//   d1 = h[{4,4,5,5,6,6,7,7}]  (pairs with win lanes 4..7)
	idxLoMem := GLOBL("foldhop_idx_lo", attr.RODATA|attr.NOPTR)
	for lane, v := range []uint32{0, 0, 1, 1, 2, 2, 3, 3} {
		DATA(lane*4, U32(v))
	}
	idxHiMem := GLOBL("foldhop_idx_hi", attr.RODATA|attr.NOPTR)
	for lane, v := range []uint32{4, 4, 5, 5, 6, 6, 7, 7} {
		DATA(lane*4, U32(v))
	}

	// foldHopAVX2 computes fold[i] = Σ_q win[q*k+i]*proto[q*k+i] for i in [0,kv),
	// kv a multiple of 8 bins. Vectorizes across bins: 8 bins = 16 float32 = 2
	// YMM, accumulated over the p taps; proto reals are duplicated into re/im
	// lanes. win/fold are complex64 (interleaved re,im); proto is float32.
	TEXT("foldHopAVX2", NOSPLIT, "func(win []complex64, proto []float32, fold []complex64, kv int, k int, p int)")
	Doc("foldHopAVX2 is the AVX2 8-bin-block polyphase fold; the Go dispatch handles the scalar tail.")

	winPtr := Load(Param("win").Base(), GP64())
	protoPtr := Load(Param("proto").Base(), GP64())
	foldPtr := Load(Param("fold").Base(), GP64())
	kv := Load(Param("kv"), GP64())
	k := Load(Param("k"), GP64())
	p := Load(Param("p"), GP64())

	// Byte strides between taps.
	kBytesC := GP64() // k complex64 = k*8 bytes (tap stride in win/fold)
	MOVQ(k, kBytesC)
	SHLQ(Imm(3), kBytesC)
	kBytesF := GP64() // k float32 = k*4 bytes (tap stride in proto)
	MOVQ(k, kBytesF)
	SHLQ(Imm(2), kBytesF)

	// Load the permute index vectors once.
	idxLo := YMM()
	idxHi := YMM()
	VMOVUPS(idxLoMem, idxLo)
	VMOVUPS(idxHiMem, idxHi)

	i := GP64() // bin index, steps by 8
	XORQ(i, i)

	Label("binloop")
	CMPQ(i, kv)
	JGE(LabelRef("done"))

	// Accumulators for 8 bins (16 floats) = 2 YMM, zeroed.
	acc0 := YMM()
	acc1 := YMM()
	VXORPS(acc0, acc0, acc0)
	VXORPS(acc1, acc1, acc1)

	// win element pointer for this bin block: winPtr + i*8 bytes.
	wp := GP64()
	MOVQ(i, wp)
	SHLQ(Imm(3), wp)
	ADDQ(winPtr, wp)
	// proto element pointer for this bin block: protoPtr + i*4 bytes.
	pp := GP64()
	MOVQ(i, pp)
	SHLQ(Imm(2), pp)
	ADDQ(protoPtr, pp)

	q := GP64()
	XORQ(q, q)
	Label("taploop")
	CMPQ(q, p)
	JGE(LabelRef("tapdone"))

	// Load 8 complex64 (16 floats) from win at wp -> 2 YMM.
	w0 := YMM()
	w1 := YMM()
	VMOVUPS(Mem{Base: wp}, w0)
	VMOVUPS(Mem{Base: wp, Disp: 32}, w1)

	// Load 8 proto floats from pp into one YMM, then duplicate each into its
	// re/im pair across two YMM via VPERMPS (Go order: src, index, dst).
	h := YMM()
	VMOVUPS(Mem{Base: pp}, h)
	d0 := YMM()
	d1 := YMM()
	VPERMPS(h, idxLo, d0)
	VPERMPS(h, idxHi, d1)

	// acc += w * d  (fused; acc += d*w).
	VFMADD231PS(w0, d0, acc0)
	VFMADD231PS(w1, d1, acc1)

	ADDQ(kBytesC, wp)
	ADDQ(kBytesF, pp)
	INCQ(q)
	JMP(LabelRef("taploop"))

	Label("tapdone")
	// Store 8 results (16 floats) to fold at foldPtr + i*8.
	fp := GP64()
	MOVQ(i, fp)
	SHLQ(Imm(3), fp)
	ADDQ(foldPtr, fp)
	VMOVUPS(acc0, Mem{Base: fp})
	VMOVUPS(acc1, Mem{Base: fp, Disp: 32})

	ADDQ(Imm(8), i)
	JMP(LabelRef("binloop"))

	Label("done")
	VZEROUPPER()
	RET()

	genFoldHop4(idxLoMem, idxHiMem)

	Generate()
}

// genFoldHop4 emits foldHop4AVX2, which folds FOUR overlapping hop-windows in a
// single pass over the prototype so each proto tap is loaded and permuted ONCE
// (the port-5 VPERMPS bottleneck) and shared across all four lanes via eight YMM
// accumulators (two per lane). Lane L's window base is win + L·r complex64 and
// its output row is fold + L·k complex64; the four output rows are contiguous.
//
// The per-lane arithmetic is byte-identical to foldHopAVX2 for that window: same
// tap order, same acc0/acc1 split across the 8-float halves, same FMA operands —
// only the proto load+permute is hoisted out of the lane loop. The Go dispatch
// handles the k%8 scalar tail per lane exactly as for the single-window kernel.
func genFoldHop4(idxLoMem, idxHiMem Mem) {
	TEXT("foldHop4AVX2", NOSPLIT, "func(win []complex64, proto []float32, fold []complex64, kv int, k int, p int, r int)")
	Doc("foldHop4AVX2 folds four overlapping hop-windows in one proto pass (8-bin blocks); the Go dispatch handles the scalar tail.")

	winPtr := Load(Param("win").Base(), GP64())
	protoPtr := Load(Param("proto").Base(), GP64())
	foldPtr := Load(Param("fold").Base(), GP64())
	kv := Load(Param("kv"), GP64())
	k := Load(Param("k"), GP64())
	p := Load(Param("p"), GP64())
	r := Load(Param("r"), GP64())

	// Byte strides.
	kBytesC := GP64() // k complex64 = k*8: tap stride in win, lane stride in fold
	MOVQ(k, kBytesC)
	SHLQ(Imm(3), kBytesC)
	kBytesF := GP64() // k float32 = k*4: tap stride in proto
	MOVQ(k, kBytesF)
	SHLQ(Imm(2), kBytesF)
	rBytesC := GP64() // r complex64 = r*8: lane stride in win
	MOVQ(r, rBytesC)
	SHLQ(Imm(3), rBytesC)

	idxLo := YMM()
	idxHi := YMM()
	VMOVUPS(idxLoMem, idxLo)
	VMOVUPS(idxHiMem, idxHi)

	i := GP64() // bin index, steps by 8
	XORQ(i, i)

	Label("binloop4")
	CMPQ(i, kv)
	JGE(LabelRef("done4"))

	// Two YMM accumulators per lane (4 lanes) covering 8 bins = 16 floats each.
	acc00, acc01 := YMM(), YMM()
	acc10, acc11 := YMM(), YMM()
	acc20, acc21 := YMM(), YMM()
	acc30, acc31 := YMM(), YMM()
	VXORPS(acc00, acc00, acc00)
	VXORPS(acc01, acc01, acc01)
	VXORPS(acc10, acc10, acc10)
	VXORPS(acc11, acc11, acc11)
	VXORPS(acc20, acc20, acc20)
	VXORPS(acc21, acc21, acc21)
	VXORPS(acc30, acc30, acc30)
	VXORPS(acc31, acc31, acc31)

	// Per-lane win tap-pointers at this bin block: winPtr + i*8 + L*rBytesC.
	wp0 := GP64()
	MOVQ(i, wp0)
	SHLQ(Imm(3), wp0)
	ADDQ(winPtr, wp0)
	wp1 := GP64()
	LEAQ(Mem{Base: wp0, Index: rBytesC, Scale: 1}, wp1)
	wp2 := GP64()
	LEAQ(Mem{Base: wp1, Index: rBytesC, Scale: 1}, wp2)
	wp3 := GP64()
	LEAQ(Mem{Base: wp2, Index: rBytesC, Scale: 1}, wp3)

	// proto tap-pointer for this bin block: protoPtr + i*4.
	pp := GP64()
	MOVQ(i, pp)
	SHLQ(Imm(2), pp)
	ADDQ(protoPtr, pp)

	// w0/w1 are reloaded per lane but reuse two registers; d0/d1 are the shared
	// once-per-tap permuted proto.
	w0 := YMM()
	w1 := YMM()
	d0 := YMM()
	d1 := YMM()

	q := GP64()
	XORQ(q, q)
	Label("taploop4")
	CMPQ(q, p)
	JGE(LabelRef("tapdone4"))

	// Load 8 proto floats once and duplicate each real into its re/im pair; this
	// VPERMPS pair is the work shared across all four lanes.
	h := YMM()
	VMOVUPS(Mem{Base: pp}, h)
	VPERMPS(h, idxLo, d0)
	VPERMPS(h, idxHi, d1)

	// Lane 0.
	VMOVUPS(Mem{Base: wp0}, w0)
	VMOVUPS(Mem{Base: wp0, Disp: 32}, w1)
	VFMADD231PS(w0, d0, acc00)
	VFMADD231PS(w1, d1, acc01)
	// Lane 1.
	VMOVUPS(Mem{Base: wp1}, w0)
	VMOVUPS(Mem{Base: wp1, Disp: 32}, w1)
	VFMADD231PS(w0, d0, acc10)
	VFMADD231PS(w1, d1, acc11)
	// Lane 2.
	VMOVUPS(Mem{Base: wp2}, w0)
	VMOVUPS(Mem{Base: wp2, Disp: 32}, w1)
	VFMADD231PS(w0, d0, acc20)
	VFMADD231PS(w1, d1, acc21)
	// Lane 3.
	VMOVUPS(Mem{Base: wp3}, w0)
	VMOVUPS(Mem{Base: wp3, Disp: 32}, w1)
	VFMADD231PS(w0, d0, acc30)
	VFMADD231PS(w1, d1, acc31)

	ADDQ(kBytesC, wp0)
	ADDQ(kBytesC, wp1)
	ADDQ(kBytesC, wp2)
	ADDQ(kBytesC, wp3)
	ADDQ(kBytesF, pp)
	INCQ(q)
	JMP(LabelRef("taploop4"))

	Label("tapdone4")
	// Store each lane to fold + L*kBytesC at bin offset i*8.
	fp0 := GP64()
	MOVQ(i, fp0)
	SHLQ(Imm(3), fp0)
	ADDQ(foldPtr, fp0)
	VMOVUPS(acc00, Mem{Base: fp0})
	VMOVUPS(acc01, Mem{Base: fp0, Disp: 32})
	fp1 := GP64()
	LEAQ(Mem{Base: fp0, Index: kBytesC, Scale: 1}, fp1)
	VMOVUPS(acc10, Mem{Base: fp1})
	VMOVUPS(acc11, Mem{Base: fp1, Disp: 32})
	fp2 := GP64()
	LEAQ(Mem{Base: fp1, Index: kBytesC, Scale: 1}, fp2)
	VMOVUPS(acc20, Mem{Base: fp2})
	VMOVUPS(acc21, Mem{Base: fp2, Disp: 32})
	fp3 := GP64()
	LEAQ(Mem{Base: fp2, Index: kBytesC, Scale: 1}, fp3)
	VMOVUPS(acc30, Mem{Base: fp3})
	VMOVUPS(acc31, Mem{Base: fp3, Disp: 32})

	ADDQ(Imm(8), i)
	JMP(LabelRef("binloop4"))

	Label("done4")
	VZEROUPPER()
	RET()
}
