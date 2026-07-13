//go:build ignore

// Command fftc_batch generates AVX2 four-transform Stockham FFT kernels.
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

	signMem := GLOBL("fftc_batch_jrot_sign", attr.RODATA|attr.NOPTR)
	for lane, v := range []uint32{0, 0x80000000, 0, 0x80000000, 0, 0x80000000, 0, 0x80000000} {
		DATA(lane*4, U32(v))
	}

	// Radix-3 constants are shared by the standalone and unpack-fused kernels,
	// so declare them once here and broadcast per kernel from these Mems.
	r3Half := declareScalar("fftc_batch_neg_half", 0xbf000000)
	r3Sin := declareScalar("fftc_batch_sqrt3_over_2", 0x3f5db3d7)

	genPack()
	genUnpack()
	genRadix2()
	genRadix3(signMem, r3Half, r3Sin)
	genRadix4(signMem)
	genRadix5(signMem)
	genRadix4Pack(signMem)
	genRadix3Unpack(signMem, r3Half, r3Sin)
	Generate()
}

// Fusion modes for the edge stages. In a PFB plan whose first radix is 4 and
// last is 3 (e.g. K=384 -> [4,4,4,2,3]), the standalone packBatch4/unpackBatch4
// AoSoA transposes each cost one L1 round-trip. The fused kernels fold the pack
// into stage-0's load (gather from the row-major src, interleaving exactly as
// packBatch4 does) and the unpack into the last stage's store (scatter to the
// four row-major dst rows, deinterleaving exactly as unpackBatch4 does). Every
// arithmetic op — the unity twiddle multiply at stage 0, the butterflies — is
// left byte-for-byte identical to the standalone kernels, so the fused output
// is bit-identical; only the memory endpoint moves.
const (
	modeNormal = iota
	modePack   // stage-0 radix-4: gather load replaces the AoSoA load
	modeUnpack // last-stage radix-3: scatter store replaces the AoSoA store
)

func genPack() {
	TEXT("packBatch4AVX2", NOSPLIT, "func(dst, src []complex64, n int)")
	dst := Load(Param("dst").Base(), GP64())
	src0 := Load(Param("src").Base(), GP64())
	n := Load(Param("n"), GP64())
	stride := GP64()
	MOVQ(n, stride)
	SHLQ(Imm(3), stride)
	src1, src2, src3 := GP64(), GP64(), GP64()
	LEAQ(Mem{Base: src0, Index: stride, Scale: 1}, src1)
	LEAQ(Mem{Base: src1, Index: stride, Scale: 1}, src2)
	LEAQ(Mem{Base: src2, Index: stride, Scale: 1}, src3)
	off := GP64()
	XORQ(off, off)
	Label("pack_loop")
	CMPQ(off, stride)
	JGE(LabelRef("pack_done"))
	x0, x1, x2, x3 := XMM(), XMM(), XMM(), XMM()
	VMOVQ(Mem{Base: src0, Index: off, Scale: 1}, x0)
	VMOVQ(Mem{Base: src1, Index: off, Scale: 1}, x1)
	VMOVQ(Mem{Base: src2, Index: off, Scale: 1}, x2)
	VMOVQ(Mem{Base: src3, Index: off, Scale: 1}, x3)
	lo, hi := XMM(), XMM()
	VPUNPCKLQDQ(x1, x0, lo)
	VPUNPCKLQDQ(x3, x2, hi)
	v := YMM()
	VINSERTI128(Imm(0), lo, v, v)
	VINSERTI128(Imm(1), hi, v, v)
	VMOVUPS(v, Mem{Base: dst, Index: off, Scale: 4})
	ADDQ(Imm(8), off)
	JMP(LabelRef("pack_loop"))
	Label("pack_done")
	VZEROUPPER()
	RET()
}

func genUnpack() {
	TEXT("unpackBatch4AVX2", NOSPLIT, "func(dst, src []complex64, n int)")
	dst0 := Load(Param("dst").Base(), GP64())
	src := Load(Param("src").Base(), GP64())
	n := Load(Param("n"), GP64())
	stride := GP64()
	MOVQ(n, stride)
	SHLQ(Imm(3), stride)
	dst1, dst2, dst3 := GP64(), GP64(), GP64()
	LEAQ(Mem{Base: dst0, Index: stride, Scale: 1}, dst1)
	LEAQ(Mem{Base: dst1, Index: stride, Scale: 1}, dst2)
	LEAQ(Mem{Base: dst2, Index: stride, Scale: 1}, dst3)
	off := GP64()
	XORQ(off, off)
	Label("unpack_loop")
	CMPQ(off, stride)
	JGE(LabelRef("unpack_done"))
	v := YMM()
	VMOVUPS(Mem{Base: src, Index: off, Scale: 4}, v)
	lo, hi := XMM(), XMM()
	VEXTRACTI128(Imm(0), v, lo)
	VEXTRACTI128(Imm(1), v, hi)
	VMOVQ(lo, Mem{Base: dst0, Index: off, Scale: 1})
	tmp := XMM()
	VPSRLDQ(Imm(8), lo, tmp)
	VMOVQ(tmp, Mem{Base: dst1, Index: off, Scale: 1})
	VMOVQ(hi, Mem{Base: dst2, Index: off, Scale: 1})
	VPSRLDQ(Imm(8), hi, tmp)
	VMOVQ(tmp, Mem{Base: dst3, Index: off, Scale: 1})
	ADDQ(Imm(8), off)
	JMP(LabelRef("unpack_loop"))
	Label("unpack_done")
	VZEROUPPER()
	RET()
}

// stage holds the running pointers and precomputed byte-strides for one
// Stockham kernel. The old code recomputed every load/store/twiddle address
// from scratch per butterfly (section*butterflies + q*step, then a MOVQ/SHLQ/
// ADDQ to scale to bytes) which made the inner loop integer-ALU bound. Here the
// leg strides are loop invariants precomputed once at kernel entry, and the
// q=0 base of each array is walked with a running byte-pointer advanced by a
// constant at the bottom of the p loop. No per-butterfly IMULQ remains.
type stage struct {
	// Running byte-pointers to the q=0 element of the current butterfly.
	srcPtr, dstPtr, twPtr reg.GPVirtual
	twBase                reg.GPVirtual // original tw base, for per-section reset
	// Loop control.
	butterflies, sections, section, p reg.GPVirtual
	// Loop-invariant byte-strides (see the q-addressing helpers below).
	inputStepBytes, bBytes32, bBytes8      reg.GPVirtual
	loadLeg3, storeLeg3, twLeg3, dstSecAdj reg.GPVirtual

	// Fusion state (modePack / modeUnpack). laneBase[L] is the running
	// row-major byte-pointer to the current (q=0) element of lane L's transform,
	// advanced +8 per outer iteration (per section on the pack, per butterfly on
	// the unpack). A leg q sits at laneBase[L] + q*legBytesRM (legBytesRM =
	// inputStep*8 on the pack load, butterflies*8 on the unpack store); legRM3 =
	// 3*legBytesRM handles the q=3 leg the SIB scale can't express.
	mode                    int
	radix                   int
	laneBase                [4]reg.GPVirtual
	legBytesRM, legRM3      reg.GPVirtual
}

func beginStage(name string, radix uint64) stage {
	return beginStageMode(name, radix, modeNormal)
}

func beginStageMode(name string, radix uint64, mode int) stage {
	TEXT(name, NOSPLIT, "func(dst, src, tw []complex64, butterflies, sections, n int)")
	dst, src, tw := GP64(), GP64(), GP64()
	butterflies, sections, n := GP64(), GP64(), GP64()
	Load(Param("dst").Base(), dst)
	Load(Param("src").Base(), src)
	Load(Param("tw").Base(), tw)
	Load(Param("butterflies"), butterflies)
	Load(Param("sections"), sections)
	Load(Param("n"), n)

	// inputStep = n / radix (complex64 elements).
	inputStep := GP64()
	MOVQ(n, reg.RAX)
	XORQ(reg.RDX, reg.RDX)
	divisor := GP64()
	MOVQ(U64(radix), divisor)
	DIVQ(divisor)
	MOVQ(reg.RAX, inputStep)

	r := int(radix)

	// Precompute the loop-invariant byte-strides. A logical (32-byte) element
	// spans 4 batched complex64; the src/dst legs stride by 32 bytes per
	// logical index, the twiddle by 8 bytes per complex64 index.
	//
	// The fused edge kernels leave one side of the AoSoA transpose in the work
	// buffer and the other side row-major, and each collapses a degenerate loop
	// (see start/end): modePack is stage 0 (butterflies==1, no p loop; reads
	// row-major so the AoSoA load strides are dead; its AoSoA store strides are
	// compile-time constants used as displacements, so no register). modeUnpack
	// is the last stage (sections==1, no section loop; writes row-major so the
	// AoSoA store strides and dstSecAdj are dead). Only allocating what each
	// mode actually uses keeps GP-register pressure within the 16-GPR budget.
	var inputStepBytes, loadLeg3 reg.GPVirtual
	if mode != modePack { // AoSoA load side (modeNormal + modeUnpack)
		inputStepBytes = GP64()
		MOVQ(inputStep, inputStepBytes)
		SHLQ(Imm(5), inputStepBytes) // inputStep * 32 (load leg q=1)
		if r >= 4 {
			// q=3 legs need a *3 stride, which the x86 SIB scale cannot express.
			loadLeg3 = GP64()
			LEAQ(Mem{Base: inputStepBytes, Index: inputStepBytes, Scale: 2}, loadLeg3) // 3 * inputStepBytes
		}
	}

	var bBytes32, storeLeg3, dstSecAdj reg.GPVirtual
	if mode == modeNormal { // AoSoA store side; modePack uses displacements, modeUnpack scatters
		bBytes32 = GP64()
		MOVQ(butterflies, bBytes32)
		SHLQ(Imm(5), bBytes32) // butterflies * 32 (store leg q=1)
		if r >= 4 {
			storeLeg3 = GP64()
			LEAQ(Mem{Base: bBytes32, Index: bBytes32, Scale: 2}, storeLeg3) // 3 * bBytes32
		}
		// dstSecAdj = (radix-1)*bBytes32. The p loop advances dstPtr by
		// butterflies*32, but the dst section stride is butterflies*radix*32, so
		// close the gap once per section. For radix 4 this equals storeLeg3.
		switch r {
		case 2:
			dstSecAdj = bBytes32
		case 3:
			dstSecAdj = GP64()
			MOVQ(bBytes32, dstSecAdj)
			SHLQ(Imm(1), dstSecAdj) // 2 * bBytes32
		case 4:
			dstSecAdj = storeLeg3
		default: // radix 5
			dstSecAdj = GP64()
			MOVQ(bBytes32, dstSecAdj)
			SHLQ(Imm(2), dstSecAdj) // 4 * bBytes32
		}
	}

	var bBytes8, twLeg3 reg.GPVirtual
	if r >= 3 && mode != modePack { // twiddle stride; modePack (butterflies==1) uses displacements
		bBytes8 = GP64()
		MOVQ(butterflies, bBytes8)
		SHLQ(Imm(3), bBytes8) // butterflies * 8 (twiddle leg (q-1)=1)
	}
	if r >= 5 {
		twLeg3 = GP64()
		LEAQ(Mem{Base: bBytes8, Index: bBytes8, Scale: 2}, twLeg3) // 3 * bBytes8 (twiddle leg (q-1)=3)
	}

	// The q=0 base pointers start at the head of each array. src walks with a
	// stride equal to its section stride, so it needs no per-section reset;
	// dst gets dstSecAdj added per section; tw is section-independent so it is
	// reset to twBase each section.
	twPtr := GP64()
	MOVQ(tw, twPtr)

	s := stage{
		srcPtr: src, dstPtr: dst, twPtr: twPtr, twBase: tw,
		butterflies: butterflies, sections: sections,
		section: GP64(), p: GP64(),
		inputStepBytes: inputStepBytes, bBytes32: bBytes32, bBytes8: bBytes8,
		loadLeg3: loadLeg3, storeLeg3: storeLeg3, twLeg3: twLeg3, dstSecAdj: dstSecAdj,
		mode:   mode, radix: r,
	}

	if mode != modeNormal {
		// Set up the row-major side. The AoSoA side (dst on pack, src on unpack)
		// still uses the normal running pointers above. laneBase[L] = base +
		// L*n*8 points at lane L's row 0; legBytesRM is the intra-row byte stride
		// between successive q legs, legRM3 = 3*legBytesRM for the q=3 leg.
		nBytes := GP64()
		MOVQ(n, nBytes)
		SHLQ(Imm(3), nBytes) // n * 8 (one row-major transform, complex64)
		var rmBase reg.GPVirtual
		if mode == modePack {
			rmBase = src // gather source rows
		} else {
			rmBase = dst // scatter destination rows
		}
		s.laneBase[0] = GP64()
		MOVQ(rmBase, s.laneBase[0])
		for L := 1; L < 4; L++ {
			s.laneBase[L] = GP64()
			LEAQ(Mem{Base: s.laneBase[L-1], Index: nBytes, Scale: 1}, s.laneBase[L])
		}
		if mode == modePack {
			// Stage 0 has butterflies==1, so the pack load leg stride is
			// inputStep*8 = sections*8. Reuse sections (its loop counter is still
			// live) scaled to bytes; the q=3 leg needs 3x that.
			s.legBytesRM = GP64()
			MOVQ(inputStep, s.legBytesRM)
			SHLQ(Imm(3), s.legBytesRM)
			if r >= 4 {
				s.legRM3 = GP64()
				LEAQ(Mem{Base: s.legBytesRM, Index: s.legBytesRM, Scale: 2}, s.legRM3)
			}
		} else {
			// Last stage has sections==1; the unpack store leg stride is
			// butterflies*8, which is exactly bBytes8 (already computed). radix 3
			// so no q=3 leg.
			s.legBytesRM = bBytes8
		}
	}

	return s
}

// rmMem is the row-major address of leg q for lane L in a fused kernel:
// laneBase[L] + q*legBytesRM.
func (s stage) rmMem(L, q int) Mem {
	switch q {
	case 0:
		return Mem{Base: s.laneBase[L]}
	case 1:
		return Mem{Base: s.laneBase[L], Index: s.legBytesRM, Scale: 1}
	case 2:
		return Mem{Base: s.laneBase[L], Index: s.legBytesRM, Scale: 2}
	case 3:
		return Mem{Base: s.laneBase[L], Index: s.legRM3, Scale: 1}
	}
	panic("bad row-major leg")
}

func (s stage) start(name string) {
	switch s.mode {
	case modePack:
		// Stage 0: butterflies==1, so there is no p loop — one butterfly per
		// section. Iterate sections only.
		XORQ(s.section, s.section)
		Label(name + "_section")
		CMPQ(s.section, s.sections)
		JGE(LabelRef(name + "_done"))
	case modeUnpack:
		// Last stage: sections==1, so there is no section loop. Iterate p only.
		XORQ(s.p, s.p)
		Label(name + "_p")
		CMPQ(s.p, s.butterflies)
		JGE(LabelRef(name + "_done"))
	default:
		XORQ(s.section, s.section)
		Label(name + "_section")
		CMPQ(s.section, s.sections)
		JGE(LabelRef(name + "_done"))
		XORQ(s.p, s.p)
		Label(name + "_p")
		CMPQ(s.p, s.butterflies)
		JGE(LabelRef(name + "_next_section"))
	}
}

// srcMem is the load address for leg q: srcPtr + q*inputStepBytes.
func (s stage) srcMem(q int) Mem {
	switch q {
	case 0:
		return Mem{Base: s.srcPtr}
	case 1:
		return Mem{Base: s.srcPtr, Index: s.inputStepBytes, Scale: 1}
	case 2:
		return Mem{Base: s.srcPtr, Index: s.inputStepBytes, Scale: 2}
	case 3:
		return Mem{Base: s.srcPtr, Index: s.loadLeg3, Scale: 1}
	case 4:
		return Mem{Base: s.srcPtr, Index: s.inputStepBytes, Scale: 4}
	}
	panic("bad load leg")
}

// dstMem is the store address for leg q: dstPtr + q*bBytes32. In modePack
// (butterflies==1) bBytes32 is the constant 32, used as a displacement so no
// register is spent on the AoSoA store strides.
func (s stage) dstMem(q int) Mem {
	if s.mode == modePack {
		return Mem{Base: s.dstPtr, Disp: q * 32}
	}
	switch q {
	case 0:
		return Mem{Base: s.dstPtr}
	case 1:
		return Mem{Base: s.dstPtr, Index: s.bBytes32, Scale: 1}
	case 2:
		return Mem{Base: s.dstPtr, Index: s.bBytes32, Scale: 2}
	case 3:
		return Mem{Base: s.dstPtr, Index: s.storeLeg3, Scale: 1}
	case 4:
		return Mem{Base: s.dstPtr, Index: s.bBytes32, Scale: 4}
	}
	panic("bad store leg")
}

// twMem is the twiddle address for load leg q (q>=1): twPtr + (q-1)*bBytes8. In
// modePack (butterflies==1) bBytes8 is the constant 8 and twPtr never advances,
// so it is a plain displacement off twPtr.
func (s stage) twMem(q int) Mem {
	if s.mode == modePack {
		return Mem{Base: s.twPtr, Disp: (q - 1) * 8}
	}
	switch q {
	case 1:
		return Mem{Base: s.twPtr}
	case 2:
		return Mem{Base: s.twPtr, Index: s.bBytes8, Scale: 1}
	case 3:
		return Mem{Base: s.twPtr, Index: s.bBytes8, Scale: 2}
	case 4:
		return Mem{Base: s.twPtr, Index: s.twLeg3, Scale: 1}
	}
	panic("bad twiddle leg")
}

func (s stage) load(q, radix int) reg.VecVirtual {
	v := YMM()
	if s.mode == modePack {
		// Fused stage-0 load: gather leg q from the four row-major source rows
		// and interleave into an AoSoA YMM, exactly as packBatch4AVX2 does
		// (VMOVQ each lane, VPUNPCKLQDQ the pairs, VINSERTI128 the halves). The
		// resulting register is bit-identical to VMOVUPS from the AoSoA buffer.
		x0, x1, x2, x3 := XMM(), XMM(), XMM(), XMM()
		VMOVQ(s.rmMem(0, q), x0)
		VMOVQ(s.rmMem(1, q), x1)
		VMOVQ(s.rmMem(2, q), x2)
		VMOVQ(s.rmMem(3, q), x3)
		lo, hi := XMM(), XMM()
		VPUNPCKLQDQ(x1, x0, lo)
		VPUNPCKLQDQ(x3, x2, hi)
		VINSERTI128(Imm(0), lo, v, v)
		VINSERTI128(Imm(1), hi, v, v)
	} else {
		VMOVUPS(s.srcMem(q), v)
	}
	if q > 0 {
		w := YMM()
		VBROADCASTSD(s.twMem(q), w)
		v = complexMul(v, w)
	}
	return v
}

func (s stage) store(q, radix int, v reg.VecVirtual) {
	if s.mode == modeUnpack {
		// Fused last-stage store: scatter leg q of the AoSoA result YMM out to
		// the four row-major destination rows, exactly as unpackBatch4AVX2 does
		// (VEXTRACTI128 the halves, VPSRLDQ to reach the odd lanes, VMOVQ each).
		lo, hi := XMM(), XMM()
		VEXTRACTI128(Imm(0), v, lo)
		VEXTRACTI128(Imm(1), v, hi)
		VMOVQ(lo, s.rmMem(0, q))
		tmp := XMM()
		VPSRLDQ(Imm(8), lo, tmp)
		VMOVQ(tmp, s.rmMem(1, q))
		VMOVQ(hi, s.rmMem(2, q))
		VPSRLDQ(Imm(8), hi, tmp)
		VMOVQ(tmp, s.rmMem(3, q))
		return
	}
	VMOVUPS(v, s.dstMem(q))
}

func (s stage) end(name string) {
	switch s.mode {
	case modePack:
		// Stage 0 (butterflies==1): one section step per iteration. The AoSoA
		// output section stride is butterflies*radix*32 = radix*32 (a constant,
		// since butterflies==1); the gather index tracks section, so advance each
		// lane's row-major pointer +8 (one complex64). srcPtr/twPtr are dead here
		// (loads gather via laneBase; twiddles are constant displacements off the
		// never-advanced twPtr).
		ADDQ(Imm(uint64(s.radix)*32), s.dstPtr)
		for L := 0; L < 4; L++ {
			ADDQ(Imm(8), s.laneBase[L])
		}
		INCQ(s.section)
		JMP(LabelRef(name + "_section"))
	case modeUnpack:
		// Last stage (sections==1): one butterfly step per iteration. srcPtr and
		// twPtr advance as usual; dstPtr is dead (stores scatter via laneBase).
		// The scatter index tracks p, so advance each lane's row-major pointer +8.
		INCQ(s.p)
		ADDQ(Imm(32), s.srcPtr)
		ADDQ(Imm(8), s.twPtr)
		for L := 0; L < 4; L++ {
			ADDQ(Imm(8), s.laneBase[L])
		}
		JMP(LabelRef(name + "_p"))
	default:
		INCQ(s.p)
		// Advance every q=0 base pointer by one butterfly: +32 bytes per logical
		// element for src/dst, +8 bytes per complex64 for the twiddle.
		ADDQ(Imm(32), s.srcPtr)
		ADDQ(Imm(32), s.dstPtr)
		ADDQ(Imm(8), s.twPtr)
		JMP(LabelRef(name + "_p"))
		Label(name + "_next_section")
		ADDQ(s.dstSecAdj, s.dstPtr)
		MOVQ(s.twBase, s.twPtr)
		INCQ(s.section)
		JMP(LabelRef(name + "_section"))
	}
	Label(name + "_done")
	VZEROUPPER()
	RET()
}

func genRadix2() {
	const name = "stockhamRadix2AVX2"
	s := beginStage(name, 2)
	s.start(name)
	x0, x1 := s.load(0, 2), s.load(1, 2)
	y0, y1 := YMM(), YMM()
	VADDPS(x1, x0, y0)
	VSUBPS(x1, x0, y1)
	s.store(0, 2, y0)
	s.store(1, 2, y1)
	s.end(name)
}

func genRadix3(signMem, r3Half, r3Sin Mem) {
	const name = "stockhamRadix3AVX2"
	s := beginStage(name, 3)
	radix3Body(name, s, signMem, r3Half, r3Sin)
}

// genRadix3Unpack is the last-stage radix-3 kernel with the AoSoA-to-row-major
// unpack folded into its store (see modeUnpack). Its arithmetic is emitted by
// the same radix3Body as the standalone kernel, so bin output is bit-identical.
func genRadix3Unpack(signMem, r3Half, r3Sin Mem) {
	const name = "stockhamRadix3UnpackAVX2"
	s := beginStageMode(name, 3, modeUnpack)
	radix3Body(name, s, signMem, r3Half, r3Sin)
}

func radix3Body(name string, s stage, signMem, r3Half, r3Sin Mem) {
	sign := YMM()
	VMOVUPS(signMem, sign)
	half := broadcastMem(r3Half)
	sin := broadcastMem(r3Sin)
	s.start(name)
	x0, x1, x2 := s.load(0, 3), s.load(1, 3), s.load(2, 3)
	sum, dif := YMM(), YMM()
	VADDPS(x2, x1, sum)
	VSUBPS(x2, x1, dif)
	y0 := YMM()
	VADDPS(sum, x0, y0)
	hs, a := YMM(), YMM()
	VMULPS(half, sum, hs)
	VADDPS(hs, x0, a)
	w := YMM()
	VMULPS(sin, dif, w)
	rot := jRotNeg(sign, w)
	y1, y2 := YMM(), YMM()
	VADDPS(rot, a, y1)
	VSUBPS(rot, a, y2)
	s.store(0, 3, y0)
	s.store(1, 3, y1)
	s.store(2, 3, y2)
	s.end(name)
}

func genRadix4(signMem Mem) {
	const name = "stockhamRadix4AVX2"
	s := beginStage(name, 4)
	radix4Body(name, s, signMem)
}

// genRadix4Pack is the stage-0 radix-4 kernel with the row-major-to-AoSoA pack
// folded into its load (see modePack). Its arithmetic is emitted by the same
// radix4Body as the standalone kernel, so bin output is bit-identical.
func genRadix4Pack(signMem Mem) {
	const name = "stockhamRadix4PackAVX2"
	s := beginStageMode(name, 4, modePack)
	radix4Body(name, s, signMem)
}

func radix4Body(name string, s stage, signMem Mem) {
	sign := YMM()
	VMOVUPS(signMem, sign)
	s.start(name)
	x0, x1, x2, x3 := s.load(0, 4), s.load(1, 4), s.load(2, 4), s.load(3, 4)
	d0, a0, a1, d1 := YMM(), YMM(), YMM(), YMM()
	VSUBPS(x2, x0, d0)
	VADDPS(x2, x0, a0)
	VADDPS(x3, x1, a1)
	VSUBPS(x3, x1, d1)
	y0, y2 := YMM(), YMM()
	VADDPS(a1, a0, y0)
	VSUBPS(a1, a0, y2)
	rot := jRotNeg(sign, d1)
	y1, y3 := YMM(), YMM()
	VADDPS(rot, d0, y1)
	VSUBPS(rot, d0, y3)
	s.store(0, 4, y0)
	s.store(1, 4, y1)
	s.store(2, 4, y2)
	s.store(3, 4, y3)
	s.end(name)
}

func genRadix5(signMem Mem) {
	const name = "stockhamRadix5AVX2"
	s := beginStage(name, 5)
	sign := YMM()
	VMOVUPS(signMem, sign)
	c1 := broadcast("fftc_batch_radix5_c1", 0x3e9e377a)
	c2 := broadcast("fftc_batch_radix5_c2", 0xbf4f1bbd)
	s1 := broadcast("fftc_batch_radix5_s1", 0x3f737871)
	s2 := broadcast("fftc_batch_radix5_s2", 0x3f167918)
	s.start(name)
	x0, x1, x2, x3, x4 := s.load(0, 5), s.load(1, 5), s.load(2, 5), s.load(3, 5), s.load(4, 5)
	a1, d1, a2, d2 := YMM(), YMM(), YMM(), YMM()
	VADDPS(x4, x1, a1)
	VSUBPS(x4, x1, d1)
	VADDPS(x3, x2, a2)
	VSUBPS(x3, x2, d2)
	t := YMM()
	VADDPS(a1, x0, t)
	y0 := YMM()
	VADDPS(a2, t, y0)
	m1a, m1b, m1 := YMM(), YMM(), YMM()
	VMULPS(c1, a1, m1a)
	VMULPS(c2, a2, m1b)
	VADDPS(m1a, x0, m1)
	VADDPS(m1b, m1, m1)
	m2a, m2b, m2 := YMM(), YMM(), YMM()
	VMULPS(c2, a1, m2a)
	VMULPS(c1, a2, m2b)
	VADDPS(m2a, x0, m2)
	VADDPS(m2b, m2, m2)
	r1a, r1b, r1 := YMM(), YMM(), YMM()
	VMULPS(s1, d1, r1a)
	VMULPS(s2, d2, r1b)
	VADDPS(r1b, r1a, r1)
	r2a, r2b, r2 := YMM(), YMM(), YMM()
	VMULPS(s2, d1, r2a)
	VMULPS(s1, d2, r2b)
	VSUBPS(r2b, r2a, r2)
	rot1, rot2 := jRotNeg(sign, r1), jRotNeg(sign, r2)
	y1, y4, y2, y3 := YMM(), YMM(), YMM(), YMM()
	VADDPS(rot1, m1, y1)
	VSUBPS(rot1, m1, y4)
	VADDPS(rot2, m2, y2)
	VSUBPS(rot2, m2, y3)
	s.store(0, 5, y0)
	s.store(1, 5, y1)
	s.store(2, 5, y2)
	s.store(3, 5, y3)
	s.store(4, 5, y4)
	s.end(name)
}

// declareScalar defines a single-float RODATA global once. Kernels that reuse
// the same constant (e.g. radix-3 shared by the standalone and unpack kernels)
// must broadcast from the returned Mem rather than re-declaring the symbol.
func declareScalar(name string, bits uint32) Mem {
	m := GLOBL(name, attr.RODATA|attr.NOPTR)
	DATA(0, U32(bits))
	return m
}

func broadcast(name string, bits uint32) reg.VecVirtual {
	return broadcastMem(declareScalar(name, bits))
}

func broadcastMem(m Mem) reg.VecVirtual {
	v := YMM()
	VBROADCASTSS(m, v)
	return v
}

func jRotNeg(sign, v reg.VecVirtual) reg.VecVirtual {
	rot := YMM()
	VPERMILPS(Imm(0xB1), v, rot)
	VXORPS(sign, rot, rot)
	return rot
}

func complexMul(a, b reg.VecVirtual) reg.VecVirtual {
	br, bi := YMM(), YMM()
	VSHUFPS(Imm(0xA0), b, b, br)
	VSHUFPS(Imm(0xF5), b, b, bi)
	sw := YMM()
	VPERMILPS(Imm(0xB1), a, sw)
	out := YMM()
	VMULPS(bi, sw, out)
	VFMADDSUB231PS(a, br, out)
	return out
}
