//go:build tools

// Package tools pins build-time-only dependencies so `go mod tidy` keeps them.
// avo is used by the asm/{fftc,fftc_batch,filter,foldhop} (//go:build ignore)
// generators, which tidy does not scan; this blank import keeps avo in
// go.mod/go.sum so `go generate ./...` can regenerate the committed kernels.
package tools

import (
	_ "github.com/mmcloughlin/avo/build"
	_ "github.com/mmcloughlin/avo/operand"
)
