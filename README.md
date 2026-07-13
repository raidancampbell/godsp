# godsp

Digital-signal-processing primitives in Go, with no runtime dependencies.

```
import "github.com/raidancampbell/godsp"
```

The package name is `dsp`.

## What's here

- FIR and IIR (biquad) filtering
- FM demodulation
- FFTs and complex FFTs
- Polyphase filterbanks (channelizers) and rational resampling
- Oscillators
- Noise-floor and SNR estimation
- Speech-enhancement building blocks (post-filtering, denoising)

There are **no runtime dependencies** — the module imports nothing outside the
standard library at build time.

## SIMD kernels

Performance-critical kernels have hand-tuned AVX2 implementations for `amd64`,
generated with [avo](https://github.com/mmcloughlin/avo). The generators live
under `asm/` and are tagged `//go:build ignore`; `avo` is therefore a
**generate-time-only** dependency (pinned via `tools.go`), never required by
consumers.

The committed `*_amd64.s` and `*_amd64_stub.go` files are the source of truth.
To regenerate them:

```
go generate ./...
```

Regenerated output is expected to be byte-identical to what is committed
(`git diff --exit-code` after `go generate` should be clean). Non-amd64
platforms use the portable Go fallbacks (`*_generic.go` / `*_scalar.go`).

## License

MIT — see [LICENSE](LICENSE).
