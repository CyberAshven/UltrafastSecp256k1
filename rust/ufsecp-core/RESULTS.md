# Validation record

Initial local validation on 2026-09-27, Windows x86-64, RTX 5070 Ti Laptop GPU
(SM 12.0), driver 616.92. Rust NVPTX: nightly-2026-04-02, LLVM 22; native GPU:
CUDA 13.4, upstream `5536321b3af2322cd2b8e90221a5bb9e4587fb7d`.

- Host: 1,539 field vectors and 1,539 scalar vectors against `num-bigint`, with
  full-width point samples against `libsecp256k1`; release tests passed.
- Actual GPU: 773 boundary/random cases of field/scalar operations, including
  generic Barrett multiplication, scalar inversion, and full-width generator
  multiplication, matched the independently checked host path.
- Downstream Pickaxe: 13,920 reconstructed mining candidates passed across
  eight transaction ages, three synthetic keys, partial batches, index
  boundaries, target boundaries, and capped winner readback.
- BCH 2026 VM: 103 accepted and 113 rejected transactions matched both standard
  policy and consensus checks.

Initial complete downstream pipeline trials used 565,248 candidates per batch,
32 points per walk lane, 16 inversions per batch lane, 45 seconds warmup, one
second settling, and eight 12-second ABBAABBA trials. One process held the GPU
exclusively; no compilation ran during timing. Rates aggregate total candidates
over total elapsed time. All trials are retained, including low-clock samples.

| Matched session | Native baseline (million candidates/s) | Rust port | Change |
|---|---:|---:|---:|
| Original Pickaxe | 114.5414 | 122.0914 | +6.59% |
| Current native upstream adapter | 114.7347 | 119.7729 | +4.39% |

These are **mining pipeline** measurements, not standalone library arithmetic
benchmarks. Pickaxe retains its own Rust SHA-256, transaction construction,
fixed-key Montgomery multiplication, and scheduling. The native upstream
adapter uses the same fixed-key specialization. The improvement must not be
attributed entirely to the arithmetic port or extrapolated to CPU signing,
other GPUs, other algorithms, or all UltrafastSecp256k1 APIs.

This is an additive experimental backend. It does not replace native defaults.
Portable CPU operations have functional coverage, not a claim to match the
native assembly/GLV engine's performance. Constant-time auditing, AMD/Intel GPU
backends, and physical testing beyond this NVIDIA device are outside this port.

Final packaging checks, repeat measurements and native regression-suite status
will be recorded before requesting review. The first native Windows build hit
a long output-path compiler error; the retry uses a shorter out-of-tree path.
