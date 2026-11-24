//! RSTSR CUDA device backend.
//!
//! - [`buffer::CudaBuffer`]: the raw storage handle (deep-copy `Clone`, logical length for
//!   `unique*` truncation, byte-erased device pointer).
//! - [`device::DeviceCuda`]: registry-canonical per-ordinal state, an execution-stream selector, a
//!   flags override, `default_order`.
//! - [`flags`]: `Option<CudaRuntimeFlags>` per device + env-seeded global default; compared as
//!   stored by `same_device`.
//! - [`stream::CudaStreamObj`]: user-visible, strictly compared.
//!
//! Execution model: every op is issued on the device value's stream and
//! returns immediately; host reads (`to_vec`, `to_scalar`, `Display`, ...)
//! synchronize explicitly. Buffer frees are stream-ordered on the buffer's
//! allocation stream.
//!
//! # Divergences from the CPU reference devices
//!
//! - `where` and `clip` are same-dtype only; the CPU devices promote scalar and mixed tensor
//!   operands through `DTypePromoteAPI`.
//! - The promoting binaries (`maximum`, `minimum`, `pow`, `atan2`, ...) are same-dtype only for the
//!   same reason. `ext_add`/`ext_sub`/`ext_mul`/`ext_div` do follow the CPU promotion table
//!   (`integer`/`float`, `real`/`complex` pairs).
//! - Transcendental unaries accept `f32`/`f64`/complex only; the CPU devices widen integer inputs
//!   to float.
//! - Float-to-integer casts use the C conversion rather than Rust's saturating `as` (`NaN -> 0` and
//!   out-of-range clamping differ); integer-to-float and width changes follow the CPU.
//! - Integer division and remainder by zero yield `0` (NumPy); the CPU reference devices panic on
//!   Rust's `/` and `%`.
//! - `fill`/`ones`/`zeros` take the kernel path for every supported dtype (complex included).
//!
//! # What is implemented
//!
//! | family | coverage |
//! |---|---|
//! | storage / transfer | creation, fill, strided copy with cast, `to_device` both ways, streams |
//! | elementwise | arithmetic, comparison, transcendentals, unary float families, `where`, `clip` (same dtype) |
//! | reductions | `sum`/`prod`/`mean`/`l2_norm`/`min`/`max`/`all`/`any`/`count_nonzero` (+ dtype variants), `var`/`std` (+ dtype), `argmin`/`argmax`/`nanarg*` (raveled, all + axes), `cumsum`/`cumprod` (real dtypes) |
//! | linalg | `outer`, `cross`, `matmul`/`matmul_uninit` (cuBLAS, tiled kernel as fallback), `tensordot` (view-only tier) |
//! | indexing | `index_select`, `index_put` |
//!
//! **Absent, and why** — these are deliberate, not pending:
//!
//! - `take_along_axis`/`put_along_axis`, `mask_select`: the first two take a *device* index buffer,
//!   so an out-of-range index cannot be caught without a read-back on every call; `mask_select` is a
//!   compaction and needs a mask prefix sum to be deterministic. No caller in the current programs.
//! - `sort`, `argsort`, `unique`, `searchsorted`, `isin`: no caller. A GPU sort is a bitonic or
//!   radix kernel family, not an afternoon.
//! - `nonzero`: compaction, same shape of work as `mask_select`.
//! - `tril`/`triu`, `cumsum`/`cumprod` `_dtype`, `unraveled_arg*`: small, unimplemented, no caller.
//! - `allclose`: `allclose_axes` is `unimplemented!()` on the CPU devices too.
//! - `mapv` and the `Op_Mut*_Ref*` closure traits: a host closure cannot cross into a kernel, so
//!   these are unimplementable on any accelerator rather than missing here.
//!
//! # Numerics: what is guaranteed, and what is not
//!
//! A GPU backend owes the reader an answer to "will this match the CPU device?", and the honest
//! answer has three tiers rather than one.
//!
//! **Bit-identical: elementwise arithmetic and comparisons.** One IEEE operation per element, in the
//! same order, so `a * b` produces the same bits on every element whatever the device.
//!
//! **Deterministic, differently ordered: reductions.** A shuffle tree adds in a different order than
//! a left-to-right loop, and float addition is not associative. Two things follow, and the second is
//! the one worth stating:
//!
//! - The difference is *bounded*, not random. No kernel here uses an atomic, and the plan
//!   (`kernels::reduce_plan`) is a pure function of shape, strides and the device's default order —
//!   `MIN_BLOCKS` is a policy constant rather than a device query, so the tree does not change with
//!   SM count, block count or occupancy. Two devices running the same plan produce **the same bits**,
//!   which is a stronger promise than "reproducible on this machine".
//! - Bit-identity with the CPU is *not worth buying*. It would mean one lane per output and a serial
//!   walk over the contraction: the kernel's entire reason to exist, surrendered for a discrepancy
//!   at the level of ordinary summation error. The trade is deliberate. What a caller gets instead is
//!   a result that is stable across runs and across devices of the same capability.
//!
//! FMA contraction belongs to this tier too. `-fmad=true` is NVRTC's **default**, so `l2_norm`,
//! `var` and `vecdot` may compute `fma(x, x, a)` where the CPU computes `fl(a + fl(x * x))` — one
//! rounding instead of two. `RSTSR_CUDA_STRICT_FP=1` disables it at a throughput cost; the tree order
//! stays different either way, so it narrows the gap rather than closing it.
//!
//! **Different implementation: transcendentals.** `sinf`/`expf`/`pow` are NVIDIA's implementations of
//! the same mathematical function — accurate to the last ulp, by a different route than Rust's libm.
//! No flag makes them bit-identical, and nothing here pretends otherwise.
//!
//! **Narrower arithmetic, on request.** `RSTSR_CUDA_COMPUTE_TYPE_32F=fasttf32|fast16f|fast16bf` routes
//! f32 `matmul` through `cublasGemmEx` with a down-converting compute type: the operands stay f32 in
//! memory, but the tensor core multiplies 10- or 7-bit mantissas. This is the only knob that changes
//! results *on purpose* rather than as a side effect of ordering, and it is off by default —
//! `Default` keeps the typed `Sgemm`, which is IEEE fp32. It buys the tensor-core rate (on an A100,
//! 19.5 → 156 → 312 TFLOP/s for fp32/TF32/bf16 math); whether the narrower mantissa is acceptable is
//! a property of the caller's algorithm, not of this crate, so nothing here decides it for you. `f64`
//! ignores the flag: there is no 64-bit compute type that trades precision for rate.
//!
//! The mantissa is not the only thing that changes. `fast16f` also bounds the *inputs* (below ≈6e-5
//! and above 65504, the range of a 5-bit exponent), so on data spanning many decades it fails
//! differently rather than merely worse; `fasttf32` and `fast16bf` keep fp32's 8-bit exponent and
//! have no such bound. `tests/test_kernels.rs` pins both statements: `rel_l2` against an f64
//! reference built from the same f32 operands, since the flag is about the arithmetic and nothing
//! else.
//!
//! The practical rule: compare against the CPU with a **tolerance**, never with `==`. That is what
//! this crate's tests do (`assert_close`, `ATOL 1e-9`), and what any cross-implementation comparison
//! of floating point has to do. Reproducibility *across runs* is guaranteed; identity *with the CPU*
//! is not, and buying it would cost the kernel.
//!
//! # Running the tests
//!
//! ```text
//! CUDARC_CUDA_VERSION=12060 cargo test -p rstsr-cuda --lib            # no device needed
//! CUDARC_CUDA_VERSION=12060 cargo test -p rstsr-cuda --test test_kernels
//! CUDARC_CUDA_VERSION=12060 cargo test -p rstsr-cuda --test test_matmul_tiled
//! ```
//!
//! Without a visible driver every hardware test prints `skipping: no CUDA device visible` and
//! passes, so a green run on a machine without a GPU means *nothing ran*. `--lib` is the half that
//! runs anywhere: the reduction plan, the kernel-token checks, the cuBLAS argument derivation and
//! the addressing rules are all pure functions and are tested there.
//!
//! Environment knobs, both read once per process:
//!
//! - `RSTSR_CUDA_MATMUL=tiled` forces the tiled GEMM instead of cuBLAS. It exists so the fallback
//!   keeps hardware coverage; `tests/test_matmul_tiled.rs` sets it.
//! - `RSTSR_CUDA_DEBUG_GEMM=1` prints the resolved geometry, the cuBLAS arguments and the pointer
//!   bases once per process. This is the instrument that turned a stride-units bug into a two-minute
//!   read after three sessions of inference.
//! - `RSTSR_CUDA_STRICT_FP=1` compiles with `--fmad=false`, removing FMA contraction from the
//!   kernels that can use it (`l2_norm`, `var`, `vecdot`). See the numerics section above for what
//!   this does and does not recover.
//! - `RSTSR_CUDA_COMPUTE_TYPE_32F=default|pedantic|fasttf32|fast16f|fast16bf` selects the cuBLAS
//!   compute type for **f32** `matmul` (`f64` ignores it). `fasttf32` is the one to try first: 10-bit
//!   mantissa, f32 exponent range, ~8× the f32 rate on Ampere. `fast16f` is the variant to avoid on
//!   integrals — fp16's 5-bit exponent underflows below ≈6e-5 and overflows above 65504, so it is a
//!   mode with an input *range*, not merely a wider error bar; bf16 has fp32's range. A compute type
//!   the device refuses falls back to exact arithmetic with a one-time warning, so "the flag did
//!   nothing" is distinguishable from "the flag never arrived".
//! - `RSTSR_CUDA_ARCH=compute_80` compiles for an arch other than the card's own. Diagnostic only:
//!   PTX is forward-compatible, so an sm_86 card will host `compute_75`/`compute_80`/`compute_86`
//!   and an sm_90 will host everything below it. That is what makes an arch matrix possible from a
//!   single GPU — see below.
//!
//! # Kernel compilation has two accept/reject points, not one
//!
//! `NVRTC` compiles the generated source to PTX; the **driver** then loads that PTX and JITs it for
//! the actual device. Both can refuse, independently, and only the first happens on the machine that
//! wrote the kernel. Reference case: `max.NaN.f64` is accepted by NVRTC and rejected by one driver's
//! JIT with `CUDA_ERROR_INVALID_PTX`.
//!
//! So "it compiled here" says nothing about the second gate, and the failure surfaces at first use,
//! on the user's machine, as a launch error. The mitigation is to move the arch flag rather than the
//! code: `RSTSR_CUDA_ARCH=compute_NN cargo test -p rstsr-cuda` exercises the whole suite — hence
//! every kernel it touches — compiled for a different arch, with the disk cache keyed on that arch
//! so the runs do not collide. Four runs on one card cover the four arches that card can host.
//!
//! What this does **not** cover: different driver versions (the driver JIT is host-side, so that
//! needs a second host), and arches above the card's own (`compute_89`/`compute_90` need those
//! cards).
//!
//! `scripts/arch-matrix.sh` runs that loop: it derives the arch list from the card, gives each run
//! its own cache directory, and **refuses to report a pass that compiled nothing** — a run served
//! from a stale cache would otherwise look exactly like a successful one.

#![allow(clippy::needless_return)]

pub mod adv_index_ops;
pub mod arg_ops;
pub mod assignment;
pub mod buffer;
pub mod conversion;
pub mod cpu;
pub mod creation;
pub mod device;
pub mod dtype;
pub mod error;
pub mod flags;
pub mod format;
pub mod kernel;
pub mod linalg;
pub(crate) mod kernels;

pub(crate) mod kernels_ops;
pub mod operators;
pub mod reduce_ops;
pub mod scan_ops;
pub mod registry;
pub mod stream;
pub mod transfer;

pub mod prelude_dev {
    pub use crate::buffer::CudaBuffer;
    pub use crate::device::DeviceCuda;
    pub use crate::flags::{default_flags, set_default_flags, ComputeType32F, CudaRuntimeFlags};
    pub use crate::registry::CudaOrdinalObj;
    pub use crate::stream::CudaStreamObj;
    pub use rstsr_core::prelude_dev::*;
}

/// Device-level free functions (`rt::cuda::*` candidates for the facade).
pub mod funcs {
    pub use crate::flags::{default_flags, set_default_flags};
    use crate::prelude_dev::*;
    use cudarc::driver::CudaContext;

    /// A device on ordinal `ordinal` (registry-canonical).
    pub fn device(ordinal: usize) -> DeviceCuda {
        DeviceCuda::new(ordinal)
    }

    /// Number of visible CUDA devices (0 if CUDA is unavailable).
    pub fn device_count() -> usize {
        CudaContext::device_count().unwrap_or(0).max(0) as usize
    }

    /// Whether any CUDA device is available.
    pub fn is_available() -> bool {
        device_count() > 0
    }

    /// `(free, total)` memory of `device` in bytes.
    pub fn memory_info(device: &DeviceCuda) -> Result<(usize, usize)> {
        device.memory_info()
    }

    /// Block until all work on `device`'s execution stream completes.
    pub fn synchronize(device: &DeviceCuda) -> Result<()> {
        device.synchronize()
    }
}
