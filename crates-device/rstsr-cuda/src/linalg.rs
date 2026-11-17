//! `DeviceOuterAPI`, `DeviceCrossAPI`, `DeviceMatMulAPI` and `DeviceTensordotAPI` for `DeviceCuda`.
//!
//! Four very different amounts of work, and they sit in one file because the split is about *who
//! owns the geometry*, not about the operator:
//!
//! - `outer` is a planned **binary** launch. The two 1-D operands are widened onto the `(n, m)`
//!   output with a zero stride on the axis they do not own — the same trick the CPU implementation
//!   uses — and the expression is `a[ia] * b[ib]`. No new kernel.
//! - `cross` reads four elements and writes three per slot (`a` and `b` at the axis offsets +1 and
//!   +2), which no one-value-per-operand template can express, so it has its own entry:
//!   [`elementwise::cross_launch`]. The iteration, the three plan variants and the cache are shared.
//! - `matmul` is the only operation here whose *thread geometry* is the point: a tiled GEMM in
//!   [`gemm`], where the block owns a tile of the output and the contraction streams through shared
//!   memory. It is also the only one that needs the batch axes, which ride in `grid.z`.
//! - `tensordot` owns no geometry at all: it is a *scheduler* over `matmul`. The axis reorder and the
//!   block merges are a pure function of layouts shared with every other backend, so the impl is the
//!   planner plus one call — and the interesting part is what it does when the planner declines.

use crate::dtype::{CudaArith, CudaDType};
use crate::kernels::elementwise;
use crate::kernels::cublas_gemm::CublasScalar;
use crate::kernels::gemm;
use crate::prelude_dev::*;

impl<TA, TB, TC> DeviceOuterAPI<TA, TB, TC> for DeviceCuda
where
    Self: DeviceRawAPI<TA, Raw = CudaBuffer<TA>>
        + DeviceRawAPI<TB, Raw = CudaBuffer<TB>>
        + DeviceRawAPI<MaybeUninit<TC>, Raw = CudaBuffer<MaybeUninit<TC>>>,
    TA: CudaDType + CudaArith,
    TB: CudaDType + CudaArith,
    TC: CudaDType + CudaArith,
{
    fn outer(
        &self,
        c: &mut CudaBuffer<MaybeUninit<TC>>,
        lc: &Layout<Ix2>,
        a: &CudaBuffer<TA>,
        la: &Layout<Ix1>,
        b: &CudaBuffer<TB>,
        lb: &Layout<Ix1>,
    ) -> Result<()> {
        let (n, m) = (la.shape()[0], lb.shape()[0]);
        rstsr_assert_eq!(
            lc.shape(),
            &[n, m],
            InvalidLayout,
            "the outer-product output should have shape (a.len, b.len)"
        )?;

        // Widen both operands onto the (n, m) output: the axis an operand does not own gets stride
        // 0. The plans already handle arbitrary per-axis strides including zero, so this needs no
        // special case downstream — `binary_launch` sees three rank-2 layouts.
        let lam = Layout::<Ix2>::new([n, m], [la.stride()[0], 0], la.offset())?;
        let lbm = Layout::<Ix2>::new([n, m], [0, lb.stride()[0]], lb.offset())?;

        elementwise::binary_launch::<TC, TA, TB>(self, c, lc, a, &lam, b, &lbm, "(a[ia]) * (b[ib])")
    }
}

impl<TA, TB, TC, D> DeviceCrossAPI<TA, TB, TC, D> for DeviceCuda
where
    D: DimAPI,
    Self: DeviceRawAPI<TA, Raw = CudaBuffer<TA>>
        + DeviceRawAPI<TB, Raw = CudaBuffer<TB>>
        + DeviceRawAPI<MaybeUninit<TC>, Raw = CudaBuffer<MaybeUninit<TC>>>,
    TA: CudaDType + CudaArith,
    TB: CudaDType + CudaArith,
    TC: CudaDType + CudaArith,
{
    fn cross(
        &self,
        c: &mut CudaBuffer<MaybeUninit<TC>>,
        lc: &Layout<D>,
        a: &CudaBuffer<TA>,
        la: &Layout<D>,
        b: &CudaBuffer<TB>,
        lb: &Layout<D>,
        axis: usize,
    ) -> Result<()> {
        // `axis` is already a valid index into the (broadcast) output rank, and all three layouts
        // share one shape with `shape[axis] == 3` — the tier checked both.
        elementwise::cross_launch(self, c, lc, a, la, b, lb, axis)
    }
}

impl<T, DA, DB, DC> DeviceMatMulAPI<T, T, T, DA, DB, DC> for DeviceCuda
where
    DA: DimAPI,
    DB: DimAPI,
    DC: DimAPI,
    Self: DeviceRawAPI<T, Raw = CudaBuffer<T>>
        + DeviceRawAPI<MaybeUninit<T>, Raw = CudaBuffer<MaybeUninit<T>>>,
    // `num::Zero` is only used for the fresh-output path's `beta`, which both paths then treat as
    // "do not read c" (cuBLAS has the same convention). It is not part of the arithmetic.
    //
    // `CublasScalar` is the bound that matters: it names exactly the two dtypes cuBLAS's typed
    // `gemm` covers, so it also excludes complex — and complex stays out on purpose. The tiled
    // kernel spells its zero as `(@T@)0`, which `rstsr_complex` has no single-argument constructor
    // for, so a complex instantiation would fail inside NVRTC (on a rented GPU) rather than here.
    // A complex GEMM needs a zero spelled as a two-argument construction and the mixed-precision
    // expression builders — separate work.
    //
    // Integers are excluded too. They *were* handled by the tiled kernel and are now not handled at
    // all; that is a deliberate narrowing, because an integer GEMM path that has never run on
    // hardware is worse than a compile error.
    T: CublasScalar + num::Zero,
{
    fn matmul(
        &self,
        c: &mut CudaBuffer<T>,
        lc: &Layout<DC>,
        a: &CudaBuffer<T>,
        la: &Layout<DA>,
        b: &CudaBuffer<T>,
        lb: &Layout<DB>,
        alpha: T,
        beta: T,
    ) -> Result<()> {
        gemm::gemm_launch(
            self,
            c,
            &lc.to_dim::<IxD>()?,
            a,
            &la.to_dim::<IxD>()?,
            b,
            &lb.to_dim::<IxD>()?,
            alpha,
            beta,
        )
    }

    fn matmul_uninit(
        &self,
        c: &mut CudaBuffer<MaybeUninit<T>>,
        lc: &Layout<DC>,
        a: &CudaBuffer<T>,
        la: &Layout<DA>,
        b: &CudaBuffer<T>,
        lb: &Layout<DB>,
        alpha: T,
    ) -> Result<()> {
        // `beta = 0` is the whole of the "never read c" contract: the epilogue is guarded by
        // `beta == 0`, so the `MaybeUninit` slots are written and not touched beforehand.
        gemm::gemm_launch(
            self,
            c,
            &lc.to_dim::<IxD>()?,
            a,
            &la.to_dim::<IxD>()?,
            b,
            &lb.to_dim::<IxD>()?,
            alpha,
            <T as num::Zero>::zero(),
        )
    }
}

/// `tensordot` as a single GEMM — tier 0 only: **views, never a copy.**
///
/// # Why this is only a few lines
///
/// A contraction is not a BLAS operation. cuBLAS's contract is exactly "one 2-D product, each
/// operand addressed by a leading dimension on one axis" (see `kernels::cublas_gemm`), so
/// the whole of `tensordot` above that call is *scheduling*: reorder each operand to
/// `[free…, ctr…]` / `[ctr…, free…]`, and ask whether each of the three blocks collapses to 2-D
/// without materializing data. That decision already exists as
/// [`tensordot_gemm_layouts`] — a pure function of layouts, shared by every backend in the
/// workspace — so this impl is the planner
/// plus [`Self::matmul_uninit`], which lands in cuBLAS.
///
/// # Where tier 0 actually stops
///
/// Verified host-side before the GPU ever sees it (`probe/geomprobe`, bin `tensordot`, which walks
/// every `(m, k)` of the returned blocks and checks that their offsets are the elements the
/// contraction names):
///
/// - **Both orientations are covered.** A block that is contiguous on its `M` axis instead of `N`
///   comes back as a 2-D layout with unit stride on `M`, which the cuBLAS argument derivation
///   turns into `OP_T`. So contracting a *leading* axis of `a` — `(3,4,5) over [0]` — is a GEMM, not
///   a pack: `a` arrives as `[20, 3]` with strides `[1, 20]`. This is better than expected and worth
///   stating, because the opposite was the natural assumption.
/// - **What declines is a `K` that is not one stride.** Contracting `a`'s axes `[0, 2]` leaves the
///   axes with a hole between them (`24` and `1`), so no single leading dimension spans `K`. Same for
///   any free/contracted interleave that no one-stride pair covers.
///
/// The planner also only ever inspects layouts, so its verdict is a property of the *shape and
/// strides*, never of the data: a refusal is reproducible without a GPU.
///
/// # What is deliberately absent: the pack
///
/// When the planner declines, the CPU implementations fall back to a naive scalar kernel, which is
/// within a few times of BLAS there. On a GPU that fallback is not a fallback: a naive contraction is
/// DRAM-bound at `2k/(8(k+1)) ≈ 0.25` flop/byte, about 0.4 TFLOP/s on an A100, against `dgemm`'s
/// 19.5 — a factor of ~50. So the honest GPU answer is a *third* tier, "pack the operand that does
/// not collapse, then the same GEMM", not the scalar loop.
///
/// That tier is not implemented, and until it is this function **refuses by name** rather than
/// silently unpacking into something slow: a pack costs one extra pass over `m*k` elements and buys
/// `2*m*k*n` flops, so it is nearly always the right trade — it just needs its own measurement
/// before it becomes the default. The refusal prints the three layouts, because "which block failed
/// to merge" is otherwise invisible.
///
/// # One asymmetry with `matmul`, on purpose
///
/// `matmul` refuses an inner dot ("single-axis contraction") and points at the reduce kernel,
/// because the tiled fallback would put 256 threads on a `1 x 1` output. `tensordot` accepts it: the
/// planner yields `m = n = 1`, and cuBLAS runs a `1 x 1 x k` product without wasting anything. Same
/// arithmetic, and here the fast path is the one that is already correct.
///
/// # What is refused for a different reason
///
/// `DeviceExtTensordotAPI` (mixed input dtypes) is not implemented at all: cuBLAS's typed `gemm`
/// takes one dtype, and a mixed product needs `cublasGemmEx` with per-operand types. The tiled
/// fallback is single-`@T@` too, so there is nothing here to fall back to.
impl<T, DA, DB, DC> DeviceTensordotAPI<T, T, T, DA, DB, DC> for DeviceCuda
where
    DA: DimAPI,
    DB: DimAPI,
    DC: DimAPI,
    Self: DeviceRawAPI<T, Raw = CudaBuffer<T>>
        + DeviceRawAPI<MaybeUninit<T>, Raw = CudaBuffer<MaybeUninit<T>>>
        + DeviceMatMulAPI<T, T, T, Ix2, Ix2, Ix2>,
    T: CublasScalar + num::Zero + num::One,
{
    fn tensordot(
        &self,
        c: &mut CudaBuffer<MaybeUninit<T>>,
        lc: &Layout<DC>,
        a: &CudaBuffer<T>,
        la: &Layout<DA>,
        b: &CudaBuffer<T>,
        lb: &Layout<DB>,
        axes_a: &[isize],
        axes_b: &[isize],
    ) -> Result<()> {
        let order = self.default_order();
        let Some((la2, lb2, lc2)) = tensordot_gemm_layouts(la, axes_a, lb, axes_b, lc, order)? else {
            rstsr_raise!(
                UnImplemented,
                "cuda tensordot: this contraction is not a GEMM over views, so it would need a \
                 packing pass first (implemented on CPU as a naive kernel, deliberately not here: a \
                 scalar contraction on a GPU is ~50x off cuBLAS). axes_a {axes_a:?}, axes_b \
                 {axes_b:?}; a shape {:?} stride {:?}; b shape {:?} stride {:?}; c shape {:?} \
                 stride {:?}",
                la.shape(),
                la.stride(),
                lb.shape(),
                lb.stride(),
                lc.shape(),
                lc.stride()
            )?
        };
        // The three returned layouts address the same buffers `c`, `a`, `b` already are: the planner
        // only reordered axes and merged blocks, so every element the GEMM reads or writes is one the
        // caller handed us. `probe/geomprobe` walks those offsets element by element.
        self.matmul_uninit(c, &lc2, a, &la2, b, &lb2, <T as num::One>::one())
    }
}
