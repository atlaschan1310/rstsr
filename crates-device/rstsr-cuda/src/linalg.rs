//! `DeviceOuterAPI`, `DeviceCrossAPI` and `DeviceMatMulAPI` for `DeviceCuda`.
//!
//! Three very different amounts of work, and they sit in one file because the split is about *who
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
