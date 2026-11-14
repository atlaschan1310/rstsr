//! `DeviceOuterAPI` and `DeviceCrossAPI` for `DeviceCuda`.
//!
//! Both are elementwise-shaped: the tensor tier has already resolved the axis, the broadcasting and
//! the output layout, so the device work is "visit N slots, compute a small fixed formula". They
//! differ in how much of the existing launcher they can reuse:
//!
//! - `outer` is a planned **binary** launch. The two 1-D operands are widened onto the `(n, m)`
//!   output with a zero stride on the axis they do not own — the same trick the CPU implementation
//!   uses — and the expression is `a[ia] * b[ib]`. No new kernel.
//! - `cross` reads four elements and writes three per slot (`a` and `b` at the axis offsets +1 and
//!   +2), which no one-value-per-operand template can express, so it has its own entry:
//!   [`elementwise::cross_launch`]. The iteration, the three plan variants and the cache are shared.

use crate::dtype::{CudaArith, CudaDType};
use crate::kernels::elementwise;
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
