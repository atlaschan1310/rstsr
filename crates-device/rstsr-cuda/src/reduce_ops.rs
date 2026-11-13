//! Unary reductions on the CUDA backend: the nine `OpReduce*API` traits that fit a
//! single-accumulator associative fold.
//!
//! | trait | op | acc | identity | note |
//! |---|---|---|---|---|
//! | `OpSumAPI` | `Sum` | `T` | `0` | |
//! | `OpProdAPI` | `Prod` | `T` | `1` | |
//! | `OpMeanAPI` | `Mean` | `T` | NaN | empty input divides 0/0, like the CPU |
//! | `OpL2NormAPI` | `L2Norm` | `T` | `0` | real dtypes only (`Real = Self`) |
//! | `OpMinAPI` | `Min` | `T` | `+inf` | NaN-propagating; empty input raises |
//! | `OpMaxAPI` | `Max` | `T` | `-inf` | NaN-propagating; empty input raises |
//! | `OpSumDtypeAPI` | `Sum` | `TOut` | `0` | `f32 → f64`, `i32 → f64` accumulate |
//! | `OpProdDtypeAPI` | `Prod` | `TOut` | `1` | |
//! | `OpMeanDtypeAPI` | `Mean` | `TOut` | NaN | |
//!
//! This is a sibling module to `operators.rs` rather than a block inside it: the nine impls, their
//! two helpers and their generic bounds are ~350 lines of one subject, and they are the only thing in
//! the crate with an accumulator type distinct from the operand type.
//!
//! # Why not the other six
//!
//! Six rows of [`OpReduceAPI`]'s table are absent, and none of them is "not typed out yet":
//!
//! - **`OpVarAPI`, `OpStdAPI`** (+ their dtype variants): the accumulator is a *pair*,
//!   `(Σx, Σ|x|²)`, because the CPU computes `Σ|x|²/n − |mean|²` in one pass. One accumulator cannot
//!   hold it, and two passes over the data would double the bytes read for an op that is
//!   memory-bound. Needs a second register and a second shuffle, not a new plan.
//! - **`OpArgMinAPI`, `OpArgMaxAPI`, `OpNanArgMinAPI`, `OpNanArgMaxAPI`**: the accumulator carries an
//!   index alongside the value, and ties resolve to the *first* index. That is achievable by packing
//!   `(value, index)` and comparing lexicographically, but "first" is first in the CPU's traversal
//!   order, and `block_step` deliberately allows the contracted axes in any order — so the tie-break
//!   would silently depend on memory layout. Needs a decision, not code.
//! - **`OpAllAPI`, `OpAnyAPI`, `OpCountNonZeroAPI`, `OpSumBoolAPI`**: `TOut` is `bool`/`usize`. Both
//!   are plausible `CudaDType`s (`rt::lt` already yields `CudaBuffer<bool>`), but neither is confirmed,
//!   and a wrong guess is a compile error in the impl header rather than at the call site.
//! - **`OpCumSumAPI`, `OpCumProdAPI`** (+ dtype variants): a *scan*, not a reduction. Different
//!   algorithm, different kernel, and the plan does not describe it.
//! - **`OpAllCloseAPI`**: `allclose_axes` is `unimplemented!()` on the CPU too.
//! - **`OpReduceCustomAPI`, `OpUnraveledArgMin/MaxAPI`**: closures, and an `IxD` output.

#![allow(clippy::type_complexity)]

use core::mem::MaybeUninit;

use crate::kernels::reduce_plan::classify_reduce;
use crate::kernels::reduce_unary::{reduce_unary_launch, AccSpec, ReduceAcc, UnaryOp};
use crate::prelude_dev::*;

/// The four things every one of these methods does, in one place.
///
/// The output layout is the only part that mirrors the CPU (`layout_for_array_copy(&lm, K)`); it is
/// part of the observable contract, because a different axis order would give the same numbers with
/// different strides depending on which device ran the op.
///
/// The `DeviceRawAPI<TOut, Raw = CudaBuffer<TOut>>` bound is not redundant with
/// `DeviceRawAPI<MaybeUninit<TOut>, …>`: `full_impl`/`empty_impl`/`to_cpu_vec` all name the *plain*
/// projection, and a projection on a generic parameter has to be pinned explicitly or it stays
/// opaque. For the six same-type impls the compiler gets it from `DeviceRawAPI<T, Raw = CudaBuffer<T>>`
/// for free, which is why only the generic-`TOut` cases need it spelled out.
use crate::dtype::CudaDType;
fn reduce_axes_impl<TIn, TOut>(
    device: &DeviceCuda,
    op: UnaryOp,
    acc: AccSpec,
    a: &CudaBuffer<TIn>,
    la: &Layout<IxD>,
    axes: &[isize],
    empty_fill: TOut,
) -> Result<(Storage<DataOwned<CudaBuffer<TOut>>, TOut, DeviceCuda>, Layout<IxD>)>
where
    TIn: CudaDType,
    TOut: CudaDType,
    DeviceCuda: DeviceRawAPI<TIn, Raw = CudaBuffer<TIn>>
        + DeviceRawAPI<TOut, Raw = CudaBuffer<TOut>>
        + DeviceRawAPI<MaybeUninit<TOut>, Raw = CudaBuffer<MaybeUninit<TOut>>>
        + DeviceCreationAnyAPI<TOut>,
{
    let (las, lm) = la.dim_split_axes(axes)?;
    let lo = layout_for_array_copy(&lm, TensorIterOrder::K)?;
    let n_out = lo.size();

    // A unary reduction has no second operand, so the same layout goes in twice and the plan's `b`
    // strides come out as duplicates that the launcher ignores.
    let plan = classify_reduce(la, la, &lo, axes, axes, device.default_order())?;

    // An empty contraction writes nothing — the launcher returns before touching the buffer — so the
    // output is filled with the identity instead of being left uninitialised. `sum([]) == 0` must not
    // read device memory that happens to look like zero.
    let mut st: Storage<DataOwned<CudaBuffer<TOut>>, TOut, DeviceCuda> =
        if plan.n_contract == 0 || n_out == 0 {
            device.full_impl(n_out, empty_fill)?
        } else {
            // SAFETY: the kernel writes all `n_out` elements before the buffer is read, which is the
            // contract `empty_impl` asks the caller to uphold.
            unsafe { device.empty_impl(n_out)? }
        };

    if plan.n_contract > 0 && n_out > 0 {
        let buf: &mut CudaBuffer<TOut> = st.raw_mut();
        let buf_uninit: &mut CudaBuffer<MaybeUninit<TOut>> =
            unsafe { &mut *(buf as *mut CudaBuffer<TOut> as *mut CudaBuffer<MaybeUninit<TOut>>) };

        reduce_unary_launch(device, op, acc, buf_uninit, &lo, a, &las, &plan)?;
    }

    Ok((st, lo))
}

/// `*_all`: reduce every axis and take the single element back.
fn reduce_all_impl<TIn, TOut>(
    device: &DeviceCuda,
    op: UnaryOp,
    acc: AccSpec,
    a: &CudaBuffer<TIn>,
    la: &Layout<IxD>,
    empty_fill: TOut,
) -> Result<TOut>
where
    TIn: CudaDType,
    TOut: CudaDType,
    DeviceCuda: DeviceRawAPI<TIn, Raw = CudaBuffer<TIn>>
        + DeviceRawAPI<TOut, Raw = CudaBuffer<TOut>>
        + DeviceRawAPI<MaybeUninit<TOut>, Raw = CudaBuffer<MaybeUninit<TOut>>>
        + DeviceCreationAnyAPI<TOut>
        + DeviceToCpuAPI<TOut>,
{
    // Every axis, which the classifier turns into a rank-0 output of one element.
    let axes: Vec<isize> = (0..la.ndim() as isize).collect();
    let (st, _lo) = reduce_axes_impl(device, op, acc, a, la, &axes, empty_fill)?;

    let host: Vec<TOut> = DeviceToCpuAPI::<TOut>::to_cpu_vec::<DataOwned<CudaBuffer<TOut>>>(&st)?;
    match host.into_iter().next() {
        Some(v) => Ok(v),
        None => rstsr_raise!(InvalidValue, "cuda reduce: full reduction produced no element"),
    }
}

impl<T, D> OpSumAPI<T, D> for DeviceCuda
where
    Self: DeviceRawAPI<T, Raw = CudaBuffer<T>>
        + DeviceRawAPI<MaybeUninit<T>, Raw = CudaBuffer<MaybeUninit<T>>>
        + DeviceCreationAnyAPI<T>
        + DeviceToCpuAPI<T>,
    // Only `Zero` is needed, and only for the empty-input fill: the addition happens in the kernel.
    T: CudaDType + ReduceAcc + num::Zero,
    D: DimAPI,
{
    type TOut = T;

    fn sum_all(&self, a: &CudaBuffer<T>, la: &Layout<D>) -> Result<T> {
        reduce_all_impl(self, UnaryOp::Sum, T::SPEC, a, &la.to_dim::<IxD>()?, T::zero())
    }

    fn sum_axes(
        &self,
        a: &CudaBuffer<T>,
        la: &Layout<D>,
        axes: &[isize],
    ) -> Result<
        (Storage<DataOwned<<Self as DeviceRawAPI<Self::TOut>>::Raw>, Self::TOut, Self>, Layout<IxD>),
    > {
        reduce_axes_impl(self, UnaryOp::Sum, T::SPEC, a, &la.to_dim::<IxD>()?, axes, T::zero())
    }
}

impl<T, D> OpProdAPI<T, D> for DeviceCuda
where
    Self: DeviceRawAPI<T, Raw = CudaBuffer<T>>
        + DeviceRawAPI<MaybeUninit<T>, Raw = CudaBuffer<MaybeUninit<T>>>
        + DeviceCreationAnyAPI<T>
        + DeviceToCpuAPI<T>,
    T: CudaDType + ReduceAcc + num::One,
    D: DimAPI,
{
    type TOut = T;

    fn prod_all(&self, a: &CudaBuffer<T>, la: &Layout<D>) -> Result<T> {
        reduce_all_impl(self, UnaryOp::Prod, T::SPEC, a, &la.to_dim::<IxD>()?, T::one())
    }

    fn prod_axes(
        &self,
        a: &CudaBuffer<T>,
        la: &Layout<D>,
        axes: &[isize],
    ) -> Result<
        (Storage<DataOwned<<Self as DeviceRawAPI<Self::TOut>>::Raw>, Self::TOut, Self>, Layout<IxD>),
    > {
        reduce_axes_impl(self, UnaryOp::Prod, T::SPEC, a, &la.to_dim::<IxD>()?, axes, T::one())
    }
}

impl<T, D> OpMeanAPI<T, D> for DeviceCuda
where
    Self: DeviceRawAPI<T, Raw = CudaBuffer<T>>
        + DeviceRawAPI<MaybeUninit<T>, Raw = CudaBuffer<MaybeUninit<T>>>
        + DeviceCreationAnyAPI<T>
        + DeviceToCpuAPI<T>,
    // `Float` covers the empty-input fill: an empty mean is `0/0`, and the CPU returns NaN too.
    T: CudaDType + ReduceAcc + num::Float,
    D: DimAPI,
{
    type TOut = T;

    fn mean_all(&self, a: &CudaBuffer<T>, la: &Layout<D>) -> Result<T> {
        reduce_all_impl(self, UnaryOp::Mean, T::SPEC, a, &la.to_dim::<IxD>()?, T::nan())
    }

    fn mean_axes(
        &self,
        a: &CudaBuffer<T>,
        la: &Layout<D>,
        axes: &[isize],
    ) -> Result<
        (Storage<DataOwned<<Self as DeviceRawAPI<Self::TOut>>::Raw>, Self::TOut, Self>, Layout<IxD>),
    > {
        reduce_axes_impl(self, UnaryOp::Mean, T::SPEC, a, &la.to_dim::<IxD>()?, axes, T::nan())
    }
}

impl<T, D> OpL2NormAPI<T, D> for DeviceCuda
where
    Self: DeviceRawAPI<T, Raw = CudaBuffer<T>>
        + DeviceRawAPI<MaybeUninit<T>, Raw = CudaBuffer<MaybeUninit<T>>>
        + DeviceCreationAnyAPI<T>
        + DeviceToCpuAPI<T>,
    // `Real = Self`, so the accumulator and the output are the operand's own type. A complex operand
    // needs `|x|²` in the fold and a real output buffer: not wired up, and refused at the call site
    // because `Complex` is not `Float`.
    T: CudaDType + ReduceAcc + num::Float,
    D: DimAPI,
{
    type TOut = T;

    fn l2_norm_all(&self, a: &CudaBuffer<T>, la: &Layout<D>) -> Result<T> {
        reduce_all_impl(self, UnaryOp::L2Norm, T::SPEC, a, &la.to_dim::<IxD>()?, T::zero())
    }

    fn l2_norm_axes(
        &self,
        a: &CudaBuffer<T>,
        la: &Layout<D>,
        axes: &[isize],
    ) -> Result<
        (Storage<DataOwned<<Self as DeviceRawAPI<Self::TOut>>::Raw>, Self::TOut, Self>, Layout<IxD>),
    > {
        reduce_axes_impl(self, UnaryOp::L2Norm, T::SPEC, a, &la.to_dim::<IxD>()?, axes, T::zero())
    }
}

impl<T, D> OpMinAPI<T, D> for DeviceCuda
where
    Self: DeviceRawAPI<T, Raw = CudaBuffer<T>>
        + DeviceRawAPI<MaybeUninit<T>, Raw = CudaBuffer<MaybeUninit<T>>>
        + DeviceCreationAnyAPI<T>
        + DeviceToCpuAPI<T>,
    T: CudaDType + ReduceAcc + num::Float,
    D: DimAPI,
{
    type TOut = T;

    fn min_all(&self, a: &CudaBuffer<T>, la: &Layout<D>) -> Result<T> {
        if la.size() == 0 {
            rstsr_raise!(InvalidValue, "zero-size array is not supported for min")?;
        }
        // The fill is unreachable: a zero-size input raises above, so `n_contract` is never 0 here.
        reduce_all_impl(self, UnaryOp::Min, T::SPEC, a, &la.to_dim::<IxD>()?, T::zero())
    }

    fn min_axes(
        &self,
        a: &CudaBuffer<T>,
        la: &Layout<D>,
        axes: &[isize],
    ) -> Result<
        (Storage<DataOwned<<Self as DeviceRawAPI<Self::TOut>>::Raw>, Self::TOut, Self>, Layout<IxD>),
    > {
        if la.size() == 0 {
            rstsr_raise!(InvalidValue, "zero-size array is not supported for min")?;
        }
        reduce_axes_impl(self, UnaryOp::Min, T::SPEC, a, &la.to_dim::<IxD>()?, axes, T::zero())
    }
}

impl<T, D> OpMaxAPI<T, D> for DeviceCuda
where
    Self: DeviceRawAPI<T, Raw = CudaBuffer<T>>
        + DeviceRawAPI<MaybeUninit<T>, Raw = CudaBuffer<MaybeUninit<T>>>
        + DeviceCreationAnyAPI<T>
        + DeviceToCpuAPI<T>,
    T: CudaDType + ReduceAcc + num::Float,
    D: DimAPI,
{
    type TOut = T;

    fn max_all(&self, a: &CudaBuffer<T>, la: &Layout<D>) -> Result<T> {
        if la.size() == 0 {
            rstsr_raise!(InvalidValue, "zero-size array is not supported for max")?;
        }
        reduce_all_impl(self, UnaryOp::Max, T::SPEC, a, &la.to_dim::<IxD>()?, T::zero())
    }

    fn max_axes(
        &self,
        a: &CudaBuffer<T>,
        la: &Layout<D>,
        axes: &[isize],
    ) -> Result<
        (Storage<DataOwned<<Self as DeviceRawAPI<Self::TOut>>::Raw>, Self::TOut, Self>, Layout<IxD>),
    > {
        if la.size() == 0 {
            rstsr_raise!(InvalidValue, "zero-size array is not supported for max")?;
        }
        reduce_axes_impl(self, UnaryOp::Max, T::SPEC, a, &la.to_dim::<IxD>()?, axes, T::zero())
    }
}

// --- explicit-output-dtype variants -------------------------------------------------------------
//
// The same kernel with a wider accumulator: `@TACC@` is `TOut`, and the fold casts each element into
// it, which is what `into_cast` does on the CPU. The cast happens in the kernel, so the host never
// touches a value — hence no `Add`/`Mul`/`Clone` bound on `T`, and `T` need not implement
// `ReduceAcc`: only the accumulator type does.

impl<T, TOut, D> OpSumDtypeAPI<T, TOut, D> for DeviceCuda
where
    Self: DeviceRawAPI<T, Raw = CudaBuffer<T>>
        + DeviceRawAPI<TOut, Raw = CudaBuffer<TOut>>
        + DeviceRawAPI<MaybeUninit<TOut>, Raw = CudaBuffer<MaybeUninit<TOut>>>
        + DeviceCreationAnyAPI<TOut>,
    T: CudaDType,
    TOut: CudaDType + ReduceAcc + num::Zero,
    D: DimAPI,
{
    fn sum_axes_dtype(
        &self,
        a: &CudaBuffer<T>,
        la: &Layout<D>,
        axes: &[isize],
    ) -> Result<(Storage<DataOwned<<Self as DeviceRawAPI<TOut>>::Raw>, TOut, Self>, Layout<IxD>)> {
        reduce_axes_impl(self, UnaryOp::Sum, TOut::SPEC, a, &la.to_dim::<IxD>()?, axes, TOut::zero())
    }
}

impl<T, TOut, D> OpProdDtypeAPI<T, TOut, D> for DeviceCuda
where
    Self: DeviceRawAPI<T, Raw = CudaBuffer<T>>
        + DeviceRawAPI<TOut, Raw = CudaBuffer<TOut>>
        + DeviceRawAPI<MaybeUninit<TOut>, Raw = CudaBuffer<MaybeUninit<TOut>>>
        + DeviceCreationAnyAPI<TOut>,
    T: CudaDType,
    TOut: CudaDType + ReduceAcc + num::One,
    D: DimAPI,
{
    fn prod_axes_dtype(
        &self,
        a: &CudaBuffer<T>,
        la: &Layout<D>,
        axes: &[isize],
    ) -> Result<(Storage<DataOwned<<Self as DeviceRawAPI<TOut>>::Raw>, TOut, Self>, Layout<IxD>)> {
        reduce_axes_impl(self, UnaryOp::Prod, TOut::SPEC, a, &la.to_dim::<IxD>()?, axes, TOut::one())
    }
}

impl<T, TOut, D> OpMeanDtypeAPI<T, TOut, D> for DeviceCuda
where
    Self: DeviceRawAPI<T, Raw = CudaBuffer<T>>
        + DeviceRawAPI<TOut, Raw = CudaBuffer<TOut>>
        + DeviceRawAPI<MaybeUninit<TOut>, Raw = CudaBuffer<MaybeUninit<TOut>>>
        + DeviceCreationAnyAPI<TOut>,
    T: CudaDType,
    TOut: CudaDType + ReduceAcc + num::Float,
    D: DimAPI,
{
    fn mean_axes_dtype(
        &self,
        a: &CudaBuffer<T>,
        la: &Layout<D>,
        axes: &[isize],
    ) -> Result<(Storage<DataOwned<<Self as DeviceRawAPI<TOut>>::Raw>, TOut, Self>, Layout<IxD>)> {
        reduce_axes_impl(self, UnaryOp::Mean, TOut::SPEC, a, &la.to_dim::<IxD>()?, axes, TOut::nan())
    }
}
