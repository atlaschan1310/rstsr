//! Host-side plan selection for reduction kernels.
//!
//! A GPU reduction must commit to a thread mapping *before* it launches, and the only input to that
//! decision is shapes and strides. [`classify_reduce`] turns those into a [`ReducePlan`], or refuses
//! with a named error.
//!
//! # Two knobs, one equation
//!
//! The geometry is two numbers:
//!
//! - `lanes` — threads cooperating on **one** output element. `1` means each thread owns an output
//!   and there is no cross-thread combination at all.
//! - `outputs_per_block` — how many outputs share a block; `lanes * outputs_per_block == block`.
//!
//! They come from one expression, with no branch between the two regimes:
//!
//! ```text
//! lanes = max(free, need).next_power_of_two().min(cap)
//! ```
//!
//! - `free` — what coalescing wants and gets for nothing. Lanes on a unit-stride contraction cost no
//!   extra transactions, so take up to a warp; on a strided contraction the coalesced choice is one
//!   lane, so `free = 1`.
//! - `need` — the lanes that would lift the grid to [`MIN_BLOCKS`]. Since
//!   `grid = ceil(n_out * lanes / BLOCK)`, lanes per output are the *only* occupancy lever short of
//!   splitting the contraction across blocks.
//! - `cap` — what the contraction can actually fill.
//!
//! `need <= free` exactly when the grid is already adequate, so `max` never escalates when it does
//! not have to, and `lanes = 1` (thread-per-output) is the fixed point rather than a branch.
//!
//! **The lanes sit on the unit-stride axis, in the low thread bits**, so consecutive lanes are
//! consecutive addresses *and* co-lanes share a warp. That second half is what keeps the tree a
//! `__shfl_down_sync` chain while `lanes <= 32`.
//!
//! # One linear run per part
//!
//! Both the contracted part and the output part are reduced to a single number: the element step per
//! unit of the index. Two tests, because the two parts have different obligations: [`block_step`]
//! asks only that the contracted axes form one contiguous block *as a set* (axis order is free — no
//! summation order is promised), while [`output_step`] additionally requires shape order, because
//! `c[j]` counts in shape order.
//!
//! That one test replaces the previous per-axis contiguity-prefix intersection, and it is what lets
//! multi-axis contractions work at all: `(p, i, j)` contracted over `(i, j)` yields a `[i, j]` part
//! with strides `[j, 1]`, step 1, and the kernel walks it with `m` exactly as it walks a single axis.
//! A rank-0 output — a full reduction — is the same story with `n_out = 1` and no step.
//!
//! # Lanes above a warp are allowed, and cost something
//!
//! [`WARP_LANES`] is the *free* ceiling, not the hard one. Past 32 the partial sums of one output
//! span several warps, so the kernel needs a shared-memory stage and a `__syncthreads()` — roughly
//! one round trip, fixed. Worth paying when the alternative is a starved grid.
//!
//! # Reproducibility
//!
//! `lanes` and `outputs_per_block` are derived from shapes only — never from the driver, the device
//! or an occupancy query. The summation tree is a function of the input shape and the binary, which
//! keeps results reproducible; it is *not* a function of how many SMs happen to be free.
//!
//! This module is **host-only**. No CUDA, no NVRTC, no device — so every decision here is testable on
//! a machine with no GPU.

use crate::prelude_dev::*;

/// Threads per block for reduction kernels.
///
/// Must stay a **compile-time constant**: sizing it with `cudaOccupancyMaxPotentialBlockSize` would
/// make the summation tree depend on the driver and device.
pub(crate) const BLOCK: usize = 256;

/// Lanes per output that keep the summation tree inside one warp.
///
/// Not a limit — the point past which a second reduction stage appears. See [`MAX_LANES`].
pub(crate) const WARP_LANES: u32 = 32;

/// Hard ceiling on lanes per output.
///
/// A whole block, because a block is the largest unit that can cooperate without a second kernel.
/// Above 32 needs the shared-memory stage, which is why it is only reached when the grid demands it.
pub(crate) const MAX_LANES: u32 = BLOCK as u32;

/// Block count the plan aims for before it stops spending lanes on occupancy.
///
/// Deliberately a *policy* number rather than a device query, so the plan stays a pure function of
/// shapes and testable without a GPU. 512 blocks gives a discrete GPU several blocks per SM, so a
/// latency-hiding wave always exists.
pub(crate) const MIN_BLOCKS: usize = 512;

/// A fully specified launch geometry. Counts and strides only — the launch itself is the caller's
/// job.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ReducePlan {
    /// Threads cooperating on one output. Power of two, `1..=MAX_LANES`.
    pub(crate) lanes: u32,
    /// Outputs per block; `lanes * outputs_per_block == block`.
    pub(crate) outputs_per_block: u32,
    /// Threads per block.
    pub(crate) block: u32,
    /// Contracted length walked by each lane group, and its element step per contracted index.
    pub(crate) n_contract: usize,
    pub(crate) step_a: isize,
    pub(crate) step_b: isize,
    /// Output count, and the element step per output index in each operand. `0` when the output is
    /// rank 0 — there is only one output, so the step is never taken.
    pub(crate) n_out: usize,
    pub(crate) out_step_a: isize,
    pub(crate) out_step_b: isize,
}

impl ReducePlan {
    /// Blocks to launch.
    pub(crate) fn grid_blocks(&self) -> usize {
        let opb = self.outputs_per_block as usize;
        (self.n_out + opb - 1) / opb
    }
}

/// Element step per unit of the **contracted** index, when those axes form one contiguous block as a
/// set — in *any* axis order.
///
/// Order is free here. A reduction makes no claim about summation order (see `reduce.rs`), so walking
/// a column-major `[i, j]` block as `m = 0..i*j` with step 1 reads every element exactly once; only
/// the accumulation order differs, and that is not part of the contract. So the test is "are the axes
/// sorted by `|stride|` an arithmetic progression", not "are they row-major".
///
/// The step reported is the innermost one, so the block may be non-unit (`[64, 2]` over `[8, 8]`) —
/// the caller only gets to assume *constant* stride, not unit stride.
///
/// `None` when the block has a gap (strides `[16, 1]` over `[8, 8]`), a repeated stride, or when the
/// axes disagree about direction.
fn block_step(l: &Layout<IxD>) -> Option<isize> {
    let n = l.ndim();
    if n == 0 {
        return None;
    }
    // Innermost axis first, by absolute stride; the *order* of the axes is exactly what we are
    // allowed to ignore, so sorting is the whole point rather than a convenience.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by_key(|&k| l.stride()[k].unsigned_abs());

    let step = l.stride()[order[0]];
    let mut expect = step;
    for &k in &order {
        if l.stride()[k] != expect {
            return None;
        }
        expect *= l.shape()[k] as isize;
    }
    Some(step)
}

/// Element step per unit of the **output** index, when the axes are a row-major block in shape order.
///
/// Order is *not* free here. The kernel writes `c[j]` with `j` counting in shape order, into a fresh
/// layout that is contiguous in the device's keep order, so `a[j] = base + j * step` addresses the
/// right element only if the operand's output axes advance in that same order. A column-major output
/// block would need `c[perm(j)]`, a reordering we do not perform — so it is refused rather than
/// silently transposed.
///
/// `None` for a rank-0 layout (nothing to walk) and for anything with a gap or a permuted axis.
fn output_step(l: &Layout<IxD>) -> Option<isize> {
    let n = l.ndim();
    if n == 0 {
        return None;
    }
    let step = l.stride()[n - 1];
    let mut expect = step;
    for k in (0..n).rev() {
        if l.stride()[k] != expect {
            return None;
        }
        expect *= l.shape()[k] as isize;
    }
    Some(step)
}

/// Decide how to decompose a contraction or a reduction over `axes`.
///
/// `la` / `lb` are the two operands of a contraction; a unary reduction passes the same layout for
/// both, so `step_b` / `out_step_b` come out equal to their `a` counterparts and are simply unused.
/// `axes_a` / `axes_b` are the contracted axes in the normalised `isize` form the device trait takes.
/// `order` is the device's default order, needed because the output layout is aligned against the
/// operands' remaining axes before the parts are tested.
pub(crate) fn classify_reduce(
    la: &Layout<IxD>,
    lb: &Layout<IxD>,
    lc: &Layout<IxD>,
    axes_a: &[isize],
    axes_b: &[isize],
    order: FlagOrder,
) -> Result<ReducePlan> {
    let (las, lam) = la.dim_split_axes(axes_a)?;
    let (lbs, lbm) = lb.dim_split_axes(axes_b)?;

    // The contracted shapes must agree. This is also what validates the dispatch assumptions
    // downstream, so it is a hard error rather than a refusal.
    rstsr_assert_eq!(
        las.shape(),
        lbs.shape(),
        InvalidLayout,
        "the dimensions of a and b along the contracted axis should be the same"
    )?;

    // Align the output against the operands' remaining axes (inserts stride-0 broadcast axes where
    // the output is lower-rank). The equality check verifies that `lc` really is a broadcast of the
    // remaining part, i.e. that this contraction is well formed.
    let (lc_aligned, lam) = broadcast_layout_to_first(lc, &lam, order)?;
    let (lc_aligned, lbm) = broadcast_layout_to_first(&lc_aligned, &lbm, order)?;
    rstsr_assert_eq!(lc_aligned, *lc, InvalidLayout, "layout of c is not a broadcast of a/b after axis sum")?;

    let n_contract = las.size();
    let n_out = lc.size();

    // The output must be walkable as `c[j]`: a row-major block whose innermost step is 1. A rank-0
    // output is the single-element case and needs no step at all.
    let c_ok = if lc.ndim() == 0 { n_out == 1 } else { output_step(&lc) == Some(1) };
    if !c_ok {
        rstsr_raise!(
            InvalidLayout,
            "cuda reduce: output must be a contiguous run with unit step (rank {}, strides {:?})",
            lc.ndim(),
            lc.stride()
        )?;
    }

    // Empty contraction or output: a zero-block launch. `next_power_of_two` maps 0 to 1, so the
    // arithmetic below stays well defined without a special case.
    if n_contract == 0 || n_out == 0 {
        return Ok(ReducePlan {
            lanes: 1,
            outputs_per_block: BLOCK as u32,
            block: BLOCK as u32,
            n_contract,
            step_a: 1,
            step_b: 1,
            n_out,
            out_step_a: 0,
            out_step_b: 0,
        });
    }

    // --- one linear run per part ----------------------------------------------------------------
    // A rank-0 contracted part is *one element* — `axes` was empty, or the operand is 0-d — so there
    // is nothing to walk and any step is correct. This is `sum(t, &[])`, which reduces to a copy, and
    // `sum` of a 0-d tensor. Without this, `block_step` answers "nothing to walk" and every such call
    // is refused.
    let step_a = if las.ndim() == 0 {
        1
    } else {
        let Some(step) = block_step(&las) else {
            rstsr_raise!(
                UnImplemented,
                "cuda reduce: contracted part is not one contiguous block (shape {:?}, strides {:?})",
                las.shape(),
                las.stride()
            )?
        };
        step
    };
    let step_b = if lbs.ndim() == 0 {
        1
    } else {
        let Some(step) = block_step(&lbs) else {
            rstsr_raise!(
                UnImplemented,
                "cuda reduce: contracted part of b is not one contiguous block (shape {:?}, strides {:?})",
                lbs.shape(),
                lbs.stride()
            )?
        };
        step
    };

    // A rank-0 output is one element; its step is never taken, and reporting 0 keeps the kernel's
    // address arithmetic well defined rather than something that could overflow.
    let (out_step_a, out_step_b) = if lam.ndim() == 0 && lbm.ndim() == 0 {
        (0, 0)
    } else {
        let Some(a) = output_step(&lam) else {
            rstsr_raise!(
                UnImplemented,
                "cuda reduce: output part is not a row-major run in shape order (shape {:?}, strides {:?})",
                lam.shape(),
                lam.stride()
            )?
        };
        let Some(b) = output_step(&lbm) else {
            rstsr_raise!(
                UnImplemented,
                "cuda reduce: output part of b is not a row-major run in shape order (shape {:?}, strides {:?})",
                lbm.shape(),
                lbm.stride()
            )?
        };
        (a, b)
    };

    // Does the contraction stream unit-stride (so lanes are free), and does the output stream
    // unit-stride (so one thread per output is coalesced)?
    let con_is_run = step_a == 1 && step_b == 1;
    let out_is_run = if lam.ndim() == 0 { n_out == 1 } else { out_step_a == 1 && out_step_b == 1 };

    // Neither: no thread mapping is coalesced. Packing would fix it; there is no packing pass yet,
    // so refuse rather than launch something known to be bad.
    if !con_is_run && !out_is_run {
        rstsr_raise!(
            UnImplemented,
            "cuda reduce: no unit-stride run on either the contracted or the output axis \
             (steps {step_a}/{step_b} in, {out_step_a}/{out_step_b} out)"
        )?;
    }

    // --- lanes, with no branch ------------------------------------------------------------------
    //
    // `need >= free` exactly when the grid is short, so the `max` alone decides the regime.
    let lanes_free = if con_is_run { n_contract.next_power_of_two().min(WARP_LANES as usize) } else { 1 };
    let lanes_cap = n_contract.next_power_of_two().min(MAX_LANES as usize);
    let lanes_need = (MIN_BLOCKS * BLOCK).div_ceil(n_out);

    let lanes = lanes_free.max(lanes_need).next_power_of_two().min(lanes_cap) as u32;

    Ok(ReducePlan {
        lanes,
        outputs_per_block: BLOCK as u32 / lanes,
        block: BLOCK as u32,
        n_contract,
        step_a,
        step_b,
        n_out,
        out_step_a,
        out_step_b,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a layout, panicking if RSTSR rejects it.
    ///
    /// The constructor's no-overlap check is stricter than "no two elements collide": walking the
    /// axes in increasing `|stride|`, it keeps `elem_cum = max_index_seen + 1` and requires
    /// `elem_cum < next_stride` (`layoutbase.rs:337`). So a *gapped* block is legal only if the next
    /// axis's stride clears the block's whole span, holes included — 64 elements laid out as eight
    /// 8-element runs 16 apart span 120 slots, and the axis above them needs stride ≥ 120, not 64.
    fn layout(shape: &[usize], stride: &[isize], offset: usize) -> Layout<IxD> {
        Layout::<IxD>::new(shape.to_vec().into(), stride.to_vec().into(), offset).unwrap()
    }

    /// C-order 2-D operand layout plus a contiguous 1-D output layout of `n_out`.
    fn pair(shape: [usize; 2], n_out: usize) -> (Layout<IxD>, Layout<IxD>) {
        (layout(&shape, &[shape[1] as isize, 1], 0), layout(&[n_out], &[1], 0))
    }

    fn geometry(plan: ReducePlan) -> (u32, u32, usize) {
        (plan.lanes, plan.outputs_per_block, plan.grid_blocks())
    }

    /// The contracted part only has to be a contiguous block *as a set*: axis order is free, because
    /// no summation order is promised.
    #[test]
    fn block_step_ignores_axis_order() {
        // one axis, any stride
        assert_eq!(block_step(&layout(&[64], &[1], 0)), Some(1));
        assert_eq!(block_step(&layout(&[64], &[1000], 0)), Some(1000));
        // row-major block
        assert_eq!(block_step(&layout(&[8, 8], &[8, 1], 0)), Some(1));
        // constant but not unit
        assert_eq!(block_step(&layout(&[8, 8], &[16, 2], 0)), Some(2));
        // column-major block: same element set, enumerated the other way round — admissible
        assert_eq!(block_step(&layout(&[8, 8], &[1, 8], 0)), Some(1));
        // a gap between the two runs: not one block
        assert_eq!(block_step(&layout(&[8, 8], &[16, 1], 0)), None);
        // a hole inside the outer axis
        assert_eq!(block_step(&layout(&[8, 8], &[9, 1], 0)), None);
        // rank 0 has nothing to walk
        assert_eq!(block_step(&layout(&[], &[], 0)), None);
    }

    /// The output part does require shape order: `c[j]` counts in shape order, so a column-major
    /// output block would need `c[perm(j)]`, which we do not do.
    #[test]
    fn output_step_requires_shape_order() {
        assert_eq!(output_step(&layout(&[64], &[1], 0)), Some(1));
        assert_eq!(output_step(&layout(&[64], &[1000], 0)), Some(1000));
        assert_eq!(output_step(&layout(&[8, 8], &[8, 1], 0)), Some(1));
        assert_eq!(output_step(&layout(&[8, 8], &[16, 2], 0)), Some(2));
        // column-major: refused here, admitted by `block_step`
        assert_eq!(output_step(&layout(&[8, 8], &[1, 8], 0)), None);
        assert_eq!(output_step(&layout(&[8, 8], &[9, 1], 0)), None);
        assert_eq!(output_step(&layout(&[], &[], 0)), None);
    }

    /// Contraction is the last axis, so it is the unit-stride one. `n_out = 1000` is far short of the
    /// grid target, so lanes escalate past a warp: 1000 blocks at one output each.
    #[test]
    fn contracted_run_escalates_lanes_to_fill_the_grid() {
        let (n_p, n_i) = (1000usize, 10240usize);
        let (la, lc) = pair([n_p, n_i], n_p);
        let plan = classify_reduce(&la, &la, &lc, &[1], &[1], RowMajor).unwrap();

        assert_eq!(geometry(plan), (256, 1, 1000));
        assert_eq!(plan.step_a, 1, "contracted axis is unit-stride");
        assert_eq!(plan.out_step_a, n_i as isize, "output axis is strided");
    }

    /// Wide output, strided contraction: 782 blocks are already enough, so `need` collapses to 1 and
    /// the plan is thread-per-output.
    #[test]
    fn wide_output_is_thread_per_output() {
        let (n_i, n_p) = (1024usize, 200_000usize);
        let (la, lc) = pair([n_i, n_p], n_p);
        let plan = classify_reduce(&la, &la, &lc, &[0], &[0], RowMajor).unwrap();

        assert_eq!(geometry(plan), (1, 256, 782));
        assert_eq!(plan.step_a, n_p as isize, "contracted axis is strided");
        assert_eq!(plan.out_step_a, 1, "output axis is unit-stride, so the warp is contiguous");
    }

    /// Starved output: 4 blocks thread-per-output, so lanes go on the *strided* contraction and buy
    /// 1000 blocks at 4x the transactions.
    #[test]
    fn starved_output_prefers_lanes() {
        let (n_i, n_p) = (10240usize, 1000usize);
        let (la, lc) = pair([n_i, n_p], n_p);
        let plan = classify_reduce(&la, &la, &lc, &[0], &[0], RowMajor).unwrap();
        assert_eq!(geometry(plan), (256, 1, 1000));
    }

    /// Two contracted axes that form one row-major block. This is what `sum(axis=(i,j))` and a
    /// two-axis `vecdot` both produce, and it was refused before the linear-run test.
    #[test]
    fn multi_axis_contraction_is_one_run() {
        // (p, i, j) C-order, contracting (i, j) => contracted part [i, j] with strides [j, 1].
        let (p, i, j) = (4usize, 8usize, 8usize);
        let la = layout(&[p, i, j], &[(i * j) as isize, j as isize, 1], 0);
        let lc = layout(&[p], &[1], 0);

        let plan = classify_reduce(&la, &la, &lc, &[1, 2], &[1, 2], RowMajor).unwrap();
        assert_eq!(plan.n_contract, i * j);
        assert_eq!(plan.n_out, p);
        assert_eq!(plan.step_a, 1, "the [i, j] block is row-major, so m walks it at stride 1");
        assert_eq!(plan.out_step_a, (i * j) as isize);
        assert_eq!(geometry(plan), (64, 4, 1));
    }

    /// Two output axes that form one row-major block: contract the leading axis of a rank-3 tensor.
    #[test]
    fn multi_axis_output_is_one_run() {
        let (r, a, b) = (4usize, 8usize, 8usize);
        let la = layout(&[r, a, b], &[(a * b) as isize, b as isize, 1], 0);
        let lc = layout(&[a, b], &[b as isize, 1], 0);

        let plan = classify_reduce(&la, &la, &lc, &[0], &[0], RowMajor).unwrap();
        assert_eq!(plan.n_contract, r);
        assert_eq!(plan.n_out, a * b);
        assert_eq!(plan.out_step_a, 1, "the [a, b] output block is contiguous");
        assert_eq!(plan.step_a, (a * b) as isize, "the contracted axis is strided");
    }

    /// A full reduction: rank-0 output, one element, one block, everything cooperating.
    #[test]
    fn rank_zero_output_is_one_block() {
        let la = layout(&[100, 1024], &[1024, 1], 0);
        let lc = layout(&[], &[], 0);

        let plan = classify_reduce(&la, &la, &lc, &[0, 1], &[0, 1], RowMajor).unwrap();
        assert_eq!(plan.n_out, 1);
        assert_eq!(plan.n_contract, 100 * 1024);
        assert_eq!(plan.out_step_a, 0, "a single output never takes a step");
        assert_eq!(geometry(plan), (256, 1, 1));
    }

    /// `axes == []` and 0-d operands: the contracted part is one element, so the plan is a copy.
    /// `sum(t, &[])` and `sum` of a 0-d tensor both land here, and both used to be refused because
    /// `block_step` answers "nothing to walk" for a rank-0 layout.
    #[test]
    fn empty_axes_is_a_copy() {
        let la = layout(&[5], &[1], 0);
        let lc = layout(&[5], &[1], 0);
        let plan = classify_reduce(&la, &la, &lc, &[], &[], RowMajor).unwrap();
        assert_eq!((plan.n_contract, plan.n_out), (1, 5));
        assert_eq!(plan.lanes, 1, "one element per output: nothing to share");
        assert_eq!(plan.out_step_a, 1);
        assert_eq!(plan.grid_blocks(), 1);

        // 0-d operand: one element in, one element out.
        let l0 = layout(&[], &[], 0);
        let plan = classify_reduce(&l0, &l0, &l0, &[], &[], RowMajor).unwrap();
        assert_eq!((plan.n_contract, plan.n_out), (1, 1));
        assert_eq!(plan.out_step_a, 0, "a single output never takes a step");
        assert_eq!(plan.grid_blocks(), 1);
    }

    /// A column-major *contracted* block is admitted: the element set is contiguous, and no summation
    /// order is promised, so enumerating it in stride order reads exactly the same elements.
    #[test]
    fn column_major_contraction_is_admitted() {
        // i and j transposed in memory: element (p, i, j) lives at p*64 + i + j*8, so each output's
        // 64 contracted elements are still one contiguous run.
        let la = layout(&[4, 8, 8], &[64, 1, 8], 0);
        let lc = layout(&[4], &[1], 0);
        let plan = classify_reduce(&la, &la, &lc, &[1, 2], &[1, 2], RowMajor).unwrap();
        assert_eq!(plan.n_contract, 64);
        assert_eq!(plan.step_a, 1, "contiguous block; only the axis order differs");
    }

    /// A contracted block with a *gap* cannot be walked with one step: refused, because guessing
    /// would silently read the wrong elements.
    #[test]
    fn gapped_contraction_is_refused() {
        // Eight 8-element runs, 16 apart: the 64 contracted elements span 120 slots with 56 holes,
        // so the output axis must clear 120 (hence 128, not 64) or the layout is rejected as
        // overlapping before `classify_reduce` ever sees it.
        let la = layout(&[4, 8, 8], &[128, 16, 1], 0);
        let lc = layout(&[4], &[1], 0);
        let e = classify_reduce(&la, &la, &lc, &[1, 2], &[1, 2], RowMajor).unwrap_err();
        let msg = format!("{e}");
        assert!(msg.contains("not one contiguous block"), "got: {msg}");
    }

    /// No unit-stride step anywhere: correct but uncoalesced, so refused until packing exists.
    #[test]
    fn stepped_on_both_sides_is_refused() {
        let la = layout(&[64, 512], &[1024, 2], 0);
        let lc = layout(&[64], &[1], 0);
        let e = classify_reduce(&la, &la, &lc, &[1], &[1], RowMajor).unwrap_err();
        let msg = format!("{e}");
        assert!(msg.contains("no unit-stride run"), "got: {msg}");
        assert!(msg.contains("UnImplemented"), "got: {msg}");
    }

    /// A non-contiguous output is a caller defect, reported as such rather than as a missing kernel.
    #[test]
    fn non_contiguous_output_is_a_layout_error() {
        let la = layout(&[64, 512], &[1024, 1], 0);
        let lc = layout(&[64], &[2], 0);
        let e = classify_reduce(&la, &la, &lc, &[1], &[1], RowMajor).unwrap_err();
        let msg = format!("{e}");
        assert!(msg.contains("output must be a contiguous run"), "got: {msg}");
        assert!(msg.contains("InvalidLayout"), "got: {msg}");
    }

    /// Short contractions shrink `lanes` instead of falling out of the plan.
    #[test]
    fn short_contraction_shrinks_lanes() {
        let (n_p, n_i) = (1_000_000usize, 8usize);
        let (la, lc) = pair([n_p, n_i], n_p);
        assert_eq!(geometry(classify_reduce(&la, &la, &lc, &[1], &[1], RowMajor).unwrap()), (8, 32, 31_250));

        let (n_p, n_i) = (100usize, 3usize);
        let (la, lc) = pair([n_p, n_i], n_p);
        assert_eq!(geometry(classify_reduce(&la, &la, &lc, &[1], &[1], RowMajor).unwrap()), (4, 64, 2));

        let (n_p, n_i) = (4096usize, 1usize);
        let (la, lc) = pair([n_p, n_i], n_p);
        assert_eq!(geometry(classify_reduce(&la, &la, &lc, &[1], &[1], RowMajor).unwrap()), (1, 256, 16));
    }

    /// The shapes the GPU integration tests use, one per lane count. Each lane count is a separately
    /// compiled kernel (`@LANES@` is a token), so this is the host half of that contract: it fails
    /// before anyone rents a GPU. Keep in step with `VECDOT_CASES` in `tests/test_kernels.rs`.
    #[test]
    fn one_gpu_case_per_lane_count() {
        let cases: &[(usize, usize, isize, u32)] = &[
            (2, 131_072, 0, 1),
            (8, 100_000, 0, 2),
            (20_000, 3, 1, 4),
            (20_000, 8, 1, 8),
            (100, 10_024, 0, 16),
            (20_000, 32, 1, 32),
            (3_000, 64, 1, 64),
            (1_500, 128, 1, 128),
            (500, 1_024, 1, 256),
        ];

        for &(rows, cols, axis, want_lanes) in cases {
            let la = layout(&[rows, cols], &[cols as isize, 1], 0);
            let n_out = if axis == 1 { rows } else { cols };
            let lc = layout(&[n_out], &[1], 0);

            let plan = classify_reduce(&la, &la, &lc, &[axis], &[axis], RowMajor).unwrap();
            assert_eq!(
                plan.lanes, want_lanes,
                "({rows} x {cols}, axis {axis}): the GPU test for lanes={want_lanes} would instead \
                 run the lanes={} kernel",
                plan.lanes
            );
        }
    }

    /// The plan must stay a legal geometry for every shape it accepts.
    #[test]
    fn plan_invariants_hold() {
        let cases: &[(usize, usize)] = &[
            (1, 1),
            (1, 1000),
            (8, 8),
            (8, 10_000),
            (64, 64),
            (1000, 10240),
            (1023, 11),
            (4096, 37),
            (10_000, 999),
        ];

        for &(n_contract, n_out) in cases {
            for contract_last in [true, false] {
                let (shape, axis) =
                    if contract_last { ([n_out, n_contract], 1isize) } else { ([n_contract, n_out], 0isize) };
                let la = layout(&shape, &[shape[1] as isize, 1], 0);
                let lc = layout(&[n_out], &[1], 0);

                let plan = classify_reduce(&la, &la, &lc, &[axis], &[axis], RowMajor).unwrap_or_else(|e| {
                    panic!("({n_contract}, {n_out}, last={contract_last}) should be plannable: {e}")
                });

                assert!(plan.lanes.is_power_of_two(), "lanes must be a power of two: {}", plan.lanes);
                assert!(plan.lanes <= MAX_LANES, "lanes capped at {MAX_LANES}: {}", plan.lanes);
                assert_eq!(plan.block % plan.lanes, 0, "block must divide by lanes");
                assert_eq!(plan.lanes * plan.outputs_per_block, plan.block);
                assert_eq!(plan.n_contract, n_contract);
                assert_eq!(plan.n_out, n_out);

                let blocks = plan.grid_blocks();
                assert!(
                    blocks * plan.outputs_per_block as usize >= n_out,
                    "({n_contract}, {n_out}): {blocks} blocks of {} do not cover {n_out} outputs",
                    plan.outputs_per_block
                );
            }
        }
    }
}
