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
//! `lanes > 1` is the old `ContigS` shape and `lanes == 1` is the old `ContigM` shape, but they are
//! the two ends of one parameterised kernel rather than two kernels, and they are *selected by the
//! same expression* — there is no branch between them:
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
//! not have to, and the old step function (`n_out` one below the threshold → 1 lane, one above → 32)
//! is gone: `lanes` tracks `MIN_BLOCKS * BLOCK / n_out` continuously.
//!
//! The CPU's split is inherited from a different question ("is there a slice to hand to
//! `unrolled_binary_reduce`?"). On the GPU the questions are only: which axis is unit-stride, and is
//! there enough work to fill a machine.
//!
//! **The lanes sit on the unit-stride axis, in the low thread bits**, so consecutive lanes are
//! consecutive addresses *and* co-lanes share a warp. That second half is what keeps the tree a
//! `__shfl_down_sync` chain while `lanes <= 32`; CuPy puts the output index in the low bits instead
//! and pays for it with a shared-memory tree in every reduction it runs.
//!
//! # Lanes above a warp are allowed, and cost something
//!
//! [`WARP_LANES`] is the *free* ceiling, not the hard one. Past 32 the partial sums of one output
//! span several warps, so the kernel needs a shared-memory stage and a `__syncthreads()` — roughly
//! one round trip, fixed. That is worth paying when the alternative is a starved grid: at
//! `n_out = 1000` with a long contraction, 32 lanes give 125 blocks (1.2 per SM) and 256 lanes give
//! 1000 blocks. The tree costs tens of cycles; the occupancy difference is several-fold bandwidth.
//!
//! So [`MAX_LANES`] is a whole block and the kernel carries both stages. `lanes` is substituted at
//! NVRTC time (`@LANES@`), so the stage is compiled out whenever `lanes <= 32`.
//!
//! The one term this deliberately does not model: with a strided contraction, a large `lanes` drives
//! `outputs_per_block` down, and neighbouring outputs are what share a 32 B sector. At
//! `lanes = BLOCK` each block owns one output, so sector sharing moves from L1 to L2. DRAM traffic is
//! unchanged either way and occupancy dominates in the starved regime — but this is the term to
//! revisit once there are numbers.
//!
//! # What is refused
//!
//! Only shapes the kernel set can address at all are planned. Everything else raises by name:
//!
//! - a multi-axis contracted or output part (the kernel walks `m` and writes `c[j]` with a single
//!   step each) — `UnImplemented`;
//! - an output that is not contiguous — `InvalidLayout`, since that is a caller defect rather than a
//!   missing kernel;
//! - no unit-stride run on either axis — `UnImplemented`. Packing a contiguous copy and re-planning
//!   is the eventual answer here; there is no plan for it yet, and pretending otherwise would hide
//!   the refusal behind a slow path.
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
    /// Contracted length walked by each lane group, and its element stride in each operand.
    pub(crate) n_contract: usize,
    pub(crate) step_a: isize,
    pub(crate) step_b: isize,
    /// Output count, and the element stride of the (single) output axis in each operand.
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

/// Longest run of axes contiguous in *both* operand lists, taken from the front.
///
/// `get_axes_composition` returns the contiguous axes as a prefix built by walking `|stride|` up
/// from 1, so a common prefix across two operands is a run both can stream together. Semantics
/// copied from the CPU's `flag_contig_s` (`cpu_serial/vecdot.rs:46-54`) — they are load-bearing.
fn shared_prefix_2(a: &[usize], b: &[usize]) -> Vec<usize> {
    let mut out = Vec::new();
    for (ia, ib) in izip!(a.iter(), b.iter()) {
        if ia == ib {
            out.push(*ia);
        } else {
            break;
        }
    }
    out
}

/// Same, for the three-way intersection the output side needs (`a`, `b` and `c`).
fn shared_prefix_3(a: &[usize], b: &[usize], c: &[usize]) -> Vec<usize> {
    let mut out = Vec::new();
    for (ia, ib, ic) in izip!(a.iter(), b.iter(), c.iter()) {
        if ia == ib && ib == ic {
            out.push(*ia);
        } else {
            break;
        }
    }
    out
}

/// Decide how to decompose `c[out] = sum_contract a * b`.
///
/// `axes_a` / `axes_b` are the contracted axes in the normalised `isize` form the device trait
/// takes. `order` is the device's default order, needed because the output layout is aligned against
/// the operands' remaining axes before the runs are compared.
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

    // --- admissibility -------------------------------------------------------------------------
    //
    // One axis per part, because the kernel walks `m` and writes `c[j]` with a single step each.
    // Collapsing several axes into a run is the next increment: the run lengths below are already
    // computed correctly for it, only the linear index arithmetic is missing.
    if las.ndim() != 1 || lbs.ndim() != 1 {
        rstsr_raise!(
            UnImplemented,
            "cuda vecdot: contracted part is not a single axis (ranks {} and {})",
            las.ndim(),
            lbs.ndim()
        )?;
    }
    if lam.ndim() != 1 || lbm.ndim() != 1 {
        rstsr_raise!(
            UnImplemented,
            "cuda vecdot: output part is not a single axis (ranks {} and {})",
            lam.ndim(),
            lbm.ndim()
        )?;
    }
    // Checked after the rank tests but before the layout analysis: a strided output layout is a
    // defect of the *output*, and reporting "no unit-stride run" for it would blame the operand
    // strides instead.
    if lc.ndim() != 1 || lc.stride()[0] != 1 {
        rstsr_raise!(InvalidLayout, "cuda vecdot: output must be contiguous (rank {}, stride {:?})", lc.ndim(), lc.stride())?;
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
            out_step_a: 1,
            out_step_b: 1,
        });
    }

    let step_a = las.stride()[0];
    let step_b = lbs.stride()[0];
    let out_step_a = lam.stride()[0];
    let out_step_b = lbm.stride()[0];

    // --- which axis can carry the lanes ---------------------------------------------------------
    let (_, _, asc_a, _) = get_axes_composition(&las);
    let (_, _, asc_b, _) = get_axes_composition(&lbs);
    let (_, _, amc_a, _) = get_axes_composition(&lam);
    let (_, _, amc_b, _) = get_axes_composition(&lbm);
    let (_, _, amc_c, _) = get_axes_composition(&lc_aligned);

    let con_run: usize = shared_prefix_2(&asc_a, &asc_b).iter().map(|&i| las.shape()[i]).product();
    let out_run: usize = shared_prefix_3(&amc_a, &amc_b, &amc_c).iter().map(|&i| lam.shape()[i]).product();

    // A run of length 1 is trivially contiguous, which is why the products start at 1 rather than
    // the flags being guarded by `is_empty()`.
    let con_is_run = con_run == n_contract;
    let out_is_run = out_run == n_out;

    // No unit-stride axis anywhere: no thread mapping is coalesced. Packing would fix it; there is
    // no packing pass yet, so refuse rather than launch something known to be bad.
    if !con_is_run && !out_is_run {
        rstsr_raise!(
            UnImplemented,
            "cuda vecdot: no unit-stride run on either the contracted or the output axis \
             (contracted run {con_run}, output run {out_run} of {n_contract} x {n_out})"
        )?;
    }

    // --- lanes, with no branch ------------------------------------------------------------------
    //
    // `need >= free` exactly when the grid is short, so the `max` alone decides the regime: a large
    // `n_out` collapses `need` to 1 and the plan is thread-per-output; a small one drives it up
    // until the grid is full or the contraction cannot fill more lanes.
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

    fn layout(shape: &[usize], stride: &[isize], offset: usize) -> Layout<IxD> {
        // `Layout::new` takes `offset: usize`; keep this helper's signature matching it rather than
        // converting at every call site.
        Layout::<IxD>::new(shape.to_vec().into(), stride.to_vec().into(), offset).unwrap()
    }

    /// C-order 2-D operand layout plus a contiguous 1-D output layout of `n_out`.
    fn pair(shape: [usize; 2], n_out: usize) -> (Layout<IxD>, Layout<IxD>) {
        (layout(&shape, &[shape[1] as isize, 1], 0), layout(&[n_out], &[1], 0))
    }

    /// `(lanes, outputs_per_block, grid)`.
    fn geometry(plan: ReducePlan) -> (u32, u32, usize) {
        (plan.lanes, plan.outputs_per_block, plan.grid_blocks())
    }

    /// Contraction is the last axis, so it is the unit-stride one. `n_out = 1000` is far short of
    /// the grid target, so lanes escalate past a warp: 1000 blocks at one output each, not 125 at
    /// eight.
    #[test]
    fn contracted_run_escalates_lanes_to_fill_the_grid() {
        let (n_p, n_i) = (1000usize, 10240usize);
        let (la, lc) = pair([n_p, n_i], n_p);
        let plan = classify_reduce(&la, &la, &lc, &[1], &[1], RowMajor).unwrap();

        assert_eq!(geometry(plan), (256, 1, 1000));
        assert!(plan.lanes > 1, "cooperative");
        assert!(plan.lanes > WARP_LANES, "256 lanes need the cross-warp stage");
        assert_eq!(plan.step_a, 1, "contracted axis is unit-stride");
        assert_eq!(plan.out_step_a, n_i as isize, "output axis is strided");
        assert_eq!(plan.n_contract, n_i);
    }

    /// The same tensor with a wide output: 782 blocks are already enough, so `need` collapses to 1
    /// and the plan is thread-per-output — the old `ContigM`, reached as a fixed point rather than
    /// by a branch.
    #[test]
    fn wide_output_is_thread_per_output() {
        let (n_i, n_p) = (1024usize, 200_000usize);
        let (la, lc) = pair([n_i, n_p], n_p);
        let plan = classify_reduce(&la, &la, &lc, &[0], &[0], RowMajor).unwrap();

        assert_eq!(geometry(plan), (1, 256, 782));
        assert_eq!(plan.lanes, 1, "one thread per output, no tree");
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
        assert_eq!(plan.step_a, n_p as isize, "lanes are on the strided axis and that is deliberate");
    }

    /// 8 outputs. Thread-per-output would launch 8 threads; the grid target gives 8 *blocks* of 256.
    /// Splitting further is split-K's job, not the plan's.
    #[test]
    fn tiny_output_gets_one_block_per_output() {
        let (n_i, n_p) = (10024usize, 8usize);
        let (la, lc) = pair([n_i, n_p], n_p);
        let plan = classify_reduce(&la, &la, &lc, &[0], &[0], RowMajor).unwrap();
        assert_eq!(geometry(plan), (256, 1, 8));
    }

    /// Short contractions cannot fill many lanes, so `cap` binds and the lane count shrinks instead
    /// of the plan being refused.
    #[test]
    fn short_contraction_shrinks_lanes() {
        // i = 8 => 8 lanes, 32 outputs per block.
        let (n_p, n_i) = (1_000_000usize, 8usize);
        let (la, lc) = pair([n_p, n_i], n_p);
        let plan = classify_reduce(&la, &la, &lc, &[1], &[1], RowMajor).unwrap();
        assert_eq!(geometry(plan), (8, 32, 31_250));

        // i = 3 => 4 lanes, 64 outputs per block.
        let (n_p, n_i) = (100usize, 3usize);
        let (la, lc) = pair([n_p, n_i], n_p);
        let plan = classify_reduce(&la, &la, &lc, &[1], &[1], RowMajor).unwrap();
        assert_eq!(geometry(plan), (4, 64, 2));

        // i = 1 => nothing to split, so the plan is the degenerate thread-per-output.
        let (n_p, n_i) = (4096usize, 1usize);
        let (la, lc) = pair([n_p, n_i], n_p);
        let plan = classify_reduce(&la, &la, &lc, &[1], &[1], RowMajor).unwrap();
        assert_eq!(geometry(plan), (1, 256, 16));
    }

    /// The old `4 * BLOCK` gate refused this shape. 1023 elements cannot fill 256 lanes, so the cap
    /// binds and each lane gets four elements — a geometry question, not a refusal.
    #[test]
    fn below_the_old_gate_still_gets_a_plan() {
        let (n_p, n_i) = (11usize, 1023usize);
        let (la, lc) = pair([n_p, n_i], n_p);
        let plan = classify_reduce(&la, &la, &lc, &[1], &[1], RowMajor).unwrap();
        assert_eq!(geometry(plan), (256, 1, 11));
    }

    /// A scalar output is one block. The plan says so rather than pretending; splitting it needs a
    /// second pass, which is a separate feature.
    #[test]
    fn scalar_output_is_one_block() {
        let (la, lc) = pair([1, 100_000], 1);
        let plan = classify_reduce(&la, &la, &lc, &[1], &[1], RowMajor).unwrap();
        assert_eq!(geometry(plan), (256, 1, 1));
    }

    /// Each refusal names its own reason. The first used to be asserted against the wrong message: a
    /// strided `lc` is caught by the contiguity guard, whose reason is "output must be contiguous"
    /// and never "unit-stride".
    #[test]
    fn inadmissible_shapes_are_rejected_by_name() {
        // Strided output layout: caught by the contiguity guard, and reported as InvalidLayout.
        let la = layout(&[64, 512], &[1024, 2], 0);
        let lc = layout(&[64], &[2], 0);
        let e = classify_reduce(&la, &la, &lc, &[1], &[1], RowMajor).unwrap_err();
        let msg = format!("{e}");
        assert!(msg.contains("output must be contiguous"), "got: {msg}");
        assert!(msg.contains("InvalidLayout"), "got: {msg}");

        // Contiguous output, but no unit-stride axis in either operand: nothing to coalesce on.
        let la = layout(&[64, 512], &[1024, 2], 0);
        let lc = layout(&[64], &[1], 0);
        let e = classify_reduce(&la, &la, &lc, &[1], &[1], RowMajor).unwrap_err();
        let msg = format!("{e}");
        assert!(msg.contains("no unit-stride run"), "got: {msg}");
        assert!(msg.contains("UnImplemented"), "got: {msg}");

        // Multi-axis output part: correct, but not expressible with one `out_step`. The output
        // layout must already have the contracted axis' rank removed, or the broadcast check fires
        // first.
        let la = layout(&[4, 8, 8], &[64, 8, 1], 0);
        let lc = layout(&[4, 8], &[8, 1], 0);
        let e = classify_reduce(&la, &la, &lc, &[2], &[2], RowMajor).unwrap_err();
        let msg = format!("{e}");
        assert!(msg.contains("output part is not a single axis"), "got: {msg}");
    }

    /// The lane count must track the grid target continuously: shrinking `n_out` by 2x must raise
    /// `lanes` by at most 2x, never jump from 1 straight to a whole block.
    #[test]
    fn lane_count_is_continuous_across_the_threshold() {
        let n_i = 1024usize;
        let mut previous = 0u32;
        for n_p in [262_144usize, 131_072, 65_536, 8192, 4096, 2048, 1024, 512] {
            let (la, lc) = pair([n_p, n_i], n_p);
            let plan = classify_reduce(&la, &la, &lc, &[1], &[1], RowMajor).unwrap();
            let (lanes, _, blocks) = geometry(plan);

            assert!(blocks >= MIN_BLOCKS, "n_out={n_p}: {blocks} blocks is short of {MIN_BLOCKS}");
            assert!(lanes >= previous, "n_out={n_p}: lanes fell from {previous} to {lanes}");
            if previous != 0 {
                assert!(lanes <= previous * 2, "n_out={n_p}: lanes jumped {previous} -> {lanes}");
            }
            previous = lanes;
        }
        // 512 outputs of a 1024-long contraction: one block per output, the old `ContigS` corner.
        let (la, lc) = pair([512, n_i], 512);
        assert_eq!(geometry(classify_reduce(&la, &la, &lc, &[1], &[1], RowMajor).unwrap()), (256, 1, 512));
    }

    /// The shapes the GPU integration tests use, one per lane count the classifier can emit.
    ///
    /// Each lane count is a separately compiled kernel (`@LANES@` is a token), so the integration
    /// tests are only meaningful if these shapes still land on those counts. This test is the host
    /// half of that contract: it fails before anyone rents a GPU. Keep it in step with
    /// `VECDOT_CASES` in `tests/test_kernels.rs`.
    #[test]
    fn one_gpu_case_per_lane_count() {
        // (rows, cols, contracted axis, expected lanes)
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

    /// Invariants that must hold for every shape that is planned at all, in both orientations. This
    /// is the test that would have caught the earlier vacuous gate test: it asserts properties of
    /// the *plan*, so it cannot pass by refusing.
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
                assert!(
                    plan.lanes <= n_contract.next_power_of_two().min(MAX_LANES as usize) as u32,
                    "lanes exceed what a {n_contract}-long contraction can fill: {}",
                    plan.lanes
                );
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
