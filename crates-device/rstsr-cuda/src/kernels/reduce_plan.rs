//! Host-side plan selection for reduction kernels.
//!
//! A GPU reduction must commit to a thread mapping *before* it launches, and the only input to that
//! decision is strides. There are exactly two mappings worth having:
//!
//! - [`ReducePlan::ContigS`] — one **block per output element**; the block's threads split the
//!   contiguous contracted run. Needs `n_contig` large enough to keep the block busy and to amortise
//!   the closing block reduction.
//! - [`ReducePlan::ContigM`] — one **thread per output element**; each thread walks the contracted
//!   axis serially with a private accumulator. Needs the *output* axis to be contiguous so that 32
//!   lanes cover 32 consecutive outputs at each contraction step. Has no cross-thread reduction at all.
//! - [`ReducePlan::General`] — neither holds. Route through the existing tiled transpose to pack a
//!   contiguous copy, then re-plan. Never fall back to a scalar loop: on GPU that is a 10–30x cliff.
//!
//! The CPU makes the same three-way decision (`flag_contig_s` / `flag_contig_m` / general in
//! `rstsr-native-impl/src/cpu_serial/vecdot.rs`), but prefers `ContigS` unconditionally — a short run
//! just means more outer iterations, and cost stays linear. **On GPU it must not:** a block per output
//! with `n_contig = 4` puts 4 of 256 threads to work. Hence [`MIN_RUN_PER_THREAD`], which is the only
//! policy difference from the CPU.
//!
//! This module is **host-only**. No CUDA, no NVRTC, no device — so every decision here is testable on
//! a machine with no GPU.

use crate::prelude_dev::*;

/// Threads per block for reduction kernels.
///
/// Must match the elementwise launcher's block size, and must stay a **compile-time constant**:
/// sizing it with `cudaOccupancyMaxPotentialBlockSize` would make the block reduction's summation
/// tree depend on the driver and device, and the kernel would stop producing reproducible sums.
pub(crate) const BLOCK: usize = 256;

/// Elements of work per thread required before a block-split run is worth launching.
///
/// Below roughly this ratio the block reduction's tree costs more than the accumulate it is closing.
const MIN_RUN_PER_THREAD: usize = 4;

/// How a reduction decomposes into threads. Counts only — the addressing plan is a separate, later
/// concern (`PlanBytes`-style `#[repr(C)]` marshalling).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum ReducePlan {
    /// Block per output; threads split the contiguous contracted run.
    ContigS {
        /// Length of the contracted contiguous run, in elements.
        n_contig: usize,
        /// Iterations over the *discontiguous* remainder of the contracted axes (usually 1).
        n_rest: usize,
        /// Number of output elements, i.e. the grid dimension.
        n_out: usize,
    },
    /// Thread per output; each thread walks the contracted axes serially.
    ContigM {
        /// Length of the contiguous run over the *remaining* (output) axes, in elements.
        n_contig: usize,
        /// Total contracted length walked serially by each thread.
        n_contract: usize,
        /// Number of output elements.
        n_out: usize,
    },
    /// Neither mapping is admissible; pack first.
    General,
}

impl ReducePlan {
    pub(crate) fn name(&self) -> &'static str {
        match self {
            Self::ContigS { .. } => "contig_s",
            Self::ContigM { .. } => "contig_m",
            Self::General => "general",
        }
    }
}

/// Decide how to decompose `vecdot`-style contraction `c[out] = sum_contract a * b`.
///
/// `axes_a` / `axes_b` are the contracted axes, in the same normalised `isize` form the device trait
/// takes. `order` is the device's default order, needed because the output layout is aligned against
/// the operands' remaining axes before the contiguous runs are compared.
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
    // downstream, so it is a hard error rather than a fallback.
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

    // --- flag_contig_s: the *contracted* part has a run contiguous in BOTH operands -------------
    //
    // `get_axes_composition` returns (singleton, broadcast, contiguous, discontiguous) axis lists.
    // The contiguous list is the leading run, built by walking |stride| up from 1, so its common
    // prefix across operands is a run both can stream together.
    let (_, _, asc_a, _) = get_axes_composition(&las);
    let (_, _, asc_b, _) = get_axes_composition(&lbs);
    let mut asc: Vec<usize> = Vec::new();
    for (ic_a, ic_b) in asc_a.iter().zip(asc_b.iter()) {
        if ic_a == ic_b {
            asc.push(*ic_a);
        } else {
            break;
        }
    }
    let flag_contig_s = !asc.is_empty();

    // --- flag_contig_m: the *remaining* part has a run contiguous in a, b AND c ------------------
    //
    // The extra `c` term is deliberate: `ContigM` writes `n_contig` outputs at once, so the output
    // run has to be contiguous too.
    let (_, _, amc_a, _) = get_axes_composition(&lam);
    let (_, _, amc_b, _) = get_axes_composition(&lbm);
    let (_, _, amc_c, _) = get_axes_composition(&lc_aligned);
    let mut amc: Vec<usize> = Vec::new();
    for (ic_a, ic_b, ic_c) in izip!(amc_a.iter(), amc_b.iter(), amc_c.iter()) {
        if ic_a == ic_b && ic_b == ic_c {
            amc.push(*ic_a);
        } else {
            break;
        }
    }
    let flag_contig_m = !amc.is_empty();

    let n_out = lc.size();
    let n_contract_total = las.size();

    // `ContigS` is preferred, but only when the run alone fills the block. This gate is the one
    // policy difference from the CPU and the whole reason this module exists.
    if flag_contig_s {
        let n_contig = asc.iter().map(|&i| las.shape()[i]).product::<usize>();
        let n_rest = if n_contig == 0 { 0 } else { n_contract_total / n_contig };
        if n_contig >= MIN_RUN_PER_THREAD * BLOCK {
            return Ok(ReducePlan::ContigS { n_contig, n_rest, n_out });
        }
    }

    if flag_contig_m {
        let n_contig = amc.iter().map(|&i| lam.shape()[i]).product::<usize>();
        return Ok(ReducePlan::ContigM { n_contig, n_contract: n_contract_total, n_out });
    }

    Ok(ReducePlan::General)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layout(shape: &[usize], stride: &[isize], offset: isize) -> Layout<IxD> {
        // Adjust to the local constructor if the signature differs.
        Layout::<IxD>::new(shape.to_vec().into(), stride.to_vec().into(), offset.try_into().unwrap()).unwrap()
    }

    #[test]
    fn row_major_contracted_axis_is_contig_s() {
        // A: (p=4, i=100_000) row-major, contracting i. las = axis 1, stride 1 -> one long run.
        let n_i = 4 * BLOCK * 8; // comfortably over the gate
        let la = layout(&[4, n_i], &[n_i as isize, 1], 0);
        let lb = layout(&[4, n_i], &[n_i as isize, 1], 0);
        let lc = layout(&[4], &[1], 0);

        let plan = classify_reduce(&la, &lb, &lc, &[1], &[1], RowMajor).unwrap();
        match plan {
            ReducePlan::ContigS { n_contig, n_rest, n_out } => {
                assert_eq!(n_contig, n_i);
                assert_eq!(n_rest, 1);
                assert_eq!(n_out, 4);
            },
            other => panic!("expected ContigS, got {}", other.name()),
        }
    }

    #[test]
    fn transposed_layout_is_contig_m() {
        // A: (i=4*BLOCK*8, p=4) row-major, contracting i (axis 0, stride 4). The contracted part has
        // no contiguous run, but the remaining axis (p, stride 1) is contiguous in a, b and c.
        let n_i = 4 * BLOCK * 8;
        let la = layout(&[n_i, 4], &[4, 1], 0);
        let lb = layout(&[n_i, 4], &[4, 1], 0);
        let lc = layout(&[4], &[1], 0);

        let plan = classify_reduce(&la, &lb, &lc, &[0], &[0], RowMajor).unwrap();
        match plan {
            ReducePlan::ContigM { n_contig, n_contract, n_out } => {
                assert_eq!(n_contig, 4);
                assert_eq!(n_contract, n_i);
                assert_eq!(n_out, 4);
            },
            other => panic!("expected ContigM, got {}", other.name()),
        }
    }

    #[test]
    fn short_run_does_not_select_contig_s() {
        // Contracting a length-2 axis: `ContigS` would launch a 256-thread block for 2 elements.
        // The run is contiguous, so the CPU would fire `flag_contig_s` here; the GPU must not.
        let la = layout(&[8, 2], &[2, 1], 0);
        let lb = layout(&[8, 2], &[2, 1], 0);
        let lc = layout(&[8], &[1], 0);

        let plan = classify_reduce(&la, &lb, &lc, &[1], &[1], RowMajor).unwrap();
        assert_ne!(plan.name(), "contig_s", "size gate must reject a 2-element run");
    }

    #[test]
    fn strided_both_axes_is_general() {
        // Contract axis 1 with stride 2 (stepped), output axis 0 with stride 4: no unit-stride run
        // anywhere, so neither mapping is admissible.
        let la = layout(&[4, 16], &[64, 2], 0);
        let lb = layout(&[4, 16], &[64, 2], 0);
        let lc = layout(&[4], &[4], 0);

        let plan = classify_reduce(&la, &lb, &lc, &[1], &[1], RowMajor).unwrap();
        assert_eq!(plan.name(), "general");
    }
    
    #[test]
    fn plan_counts_partition_the_contracted_space() {
        // (shape_a, stride_a, contract_axis, shape_c, stride_c)
        let cases: &[(&[usize], &[isize], usize, &[usize], &[isize])] = &[
            (&[4, 8192], &[8192, 1], 1, &[4], &[1]),          // row-major, run over axis 1 -> ContigS
            (&[8192, 4], &[4, 1], 0, &[4], &[1]),             // transposed, run over axis 0 -> ContigM
            (&[3, 5, 4096], &[20480, 4096, 1], 2, &[3, 5], &[5, 1]), // rank 3, run over axis 2
            (&[4096, 5, 3], &[15, 3, 1], 0, &[5, 3], &[3, 1]),       // rank 3, output contiguous
        ];
        for (sa, ta, axis, sc, tc) in cases {
            let la = layout(sa, ta, 0);
            let lc = layout(sc, tc, 0);
            let total = la.size();
            let plan = classify_reduce(&la, &la, &lc, &[*axis as isize], &[*axis as isize], RowMajor)
                .unwrap_or_else(|e| panic!("{sa:?} axis {axis}: classify failed: {e:?}"));
    
            match plan {
                ReducePlan::ContigS { n_contig, n_rest, n_out } => {
                    assert_eq!(
                        n_out * n_contig * n_rest,
                        total,
                        "{sa:?} axis {axis}: contig_s counts must partition the tensor"
                    );
                },
                ReducePlan::ContigM { n_contig, n_contract, n_out } => {
                    assert_eq!(
                        n_out * n_contract,
                        total,
                        "{sa:?} axis {axis}: contig_m counts must cover the tensor"
                    );
                    assert_eq!(n_out % n_contig, 0, "{sa:?} axis {axis}: output run must divide n_out");
                },
                ReducePlan::General => {
                    panic!("{sa:?} axis {axis}: both layouts are contiguous somewhere, expected a plan");
                },
            }
        }
    }

}
