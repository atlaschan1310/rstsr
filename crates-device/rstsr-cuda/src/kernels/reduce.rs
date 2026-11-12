//! Reduction kernels.
//!
//! One kernel, `vecdot` (`pi,pi->p`), parameterised by the plan's `lanes` / `outputs_per_block` pair
//! from [`super::reduce_plan`]. There is no longer a `ContigS` kernel and a `ContigM` kernel: those
//! names now describe the two ends of the same source, and which end you get is decided by
//! `classify_reduce`, not by a branch here.
//!
//! # Thread mapping
//!
//! ```text
//! lane   = threadIdx.x & (LANES - 1)                        // low bits: the unit-stride axis
//! output = blockIdx.x * OPB + (threadIdx.x >> LOG2_LANES)    // high bits: which output
//! ```
//!
//! Lane in the **low** bits is the whole design. It puts consecutive lanes on consecutive addresses
//! when the contracted axis is unit-stride, and — just as important — it keeps the lanes that share
//! an output inside one warp, which is what lets the tree be a `__shfl_down_sync` chain instead of a
//! shared-memory reduction. CuPy does the opposite (`output = tid & (block_stride - 1)`) and pays for
//! it with a shared-memory tree in every reduction it runs.
//!
//! # Two stages, one of which is usually compiled out
//!
//! - **Stage 1** butterflies over the lanes of one output, with offsets below `min(LANES, warpSize)`.
//!   When `LANES < warpSize` several outputs share a warp, so the cross-group value the shuffle
//!   still delivers is discarded with `lane + off < LANES` rather than by making the loop divergent.
//! - **Stage 2** appears only when one output spans more than one warp (`LANES > warpSize`): each
//!   warp's partial goes to shared memory and the first warp of each output's group finishes the
//!   reduction. `LANES` is a compile-time token, so for `LANES <= 32` the whole stage — shared array
//!   included — is dead code and disappears.
//!
//! # The guard is not a `return`
//!
//! `if (j >= n_out) return;` would take part of a warp out of the shuffles below and make the mask
//! undefined. The guard wraps only the accumulation and the final store; an out-of-range thread keeps
//! `acc == 0`, participates in the tree, and writes nothing. This matters as soon as
//! `outputs_per_block > 1`, i.e. whenever several outputs share a warp.
//!
//! # Arithmetic
//!
//! `acc + x * y` as written — NVRTC contracts it to a single-rounding FMA. No CPU parity is claimed,
//! so `__dadd_rn` / `__dmul_rn` are not needed; see `reduce_plan`'s module docs.
//!
//! # Reproducibility
//!
//! The accumulation order is fixed by `LANES` and `n_contract`, both functions of the shape, so the
//! same input on the same binary gives the same bits. `__activemask()` is used instead of a literal
//! mask because the mask width differs between NVIDIA (32) and CDNA (64); it is only meaningful
//! because `__syncwarp()` forces the warp back together after the output guard, which is divergent
//! whenever several outputs share a warp.

use crate::prelude_dev::*;
use std::fmt::Write as _;
use std::sync::Arc;

use cudarc::driver::CudaFunction;
use crate::dtype::CudaDType;
use super::reduce_plan::{ReducePlan, BLOCK};

/// Largest number of warps a block can contain, assuming a warp is at least 32 lanes.
///
/// Only a shared-array bound; the real warp count comes from the `warpSize` builtin.
const MAX_WARPS_PER_BLOCK: usize = BLOCK / 32;

/// Kernel body. `@NAME@` tokens are substituted by [`render`]; the plan's geometry is baked in, so
/// `LANES`, `LOG2_LANES` and `OPB` are compile-time constants.
const VECDOT_ENTRY: &str = r#"
extern "C" __global__ void rstsr_vecdot(
    @T@* __restrict__ c,
    const @T@* __restrict__ a,
    const @T@* __restrict__ b,
    rstsr_usz n_out,
    rstsr_usz n_contract,
    rstsr_isz sa,      // contracted-axis stride of a, elements
    rstsr_isz sb,      // contracted-axis stride of b, elements
    rstsr_isz sa_out,  // output-axis stride of a, elements
    rstsr_isz sb_out)  // output-axis stride of b, elements
{
    const rstsr_usz lane = (rstsr_usz)threadIdx.x & (@LANES@ - 1u);
    const rstsr_usz j = (rstsr_usz)blockIdx.x * @OPB@ + ((rstsr_usz)threadIdx.x >> @LOG2_LANES@);

    @T@ acc = (@T@)0;
    if (j < n_out) {
        const rstsr_isz ia = (rstsr_isz)j * sa_out;
        const rstsr_isz ib = (rstsr_isz)j * sb_out;
        for (rstsr_usz m = lane; m < n_contract; m += @LANES@) {
            acc = acc + a[ia + (rstsr_isz)m * sa] * b[ib + (rstsr_isz)m * sb];
        }
    }

    // Stage 1: within a warp. Offsets never reach the next output's group when LANES < warpSize, and
    // the mask term throws away the cross-group value the shuffle still delivers.
    //
    // `__syncwarp()` is not decoration. When LANES < warpSize several outputs share a warp, so the
    // guard above is divergent *within* the warp on the last block. Since Volta the hardware may not
    // reconverge at the end of a divergent region on its own, and `__activemask()` would then return
    // a partial set — the shuffle would silently read lanes that never participated. Forcing the
    // reconvergence here is the portable fix, and it is a no-op when the warp is already converged.
    __syncwarp();
    const int ws = warpSize;
    for (int off = (@LANES@ <= 32 ? (@LANES@ >> 1) : (ws >> 1)); off > 0; off >>= 1) {
        @T@ other = __shfl_down_sync(__activemask(), acc, off);
        acc = acc + (((rstsr_usz)off + lane < @LANES@) ? other : (@T@)0);
    }

    // Stage 2: only when one output spans several warps. Statically dead for LANES <= 32.
    const int nw = @LANES@ / ws;
    if (@LANES@ > 32 && nw > 1) {
        __shared__ @T@ wpart[@MAXW@];
        const int tl = (int)((unsigned)threadIdx.x & (unsigned)(ws - 1));
        const int wid = (int)((unsigned)threadIdx.x / (unsigned)ws);
        if (tl == 0) wpart[wid] = acc;
        __syncthreads();

        if (wid % nw == 0) {
            // Every lane of this warp runs: the butterfly below uses a full-warp mask.
            @T@ v = (tl < nw) ? wpart[(wid / nw) * nw + tl] : (@T@)0;
            for (int off = nw >> 1; off > 0; off >>= 1) {
                @T@ o = __shfl_down_sync(__activemask(), v, off);
                v = v + (((int)tl + off < nw) ? o : (@T@)0);
            }
            if (tl == 0 && j < n_out) c[j] = v;
        }
        return;
    }

    // No cross-warp stage: lane 0 of each group holds its output's total.
    if (lane == 0 && j < n_out) c[j] = acc;
}
"#;

/// Fold an element offset into a base device pointer, in bytes.
///
/// Mirrors `kernels::elementwise::offset_ptr`: the read event must be recorded on the **launching**
/// stream, not the allocation stream, or a cross-stream free can race the kernel.
fn offset_ptr<T>(exec: &Arc<CudaStreamObj>, buf: &CudaBuffer<T>, offset: usize) -> u64 {
    if buf.is_empty() {
        0
    } else {
        (buf.device_ptr_on(exec) as u64)
            .wrapping_add(offset as u64 * core::mem::size_of::<T>() as u64)
    }
}

fn render(template: &str, subs: &[(&str, &str)]) -> String {
    let mut s = template.to_string();
    for (name, text) in subs {
        s = s.replace(name, text);
    }
    s
}

fn assemble(includes: &[&str], body: &str) -> String {
    let mut s = String::from(crate::kernel::CUDA_PRELUDE);
    s.push_str(&super::splice_includes(includes));
    let _ = write!(s, "\n{body}");
    s
}

/// Compile-or-fetch the reduction entry, interned per ordinal on the specialization discriminator.
///
/// The discriminator carries the geometry, not just the dtype: two instantiations with the same `T`
/// but different `LANES` are different kernels, and the source hash would catch it anyway — this
/// keeps the in-memory interning key honest too.
pub(crate) fn reduce_function(
    device: &DeviceCuda,
    name: &'static str,
    includes: &[&str],
    disc_extra: &str,
    entries: impl FnOnce() -> String,
) -> Result<Arc<CudaFunction>> {
    let disc = format!("{name}|{}|{disc_extra}", includes.concat());
    let cache = device.ordinal_obj().planned_cache();
    if let Some(f) = cache.entries.lock().unwrap_or_else(|e| e.into_inner()).get(&disc) {
        return Ok(f.clone());
    }
    let src = assemble(includes, &entries());
    let key = crate::kernel::make_key(device.ordinal_obj(), name, &src, &[])?;
    let f = device.ordinal_obj().kernel_function(&key)?;
    cache.entries.lock().unwrap_or_else(|e| e.into_inner()).insert(disc, f.clone());
    Ok(f)
}

/// Launch `vecdot` for a plan produced by [`super::reduce_plan::classify_reduce`].
///
/// `las` / `lbs` are the contracted parts of the operands; their offsets are the base pointers, and
/// `dim_split_axes` keeps the parent offset on both halves, so one offset serves the whole operand.
/// All other addressing comes from `plan`, so no geometry is derived twice.
///
/// Refusals live in `classify_reduce`, not here: by the time a plan exists the launch is possible.
#[allow(clippy::too_many_arguments)]
pub(crate) fn vecdot_launch<T>(
    device: &DeviceCuda,
    c: &mut CudaBuffer<MaybeUninit<T>>,
    lc: &Layout<IxD>,
    a: &CudaBuffer<T>,
    las: &Layout<IxD>,
    b: &CudaBuffer<T>,
    lbs: &Layout<IxD>,
    plan: &ReducePlan,
) -> Result<()>
where
    T: CudaDType,
{
    let ReducePlan {
        lanes,
        outputs_per_block,
        block,
        n_contract,
        step_a,
        step_b,
        n_out,
        out_step_a,
        out_step_b,
    } = *plan;

    if n_out == 0 || n_contract == 0 {
        return Ok(());
    }

    let opb = outputs_per_block as usize;
    let grid_blocks = (n_out + opb - 1) / opb;
    // one block per output tile: the grid is a u32 in the driver API
    if grid_blocks > u32::MAX as usize {
        rstsr_raise!(UnImplemented, "cuda vecdot: grid exceeds u32")?;
    }
    if !lanes.is_power_of_two() || block % lanes != 0 {
        // A malformed plan is a bug in `classify_reduce`, not a user error; say so loudly.
        rstsr_raise!(InvalidLayout, "cuda vecdot: plan has lanes={lanes}, block={block}")?;
    }

    let exec = device.stream();
    let _ga = crate::kernel::order_input(a, exec)?;
    let _gb = crate::kernel::order_input(b, exec)?;

    let disc = format!("{}|L{lanes}|B{block}", T::CNAME);
    let f = reduce_function(device, "rstsr_vecdot", T::INCLUDES, &disc, || {
        render(
            VECDOT_ENTRY,
            &[
                ("@T@", T::CNAME),
                ("@LANES@", &lanes.to_string()),
                ("@LOG2_LANES@", &lanes.trailing_zeros().to_string()),
                ("@OPB@", &outputs_per_block.to_string()),
                ("@MAXW@", &MAX_WARPS_PER_BLOCK.to_string()),
            ],
        )
    })?;

    // SAFETY: `MaybeUninit<T>` and `T` are layout-identical; the kernel writes every visited element
    // (exactly once per `j < n_out`, by the lane-0 or tile-leader thread) and `c` holds `n_out`.
    let c_typed: &mut CudaBuffer<T> =
        unsafe { &mut *(c as *mut CudaBuffer<MaybeUninit<T>> as *mut CudaBuffer<T>) };

    crate::kernel::FlatLaunch::new(exec, &f, grid_blocks)
        .arg_u64(offset_ptr(exec, c_typed, lc.offset()))
        .arg_u64(offset_ptr(exec, a, las.offset()))
        .arg_u64(offset_ptr(exec, b, lbs.offset()))
        .arg_usize(n_out)
        .arg_usize(n_contract)
        .arg_bytes(&step_a.to_ne_bytes())
        .arg_bytes(&step_b.to_ne_bytes())
        .arg_bytes(&out_step_a.to_ne_bytes())
        .arg_bytes(&out_step_b.to_ne_bytes())
        // The default flat geometry would launch `n / block` blocks, which is not
        // `ceil(n_out / outputs_per_block)`. Both dimensions are passed separately.
        .explicit_shape((grid_blocks as u32, 1, 1), (block, 1, 1))
        .launch()?;

    // publish the write to the output buffer's allocation stream
    let _go = crate::kernel::order_output(c, exec)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::reduce_plan::{classify_reduce, MAX_LANES};
    use super::*;

    fn layout(shape: &[usize], stride: &[isize], offset: usize) -> Layout<IxD> {
        // `Layout::new` takes `offset: usize`; keep this helper's signature matching it.
        Layout::<IxD>::new(shape.to_vec().into(), stride.to_vec().into(), offset).unwrap()
    }

    /// Mirror of the kernel's `lane` / `j` split, in host arithmetic.
    ///
    /// `tid >> log2(lanes)` and `tid & (lanes - 1)` are only equivalent to `/` and `%` because
    /// `lanes` is a power of two; this checks the two against each other, and checks that every
    /// output is visited exactly `lanes` times — which is what makes the single lane-0 write safe.
    #[test]
    fn thread_split_matches_block_geometry() {
        for lanes in [1usize, 2, 4, 8, 16, 32, 64, 128, 256] {
            let opb = BLOCK / lanes;
            let log2 = lanes.trailing_zeros();
            for n_out in [1usize, 7, 64, 999, 4096] {
                let grid = (n_out + opb - 1) / opb;

                let mut hits = vec![0usize; n_out];
                for blk in 0..grid {
                    for tid in 0..BLOCK {
                        let lane = tid & (lanes - 1);
                        let j = blk * opb + (tid >> log2);
                        assert_eq!(lane, tid % lanes, "shift and modulo must agree");
                        assert!(lane < lanes);
                        if j < n_out {
                            hits[j] += 1;
                        }
                    }
                }

                for (j, &h) in hits.iter().enumerate() {
                    assert_eq!(h, lanes, "lanes={lanes} n_out={n_out}: output {j} hit {h} times");
                }
            }
        }
    }

    /// The kernel addresses output `j` at `c_base + j`, and reads `a` at
    /// `a_base + j * out_step + m * step`. `offset_ptr` folds `layout.offset()` into the base
    /// pointer, so this must equal `layout.index_uncheck` for every element.
    ///
    /// The failure mode guarded here is wrong data, not a crash — hence the exhaustive loop.
    fn check_addresses(la: &Layout<IxD>, n_contract: usize, n_out: usize) {
        let offset: i64 = la.offset() as i64;
        let step: i64 = la.stride()[0] as i64; // contracted axis
        let out_step: i64 = la.stride()[1] as i64; // output axis

        for m in 0..n_contract {
            for j in 0..n_out {
                let derived: i64 = offset + (j as i64) * out_step + (m as i64) * step;
                let truth: i64 = la.index_uncheck(&[m, j]) as i64;
                assert_eq!(derived, truth, "address mismatch at (m={m}, j={j})");
            }
        }
    }

    #[test]
    fn addresses_match_layout_index() {
        // (i, p) layout: contracted axis 0 with stride p, output axis 1 with stride 1
        let (n_i, n_p) = (37usize, 11usize);
        check_addresses(&layout(&[n_i, n_p], &[n_p as isize, 1], 0), n_i, n_p);
    }

    #[test]
    fn addresses_match_for_the_contiguous_contraction() {
        // (p, i) layout: the other orientation, contracted axis unit-stride
        let (n_p, n_i) = (11usize, 37usize);
        check_addresses(&layout(&[n_p, n_i], &[n_i as isize, 1], 0), n_i, n_p);
    }

    #[test]
    fn addresses_match_with_offset() {
        // non-zero offset: this is where the "double count offset" bug lives
        let (n_i, n_p) = (9usize, 5usize);
        check_addresses(&layout(&[n_i, n_p], &[n_p as isize, 1], 3), n_i, n_p);
    }

    /// The plan is the launcher's only source of geometry, so anything the classifier emits must be
    /// launchable. Cheap sweep rather than a second derivation.
    #[test]
    fn every_plan_the_classifier_emits_is_launchable() {
        for n_contract in [1usize, 3, 8, 1023, 4096, 100_000] {
            for n_out in [1usize, 8, 1000, 200_000] {
                for contract_last in [true, false] {
                    let shape = if contract_last { [n_out, n_contract] } else { [n_contract, n_out] };
                    let la = layout(&shape, &[shape[1] as isize, 1], 0);
                    let lc = layout(&[n_out], &[1], 0);
                    let axis = if contract_last { 1isize } else { 0isize };
                    let plan = classify_reduce(&la, &la, &lc, &[axis], &[axis], RowMajor).unwrap_or_else(|e| {
                        panic!("({n_contract}, {n_out}, last={contract_last}) should be plannable: {e}")
                    });
                    assert!(plan.lanes.is_power_of_two() && plan.lanes <= MAX_LANES);
                    assert_eq!(plan.block, BLOCK as u32);
                    assert_eq!(plan.block % plan.lanes, 0, "the launcher would reject this plan");
                    assert_eq!(plan.lanes * plan.outputs_per_block, plan.block);
                }
            }
        }
    }
}
