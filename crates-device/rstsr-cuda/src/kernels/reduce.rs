//! Reduction kernels.
//!
//! First slice: `vecdot` in the `ContigM` shape — **one thread per output element**, each thread
//! walking the contracted axis serially with private accumulators. No cross-thread communication at
//! all, so the sum is order-stable by construction and every element of the contraction is visited in
//! a fixed order.
//!
//! Shape admitted here (everything else returns `UnImplemented`):
//!
//! - both operands rank 2 after the axis split, one contracted axis and one output axis each
//! - the **output axis is unit-stride** in both operands (`(i, p)` layout — this is `ContigM`)
//! - the output tensor is contiguous, so output addressing is just `c[j]`
//! - one dtype for a, b and c; `f32` / `f64` (f16/bf16 once the accumulate semantics are settled)
//!
//! This is exactly `pi,pi->p` on an `(i, p)` array. The `(p, i)` orientation classifies as `ContigS`
//! and is not handled here yet.
//!
//! # Accumulation order
//!
//! **One accumulator, contracted index ascending** — the CPU's `flag_contig_m` branch
//! (`cpu_serial/vecdot.rs:110-135`) does `c[t] += a[t] * b[t]` once per contracted index, in
//! increasing order, with no unrolling. Reproducing that is the only way this kernel can be
//! **bitwise identical** to the CPU device.
//!
//! Two things that look like optimisations and both break parity:
//!
//! - **Eight accumulators.** That tree belongs to `flag_contig_s` / `unrolled_binary_reduce`, a
//!   different branch. Mixing it in changes the last bits.
//! - **FMA contraction.** `acc + x * y` becomes a single-rounding `fma` under NVRTC (fmad defaults
//!   to on), while Rust performs two roundings. `__fadd_rn(acc, __fmul_rn(x, y))` forces the split.
//!
//! Neither costs anything measurable: the loop is DRAM-bound, one load pair per element, and a
//! dependent fp64 add is ~26x shorter than the memory latency it hides behind.

use crate::prelude_dev::*;
use std::fmt::Write as _;
use std::sync::Arc;

use cudarc::driver::CudaFunction;
use crate::dtype::CudaDType;
/// Kernel body. Token substitution uses `@NAME@` (see `render`).
const VECDOT_CM_ENTRY: &str = r#"
extern "C" __global__ void rstsr_vecdot_cm(
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
    rstsr_usz j = (rstsr_usz)blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n_out) return;

    // base of this output's two rows
    rstsr_isz ia = (rstsr_isz)j * sa_out;
    rstsr_isz ib = (rstsr_isz)j * sb_out;

    // One accumulator in contracted order: the CPU's flag_contig_m branch accumulates
    // sequentially, so any unrolling here breaks bitwise parity. Do not "optimise" this into
    // multiple accumulators without also deciding to give up CPU-identical results.
    //
    // __fmul_rn / __fadd_rn force two roundings. Plain `acc + x * y` contracts to a
    // single-rounding FMA (fmad is on by default in NVRTC), which also breaks parity.
    @T@ acc = (@T@)0;
    for (rstsr_usz m = 0; m < n_contract; ++m) {
        @T@ x = a[ia + (rstsr_isz)m * sa];
        @T@ y = b[ib + (rstsr_isz)m * sb];
        acc = __fadd_rn(acc, __fmul_rn(x, y));
    }

    c[j] = acc;
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
/// Same shape as `kernels::elementwise::planned_function`; that one is private to its module, so
/// this is a local copy. Lifting both into `kernels/mod.rs` is a ~10 line refactor that would remove
/// the duplication, if the maintainer prefers.
fn reduce_function(
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

/// `vecdot` for the `ContigM` shape: one thread per output, serial contraction.
///
/// `las` / `lbs` are the contracted parts, `lam` / `lbm` the remaining parts, both from
/// `Layout::dim_split_axes`. `lc` is the broadcast-aligned output layout.
///
/// Returns `UnImplemented` for any shape this slice does not cover — never falls back to the host.
#[allow(clippy::too_many_arguments)]
pub(crate) fn vecdot_contig_m_launch<T>(
    device: &DeviceCuda,
    c: &mut CudaBuffer<MaybeUninit<T>>,
    lc: &Layout<IxD>,
    a: &CudaBuffer<T>,
    las: &Layout<IxD>,
    lam: &Layout<IxD>,
    b: &CudaBuffer<T>,
    lbs: &Layout<IxD>,
    lbm: &Layout<IxD>,
) -> Result<()>
where
    T: CudaDType,
{
    // --- preconditions of this slice -------------------------------------------------------------
    if las.ndim() != 1 || lbs.ndim() != 1 {
        rstsr_raise!(UnImplemented, "cuda vecdot (contig_m, single axis): only one contracted axis is implemented")?;
    }
    if lam.ndim() != 1 || lbm.ndim() != 1 {
        rstsr_raise!(UnImplemented, "cuda vecdot (contig_m, single axis): only one output axis is implemented")?;
    }
    // output must be fully contiguous, so that `c[j]` addresses it
    if lc.ndim() != 1 || lc.stride()[0] != 1 {
        rstsr_raise!(UnImplemented, "cuda vecdot (contig_m, single axis): output must be contiguous")?;
    }
    // the output axis must be unit-stride in both operands -- this is what makes the read coalesced
    if lam.stride()[0] != 1 || lbm.stride()[0] != 1 {
        rstsr_raise!(UnImplemented, "cuda vecdot (contig_m, single axis): output axis must be unit-stride in both operands")?;
    }
    if las.shape()[0] != lbs.shape()[0] {
        rstsr_raise!(InvalidLayout, "contracted dimensions of a and b differ")?;
    }

    let n_out: usize = lc.size();
    let n_contract: usize = las.shape()[0];
    if n_out == 0 || n_contract == 0 {
        return Ok(());
    }

    // strides are `isize`; widen to i64 for the kernel so negative strides stay exact
    let sa: i64 = las.stride()[0] as i64;
    let sb: i64 = lbs.stride()[0] as i64;
    let sa_out: i64 = lam.stride()[0] as i64;
    let sb_out: i64 = lbm.stride()[0] as i64;

    // --- launch ----------------------------------------------------------------------------------
    let exec = device.stream();
    let _ga = crate::kernel::order_input(a, exec)?;
    let _gb = crate::kernel::order_input(b, exec)?;

    let f = reduce_function(device, "rstsr_vecdot_cm", T::INCLUDES, T::CNAME, || {
        render(VECDOT_CM_ENTRY, &[("@T@", T::CNAME)])
    })?;

    // SAFETY: `MaybeUninit<T>` and `T` are layout-identical; the kernel writes every visited element
    // (exactly one per `j < n_out`), and `c` holds `n_out` elements.
    let c_typed: &mut CudaBuffer<T> =
        unsafe { &mut *(c as *mut CudaBuffer<MaybeUninit<T>> as *mut CudaBuffer<T>) };

    crate::kernel::FlatLaunch::new(exec, &f, n_out)
        .arg_u64(offset_ptr(exec, c_typed, lc.offset()))
        .arg_u64(offset_ptr(exec, a, las.offset()))
        .arg_u64(offset_ptr(exec, b, lbs.offset()))
        .arg_usize(n_out)
        .arg_usize(n_contract)
        .arg_bytes(&sa.to_ne_bytes())
        .arg_bytes(&sb.to_ne_bytes())
        .arg_bytes(&sa_out.to_ne_bytes())
        .arg_bytes(&sb_out.to_ne_bytes())
        .launch()?;

    // publish the write to the output buffer's allocation stream
    let _go = crate::kernel::order_output(c, exec)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a 2-D layout. **Adjust here if the local constructor differs** — this is the only place
    /// the offset type is written. `Layout::offset()` is read back as `i64` everywhere below, so the
    /// arithmetic is unaffected by whether it is stored as `usize` or `isize`.
    fn layout2(shape: [usize; 2], stride: [isize; 2], offset: usize) -> Layout<IxD> {
        Layout::<IxD>::new(vec![shape[0], shape[1]], vec![stride[0], stride[1]], offset).unwrap()
    }

    /// The kernel addresses output `j` at `c_base + j`, and reads `a` at
    /// `a_base + j * sa_out + m * sa`. `offset_ptr` folds `layout.offset()` into the base pointer,
    /// so this arithmetic must equal `layout.index_uncheck(&[m, j])` for every element.
    ///
    /// The failure mode this guards is wrong data, not a crash — hence the exhaustive loop rather
    /// than a couple of spot checks.
    fn check_addresses(la: &Layout<IxD>, n_contract: usize, n_out: usize) {
        let offset: i64 = la.offset() as i64;
        let sa: i64 = la.stride()[0] as i64; // contracted axis
        let sa_out: i64 = la.stride()[1] as i64; // output axis

        for m in 0..n_contract {
            for j in 0..n_out {
                let derived: i64 = offset + (j as i64) * sa_out + (m as i64) * sa;
                let truth: i64 = la.index_uncheck(&[m, j]) as i64;
                assert_eq!(derived, truth, "address mismatch at (m={m}, j={j})");
            }
        }
    }

    #[test]
    fn addresses_match_layout_index() {
        // (i, p) layout: contracted axis 0 with stride p, output axis 1 with stride 1
        let (n_i, n_p) = (37usize, 11usize);
        let la = layout2([n_i, n_p], [n_p as isize, 1], 0);
        check_addresses(&la, n_i, n_p);
    }

    #[test]
    fn addresses_match_with_offset() {
        // non-zero offset: this is where the "double count offset" bug lives
        let (n_i, n_p) = (9usize, 5usize);
        let la = layout2([n_i, n_p], [n_p as isize, 1], 3);
        check_addresses(&la, n_i, n_p);
    }

    #[test]
    fn addresses_match_for_strided_contract() {
        // contracted axis every other row: stride 2 * n_p
        let (n_i, n_p) = (6usize, 4usize);
        let la = layout2([n_i, n_p], [2 * n_p as isize, 1], 0);
        check_addresses(&la, n_i, n_p);
    }

    #[test]
    fn addresses_match_for_strided_output() {
        // output axis with stride 2 -- admissible via the launcher's stride arguments even though
        // the classifier would reject it as non-contiguous; this checks the arithmetic, not the policy
        let (n_i, n_p) = (5usize, 7usize);
        let la = layout2([n_i, n_p], [2 * n_p as isize, 2], 0);
        check_addresses(&la, n_i, n_p);
    }
}
