//! Planned elementwise kernels: flat (F) and magic-divmod parameterized
//! (P) variants, dispatched by per-operand [`OperandPlan`] analysis.
//!
//! Operand plans travel **by value** in the kernel parameter list (136 B
//! structs; the driver stages them in constant memory), so the planned
//! path needs no per-launch scratch upload or allocation. The
//! rank-generic kernel in `kernels_ops` remains the correctness floor
//! (`Generic` plans) and the home of assign-style positional-zip
//! semantics.
//!
//! Each program carries both the `_p` (parameterized) and `_f` (flat)
//! entry points for one (dtype, expression) specialization; the variant
//! picks the entry name. A flat operand marshals as a rank-0 plan (the
//! device folds the linear id), so a mixed all-flat selection takes the
//! argument-lean `_f` entry.

use super::plan::{plan_operand, OperandPlan};
use crate::buffer::CudaBuffer;
use crate::dtype::CudaDType;
use crate::kernel::{make_key, order_input, FlatLaunch};
use crate::kernels_ops;
use crate::prelude_dev::*;
use std::sync::Arc;

/// Device-side plan accessors: `rank == 0` is the flat encoding (offset =
/// linear id); division is the round-up magic (`__umul64hi` high word).
const PLAN_HELPER: &str = r#"
struct rstsr_axis { rstsr_usz d; rstsr_usz s; rstsr_usz m; int sh; };
struct rstsr_plan { int rank; rstsr_axis ax[4]; };

__device__ rstsr_usz rstsr_plan_offset(rstsr_usz lin, rstsr_plan p, rstsr_usz base) {
    if (p.rank == 0) return base + lin;
    rstsr_usz off = base;
    for (int i = 0; i < 4; ++i) {
        if (i >= p.rank) break;
        const rstsr_axis* ax = &p.ax[i];
        rstsr_usz q = (ax->m == 0) ? (lin >> ax->sh) : (__umul64hi(lin, ax->m) >> ax->sh);
        off += (lin - q * ax->d) * ax->s;
        lin = q;
    }
    return off;
}
"#;

/// 32-bit plan helper (`_pn` entries): narrow divisors/strides, offset
/// walk in `i32` (negative strides wrap exactly in two's complement), the
/// signed result sign-extends into pointer arithmetic.
const PLAN_HELPER_N: &str = r#"
struct rstsr_axis_n { rstsr_u32 d; rstsr_i32 s; rstsr_u32 m; rstsr_i32 sh; };
struct rstsr_plan_n { int rank; rstsr_axis_n ax[4]; };

__device__ rstsr_isz rstsr_plan_offset_n(int lin, rstsr_plan_n p) {
    if (p.rank == 0) return (rstsr_isz)lin;
    rstsr_i32 off = 0;
    for (int i = 0; i < 4; ++i) {
        if (i >= p.rank) break;
        const rstsr_axis_n* ax = &p.ax[i];
        rstsr_u32 lu = (rstsr_u32)lin;
        rstsr_u32 q = (ax->m == 0) ? (lu >> ax->sh) : (__umulhi(lu, ax->m) >> ax->sh);
        off += ((rstsr_i32)lin - (rstsr_i32)(q * ax->d)) * ax->s;
        lin = (int)q;
    }
    return (rstsr_isz)off;
}
"#;

/// Planned-source assembly: shared prelude + plan/math helpers + entries.
/// The narrow helper is included unconditionally (unused struct is free;
/// keeps one source shape per family).
fn assemble_planned(includes: &[&str], entries: &str) -> String {
    let mut s = String::from(crate::kernel::CUDA_PRELUDE);
    s.push_str(&super::splice_includes(includes));
    let _ = write!(s, "\n{PLAN_HELPER}\n{PLAN_HELPER_N}\n{}\n{entries}", crate::kernel::MATH_HELPER);
    s
}

/// Render `template` by token replacement (`@NAME@` -> `text`); avoids
/// brace-escaping the CUDA C bodies through `format!`.
fn render(template: &str, subs: &[(&str, &str)]) -> String {
    let mut s = template.to_string();
    for (name, text) in subs {
        s = s.replace(name, text);
    }
    s
}

// /* #region source templates (F and P entries per program) */
const BINARY_ENTRIES: &str = r#"
extern "C" __global__ void rstsr_binary_p(
    @TO@* out, rstsr_plan op,
    const @TA@* a, rstsr_plan ap,
    const @TB@* b, rstsr_plan bp,
    rstsr_usz n)
{
    rstsr_usz lin = (rstsr_usz)blockIdx.x * (rstsr_usz)blockDim.x + (rstsr_usz)threadIdx.x;
    if (lin >= n) return;
    rstsr_usz oo = rstsr_plan_offset(lin, op, 0);
    rstsr_usz ia = rstsr_plan_offset(lin, ap, 0);
    rstsr_usz ib = rstsr_plan_offset(lin, bp, 0);
    out[oo] = @EXPR@;
}

extern "C" __global__ void rstsr_binary_pn(
    @TO@* out, rstsr_plan_n op,
    const @TA@* a, rstsr_plan_n ap,
    const @TB@* b, rstsr_plan_n bp,
    int n)
{
    int lin = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    int w = gridDim.x * blockDim.x;
    for (int rep = 0; rep < 4; ++rep) {
        if (lin < n) {
            rstsr_isz oo = rstsr_plan_offset_n(lin, op);
            rstsr_isz ia = rstsr_plan_offset_n(lin, ap);
            rstsr_isz ib = rstsr_plan_offset_n(lin, bp);
            out[oo] = @EXPR@;
        }
        lin += w;
    }
}

extern "C" __global__ void rstsr_binary_f(
    @TO@* out, const @TA@* a, const @TB@* b, rstsr_usz n)
{
    rstsr_usz g = (rstsr_usz)blockIdx.x * (rstsr_usz)blockDim.x + (rstsr_usz)threadIdx.x;
    rstsr_usz w = (rstsr_usz)gridDim.x * (rstsr_usz)blockDim.x;
    #define RSTSR_STEP(I) { rstsr_usz oo = g + (I) * w; rstsr_usz ia = oo, ib = oo; if (oo < n) { out[oo] = @EXPR@; } }
    RSTSR_STEP(0) RSTSR_STEP(1) RSTSR_STEP(2) RSTSR_STEP(3)
    #undef RSTSR_STEP
}
"#;

/// Cross product over one size-3 axis: four reads, three writes per visited slot.
///
/// Its own entry rather than a `binary_launch` expression because the body is not one value per
/// operand — it reads `a` and `b` at the axis offsets +1 and +2 and writes all three output
/// components. The iteration is the same plan machinery: the size-3 axis is collapsed to shape 1 /
/// stride 0, so one visited slot is one vector, and the three components are reached by the axis
/// strides that the launcher passes separately.
const CROSS_ENTRIES: &str = r#"
__device__ __forceinline__ void rstsr_cross3(
    @TO@* cv, const @TA@* av, const @TB@* bv,
    rstsr_isz sa, rstsr_isz sb, rstsr_isz sc)
{
    // the cyclic formula, with 0 * s == 0 so a broadcast (stride 0) operand reads one element
    cv[0 * sc] = (av[1 * sa] * bv[2 * sb]) - (av[2 * sa] * bv[1 * sb]);
    cv[1 * sc] = (av[2 * sa] * bv[0 * sb]) - (av[0 * sa] * bv[2 * sb]);
    cv[2 * sc] = (av[0 * sa] * bv[1 * sb]) - (av[1 * sa] * bv[0 * sb]);
}

extern "C" __global__ void rstsr_cross_p(
    @TO@* out, rstsr_plan op,
    const @TA@* a, rstsr_plan ap,
    const @TB@* b, rstsr_plan bp,
    rstsr_usz n, rstsr_isz sa, rstsr_isz sb, rstsr_isz sc)
{
    rstsr_usz lin = (rstsr_usz)blockIdx.x * (rstsr_usz)blockDim.x + (rstsr_usz)threadIdx.x;
    if (lin >= n) return;
    rstsr_usz oo = rstsr_plan_offset(lin, op, 0);
    rstsr_usz ia = rstsr_plan_offset(lin, ap, 0);
    rstsr_usz ib = rstsr_plan_offset(lin, bp, 0);
    rstsr_cross3(out + (rstsr_isz)oo, a + (rstsr_isz)ia, b + (rstsr_isz)ib, sa, sb, sc);
}

extern "C" __global__ void rstsr_cross_pn(
    @TO@* out, rstsr_plan_n op,
    const @TA@* a, rstsr_plan_n ap,
    const @TB@* b, rstsr_plan_n bp,
    int n, rstsr_isz sa, rstsr_isz sb, rstsr_isz sc)
{
    int lin = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    int w = gridDim.x * blockDim.x;
    for (int rep = 0; rep < 4; ++rep) {
        if (lin < n) {
            rstsr_isz oo = rstsr_plan_offset_n(lin, op);
            rstsr_isz ia = rstsr_plan_offset_n(lin, ap);
            rstsr_isz ib = rstsr_plan_offset_n(lin, bp);
            rstsr_cross3(out + oo, a + ia, b + ib, sa, sb, sc);
        }
        lin += w;
    }
}

extern "C" __global__ void rstsr_cross_f(
    @TO@* out, const @TA@* a, const @TB@* b,
    rstsr_usz n, rstsr_isz sa, rstsr_isz sb, rstsr_isz sc)
{
    rstsr_usz g = (rstsr_usz)blockIdx.x * (rstsr_usz)blockDim.x + (rstsr_usz)threadIdx.x;
    rstsr_usz w = (rstsr_usz)gridDim.x * (rstsr_usz)blockDim.x;
    #define RSTSR_CSTEP(I) { rstsr_usz oo = g + (I) * w; if (oo < n) { rstsr_cross3(out + (rstsr_isz)oo, a + (rstsr_isz)oo, b + (rstsr_isz)oo, sa, sb, sc); } }
    RSTSR_CSTEP(0) RSTSR_CSTEP(1) RSTSR_CSTEP(2) RSTSR_CSTEP(3)
    #undef RSTSR_CSTEP
}
"#;

const UNARY_ENTRIES: &str = r#"
extern "C" __global__ void rstsr_unary_p(
    @TO@* out, rstsr_plan op,
    const @TB@* b, rstsr_plan bp,
    rstsr_usz n)
{
    rstsr_usz lin = (rstsr_usz)blockIdx.x * (rstsr_usz)blockDim.x + (rstsr_usz)threadIdx.x;
    if (lin >= n) return;
    rstsr_usz oo = rstsr_plan_offset(lin, op, 0);
    rstsr_usz ib = rstsr_plan_offset(lin, bp, 0);
    out[oo] = @EXPR@;
}

extern "C" __global__ void rstsr_unary_pn(
    @TO@* out, rstsr_plan_n op,
    const @TB@* b, rstsr_plan_n bp,
    int n)
{
    int lin = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    int w = gridDim.x * blockDim.x;
    for (int rep = 0; rep < 4; ++rep) {
        if (lin < n) {
            rstsr_isz oo = rstsr_plan_offset_n(lin, op);
            rstsr_isz ib = rstsr_plan_offset_n(lin, bp);
            out[oo] = @EXPR@;
        }
        lin += w;
    }
}

extern "C" __global__ void rstsr_unary_f(
    @TO@* out, const @TB@* b, rstsr_usz n)
{
    rstsr_usz g = (rstsr_usz)blockIdx.x * (rstsr_usz)blockDim.x + (rstsr_usz)threadIdx.x;
    rstsr_usz w = (rstsr_usz)gridDim.x * (rstsr_usz)blockDim.x;
    #define RSTSR_STEP(I) { rstsr_usz oo = g + (I) * w; rstsr_usz ib = oo; if (oo < n) { out[oo] = @EXPR@; } }
    RSTSR_STEP(0) RSTSR_STEP(1) RSTSR_STEP(2) RSTSR_STEP(3)
    #undef RSTSR_STEP
}
"#;

const SCALAR_ENTRIES: &str = r#"
extern "C" __global__ void rstsr_scalar_p(
    @TO@* out, rstsr_plan op,
    const @TT@* t, rstsr_plan tp,
    @TS@ v, rstsr_usz n)
{
    rstsr_usz lin = (rstsr_usz)blockIdx.x * (rstsr_usz)blockDim.x + (rstsr_usz)threadIdx.x;
    if (lin >= n) return;
    rstsr_usz oo = rstsr_plan_offset(lin, op, 0);
    rstsr_usz it = rstsr_plan_offset(lin, tp, 0);
    out[oo] = @EXPR@;
}

extern "C" __global__ void rstsr_scalar_pn(
    @TO@* out, rstsr_plan_n op,
    const @TT@* t, rstsr_plan_n tp,
    @TS@ v, int n)
{
    int lin = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    int w = gridDim.x * blockDim.x;
    for (int rep = 0; rep < 4; ++rep) {
        if (lin < n) {
            rstsr_isz oo = rstsr_plan_offset_n(lin, op);
            rstsr_isz it = rstsr_plan_offset_n(lin, tp);
            out[oo] = @EXPR@;
        }
        lin += w;
    }
}

extern "C" __global__ void rstsr_scalar_f(
    @TO@* out, const @TT@* t, @TS@ v, rstsr_usz n)
{
    rstsr_usz g = (rstsr_usz)blockIdx.x * (rstsr_usz)blockDim.x + (rstsr_usz)threadIdx.x;
    rstsr_usz w = (rstsr_usz)gridDim.x * (rstsr_usz)blockDim.x;
    #define RSTSR_STEP(I) { rstsr_usz oo = g + (I) * w; rstsr_usz it = oo; if (oo < n) { out[oo] = @EXPR@; } }
    RSTSR_STEP(0) RSTSR_STEP(1) RSTSR_STEP(2) RSTSR_STEP(3)
    #undef RSTSR_STEP
}
"#;

// /* #endregion */
// /* #region plan marshaling (kernel-parameter bytes) */
/// Device `rstsr_axis` bytes (32 B: d, s, m, sh, pad).
#[repr(C)]
#[derive(Clone, Copy)]
struct AxisBytes {
    d: u64,
    s: u64,
    m: u64,
    sh: i32,
    _pad: u32,
}

/// Device `rstsr_plan` bytes (136 B: rank, pad, 4 axes). A flat operand
/// marshals as the all-zero struct (`rank == 0` sentinel).
#[repr(C)]
struct PlanBytes {
    rank: u32,
    _pad: u32,
    axes: [AxisBytes; 4],
}

impl PlanBytes {
    fn zeroed() -> Self {
        Self { rank: 0, _pad: 0, axes: [AxisBytes { d: 0, s: 0, m: 0, sh: 0, _pad: 0 }; 4] }
    }

    fn of_params(pack: &super::plan::AxesPack) -> Self {
        let mut axes = [AxisBytes { d: 0, s: 0, m: 0, sh: 0, _pad: 0 }; 4];
        for (ax, &src) in axes.iter_mut().zip(pack.axes.iter()) {
            *ax = AxisBytes { d: src.d, s: src.s, m: src.m, sh: src.sh as i32, _pad: 0 };
        }
        Self { rank: pack.rank as u32, _pad: 0, axes }
    }

    /// Kernel-parameter bytes of this plan.
    fn bytes(&self) -> &[u8] {
        // SAFETY: `PlanBytes` is `repr(C)` with fixed layout matching the
        // device struct; its bytes are the parameter representation
        unsafe { core::slice::from_raw_parts(self as *const Self as *const u8, core::mem::size_of::<Self>()) }
    }
}

/// Device `rstsr_axis_n` bytes (16 B).
#[repr(C)]
#[derive(Clone, Copy)]
struct AxisBytesN {
    d: u32,
    s: i32,
    m: u32,
    sh: i32,
}

/// Device `rstsr_plan_n` bytes (68 B: rank + 4 16-B axes; 4-byte aligned,
/// no tail padding on the device struct). A flat operand marshals as the
/// all-zero struct (`rank == 0` sentinel; the device returns the linear
/// id itself).
#[repr(C)]
struct PlanBytesN {
    rank: u32,
    axes: [AxisBytesN; 4],
}

impl PlanBytesN {
    fn zeroed() -> Self {
        Self { rank: 0, axes: [AxisBytesN { d: 0, s: 0, m: 0, sh: 0 }; 4] }
    }

    fn of_params(pack: &super::plan::AxesPackN) -> Self {
        let mut axes = [AxisBytesN { d: 0, s: 0, m: 0, sh: 0 }; 4];
        for (ax, &src) in axes.iter_mut().zip(pack.axes.iter()) {
            *ax = AxisBytesN { d: src.d, s: src.s, m: src.m, sh: src.sh as i32 };
        }
        Self { rank: pack.rank as u32, axes }
    }

    /// Kernel-parameter bytes of this plan.
    fn bytes(&self) -> &[u8] {
        // SAFETY: `PlanBytesN` is `repr(C)` with fixed layout matching the
        // device struct; its bytes are the parameter representation
        unsafe { core::slice::from_raw_parts(self as *const Self as *const u8, core::mem::size_of::<Self>()) }
    }
}

/// Entry variant for a planned family: all-flat (`_f`), all-narrow
/// (`_pn`; Flat marshals as rank-0), or full 64-bit (`_p`).
enum PlanVariant {
    Flat,
    Narrow,
    Wide,
}

fn plan_variant(plans: &[&OperandPlan]) -> PlanVariant {
    if plans.iter().all(|p| matches!(p, OperandPlan::Flat)) {
        return PlanVariant::Flat;
    }
    if plans.iter().all(|p| matches!(p, OperandPlan::Flat | OperandPlan::ParamsN(_))) {
        return PlanVariant::Narrow;
    }
    PlanVariant::Wide
}

/// Pointer argument folded with the element-unit offset (planned kernels
/// take no separate offset parameters).
fn offset_ptr<T>(exec: &Arc<CudaStreamObj>, buf: &CudaBuffer<T>, offset: usize) -> u64 {
    if buf.is_empty() {
        0
    } else {
        // the read event must land on the launching stream (not the
        // allocation stream), or a cross-stream free can race the kernel
        (buf.device_ptr_on(exec) as u64).wrapping_add(offset as u64 * core::mem::size_of::<T>() as u64)
    }
}

// /* #endregion */
// /* #region launchers */
/// Cheap discriminator of a specialization: every template substitution in
/// order (the template itself is fixed by `name`).
fn subs_disc(subs: &[(&str, &str)]) -> String {
    let mut s = String::with_capacity(subs.len() * 24);
    for (_, v) in subs {
        s.push('|');
        s.push_str(v);
    }
    s
}

/// Compile/fetch (interned) the entry `name` of a planned program.
/// `disc_extra` must name every substitution the rendered source depends on;
/// the source is assembled and NVRTC-compiled only on an intern miss.
fn planned_function(
    device: &DeviceCuda,
    name: &'static str,
    includes: &[&str],
    disc_extra: &str,
    entries: impl FnOnce() -> String,
) -> Result<std::sync::Arc<cudarc::driver::CudaFunction>> {
    let disc = format!("{name}|{}|{disc_extra}", includes.concat());
    let cache = device.ordinal_obj().planned_cache();
    if let Some(f) = cache.entries.lock().unwrap_or_else(|e| e.into_inner()).get(&disc) {
        return Ok(f.clone());
    }
    let src = assemble_planned(includes, &entries());
    let key = make_key(device.ordinal_obj(), name, &src, &[])?;
    let f = device.ordinal_obj().kernel_function(&key)?;
    cache.entries.lock().unwrap_or_else(|e| e.into_inner()).insert(disc, f.clone());
    Ok(f)
}

/// Planned binary elementwise launch; falls back to the rank-generic
/// kernel when any operand exceeds the parameterized walk.
#[allow(clippy::too_many_arguments)]
pub(crate) fn binary_launch<TOUT, TA, TB>(
    device: &DeviceCuda,
    c: &mut CudaBuffer<MaybeUninit<TOUT>>,
    lc: &Layout<impl DimAPI>,
    a: &CudaBuffer<TA>,
    la: &Layout<impl DimAPI>,
    b: &CudaBuffer<TB>,
    lb: &Layout<impl DimAPI>,
    expr: &str,
) -> Result<()>
where
    TOUT: CudaDType,
    TA: CudaDType,
    TB: CudaDType,
{
    let n = lc.size();
    if n == 0 {
        return Ok(());
    }
    let order = device.default_order();
    let po = plan_operand(lc, order, n);
    let pa = plan_operand(la, order, n);
    let pb = plan_operand(lb, order, n);
    if po.is_generic() || pa.is_generic() || pb.is_generic() {
        return kernels_ops::launch_binary_generic(device, c, lc, a, la, b, lb, expr);
    }
    let exec = device.stream();
    let _ga = order_input(a, exec)?;
    let _gb = order_input(b, exec)?;
    let variant = plan_variant(&[&po, &pa, &pb]);
    let entry = match variant {
        PlanVariant::Flat => "rstsr_binary_f",
        PlanVariant::Narrow => "rstsr_binary_pn",
        PlanVariant::Wide => "rstsr_binary_p",
    };
    let includes = [TA::INCLUDES, TB::INCLUDES, TOUT::INCLUDES].concat();
    let subs = [("@TO@", TOUT::CNAME), ("@TA@", TA::CNAME), ("@TB@", TB::CNAME), ("@EXPR@", expr)];
    let f = planned_function(device, entry, &includes, &subs_disc(&subs), || render(BINARY_ENTRIES, &subs))?;
    // SAFETY: MaybeUninit<TOUT> and TOUT are layout-identical; the kernel
    // writes every visited element
    let c_typed: &mut CudaBuffer<TOUT> =
        unsafe { &mut *(c as *mut CudaBuffer<MaybeUninit<TOUT>> as *mut CudaBuffer<TOUT>) };
    let mut launch = FlatLaunch::new(exec, &f, n);
    match variant {
        PlanVariant::Flat => {
            launch = launch
                .elems_per_thread(4)
                .arg_u64(offset_ptr(exec, c_typed, lc.offset()))
                .arg_u64(offset_ptr(exec, a, la.offset()))
                .arg_u64(offset_ptr(exec, b, lb.offset()))
                .arg_usize(n);
        },
        PlanVariant::Narrow => {
            let zn = PlanBytesN::zeroed();
            let (nop, nap, nbp) = (plan_bytes_n(&po), plan_bytes_n(&pa), plan_bytes_n(&pb));
            launch = launch
                .elems_per_thread(4)
                .arg_u64(offset_ptr(exec, c_typed, lc.offset()))
                .arg_bytes(nop.as_ref().unwrap_or(&zn).bytes())
                .arg_u64(offset_ptr(exec, a, la.offset()))
                .arg_bytes(nap.as_ref().unwrap_or(&zn).bytes())
                .arg_u64(offset_ptr(exec, b, lb.offset()))
                .arg_bytes(nbp.as_ref().unwrap_or(&zn).bytes())
                .arg_i32(n as i32);
        },
        PlanVariant::Wide => {
            let zero = PlanBytes::zeroed();
            let (bop, bap, bbp) = (plan_bytes(&po), plan_bytes(&pa), plan_bytes(&pb));
            launch = launch
                .arg_u64(offset_ptr(exec, c_typed, lc.offset()))
                .arg_bytes(bop.as_ref().unwrap_or(&zero).bytes())
                .arg_u64(offset_ptr(exec, a, la.offset()))
                .arg_bytes(bap.as_ref().unwrap_or(&zero).bytes())
                .arg_u64(offset_ptr(exec, b, lb.offset()))
                .arg_bytes(bbp.as_ref().unwrap_or(&zero).bytes())
                .arg_usize(n);
        },
    }
    launch.launch()?;
    // write ordering for an output buffer not allocated on this stream
    let _go = crate::kernel::order_output(c, exec)?;
    Ok(())
}

/// Collapse `axis` to shape 1 / stride 0: one visited slot becomes one vector, and the three
/// components are then reached by the axis stride the caller captured beforehand.
fn collapse_axis<D: DimAPI>(l: &Layout<D>, axis: usize) -> Result<Layout<D>> {
    let mut shape = l.shape().clone();
    let mut stride = l.stride().clone();
    shape[axis] = 1;
    stride[axis] = 0;
    Layout::new(shape, stride, l.offset())
}

/// Planned cross-product launch over one size-3 `axis`.
///
/// Refuses the rank-generic plan walk for now: that path needs a four-input, three-output generic
/// kernel of its own, and the shapes that reach it (very high rank or offsets past 32 bits) are not
/// what `cross` is used for. A named error beats a silently wrong answer.
#[allow(clippy::too_many_arguments)]
pub(crate) fn cross_launch<TOUT, TA, TB>(
    device: &DeviceCuda,
    c: &mut CudaBuffer<MaybeUninit<TOUT>>,
    lc: &Layout<impl DimAPI>,
    a: &CudaBuffer<TA>,
    la: &Layout<impl DimAPI>,
    b: &CudaBuffer<TB>,
    lb: &Layout<impl DimAPI>,
    axis: usize,
) -> Result<()>
where
    TOUT: CudaDType,
    TA: CudaDType,
    TB: CudaDType,
{
    // read the axis strides before collapsing: these are what select the three components
    let (sa, sb, sc) = (la.stride()[axis], lb.stride()[axis], lc.stride()[axis]);
    let lcc = collapse_axis(lc, axis)?;
    let lac = collapse_axis(la, axis)?;
    let lbc = collapse_axis(lb, axis)?;

    let n = lcc.size();
    if n == 0 {
        return Ok(());
    }

    let order = device.default_order();
    let po = plan_operand(&lcc, order, n);
    let pa = plan_operand(&lac, order, n);
    let pb = plan_operand(&lbc, order, n);
    if po.is_generic() || pa.is_generic() || pb.is_generic() {
        rstsr_raise!(
            UnImplemented,
            "cuda cross: an operand needs the rank-generic plan walk, which this kernel does not have"
        )?;
    }

    let exec = device.stream();
    let _ga = order_input(a, exec)?;
    let _gb = order_input(b, exec)?;

    let variant = plan_variant(&[&po, &pa, &pb]);
    let entry = match variant {
        PlanVariant::Flat => "rstsr_cross_f",
        PlanVariant::Narrow => "rstsr_cross_pn",
        PlanVariant::Wide => "rstsr_cross_p",
    };
    let includes = [TA::INCLUDES, TB::INCLUDES, TOUT::INCLUDES].concat();
    let subs = [("@TO@", TOUT::CNAME), ("@TA@", TA::CNAME), ("@TB@", TB::CNAME)];
    let f = planned_function(device, entry, &includes, &subs_disc(&subs), || {
        render(CROSS_ENTRIES, &subs)
    })?;

    // SAFETY: `MaybeUninit<TOUT>` and `TOUT` are layout-identical; the kernel writes all three
    // components of every visited slot, and `lc` holds exactly those slots.
    let c_typed: &mut CudaBuffer<TOUT> =
        unsafe { &mut *(c as *mut CudaBuffer<MaybeUninit<TOUT>> as *mut CudaBuffer<TOUT>) };

    let mut launch = FlatLaunch::new(exec, &f, n);
    match variant {
        PlanVariant::Flat => {
            launch = launch
                .elems_per_thread(4)
                .arg_u64(offset_ptr(exec, c_typed, lc.offset()))
                .arg_u64(offset_ptr(exec, a, la.offset()))
                .arg_u64(offset_ptr(exec, b, lb.offset()))
                .arg_usize(n)
                .arg_bytes(&sa.to_ne_bytes())
                .arg_bytes(&sb.to_ne_bytes())
                .arg_bytes(&sc.to_ne_bytes());
        },
        PlanVariant::Narrow => {
            let zn = PlanBytesN::zeroed();
            let (nop, nap, nbp) = (plan_bytes_n(&po), plan_bytes_n(&pa), plan_bytes_n(&pb));
            launch = launch
                .elems_per_thread(4)
                .arg_u64(offset_ptr(exec, c_typed, lc.offset()))
                .arg_bytes(nop.as_ref().unwrap_or(&zn).bytes())
                .arg_u64(offset_ptr(exec, a, la.offset()))
                .arg_bytes(nap.as_ref().unwrap_or(&zn).bytes())
                .arg_u64(offset_ptr(exec, b, lb.offset()))
                .arg_bytes(nbp.as_ref().unwrap_or(&zn).bytes())
                .arg_i32(n as i32)
                .arg_bytes(&sa.to_ne_bytes())
                .arg_bytes(&sb.to_ne_bytes())
                .arg_bytes(&sc.to_ne_bytes());
        },
        PlanVariant::Wide => {
            let zero = PlanBytes::zeroed();
            let (bop, bap, bbp) = (plan_bytes(&po), plan_bytes(&pa), plan_bytes(&pb));
            launch = launch
                .arg_u64(offset_ptr(exec, c_typed, lc.offset()))
                .arg_bytes(bop.as_ref().unwrap_or(&zero).bytes())
                .arg_u64(offset_ptr(exec, a, la.offset()))
                .arg_bytes(bap.as_ref().unwrap_or(&zero).bytes())
                .arg_u64(offset_ptr(exec, b, lb.offset()))
                .arg_bytes(bbp.as_ref().unwrap_or(&zero).bytes())
                .arg_usize(n)
                .arg_bytes(&sa.to_ne_bytes())
                .arg_bytes(&sb.to_ne_bytes())
                .arg_bytes(&sc.to_ne_bytes());
        },
    }
    launch.launch()?;

    let _go = crate::kernel::order_output(c, exec)?;
    Ok(())
}

/// Planned unary elementwise launch.
pub(crate) fn unary_launch<TOUT, TB>(
    device: &DeviceCuda,
    c: &mut CudaBuffer<MaybeUninit<TOUT>>,
    lc: &Layout<impl DimAPI>,
    b: &CudaBuffer<TB>,
    lb: &Layout<impl DimAPI>,
    expr: &str,
) -> Result<()>
where
    TOUT: CudaDType,
    TB: CudaDType,
{
    let n = lc.size();
    if n == 0 {
        return Ok(());
    }
    let order = device.default_order();
    let po = plan_operand(lc, order, n);
    let pb = plan_operand(lb, order, n);
    if po.is_generic() || pb.is_generic() {
        return kernels_ops::launch_unary_generic(device, c, lc, b, lb, expr);
    }
    let exec = device.stream();
    let _gb = order_input(b, exec)?;
    let variant = plan_variant(&[&po, &pb]);
    let entry = match variant {
        PlanVariant::Flat => "rstsr_unary_f",
        PlanVariant::Narrow => "rstsr_unary_pn",
        PlanVariant::Wide => "rstsr_unary_p",
    };
    let includes = [TB::INCLUDES, TOUT::INCLUDES].concat();
    let subs = [("@TO@", TOUT::CNAME), ("@TB@", TB::CNAME), ("@EXPR@", expr)];
    let f = planned_function(device, entry, &includes, &subs_disc(&subs), || render(UNARY_ENTRIES, &subs))?;
    // SAFETY: layout-identical MaybeUninit transmute (crate invariant)
    let c_typed: &mut CudaBuffer<TOUT> =
        unsafe { &mut *(c as *mut CudaBuffer<MaybeUninit<TOUT>> as *mut CudaBuffer<TOUT>) };
    let mut launch = FlatLaunch::new(exec, &f, n);
    match variant {
        PlanVariant::Flat => {
            launch = launch
                .elems_per_thread(4)
                .arg_u64(offset_ptr(exec, c_typed, lc.offset()))
                .arg_u64(offset_ptr(exec, b, lb.offset()))
                .arg_usize(n);
        },
        PlanVariant::Narrow => {
            let zn = PlanBytesN::zeroed();
            let (nop, nbp) = (plan_bytes_n(&po), plan_bytes_n(&pb));
            launch = launch
                .elems_per_thread(4)
                .arg_u64(offset_ptr(exec, c_typed, lc.offset()))
                .arg_bytes(nop.as_ref().unwrap_or(&zn).bytes())
                .arg_u64(offset_ptr(exec, b, lb.offset()))
                .arg_bytes(nbp.as_ref().unwrap_or(&zn).bytes())
                .arg_i32(n as i32);
        },
        PlanVariant::Wide => {
            let zero = PlanBytes::zeroed();
            launch = launch
                .arg_u64(offset_ptr(exec, c_typed, lc.offset()))
                .arg_bytes(plan_bytes(&po).as_ref().unwrap_or(&zero).bytes())
                .arg_u64(offset_ptr(exec, b, lb.offset()))
                .arg_bytes(plan_bytes(&pb).as_ref().unwrap_or(&zero).bytes())
                .arg_usize(n);
        },
    }
    launch.launch()?;
    let _go = crate::kernel::order_output(c, exec)?;
    Ok(())
}

/// Planned in-place unary launch: the buffer is both destination and
/// source (each thread reads and writes its own element).
pub(crate) fn unary_launch_inplace<T>(
    device: &DeviceCuda,
    a: &mut CudaBuffer<T>,
    la: &Layout<impl DimAPI>,
    expr: &str,
) -> Result<()>
where
    T: CudaDType,
{
    let n = la.size();
    if n == 0 {
        return Ok(());
    }
    let order = device.default_order();
    let pa = plan_operand(la, order, n);
    if pa.is_generic() {
        return kernels_ops::unary_float_inplace(device, a, la, expr);
    }
    let exec = device.stream();
    let _ga = order_input(a, exec)?;
    let includes = T::INCLUDES;
    let variant = plan_variant(&[&pa]);
    let entry = match variant {
        PlanVariant::Flat | PlanVariant::Narrow => {
            if matches!(pa, OperandPlan::Flat) {
                "rstsr_unary_inplace"
            } else {
                "rstsr_unary_inplace_n"
            }
        },
        PlanVariant::Wide => "rstsr_unary_inplace",
    };
    let subs = [("@T@", T::CNAME), ("@EXPR@", expr)];
    let f = planned_function(device, entry, includes, &subs_disc(&subs), || render(UNARY_INPLACE_ENTRIES, &subs))?;
    let mut launch = FlatLaunch::new(exec, &f, n);
    match (variant, &pa) {
        (PlanVariant::Wide, _) | (PlanVariant::Flat, OperandPlan::Flat) => {
            let zero = PlanBytes::zeroed();
            launch = launch
                .arg_u64(offset_ptr(exec, a, la.offset()))
                .arg_bytes(plan_bytes(&pa).as_ref().unwrap_or(&zero).bytes())
                .arg_usize(n);
        },
        _ => {
            let zn = PlanBytesN::zeroed();
            launch = launch
                .elems_per_thread(4)
                .arg_u64(offset_ptr(exec, a, la.offset()))
                .arg_bytes(plan_bytes_n(&pa).as_ref().unwrap_or(&zn).bytes())
                .arg_i32(n as i32);
        },
    }
    launch.launch()?;
    let _go = crate::kernel::order_output(a, exec)?;
    Ok(())
}

const UNARY_INPLACE_ENTRIES: &str = r#"
extern "C" __global__ void rstsr_unary_inplace(
    @T@* a, rstsr_plan ap, rstsr_usz n)
{
    rstsr_usz lin = (rstsr_usz)blockIdx.x * (rstsr_usz)blockDim.x + (rstsr_usz)threadIdx.x;
    if (lin >= n) return;
    rstsr_usz io = rstsr_plan_offset(lin, ap, 0);
    a[io] = @EXPR@;
}

extern "C" __global__ void rstsr_unary_inplace_n(
    @T@* a, rstsr_plan_n ap, int n)
{
    int lin = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    int w = gridDim.x * blockDim.x;
    for (int rep = 0; rep < 4; ++rep) {
        if (lin < n) {
            rstsr_isz io = rstsr_plan_offset_n(lin, ap);
            a[io] = @EXPR@;
        }
        lin += w;
    }
}
"#;

/// Planned tensor-scalar elementwise launch (scalar rides in the
/// parameter list; the output and tensor operands are planned).
pub(crate) fn scalar_launch<TOUT, TT, TS>(
    device: &DeviceCuda,
    c: &mut CudaBuffer<MaybeUninit<TOUT>>,
    lc: &Layout<impl DimAPI>,
    t: &CudaBuffer<TT>,
    lt: &Layout<impl DimAPI>,
    s: &TS,
    expr: &str,
) -> Result<()>
where
    TOUT: CudaDType,
    TT: CudaDType,
    TS: CudaDType,
{
    let n = lc.size();
    if n == 0 {
        return Ok(());
    }
    let order = device.default_order();
    let po = plan_operand(lc, order, n);
    let pt = plan_operand(lt, order, n);
    if po.is_generic() || pt.is_generic() {
        return kernels_ops::launch_scalar_generic(device, c, lc, t, lt, s, expr);
    }
    let exec = device.stream();
    let _gt = order_input(t, exec)?;
    let variant = plan_variant(&[&po, &pt]);
    let entry = match variant {
        PlanVariant::Flat => "rstsr_scalar_f",
        PlanVariant::Narrow => "rstsr_scalar_pn",
        PlanVariant::Wide => "rstsr_scalar_p",
    };
    let includes = [TT::INCLUDES, TS::INCLUDES, TOUT::INCLUDES].concat();
    let subs = [("@TO@", TOUT::CNAME), ("@TT@", TT::CNAME), ("@TS@", TS::CNAME), ("@EXPR@", expr)];
    let f = planned_function(device, entry, &includes, &subs_disc(&subs), || render(SCALAR_ENTRIES, &subs))?;
    // SAFETY: layout-identical MaybeUninit transmute (crate invariant)
    let c_typed: &mut CudaBuffer<TOUT> =
        unsafe { &mut *(c as *mut CudaBuffer<MaybeUninit<TOUT>> as *mut CudaBuffer<TOUT>) };
    let nbytes = core::mem::size_of::<TS>();
    let mut sbytes = vec![0u8; nbytes];
    // SAFETY: TS is a plain scalar; its bytes are its on-device representation
    unsafe {
        core::ptr::copy_nonoverlapping(core::ptr::addr_of!(*s) as *const u8, sbytes.as_mut_ptr(), nbytes);
    }
    let mut launch = FlatLaunch::new(exec, &f, n);
    match variant {
        PlanVariant::Flat => {
            launch = launch
                .elems_per_thread(4)
                .arg_u64(offset_ptr(exec, c_typed, lc.offset()))
                .arg_u64(offset_ptr(exec, t, lt.offset()))
                .arg_bytes(&sbytes)
                .arg_usize(n);
        },
        PlanVariant::Narrow => {
            let zn = PlanBytesN::zeroed();
            let (nop, ntp) = (plan_bytes_n(&po), plan_bytes_n(&pt));
            launch = launch
                .elems_per_thread(4)
                .arg_u64(offset_ptr(exec, c_typed, lc.offset()))
                .arg_bytes(nop.as_ref().unwrap_or(&zn).bytes())
                .arg_u64(offset_ptr(exec, t, lt.offset()))
                .arg_bytes(ntp.as_ref().unwrap_or(&zn).bytes())
                .arg_bytes(&sbytes)
                .arg_i32(n as i32);
        },
        PlanVariant::Wide => {
            let zero = PlanBytes::zeroed();
            launch = launch
                .arg_u64(offset_ptr(exec, c_typed, lc.offset()))
                .arg_bytes(plan_bytes(&po).as_ref().unwrap_or(&zero).bytes())
                .arg_u64(offset_ptr(exec, t, lt.offset()))
                .arg_bytes(plan_bytes(&pt).as_ref().unwrap_or(&zero).bytes())
                .arg_bytes(&sbytes)
                .arg_usize(n);
        },
    }
    launch.launch()?;
    let _go = crate::kernel::order_output(c, exec)?;
    Ok(())
}

// /* #endregion */
// /* #region fill + copy (planned) */
const FILL_ENTRIES: &str = r#"
extern "C" __global__ void rstsr_fill_p(
    @TC@* out, rstsr_plan op, @TV@ v, rstsr_usz n)
{
    rstsr_usz lin = (rstsr_usz)blockIdx.x * (rstsr_usz)blockDim.x + (rstsr_usz)threadIdx.x;
    if (lin >= n) return;
    rstsr_usz oo = rstsr_plan_offset(lin, op, 0);
    out[oo] = v;
}

extern "C" __global__ void rstsr_fill_f(
    @TC@* out, @TV@ v, rstsr_usz n)
{
    rstsr_usz g = (rstsr_usz)blockIdx.x * (rstsr_usz)blockDim.x + (rstsr_usz)threadIdx.x;
    rstsr_usz w = (rstsr_usz)gridDim.x * (rstsr_usz)blockDim.x;
    #define RSTSR_STEP(I) { rstsr_usz oo = g + (I) * w; if (oo < n) { out[oo] = v; } }
    RSTSR_STEP(0) RSTSR_STEP(1) RSTSR_STEP(2) RSTSR_STEP(3)
    #undef RSTSR_STEP
}

// vector fill: a two-element struct store per pair (16 B for f64, 8 B for
// f32, wider for complex), scalar remainder loop for the tail.
extern "C" __global__ void rstsr_fill_v(
    @TC@* out, @TV@ v, rstsr_usz nv4, rstsr_usz n)
{
    rstsr_usz g = (rstsr_usz)blockIdx.x * (rstsr_usz)blockDim.x + (rstsr_usz)threadIdx.x;
    rstsr_usz w = (rstsr_usz)gridDim.x * (rstsr_usz)blockDim.x;
    typedef struct { @TC@ x, y; } rstsr_pair2;
    rstsr_pair2 p2; p2.x = v; p2.y = v;
    #define RSTSR_STEP(I) { rstsr_usz oo = g + (I) * w; rstsr_usz o2 = oo * 2; if (oo < nv4 && o2 + 1 < n) { *((rstsr_pair2*)(out + o2)) = p2; } }
    RSTSR_STEP(0) RSTSR_STEP(1) RSTSR_STEP(2) RSTSR_STEP(3)
    #undef RSTSR_STEP
    rstsr_usz tail = nv4 * 2;
    for (rstsr_usz oo = tail + g; oo < n; oo += w) out[oo] = v;
}

extern "C" __global__ void rstsr_fill_pn(
    @TC@* out, rstsr_plan_n op, @TV@ v, int n)
{
    int lin = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    int w = gridDim.x * blockDim.x;
    for (int rep = 0; rep < 4; ++rep) {
        if (lin < n) {
            rstsr_isz oo = rstsr_plan_offset_n(lin, op);
            out[oo] = v;
        }
        lin += w;
    }
}
"#;

/// Planned strided fill (the `rstsr_fill_cast` generic kernel remains for
/// cross-dtype fill values, which need the in-kernel cast expression).
pub(crate) fn fill_launch<TC, TV>(
    device: &DeviceCuda,
    c: &mut CudaBuffer<TC>,
    lc: &Layout<impl DimAPI>,
    v: &TV,
) -> Result<()>
where
    TC: CudaDType,
    TV: CudaDType,
{
    let n = lc.size();
    if n == 0 {
        return Ok(());
    }
    let po = plan_operand(lc, device.default_order(), n);
    if po.is_generic() {
        // rank-generic strided kernel; must not re-enter this function
        return kernels_ops::fill_generic_same(device, c, lc, v);
    }
    let exec = device.stream();
    let _gc = order_input(c, exec)?;
    let variant = plan_variant(&[&po]);
    // flat fill of a whole 16-B-aligned buffer takes the vector entry
    let is_flat_all = matches!(po, OperandPlan::Flat) && lc.offset() == 0;
    let vec_fill = is_flat_all && n >= 1024 && n % 2 == 0;
    let entry = match variant {
        PlanVariant::Flat if vec_fill => "rstsr_fill_v",
        PlanVariant::Flat => "rstsr_fill_f",
        PlanVariant::Narrow => "rstsr_fill_pn",
        PlanVariant::Wide => "rstsr_fill_p",
    };
    let includes = [TC::INCLUDES, TV::INCLUDES].concat();
    let subs = [("@TC@", TC::CNAME), ("@TV@", TV::CNAME)];
    let f = planned_function(device, entry, &includes, &subs_disc(&subs), || render(FILL_ENTRIES, &subs))?;
    let nbytes = core::mem::size_of::<TV>();
    let mut vbytes = vec![0u8; nbytes];
    // SAFETY: TV is a plain scalar; its bytes are its on-device representation
    unsafe {
        core::ptr::copy_nonoverlapping(core::ptr::addr_of!(*v) as *const u8, vbytes.as_mut_ptr(), nbytes);
    }
    #[cfg(feature = "launch_trace")]
    eprintln!("[rstsr-cuda] fill entry={entry} n={n}");
    // the vector entry works in pairs; size the grid from the pair count
    let launch_n = if vec_fill { n / 2 } else { n };
    let mut launch = FlatLaunch::new(exec, &f, launch_n);
    match variant {
        PlanVariant::Flat if vec_fill => {
            // one quad-store per thread slot; scalar remainder rides the tail loop
            launch = launch
                .elems_per_thread(4)
                .arg_u64(offset_ptr(exec, c, lc.offset()))
                .arg_bytes(&vbytes)
                .arg_usize(n / 2)
                .arg_usize(n);
        },
        PlanVariant::Flat => {
            launch =
                launch.elems_per_thread(4).arg_u64(offset_ptr(exec, c, lc.offset())).arg_bytes(&vbytes).arg_usize(n);
        },
        PlanVariant::Narrow => {
            let zn = PlanBytesN::zeroed();
            launch = launch
                .elems_per_thread(4)
                .arg_u64(offset_ptr(exec, c, lc.offset()))
                .arg_bytes(plan_bytes_n(&po).as_ref().unwrap_or(&zn).bytes())
                .arg_bytes(&vbytes)
                .arg_i32(n as i32);
        },
        PlanVariant::Wide => {
            let zero = PlanBytes::zeroed();
            launch = launch
                .arg_u64(offset_ptr(exec, c, lc.offset()))
                .arg_bytes(plan_bytes(&po).as_ref().unwrap_or(&zero).bytes())
                .arg_bytes(&vbytes)
                .arg_usize(n);
        },
    }
    launch.launch()?;
    let _go = crate::kernel::order_output(c, exec)?;
    Ok(())
}

const COPY_ENTRIES: &str = r#"
extern "C" __global__ void rstsr_copy_p(
    @TC@* out, rstsr_plan op, const @TA@* a, rstsr_plan ap, rstsr_usz n)
{
    rstsr_usz lin = (rstsr_usz)blockIdx.x * (rstsr_usz)blockDim.x + (rstsr_usz)threadIdx.x;
    if (lin >= n) return;
    rstsr_usz oo = rstsr_plan_offset(lin, op, 0);
    rstsr_usz ia = rstsr_plan_offset(lin, ap, 0);
    out[oo] = (@CAST@);
}

extern "C" __global__ void rstsr_copy_f(
    @TC@* out, const @TA@* a, rstsr_usz n)
{
    rstsr_usz g = (rstsr_usz)blockIdx.x * (rstsr_usz)blockDim.x + (rstsr_usz)threadIdx.x;
    rstsr_usz w = (rstsr_usz)gridDim.x * (rstsr_usz)blockDim.x;
    #define RSTSR_STEP(I) { rstsr_usz oo = g + (I) * w; rstsr_usz ia = oo; if (oo < n) { out[oo] = (@CAST@); } }
    RSTSR_STEP(0) RSTSR_STEP(1) RSTSR_STEP(2) RSTSR_STEP(3)
    #undef RSTSR_STEP
}

extern "C" __global__ void rstsr_copy_pn(
    @TC@* out, rstsr_plan_n op, const @TA@* a, rstsr_plan_n ap, int n)
{
    int lin = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    int w = gridDim.x * blockDim.x;
    for (int rep = 0; rep < 4; ++rep) {
        if (lin < n) {
            rstsr_isz oo = rstsr_plan_offset_n(lin, op);
            rstsr_isz ia = rstsr_plan_offset_n(lin, ap);
            out[oo] = (@CAST@);
        }
        lin += w;
    }
}
"#;

/// Tiled 2-D transpose copy for same-dtype axis-swapped operands: shared-
/// memory tile (32x32, +1 padded row against bank conflicts) with coalesced
/// reads AND writes on both sides (algorithm per the CUDA Samples
/// `transpose.cu` diagonal-tile kernel; the same tiling PyTorch's
/// TensorIterator copy and cublas geam use for 2-D transposes).
const COPY_TRANSPOSE_ENTRIES: &str = r#"
#define RSTSR_TILE 32
extern "C" __global__ void rstsr_copy_transpose(
    @TC@* out, const @TA@* in, rstsr_usz rows, rstsr_usz cols, int col_major, rstsr_usz n)
{
    // `in` is the contiguous buffer laid out as [rows, cols] in the device
    // order; `out` is the same data as [cols, rows] in that order. Tile
    // staging is order-independent; only the linear addresses change
    // (row-major `r*ld + c`, column-major `r + c*ld`). The +1 tile column
    // keeps the shared-memory bank layout conflict-free for both accesses.
    __shared__ @TA@ tile[RSTSR_TILE][RSTSR_TILE + 1];
    rstsr_usz c0 = (rstsr_usz)blockIdx.x * RSTSR_TILE;
    rstsr_usz r0 = (rstsr_usz)blockIdx.y * RSTSR_TILE;
    rstsr_usz tx = threadIdx.x, ty = threadIdx.y;
    rstsr_usz r = r0 + ty, c = c0 + tx;
    if (r < rows && c < cols) {
        rstsr_usz ii = col_major ? (r + c * rows) : (r * cols + c);
        tile[ty][tx] = in[ii];
    }
    __syncthreads();
    r = c0 + ty;
    c = r0 + tx;
    if (r < cols && c < rows) {
        rstsr_usz oo = col_major ? (r + c * cols) : (r * rows + c);
        if (oo < n) out[oo] = tile[tx][ty];
    }
}
#undef RSTSR_TILE
"#;

/// Planned same-shape strided copy/cast (`out[oo] = cast(a[ia])`); the
/// rank/shape-changing positional-zip assigns stay on the generic kernel.
/// `cast_expr` is the full element expression from `a[ia]` to TC.
/// Orientation of a 2-D transpose copy. `OutContigInSwap`: the output
/// layout is contiguous in the device order and the input is its axis-swap
/// (the `to_contig(T)` direction); `InContigOutSwap`: the mirror
/// (assign-into-transposed-view direction). Both use one tiled kernel whose
/// guards cover non-multiple-of-32 margins.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Transpose2D {
    OutContigInSwap { rows: usize, cols: usize },
    InContigOutSwap { rows: usize, cols: usize },
}

/// Recognize a 2-D axis-swapped same-shape pair: one operand row-major
/// contiguous, the other striding (1, cols) (pure 2-D swapaxes). Any
/// offset is fine (folded into the base pointer).
fn transpose2d_plan<D1: DimAPI, D2: DimAPI>(lc: &Layout<D1>, la: &Layout<D2>, order: FlagOrder) -> Option<Transpose2D> {
    let (sc, sa) = (lc.shape().as_ref(), la.shape().as_ref());
    let (tc, ta) = (lc.stride().as_ref(), la.stride().as_ref());
    if sc.len() != 2 || sa.len() != 2 || sc != sa {
        return None;
    }
    let contig = |st: &[isize], sh: &[usize]| match order {
        RowMajor => st == [sh[1] as isize, 1],
        ColMajor => st == [1, sh[0] as isize],
    };
    let swapped = |st: &[isize], sh: &[usize]| match order {
        RowMajor => st == [1, sh[0] as isize],
        ColMajor => st == [sh[1] as isize, 1],
    };
    let (r, c) = (sc[0], sc[1]);
    let candidate = if contig(tc, sc) && swapped(ta, sa) && c >= r {
        Transpose2D::OutContigInSwap { rows: r, cols: c }
    } else if swapped(tc, sc) && contig(ta, sa) {
        Transpose2D::InContigOutSwap { rows: r, cols: c }
    } else {
        return None;
    };
    // grid.y is capped at 65535 tiles of 32 rows; taller cases fall back
    // to the planned copy rather than fail the launch
    let t_rows = match candidate {
        Transpose2D::OutContigInSwap { cols, .. } => cols,
        Transpose2D::InContigOutSwap { rows, .. } => rows,
    };
    if t_rows > 65535 * 32 {
        return None;
    }
    Some(candidate)
}

/// Tiled 2-D transpose copy launch (both orientations; grid over output
/// tiles of 32x32, one value per shared-memory stage).
fn copy_transpose_launch<TC, TA>(
    device: &DeviceCuda,
    c: &mut CudaBuffer<TC>,
    lc: &Layout<impl DimAPI>,
    a: &CudaBuffer<TA>,
    la: &Layout<impl DimAPI>,
    plan: Transpose2D,
    n: usize,
) -> Result<()>
where
    TC: CudaDType,
    TA: CudaDType,
{
    let exec = device.stream();
    let _gc = order_input(c, exec)?;
    let _ga = order_input(a, exec)?;
    // Kernel roles (device `in` = the CONTIGUOUS buffer, read coalesced;
    // device `out` = the swapped-view buffer, written transposed):
    //   InContigOutSwap { rows: m, cols: n }: `a` is contiguous [m, n];
    //     kernel in=(m, n), out = c-buffer.
    //   OutContigInSwap { rows: m, cols: n }: `c` is contiguous [m, n];
    //     the swapped input view(r, c) = a_buf[c * m + r], i.e. a_buf holds
    //     the transpose data as [n, m] row-major: kernel in=(n, m) (a-buffer
    //     read row-major), out = c-buffer.
    let (t_rows, t_cols, out_ptr, in_ptr) = match plan {
        Transpose2D::OutContigInSwap { rows, cols } => {
            (cols, rows, offset_ptr(exec, c, lc.offset()), offset_ptr(exec, a, la.offset()))
        },
        Transpose2D::InContigOutSwap { rows, cols } => {
            (rows, cols, offset_ptr(exec, c, lc.offset()), offset_ptr(exec, a, la.offset()))
        },
    };
    let includes = [TC::INCLUDES, TA::INCLUDES].concat();
    let subs = [("@TC@", TC::CNAME), ("@TA@", TA::CNAME)];
    let f = planned_function(device, "rstsr_copy_transpose", &includes, &subs_disc(&subs), || {
        render(COPY_TRANSPOSE_ENTRIES, &subs)
    })?;
    let grid = (t_cols.div_ceil(32) as u32, t_rows.div_ceil(32) as u32, 1);
    let col_major = (device.default_order() == ColMajor) as i32;
    let launch = FlatLaunch::new(exec, &f, 0)
        .explicit_shape(grid, (32, 32, 1))
        .arg_u64(out_ptr)
        .arg_u64(in_ptr)
        .arg_usize(t_rows)
        .arg_usize(t_cols)
        .arg_i32(col_major)
        .arg_usize(n);
    launch.launch()?;
    let _go = crate::kernel::order_output(c, exec)?;
    Ok(())
}

pub(crate) fn copy_launch<TC, TA>(
    device: &DeviceCuda,
    c: &mut CudaBuffer<TC>,
    lc: &Layout<impl DimAPI>,
    a: &CudaBuffer<TA>,
    la: &Layout<impl DimAPI>,
    cast_expr: &str,
) -> Result<()>
where
    TC: CudaDType,
    TA: CudaDType,
{
    let n = lc.size();
    if n == 0 {
        return Ok(());
    }
    // A non-identity copy into the same allocation (e.g. an in-place
    // transpose) would let threads overwrite source elements other threads
    // have not read yet; the safe tensor API cannot build this, but a raw /
    // Cow repr can.
    if !c.is_empty()
        && !a.is_empty()
        && std::ptr::eq(c.device_ptr() as *const (), a.device_ptr() as *const ())
        && (lc.offset() != la.offset()
            || lc.shape().as_ref() != la.shape().as_ref()
            || lc.stride().as_ref() != la.stride().as_ref())
    {
        rstsr_raise!(InvalidValue, "copy source and destination alias with different layouts")?;
    }
    // positional zip is only guaranteed shape-equal here; everything else
    // (reshape-class assigns) belongs to the generic kernel
    if lc.shape().as_ref() != la.shape().as_ref() {
        return kernels_ops::copy_dispatch_generic(device, c, lc, a, la, cast_expr);
    }
    let order = device.default_order();
    // tiled 2-D transpose fast path: same dtype, plain copy, 2-D shape
    // [m, n] where one operand reads along its contiguous axis and the
    // other is the axis-swap. Handles both directions (transposed-read and
    // transposed-write); sub-tile margins fall back to the gather kernel.
    if cast_expr == "a[ia]" && core::mem::size_of::<TC>() == core::mem::size_of::<TA>() {
        if let Some(plan) = transpose2d_plan(lc, la, order) {
            if n >= 4096 {
                return copy_transpose_launch::<TC, TA>(device, c, lc, a, la, plan, n);
            }
        }
    }
    // (tiled path below stages the raw tile; cast copies keep the generic
    // planned entries)
    let (po, pa) = (plan_operand(lc, order, n), plan_operand(la, order, n));
    if po.is_generic() || pa.is_generic() {
        return kernels_ops::copy_dispatch_generic(device, c, lc, a, la, cast_expr);
    }
    let exec = device.stream();
    let _gc = order_input(c, exec)?;
    let _ga = order_input(a, exec)?;
    let variant = plan_variant(&[&po, &pa]);
    let entry = match variant {
        PlanVariant::Flat => "rstsr_copy_f",
        PlanVariant::Narrow => "rstsr_copy_pn",
        PlanVariant::Wide => "rstsr_copy_p",
    };
    let includes = [TC::INCLUDES, TA::INCLUDES].concat();
    let subs = [("@TC@", TC::CNAME), ("@TA@", TA::CNAME), ("@CAST@", cast_expr)];
    let f = planned_function(device, entry, &includes, &subs_disc(&subs), || render(COPY_ENTRIES, &subs))?;
    let mut launch = FlatLaunch::new(exec, &f, n);
    match variant {
        PlanVariant::Flat => {
            launch = launch
                .elems_per_thread(4)
                .arg_u64(offset_ptr(exec, c, lc.offset()))
                .arg_u64(offset_ptr(exec, a, la.offset()))
                .arg_usize(n);
        },
        PlanVariant::Narrow => {
            let zn = PlanBytesN::zeroed();
            let (nop, nap) = (plan_bytes_n(&po), plan_bytes_n(&pa));
            launch = launch
                .elems_per_thread(4)
                .arg_u64(offset_ptr(exec, c, lc.offset()))
                .arg_bytes(nop.as_ref().unwrap_or(&zn).bytes())
                .arg_u64(offset_ptr(exec, a, la.offset()))
                .arg_bytes(nap.as_ref().unwrap_or(&zn).bytes())
                .arg_i32(n as i32);
        },
        PlanVariant::Wide => {
            let zero = PlanBytes::zeroed();
            launch = launch
                .arg_u64(offset_ptr(exec, c, lc.offset()))
                .arg_bytes(plan_bytes(&po).as_ref().unwrap_or(&zero).bytes())
                .arg_u64(offset_ptr(exec, a, la.offset()))
                .arg_bytes(plan_bytes(&pa).as_ref().unwrap_or(&zero).bytes())
                .arg_usize(n);
        },
    }
    launch.launch()?;
    let _go = crate::kernel::order_output(c, exec)?;
    Ok(())
}

const WHERE_SC_ENTRIES: &str = r#"
extern "C" __global__ void rstsr_where_sc_p(
    @TD@* d, rstsr_plan dp,
    const bool* a, rstsr_plan ap,
    const @TB@* b, rstsr_plan bp,
    @TC@ v, int inv, rstsr_usz n)
{
    rstsr_usz lin = (rstsr_usz)blockIdx.x * (rstsr_usz)blockDim.x + (rstsr_usz)threadIdx.x;
    if (lin >= n) return;
    rstsr_usz id = rstsr_plan_offset(lin, dp, 0);
    rstsr_usz ia = rstsr_plan_offset(lin, ap, 0);
    rstsr_usz ib = rstsr_plan_offset(lin, bp, 0);
    d[id] = (a[ia] ^ inv) ? b[ib] : v;
}

extern "C" __global__ void rstsr_where_sc_pn(
    @TD@* d, rstsr_plan_n dp,
    const bool* a, rstsr_plan_n ap,
    const @TB@* b, rstsr_plan_n bp,
    @TC@ v, int inv, int n)
{
    int lin = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    int w = gridDim.x * blockDim.x;
    for (int rep = 0; rep < 4; ++rep) {
        if (lin < n) {
            rstsr_isz id = rstsr_plan_offset_n(lin, dp);
            rstsr_isz ia = rstsr_plan_offset_n(lin, ap);
            rstsr_isz ib = rstsr_plan_offset_n(lin, bp);
            d[id] = (a[ia] ^ inv) ? b[ib] : v;
        }
        lin += w;
    }
}

extern "C" __global__ void rstsr_where_sc_f(
    @TD@* d, const bool* a, const @TB@* b, @TC@ v, int inv, rstsr_usz n)
{
    rstsr_usz g = (rstsr_usz)blockIdx.x * (rstsr_usz)blockDim.x + (rstsr_usz)threadIdx.x;
    rstsr_usz w = (rstsr_usz)gridDim.x * (rstsr_usz)blockDim.x;
    #define RSTSR_STEP(I) { rstsr_usz id = g + (I) * w; if (id < n) { d[id] = (a[id] ^ inv) ? b[id] : v; } }
    RSTSR_STEP(0) RSTSR_STEP(1) RSTSR_STEP(2) RSTSR_STEP(3)
    #undef RSTSR_STEP
}
"#;

/// Where with a tensor `b` and scalar `c` (same dtype; the scalar rides in
/// the parameter list). `invert` flips the condition in-kernel, which
/// serves the scalar-`b` arm (`d = cond ? b : c` becomes
/// `d = !cond ? c : b`) without a scratch mask.
pub(crate) fn where_launch_sc<TD, TB, TC>(
    device: &DeviceCuda,
    d: &mut CudaBuffer<MaybeUninit<TD>>,
    ld: &Layout<impl DimAPI>,
    a: &CudaBuffer<bool>,
    la: &Layout<impl DimAPI>,
    b: &CudaBuffer<TB>,
    lb: &Layout<impl DimAPI>,
    c: &TC,
    invert: bool,
) -> Result<()>
where
    TD: CudaDType,
    TB: CudaDType,
    TC: CudaDType,
{
    let n = ld.size();
    if n == 0 {
        return Ok(());
    }
    let order = device.default_order();
    let (pd, pa, pb) = (plan_operand(ld, order, n), plan_operand(la, order, n), plan_operand(lb, order, n));
    if pd.is_generic() || pa.is_generic() || pb.is_generic() {
        return kernels_ops::where_sc_generic(device, d, ld, a, la, b, lb, c, invert);
    }
    let exec = device.stream();
    let _ga = order_input(a, exec)?;
    let _gb = order_input(b, exec)?;
    let variant = plan_variant(&[&pd, &pa, &pb]);
    let entry = match variant {
        PlanVariant::Flat => "rstsr_where_sc_f",
        PlanVariant::Narrow => "rstsr_where_sc_pn",
        PlanVariant::Wide => "rstsr_where_sc_p",
    };
    let includes = [TB::INCLUDES, TC::INCLUDES, TD::INCLUDES].concat();
    let subs = [("@TD@", TD::CNAME), ("@TB@", TB::CNAME), ("@TC@", TC::CNAME)];
    let f = planned_function(device, entry, &includes, &subs_disc(&subs), || render(WHERE_SC_ENTRIES, &subs))?;
    // SAFETY: layout-identical MaybeUninit transmute (crate invariant)
    let d_typed: &mut CudaBuffer<TD> = unsafe { &mut *(d as *mut CudaBuffer<MaybeUninit<TD>> as *mut CudaBuffer<TD>) };
    let nb = core::mem::size_of::<TC>();
    let mut cb = vec![0u8; nb];
    // SAFETY: TC is a plain scalar; bytes are its device representation
    unsafe {
        core::ptr::copy_nonoverlapping(core::ptr::addr_of!(*c) as *const u8, cb.as_mut_ptr(), nb);
    }
    let inv = invert as i32;
    let mut launch = FlatLaunch::new(exec, &f, n);
    match variant {
        PlanVariant::Flat => {
            launch = launch
                .elems_per_thread(4)
                .arg_u64(offset_ptr(exec, d_typed, ld.offset()))
                .arg_u64(offset_ptr(exec, a, la.offset()))
                .arg_u64(offset_ptr(exec, b, lb.offset()))
                .arg_bytes(&cb)
                .arg_i32(inv)
                .arg_usize(n);
        },
        PlanVariant::Narrow => {
            let zn = PlanBytesN::zeroed();
            let (ndp, nap, nbp) = (plan_bytes_n(&pd), plan_bytes_n(&pa), plan_bytes_n(&pb));
            launch = launch
                .elems_per_thread(4)
                .arg_u64(offset_ptr(exec, d_typed, ld.offset()))
                .arg_bytes(ndp.as_ref().unwrap_or(&zn).bytes())
                .arg_u64(offset_ptr(exec, a, la.offset()))
                .arg_bytes(nap.as_ref().unwrap_or(&zn).bytes())
                .arg_u64(offset_ptr(exec, b, lb.offset()))
                .arg_bytes(nbp.as_ref().unwrap_or(&zn).bytes())
                .arg_bytes(&cb)
                .arg_i32(inv)
                .arg_i32(n as i32);
        },
        PlanVariant::Wide => {
            let zero = PlanBytes::zeroed();
            launch = launch
                .arg_u64(offset_ptr(exec, d_typed, ld.offset()))
                .arg_bytes(plan_bytes(&pd).as_ref().unwrap_or(&zero).bytes())
                .arg_u64(offset_ptr(exec, a, la.offset()))
                .arg_bytes(plan_bytes(&pa).as_ref().unwrap_or(&zero).bytes())
                .arg_u64(offset_ptr(exec, b, lb.offset()))
                .arg_bytes(plan_bytes(&pb).as_ref().unwrap_or(&zero).bytes())
                .arg_bytes(&cb)
                .arg_i32(inv)
                .arg_usize(n);
        },
    }
    launch.launch()?;
    let _go = crate::kernel::order_output(d, exec)?;
    Ok(())
}

// /* #endregion */
fn plan_bytes(plan: &OperandPlan) -> Option<PlanBytes> {
    match plan {
        OperandPlan::Params(pack) => Some(PlanBytes::of_params(pack)),
        // a wide launch may mix widths; the narrow form widens exactly
        OperandPlan::ParamsN(pack) => Some(PlanBytes::of_params(&super::plan::widen_pack(pack))),
        _ => None,
    }
}

fn plan_bytes_n(plan: &OperandPlan) -> Option<PlanBytesN> {
    match plan {
        OperandPlan::ParamsN(pack) => Some(PlanBytesN::of_params(pack)),
        _ => None,
    }
}

// /* #region in-place binary (compound assign) launchers */
const BINARY_ASSIGN_ENTRIES: &str = r#"
extern "C" __global__ void rstsr_binary_assign_p(
    @T@* t, rstsr_plan tp, const @TB@* b, rstsr_plan bp, rstsr_usz n)
{
    rstsr_usz lin = (rstsr_usz)blockIdx.x * (rstsr_usz)blockDim.x + (rstsr_usz)threadIdx.x;
    if (lin >= n) return;
    rstsr_usz it = rstsr_plan_offset(lin, tp, 0);
    rstsr_usz ib = rstsr_plan_offset(lin, bp, 0);
    t[it] = @EXPR@;
}

extern "C" __global__ void rstsr_binary_assign_pn(
    @T@* t, rstsr_plan_n tp, const @TB@* b, rstsr_plan_n bp, int n)
{
    int lin = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    int w = gridDim.x * blockDim.x;
    for (int rep = 0; rep < 4; ++rep) {
        if (lin < n) {
            rstsr_isz it = rstsr_plan_offset_n(lin, tp);
            rstsr_isz ib = rstsr_plan_offset_n(lin, bp);
            t[it] = @EXPR@;
        }
        lin += w;
    }
}
"#;

const SCALAR_ASSIGN_ENTRIES: &str = r#"
extern "C" __global__ void rstsr_scalar_assign_p(
    @T@* t, rstsr_plan tp, @TB@ v, rstsr_usz n)
{
    rstsr_usz lin = (rstsr_usz)blockIdx.x * (rstsr_usz)blockDim.x + (rstsr_usz)threadIdx.x;
    if (lin >= n) return;
    rstsr_usz it = rstsr_plan_offset(lin, tp, 0);
    t[it] = @EXPR@;
}

extern "C" __global__ void rstsr_scalar_assign_f(
    @T@* t, @TB@ v, rstsr_usz n)
{
    rstsr_usz g = (rstsr_usz)blockIdx.x * (rstsr_usz)blockDim.x + (rstsr_usz)threadIdx.x;
    rstsr_usz w = (rstsr_usz)gridDim.x * (rstsr_usz)blockDim.x;
    #define RSTSR_STEP(I) { rstsr_usz oo = g + (I) * w; rstsr_usz it = oo; if (oo < n) { t[it] = @EXPR@; } }
    RSTSR_STEP(0) RSTSR_STEP(1) RSTSR_STEP(2) RSTSR_STEP(3)
    #undef RSTSR_STEP
}

extern "C" __global__ void rstsr_scalar_assign_pn(
    @T@* t, rstsr_plan_n tp, @TB@ v, int n)
{
    int lin = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    int w = gridDim.x * blockDim.x;
    for (int rep = 0; rep < 4; ++rep) {
        if (lin < n) {
            rstsr_isz it = rstsr_plan_offset_n(lin, tp);
            t[it] = @EXPR@;
        }
        lin += w;
    }
}
"#;

/// In-place binary launch (compound assign): destination planned like an
/// output, RHS like an input; write ordering mirrors the assign paths.
#[allow(clippy::too_many_arguments)]
pub(crate) fn binary_assign_launch<TA, TB>(
    device: &DeviceCuda,
    a: &mut CudaBuffer<TA>,
    la: &Layout<impl DimAPI>,
    b: &CudaBuffer<TB>,
    lb: &Layout<impl DimAPI>,
    expr: &str,
) -> Result<()>
where
    TA: CudaDType,
    TB: CudaDType,
{
    let n = la.size();
    if n == 0 {
        return Ok(());
    }
    let order = device.default_order();
    let pa = plan_operand(la, order, n);
    let pb = plan_operand(lb, order, n);
    if pa.is_generic() || pb.is_generic() {
        return kernels_ops::binary_assign_generic(device, a, la, b, lb, expr);
    }
    let exec = device.stream();
    let _ga = order_input(a, exec)?;
    let _gb = order_input(b, exec)?;
    let variant = plan_variant(&[&pa, &pb]);
    let entry = match variant {
        PlanVariant::Flat | PlanVariant::Narrow => {
            if matches!(pa, OperandPlan::Flat) && matches!(pb, OperandPlan::Flat) {
                "rstsr_binary_assign_p"
            } else {
                "rstsr_binary_assign_pn"
            }
        },
        PlanVariant::Wide => "rstsr_binary_assign_p",
    };
    let includes = [TA::INCLUDES, TB::INCLUDES].concat();
    let subs = [("@T@", TA::CNAME), ("@TB@", TB::CNAME), ("@EXPR@", expr)];
    let f = planned_function(device, entry, &includes, &subs_disc(&subs), || render(BINARY_ASSIGN_ENTRIES, &subs))?;
    let mut launch = FlatLaunch::new(exec, &f, n);
    match variant {
        PlanVariant::Wide => {
            let zero = PlanBytes::zeroed();
            launch = launch
                .arg_u64(offset_ptr(exec, a, la.offset()))
                .arg_bytes(plan_bytes(&pa).as_ref().unwrap_or(&zero).bytes())
                .arg_u64(offset_ptr(exec, b, lb.offset()))
                .arg_bytes(plan_bytes(&pb).as_ref().unwrap_or(&zero).bytes())
                .arg_usize(n);
        },
        PlanVariant::Flat => {
            let zero = PlanBytes::zeroed();
            launch = launch
                .arg_u64(offset_ptr(exec, a, la.offset()))
                .arg_bytes(plan_bytes(&pa).as_ref().unwrap_or(&zero).bytes())
                .arg_u64(offset_ptr(exec, b, lb.offset()))
                .arg_bytes(plan_bytes(&pb).as_ref().unwrap_or(&zero).bytes())
                .arg_usize(n);
        },
        PlanVariant::Narrow => {
            let zn = PlanBytesN::zeroed();
            let (nap, nbp) = (plan_bytes_n(&pa), plan_bytes_n(&pb));
            launch = launch
                .elems_per_thread(4)
                .arg_u64(offset_ptr(exec, a, la.offset()))
                .arg_bytes(nap.as_ref().unwrap_or(&zn).bytes())
                .arg_u64(offset_ptr(exec, b, lb.offset()))
                .arg_bytes(nbp.as_ref().unwrap_or(&zn).bytes())
                .arg_i32(n as i32);
        },
    }
    launch.launch()?;
    let _go = crate::kernel::order_output(a, exec)?;
    Ok(())
}

/// In-place tensor-scalar launch (compound assign, scalar RHS).
pub(crate) fn scalar_assign_launch<TA, TB>(
    device: &DeviceCuda,
    a: &mut CudaBuffer<TA>,
    la: &Layout<impl DimAPI>,
    b: &TB,
    expr: &str,
) -> Result<()>
where
    TA: CudaDType,
    TB: CudaDType,
{
    let n = la.size();
    if n == 0 {
        return Ok(());
    }
    let order = device.default_order();
    let pa = plan_operand(la, order, n);
    if pa.is_generic() {
        return kernels_ops::scalar_assign_generic(device, a, la, b, expr);
    }
    let exec = device.stream();
    let _ga = order_input(a, exec)?;
    let variant = plan_variant(&[&pa]);
    let entry = match variant {
        PlanVariant::Flat => "rstsr_scalar_assign_f",
        PlanVariant::Narrow => "rstsr_scalar_assign_pn",
        PlanVariant::Wide => "rstsr_scalar_assign_p",
    };
    let includes = [TA::INCLUDES, TB::INCLUDES].concat();
    let subs = [("@T@", TA::CNAME), ("@TB@", TB::CNAME), ("@EXPR@", expr)];
    let f = planned_function(device, entry, &includes, &subs_disc(&subs), || render(SCALAR_ASSIGN_ENTRIES, &subs))?;
    let nbytes = core::mem::size_of::<TB>();
    let mut bbytes = vec![0u8; nbytes];
    // SAFETY: TB is a plain scalar; its bytes are its on-device representation
    unsafe {
        core::ptr::copy_nonoverlapping(core::ptr::addr_of!(*b) as *const u8, bbytes.as_mut_ptr(), nbytes);
    }
    let mut launch = FlatLaunch::new(exec, &f, n);
    match variant {
        PlanVariant::Flat => {
            launch =
                launch.elems_per_thread(4).arg_u64(offset_ptr(exec, a, la.offset())).arg_bytes(&bbytes).arg_usize(n);
        },
        PlanVariant::Narrow => {
            let zn = PlanBytesN::zeroed();
            launch = launch
                .elems_per_thread(4)
                .arg_u64(offset_ptr(exec, a, la.offset()))
                .arg_bytes(plan_bytes_n(&pa).as_ref().unwrap_or(&zn).bytes())
                .arg_bytes(&bbytes)
                .arg_i32(n as i32);
        },
        PlanVariant::Wide => {
            let zero = PlanBytes::zeroed();
            launch = launch
                .arg_u64(offset_ptr(exec, a, la.offset()))
                .arg_bytes(plan_bytes(&pa).as_ref().unwrap_or(&zero).bytes())
                .arg_bytes(&bbytes)
                .arg_usize(n);
        },
    }
    launch.launch()?;
    let _go = crate::kernel::order_output(a, exec)?;
    Ok(())
}

// /* #endregion */
// /* #region where / clip (ternary, B7) */
const WHERE_ENTRIES: &str = r#"
extern "C" __global__ void rstsr_where_p(
    @TD@* d, rstsr_plan dp,
    const bool* a, rstsr_plan ap,
    const @TB@* b, rstsr_plan bp,
    const @TC@* c, rstsr_plan cp,
    rstsr_usz n)
{
    rstsr_usz lin = (rstsr_usz)blockIdx.x * (rstsr_usz)blockDim.x + (rstsr_usz)threadIdx.x;
    if (lin >= n) return;
    rstsr_usz id = rstsr_plan_offset(lin, dp, 0);
    rstsr_usz ia = rstsr_plan_offset(lin, ap, 0);
    rstsr_usz ib = rstsr_plan_offset(lin, bp, 0);
    rstsr_usz ic = rstsr_plan_offset(lin, cp, 0);
    d[id] = a[ia] ? b[ib] : c[ic];
}

extern "C" __global__ void rstsr_where_f(
    @TD@* d, const bool* a, const @TB@* b, const @TC@* c, rstsr_usz n)
{
    rstsr_usz g = (rstsr_usz)blockIdx.x * (rstsr_usz)blockDim.x + (rstsr_usz)threadIdx.x;
    rstsr_usz w = (rstsr_usz)gridDim.x * (rstsr_usz)blockDim.x;
    #define RSTSR_STEP(I) { rstsr_usz id = g + (I) * w; if (id < n) { d[id] = a[id] ? b[id] : c[id]; } }
    RSTSR_STEP(0) RSTSR_STEP(1) RSTSR_STEP(2) RSTSR_STEP(3)
    #undef RSTSR_STEP
}

extern "C" __global__ void rstsr_where_pn(
    @TD@* d, rstsr_plan_n dp,
    const bool* a, rstsr_plan_n ap,
    const @TB@* b, rstsr_plan_n bp,
    const @TC@* c, rstsr_plan_n cp,
    int n)
{
    int lin = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    int w = gridDim.x * blockDim.x;
    for (int rep = 0; rep < 4; ++rep) {
        if (lin < n) {
            rstsr_isz id = rstsr_plan_offset_n(lin, dp);
            rstsr_isz ia = rstsr_plan_offset_n(lin, ap);
            rstsr_isz ib = rstsr_plan_offset_n(lin, bp);
            rstsr_isz ic = rstsr_plan_offset_n(lin, cp);
            d[id] = a[ia] ? b[ib] : c[ic];
        }
        lin += w;
    }
}
"#;

/// Planned ternary select launch (`d = a ? b : c`); all four operands
/// planned; generic fallback when any layout exceeds the pack.
#[allow(clippy::too_many_arguments)]
pub(crate) fn where_launch<TD, TB, TC>(
    device: &DeviceCuda,
    d: &mut CudaBuffer<MaybeUninit<TD>>,
    ld: &Layout<impl DimAPI>,
    a: &CudaBuffer<bool>,
    la: &Layout<impl DimAPI>,
    b: &CudaBuffer<TB>,
    lb: &Layout<impl DimAPI>,
    c: &CudaBuffer<TC>,
    lc: &Layout<impl DimAPI>,
) -> Result<()>
where
    TD: CudaDType,
    TB: CudaDType,
    TC: CudaDType,
{
    let n = ld.size();
    if n == 0 {
        return Ok(());
    }
    let order = device.default_order();
    let (pd, pa, pb, pc) = (
        plan_operand(ld, order, n),
        plan_operand(la, order, n),
        plan_operand(lb, order, n),
        plan_operand(lc, order, n),
    );
    if pd.is_generic() || pa.is_generic() || pb.is_generic() || pc.is_generic() {
        return kernels_ops::where_generic(device, d, ld, a, la, b, lb, c, lc);
    }
    let exec = device.stream();
    let _ga = order_input(a, exec)?;
    let _gb = order_input(b, exec)?;
    let _gc = order_input(c, exec)?;
    let variant = plan_variant(&[&pd, &pa, &pb, &pc]);
    let entry = match variant {
        PlanVariant::Flat => "rstsr_where_f",
        PlanVariant::Narrow => "rstsr_where_pn",
        PlanVariant::Wide => "rstsr_where_p",
    };
    let includes = [TB::INCLUDES, TC::INCLUDES, TD::INCLUDES].concat();
    let subs = [("@TD@", TD::CNAME), ("@TB@", TB::CNAME), ("@TC@", TC::CNAME)];
    let f = planned_function(device, entry, &includes, &subs_disc(&subs), || render(WHERE_ENTRIES, &subs))?;
    // SAFETY: layout-identical MaybeUninit transmute (crate invariant)
    let d_typed: &mut CudaBuffer<TD> = unsafe { &mut *(d as *mut CudaBuffer<MaybeUninit<TD>> as *mut CudaBuffer<TD>) };
    let mut launch = FlatLaunch::new(exec, &f, n);
    match variant {
        PlanVariant::Flat => {
            launch = launch
                .elems_per_thread(4)
                .arg_u64(offset_ptr(exec, d_typed, ld.offset()))
                .arg_u64(offset_ptr(exec, a, la.offset()))
                .arg_u64(offset_ptr(exec, b, lb.offset()))
                .arg_u64(offset_ptr(exec, c, lc.offset()))
                .arg_usize(n);
        },
        PlanVariant::Narrow => {
            let zn = PlanBytesN::zeroed();
            let (ndp, nap, nbp, ncp) = (plan_bytes_n(&pd), plan_bytes_n(&pa), plan_bytes_n(&pb), plan_bytes_n(&pc));
            launch = launch
                .elems_per_thread(4)
                .arg_u64(offset_ptr(exec, d_typed, ld.offset()))
                .arg_bytes(ndp.as_ref().unwrap_or(&zn).bytes())
                .arg_u64(offset_ptr(exec, a, la.offset()))
                .arg_bytes(nap.as_ref().unwrap_or(&zn).bytes())
                .arg_u64(offset_ptr(exec, b, lb.offset()))
                .arg_bytes(nbp.as_ref().unwrap_or(&zn).bytes())
                .arg_u64(offset_ptr(exec, c, lc.offset()))
                .arg_bytes(ncp.as_ref().unwrap_or(&zn).bytes())
                .arg_i32(n as i32);
        },
        PlanVariant::Wide => {
            let zero = PlanBytes::zeroed();
            launch = launch
                .arg_u64(offset_ptr(exec, d_typed, ld.offset()))
                .arg_bytes(plan_bytes(&pd).as_ref().unwrap_or(&zero).bytes())
                .arg_u64(offset_ptr(exec, a, la.offset()))
                .arg_bytes(plan_bytes(&pa).as_ref().unwrap_or(&zero).bytes())
                .arg_u64(offset_ptr(exec, b, lb.offset()))
                .arg_bytes(plan_bytes(&pb).as_ref().unwrap_or(&zero).bytes())
                .arg_u64(offset_ptr(exec, c, lc.offset()))
                .arg_bytes(plan_bytes(&pc).as_ref().unwrap_or(&zero).bytes())
                .arg_usize(n);
        },
    }
    launch.launch()?;
    let _go = crate::kernel::order_output(d, exec)?;
    Ok(())
}

// /* #endregion */
// /* #region clip (all bound combinations, single pass) */
/// Lean both-scalar clip: `out = min(max(in, lo), hi)`, flat/narrow only
/// (the generic combinations stay on `rstsr_clip_p/pn`).
const CLIP_SS_ENTRIES: &str = r#"
extern "C" __global__ void rstsr_clip_ss_f(
    @TD@* out, const @TD@* a, @TD@ lo, @TD@ hi, rstsr_usz n)
{
    rstsr_usz g = (rstsr_usz)blockIdx.x * (rstsr_usz)blockDim.x + (rstsr_usz)threadIdx.x;
    rstsr_usz w = (rstsr_usz)gridDim.x * (rstsr_usz)blockDim.x;
    #define RSTSR_STEP(I) { rstsr_usz io = g + (I) * w; if (io < n) { @TD@ val = a[io]; val = @MAXEXPR@; out[io] = @MINEXPR@; } }
    RSTSR_STEP(0) RSTSR_STEP(1) RSTSR_STEP(2) RSTSR_STEP(3)
    #undef RSTSR_STEP
}

extern "C" __global__ void rstsr_clip_ss_pn(
    @TD@* out, rstsr_plan_n op, const @TD@* a, rstsr_plan_n ap, @TD@ lo, @TD@ hi, int n)
{
    int lin = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    int w = gridDim.x * blockDim.x;
    for (int rep = 0; rep < 4; ++rep) {
        if (lin < n) {
            rstsr_isz io = rstsr_plan_offset_n(lin, op);
            rstsr_isz ia = rstsr_plan_offset_n(lin, ap);
            @TD@ val = a[ia];
            val = @MAXEXPR@;
            out[io] = @MINEXPR@;
        }
        lin += w;
    }
}
"#;

/// Clip bound descriptor (host side). `D` is the bound tensor's dim.
pub(crate) enum ClipBound<'a, T, D: DimAPI> {
    None,
    Scalar(&'a T),
    Tensor(&'a CudaBuffer<T>, &'a Layout<D>),
}

const CLIP_ENTRIES: &str = r#"
extern "C" __global__ void rstsr_clip_p(
    @TD@* d, rstsr_plan dp,
    const @TD@* a, rstsr_plan ap,
    const @TB@* b, rstsr_plan bp, int b_kind,
    const @TC@* c, rstsr_plan cp, int c_kind,
    @TB@ v_lo, @TC@ v_hi, rstsr_usz n)
{
    rstsr_usz lin = (rstsr_usz)blockIdx.x * (rstsr_usz)blockDim.x + (rstsr_usz)threadIdx.x;
    if (lin >= n) return;
    rstsr_usz id = rstsr_plan_offset(lin, dp, 0);
    rstsr_usz x = rstsr_plan_offset(lin, ap, 0);
    @TD@ val = a[x];
    if (b_kind != 0) {
        @TD@ lo = (b_kind == 1) ? b[rstsr_plan_offset(lin, bp, 0)] : v_lo;
        val = @MAXEXPR@;
    }
    if (c_kind != 0) {
        @TD@ hi = (c_kind == 1) ? c[rstsr_plan_offset(lin, cp, 0)] : v_hi;
        val = @MINEXPR@;
    }
    d[id] = val;
}

extern "C" __global__ void rstsr_clip_pn(
    @TD@* d, rstsr_plan_n dp,
    const @TD@* a, rstsr_plan_n ap,
    const @TB@* b, rstsr_plan_n bp, int b_kind,
    const @TC@* c, rstsr_plan_n cp, int c_kind,
    @TB@ v_lo, @TC@ v_hi, int n)
{
    int lin = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    int w = gridDim.x * blockDim.x;
    for (int rep = 0; rep < 4; ++rep) {
        if (lin < n) {
            rstsr_isz id = rstsr_plan_offset_n(lin, dp);
            rstsr_isz x = rstsr_plan_offset_n(lin, ap);
            @TD@ val = a[x];
            if (b_kind != 0) {
                rstsr_isz ib = rstsr_plan_offset_n(lin, bp);
                @TD@ lo = (b_kind == 1) ? b[ib] : v_lo;
                val = @MAXEXPR@;
            }
            if (c_kind != 0) {
                rstsr_isz ic = rstsr_plan_offset_n(lin, cp);
                @TD@ hi = (c_kind == 1) ? c[ic] : v_hi;
                val = @MINEXPR@;
            }
            d[id] = val;
        }
        lin += w;
    }
}
"#;

/// Single-pass clip launch over any bound combination. `MAXEXPR`/`MINEXPR`
/// are the NaN-propagating expressions on `(val, lo)` / `(val, hi)`.
pub(crate) fn clip_launch<TD, TB, TC, LD, LA, LB, LC>(
    device: &DeviceCuda,
    d: &mut CudaBuffer<MaybeUninit<TD>>,
    ld: &Layout<LD>,
    a: &CudaBuffer<TD>,
    la: &Layout<LA>,
    lo: ClipBound<TB, LB>,
    hi: ClipBound<TC, LC>,
    max_expr: &str,
    min_expr: &str,
) -> Result<()>
where
    TD: CudaDType,
    TB: CudaDType,
    TC: CudaDType,
    LD: DimAPI,
    LA: DimAPI,
    LB: DimAPI,
    LC: DimAPI,
{
    let n = ld.size();
    if n == 0 {
        return Ok(());
    }
    // the device op admits same-dtype bounds only (CPU-side promotion is
    // not mirrored on this backend)
    debug_assert_eq!(core::mem::size_of::<TB>(), core::mem::size_of::<TD>());
    debug_assert_eq!(core::mem::size_of::<TC>(), core::mem::size_of::<TD>());
    let order = device.default_order();
    let (pd, pa) = (plan_operand(ld, order, n), plan_operand(la, order, n));
    let (pb, pc) = match (&lo, &hi) {
        (ClipBound::Tensor(_, lb), ClipBound::Tensor(_, lc)) => {
            (plan_operand(lb, order, n), plan_operand(lc, order, n))
        },
        (ClipBound::Tensor(_, lb), _) => (plan_operand(lb, order, n), OperandPlan::Flat),
        (_, ClipBound::Tensor(_, lc)) => (OperandPlan::Flat, plan_operand(lc, order, n)),
        _ => (OperandPlan::Flat, OperandPlan::Flat),
    };
    if pd.is_generic() || pa.is_generic() || pb.is_generic() || pc.is_generic() {
        return kernels_ops::clip_generic(device, d, ld, a, la, lo, hi, max_expr, min_expr);
    }
    // lean both-scalar path (no bound plans, no kinds; ILP4 on the flat arm).
    // Bounds are TD-sized (the device op admits same-dtype bounds only).
    // Wide plans take the general `_p` entry below, which carries the width
    // that `rstsr_clip_ss_pn` cannot.
    let lean_ok = matches!(plan_variant(&[&pd, &pa]), PlanVariant::Flat | PlanVariant::Narrow);
    if lean_ok {
        if let (ClipBound::Scalar(lo_v), ClipBound::Scalar(hi_v)) = (&lo, &hi) {
            let nb = core::mem::size_of::<TD>();
            let mut lob = vec![0u8; nb];
            let mut hib = vec![0u8; nb];
            // SAFETY: the device op admits same-dtype bounds (TB = TC = TD)
            unsafe {
                core::ptr::copy_nonoverlapping(*lo_v as *const TB as *const u8, lob.as_mut_ptr(), nb);
                core::ptr::copy_nonoverlapping(*hi_v as *const TC as *const u8, hib.as_mut_ptr(), nb);
            }
            return clip2s_out_launch::<TD, LD, LA>(device, d, ld, a, la, lob, hib, max_expr, min_expr);
        }
    }
    let exec = device.stream();
    let _ga = order_input(a, exec)?;
    if let ClipBound::Tensor(buf, _) = &lo {
        let _ = order_input(buf, exec)?;
    }
    if let ClipBound::Tensor(buf, _) = &hi {
        let _ = order_input(buf, exec)?;
    }
    let b_kind = match &lo {
        ClipBound::None => 0i32,
        ClipBound::Scalar(_) => 2,
        ClipBound::Tensor(_, _) => 1,
    };
    let c_kind = match &hi {
        ClipBound::None => 0i32,
        ClipBound::Scalar(_) => 2,
        ClipBound::Tensor(_, _) => 1,
    };
    let variant = plan_variant(&[&pd, &pa, &pb, &pc]);
    let entry = match variant {
        PlanVariant::Flat | PlanVariant::Narrow => {
            if matches!(pd, OperandPlan::Flat)
                && matches!(pa, OperandPlan::Flat)
                && matches!(pb, OperandPlan::Flat)
                && matches!(pc, OperandPlan::Flat)
            {
                "rstsr_clip_p"
            } else {
                "rstsr_clip_pn"
            }
        },
        PlanVariant::Wide => "rstsr_clip_p",
    };
    let includes = [TD::INCLUDES, TB::INCLUDES, TC::INCLUDES].concat();
    let subs = [
        ("@TD@", TD::CNAME),
        ("@TB@", TB::CNAME),
        ("@TC@", TC::CNAME),
        ("@MAXEXPR@", max_expr),
        ("@MINEXPR@", min_expr),
    ];
    let f = planned_function(device, entry, &includes, &subs_disc(&subs), || render(CLIP_ENTRIES, &subs))?;
    // SAFETY: layout-identical MaybeUninit transmute (crate invariant)
    let d_typed: &mut CudaBuffer<TD> = unsafe { &mut *(d as *mut CudaBuffer<MaybeUninit<TD>> as *mut CudaBuffer<TD>) };
    let nb = core::mem::size_of::<TD>();
    let mut blob = vec![0u8; nb];
    let mut hib = vec![0u8; nb];
    // SAFETY: scalar bytes (TD-sized for both bounds; same-type promotion
    // at the tensor tier)
    unsafe {
        match &lo {
            ClipBound::Scalar(v) => {
                core::ptr::copy_nonoverlapping(core::ptr::addr_of!(**v) as *const u8, blob.as_mut_ptr(), nb)
            },
            _ => core::ptr::write_bytes(blob.as_mut_ptr(), 0, nb),
        }
        match &hi {
            ClipBound::Scalar(v) => {
                core::ptr::copy_nonoverlapping(core::ptr::addr_of!(**v) as *const u8, hib.as_mut_ptr(), nb)
            },
            _ => core::ptr::write_bytes(hib.as_mut_ptr(), 0, nb),
        }
    }
    // the entry selection above already resolved the all-flat case to `_p`
    let narrow = matches!(variant, PlanVariant::Narrow);
    let zero = PlanBytes::zeroed();
    let zn = PlanBytesN::zeroed();
    // pre-marshaled plan bytes (both widths; unused width stays zeroed)
    let (wd, wa, wb, wc) = (plan_bytes(&pd), plan_bytes(&pa), plan_bytes(&pb), plan_bytes(&pc));
    let (nd, na, nb_, nc) = (plan_bytes_n(&pd), plan_bytes_n(&pa), plan_bytes_n(&pb), plan_bytes_n(&pc));
    let mut launch = FlatLaunch::new(exec, &f, n).arg_u64(offset_ptr(exec, d_typed, ld.offset()));
    if narrow {
        // the `_pn` entries are 4-step grid-stride loops
        launch = launch.elems_per_thread(4);
    }
    launch = if narrow {
        launch.arg_bytes(nd.as_ref().unwrap_or(&zn).bytes())
    } else {
        launch.arg_bytes(wd.as_ref().unwrap_or(&zero).bytes())
    };
    launch = launch.arg_u64(offset_ptr(exec, a, la.offset()));
    launch = if narrow {
        launch.arg_bytes(na.as_ref().unwrap_or(&zn).bytes())
    } else {
        launch.arg_bytes(wa.as_ref().unwrap_or(&zero).bytes())
    };
    launch = match &lo {
        ClipBound::Tensor(buf, lb) => launch
            .arg_u64(offset_ptr(exec, buf, lb.offset()))
            .arg_bytes(if narrow { nb_.as_ref().unwrap_or(&zn).bytes() } else { wb.as_ref().unwrap_or(&zero).bytes() })
            .arg_i32(b_kind),
        _ => launch.arg_u64(0).arg_bytes(if narrow { zn.bytes() } else { zero.bytes() }).arg_i32(b_kind),
    };
    launch = match &hi {
        ClipBound::Tensor(buf, lc) => launch
            .arg_u64(offset_ptr(exec, buf, lc.offset()))
            .arg_bytes(if narrow { nc.as_ref().unwrap_or(&zn).bytes() } else { wc.as_ref().unwrap_or(&zero).bytes() })
            .arg_i32(c_kind),
        _ => launch.arg_u64(0).arg_bytes(if narrow { zn.bytes() } else { zero.bytes() }).arg_i32(c_kind),
    };
    launch = launch.arg_bytes(&blob).arg_bytes(&hib);
    launch = if narrow { launch.arg_i32(n as i32) } else { launch.arg_usize(n) };
    launch.launch()?;
    let _go = crate::kernel::order_output(d, exec)?;
    Ok(())
}

/// Out-of-place both-scalar clip (lean `rstsr_clip_ss_f/pn` entries);
/// `lo`/`hi` arrive as TD-sized on-device bytes.
fn clip2s_out_launch<TD, LD: DimAPI, LA: DimAPI>(
    device: &DeviceCuda,
    d: &mut CudaBuffer<MaybeUninit<TD>>,
    ld: &Layout<LD>,
    a: &CudaBuffer<TD>,
    la: &Layout<LA>,
    lo: Vec<u8>,
    hi: Vec<u8>,
    max_expr: &str,
    min_expr: &str,
) -> Result<()>
where
    TD: CudaDType,
{
    let n = ld.size();
    let order = device.default_order();
    let (po, pa) = (plan_operand(ld, order, n), plan_operand(la, order, n));
    // op_clip broadcasts scalar bounds onto the value operand's own layout,
    // so neither operand can exceed the parameterized walk here
    rstsr_assert!(!(po.is_generic() || pa.is_generic()), InvalidLayout, "both-scalar clip beyond plan walk")?;
    let exec = device.stream();
    let _ga = order_input(a, exec)?;
    let variant = plan_variant(&[&po, &pa]);
    let entry = match variant {
        PlanVariant::Flat => "rstsr_clip_ss_f",
        _ => "rstsr_clip_ss_pn",
    };
    #[cfg(feature = "launch_trace")]
    eprintln!("[rstsr-cuda] clip2s_out entry={entry} n={n}");
    #[cfg(feature = "launch_trace")]
    let _ = entry;
    // both stages read/write the local `val` (sequential in the template);
    // textual nesting would duplicate the max stage 4x with control flow the
    // compiler cannot CSE. The bound expressions already carry the CPU
    // NaN/zero-tie semantics.
    let max_stage = max_expr;
    let min_stage = min_expr;
    let subs = [("@TD@", TD::CNAME), ("@MAXEXPR@", max_stage), ("@MINEXPR@", min_stage)];
    let f = planned_function(device, entry, TD::INCLUDES, &subs_disc(&subs), || render(CLIP_SS_ENTRIES, &subs))?;
    // SAFETY: layout-identical MaybeUninit transmute (crate invariant)
    let d_typed: &mut CudaBuffer<TD> = unsafe { &mut *(d as *mut CudaBuffer<MaybeUninit<TD>> as *mut CudaBuffer<TD>) };
    let mut launch = FlatLaunch::new(exec, &f, n);
    match (variant, &po, &pa) {
        (PlanVariant::Flat, _, _) => {
            launch = launch
                .elems_per_thread(4)
                .arg_u64(offset_ptr(exec, d_typed, ld.offset()))
                .arg_u64(offset_ptr(exec, a, la.offset()))
                .arg_bytes(&lo)
                .arg_bytes(&hi)
                .arg_usize(n);
        },
        _ => {
            let zn = PlanBytesN::zeroed();
            launch = launch
                .elems_per_thread(4)
                .arg_u64(offset_ptr(exec, d_typed, ld.offset()))
                .arg_bytes(plan_bytes_n(&po).as_ref().unwrap_or(&zn).bytes())
                .arg_u64(offset_ptr(exec, a, la.offset()))
                .arg_bytes(plan_bytes_n(&pa).as_ref().unwrap_or(&zn).bytes())
                .arg_bytes(&lo)
                .arg_bytes(&hi)
                .arg_i32(n as i32);
        },
    }
    launch.launch()?;
    let _go = crate::kernel::order_output(d, exec)?;
    Ok(())
}

// /* #endregion */
