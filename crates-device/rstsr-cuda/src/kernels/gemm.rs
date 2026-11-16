//! Tiled GEMM behind `DeviceMatMulAPI` — the **fallback** path, not the default.
//!
//! [`super::cublas_gemm`] runs first and handles every layout it can express; this kernel takes what
//! is left: a stepped output, a reversed operand, a machine without `libcublas`. It is kept for
//! those cases and as the test oracle for the cuBLAS path, not because it competes — on an A100,
//! `dgemm` reaches the FP64 tensor-core pipe at 19.5 TFLOPS where a SIMT tile tops out at 9.7.
//!
//! `RSTSR_CUDA_MATMUL=tiled` forces this path, which is how `tests/test_matmul_tiled.rs` keeps it
//! on hardware evidence now that cuBLAS is the default.
//!
//! What follows describes this kernel.
//! # What this kernel is
//!
//! One block computes a `BM x BN` tile of `C = alpha * A B + beta * C` by streaming the contraction
//! axis through shared memory in `BK`-deep slabs. Per block that is `BM*BN*K` fused multiply-adds
//! for `(BM+BN)*K` element loads, i.e. `2*64*64 / (128*8) = 8` flop/byte once the tile is resident.
//! The untiled form is `2/16 = 0.125`, which is DRAM-bound on every machine — the whole reason this
//! file exists.
//!
//! 8 flop/byte is only just above the A100 fp64 ridge point (`9.7 TFLOPS / 1.55 TB/s` = 6.3), so at
//! this tile size the kernel is *marginally* compute-bound and the memory system still costs nearly
//! as much as the arithmetic. Doubling the tile to 128x128 gives `2*128*128 / (256*8) = 16` and is
//! the first thing to try after measuring this one; `BM`/`BN` are plain constants and every derived
//! count (slab depth, thread count, shared sizes) follows from them, so the change is a re-tune
//! rather than a rewrite.
//!
//! # The two things that are actually decisions
//!
//! **1. The lane map follows a unit-stride axis, not a fixed axis.** A 2-D operand arrives with
//! arbitrary strides: `A` may be `[M, K]` with `K` contiguous (`rt::matmul(a, b)`) or with `M`
//! contiguous (`rt::matmul(a.T, b)`), and `layout_matmul` admits both because it only checks shapes.
//! Instead of transposing an operand — a copy — the block's linear thread id is mapped to the tile
//! along whichever axis is unit-stride, chosen at **NVRTC time**. Four variants result (A on `k` or
//! `m`, B on `k` or `n`) and each has both global loads coalesced.
//!
//! **2. Per-thread tiles are interleaved, not blocked.** Thread `(tx, ty)` owns rows `ty + i*TY` and
//! columns `tx + j*TX`, so one shared-memory read instruction has its 16 lanes on 16 consecutive
//! doubles (and the other row group in the warp reads the same address, which broadcasts). Blocking
//! would instead make the `Bs` read stride `TN` doubles per lane — a 4-way bank conflict on every
//! one of the `TM*TN*BK` inner reads. Interleaving costs something on the store side, but not
//! bandwidth: for a fixed `(i, j)` the 16 lanes span 16 sectors holding 8 useful bytes each, and the
//! four values of `j` then fill those same sectors, so all the bytes are consumed — the request
//! count is 4x, not the traffic. Staging `C` through shared memory is the fix if that ever measures
//! badly.
//!
//! # Order
//!
//! `ColMajor` is not special-cased in the kernel. The identity the layout module documents,
//! `C[t,m,n] = A[t,m,k] B[t,k,n]` (row-major) `<==>` `C[n,m,t] = B[n,k,t] A[k,m,t]` (col-major),
//! says: reverse every layout and swap the operands. Reversing a column-major layout yields a
//! row-major one in the reversed shape, so the kernel only ever sees row-major-decomposed blocks
//! with leading dimensions.
//!
//! # What is deliberately absent
//!
//! No double buffering, no 128-bit vectorised loads, no `C` staging, no split-K, no cuBLAS. Each is
//! one measurement away, and the measurement needs this kernel to exist first. Batch axes ride in
//! `grid.z` and must be *affine* (see [`linear_step`]); a non-affine batch is refused rather than
//! silently mis-indexed.
//!
//! Complex dtypes are excluded at the `DeviceMatMulAPI` impl rather than here: this source spells
//! its zero as `(@T@)0`, and `rstsr_complex` has no single-argument constructor, so a complex
//! instantiation would fail inside NVRTC — on a rented GPU — instead of at the call site.

use std::sync::Arc;

use crate::kernel::{FlatLaunch, order_input, order_output};
use crate::kernels::cublas_gemm::{self, CublasScalar};
use crate::kernels::elementwise::planned_function;
use crate::prelude_dev::*;

/// Tile of `C` computed by one block.
const BM: u32 = 64;
const BN: u32 = 64;
/// Depth of one shared-memory slab along the contraction.
const BK: u32 = 16;
/// Outputs per thread, per axis.
const TM: u32 = 4;
const TN: u32 = 4;
const TX: u32 = BN / TN;
const TY: u32 = BM / TM;
const NTHREADS: u32 = TX * TY;

/// Elements per thread when filling one slab: `BM*BK` and `BK*BN` over the block size.
const A_SLAB: u32 = BM * BK / NTHREADS;
const B_SLAB: u32 = BK * BN / NTHREADS;

const _: () = assert!(TX * TY == 256, "the block must be 256 threads");
const _: () = assert!(BM * BK % NTHREADS == 0 && BK * BN % NTHREADS == 0);
const _: () = assert!(TM * TY == BM && TN * TX == BN);

/// Which axis of a 2-D operand the linear thread id enumerates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LaneAxis {
    /// Contraction axis (`k`).
    K,
    /// The other one: `m` for the left operand, `n` for the right.
    Other,
}

const GEMM_ENTRY: &str = r#"
extern "C" __global__ void rstsr_gemm(
    @T@* c,     rstsr_isz scm, rstsr_isz scn, rstsr_isz scb,
    const @T@* a, rstsr_isz sam, rstsr_isz sak, rstsr_isz sab,
    const @T@* b, rstsr_isz sbk, rstsr_isz sbn, rstsr_isz sbb,
    int M, int N, int K,
    @T@ alpha, @T@ beta)
{
    __shared__ @T@ As[@BK@][@BM@ + 1];
    __shared__ @T@ Bs[@BK@][@BN@ + 1];

    const int m0 = (int)blockIdx.y * @BM@;
    const int n0 = (int)blockIdx.x * @BN@;
    const int tid = (int)threadIdx.y * @TX@ + (int)threadIdx.x;
    const int tx = (int)threadIdx.x;
    const int ty = (int)threadIdx.y;

    // batch: one grid slice per output matrix, affine in the slice index
    const @T@* ap = a + (rstsr_isz)blockIdx.z * sab;
    const @T@* bp = b + (rstsr_isz)blockIdx.z * sbb;
    @T@*       cp = c + (rstsr_isz)blockIdx.z * scb;

    @T@ acc[@TM@][@TN@];
    #pragma unroll
    for (int i = 0; i < @TM@; ++i)
        #pragma unroll
        for (int j = 0; j < @TN@; ++j) acc[i][j] = (@T@)0;

    for (int k0 = 0; k0 < K; k0 += @BK@) {
        // A slab, [BM x BK], stored transposed as As[k][m] so the inner loop reads one shared
        // double per (i, k) instead of a strided column.
        #pragma unroll
        for (int t = 0; t < @A_SLAB@; ++t) {
            rstsr_usz lin = (rstsr_usz)(tid + t * @NTHREADS@);
            @A_IDX@
            const int gm = m0 + (int)mm;
            const int gk = k0 + (int)kk;
            As[kk][mm] = (gm < M && gk < K)
                ? ap[(rstsr_isz)gm * sam + (rstsr_isz)gk * sak]
                : (@T@)0;
        }
        // B slab, [BK x BN], stored as Bs[k][n].
        #pragma unroll
        for (int t = 0; t < @B_SLAB@; ++t) {
            rstsr_usz lin = (rstsr_usz)(tid + t * @NTHREADS@);
            @B_IDX@
            const int gk = k0 + (int)kk;
            const int gn = n0 + (int)nn;
            Bs[kk][nn] = (gk < K && gn < N)
                ? bp[(rstsr_isz)gk * sbk + (rstsr_isz)gn * sbn]
                : (@T@)0;
        }
        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < @BK@; ++kk) {
            @T@ av[@TM@], bv[@TN@];
            #pragma unroll
            for (int i = 0; i < @TM@; ++i) av[i] = As[kk][ty + i * @TY@];
            #pragma unroll
            for (int j = 0; j < @TN@; ++j) bv[j] = Bs[kk][tx + j * @TX@];
            #pragma unroll
            for (int i = 0; i < @TM@; ++i)
                #pragma unroll
                for (int j = 0; j < @TN@; ++j) acc[i][j] += av[i] * bv[j];
        }
        __syncthreads();
    }

    // `beta == 0` must not read `c`: that is the contract of `matmul_uninit`, whose buffer is
    // `MaybeUninit` and has no meaningful contents.
    #pragma unroll
    for (int i = 0; i < @TM@; ++i) {
        const int gm = m0 + ty + i * @TY@;
        if (gm >= M) continue;
        #pragma unroll
        for (int j = 0; j < @TN@; ++j) {
            const int gn = n0 + tx + j * @TX@;
            if (gn >= N) continue;
            @T@* p = cp + (rstsr_isz)gm * scm + (rstsr_isz)gn * scn;
            *p = (beta == (@T@)0) ? (alpha * acc[i][j]) : (alpha * acc[i][j] + beta * (*p));
        }
    }
}
"#;

fn render(template: &str, subs: &[(&str, &str)]) -> String {
    let mut s = template.to_string();
    for (name, text) in subs {
        s = s.replace(name, text);
    }
    s
}

/// Fold an element offset into a base device pointer, in bytes.
///
/// Mirrors `kernels::elementwise::offset_ptr`: the read event must be recorded on the **launching**
/// stream, not the allocation stream, or a cross-stream free can race the kernel.
fn offset_ptr<B>(exec: &Arc<CudaStreamObj>, buf: &CudaBuffer<B>, offset: usize) -> u64 {
    if buf.is_empty() {
        0
    } else {
        (buf.device_ptr_on(exec) as u64).wrapping_add(offset as u64 * core::mem::size_of::<B>() as u64)
    }
}

/// Pick the axis the linear thread id enumerates, preferring a unit-stride one.
///
/// `s_other`/`s_k` are the strides of the non-contracted and contracted axes, `n_other`/`n_k` their
/// extents. A degenerate axis (extent 1) is free to take: it contributes no address variation, so
/// following the other axis is coalesced whatever its stride happens to be.
fn lane_axis(s_other: isize, s_k: isize, n_other: usize, n_k: usize, what: &str) -> Result<LaneAxis> {
    if s_k == 1 {
        Ok(LaneAxis::K)
    } else if s_other == 1 {
        Ok(LaneAxis::Other)
    } else if n_other == 1 {
        Ok(LaneAxis::K)
    } else if n_k == 1 {
        Ok(LaneAxis::Other)
    } else {
        rstsr_raise!(
            UnImplemented,
            "cuda matmul: {what} has no unit-stride axis (strides {s_other} and {s_k}); \
             a stepped 2-D block needs a packing pass first"
        )?
    }
}

/// Element step of one batch slice, if the batch part is a single affine run.
///
/// Size-1 axes are dropped first, because they contribute no offset however their stride is
/// written: `[4, 1]` with strides `[3, 0]` steps by 3, and `[4, 1]` with `[3, 7]` steps by 3 as
/// well. What survives must then be a row-major progression, so `offset = index * step` for every
/// index in `0..size`. Broadcast axes (`stride == 0`) are affine with step 0, which covers
/// `A[i,j] @ B[p,j,k]` where the batch axis belongs to `B` alone.
///
/// `None` means a genuinely strided multi-axis batch (`[4, 3]` with strides `[6, 1]`, whose offsets
/// are not one step). Those need a real plan walk; the caller refuses them by name.
fn linear_step(l: &Layout<IxD>) -> Option<isize> {
    let mut shape: Vec<usize> = Vec::with_capacity(l.ndim());
    let mut stride: Vec<isize> = Vec::with_capacity(l.ndim());
    for k in 0..l.ndim() {
        if l.shape()[k] != 1 {
            shape.push(l.shape()[k]);
            stride.push(l.stride()[k]);
        }
    }
    if shape.is_empty() {
        return Some(0);
    }
    let n = shape.len();
    let step = stride[n - 1];
    let mut expect = step;
    for k in (0..n).rev() {
        if stride[k] != expect {
            return None;
        }
        expect *= shape[k] as isize;
    }
    Some(step)
}

/// Everything the kernel needs to address the three buffers, resolved from the layouts alone.
///
/// Separated from the launch because this is the part that can be wrong in a way a GPU run reports as
/// "wrong numbers": the col-major reversal, the batch steps, the choice of lane axis. All of it is
/// host-side arithmetic on strides, so it is tested here rather than on a rented A100.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct GemmGeom {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub n_batch: usize,
    /// `(element offset, stride of the m axis, stride of the k axis, batch step)` for the left
    /// operand of the *row-major* problem, which under `ColMajor` is the caller's `b`.
    pub a: (usize, isize, isize, isize),
    /// The same for the right operand: `(offset, stride of the k axis, stride of the n axis, batch)`.
    pub b: (usize, isize, isize, isize),
    /// `(offset, stride of the m axis, stride of the n axis, batch step)` for the output.
    pub c: (usize, isize, isize, isize),
    pub lane_a: LaneAxis,
    pub lane_b: LaneAxis,
    /// True when the geometry came from the col-major reversal, i.e. `a` and `b` are the caller's
    /// second and first operands respectively.
    pub swapped: bool,
}

/// Resolve the launch geometry. Pure: no device, no allocation, no I/O.
pub(crate) fn resolve_geom(
    order: FlagOrder,
    lc: &Layout<IxD>,
    la: &Layout<IxD>,
    lb: &Layout<IxD>,
) -> Result<GemmGeom> {
    // Col-major: reverse every layout **and swap the operands**, so the kernel only sees the
    // row-major decomposition. That is the identity `rstsr_common::layout::matmul` documents,
    // `C[n,m,t] = B[n,k,t] @ A[k,m,t]`, not a device-specific convention.
    //
    // The swap has to happen here, when asking for the config: reversing alone would feed the rule
    // table `(A_rev, B_rev)`, whose contracted dimensions are `m_a` and `n_b` — a shape mismatch for
    // anything rectangular, and a silently transposed answer for anything square.
    let swapped = matches!(order, ColMajor);
    let (la_c, lb_c, lc_c) = if swapped {
        (lb.reverse_axes(), la.reverse_axes(), lc.reverse_axes())
    } else {
        (la.clone(), lb.clone(), lc.clone())
    };

    let cfg = layout_matmul_dyn_row_major_with_lc(&la_c, &lb_c, &lc_c)?;

    // The inner-dot rule leaves `la_matmul` rank 1. That rule is a reduction, not a GEMM: running it
    // here would put one block of 256 threads on a `1 x 1` output, i.e. 256x redundant arithmetic and
    // a tile of padding. It belongs on the reduce kernel, which already exists.
    if cfg.la_matmul.ndim() != 2 || cfg.lb_matmul.ndim() != 2 {
        rstsr_raise!(
            UnImplemented,
            "cuda matmul: this is a single-axis contraction (inner dot), not a GEMM; it belongs on \
             the reduce kernel and is not wired to it yet"
        )?
    }

    let (lay_a, lay_b) = (&cfg.la_matmul, &cfg.lb_matmul);
    let (m, k) = (lay_a.shape()[0], lay_a.shape()[1]);
    let (k2, n) = (lay_b.shape()[0], lay_b.shape()[1]);
    rstsr_assert_eq!(k, k2, InvalidLayout, "the contracted dimensions of a and b should be the same")?;
    rstsr_assert_eq!(
        cfg.lc_matmul.shape(),
        &[m, n][..],
        InvalidLayout,
        "the matmul output should have shape (a.shape[0], b.shape[1])"
    )?;

    // Batch: one grid slice per output matrix, so the batch offset must be affine.
    let n_batch = cfg.lc_rest.as_ref().map(|l| l.size()).unwrap_or(1);
    let mut steps = [0isize; 3];
    for (slot, rest, what) in
        [(0usize, &cfg.la_rest, "a"), (1, &cfg.lb_rest, "b"), (2, &cfg.lc_rest, "c")]
    {
        let Some(l) = rest else { continue };
        let Some(s) = linear_step(l) else {
            rstsr_raise!(
                UnImplemented,
                "cuda matmul: the batch part of {what} is not one linear run (shape {:?}, strides {:?}); \
                 grid.z can only carry an affine batch",
                l.shape(),
                l.stride()
            )?
        };
        steps[slot] = s;
    }

    let sa = (lay_a.stride()[0], lay_a.stride()[1]);
    let sb = (lay_b.stride()[0], lay_b.stride()[1]);
    let sc = (cfg.lc_matmul.stride()[0], cfg.lc_matmul.stride()[1]);

    Ok(GemmGeom {
        m,
        n,
        k,
        n_batch,
        a: (lay_a.offset(), sa.0, sa.1, steps[0]),
        b: (lay_b.offset(), sb.0, sb.1, steps[1]),
        c: (cfg.lc_matmul.offset(), sc.0, sc.1, steps[2]),
        lane_a: lane_axis(sa.0, sa.1, m, k, "the a operand")?,
        lane_b: lane_axis(sb.1, sb.0, n, k, "the b operand")?,
        swapped,
    })
}

/// `RSTSR_CUDA_MATMUL=tiled` forces the tiled kernel.
///
/// Two uses: comparing the two paths on one machine, and getting a hardware test of the fallback
/// while cuBLAS is the default. Read once, so it cannot flip between launches.
fn tiled_forced() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| matches!(std::env::var("RSTSR_CUDA_MATMUL").as_deref(), Ok("tiled")))
}

/// `RSTSR_CUDA_DEBUG_GEMM=1` prints the resolved geometry and the exact cuBLAS arguments, once.
///
/// This exists because a wrong answer has two very different causes that look alike from the test
/// side — the geometry we derived, or how the call was issued — and the only way to tell them apart
/// on a machine we cannot attach a debugger to is to print what was actually decided. Cheap enough
/// to leave in: one env lookup, and nothing at all when the variable is absent.
fn debug_gemm() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| std::env::var_os("RSTSR_CUDA_DEBUG_GEMM").is_some())
}

/// Launch `c = alpha * a @ b + beta * c` on the layout the tier already validated.
///
/// `c` is taken by shared reference on purpose: its element type is either `T` (the `beta` path) or
/// `MaybeUninit<T>` (the fresh path) and only its address is used, so no `MaybeUninit` transmute is
/// needed — nothing here writes through the Rust type.
#[allow(clippy::too_many_arguments)]
pub(crate) fn gemm_launch<T, TBuf>(
    device: &DeviceCuda,
    c: &CudaBuffer<TBuf>,
    lc: &Layout<IxD>,
    a: &CudaBuffer<T>,
    la: &Layout<IxD>,
    b: &CudaBuffer<T>,
    lb: &Layout<IxD>,
    alpha: T,
    beta: T,
) -> Result<()>
where
    T: CublasScalar,
{
    let geom = resolve_geom(device.default_order(), lc, la, lb)?;
    let GemmGeom { m, n, k, n_batch, lane_a, lane_b, .. } = geom;

    // `geom.a` always belongs to the buffer whose layout was passed first to the rule table.
    let (buf_a, buf_b) = if geom.swapped { (b, a) } else { (a, b) };

    if m == 0 || n == 0 || n_batch == 0 {
        return Ok(());
    }
    if m > i32::MAX as usize || n > i32::MAX as usize || k > i32::MAX as usize {
        rstsr_raise!(UnImplemented, "cuda matmul: a dimension exceeds i32 ({m} x {n} x {k})")?
    }

    let grid_x = (n as u64).div_ceil(BN as u64);
    let grid_y = (m as u64).div_ceil(BM as u64);
    if grid_y > u16::MAX as u64 || n_batch as u64 > u16::MAX as u64 {
        rstsr_raise!(
            UnImplemented,
            "cuda matmul: {grid_y} row tiles x {n_batch} batch slices exceeds the 65535 grid limit \
             on y and z"
        )?
    }

    let exec = device.stream();
    let _ga = order_input(buf_a, exec)?;
    let _gb = order_input(buf_b, exec)?;
    let _gc = order_output(c, exec)?;

    // cuBLAS first: it is the fast path for every layout it can express. A refusal or a non-success
    // status falls through to the tiled kernel below, which handles the same shapes correctly and
    // more slowly — so this branch can only change the speed, never the answer.
    //
    // `launch` takes the handle's lock itself, because that handle is shared by every thread using
    // this ordinal and `cublasHandle_t` is not thread-safe. That omission is not a theoretical
    // concern: it is what made four `matmul` tests fail with answers no wrong `lda` could produce.
    if !tiled_forced() {
        if let (Some(args), Some(blas)) = (cublas_gemm::args(&geom), exec.cublas()) {
            let (pa, pb, pc) = (
                offset_ptr(exec, buf_a, geom.a.0),
                offset_ptr(exec, buf_b, geom.b.0),
                offset_ptr(exec, c, geom.c.0),
            );
            if debug_gemm() {
                static ONCE: std::sync::Once = std::sync::Once::new();
                ONCE.call_once(|| {
                    eprintln!(
                        "[rstsr-cuda gemm] geom: m={m} n={n} k={k} batch={n_batch} \
                         order={:?} swapped={} opts={:?} lanes={lane_a:?}/{lane_b:?}",
                        device.default_order(),
                        geom.swapped,
                        T::CNAME
                    );
                    eprintln!("[rstsr-cuda gemm] geom.a={:?} geom.b={:?} geom.c={:?}", geom.a, geom.b, geom.c);
                    eprintln!("[rstsr-cuda gemm] args={args:?}");
                    eprintln!(
                        "[rstsr-cuda gemm] ptr_a={pa:#x} ptr_b={pb:#x} ptr_c={pc:#x} esz={}",
                        core::mem::size_of::<T>()
                    );
                });
            }
            // SAFETY: the pointers are the buffers' bases folded with the layouts' own offsets, and
            // `args` was derived from the same `geom` those offsets came from.
            let status = unsafe {
                cublas_gemm::launch(blas, &args, &alpha, &beta, pa, pb, pc)
            };
            if matches!(status, cudarc::cublas::sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS) {
                return Ok(());
            }
        }
    }

    // The two fragments differ only in which axis the linear id follows; that is what makes each
    // variant coalesced on both operands' loads.
    let a_idx = match lane_a {
        LaneAxis::K => "rstsr_usz kk = lin % @BK@; rstsr_usz mm = lin / @BK@;",
        LaneAxis::Other => "rstsr_usz mm = lin % @BM@; rstsr_usz kk = lin / @BM@;",
    };
    let b_idx = match lane_b {
        LaneAxis::K => "rstsr_usz kk = lin % @BK@; rstsr_usz nn = lin / @BK@;",
        LaneAxis::Other => "rstsr_usz nn = lin % @BN@; rstsr_usz kk = lin / @BN@;",
    };

    // Order matters: the fragments are text that still contains the numeric tokens, so they must be
    // expanded *before* the numbers, or an `@BK@` inside a fragment survives into the CUDA source.
    let subs: Vec<(&str, String)> = vec![
        ("@A_IDX@", a_idx.to_string()),
        ("@B_IDX@", b_idx.to_string()),
        ("@T@", T::CNAME.to_string()),
        ("@BM@", BM.to_string()),
        ("@BN@", BN.to_string()),
        ("@BK@", BK.to_string()),
        ("@TM@", TM.to_string()),
        ("@TN@", TN.to_string()),
        ("@TX@", TX.to_string()),
        ("@TY@", TY.to_string()),
        ("@NTHREADS@", NTHREADS.to_string()),
        ("@A_SLAB@", A_SLAB.to_string()),
        ("@B_SLAB@", B_SLAB.to_string()),
    ];
    let subs: Vec<(&str, &str)> = subs.iter().map(|(k, v)| (*k, v.as_str())).collect();

    // The discriminator must name every substitution the source depends on, or two variants collide
    // in the cache and one launches the other's code.
    let disc = format!(
        "{}|{}|{}|BM{BM}|BN{BN}|BK{BK}|TM{TM}|TN{TN}|N{NTHREADS}",
        T::CNAME,
        match lane_a {
            LaneAxis::K => "ak",
            LaneAxis::Other => "am",
        },
        match lane_b {
            LaneAxis::K => "bk",
            LaneAxis::Other => "bn",
        },
    );
    let f = planned_function(device, "rstsr_gemm", T::INCLUDES, &disc, || render(GEMM_ENTRY, &subs))?;

    let nb = core::mem::size_of::<T>();
    let mut abytes = [0u8; 16];
    let mut bbytes = [0u8; 16];
    debug_assert!(nb <= 16, "alpha/beta wider than 16 bytes is not a scalar");
    // SAFETY: `T: CudaDType` is a plain scalar, so its bytes are its on-device representation.
    unsafe {
        core::ptr::copy_nonoverlapping(core::ptr::addr_of!(alpha) as *const u8, abytes.as_mut_ptr(), nb);
        core::ptr::copy_nonoverlapping(core::ptr::addr_of!(beta) as *const u8, bbytes.as_mut_ptr(), nb);
    }

    FlatLaunch::new(exec, &f, 0)
        .explicit_shape((grid_x as u32, grid_y as u32, n_batch as u32), (TX, TY, 1))
        .arg_u64(offset_ptr(exec, c, geom.c.0))
        .arg_bytes(&geom.c.1.to_ne_bytes())
        .arg_bytes(&geom.c.2.to_ne_bytes())
        .arg_bytes(&geom.c.3.to_ne_bytes())
        .arg_u64(offset_ptr(exec, buf_a, geom.a.0))
        .arg_bytes(&geom.a.1.to_ne_bytes())
        .arg_bytes(&geom.a.2.to_ne_bytes())
        .arg_bytes(&geom.a.3.to_ne_bytes())
        .arg_u64(offset_ptr(exec, buf_b, geom.b.0))
        .arg_bytes(&geom.b.1.to_ne_bytes())
        .arg_bytes(&geom.b.2.to_ne_bytes())
        .arg_bytes(&geom.b.3.to_ne_bytes())
        .arg_i32(m as i32)
        .arg_i32(n as i32)
        .arg_i32(k as i32)
        .arg_bytes(&abytes[..nb])
        .arg_bytes(&bbytes[..nb])
        .launch()?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layout(shape: &[usize], stride: &[isize]) -> Layout<IxD> {
        Layout::<IxD>::new(shape.to_vec(), stride.to_vec(), 0).unwrap()
    }

    /// A contiguous batch, a broadcast batch and a rank-1 batch are all affine.
    #[test]
    fn affine_batches_are_linear() {
        assert_eq!(linear_step(&layout(&[4], &[64])), Some(64));
        assert_eq!(linear_step(&layout(&[4, 3], &[3, 1])), Some(1), "contiguous = step 1 over the flat index");
        assert_eq!(linear_step(&layout(&[4], &[0])), Some(0), "broadcast");
        assert_eq!(linear_step(&layout(&[4, 1], &[3, 0])), Some(3), "size-1 axes contribute nothing");
        assert_eq!(linear_step(&layout(&[4, 3], &[0, 0])), Some(0), "all-broadcast");
        assert_eq!(linear_step(&layout(&[], &[])), Some(0), "no batch at all");
    }

    /// A batch that walks two axes with a hole between them is not one step, and is refused.
    #[test]
    fn gapped_batches_are_refused() {
        assert_eq!(linear_step(&layout(&[4, 3], &[6, 1])), None);
        assert_eq!(linear_step(&layout(&[4, 3], &[9, 1])), None);
    }

    /// The lane map prefers the unit-stride axis, and falls back to the axis that actually varies
    /// when the other is degenerate.
    #[test]
    fn lane_axis_prefers_the_unit_stride() {
        assert!(matches!(lane_axis(64, 1, 64, 64, "a"), Ok(LaneAxis::K)));
        assert!(matches!(lane_axis(1, 64, 64, 64, "a"), Ok(LaneAxis::Other)));
        // [1, K] with a strided k axis: nothing is unit-stride, but m is degenerate, so following
        // k is still the right map.
        assert!(matches!(lane_axis(0, 2, 1, 64, "a"), Ok(LaneAxis::K)));
        // genuinely stepped on both axes
        assert!(lane_axis(2, 64, 64, 64, "a").is_err());
    }

    /// The offset a layout assigns to one multi-index. This is the tensor layer's own mapping, so
    /// comparing the geom against it checks *which layout plays which role*, not the arithmetic of
    /// `offset + i*stride` twice over.
    fn off(l: &Layout<IxD>, idx: &[usize]) -> isize {
        let mut o = l.offset() as isize;
        for (k, &i) in idx.iter().enumerate() {
            o += i as isize * l.stride()[k];
        }
        o
    }

    /// Row-major: `geom.a` walks `a[i, k]`, `geom.b` walks `b[k, j]`, `geom.c` walks `c[i, j]`.
    #[test]
    fn row_major_geom_matches_the_layouts() {
        let (m, k, n) = (2usize, 3usize, 4usize);
        let la = layout(&[m, k], &[k as isize, 1]);
        let lb = layout(&[k, n], &[n as isize, 1]);
        let lc = layout(&[m, n], &[n as isize, 1]);
        let g = resolve_geom(RowMajor, &lc, &la, &lb).unwrap();

        assert_eq!((g.m, g.n, g.k, g.n_batch), (m, n, k, 1));
        assert!(!g.swapped);
        assert!(matches!(g.lane_a, LaneAxis::K), "k is unit-stride for a");
        assert!(matches!(g.lane_b, LaneAxis::Other), "n is unit-stride for b");

        for i in 0..m {
            for kk in 0..k {
                assert_eq!(g.a.0 as isize + i as isize * g.a.1 + kk as isize * g.a.2, off(&la, &[i, kk]));
            }
            for j in 0..n {
                assert_eq!(g.c.0 as isize + i as isize * g.c.1 + j as isize * g.c.2, off(&lc, &[i, j]));
            }
        }
        for kk in 0..k {
            for j in 0..n {
                assert_eq!(g.b.0 as isize + kk as isize * g.b.1 + j as isize * g.b.2, off(&lb, &[kk, j]));
            }
        }
    }

    /// Column-major: the roles swap and every index is read in the other order. This is the case a
    /// GPU run would report only as "wrong numbers", so it is pinned by walking each buffer and
    /// comparing against the tensor layout's own offset.
    #[test]
    fn col_major_geom_swaps_roles_and_index_order() {
        let (m, k, n) = (2usize, 3usize, 4usize);
        // column-major = first axis contiguous
        let la = layout(&[m, k], &[1, m as isize]);
        let lb = layout(&[k, n], &[1, k as isize]);
        let lc = layout(&[m, n], &[1, m as isize]);
        let g = resolve_geom(ColMajor, &lc, &la, &lb).unwrap();

        assert!(g.swapped, "col-major must then read the operands the other way round");
        assert_eq!((g.m, g.n, g.k), (n, m, k), "the problem is solved transposed");
        assert_eq!(g.m * g.n, m * n, "but it is still the same output matrix");

        // geom.a is the caller's b, walked as b[k, n]: geom's (i, kk) is b's (kk, i)
        for i in 0..g.m {
            for kk in 0..g.k {
                assert_eq!(g.a.0 as isize + i as isize * g.a.1 + kk as isize * g.a.2, off(&lb, &[kk, i]));
            }
        }
        // geom.b is the caller's a, walked as a[m, k]: geom's (kk, j) is a's (j, kk)
        for kk in 0..g.k {
            for j in 0..g.n {
                assert_eq!(g.b.0 as isize + kk as isize * g.b.1 + j as isize * g.b.2, off(&la, &[j, kk]));
            }
        }
        // geom.c is c[n, m]: geom's (i, j) is c's (j, i)
        for i in 0..g.m {
            for j in 0..g.n {
                assert_eq!(g.c.0 as isize + i as isize * g.c.1 + j as isize * g.c.2, off(&lc, &[j, i]));
            }
        }
    }

    /// A batched problem: the batch step must be the stride of one whole batched matrix, and a
    /// broadcast operand must have step 0.
    #[test]
    fn batch_step_is_one_matrix() {
        let (p, m, k, n) = (4usize, 5usize, 6usize, 7usize);
        let la = layout(&[p, m, k], &[(m * k) as isize, k as isize, 1]);
        let lb = layout(&[p, k, n], &[(k * n) as isize, n as isize, 1]);
        let lc = layout(&[p, m, n], &[(m * n) as isize, n as isize, 1]);
        let g = resolve_geom(RowMajor, &lc, &la, &lb).unwrap();
        assert_eq!(g.n_batch, p);
        assert_eq!(g.a.3, (m * k) as isize);
        assert_eq!(g.b.3, (k * n) as isize);
        assert_eq!(g.c.3, (m * n) as isize);

        // left operand broadcast over the batch: its batch stride is 0
        let la1 = layout(&[m, k], &[k as isize, 1]);
        let g1 = resolve_geom(RowMajor, &lc, &la1, &lb).unwrap();
        assert_eq!(g1.n_batch, p);
        assert_eq!(g1.a.3, 0, "a broadcast batch contributes no offset");
        assert_eq!(g1.b.3, (k * n) as isize);
    }

    /// The inner-dot rule has no 2-D matmul part, and is refused rather than run as a 1x1 tile.
    #[test]
    fn inner_dot_has_no_geometry() {
        let la = layout(&[6], &[1]);
        let lc = layout(&[], &[]);
        let e = resolve_geom(RowMajor, &lc, &la, &la).unwrap_err();
        assert!(format!("{e}").contains("single-axis contraction"), "got: {e}");
    }

    /// Tile geometry has to stay self-consistent: the kernel derives slab counts from it and the
    /// shared arrays from `BM`/`BN`.
    #[test]
    fn tile_geometry_is_consistent() {
        assert_eq!(TX * TY, NTHREADS);
        assert_eq!(TM * TY, BM);
        assert_eq!(TN * TX, BN);
        assert_eq!(BM * BK, A_SLAB * NTHREADS);
        assert_eq!(BK * BN, B_SLAB * NTHREADS);
        // static shared memory for both slabs must fit the 48 KB static limit
        let smem = (BK * (BM + 1) + BK * (BN + 1)) * 8;
        assert!(smem <= 48 * 1024, "static shared memory {smem} B is over the 48 KB limit");
    }
}
