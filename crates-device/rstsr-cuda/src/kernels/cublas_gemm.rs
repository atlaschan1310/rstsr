//! cuBLAS `gemm` — the **primary** GEMM path; [`super::gemm`]'s tiled kernel is the fallback.
//!
//! # Why this exists at all
//!
//! Because there is no competition. On an A100, `dgemm` reaches the FP64 tensor-core pipe (19.5
//! TFLOPS) where a SIMT tile tops out at 9.7, and on every other shape it is at least as fast for
//! a hundredth of the engineering. The tiled kernel in [`super::gemm`] is kept for the cases this
//! module cannot express (see below) and as the test oracle — not as a competitor.
//!
//! # The calling convention, and the one thing that must not be got wrong
//!
//! cuBLAS is column-major only, and `C` has **no transpose flag** — only `ldc`. So the whole mapping
//! rests on the identity `C = A·B  <=>  Cᵀ = Bᵀ·Aᵀ`, which fixes the roles completely:
//!
//! | cuBLAS slot | our operand | must present as | how |
//! |---|---|---|---|
//! | `op(A_o)` | `geom.b` (`k × n`) | `n × k` | `n` contiguous → `OP_N`, ld = `sbk`; else `k` contiguous → `OP_T`, ld = `sbn` |
//! | `op(B_o)` | `geom.a` (`m × k`) | `k × m` | `k` contiguous → `OP_N`, ld = `sam`; else `m` contiguous → `OP_T`, ld = `sak` |
//! | `C_o` | `geom.c` (`m × n`) | `n × m` | requires `scn == 1`, ld = `scm` |
//!
//! with `m_o = n`, `n_o = m`, `k_o = k`. Note that the **operands are swapped relative to their
//! mathematical roles**: `geom.b` goes in cuBLAS's A slot. Getting either the swap or one `OP_*`
//! flag wrong produces a plausible transposed answer rather than an error, which is why
//! [`args`] is pure and is tested by re-deriving the same products through the declared leading
//! dimensions ([`tests::declared_lds_reproduce_the_product`]).
//!
//! `geom` is already order-normalised: [`crate::kernels::gemm::resolve_geom`] reverses and swaps for
//! `ColMajor`, so this module only ever sees a row-major description.
//!
//! # What is refused (and falls back to the tiled kernel)
//!
//! - `scn != 1`: the output cannot be presented as `n × m` column-major. Fresh outputs never hit
//!   this; a stepped or transposed view does.
//! - neither stride of an operand is 1: cuBLAS needs one leading dimension, so a stepped 2-D block
//!   needs a pack. The tiled kernel takes it directly.
//! - a non-float dtype, a dimension over `i32`, or a failed `cublasCreate`/`cublasDgemm`.
//!
//! Every refusal is silent: the caller runs the tiled kernel and gets the right answer, slower.
//!
//! # Thread safety
//!
//! A `cublasHandle_t` is not thread-safe, and this crate shares one per ordinal (see
//! [`CudaStreamObj`][crate::stream::CudaStreamObj]). [`launch`] therefore takes the handle's lock
//! itself. That is a requirement of cuBLAS, not a precaution: two host threads entering a `gemm` on
//! one handle at the same time is undefined.

use cudarc::cublas::sys;

use crate::dtype::CudaDType;
use crate::kernels::gemm::GemmGeom;
use crate::prelude_dev::*;
use crate::stream::CublasHandle;

/// The float widths cuBLAS's typed `gemm` covers, which is exactly what `DeviceMatMulAPI` is
/// instantiated for on this device.
pub trait CublasScalar: CudaDType {
    /// # Safety
    /// `a`, `b`, `c` must be device pointers of at least the extents implied by
    /// `m, n, k, lda, ldb, ldc`, on the stream the handle is bound to.
    #[allow(clippy::too_many_arguments)]
    unsafe fn gemm(
        handle: sys::cublasHandle_t,
        transa: sys::cublasOperation_t,
        transb: sys::cublasOperation_t,
        m: i32,
        n: i32,
        k: i32,
        alpha: &Self,
        a: u64,
        lda: i32,
        b: u64,
        ldb: i32,
        beta: &Self,
        c: u64,
        ldc: i32,
    ) -> sys::cublasStatus_t;

    /// Same contract, plus `batch` matrices spaced `stride_*` elements apart.
    #[allow(clippy::too_many_arguments)]
    unsafe fn gemm_strided_batched(
        handle: sys::cublasHandle_t,
        transa: sys::cublasOperation_t,
        transb: sys::cublasOperation_t,
        m: i32,
        n: i32,
        k: i32,
        alpha: &Self,
        a: u64,
        lda: i32,
        stride_a: i64,
        b: u64,
        ldb: i32,
        stride_b: i64,
        beta: &Self,
        c: u64,
        ldc: i32,
        stride_c: i64,
        batch: i32,
    ) -> sys::cublasStatus_t;
}

macro_rules! impl_cublas_scalar {
    ($t:ty, $gemm:ident, $gemm_sb:ident) => {
        impl CublasScalar for $t {
            unsafe fn gemm(
                handle: sys::cublasHandle_t,
                transa: sys::cublasOperation_t,
                transb: sys::cublasOperation_t,
                m: i32,
                n: i32,
                k: i32,
                alpha: &Self,
                a: u64,
                lda: i32,
                b: u64,
                ldb: i32,
                beta: &Self,
                c: u64,
                ldc: i32,
            ) -> sys::cublasStatus_t {
                unsafe {
                    sys::$gemm(
                        handle,
                        transa,
                        transb,
                        m,
                        n,
                        k,
                        alpha,
                        a as *const $t,
                        lda,
                        b as *const $t,
                        ldb,
                        beta,
                        c as *mut $t,
                        ldc,
                    )
                }
            }

            unsafe fn gemm_strided_batched(
                handle: sys::cublasHandle_t,
                transa: sys::cublasOperation_t,
                transb: sys::cublasOperation_t,
                m: i32,
                n: i32,
                k: i32,
                alpha: &Self,
                a: u64,
                lda: i32,
                stride_a: i64,
                b: u64,
                ldb: i32,
                stride_b: i64,
                beta: &Self,
                c: u64,
                ldc: i32,
                stride_c: i64,
                batch: i32,
            ) -> sys::cublasStatus_t {
                unsafe {
                    sys::$gemm_sb(
                        handle,
                        transa,
                        transb,
                        m,
                        n,
                        k,
                        alpha,
                        a as *const $t,
                        lda,
                        stride_a,
                        b as *const $t,
                        ldb,
                        stride_b,
                        beta,
                        c as *mut $t,
                        ldc,
                        stride_c,
                        batch,
                    )
                }
            }
        }
    };
}

impl_cublas_scalar!(f32, cublasSgemm_v2, cublasSgemmStridedBatched);
impl_cublas_scalar!(f64, cublasDgemm_v2, cublasDgemmStridedBatched);

/// One `gemm` call, in cuBLAS's terms.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct CublasArgs {
    pub mo: i32,
    pub no: i32,
    pub ko: i32,
    pub transa: bool,
    pub transb: bool,
    pub lda: i32,
    pub ldb: i32,
    pub ldc: i32,
    /// Batch element steps, named after the **geom** operands. `batch_b` belongs to `geom.b`, which
    /// [`launch`] places in cuBLAS's A slot — the naming follows the data, not the slot.
    ///
    /// **These are element counts, not bytes.** cuBLAS documents `strideA` as "the offset in number
    /// of elements between `A[i]` and `A[i+1]`", which is the opposite of what `cublasSetVector`'s
    /// `elemSize` does two sections earlier, and getting it wrong is invisible for `batchCount == 1`.
    pub batch_a: isize,
    pub batch_b: isize,
    pub batch_c: isize,
    pub batch: i32,
}

/// Largest leading dimension cuBLAS accepts for a declared matrix of `rows` rows.
///
/// A broadcast axis (extent 1, stride 0) gives a leading dimension of 0, which cuBLAS rejects;
/// clamping to the declared row count is safe because that axis has only index 0.
fn ld(stride: isize, rows: usize) -> i32 {
    let s = stride.max(0);
    let clamped = s.max(rows as isize);
    clamped.min(i32::MAX as isize) as i32
}

/// Derive the cuBLAS arguments, or `None` when this layout is not expressible.
///
/// Pure on purpose: this is the part where a mistake produces a silently transposed answer, so it is
/// tested against the declared leading dimensions rather than only on hardware.
pub(crate) fn args(geom: &GemmGeom) -> Option<CublasArgs> {
    let (m, n, k) = (geom.m, geom.n, geom.k);
    let (_, sam, sak, sa_b) = geom.a;
    let (_, sbk, sbn, sb_b) = geom.b;
    let (_, scm, scn, sc_b) = geom.c;

    // An extent-1 axis may legitimately carry stride 0; anything else with a non-positive stride
    // (a reversed view) has no cuBLAS expression at all.
    for (stride, extent) in [(sam, m), (sak, k), (sbk, k), (sbn, n), (scm, m), (scn, n)] {
        if stride < 0 && extent > 1 {
            return None;
        }
    }
    // `C` has no transpose flag: it must be presentable as `n x m` column-major.
    if scn != 1 {
        return None;
    }
    if m > i32::MAX as usize || n > i32::MAX as usize || k > i32::MAX as usize {
        return None;
    }

    // Slot A = geom.b, which must appear as n x k.
    let (transa, lda_raw, rows_a) = if sbn == 1 {
        (false, sbk, n)
    } else if sbk == 1 {
        (true, sbn, k)
    } else {
        return None;
    };
    // Slot B = geom.a, which must appear as k x m.
    let (transb, ldb_raw, rows_b) = if sak == 1 {
        (false, sam, k)
    } else if sam == 1 {
        (true, sak, m)
    } else {
        return None;
    };

    let (lda, ldb, ldc) = (ld(lda_raw, rows_a), ld(ldb_raw, rows_b), ld(scm, n));
    // `ld(a/b)` went through `max(0)`, so a genuinely negative stride can hide there; re-check.
    if lda_raw < 0 || ldb_raw < 0 || ldc == 0 {
        return None;
    }

    if geom.n_batch > i32::MAX as usize {
        return None;
    }

    Some(CublasArgs {
        mo: n as i32,
        no: m as i32,
        ko: k as i32,
        transa,
        transb,
        lda,
        ldb,
        ldc,
        batch_a: sa_b,
        batch_b: sb_b,
        batch_c: sc_b,
        batch: geom.n_batch as i32,
    })
}

fn op(t: bool) -> sys::cublasOperation_t {
    if t {
        sys::cublasOperation_t::CUBLAS_OP_T
    } else {
        sys::cublasOperation_t::CUBLAS_OP_N
    }
}

/// The `strideA`, `strideB`, `strideC` triple cuBLAS receives, in **elements**, ordered by cuBLAS's
/// slots rather than by our operand names: `geom.b` sits in slot A, `geom.a` in slot B.
///
/// This is a separate function for one reason. cuBLAS's typed strided-batched API takes element
/// offsets — "the offset in number of elements between `A[i]` and `A[i+1]`" — while `cublasSetVector`
/// two sections earlier in the same manual takes bytes. The conversion therefore must **not** happen
/// anywhere, and a `* size_of::<T>()` inserted here is invisible to every `batchCount == 1` test: it
/// leaves the first slice exact, loses the rest, and writes the surplus into whatever the allocator
/// placed next, which is how it presented on hardware (a batched matmul with correct first `m*n`
/// outputs and zeroed later slices, plus unrelated tests failing at random). Naming the triple keeps
/// the unit in one place, where [`tests::batch_strides_are_element_counts`] can assert it.
fn batch_strides(a: &CublasArgs) -> (i64, i64, i64) {
    (a.batch_b as i64, a.batch_a as i64, a.batch_c as i64)
}

/// Issue the `gemm`. Returns the cuBLAS status; the caller falls back to the tiled kernel on any
/// non-success, because a returned status is a launch-time validation failure and nothing has been
/// enqueued.
///
/// The pointers are named after the **geom** operands; the slot swap happens here, once, and is the
/// same swap [`args`] assumes.
///
/// # The strides are element counts
///
/// `strideA/B/C` are passed straight through from [`CublasArgs`], which holds them in elements
/// because that is what cuBLAS's strided-batched API takes — "the offset in number of elements
/// between `A[i]` and `A[i+1]`". Multiplying by `size_of::<T>()` here is the obvious-looking mistake
/// and it does not fail loudly: `batchCount == 1` ignores the strides entirely, and for a batch the
/// first slice is still correct while every later slice is read and written eight times too far
/// along for `f64`. On hardware that showed up as a batched matmul whose first `m*n` outputs were
/// exact and whose remaining slices were untouched, plus *other* tests failing at random with values
/// larger than the operands could produce — the stray writes landing in whatever the caching
/// allocator had placed next.
///
/// The handle's lock is taken **here** rather than by the caller: a `cublasHandle_t` is not
/// thread-safe, and this stream's handle is shared by every thread that uses the ordinal, so this is
/// the one call that has to be serialized and the one place a guard cannot be forgotten. See
/// [`CublasHandle`].
///
/// # Safety
/// The three pointers must be valid for `geom`'s extents on the handle's stream, including the batch
/// range `batchCount * stride` reaches.
pub(crate) unsafe fn launch<T: CublasScalar>(
    blas: &CublasHandle,
    a: &CublasArgs,
    alpha: &T,
    beta: &T,
    ptr_a: u64,
    ptr_b: u64,
    ptr_c: u64,
) -> sys::cublasStatus_t {
    let _api = blas.lock();
    let handle = *blas.blas().handle();
    let (stride_a, stride_b, stride_c) = batch_strides(a);
    unsafe {
        if a.batch > 1 {
            T::gemm_strided_batched(
                handle,
                op(a.transa),
                op(a.transb),
                a.mo,
                a.no,
                a.ko,
                alpha,
                // slot A holds `geom.b`, slot B holds `geom.a`; the strides follow the buffer.
                ptr_b,
                a.lda,
                stride_a,
                ptr_a,
                a.ldb,
                stride_b,
                beta,
                ptr_c,
                a.ldc,
                stride_c,
                a.batch,
            )
        } else {
            T::gemm(
                handle, op(a.transa), op(a.transb), a.mo, a.no, a.ko, alpha, ptr_b, a.lda, ptr_a, a.ldb, beta, ptr_c,
                a.ldc,
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::gemm::resolve_geom;

    fn layout(shape: &[usize], stride: &[isize], offset: usize) -> Layout<IxD> {
        Layout::<IxD>::new(shape.to_vec(), stride.to_vec(), offset).unwrap()
    }

    /// Element index of the operand in cuBLAS's **A** slot (`geom.b`), addressed the way the
    /// *declared* matrix and `lda` say it should be — not the way `geom` says.
    fn slot_a(g: &GemmGeom, a: &CublasArgs, ni: usize, ki: usize) -> usize {
        let base = g.b.0;
        if a.transa {
            // declared k x n, read transposed
            base + ki + ni * a.lda as usize
        } else {
            // declared n x k
            base + ni + ki * a.lda as usize
        }
    }

    /// Element index of the operand in cuBLAS's **B** slot (`geom.a`).
    fn slot_b(g: &GemmGeom, a: &CublasArgs, ki: usize, mi: usize) -> usize {
        let base = g.a.0;
        if a.transb {
            base + mi + ki * a.ldb as usize
        } else {
            base + ki + mi * a.ldb as usize
        }
    }

    /// The output, as `n x m` column-major.
    fn slot_c(g: &GemmGeom, a: &CublasArgs, ni: usize, mi: usize) -> usize {
        g.c.0 + ni + mi * a.ldc as usize
    }

    fn data(n: usize, seed: usize) -> Vec<f64> {
        (0..n).map(|i| (((i * 7 + seed) % 9) as f64) - 4.0).collect()
    }

    /// Every orientation the derivation claims to support, plus a non-zero offset on all three
    /// layouts: the `gemm` result re-derived through the *declared* leading dimensions must equal the
    /// one computed through `geom`'s own strides. A wrong `OP_*` flag, a wrong `ld`, or a missing
    /// operand swap changes the answer here rather than in a rented-GPU session.
    #[test]
    fn declared_lds_reproduce_the_product() {
        let (m, k, n) = (3usize, 4usize, 5usize);
        // (a's strides, b's strides, label). One of each operand's two axes must be unit.
        let cases: &[([isize; 2], [isize; 2], &str)] = &[
            ([4, 1], [5, 1], "a k-contig, b n-contig"),
            ([1, 4], [5, 1], "a m-contig, b n-contig"),
            ([4, 1], [1, 4], "a k-contig, b k-contig"),
            ([1, 4], [1, 4], "a m-contig, b k-contig"),
        ];
        for (a_stride, b_stride, label) in cases {
            for offset in [0usize, 3] {
                let la = layout(&[m, k], a_stride, offset);
                let lb = layout(&[k, n], b_stride, offset + 1);
                let lc = layout(&[m, n], &[n as isize, 1], offset + 2);
                let g = resolve_geom(RowMajor, &lc, &la, &lb).unwrap();
                let args = args(&g).unwrap_or_else(|| panic!("{label}: should be expressible"));

                let av = data(64, 1);
                let bv = data(64, 2);

                for mi in 0..m {
                    for ni in 0..n {
                        // What the declared lds say this output element is. Note which buffer each
                        // slot reads: cuBLAS's A slot holds `geom.b` (so `bv`), its B slot holds
                        // `geom.a` (so `av`). Swapping these is the mistake this test exists to
                        // catch — and it caught it here first.
                        let declared: f64 = (0..k)
                            .map(|ki| bv[slot_a(&g, &args, ni, ki)] * av[slot_b(&g, &args, ki, mi)])
                            .sum();
                        // what the geometry says, straight from each operand's own strides
                        let reference: f64 = (0..k)
                            .map(|ki| {
                                let ai = g.a.0 + mi * g.a.1 as usize + ki * g.a.2 as usize;
                                let bi = g.b.0 + ki * g.b.1 as usize + ni * g.b.2 as usize;
                                av[ai] * bv[bi]
                            })
                            .sum();
                        assert_eq!(declared, reference, "{label}, offset {offset}, C[{mi},{ni}]");
                        // and the output slot addresses the same cell the layout does
                        assert_eq!(
                            slot_c(&g, &args, ni, mi),
                            g.c.0 + mi * g.c.1 as usize + ni * g.c.2 as usize,
                            "{label}, offset {offset}: output addressing"
                        );
                    }
                }
            }
        }
    }

    /// Refusals, at the stage where each one actually happens.
    ///
    /// The two stages are not interchangeable: `resolve_geom` admits a stepped **output** (the tiled
    /// kernel addresses `c + m*scm + n*scn`, so it does not care), and it is `args` that has to
    /// refuse it, because cuBLAS's output side has only `ldc` and no transpose flag. A stepped
    /// **operand** is refused earlier, by the lane-axis choice, since our kernel needs one
    /// unit-stride axis to put the lanes on.
    #[test]
    fn refusals_happen_at_the_earliest_stage() {
        let lb = layout(&[4, 3], &[3, 1], 0);

        // stepped output: legal for our kernel, inexpressible for cuBLAS
        let la = layout(&[2, 4], &[4, 1], 0);
        let lc = layout(&[2, 3], &[6, 2], 0);
        let g = resolve_geom(RowMajor, &lc, &la, &lb).unwrap();
        assert_eq!((g.c.1, g.c.2), (6, 2), "fixture: the output really is stepped");
        assert!(args(&g).is_none(), "a stepped output has no column-major expression");

        // stepped operand: refused by the lane-axis choice
        let la2 = layout(&[2, 4], &[16, 2], 0);
        let lc2 = layout(&[2, 3], &[3, 1], 0);
        let e = resolve_geom(RowMajor, &lc2, &la2, &lb).unwrap_err();
        assert!(format!("{e}").contains("no unit-stride axis"), "got: {e}");

        // reversed operand: accepted by the classifier (the kernel walks backwards), so this is the
        // one refusal `args` owns — cuBLAS has no negative leading dimension.
        let lc3 = layout(&[3, 3], &[3, 1], 0);
        let lb3 = layout(&[4, 3], &[3, 1], 0);
        let la3 = layout(&[3, 4], &[-4, 1], 8);
        // If the layout itself is rejected as out of bounds the case is unreachable and the guard in
        // `args` is defence in depth rather than a live branch, so only the Ok arm asserts.
        if let Ok(g) = resolve_geom(RowMajor, &lc3, &la3, &lb3) {
            assert!(args(&g).is_none(), "a reversed operand has no cuBLAS expression");
        }
    }

    /// The batch strides are **element** counts and exactly one matrix apart.
    ///
    /// The convention is cuBLAS's, not ours: the typed strided-batched API documents `strideA` as
    /// "the offset in number of elements between `A[i]` and `A[i+1]`", while `cublasSetVector` takes
    /// bytes. A `* size_of::<T>()` in [`batch_strides`] — the one place that produces these values —
    /// is invisible to every `batchCount == 1` fixture: it leaves the first slice exact, drops the
    /// rest, and writes the surplus into whatever the allocator placed next. That is not a
    /// hypothetical: it is what the first hardware run of this path did.
    #[test]
    fn batch_strides_are_element_counts() {
        let (p, m, k, n) = (5usize, 12usize, 16usize, 8usize);
        let la = layout(&[p, m, k], &[(m * k) as isize, k as isize, 1], 0);
        let lb = layout(&[p, k, n], &[(k * n) as isize, n as isize, 1], 0);
        let lc = layout(&[p, m, n], &[(m * n) as isize, n as isize, 1], 0);
        let g = resolve_geom(RowMajor, &lc, &la, &lb).unwrap();
        let a = args(&g).unwrap();
        assert_eq!(a.batch, p as i32);

        let (stride_a, stride_b, stride_c) = batch_strides(&a);
        // slot A holds `geom.b` (k x n), slot B holds `geom.a` (m x k)
        assert_eq!(stride_a, (k * n) as i64, "slot A steps by one k x n matrix, in elements");
        assert_eq!(stride_b, (m * k) as i64, "slot B steps by one m x k matrix, in elements");
        assert_eq!(stride_c, (m * n) as i64, "the output steps by one m x n matrix, in elements");

        // The last slice's last element must still be inside each buffer. This is the assertion an
        // element-size scaling breaks, and only a batched fixture can fail it.
        let last = a.batch as i64 - 1;
        for (step, per_slice, whole, what) in [
            (stride_a, (k * n) as i64, (p * k * n) as i64, "slot A (geom.b)"),
            (stride_b, (m * k) as i64, (p * m * k) as i64, "slot B (geom.a)"),
            (stride_c, (m * n) as i64, (p * m * n) as i64, "output"),
        ] {
            assert!(last * step + per_slice <= whole, "{what}: the batch walk leaves its buffer");
        }
    }

    /// A broadcast batch axis is a zero stride, and a size-1 leading dimension is clamped up rather
    /// than passed as 0 (cuBLAS rejects ld = 0).
    #[test]
    fn degenerate_axes_are_clamped() {
        let (f, m, k, n) = (4usize, 5usize, 6usize, 7usize);
        let la = layout(&[m, k], &[k as isize, 1], 0);
        let lb = layout(&[f, k, n], &[(k * n) as isize, n as isize, 1], 0);
        let lc = layout(&[f, m, n], &[(m * n) as isize, n as isize, 1], 0);
        let g = resolve_geom(RowMajor, &lc, &la, &lb).unwrap();
        let args = args(&g).unwrap();
        assert_eq!(args.batch, f as i32);
        assert_eq!(args.batch_a, 0, "a broadcast batch keeps stride 0");
        assert_eq!(args.batch_b, (k * n) as isize);
        assert_eq!(args.batch_c, (m * n) as isize);
        assert!(args.lda >= n as i32 && args.ldb >= k as i32 && args.ldc >= n as i32);
    }
}
