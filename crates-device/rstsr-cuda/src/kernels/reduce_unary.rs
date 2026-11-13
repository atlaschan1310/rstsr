//! Unary reduction kernels: one tokenised source, one row per operator.
//!
//! # Why this is a second kernel and not a parameter of [`super::reduce`]
//!
//! `vecdot` reads two operands per element. A unary reduction reads one. The *geometry* is identical
//! — the plan from [`super::reduce_plan`] is shared verbatim — but the per-element body and the
//! argument list are not, so pretending they are one entry point would mean a null pointer and a
//! stride nobody uses. Two entries, one plan, and the unary operators share this one:
//!
//! ```text
//! op       INIT      FOLD(A, X)              COMB(A, B)    POST(V)
//! sum      0         A + X                   A + B         V
//! prod     1         A * X                   A * B         V
//! mean     0         A + X                   A + B         V / n_contract
//! l2_norm  0         A + X*X                 A + B         sqrt(V)
//! min      +inf      RSTSR_MIN(A, X)         RSTSR_MIN     V
//! max      -inf      RSTSR_MAX(A, X)         RSTSR_MAX     V
//! all      1         A & X                   A & B         V != 0
//! any      0         A | X                   A | B         V != 0
//! count_nz 0         A + (X != 0)            A + B         V
//! ```
//!
//! The last three are the rows whose accumulator is *not* the operand type: `all`/`any` fold into an
//! `int` and `count_nonzero` into an `unsigned long long`, which is why the accumulator travels as an
//! [`AccSpec`] value rather than being inferred from `T`.
//!
//! # Three types, not one
//!
//! `@TIN@` is what is loaded, `@TACC@` is what the tree adds up, `@TOUT@` is what is stored. They are
//! equal for most rows, but the split is load-bearing:
//!
//! - `sum_axes_dtype` casts a narrower input into a wider accumulator (`f32 → f64`, `i32 → f64`),
//!   which is what `into_cast` means on the CPU side. Every fold casts the element **explicitly**
//!   (`(@TACC@)(X)`) rather than relying on C++'s implicit promotion, because `acc + x` with mixed
//!   types rounds to the wider type first and then narrows on assignment — one rounding too many.
//! - `l2_norm` accumulates a *real* quantity from a possibly complex operand. Only real dtypes are
//!   wired up here (`Real = Self`), so the three names coincide; a complex operand needs `|x|^2`
//!   in the fold and is not supported yet.
//!
//! # The identity is what makes the guard safe
//!
//! Threads beyond `n_out` must not `return` — that would leave part of a warp out of the shuffles —
//! and *within* a live group, lanes at or past `n_contract` do zero iterations. Both keep
//! `acc == INIT` and take part in the tree, which is only correct because `INIT` is the operation's
//! identity: a zero cannot change a sum, `+inf` cannot change a min. Any future op must satisfy the
//! same property or the tail block silently gains elements.
//!
//! # min/max are NaN-*propagating*
//!
//! The CPU folds with `ExtReal::ext_min`, which is IEEE 754-2019 `minimum`: if either operand is NaN
//! the result is NaN. CUDA's `fmin`/`fmax` are the opposite convention — they return the non-NaN
//! operand — so the macros below add the guard, and `fmin` is kept for its signed-zero handling
//! (`minimum(-0.0, +0.0) == -0.0`, which a bare `<` test gets wrong). The *payload* of a NaN can
//! differ from the CPU's, because which operand is returned depends on the tree shape; that is not
//! observable in `x != x` terms, and summation order was never promised.
//!
//! # The tree
//!
//! Identical to [`super::reduce`]: an in-warp butterfly masked by `lane + off < LANES`, then — only
//! when `LANES > warpSize` — a shared-memory stage. It is duplicated rather than shared because the
//! two files differ in the body and in the argument list; if a third kernel appears, factor the tree
//! into an included source fragment before it becomes four copies.

use crate::prelude_dev::*;
use std::fmt::Write as _;
use std::sync::Arc;

use cudarc::driver::CudaFunction;
use crate::dtype::CudaDType;
use super::reduce_plan::{ReducePlan, BLOCK};

/// Largest number of warps a block can contain, assuming a warp is at least 32 lanes.
const MAX_WARPS_PER_BLOCK: usize = BLOCK / 32;

/// The three strings the accumulator type contributes to the generated source.
///
/// Carried **by value** into the launcher rather than as a type parameter. A phantom `TAcc` cannot
/// be inferred at the call site — nothing in the argument list mentions it — so every caller would
/// need an explicit turbofish, and the one that forgets gets `E0283`. A value has no such problem and
/// costs three `&'static str` in the launch, not in the kernel.
#[derive(Clone, Copy, Debug)]
pub(crate) struct AccSpec {
    /// C name of the accumulator — the same spelling `CudaDType::CNAME` gives this type, so the
    /// buffers and the arithmetic cannot drift apart.
    pub cname: &'static str,
    /// `+inf`/`-inf`, spelled with compiler builtins rather than `INFINITY`.
    pub pos_inf: &'static str,
    pub neg_inf: &'static str,
}

impl AccSpec {
    /// Accumulator for the predicates: `all`/`any` fold into an `int`, not a `bool`, because
    /// `__shfl_down_sync` has no `bool` overload and an integer promotion in the overload set is
    /// one silent conversion more than this needs.
    pub(crate) const INT: AccSpec = AccSpec { cname: "int", pos_inf: "", neg_inf: "" };

    /// Accumulator for the counting rows (`count_nonzero`, `sum` over `bool`). `usize` and `u64` are
    /// both `unsigned long long` on the device, which is what `CudaDType` gives them.
    pub(crate) const U64: AccSpec =
        AccSpec { cname: "unsigned long long", pos_inf: "", neg_inf: "" };
}

/// The accumulator types the fold table can be written in.
///
/// This exists for the two infinities, not for the name: NVRTC is not assumed to provide
/// `<math.h>`'s `INFINITY` without an include path. `f32`/`f64` are the only rows, which is also the
/// honest statement of which dtypes the table supports — a complex accumulator would need
/// `@TACC@` to be a struct.
pub(crate) trait ReduceAcc {
    const SPEC: AccSpec;
}

impl ReduceAcc for f32 {
    const SPEC: AccSpec = AccSpec {
        cname: "float",
        pos_inf: "(__int_as_float(0x7f800000))",
        neg_inf: "(__int_as_float(0xff800000))",
    };
}

impl ReduceAcc for f64 {
    const SPEC: AccSpec = AccSpec {
        cname: "double",
        pos_inf: "(__longlong_as_double(0x7ff0000000000000LL))",
        neg_inf: "(__longlong_as_double(0xfff0000000000000LL))",
    };
}

/// The reduction to perform. One variant per row of the table in the module docs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum UnaryOp {
    Sum,
    Prod,
    Mean,
    L2Norm,
    Min,
    Max,
    All,
    Any,
    CountNonZero,
}

impl UnaryOp {
    /// Kernel symbol suffix and cache-key component.
    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Sum => "sum",
            Self::Prod => "prod",
            Self::Mean => "mean",
            Self::L2Norm => "l2_norm",
            Self::Min => "min",
            Self::Max => "max",
            Self::All => "all",
            Self::Any => "any",
            Self::CountNonZero => "count_nonzero",
        }
    }

    /// Source text for the accumulator's initial value. Must be the operation's identity — see the
    /// module docs.
    ///
    /// `acc` is the accumulator's C name; `pos_inf`/`neg_inf` are only read by `Min`/`Max`, and the
    /// caller supplies them from [`ReduceAcc`] because they are type-dependent. `n_contract` is not
    /// needed here, only in [`Self::post`].
    fn init(self, acc: &str, pos_inf: &str, neg_inf: &str) -> String {
        match self {
            Self::Sum | Self::Mean | Self::L2Norm | Self::Any | Self::CountNonZero => {
                format!("({acc})0")
            },
            Self::Prod | Self::All => format!("({acc})1"),
            Self::Min => pos_inf.to_string(),
            Self::Max => neg_inf.to_string(),
        }
    }

    /// `acc = FOLD(acc, x)`, with `A` the accumulator and `X` the element, typed `@TIN@`.
    fn fold(self) -> &'static str {
        match self {
            Self::Sum | Self::Mean => "((A) + (@TACC@)(X))",
            Self::Prod => "((A) * (@TACC@)(X))",
            Self::Min => "RSTSR_MIN((A), (@TACC@)(X))",
            Self::Max => "RSTSR_MAX((A), (@TACC@)(X))",
            Self::L2Norm => "((A) + (@TACC@)(X) * (@TACC@)(X))",
            // `(int)(X)` on a `bool` is 0/1, so `&` and `|` are the folds and the tree stays integer.
            Self::All => "((A) & (@TACC@)(X))",
            Self::Any => "((A) | (@TACC@)(X))",
            // The cast wraps the *predicate*, not the element: there is no widening to do here.
            Self::CountNonZero => "((A) + (@TACC@)((X) != (@TIN@)0))",
        }
    }

    /// `COMB(a, b)` over two accumulators. Must be the same operation `fold` applies, and must be
    /// associative for the tree to be a valid reordering.
    fn comb(self) -> &'static str {
        match self {
            Self::Sum | Self::Mean | Self::L2Norm => "((A) + (B))",
            Self::Prod => "((A) * (B))",
            Self::Min => "RSTSR_MIN((A), (B))",
            Self::Max => "RSTSR_MAX((A), (B))",
            Self::All => "((A) & (B))",
            Self::Any => "((A) | (B))",
            Self::CountNonZero => "((A) + (B))",
        }
    }

    /// `@TOUT@` value produced from the finished accumulator. Applied once, at the writer, never
    /// inside the tree — otherwise `mean` would divide on every level.
    fn post(self) -> &'static str {
        match self {
            Self::Sum | Self::Prod | Self::Min | Self::Max => "(V)",
            // `n_contract` is the contracted size, which is exactly the divisor the CPU uses
            // (`layout_axes.size()`), and the cast keeps the division in the accumulator's type.
            Self::Mean => "((V) / (@TACC@)n_contract)",
            Self::L2Norm => "sqrt(V)",
            // The tree ran in `int`; the output dtype is `bool`.
            Self::All | Self::Any => "((V) != 0)",
            Self::CountNonZero => "(V)",
        }
    }

    /// Extra `#define`s the fold text depends on.
    fn preamble(self) -> &'static str {
        match self {
            Self::Min | Self::Max => {
                "\
#define RSTSR_MIN(x, y) (((x) != (x)) ? (x) : (((y) != (y)) ? (y) : fmin((x), (y))))
#define RSTSR_MAX(x, y) (((x) != (x)) ? (x) : (((y) != (y)) ? (y) : fmax((x), (y))))
"
            },
            _ => "",
        }
    }
}

/// `@NAME@` is the kernel symbol, so each op shows up separately in a profiler.
///
/// `RSTSR_PREAMBLE` is placed before the macros because `RSTSR_MIN`/`RSTSR_MAX` are used *inside*
/// the `RSTSR_FOLD`/`RSTSR_COMB` replacement text, and macros must be defined before they expand.
const REDUCE_UNARY_ENTRY: &str = r#"
@PREAMBLE@
#define RSTSR_INIT        (@INIT@)
#define RSTSR_FOLD(A, X)  (@FOLD@)
#define RSTSR_COMB(A, B)  (@COMB@)
#define RSTSR_POST(V)     (@POST@)

extern "C" __global__ void @NAME@(
    @TOUT@* __restrict__ c,
    const @TIN@* __restrict__ a,
    rstsr_usz n_out,
    rstsr_usz n_contract,
    rstsr_isz step,      // element step per contracted index
    rstsr_isz out_step)  // element step per output index (unused when n_out == 1)
{
    const rstsr_usz lane = (rstsr_usz)threadIdx.x & (@LANES@ - 1u);
    const rstsr_usz j = (rstsr_usz)blockIdx.x * @OPB@ + ((rstsr_usz)threadIdx.x >> @LOG2_LANES@);

    // Out-of-range threads and lanes past `n_contract` keep the identity and still reach the
    // shuffles below.
    @TACC@ acc = RSTSR_INIT;
    if (j < n_out) {
        const @TIN@* ap = a + (rstsr_isz)j * out_step;
        for (rstsr_usz m = lane; m < n_contract; m += @LANES@) {
            acc = RSTSR_FOLD(acc, ap[(rstsr_isz)m * step]);
        }
    }

    // Stage 1: within a warp. `__syncwarp()` forces reconvergence after the divergent guard, which
    // is what makes `__activemask()` meaningful here.
    __syncwarp();
    const int ws = warpSize;
    for (int off = (@LANES@ <= 32 ? (@LANES@ >> 1) : (ws >> 1)); off > 0; off >>= 1) {
        @TACC@ other = __shfl_down_sync(__activemask(), acc, off);
        acc = (((rstsr_usz)off + lane < @LANES@) ? RSTSR_COMB(acc, other) : acc);
    }

    // Stage 2: only when one output spans several warps. Statically dead for LANES <= 32.
    const int nw = @LANES@ / ws;
    if (@LANES@ > 32 && nw > 1) {
        __shared__ @TACC@ wpart[@MAXW@];
        const int tl = (int)((unsigned)threadIdx.x & (unsigned)(ws - 1));
        const int wid = (int)((unsigned)threadIdx.x / (unsigned)ws);
        if (tl == 0) wpart[wid] = acc;
        __syncthreads();

        if (wid % nw == 0) {
            @TACC@ v = (tl < nw) ? wpart[(wid / nw) * nw + tl] : RSTSR_INIT;
            for (int off = nw >> 1; off > 0; off >>= 1) {
                @TACC@ o = __shfl_down_sync(__activemask(), v, off);
                v = (((int)tl + off < nw) ? RSTSR_COMB(v, o) : v);
            }
            if (tl == 0 && j < n_out) c[j] = (@TOUT@)RSTSR_POST(v);
        }
        return;
    }

    if (lane == 0 && j < n_out) c[j] = (@TOUT@)RSTSR_POST(acc);
}
"#;

/// Fold an element offset into a base device pointer, in bytes.
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

/// Compile-or-fetch the entry. The discriminator carries the op, the three type names and the
/// geometry, because all of them are baked into the source.
fn reduce_unary_function(
    device: &DeviceCuda,
    name: &str,
    disc: &str,
    includes: &[&str],
    entries: impl FnOnce() -> String,
) -> Result<Arc<CudaFunction>> {
    let cache = device.ordinal_obj().planned_cache();
    if let Some(f) = cache.entries.lock().unwrap_or_else(|e| e.into_inner()).get(disc) {
        return Ok(f.clone());
    }
    let src = assemble(includes, &entries());
    let key = crate::kernel::make_key(device.ordinal_obj(), name, &src, &[])?;
    let f = device.ordinal_obj().kernel_function(&key)?;
    cache.entries.lock().unwrap_or_else(|e| e.into_inner()).insert(disc.to_string(), f.clone());
    Ok(f)
}

/// Launch a unary reduction for a plan produced by [`super::reduce_plan::classify_reduce`].
///
/// `c` is the caller's output buffer, already sized to `plan.n_out`; `a` is the operand and `las` its
/// contracted part (only for the base offset — all addressing comes from `plan`).
///
/// The three roles from the module docs appear here as two type parameters (`TIn` loaded, `TOut`
/// stored) and one value (`acc`, the accumulator's C name and infinities). The accumulator is a value
/// rather than a third type parameter because nothing in the argument list would constrain it, so
/// every caller would need an explicit turbofish to avoid `E0283`.
pub(crate) fn reduce_unary_launch<TIn, TOut>(
    device: &DeviceCuda,
    op: UnaryOp,
    acc: AccSpec,
    c: &mut CudaBuffer<MaybeUninit<TOut>>,
    lc: &Layout<IxD>,
    a: &CudaBuffer<TIn>,
    las: &Layout<IxD>,
    plan: &ReducePlan,
) -> Result<()>
where
    TIn: CudaDType,
    TOut: CudaDType,
{
    let ReducePlan {
        lanes,
        block,
        n_contract,
        step_a,
        n_out,
        out_step_a,
        ..
    } = *plan;

    if n_out == 0 || n_contract == 0 {
        return Ok(());
    }

    let grid_blocks = plan.grid_blocks();
    if grid_blocks > u32::MAX as usize {
        rstsr_raise!(UnImplemented, "cuda {}: grid exceeds u32", op.name())?;
    }
    if !lanes.is_power_of_two() || block % lanes != 0 {
        rstsr_raise!(InvalidLayout, "cuda {}: plan has lanes={lanes}, block={block}", op.name())?;
    }

    let exec = device.stream();
    let _ga = crate::kernel::order_input(a, exec)?;

    let name = format!("rstsr_reduce_{}", op.name());
    let disc = format!(
        "{name}|{}|{}|{}|L{lanes}|B{block}",
        TIn::CNAME,
        acc.cname,
        TOut::CNAME
    );
    let init = op.init(acc.cname, acc.pos_inf, acc.neg_inf);

    // A mixed-dtype instantiation may need both types' includes; the same-dtype case is a no-op.
    let mut includes: Vec<&'static str> = TIn::INCLUDES.to_vec();
    for inc in TOut::INCLUDES.iter().copied() {
        if !includes.contains(&inc) {
            includes.push(inc);
        }
    }

    let opb = plan.outputs_per_block.to_string();
    let lanes_s = lanes.to_string();
    let log2_s = lanes.trailing_zeros().to_string();
    let maxw_s = MAX_WARPS_PER_BLOCK.to_string();

    let f = reduce_unary_function(device, &name, &disc, &includes, || {
        render(
            REDUCE_UNARY_ENTRY,
            &[
                // Order matters: an entry that *expands to* text containing another token must come
                // first, or the inner token is never substituted. `FOLD`/`COMB`/`POST` all contain
                // `@TACC@`/`@TIN@` references, and `@PREAMBLE@` supplies macros the body uses.
                ("@PREAMBLE@", op.preamble()),
                ("@NAME@", name.as_str()),
                ("@FOLD@", op.fold()),
                ("@COMB@", op.comb()),
                ("@POST@", op.post()),
                ("@INIT@", init.as_str()),
                ("@TIN@", TIn::CNAME),
                ("@TACC@", acc.cname),
                ("@TOUT@", TOut::CNAME),
                ("@LANES@", lanes_s.as_str()),
                ("@LOG2_LANES@", log2_s.as_str()),
                ("@OPB@", opb.as_str()),
                ("@MAXW@", maxw_s.as_str()),
            ],
        )
    })?;

    // SAFETY: `MaybeUninit<TOut>` and `TOut` are layout-identical; the kernel writes every visited
    // element (exactly once per `j < n_out`) and `c` holds `n_out`.
    let c_typed: &mut CudaBuffer<TOut> =
        unsafe { &mut *(c as *mut CudaBuffer<MaybeUninit<TOut>> as *mut CudaBuffer<TOut>) };

    crate::kernel::FlatLaunch::new(exec, &f, grid_blocks)
        .arg_u64(offset_ptr(exec, c_typed, lc.offset()))
        .arg_u64(offset_ptr(exec, a, las.offset()))
        .arg_usize(n_out)
        .arg_usize(n_contract)
        .arg_bytes(&step_a.to_ne_bytes())
        .arg_bytes(&out_step_a.to_ne_bytes())
        .explicit_shape((grid_blocks as u32, 1, 1), (block, 1, 1))
        .launch()?;

    let _go = crate::kernel::order_output(c, exec)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::reduce_plan::{classify_reduce, MAX_LANES};
    use super::*;

    /// Every row of the table, so a new op cannot be added without the checks below seeing it.
    const ALL_OPS: [UnaryOp; 9] = [
        UnaryOp::Sum,
        UnaryOp::Prod,
        UnaryOp::Mean,
        UnaryOp::L2Norm,
        UnaryOp::Min,
        UnaryOp::Max,
        UnaryOp::All,
        UnaryOp::Any,
        UnaryOp::CountNonZero,
    ];

    fn layout(shape: &[usize], stride: &[isize], offset: usize) -> Layout<IxD> {
        Layout::<IxD>::new(shape.to_vec().into(), stride.to_vec().into(), offset).unwrap()
    }

    /// Every row of the table must be internally consistent, because a mistake here shows up as an
    /// NVRTC error at runtime — on a rented GPU, not in `cargo test`.
    #[test]
    fn op_table_is_well_formed() {
        for op in ALL_OPS {
            let init = op.init("double", "POSINF", "NEGINF");
            assert!(!init.is_empty(), "{:?}: empty init", op);
            // `fold` and `comb` must consume both macro parameters, or the C macro fails to expand.
            assert!(op.fold().contains("(A)"), "{:?}: fold must use the accumulator", op);
            assert!(op.fold().contains("(X)"), "{:?}: fold must use the element", op);
            assert!(op.comb().contains("(A)") && op.comb().contains("(B)"), "{:?}: comb", op);
            assert!(op.post().contains("(V)"), "{:?}: post must use the accumulator", op);
            assert!(!op.name().contains('@'), "{:?}: unresolved token in the name", op);
        }

        // The rows that sum the element itself must cast it to the accumulator; relying on C++
        // promotion would round to the wider type first and narrow on assignment. `count_nonzero`
        // is excluded on purpose: its cast wraps the predicate, not the element.
        for op in [UnaryOp::Sum, UnaryOp::Prod, UnaryOp::Mean, UnaryOp::L2Norm, UnaryOp::Min, UnaryOp::Max] {
            assert!(
                op.fold().contains("(@TACC@)(X)"),
                "{:?}: fold must cast the element to the accumulator",
                op
            );
        }
    }

    /// Every token that appears in the template or in a table row must have a substitution, and the
    /// rows that expand *into* other tokens must be substituted first. Both checks are cheap here and
    /// expensive on a rented GPU: an unsubstituted token reaches NVRTC and fails at launch time, not
    /// at `cargo test` time.
    #[test]
    fn every_token_is_substituted_in_the_right_order() {
        fn tokens(text: &str) -> Vec<&str> {
            let bytes = text.as_bytes();
            let mut out = vec![];
            let mut i = 0;
            while i < bytes.len() {
                if bytes[i] == b'@' {
                    if let Some(j) = text[i + 1..].find('@') {
                        out.push(&text[i..i + j + 2]);
                        i += j + 2;
                        continue;
                    }
                }
                i += 1;
            }
            out
        }

        // The substitution list, in the order the launcher applies it. Kept in one place so the
        // ordering assertion below is about the real order, not a copy of it.
        let order = [
            "@PREAMBLE@", "@NAME@", "@FOLD@", "@COMB@", "@POST@", "@INIT@", "@TIN@", "@TACC@",
            "@TOUT@", "@LANES@", "@LOG2_LANES@", "@OPB@", "@MAXW@",
        ];

        for op in ALL_OPS {
            for text in [op.fold(), op.comb(), op.post(), op.preamble()] {
                for t in tokens(text) {
                    assert!(order.contains(&t), "{op:?}: {t} has no substitution");
                }
            }
        }
        for t in tokens(REDUCE_UNARY_ENTRY) {
            assert!(order.contains(&t), "template uses {t}, which is not substituted");
        }

        // A token whose replacement text contains another token must come first.
        for producer in ["@FOLD@", "@COMB@", "@POST@"] {
            for consumer in ["@TIN@", "@TACC@", "@TOUT@"] {
                let (p, c) = (order.iter().position(|t| *t == producer).unwrap(),
                              order.iter().position(|t| *t == consumer).unwrap());
                assert!(p < c, "{producer} must be substituted before {consumer}");
            }
        }
    }

    /// `Min`/`Max` are the only rows that need the type-dependent infinities, and both must actually
    /// use them — a `0` there would make every negative result wrong.
    #[test]
    fn extremes_start_from_the_infinities() {
        assert_eq!(UnaryOp::Min.init("double", "POSINF", "NEGINF"), "POSINF");
        assert_eq!(UnaryOp::Max.init("double", "POSINF", "NEGINF"), "NEGINF");
        // The others ignore them entirely.
        for op in [UnaryOp::Sum, UnaryOp::Prod, UnaryOp::Mean, UnaryOp::L2Norm] {
            assert!(!op.init("f", "POSINF", "NEGINF").contains("INF"), "{op:?}");
        }
        // Both spellings must be header-free builtins, or NVRTC fails to compile the kernel.
        for inf in [<f64 as ReduceAcc>::SPEC.pos_inf, <f32 as ReduceAcc>::SPEC.neg_inf] {
            assert!(inf.starts_with("(__"), "{inf}: expected a compiler builtin");
        }
    }

    /// The post-step is applied once, at the writer; `mean`'s divisor is the contracted size.
    #[test]
    fn mean_divides_once_by_the_contraction() {
        assert_eq!(UnaryOp::Mean.post(), "((V) / (@TACC@)n_contract)");
        // The identity of `mean` is a sum, not a mean: the divisor is applied after the tree.
        assert_eq!(UnaryOp::Mean.init("double", "P", "N"), "(double)0");
    }

    /// A unary reduction passes the same layout twice; the plan must then carry duplicated `b`
    /// strides, which the launcher ignores by construction.
    #[test]
    fn unary_plan_ignores_the_second_operand() {
        let (n_p, n_i) = (1000usize, 10240usize);
        let la = layout(&[n_p, n_i], &[n_i as isize, 1], 0);
        let lc = layout(&[n_p], &[1], 0);

        let plan = classify_reduce(&la, &la, &lc, &[1], &[1], RowMajor).unwrap();
        assert_eq!(plan.step_a, plan.step_b);
        assert_eq!(plan.out_step_a, plan.out_step_b);
        assert!(plan.lanes.is_power_of_two() && plan.lanes <= MAX_LANES);
        assert_eq!(plan.outputs_per_block * plan.lanes, plan.block);
    }

    /// Full reductions are the case `*_all` needs: rank-0 output, one block, and as many lanes as the
    /// contraction can fill (which is *not* always a whole block — a 60-element reduction gets 64
    /// lanes and leaves the rest of the block idle; that is the known scalar-reduction ceiling).
    #[test]
    fn full_reduction_is_one_block() {
        for (shape, want_lanes) in [
            (&[100usize, 1024][..], 256u32),
            (&[64][..], 64),
            (&[3, 4, 5][..], 64),
        ] {
            let n: usize = shape.iter().product();
            let stride: Vec<isize> = (0..shape.len())
                .map(|k| shape[k + 1..].iter().product::<usize>() as isize)
                .collect();
            let la = layout(shape, &stride, 0);
            let lc = layout(&[], &[], 0);
            let axes: Vec<isize> = (0..shape.len() as isize).collect();

            let plan = classify_reduce(&la, &la, &lc, &axes, &axes, RowMajor).unwrap();
            assert_eq!(plan.n_contract, n);
            assert_eq!(plan.n_out, 1);
            assert_eq!(plan.grid_blocks(), 1);
            assert_eq!(
                plan.lanes, want_lanes,
                "shape {shape:?}: lanes must be clamped by what the contraction can fill"
            );
        }
    }
}
