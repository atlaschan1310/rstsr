//! Elementwise device-op trait impls for `DeviceCuda`: arithmetic,
//! comparison, and unary float families. Element bounds live on these
//! device impls (`CudaDType` sub-traits), not on the tensor tier.

use crate::buffer::CudaBuffer;
use crate::dtype::{CudaArith, CudaCmp, CudaComplexFrom, CudaDType, CudaFloat, CudaReal, CudaTranscend};
use crate::kernels::elementwise;
use crate::prelude_dev::*;

// /* #region binary arithmetic */
#[duplicate_item(
    OpAPI   opchar;
    [OpAddAPI] ['+'];
    [OpSubAPI] ['-'];
    [OpMulAPI] ['*'];
    [OpDivAPI] ['/'];
)]
impl<TA, TB, TC, D> OpAPI<TA, TB, TC, D> for DeviceCuda
where
    Self: DeviceRawAPI<TA, Raw = CudaBuffer<TA>>
        + DeviceRawAPI<TB, Raw = CudaBuffer<TB>>
        + DeviceRawAPI<MaybeUninit<TC>, Raw = CudaBuffer<MaybeUninit<TC>>>,
    TC: CudaDType + CudaArith,
    TA: CudaDType + CudaArith,
    TB: CudaDType + CudaArith,
    D: DimAPI,
{
    fn op_mutc_refa_refb(
        &self,
        c: &mut CudaBuffer<MaybeUninit<TC>>,
        lc: &Layout<D>,
        a: &CudaBuffer<TA>,
        la: &Layout<D>,
        b: &CudaBuffer<TB>,
        lb: &Layout<D>,
    ) -> Result<()> {
        let expr = crate::kernels::exprs::arith_bin(opchar, TC::CNAME, "a[ia]", "b[ib]");
        elementwise::binary_launch::<TC, TA, TB>(self, c, lc, a, la, b, lb, &expr)
    }

    fn op_mutc_refa_numb(
        &self,
        c: &mut CudaBuffer<MaybeUninit<TC>>,
        lc: &Layout<D>,
        a: &CudaBuffer<TA>,
        la: &Layout<D>,
        b: TB,
    ) -> Result<()> {
        let expr = crate::kernels::exprs::arith_bin(opchar, TC::CNAME, "t[it]", "v");
        elementwise::scalar_launch::<TC, TA, TB>(self, c, lc, a, la, &b, &expr)
    }

    fn op_mutc_numa_refb(
        &self,
        c: &mut CudaBuffer<MaybeUninit<TC>>,
        lc: &Layout<D>,
        a: TA,
        b: &CudaBuffer<TB>,
        lb: &Layout<D>,
    ) -> Result<()> {
        let expr = crate::kernels::exprs::arith_bin(opchar, TC::CNAME, "v", "t[it]");
        elementwise::scalar_launch::<TC, TB, TA>(self, c, lc, b, lb, &a, &expr)
    }
}

// /* #endregion */
// /* #region comparisons */
#[duplicate_item(
    OpAPI        opstr;
    [OpEqualAPI        ] ["=="];
    [OpNotEqualAPI     ] ["!="];
)]
impl<TA, TB, D> OpAPI<TA, TB, D> for DeviceCuda
where
    Self: DeviceRawAPI<TA, Raw = CudaBuffer<TA>>
        + DeviceRawAPI<TB, Raw = CudaBuffer<TB>>
        + DeviceRawAPI<MaybeUninit<bool>, Raw = CudaBuffer<MaybeUninit<bool>>>,
    TA: CudaDType + crate::dtype::CudaEqCmp,
    TB: CudaDType + crate::dtype::CudaEqCmp,
    D: DimAPI,
{
    type TOut = bool;

    fn op_mutc_refa_refb(
        &self,
        c: &mut CudaBuffer<MaybeUninit<bool>>,
        lc: &Layout<D>,
        a: &CudaBuffer<TA>,
        la: &Layout<D>,
        b: &CudaBuffer<TB>,
        lb: &Layout<D>,
    ) -> Result<()> {
        elementwise::binary_launch::<bool, TA, TB>(self, c, lc, a, la, b, lb, &format!("(a[ia] {} b[ib])", opstr))
    }

    fn op_mutc_refa_numb(
        &self,
        c: &mut CudaBuffer<MaybeUninit<bool>>,
        lc: &Layout<D>,
        a: &CudaBuffer<TA>,
        la: &Layout<D>,
        b: TB,
    ) -> Result<()> {
        elementwise::scalar_launch::<bool, TA, TB>(self, c, lc, a, la, &b, &format!("(t[it] {} v)", opstr))
    }

    fn op_mutc_numa_refb(
        &self,
        c: &mut CudaBuffer<MaybeUninit<bool>>,
        lc: &Layout<D>,
        a: TA,
        b: &CudaBuffer<TB>,
        lb: &Layout<D>,
    ) -> Result<()> {
        elementwise::scalar_launch::<bool, TB, TA>(self, c, lc, b, lb, &a, &format!("(v {} t[it])", opstr))
    }
}

#[duplicate_item(
    OpAPI        opstr;
    [OpGreaterAPI      ] [">" ];
    [OpGreaterEqualAPI ] [">="];
    [OpLessAPI         ] ["<" ];
    [OpLessEqualAPI    ] ["<="];
)]
impl<TA, TB, D> OpAPI<TA, TB, D> for DeviceCuda
where
    Self: DeviceRawAPI<TA, Raw = CudaBuffer<TA>>
        + DeviceRawAPI<TB, Raw = CudaBuffer<TB>>
        + DeviceRawAPI<MaybeUninit<bool>, Raw = CudaBuffer<MaybeUninit<bool>>>,
    TA: CudaDType + CudaCmp,
    TB: CudaDType + CudaCmp,
    D: DimAPI,
{
    type TOut = bool;

    fn op_mutc_refa_refb(
        &self,
        c: &mut CudaBuffer<MaybeUninit<bool>>,
        lc: &Layout<D>,
        a: &CudaBuffer<TA>,
        la: &Layout<D>,
        b: &CudaBuffer<TB>,
        lb: &Layout<D>,
    ) -> Result<()> {
        elementwise::binary_launch::<bool, TA, TB>(self, c, lc, a, la, b, lb, &format!("(a[ia] {} b[ib])", opstr))
    }

    fn op_mutc_refa_numb(
        &self,
        c: &mut CudaBuffer<MaybeUninit<bool>>,
        lc: &Layout<D>,
        a: &CudaBuffer<TA>,
        la: &Layout<D>,
        b: TB,
    ) -> Result<()> {
        elementwise::scalar_launch::<bool, TA, TB>(self, c, lc, a, la, &b, &format!("(t[it] {} v)", opstr))
    }

    fn op_mutc_numa_refb(
        &self,
        c: &mut CudaBuffer<MaybeUninit<bool>>,
        lc: &Layout<D>,
        a: TA,
        b: &CudaBuffer<TB>,
        lb: &Layout<D>,
    ) -> Result<()> {
        elementwise::scalar_launch::<bool, TB, TA>(self, c, lc, b, lb, &a, &format!("(v {} t[it])", opstr))
    }
}

// /* #endregion */
// /* #region transcendentals: real (libdevice) + complex (own header) */
/// Real transcendentals use CUDA libdevice (`sinf`/`sin`, ...); complex
/// ones use the rstsr_complex.cuh functions (Smith/Hull-Tenan formulas
/// as in CuPy complex.cuh and pytorch ComplexMath.cuh; see the header).
/// Bound is `CudaTranscend` (f32/f64/complex); integer inputs are rejected
/// at compile time (the CPU devices widen them to float — divergence).
#[duplicate_item(
    OpAPI        fn_base  is_recip  cfn        compose;
    [OpSinAPI     ] ["sin"]   [false] ["csin"]   [""];
    [OpCosAPI] ["cos"]         [false] ["ccos"]   [""];
    [OpTanAPI] ["tan"]         [false] ["ctan"]   [""];
    [OpAsinAPI] ["asin"]        [false] ["casin"]  [""];
    [OpAcosAPI] ["acos"]        [false] ["cacos"]  [""];
    [OpAtanAPI] ["atan"]        [false] ["catan"]  [""];
    [OpSinhAPI] ["sinh"]        [false] ["csinh"]  [""];
    [OpCoshAPI] ["cosh"]        [false] ["ccosh"]  [""];
    [OpTanhAPI] ["tanh"]        [false] ["ctanh"]  [""];
    [OpAsinhAPI] ["asinh"]       [false] ["casinh"] [""];
    [OpAcoshAPI] ["acosh"]       [false] ["cacosh"] [""];
    [OpAtanhAPI] ["atanh"]       [false] ["catanh"] [""];
    [OpExpAPI] ["exp"]         [false] ["cexp"]   [""];
    [OpExpm1API   ] ["expm1"]  [false] ["cexp"]   ["sub_one"];
    [OpLogAPI] ["log"]         [false] ["clog"]   [""];
    [OpLog1pAPI   ] ["log1p"]  [false] ["clog"]   ["add_one"];
    [OpLog2API    ] ["log2"]   [false] ["clog"]   ["div_ln2"];
    [OpLog10API   ] ["log10"]  [false] ["clog"]   ["div_ln10"];
    [OpSqrtAPI] ["sqrt"]        [false] ["csqrt"]  [""];
    [OpInvAPI     ] ["inv"]    [true]  ["" ]      [""];
    [OpReciprocalAPI] ["recip"] [true] ["" ]      [""];
)]
impl<T, D> OpAPI<T, D> for DeviceCuda
where
    Self: DeviceRawAPI<T, Raw = CudaBuffer<T>> + DeviceRawAPI<MaybeUninit<T>, Raw = CudaBuffer<MaybeUninit<T>>>,
    T: CudaDType + CudaTranscend,
    D: DimAPI,
{
    type TOut = T;

    fn op_muta_refb(
        &self,
        a: &mut CudaBuffer<MaybeUninit<T>>,
        la: &Layout<D>,
        b: &CudaBuffer<T>,
        lb: &Layout<D>,
    ) -> Result<()> {
        let expr = transc_expr(T::CNAME, fn_base, cfn, compose, is_recip, "b[ib]")?;
        elementwise::unary_launch::<T, T>(self, a, la, b, lb, &expr)
    }

    fn op_muta(&self, a: &mut CudaBuffer<MaybeUninit<T>>, la: &Layout<D>) -> Result<()> {
        let expr = transc_expr(T::CNAME, fn_base, cfn, compose, is_recip, "a[io]")?;
        // SAFETY: layout-identical MaybeUninit transmute (crate invariant)
        let a_typed: &mut CudaBuffer<T> = unsafe { crate::assignment::transmute_mut(a) };
        elementwise::unary_launch_inplace::<T>(self, a_typed, la, &expr)
    }
}

/// Element expression of one transcendental at C type `t`, operand `x`.
/// Real dtypes call libdevice (f-suffixed for float); complex calls the
/// header function, composing expm1/log1p/log2/log10 from cexp/clog
/// (CPU composes them the same way through num-complex).
fn transc_expr(t: &str, fn_base: &str, cfn: &str, compose: &str, is_recip: bool, x: &str) -> Result<String> {
    let s = crate::kernels::exprs::f(t);
    let complex = crate::dtype::is_complex_cname(t);
    // complex call names are template functions: one spelling for both
    // widths (overload resolution picks float/double real-math inside)
    if is_recip {
        return Ok(if complex {
            format!(
                "rstsr_cdiv(rstsr_make_complex(({})1.0, ({})0.0), {x})",
                crate::kernels::exprs::t_real(t),
                crate::kernels::exprs::t_real(t)
            )
        } else {
            format!("(1.0{s} / {x})")
        });
    }
    if compose.is_empty() {
        return Ok(if complex { format!("rstsr_{cfn}({x})") } else { format!("{fn_base}{s}({x})") });
    }
    if !complex {
        // real forms: libdevice provides the stable expm1/log1p and the
        // base-2/base-10 logarithms directly (no exp(x)-1 composition)
        return Ok(match compose {
            "sub_one" => format!("expm1{s}({x})"),
            "add_one" => format!("log1p{s}({x})"),
            "div_ln2" => format!("log2{s}({x})"),
            "div_ln10" => format!("log10{s}({x})"),
            other => rstsr_raise!(UnImplemented, "compose form {other} unknown")?,
        });
    }
    let r = crate::kernels::exprs::t_real(t);
    Ok(match compose {
        // cancellation-free forms mirrored from the CPU's ExtComplexFloat
        "sub_one" => format!("rstsr_cexpm1({x})"),
        "add_one" => format!("rstsr_clog1p({x})"),
        "div_ln2" => {
            let ln2 = format!("rstsr_make_complex(({})0.693147180559945309417232121458, ({})0.0)", r, r);
            format!("rstsr_cdiv(rstsr_clog({x}), {ln2})")
        },
        "div_ln10" => {
            let ln10 = format!("rstsr_make_complex(({})2.302585092994045684017991454684, ({})0.0)", r, r);
            format!("rstsr_cdiv(rstsr_clog({x}), {ln10})")
        },
        other => rstsr_raise!(UnImplemented, "compose form {other} unknown")?,
    })
}

// /* #endregion */
// /* #region promoting binaries (B4) */
/// The promoting-binary traits are `OpAPI<TA, TB, D>` with associated
/// `TOut`: this backend implements the same-dtype form only. The CPU devices
/// promote mixed operands inside the device op (`TOut = TA::Res`); that
/// promotion is a documented divergence and not mirrored here yet. Float-only
/// families bound `CudaFloat`; real families bound `CudaReal`; pow bounds
/// `CudaArith` (complex included). (CPU reference:
/// op_ternary_common.rs promotion tables.)
///
/// `f_suffix` aliases the exprs helper (real-math suffix by C type).
use crate::kernels::exprs::f as f_suffix;

/// Reject a negative scalar exponent for integer `pow` (CPU parity).
fn check_int_pow_exponent<T: CudaDType>(v: &T) -> Result<()> {
    // SAFETY: `CNAME` names the C representation of `T`, so the signed
    // integer views below match T's layout.
    let neg = unsafe {
        match T::CNAME {
            "signed char" => (*(v as *const T as *const i8)) < 0,
            "short" => (*(v as *const T as *const i16)) < 0,
            "int" => (*(v as *const T as *const i32)) < 0,
            "long long" => (*(v as *const T as *const i64)) < 0,
            _ => false,
        }
    };
    rstsr_assert!(!neg, InvalidValue, "integer power requires a non-negative exponent")?;
    Ok(())
}

/// Reject a negative element in an integer tensor exponent (CPU parity:
/// one D2H + sync on this rare path).
fn check_int_pow_exponent_tensor<T: CudaDType, D: DimAPI>(
    device: &DeviceCuda,
    b: &CudaBuffer<T>,
    lb: &Layout<D>,
) -> Result<()> {
    if !crate::dtype::is_signed_int_cname(T::CNAME) {
        return Ok(());
    }
    let order = match device.default_order() {
        RowMajor => TensorIterOrder::C,
        ColMajor => TensorIterOrder::F,
    };
    let offsets: Vec<usize> = rstsr_common::layout::iterator::IterLayout::new(lb, order)?.collect();
    for v in crate::kernels_ops::gather_to_host(device, b, &offsets)? {
        check_int_pow_exponent(&v)?;
    }
    Ok(())
}

macro_rules! impl_promoting_binary {
    ($OpAPI:ident, $bound:ident, $int_exp_check:expr, |$a:ident, $b:ident, $t:ident| $expr:expr) => {
        impl<T, D> $OpAPI<T, T, D> for DeviceCuda
        where
            Self: DeviceRawAPI<T, Raw = CudaBuffer<T>> + DeviceRawAPI<MaybeUninit<T>, Raw = CudaBuffer<MaybeUninit<T>>>,
            T: CudaDType + $bound,
            D: DimAPI,
        {
            type TOut = T;

            fn op_mutc_refa_refb(
                &self,
                c: &mut CudaBuffer<MaybeUninit<T>>,
                lc: &Layout<D>,
                a: &CudaBuffer<T>,
                la: &Layout<D>,
                b: &CudaBuffer<T>,
                lb: &Layout<D>,
            ) -> Result<()> {
                if $int_exp_check {
                    check_int_pow_exponent_tensor(self, b, lb)?;
                }
                let expr = {
                    let ($a, $b, $t) = ("a[ia]", "b[ib]", T::CNAME);
                    $expr.to_string()
                };
                elementwise::binary_launch::<T, T, T>(self, c, lc, a, la, b, lb, &expr)
            }

            fn op_mutc_refa_numb(
                &self,
                c: &mut CudaBuffer<MaybeUninit<T>>,
                lc: &Layout<D>,
                a: &CudaBuffer<T>,
                la: &Layout<D>,
                b: T,
            ) -> Result<()> {
                if $int_exp_check {
                    check_int_pow_exponent(&b)?;
                }
                let expr = {
                    let ($a, $b, $t) = ("t[it]", "v", T::CNAME);
                    $expr.to_string()
                };
                elementwise::scalar_launch::<T, T, T>(self, c, lc, a, la, &b, &expr)
            }

            fn op_mutc_numa_refb(
                &self,
                c: &mut CudaBuffer<MaybeUninit<T>>,
                lc: &Layout<D>,
                a: T,
                b: &CudaBuffer<T>,
                lb: &Layout<D>,
            ) -> Result<()> {
                let expr = {
                    let ($a, $b, $t) = ("v", "t[it]", T::CNAME);
                    $expr.to_string()
                };
                elementwise::scalar_launch::<T, T, T>(self, c, lc, b, lb, &a, &expr)
            }
        }
    };
}

impl_promoting_binary!(OpATan2API, CudaFloat, false, |a, b, t| format!("atan2{}({}, {})", f_suffix(t), a, b));
impl_promoting_binary!(OpCopySignAPI, CudaFloat, false, |a, b, t| format!("copysign{}({}, {})", f_suffix(t), a, b));
impl_promoting_binary!(OpHypotAPI, CudaFloat, false, |a, b, t| format!("hypot{}({}, {})", f_suffix(t), a, b));
impl_promoting_binary!(OpNextAfterAPI, CudaFloat, false, |a, b, t| format!("nextafter{}({}, {})", f_suffix(t), a, b));
// CPU: (a.exp() + b.exp()).ln() — CUDA libdevice has no logaddexp; emit
// the same formula (fmax inside for stability is a numerical upgrade we
// do NOT take: CPU parity rules).
impl_promoting_binary!(OpLogAddExpAPI, CudaFloat, false, |a, b, t| format!(
    "log{s}((exp{s}({a}) + exp{s}({b})))",
    s = f_suffix(t)
));

/// `floor_divide`: float floor(a/b); ints div_floor with zero-guard.
fn floor_divide_expr(tc: &str, x: &str, y: &str) -> String {
    if tc == "float" || tc == "double" {
        let s = f_suffix(tc);
        format!("floor{s}(({x}) / ({y}))")
    } else {
        format!(
            "({y} == 0 ? ({tc})0 : (({tc})(({x}) / ({y}))) - ({tc})((((({x}) % ({y})) != 0) && (((({x}) < 0) != (({y}) < 0)))) ? 1 : 0))"
        )
    }
}

impl_promoting_binary!(OpFloorDivideAPI, CudaReal, false, |a, b, t| floor_divide_expr(t, a, b));
impl_promoting_binary!(OpMaximumAPI, CudaReal, false, |a, b, t| crate::kernels::exprs::maximum(t, a, b));
impl_promoting_binary!(OpMinimumAPI, CudaReal, false, |a, b, t| crate::kernels::exprs::minimum(t, a, b));

// `pow`: float powf / int ipow / complex cpow (exprs::pow dispatches).
// Integer exponents are validated on the host (scalar value, or a one-pass
// D2H scan of a tensor exponent) to match the CPU InvalidValue.
impl_promoting_binary!(OpPowAPI, CudaArith, true, |a, b, t| crate::kernels::exprs::pow(t, t, t, a, b));

// /* #endregion */
// /* #region non-transcendental unaries (B5) */
// Floor/ceil/trunc/round: dtype-preserving (ints are identity on CPU);
// CUDA `floor/ceil/trunc/rint` match exactly. Same-type in/out.
macro_rules! impl_rounding_unary {
    ($OpAPI:ident, $fnbase:expr) => {
        impl<T, D> $OpAPI<T, D> for DeviceCuda
        where
            Self: DeviceRawAPI<T, Raw = CudaBuffer<T>> + DeviceRawAPI<MaybeUninit<T>, Raw = CudaBuffer<MaybeUninit<T>>>,
            T: CudaDType + CudaReal,
            D: DimAPI,
        {
            type TOut = T;

            fn op_muta_refb(
                &self,
                a: &mut CudaBuffer<MaybeUninit<T>>,
                la: &Layout<D>,
                b: &CudaBuffer<T>,
                lb: &Layout<D>,
            ) -> Result<()> {
                // `CudaReal` excludes half; the float suffix is enough
                let expr = if T::CNAME == "float" {
                    format!("{}f(b[ib])", $fnbase)
                } else if T::CNAME == "double" {
                    format!("{}(b[ib])", $fnbase)
                } else {
                    // integer: identity (CPU ext_floor/ceil/trunc/round)
                    "b[ib]".to_string()
                };
                elementwise::unary_launch::<T, T>(self, a, la, b, lb, &expr)
            }

            fn op_muta(&self, a: &mut CudaBuffer<MaybeUninit<T>>, la: &Layout<D>) -> Result<()> {
                let expr = if T::CNAME == "float" {
                    format!("{}f(a[io])", $fnbase)
                } else if T::CNAME == "double" {
                    format!("{}(a[io])", $fnbase)
                } else {
                    "a[io]".to_string()
                };
                // SAFETY: layout-identical MaybeUninit transmute (crate invariant)
                let a_typed: &mut CudaBuffer<T> = unsafe { crate::assignment::transmute_mut(a) };
                elementwise::unary_launch_inplace::<T>(self, a_typed, la, &expr)
            }
        }
    };
}

impl_rounding_unary!(OpCeilAPI, "ceil");
impl_rounding_unary!(OpFloorAPI, "floor");
impl_rounding_unary!(OpTruncAPI, "trunc");

/// `round`: ties-to-even = CUDA `rint` (CPU roundeven parity).
impl<T, D> OpRoundAPI<T, D> for DeviceCuda
where
    Self: DeviceRawAPI<T, Raw = CudaBuffer<T>> + DeviceRawAPI<MaybeUninit<T>, Raw = CudaBuffer<MaybeUninit<T>>>,
    T: CudaDType + CudaReal,
    D: DimAPI,
{
    type TOut = T;

    fn op_muta_refb(
        &self,
        a: &mut CudaBuffer<MaybeUninit<T>>,
        la: &Layout<D>,
        b: &CudaBuffer<T>,
        lb: &Layout<D>,
    ) -> Result<()> {
        let expr = crate::kernels::exprs::round_even(T::CNAME, "b[ib]");
        elementwise::unary_launch::<T, T>(self, a, la, b, lb, &expr)
    }

    fn op_muta(&self, a: &mut CudaBuffer<MaybeUninit<T>>, la: &Layout<D>) -> Result<()> {
        let expr = crate::kernels::exprs::round_even(T::CNAME, "a[io]");
        // SAFETY: layout-identical MaybeUninit transmute (crate invariant)
        let a_typed: &mut CudaBuffer<T> = unsafe { crate::assignment::transmute_mut(a) };
        elementwise::unary_launch_inplace::<T>(self, a_typed, la, &expr)
    }
}

/// `abs`: same-type for reals (unsigned identity), magnitude for
/// complex (TOut = real part type). Complex is split out below.
impl<T, D> OpAbsAPI<T, D> for DeviceCuda
where
    Self: DeviceRawAPI<T, Raw = CudaBuffer<T>>
        + DeviceRawAPI<MaybeUninit<T>, Raw = CudaBuffer<MaybeUninit<T>>>
        + DeviceRawAPI<MaybeUninit<T::AbsOut>, Raw = CudaBuffer<MaybeUninit<T::AbsOut>>>,
    T: CudaDType + crate::dtype::CudaComplexFrom + ExtNum<AbsOut: CudaDType>,
    D: DimAPI,
{
    type TOut = T::AbsOut;

    fn op_muta_refb(
        &self,
        a: &mut CudaBuffer<MaybeUninit<T::AbsOut>>,
        la: &Layout<D>,
        b: &CudaBuffer<T>,
        lb: &Layout<D>,
    ) -> Result<()> {
        let expr = crate::kernels::exprs::abs(T::CNAME, T::AbsOut::CNAME, "b[ib]");
        elementwise::unary_launch::<T::AbsOut, T>(self, a, la, b, lb, &expr)
    }
    fn op_muta(&self, a: &mut CudaBuffer<MaybeUninit<T::AbsOut>>, la: &Layout<D>) -> Result<()> {
        if !<T as ExtNum>::ABS_SAME_TYPE {
            // complex/real-mismatch abs has no in-place form (CPU parity)
            rstsr_raise!(UnImplemented, "abs in-place requires a same-type dtype")?;
        }
        let expr = crate::kernels::exprs::abs(T::CNAME, T::AbsOut::CNAME, "a[io]");
        // SAFETY: T == T::AbsOut here (same-type abs)
        let a_typed: &mut CudaBuffer<T> =
            unsafe { &mut *(a as *mut CudaBuffer<MaybeUninit<T::AbsOut>> as *mut CudaBuffer<T>) };
        elementwise::unary_launch_inplace::<T>(self, a_typed, la, &expr)
    }
}

/// `real`/`imag`: complex -> real part type; real dtypes -> identity
/// (T::AbsOut == T for reals).
impl<T, D> OpRealAPI<T, D> for DeviceCuda
where
    Self: DeviceRawAPI<T, Raw = CudaBuffer<T>>
        + DeviceRawAPI<MaybeUninit<T::AbsOut>, Raw = CudaBuffer<MaybeUninit<T::AbsOut>>>,
    T: CudaDType + ExtNum<AbsOut: CudaDType>,
    D: DimAPI,
{
    type TOut = T::AbsOut;

    fn op_muta_refb(
        &self,
        a: &mut CudaBuffer<MaybeUninit<T::AbsOut>>,
        la: &Layout<D>,
        b: &CudaBuffer<T>,
        lb: &Layout<D>,
    ) -> Result<()> {
        let expr =
            if crate::dtype::is_complex_cname(T::CNAME) { "(b[ib]).re".to_string() } else { "b[ib]".to_string() };
        elementwise::unary_launch::<T::AbsOut, T>(self, a, la, b, lb, &expr)
    }
    fn op_muta(&self, _a: &mut CudaBuffer<MaybeUninit<T::AbsOut>>, _la: &Layout<D>) -> Result<()> {
        if !<T as ExtNum>::ABS_SAME_TYPE {
            rstsr_raise!(UnImplemented, "real in-place requires a same-type dtype")?;
        }
        // real of a real dtype is the identity (no-op in place)
        Ok(())
    }
}

impl<T, D> OpImagAPI<T, D> for DeviceCuda
where
    Self: DeviceRawAPI<T, Raw = CudaBuffer<T>>
        + DeviceRawAPI<MaybeUninit<T::AbsOut>, Raw = CudaBuffer<MaybeUninit<T::AbsOut>>>,
    T: CudaDType + ExtNum<AbsOut: CudaDType>,
    D: DimAPI,
{
    type TOut = T::AbsOut;

    fn op_muta_refb(
        &self,
        a: &mut CudaBuffer<MaybeUninit<T::AbsOut>>,
        la: &Layout<D>,
        b: &CudaBuffer<T>,
        lb: &Layout<D>,
    ) -> Result<()> {
        let expr = if crate::dtype::is_complex_cname(T::CNAME) {
            "(b[ib]).im".to_string()
        } else {
            format!("({})0", T::AbsOut::CNAME)
        };
        elementwise::unary_launch::<T::AbsOut, T>(self, a, la, b, lb, &expr)
    }
    fn op_muta(&self, a: &mut CudaBuffer<MaybeUninit<T::AbsOut>>, la: &Layout<D>) -> Result<()> {
        if !<T as ExtNum>::ABS_SAME_TYPE {
            rstsr_raise!(UnImplemented, "imag in-place requires a same-type dtype")?;
        }
        // imag of a real dtype is 0: fill in place
        let expr = format!("({})0", T::AbsOut::CNAME);
        let a_typed: &mut CudaBuffer<T> =
            unsafe { &mut *(a as *mut CudaBuffer<MaybeUninit<T::AbsOut>> as *mut CudaBuffer<T>) };
        elementwise::unary_launch_inplace::<T>(self, a_typed, la, &expr)
    }
}

/// `conj`: complex -> conjugate; real -> identity (CPU conj on real is
/// identity through ExtNum::ext_conj default).
impl<T, D> OpConjAPI<T, D> for DeviceCuda
where
    Self: DeviceRawAPI<T, Raw = CudaBuffer<T>> + DeviceRawAPI<MaybeUninit<T>, Raw = CudaBuffer<MaybeUninit<T>>>,
    T: CudaDType + CudaComplexFrom,
    D: DimAPI,
{
    type TOut = T;

    fn op_muta_refb(
        &self,
        a: &mut CudaBuffer<MaybeUninit<T>>,
        la: &Layout<D>,
        b: &CudaBuffer<T>,
        lb: &Layout<D>,
    ) -> Result<()> {
        let expr = if crate::dtype::is_complex_cname(T::CNAME) {
            "rstsr_cconj(b[ib])".to_string()
        } else {
            "b[ib]".to_string()
        };
        elementwise::unary_launch::<T, T>(self, a, la, b, lb, &expr)
    }
    fn op_muta(&self, a: &mut CudaBuffer<MaybeUninit<T>>, la: &Layout<D>) -> Result<()> {
        let expr = if crate::dtype::is_complex_cname(T::CNAME) {
            "rstsr_cconj(a[io])".to_string()
        } else {
            "a[io]".to_string()
        };
        // SAFETY: layout-identical MaybeUninit transmute (crate invariant)
        let a_typed: &mut CudaBuffer<T> = unsafe { crate::assignment::transmute_mut(a) };
        elementwise::unary_launch_inplace::<T>(self, a_typed, la, &expr)
    }
}

/// `sign`: -1/0/1 (NaN->NaN floats; z/|z| complex).
impl<T, D> OpSignAPI<T, D> for DeviceCuda
where
    Self: DeviceRawAPI<T, Raw = CudaBuffer<T>> + DeviceRawAPI<MaybeUninit<T>, Raw = CudaBuffer<MaybeUninit<T>>>,
    T: CudaDType + crate::dtype::CudaComplexFrom,
    D: DimAPI,
{
    type TOut = T;

    fn op_muta_refb(
        &self,
        a: &mut CudaBuffer<MaybeUninit<T>>,
        la: &Layout<D>,
        b: &CudaBuffer<T>,
        lb: &Layout<D>,
    ) -> Result<()> {
        let expr = crate::kernels::exprs::sign(T::CNAME, "b[ib]");
        elementwise::unary_launch::<T, T>(self, a, la, b, lb, &expr)
    }
    fn op_muta(&self, a: &mut CudaBuffer<MaybeUninit<T>>, la: &Layout<D>) -> Result<()> {
        let expr = crate::kernels::exprs::sign(T::CNAME, "a[io]");
        // SAFETY: layout-identical MaybeUninit transmute (crate invariant)
        let a_typed: &mut CudaBuffer<T> = unsafe { crate::assignment::transmute_mut(a) };
        elementwise::unary_launch_inplace::<T>(self, a_typed, la, &expr)
    }
}

/// `square`: x*x, any CudaArith dtype (CPU bound Mul<Output=T>).
impl<T, D> OpSquareAPI<T, D> for DeviceCuda
where
    Self: DeviceRawAPI<T, Raw = CudaBuffer<T>> + DeviceRawAPI<MaybeUninit<T>, Raw = CudaBuffer<MaybeUninit<T>>>,
    T: CudaDType + CudaArith,
    D: DimAPI,
{
    type TOut = T;

    fn op_muta_refb(
        &self,
        a: &mut CudaBuffer<MaybeUninit<T>>,
        la: &Layout<D>,
        b: &CudaBuffer<T>,
        lb: &Layout<D>,
    ) -> Result<()> {
        elementwise::unary_launch::<T, T>(self, a, la, b, lb, "(b[ib] * b[ib])")
    }
    fn op_muta(&self, a: &mut CudaBuffer<MaybeUninit<T>>, la: &Layout<D>) -> Result<()> {
        // SAFETY: layout-identical MaybeUninit transmute (crate invariant)
        let a_typed: &mut CudaBuffer<T> = unsafe { crate::assignment::transmute_mut(a) };
        elementwise::unary_launch_inplace::<T>(self, a_typed, la, "(a[io] * a[io])")
    }
}

/// `ext_neg`: two's-complement wrap for unsigned, plain negate else.
impl<T, D> OpExtNegAPI<T, D> for DeviceCuda
where
    Self: DeviceRawAPI<T, Raw = CudaBuffer<T>> + DeviceRawAPI<MaybeUninit<T>, Raw = CudaBuffer<MaybeUninit<T>>>,
    T: CudaDType + CudaArith,
    D: DimAPI,
{
    type TOut = T;

    fn op_muta_refb(
        &self,
        a: &mut CudaBuffer<MaybeUninit<T>>,
        la: &Layout<D>,
        b: &CudaBuffer<T>,
        lb: &Layout<D>,
    ) -> Result<()> {
        elementwise::unary_launch::<T, T>(self, a, la, b, lb, "(-b[ib])")
    }

    fn op_muta(&self, a: &mut CudaBuffer<MaybeUninit<T>>, la: &Layout<D>) -> Result<()> {
        // SAFETY: layout-identical MaybeUninit transmute (crate invariant)
        let a_typed: &mut CudaBuffer<T> = unsafe { crate::assignment::transmute_mut(a) };
        elementwise::unary_launch_inplace::<T>(self, a_typed, la, "(-a[io])")
    }
}

/// `not` (logical_not / bitwise invert): bool -> `!`, ints -> `~`
/// (CPU bound: `!` applies to bool and integers; same-type in/out).
impl<T, D> OpNotAPI<T, T, D> for DeviceCuda
where
    Self: DeviceRawAPI<MaybeUninit<T>, Raw = CudaBuffer<MaybeUninit<T>>> + DeviceRawAPI<T, Raw = CudaBuffer<T>>,
    T: CudaDType + crate::dtype::CudaBitwise,
    D: DimAPI,
{
    fn op_muta_refb(
        &self,
        a: &mut CudaBuffer<MaybeUninit<T>>,
        la: &Layout<D>,
        b: &CudaBuffer<T>,
        lb: &Layout<D>,
    ) -> Result<()> {
        let expr = if T::CNAME == "bool" { "(!b[ib])" } else { "(~b[ib])" };
        elementwise::unary_launch::<T, T>(self, a, la, b, lb, expr)
    }

    fn op_muta(&self, a: &mut CudaBuffer<T>, la: &Layout<D>) -> Result<()> {
        let expr = if T::CNAME == "bool" { "(!a[io])" } else { "(~a[io])" };
        elementwise::unary_launch_inplace::<T>(self, a, la, expr)
    }
}

/// boolean-output queries: is_nan/is_inf/is_finite (floats+complex on
/// CPU); generated per-component for complex (true if any component NaN).
macro_rules! impl_query_unary {
    ($OpAPI:ident, $real:expr, $c32:expr, $c64:expr) => {
        impl<T, D> $OpAPI<T, D> for DeviceCuda
        where
            Self: DeviceRawAPI<T, Raw = CudaBuffer<T>>
                + DeviceRawAPI<MaybeUninit<bool>, Raw = CudaBuffer<MaybeUninit<bool>>>,
            T: CudaDType + crate::dtype::CudaQuery,
            D: DimAPI,
        {
            type TOut = bool;

            fn op_muta_refb(
                &self,
                a: &mut CudaBuffer<MaybeUninit<bool>>,
                la: &Layout<D>,
                b: &CudaBuffer<T>,
                lb: &Layout<D>,
            ) -> Result<()> {
                // NVRTC defines the unsuffixed overloads only; float and
                // half promote implicitly (there is no `isnanf`/`isinff`)
                let expr = match T::CNAME {
                    "float" | "double" => format!("{}(b[ib])", $real),
                    "__half" => format!("{}(__half2float(b[ib]))", $real),
                    "__nv_bfloat16" => format!("{}(__bfloat162float(b[ib]))", $real),
                    "rstsr_c32" => $c32.to_string(),
                    "rstsr_c64" => $c64.to_string(),
                    other => rstsr_raise!(UnImplemented, "query unary dtype {other} not covered")?,
                };
                elementwise::unary_launch::<bool, T>(self, a, la, b, lb, &expr)
            }

            fn op_muta(&self, _a: &mut CudaBuffer<MaybeUninit<bool>>, _la: &Layout<D>) -> Result<()> {
                // bool output has no in-place form (CPU: unreachable!())
                rstsr_raise!(UnImplemented, "query unaries have no in-place form")?
            }
        }
    };
}

impl_query_unary!(OpIsNanAPI, "isnan", "(isnan(b[ib].re) || isnan(b[ib].im))", "(isnan(b[ib].re) || isnan(b[ib].im))");
impl_query_unary!(OpIsInfAPI, "isinf", "(isinf(b[ib].re) || isinf(b[ib].im))", "(isinf(b[ib].re) || isinf(b[ib].im))");
impl_query_unary!(
    OpIsFiniteAPI,
    "isfinite",
    "(isfinite(b[ib].re) && isfinite(b[ib].im))",
    "(isfinite(b[ib].re) && isfinite(b[ib].im))"
);

/// `signbit`: bool out; real dtypes (CPU bound ExtReal).
impl<T, D> OpSignBitAPI<T, D> for DeviceCuda
where
    Self: DeviceRawAPI<T, Raw = CudaBuffer<T>> + DeviceRawAPI<MaybeUninit<bool>, Raw = CudaBuffer<MaybeUninit<bool>>>,
    T: CudaDType + crate::dtype::CudaReal,
    D: DimAPI,
{
    type TOut = bool;

    fn op_muta_refb(
        &self,
        a: &mut CudaBuffer<MaybeUninit<bool>>,
        la: &Layout<D>,
        b: &CudaBuffer<T>,
        lb: &Layout<D>,
    ) -> Result<()> {
        let expr = crate::kernels::exprs::signbit(T::CNAME, "b[ib]");
        elementwise::unary_launch::<bool, T>(self, a, la, b, lb, &expr)
    }
    fn op_muta(&self, _a: &mut CudaBuffer<MaybeUninit<bool>>, _la: &Layout<D>) -> Result<()> {
        // bool output has no in-place form (CPU: unreachable!())
        rstsr_raise!(UnImplemented, "signbit has no in-place form")?
    }
}

/// `positive`: identity copy (CPU: clone per element).
impl<T, D> OpPositiveAPI<T, D> for DeviceCuda
where
    Self: DeviceRawAPI<T, Raw = CudaBuffer<T>> + DeviceRawAPI<MaybeUninit<T>, Raw = CudaBuffer<MaybeUninit<T>>>,
    T: CudaDType,
    D: DimAPI,
{
    fn op_muta_refb(
        &self,
        a: &mut CudaBuffer<MaybeUninit<T>>,
        la: &Layout<D>,
        b: &CudaBuffer<T>,
        lb: &Layout<D>,
    ) -> Result<()> {
        elementwise::unary_launch::<T, T>(self, a, la, b, lb, "b[ib]")
    }
}

use crate::kernels::reduce;
use crate::kernels::reduce_plan::classify_reduce;

impl<T, D> DeviceVecdotAPI<T, T, T, D, D, D> for DeviceCuda
where
    Self: DeviceRawAPI<T, Raw = CudaBuffer<T>>
        + DeviceRawAPI<MaybeUninit<T>, Raw = CudaBuffer<MaybeUninit<T>>>,
    T: CudaDType,
    D: DimAPI,
{
    fn vecdot(
        &self,
        c: &mut CudaBuffer<MaybeUninit<T>>,
        lc: &Layout<D>,
        a: &CudaBuffer<T>,
        la: &Layout<D>,
        b: &CudaBuffer<T>,
        lb: &Layout<D>,
        axes_a: &[isize],
        axes_b: &[isize],
    ) -> Result<()> {
        // The trait takes no `order`: each device supplies its own default. This is the same value
        // the CPU path receives from its caller.
        let order: FlagOrder = self.default_order();

        // Work on dynamic layouts: the split destroys static rank knowledge, and the classifier must
        // see exactly the `Layout<IxD>` the CPU code sees so the port stays faithful.
        let la_d = la.to_dim::<IxD>()?;
        let lb_d = lb.to_dim::<IxD>()?;
        let lc_d = lc.to_dim::<IxD>()?;

        let plan = classify_reduce(&la_d, &lb_d, &lc_d, axes_a, axes_b, order)?;

        // The plan already carries the strides, so the launcher needs the split layouts only for
        // their base offsets. No broadcast-alignment here: the classifier aligns its own copy and
        // the output's offset does not move under it.
        let (las, _) = la_d.dim_split_axes(axes_a)?;
        let (lbs, _) = lb_d.dim_split_axes(axes_b)?;

        reduce::vecdot_launch(self, c, &lc_d, a, &las, b, &lbs, &plan)
    }
}

// /* #endregion */

use crate::kernels::reduce_plan::{ReducePlan};
use crate::kernels::reduce_unary::{reduce_unary_launch, AccSpec, ReduceAcc, UnaryOp};

/// The four things every one of these methods does, in one place.
///
/// The output layout is the only part that mirrors the CPU (`layout_for_array_copy(&lm, K)`); it is
/// part of the observable contract, because a different axis order would give the same numbers with
/// different strides depending on which device ran the op.
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
        + DeviceRawAPI<MaybeUninit<TOut>, Raw = CudaBuffer<MaybeUninit<TOut>>>
        + DeviceCreationAnyAPI<TOut>,
{
    let (las, lm) = la.dim_split_axes(axes)?;
    let lo = layout_for_array_copy(&lm, TensorIterOrder::K)?;
    let n_out = lo.size();

    // A unary reduction has no second operand, so the same layout goes in twice and the plan's `b`
    // strides come out as duplicates that the launcher ignores.
    let plan: ReducePlan = classify_reduce(la, la, &lo, axes, axes, device.default_order())?;

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
        reduce_all_impl(self, UnaryOp::Sum, <T as ReduceAcc>::SPEC, a, &la.to_dim::<IxD>()?, T::zero())
    }

    fn sum_axes(
        &self,
        a: &CudaBuffer<T>,
        la: &Layout<D>,
        axes: &[isize],
    ) -> Result<
        (Storage<DataOwned<<Self as DeviceRawAPI<Self::TOut>>::Raw>, Self::TOut, Self>, Layout<IxD>),
    > {
        reduce_axes_impl(self, UnaryOp::Sum, <T as ReduceAcc>::SPEC, a, &la.to_dim::<IxD>()?, axes, T::zero())
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
        reduce_all_impl(self, UnaryOp::Prod, <T as ReduceAcc>::SPEC, a, &la.to_dim::<IxD>()?, T::one())
    }

    fn prod_axes(
        &self,
        a: &CudaBuffer<T>,
        la: &Layout<D>,
        axes: &[isize],
    ) -> Result<
        (Storage<DataOwned<<Self as DeviceRawAPI<Self::TOut>>::Raw>, Self::TOut, Self>, Layout<IxD>),
    > {
        reduce_axes_impl(self, UnaryOp::Prod, <T as ReduceAcc>::SPEC, a, &la.to_dim::<IxD>()?, axes, T::one())
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
        reduce_all_impl(self, UnaryOp::Mean, <T as ReduceAcc>::SPEC, a, &la.to_dim::<IxD>()?, T::nan())
    }

    fn mean_axes(
        &self,
        a: &CudaBuffer<T>,
        la: &Layout<D>,
        axes: &[isize],
    ) -> Result<
        (Storage<DataOwned<<Self as DeviceRawAPI<Self::TOut>>::Raw>, Self::TOut, Self>, Layout<IxD>),
    > {
        reduce_axes_impl(self, UnaryOp::Mean, <T as ReduceAcc>::SPEC, a, &la.to_dim::<IxD>()?, axes, T::nan())
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
        reduce_all_impl(self, UnaryOp::L2Norm, <T as ReduceAcc>::SPEC, a, &la.to_dim::<IxD>()?, T::zero())
    }

    fn l2_norm_axes(
        &self,
        a: &CudaBuffer<T>,
        la: &Layout<D>,
        axes: &[isize],
    ) -> Result<
        (Storage<DataOwned<<Self as DeviceRawAPI<Self::TOut>>::Raw>, Self::TOut, Self>, Layout<IxD>),
    > {
        reduce_axes_impl(self, UnaryOp::L2Norm, <T as ReduceAcc>::SPEC, a, &la.to_dim::<IxD>()?, axes, T::zero())
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
        reduce_all_impl(self, UnaryOp::Min, <T as ReduceAcc>::SPEC, a, &la.to_dim::<IxD>()?, T::zero())
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
        reduce_axes_impl(self, UnaryOp::Min, <T as ReduceAcc>::SPEC, a, &la.to_dim::<IxD>()?, axes, T::zero())
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
        reduce_all_impl(self, UnaryOp::Max, <T as ReduceAcc>::SPEC, a, &la.to_dim::<IxD>()?, T::zero())
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
        reduce_axes_impl(self, UnaryOp::Max, <T as ReduceAcc>::SPEC, a, &la.to_dim::<IxD>()?, axes, T::zero())
    }
}

// --- explicit-output-dtype variants -------------------------------------------------------------
//
// These are the same kernel with a wider accumulator: `@TACC@` is `TOut`, and the fold casts each
// element into it, which is what `into_cast` does on the CPU. The cast is in the kernel, so the host
// never touches a value — hence no `Add`/`Mul`/`Clone` bound on `T`.

impl<T, TOut, D> OpSumDtypeAPI<T, TOut, D> for DeviceCuda
where
    Self: DeviceRawAPI<T, Raw = CudaBuffer<T>>
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
        reduce_axes_impl(self, UnaryOp::Sum, <TOut as ReduceAcc>::SPEC, a, &la.to_dim::<IxD>()?, axes, TOut::zero())
    }
}

impl<T, TOut, D> OpProdDtypeAPI<T, TOut, D> for DeviceCuda
where
    Self: DeviceRawAPI<T, Raw = CudaBuffer<T>>
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
        reduce_axes_impl(self, UnaryOp::Prod, <TOut as ReduceAcc>::SPEC, a, &la.to_dim::<IxD>()?, axes, TOut::one())
    }
}

impl<T, TOut, D> OpMeanDtypeAPI<T, TOut, D> for DeviceCuda
where
    Self: DeviceRawAPI<T, Raw = CudaBuffer<T>>
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
        reduce_axes_impl(self, UnaryOp::Mean, <TOut as ReduceAcc>::SPEC, a, &la.to_dim::<IxD>()?, axes, TOut::nan())
    }
}
