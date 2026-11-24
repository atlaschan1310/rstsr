//! Kernel-round tests: strided/offset assigns, cast assigns, sequence
//! creation, elementwise ops, stream mixing — all vs CPU reference
//! (copy-and-compare parity shape).

use rstsr::prelude::*;
use rstsr_cuda::funcs;
use rstsr_cuda::prelude_dev::*;
use std::sync::Arc;

/// D2H of a tensor's visible elements by layout gather (house helper).
///
/// **1-D only**: it goes through `to_raw_f`, which asserts `ndim == 1`. Use [`host_nd`] for anything
/// of higher rank and `to_scalar` for a 0-D result.
fn host_vec1<T, B, D, R>(t: &TensorAny<R, T, B, D>) -> Vec<T>
where
    R: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw>,
    T: Clone + Send + Sync + 'static,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceToCpuAPI<T> + DeviceCreationAnyAPI<T> + OpAssignAPI<T, Ix1>,
{
    let raw = t.to_raw_f().rstsr_unwrap();
    DeviceToCpuAPI::<T>::to_cpu_vec::<DataOwned<<B as DeviceRawAPI<T>>::Raw>>(&Storage::new(
        DataOwned::from(raw),
        t.device().clone(),
    ))
    .rstsr_unwrap()
}

/// Layout-ordered host values of a tensor (N-d safe: the D2H goes
/// through the trusted CPU device, so strided views gather correctly).
///
/// Generic in the element type because a reduction is instantiated at every width it supports, and
/// testing only the `f64` instantiation would leave the `f32` template text unexercised. Inference
/// picks `T` from the argument, so the existing `f64` call sites read unchanged.
fn host_nd<'a, T, Tr, B, D>(t: &TensorAny<Tr, T, B, D>) -> Vec<T>
where
    T: Clone + Send + Sync + 'static,
    Tr: DataAPI<Data = <B as DeviceRawAPI<T>>::Raw> + DataIntoCowAPI<'a>,
    D: DimAPI,
    B: DeviceAPI<T> + DeviceToCpuAPI<T> + DeviceCreationAnyAPI<T> + OpAssignArbitaryAPI<T, D, D>,
{
    let contig = t.to_contig(RowMajor);
    DeviceToCpuAPI::<T>::to_cpu_vec(contig.storage()).rstsr_unwrap()
}

macro_rules! require_cuda {
    () => {
        // `cudarc` *panics* when the driver library is absent (nothing to dlopen) instead of
        // reporting zero devices, so `is_available()` is not a probe on a machine without CUDA —
        // it aborts the test. Treat the panic as "no device" so this file skips on a laptop
        // instead of failing every GPU test, and still runs normally where a driver is present.
        let available = std::panic::catch_unwind(funcs::is_available).unwrap_or(false);
        if !available {
            eprintln!("skipping: no CUDA device visible");
            return;
        }
    };
}

/* #region assign */

#[test]
fn assign_strided_view_sources() {
    require_cuda!();
    let d = DeviceCuda::new(0);
    let a: Tensor<f64, _, _> = rt::arange((0., 12., 1., &d)).into_shape([3, 4]);

    // transpose (strided view) -> to_contig: the strided-source regression shape
    let at = a.swapaxes(0, 1);
    let at_contig = at.to_contig(RowMajor);
    let want = host_nd(&at);
    assert_eq!(host_nd(&at_contig), want);

    // row slice of the transpose (partially strided, offset != 0)
    let row_view = at.i(1);
    let row = row_view.to_contig(RowMajor);
    let want_row = host_vec1(&row_view);
    assert_eq!(host_vec1(&row), want_row);

    // assign through a strided source without intermediate to_contig
    let mut c: Tensor<f64, _, _> = rt::zeros(([4, 3], &d));
    c.assign(&at);
    assert_eq!(host_nd(&c), want);
}

#[test]
fn assign_cast_pairs() {
    require_cuda!();
    let d = DeviceCuda::new(0);
    let a: Tensor<f64, _, _> = rt::arange((0., 16., 1., &d));

    // f64 -> f32
    let mut c: Tensor<f32, _, _> = rt::zeros(([16], &d));
    c.assign(&a);
    let want: Vec<f32> = (0..16).map(|v| v as f32).collect();
    assert_eq!(host_vec1(&c), want);

    // f64 -> i32
    let mut c2: Tensor<i32, _, _> = rt::zeros(([16], &d));
    c2.assign(&a);
    let want2: Vec<i32> = (0..16).collect();
    assert_eq!(host_vec1(&c2), want2);

    // bool -> f32
    // (pin the host device: DeviceCpu resolves to DeviceFaer under
    // faer_as_default, whose conversion impls are feature-gated)
    let b_cpu: Tensor<bool, DeviceCpuSerial, _> =
        rt::asarray((vec![true, false, true, true], &DeviceCpuSerial::default()));
    let b = b_cpu.to_device(&d);
    let mut c3: Tensor<f32, _, _> = rt::zeros(([4], &d));
    c3.assign(&b);
    assert_eq!(host_vec1(&c3), vec![1.0f32, 0.0, 1.0, 1.0]);
}

#[test]
fn fill_strided_and_full() {
    require_cuda!();
    let d = DeviceCuda::new(0);
    let c: Tensor<f64, _, _> = rt::full(([2, 3], 2.5f64, &d));
    assert_eq!(host_nd(&c), vec![2.5f64; 6]);
    // strided fill through a mutable slice view (partially strided dst)
    let mut d2: Tensor<f64, _, _> = rt::zeros(([4, 3], &d));
    d2.i_mut(1..3).fill(7.0f64);
    let host = host_nd(&d2);
    assert_eq!(&host[3..9], &vec![7.0f64; 6][..]);
    assert_eq!(&host[0..3], &vec![0.0f64; 3][..]);
}

/* #endregion */

/* #region creation */

#[test]
fn arange_linspace_values() {
    require_cuda!();
    let d = DeviceCuda::new(0);
    let a: Tensor<f64, _, _> = rt::arange((0., 10., 2., &d));
    assert_eq!(host_vec1(&a), vec![0.0f64, 2.0, 4.0, 6.0, 8.0]);
    let l: Tensor<f64, _, _> = rt::linspace((0., 1., 5, &d));
    assert_eq!(host_vec1(&l), vec![0.0f64, 0.25, 0.5, 0.75, 1.0]);
}

/* #endregion */

/* #region elementwise ops */

#[test]
fn binary_arith_vs_cpu() {
    require_cuda!();
    let d = DeviceCuda::new(0);
    let dcpu = DeviceCpuSerial::default();
    let a = rt::arange((0., 32., 1., &d)).into_shape([4, 8]);
    let b = rt::arange((100., 132., 1., &d)).into_shape([4, 8]);
    let a_cpu = a.to_device(&dcpu);
    let b_cpu = b.to_device(&dcpu);

    let s = rt::add(&a, &b);
    assert_eq!(host_nd(&s), host_nd(&rt::add(&a_cpu, &b_cpu)));

    // strided inputs: row-slice views (partially strided, offset != 0)
    let a_sl = a.i(1..3);
    let b_sl = b.i(1..3);
    let st = rt::add(&a_sl, &b_sl);
    let want = rt::add(&a_cpu.i(1..3), &b_cpu.i(1..3));
    assert_eq!(host_nd(&st), host_nd(&want));

    // broadcast input: [4,1] column view + [1,8] row view
    let col = a.i((.., 0..1));
    let row_cpu = b_cpu.i((0..1, ..));
    let row = row_cpu.to_device(&d);
    let sb = rt::add(&col, &row);
    let want_b = rt::add(&a_cpu.i((.., 0..1)), &row_cpu);
    assert_eq!(host_nd(&sb), host_nd(&want_b));
}

#[test]
fn compare_ops_vs_cpu() {
    require_cuda!();
    let d = DeviceCuda::new(0);
    let dcpu = DeviceCpuSerial::default();
    let a = rt::arange((0., 8., 1., &d));
    let b = rt::arange((4., 12., 1., &d));
    let a_cpu = a.to_device(&dcpu);
    let b_cpu = b.to_device(&dcpu);
    let m = rt::lt(&a, &b);
    assert_eq!(host_vec1(&m), host_vec1(&rt::lt(&a_cpu, &b_cpu)));
    let e = rt::eq(&a, &b);
    assert_eq!(host_vec1(&e), host_vec1(&rt::eq(&a_cpu, &b_cpu)));
}

#[test]
fn unary_float_vs_cpu() {
    require_cuda!();
    let d = DeviceCuda::new(0);
    let dcpu = DeviceCpuSerial::default();
    let a = rt::linspace((0.1, 1.0, 16, &d));
    let a_cpu = a.to_device(&dcpu);
    let s = rt::sin(&a);
    let want: Vec<f64> = host_vec1(&rt::sin(&a_cpu));
    let got = host_vec1(&s);
    assert_eq!(got.len(), want.len());
    assert!(got.iter().zip(want.iter()).all(|(x, y)| (x - y).abs() < 1e-12));
    let e = rt::exp(&a);
    let want_e: Vec<f64> = host_vec1(&rt::exp(&a_cpu));
    let got_e = host_vec1(&e);
    assert_eq!(got_e.len(), want_e.len());
    assert!(got_e.iter().zip(want_e.iter()).all(|(x, y)| (x - y).abs() < 1e-12));
}

/// `pow` is generated by `impl_promoting_binary!`, not written as an `impl … for DeviceCuda`, which
/// is why an `impl`-grep once reported it missing (see `reduce_ops.rs`, "Why not the rest"). It has
/// been implemented all along and had no hardware test — this is that test, and nothing else.
///
/// The exponents are integer-valued on purpose: `(-3)^0.5` is NaN and a NaN comparison proves
/// nothing, whereas integer exponents keep every result exact integer arithmetic on both sides.
#[test]
fn pow_vs_cpu() {
    require_cuda!();
    let d = DeviceCuda::new(0);
    let dcpu = DeviceCpuSerial::default();
    let base_cpu: Tensor<f64, DeviceCpuSerial, _> =
        rt::asarray((exact_data(48, 95), &dcpu)).into_shape([6, 8]);
    let exp_cpu: Tensor<f64, DeviceCpuSerial, _> =
        rt::asarray((exact_data(48, 96).into_iter().map(|v| v.abs()).collect::<Vec<f64>>(), &dcpu))
            .into_shape([6, 8]);
    let (base, exp) = (base_cpu.to_device(&d), exp_cpu.to_device(&d));
    assert_close(&host_nd(&rt::pow(&base, &exp)), &host_nd(&rt::pow(&base_cpu, &exp_cpu)), "pow(a, b)");
    // scalar exponent, the other branch of the promoting macro
    assert_close(&host_nd(&rt::pow(&base, 3.0)), &host_nd(&rt::pow(&base_cpu, 3.0)), "pow(a, 3.0)");
}

/* #endregion */

/* #region streams + same_device */

#[test]
fn same_device_relaxed_streams() {
    require_cuda!();
    let d1 = DeviceCuda::new(0);
    let s2 = d1.ordinal_obj().context().new_stream().unwrap();
    let s2obj = Arc::new(CudaStreamObj::wrap(d1.ordinal(), s2));
    let mut d2 = d1.clone();
    d2.set_stream(s2obj);
    // different streams, same ordinal: same device
    assert!(d1.same_device(&d2));
    // a cross-stream op succeeds and orders correctly
    let a = rt::arange((0., 16., 1., &d1));
    let b = rt::arange((0., 16., 1., &d2));
    let s = rt::add(&a, &b);
    let want: Vec<f64> = (0..16).map(|v| (v as f64) * 2.0).collect();
    assert_eq!(host_vec1(&s), want);
}

#[test]
fn mixed_stream_assign() {
    require_cuda!();
    // produce on stream A, assign on stream B (foreign-stream input)
    let d1 = DeviceCuda::new(0);
    let s2 = d1.ordinal_obj().context().new_stream().unwrap();
    let s2obj = Arc::new(CudaStreamObj::wrap(d1.ordinal(), s2));
    let mut d2 = d1.clone();
    d2.set_stream(s2obj);

    let a = rt::arange((0., 8., 1., &d1));
    let mut c: Tensor<f64, _, _> = rt::zeros(([8], &d2));
    c.assign(&a);
    let want: Vec<f64> = (0..8).map(|v| v as f64).collect();
    assert_eq!(host_vec1(&c), want);
}

/* #endregion */

/* #region DeviceFaer interop (feature = "faer") */

#[cfg(feature = "faer")]
#[test]
fn faer_to_cuda_and_back() {
    require_cuda!();
    let dcpu = DeviceFaer::default();
    let d = DeviceCuda::new(0);
    let a_cpu = rt::arange((0., 12., 1., &dcpu)).into_shape([3, 4]);
    let a = a_cpu.to_device(&d);
    let s = rt::add(&a, &a);
    let s_back = s.to_device(&dcpu);
    let want = host_nd(&rt::add(&a_cpu, &a_cpu));
    assert_eq!(host_nd(&s_back), want);

    // into_device (owned transfer)
    let owned = a_cpu.clone().into_device(&d);
    assert_eq!(host_nd(&owned), host_nd(&a_cpu));
}

/* #endregion */

#[test]
fn linspace_endpoint_false_and_complex() {
    require_cuda!();
    let d = DeviceCuda::new(0);
    let dcpu = DeviceCpuSerial::default();

    // endpoint = false (and the n <= 1 boundary: output is just `start`)
    let a: Tensor<f64, _, _> = rt::linspace((0., 1., 5, false, &d));
    let a_cpu: Tensor<f64, _, _> = rt::linspace((0., 1., 5, false, &dcpu));
    assert_eq!(host_vec1(&a), host_vec1(&a_cpu));
    // exact-representable step so the literal is a readability check
    let b: Tensor<f64, _, _> = rt::linspace((0., 1., 4, false, &d));
    assert_eq!(host_vec1(&b), vec![0.0f64, 0.25, 0.5, 0.75]);
    let one: Tensor<f64, _, _> = rt::linspace((3., 9., 1, false, &d));
    assert_eq!(host_vec1(&one), vec![3.0f64]);

    // complex linspace: component-wise kernel (no struct<->scalar casts)
    let z0 = num::Complex::new(0.0f64, 1.0);
    let z1 = num::Complex::new(1.0f64, 2.0);
    let z = rt::linspace((z0, z1, 5, &d));
    let z_cpu: Tensor<num::Complex<f64>, _, _> = rt::linspace((z0, z1, 5, &dcpu));
    let got = host_vec1(&z);
    let want = host_vec1(&z_cpu);
    assert_eq!(got.len(), want.len());
    for (g, w) in got.iter().zip(want.iter()) {
        assert!((g.re - w.re).abs() < 1e-15 && (g.im - w.im).abs() < 1e-15, "{g} vs {w}");
    }
}

#[test]
fn scalar_ops_and_inplace_unary() {
    require_cuda!();
    let d = DeviceCuda::new(0);
    let dcpu = DeviceCpuSerial::default();
    let a = rt::arange((0., 16., 1., &d)).into_shape([4, 4]);
    let a_cpu = rt::arange((0., 16., 1., &dcpu)).into_shape([4, 4]);

    // tensor-scalar arithmetic (scalar is a kernel argument, no staging sync)
    assert_eq!(host_nd(&(&a + 1.5)), host_nd(&(&a_cpu + 1.5)));
    assert_eq!(host_nd(&(2.0 - &a)), host_nd(&(2.0 - &a_cpu)));
    assert_eq!(host_nd(&(&a * 3.0)), host_nd(&(&a_cpu * 3.0)));

    // tensor-scalar comparisons, both orientations (1-D for to_vec)
    let v = rt::arange((0., 16., 1., &d));
    let v_cpu = rt::arange((0., 16., 1., &dcpu));
    assert_eq!(host_vec1(&rt::lt(&v, 7.0)), host_vec1(&rt::lt(&v_cpu, 7.0)));
    assert_eq!(host_vec1(&rt::gt(7.0, &v)), host_vec1(&rt::gt(7.0, &v_cpu)));
    assert_eq!(host_vec1(&rt::eq(&v, 3.0)), host_vec1(&rt::eq(&v_cpu, 3.0)));

    // in-place unary on an owned tensor
    // `f64` spelled out: `host_nd` is generic over the element type, so the literal no longer pins it.
    let b = rt::arange((0.1f64, 16.1, 1., &d)).into_shape([4, 4]);
    let b_cpu = rt::arange((0.1f64, 16.1, 1., &dcpu)).into_shape([4, 4]);
    let s = rt::sin(b);
    let got = host_nd(&s);
    let want = host_nd(&rt::sin(&b_cpu));
    assert_eq!(got.len(), want.len());
    assert!(got.iter().zip(want.iter()).all(|(x, y)| (x - y).abs() < 1e-12));
}

#[test]
fn elementwise_negative_stride() {
    require_cuda!();
    let d = DeviceCuda::new(0);
    let dcpu = DeviceCpuSerial::default();
    let v = rt::arange((0., 32., 1., &d));
    let v_cpu = rt::arange((0., 32., 1., &dcpu));
    let r = SliceI::new(None::<isize>, None::<isize>, Some(-2isize));
    let a = v.i(r);
    let a_cpu = v_cpu.i(r);
    assert_eq!(a.layout().stride()[0], -2);
    assert_eq!(host_vec1(&rt::add(&a, &a)), host_vec1(&rt::add(&a_cpu, &a_cpu)));
    let s_cuda: Vec<f64> = host_vec1(&rt::sin(&a));
    let s_cpu: Vec<f64> = host_vec1(&rt::sin(&a_cpu));
    assert_eq!(s_cuda.len(), s_cpu.len());
    assert!(s_cuda.iter().zip(s_cpu.iter()).all(|(x, y)| (x - y).abs() < 1e-12));
}

#[test]
fn cross_stream_ordering_loop() {
    require_cuda!();
    // produce on stream A, consume on stream B, every iteration; event
    // ordering must make each read see the produced values
    let d1 = DeviceCuda::new(0);
    let s2 = d1.ordinal_obj().context().new_stream().unwrap();
    let s2obj = Arc::new(CudaStreamObj::wrap(d1.ordinal(), s2));
    let mut d2 = d1.clone();
    d2.set_stream(s2obj);
    for _ in 0..50 {
        let a = rt::arange((0., 4096., 1., &d1));
        let mut c: Tensor<f64, _, _> = rt::zeros(([4096], &d2));
        c.assign(&a);
        let r = rt::add(&c, &c);
        let v = host_vec1(&r);
        assert_eq!(v.len(), 4096);
        assert_eq!(v[0], 0.0);
        assert_eq!(v[1], 2.0);
        assert_eq!(v[4095], 8190.0);
    }
}

/* #region assign order/shape regressions */

#[test]
fn assign_reshape_of_strided_view() {
    require_cuda!();
    // rank/shape-changing assign: each side decomposes by its own shape, so
    // the result must match the CPU devices' positional zip
    let d = DeviceCuda::new(0);
    let dcpu = DeviceCpuSerial::default();

    let base = rt::arange((0., 12., 1., &d)).into_shape([3, 4]);
    let a = base.swapaxes(0, 1);
    let got = host_nd(&a.reshape([2, 6]).into_owned());

    let base_cpu = rt::arange((0., 12., 1., &dcpu)).into_shape([3, 4]);
    let a_cpu = base_cpu.swapaxes(0, 1);
    let want = host_nd(&a_cpu.reshape([2, 6]).into_owned());

    assert_eq!(got, want);
}

#[test]
fn assign_reshape_flatten_of_strided_view() {
    require_cuda!();
    // flattening a non-contiguous view: same-rank source/dest must not take
    // the linear-copy fast path
    let d = DeviceCuda::new(0);
    let dcpu = DeviceCpuSerial::default();

    let base = rt::arange((0., 12., 1., &d)).into_shape([3, 4]);
    let a = base.swapaxes(0, 1);
    let got = host_nd(&a.reshape([12]).into_owned());

    let base_cpu = rt::arange((0., 12., 1., &dcpu)).into_shape([3, 4]);
    let a_cpu = base_cpu.swapaxes(0, 1);
    let want = host_nd(&a_cpu.reshape([12]).into_owned());

    assert_eq!(got, want);
}

#[test]
fn assign_shape_change_on_col_major_device() {
    require_cuda!();
    // the same shape-changing assign under ColMajor must follow the
    // column-major positional zip
    let mut d = DeviceCuda::new(0);
    d.set_default_order(ColMajor);
    let mut dcpu = DeviceCpuSerial::default();
    dcpu.set_default_order(ColMajor);

    let base = rt::arange((0., 12., 1., &d)).into_shape([3, 4]);
    let a = base.swapaxes(0, 1);
    let got = host_nd(&a.reshape([2, 6]).into_owned());

    let base_cpu = rt::arange((0., 12., 1., &dcpu)).into_shape([3, 4]);
    let a_cpu = base_cpu.swapaxes(0, 1);
    let want = host_nd(&a_cpu.reshape([2, 6]).into_owned());

    assert_eq!(got, want);
}

#[test]
fn assign_same_size_cast_pairs() {
    require_cuda!();
    // equal-size dtype pairs must cast element-wise, not byte-copy
    let d = DeviceCuda::new(0);
    let dcpu = DeviceCpuSerial::default();

    let a: Tensor<i64, _, _> = rt::arange((0i64, 8i64, 1i64, &d));
    let mut c: Tensor<f64, _, _> = rt::zeros(([8], &d));
    c.assign(&a);
    assert_eq!(host_vec1(&c), (0..8).map(|i| i as f64).collect::<Vec<_>>());

    let b: Tensor<f32, _, _> = rt::arange((0f32, 8f32, 1f32, &d));
    let mut ci: Tensor<i32, _, _> = rt::zeros(([8], &d));
    ci.assign(&b);
    assert_eq!(host_vec1(&ci), (0..8).collect::<Vec<i32>>());

    // strided source (same shape) goes through the cast kernel
    let base = rt::arange((0i64, 12i64, 1i64, &d)).into_shape([3, 4]);
    let at = base.swapaxes(0, 1);
    let mut cs: Tensor<f64, _, _> = rt::zeros(([4, 3], &d));
    cs.assign(&at);

    let base_cpu = rt::arange((0i64, 12i64, 1i64, &dcpu)).into_shape([3, 4]);
    let at_cpu = base_cpu.swapaxes(0, 1);
    let mut cs_cpu: Tensor<f64, _, _> = rt::zeros(([4, 3], &dcpu));
    cs_cpu.assign(&at_cpu);
    let view_cpu = cs_cpu.view();
    assert_eq!(host_nd(&cs), host_nd(&view_cpu));
}

/* #endregion */

/* #region stream write-target ordering */

#[test]
fn inplace_after_stream_retag() {
    require_cuda!();
    // a re-tagged buffer lives on its allocation stream while ops run on the
    // device's execution stream; host reads must still see the op's result
    let d1 = DeviceCuda::new(0);
    let s2 = d1.ordinal_obj().context().new_stream().unwrap();
    let mut d2 = d1.clone();
    d2.set_stream(Arc::new(CudaStreamObj::wrap(d1.ordinal(), s2)));

    for _ in 0..40 {
        let a = rt::arange((0., 4096., 1., &d1));
        let owned = a.into_device(&d2);
        let s = rt::sin(owned);
        let v = host_vec1(&s);
        assert_eq!(v.len(), 4096);
        for (i, x) in v.iter().enumerate() {
            assert!((*x - (i as f64).sin()).abs() < 1e-12);
        }
    }
}

/* #endregion */

/* #region vecdot */

/// Deterministic, deliberately non-representable f64 data.
fn probe_data(n: usize, seed: u64) -> Vec<f64> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u = ((s >> 11) as f64) / ((1u64 << 53) as f64); // [0, 1)
            (u - 0.5) * 8.0 // [-4, 4)
        })
        .collect()
}

/// Small integers as f64: every product is exact, every sum below 2^53 is exact, so the result is
/// independent of summation order and may be compared with `assert_eq!`.
fn exact_data(n: usize, seed: u64) -> Vec<f64> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 33) % 9) as f64 - 4.0 // integers in [-4, 4]
        })
        .collect()
}

/// `err <= atol + rtol * |want|`, reported with the offending index.
///
/// `atol` is not slack: for a length-1024 dot product of O(1) terms two legitimate summation orders
/// differ by ~1e-12 absolute when the result cancels, and a purely relative criterion would fail on
/// the near-zero outputs.
fn assert_close(got: &[f64], want: &[f64], ctx: &str) {
    const ATOL: f64 = 1e-9;
    const RTOL: f64 = 1e-12;
    assert_eq!(got.len(), want.len(), "{ctx}: length mismatch");
    let mut worst = 0.0f64;
    let mut at = 0usize;
    for i in 0..got.len() {
        let err = (got[i] - want[i]).abs();
        let budget = ATOL + RTOL * want[i].abs();
        if err / budget > worst {
            worst = err / budget;
            at = i;
        }
    }
    assert!(
        worst <= 1.0,
        "{ctx}: |got - want| = {:e} at index {at} (got {:?}, want {:?}) exceeds {ATOL:e} + \
         {RTOL:e}*|want|. A scattered pattern of small errors means the addressing is right and only \
         the summation order differs; a large or structured one means the data is wrong.",
        (got[at] - want[at]).abs(),
        got[at],
        want[at],
    );
}

/// One shape per lane count the classifier can emit: `(rows, cols, contracted axis)`.
///
/// C-order, so contracting the last axis makes the contraction unit-stride and contracting the first
/// axis makes the output unit-stride. Row 5 is the interesting one for addressing: the contraction is
/// *strided* and lanes > 1, which the old `ContigS` kernel never had to handle.
///
/// The last three rows look transposed because they are: to reach 64/128/256 lanes the *output* has
/// to be small (fewer than ~4000 elements) while the contraction is at least as long as the lane
/// count, so the long dimension has to be the contracted one.
const VECDOT_CASES: &[(usize, usize, isize)] = &[
    (2, 131_072, 0),  // lanes 1   — thread per output, no tree at all
    (8, 100_000, 0),  // lanes 2   — 16 outputs per warp
    (20_000, 3, 1),   // lanes 4   — 8 outputs per warp
    (20_000, 8, 1),   // lanes 8   — 4 outputs per warp
    (100, 10_024, 0), // lanes 16  — strided contraction, step_a = 10024
    (20_000, 32, 1),  // lanes 32  — exactly one warp per output
    (3_000, 64, 1),   // lanes 64  — 2 warps per output
    (1_500, 128, 1),  // lanes 128 — 4 warps per output
    (500, 1_024, 1),  // lanes 256 — 8 warps per output, whole block
];

/// One case, built on the CPU device, copied to the GPU, and compared back.
fn vecdot_case(rows: usize, cols: usize, axis: isize, exact: bool) {
    let d = DeviceCuda::new(0);
    // Pin the host device: `DeviceCpu` resolves to `DeviceFaer` under `faer_as_default`, whose impls
    // are feature-gated.
    let dcpu = DeviceCpuSerial::default();
    let gen = if exact { exact_data } else { probe_data };

    let seed = (rows as u64) * 1_000_003 + cols as u64;
    let a_cpu: Tensor<f64, DeviceCpuSerial, _> =
        rt::asarray((gen(rows * cols, seed), &dcpu)).into_shape([rows, cols]);
    let b_cpu: Tensor<f64, DeviceCpuSerial, _> =
        rt::asarray((gen(rows * cols, seed ^ 0x9e37_79b9), &dcpu)).into_shape([rows, cols]);

    let want: Vec<f64> = host_vec1(&rt::vecdot(&a_cpu, &b_cpu, axis));

    let a = a_cpu.to_device(&d);
    let b = b_cpu.to_device(&d);
    let got: Vec<f64> = host_vec1(&rt::vecdot(&a, &b, axis));

    let ctx = format!("vecdot ({rows} x {cols}, axis {axis}, exact={exact})");
    assert_eq!(got.len(), want.len(), "{ctx}: output length");
    if exact {
        assert_eq!(got, want, "{ctx}: integer-valued data must agree bitwise in any summation order");
    } else {
        assert_close(&got, &want, &ctx);
    }
}

/// Every lane count, on data where the summation order cannot matter.
///
/// Named for what it asserts, not for the fixture it uses: `exact_data` is the generator
/// below and a test of the same name would shadow it.
#[test]
fn vecdot_all_lane_counts_exact() {
    require_cuda!();
    for &(rows, cols, axis) in VECDOT_CASES {
        vecdot_case(rows, cols, axis, true);
    }
}

/// Every lane count, on data where only the tolerance applies.
#[test]
fn vecdot_all_lane_counts_vs_cpu() {
    require_cuda!();
    for &(rows, cols, axis) in VECDOT_CASES {
        vecdot_case(rows, cols, axis, false);
    }
}

/// Non-zero `layout.offset()` on the cooperative path.
///
/// The launcher folds the split layouts' offset into the base pointer and the kernel then walks
/// `ia = j * out_step + m * step`. A double-counted or dropped offset shows up here and nowhere else,
/// and it shows up as wrong numbers rather than as an error — hence exact data, so the comparison is
/// bitwise.
#[test]
fn vecdot_offset_view() {
    require_cuda!();
    let d = DeviceCuda::new(0);
    let dcpu = DeviceCpuSerial::default();
    let (rows, cols) = (100usize, 10_024usize);

    let a_cpu: Tensor<f64, DeviceCpuSerial, _> =
        rt::asarray((exact_data(rows * cols, 5), &dcpu)).into_shape([rows, cols]);
    let b_cpu: Tensor<f64, DeviceCpuSerial, _> =
        rt::asarray((exact_data(rows * cols, 6), &dcpu)).into_shape([rows, cols]);

    let want: Vec<f64> = host_vec1(&rt::vecdot(a_cpu.i(3..), b_cpu.i(3..), 0));

    let a = a_cpu.to_device(&d);
    let b = b_cpu.to_device(&d);
    let got: Vec<f64> = host_vec1(&rt::vecdot(a.i(3..), b.i(3..), 0));

    assert_eq!(got, want, "offset view (97 x 10024, axis 0): exact data must agree bitwise");
}

/// The shapes the backend refuses, each by name. All are reachable through the public API, and all
/// must fail rather than fall back to the host or silently pick a different kernel.
#[test]
fn vecdot_refuses_unplannable_shapes() {
    require_cuda!();
    let d = DeviceCuda::new(0);
    let dcpu = DeviceCpuSerial::default();

    // Stepped on both axes: neither the contracted nor the output part has a unit-stride run, so
    // there is nothing to coalesce on. Packing is the eventual answer; until then, refuse.
    let r = SliceI::new(None::<isize>, None::<isize>, Some(2isize));
    let a2: Tensor<f64, _, _> = rt::arange((0., 512. * 512., 1., &d)).into_shape([512, 512]);
    let b2: Tensor<f64, _, _> =
        rt::arange((1., (512. * 512.) + 1., 1., &d)).into_shape([512, 512]);
    match rt::vecdot_f(a2.i((r, r)), b2.i((r, r)), 0) {
        Ok(_) => panic!("stepped-on-both-axes should be refused, but it computed a result"),
        Err(e) => {
            let msg = format!("{e}");
            assert!(msg.contains("no unit-stride run"), "expected the layout reason, got: {msg}");
        },
    }

    // One `m` serves both operands, so a pair whose contracted blocks are enumerated in different
    // axis orders would multiply elements that do not correspond — a wrong answer that looks like a
    // number. The plan refuses; the fix is `to_contig` on one operand.
    let (p, i, j) = (2usize, 2usize, 3usize);
    let ac_cpu: Tensor<f64, DeviceCpuSerial, _> =
        rt::asarray((exact_data(p * i * j, 17), &dcpu)).into_shape([p, i, j]);
    let bj_cpu: Tensor<f64, DeviceCpuSerial, _> =
        rt::asarray((exact_data(p * j * i, 18), &dcpu)).into_shape([p, j, i]);
    let bt_cpu = bj_cpu.swapaxes(1, 2); // [p, i, j] with its (i, j) block F-ordered
    assert_eq!(bt_cpu.layout().stride()[1], 1, "fixture: b's contracted block must be F-ordered");
    let ac = ac_cpu.to_device(&d);
    let bt = bt_cpu.to_device(&d);

    let axes = || (AxesIndex::Vec(vec![1isize, 2]), AxesIndex::Vec(vec![1isize, 2]));
    match rt::vecdot_f(&ac, &bt, axes()) {
        Ok(_) => panic!("operands that enumerate the contraction differently should be refused"),
        Err(e) => {
            let msg = format!("{e}");
            assert!(msg.contains("different memory orders"), "expected the enumeration reason, got: {msg}");
        },
    }
    // ... while two operands that agree still work, F-ordered or not: the check is about agreement,
    // not about row-major order.
    let same: Vec<f64> = host_vec1(&rt::vecdot(&bt, &bt, axes()));
    let same_cpu: Vec<f64> = host_vec1(&rt::vecdot(&bt_cpu, &bt_cpu, axes()));
    assert_eq!(same, same_cpu, "(bt, bt): two equally-ordered blocks must agree with the CPU");

    // The second case this test used to assert — a rank-3 contraction leaving a two-axis output —
    // is no longer refused. `output_step` accepts any row-major run in shape order, not just a
    // single axis, so it is now planned and computed; `vecdot_multi_axis_output_vs_cpu` checks the
    // answer instead. The refusals that remain reachable through `rt::vecdot_f` are the stepped one,
    // a gapped contracted block and this enumeration mismatch, and `reduce_plan::tests` pins all
    // three on the host, so a classifier change fails in `cargo test --lib` rather than only on a
    // machine with a GPU.
}

/// Contracting the last axis of a rank-3 tensor leaves a **two-axis** output part.
///
/// This was refused by the first classifier, which required the output side to be a single axis, and
/// that refusal was asserted in `vecdot_refuses_unplannable_shapes` until it quietly stopped being
/// true. It is now supported: `[4, 8]` with strides `[64, 8]` is one arithmetic progression with
/// step 8, so the kernel's `j * out_step` covers it. `reduce_plan::tests::multi_axis_output_is_planned`
/// pins the same shape on the host.
#[test]
fn vecdot_multi_axis_output_vs_cpu() {
    require_cuda!();
    let d = DeviceCuda::new(0);
    let dcpu = DeviceCpuSerial::default();

    // Exact data, so any summation order agrees and the comparison stays an assert_eq!.
    let a_cpu: Tensor<f64, DeviceCpuSerial, _> =
        rt::asarray((exact_data(4 * 8 * 8, 21), &dcpu)).into_shape([4, 8, 8]);
    let b_cpu: Tensor<f64, DeviceCpuSerial, _> =
        rt::asarray((exact_data(4 * 8 * 8, 22), &dcpu)).into_shape([4, 8, 8]);
    let a = a_cpu.to_device(&d);
    let b = b_cpu.to_device(&d);

    // Output is rank 2 here, so `host_nd` rather than `host_vec1`.
    let got = host_nd(&rt::vecdot(&a, &b, 2));
    let want = host_nd(&rt::vecdot(&a_cpu, &b_cpu, 2));
    assert_eq!(got.len(), 32, "(4, 8, 8) contracted on axis 2 leaves 4 * 8 outputs");
    assert_eq!(got, want, "rank-3 input with a two-axis output part");
}


/* #endregion */

/* #region unary reductions */

/// The shapes the reduce kernels are exercised on, `(rows, cols, contracted axis)`.
///
/// Deliberately the same nine as `VECDOT_CASES`: the plan comes from the same classifier, so these
/// are the shapes that compile the 1/2/4/8/16/32/64/128/256-lane variants. Only `sum` walks all of
/// them — the tree and the addressing are shared across ops, so a per-op fold is checked on small
/// tensors instead of nine shapes each.
const REDUCE_CASES: &[(usize, usize, isize)] = &[
    (2, 131_072, 0),  // lanes 1   — thread per output, no tree at all
    (8, 100_000, 0),  // lanes 2   — 16 outputs per warp
    (20_000, 3, 1),   // lanes 4   — 8 outputs per warp
    (20_000, 8, 1),   // lanes 8   — 4 outputs per warp
    (100, 10_024, 0), // lanes 16  — strided contraction, step_a = 10024
    (20_000, 32, 1),  // lanes 32  — exactly one warp per output
    (3_000, 64, 1),   // lanes 64  — 2 warps per output, shared-memory stage
    (1_500, 128, 1),  // lanes 128 — 4 warps per output
    (500, 1_024, 1),  // lanes 256 — whole block on one output
];

/// One `(rows, cols)` pair of tensors on both devices, integer-valued so summation order cannot
/// matter, plus the probe-data twin for the ops where it can.
fn reduce_pair(rows: usize, cols: usize, seed: u64) -> (Tensor<f64, DeviceCuda, IxD>, Tensor<f64, DeviceCpuSerial, IxD>) {
    let d = DeviceCuda::new(0);
    let dcpu = DeviceCpuSerial::default();
    let cpu: Tensor<f64, DeviceCpuSerial, _> =
        rt::asarray((exact_data(rows * cols, seed), &dcpu)).into_shape([rows, cols]);
    let gpu = cpu.to_device(&d);
    (gpu, cpu)
}

/// `sum` over the contracted axis, one shape per lane count.
///
/// Every failure mode of the tree shows up here: a wrong `lane` mask sends another output's partial
/// sum into this one (values, not just a crash), a missing shared-memory stage truncates the result
/// for lanes > 32, and a wrong `step` reads the wrong element. Exact data makes the comparison an
/// `assert_eq!` rather than a tolerance, so a single flipped bit fails the test.
#[test]
fn reduce_sum_over_all_lane_counts() {
    require_cuda!();
    for &(rows, cols, axis) in REDUCE_CASES {
        let (gpu, cpu) = reduce_pair(rows, cols, rows as u64 * 1_000_003 + cols as u64);
        let got: Vec<f64> = host_vec1(&gpu.sum_axes(axis));
        let want: Vec<f64> = host_vec1(&cpu.sum_axes(axis));
        assert_eq!(got, want, "sum over axis {axis} of ({rows} x {cols})");
    }
}

/// The folds that are not a sum: `min`, `max` (order-blind) and `mean`, `l2_norm` (order-dependent).
#[test]
fn reduce_families_vs_cpu() {
    require_cuda!();
    for &(rows, cols, axis) in &[(7usize, 5usize, 0isize), (7, 5, 1), (3, 64, 1), (1_000, 33, 0)] {
        let (gpu, cpu) = reduce_pair(rows, cols, rows as u64 * 31 + cols as u64);
        let ctx = format!("({rows} x {cols}, axis {axis})");

        // Integer-valued input: min/max/sum agree bitwise whatever the visiting order.
        assert_eq!(host_vec1(&gpu.min_axes(axis)), host_vec1(&cpu.min_axes(axis)), "min {ctx}");
        assert_eq!(host_vec1(&gpu.max_axes(axis)), host_vec1(&cpu.max_axes(axis)), "max {ctx}");
        assert_eq!(host_vec1(&gpu.sum_axes(axis)), host_vec1(&cpu.sum_axes(axis)), "sum {ctx}");

        // Division and the square root do not survive reordering bit-for-bit.
        assert_close(&host_vec1(&gpu.mean_axes(axis)), &host_vec1(&cpu.mean_axes(axis)), &format!("mean {ctx}"));
        assert_close(&host_vec1(&gpu.l2_norm_axes(axis)), &host_vec1(&cpu.l2_norm_axes(axis)), &format!("l2 {ctx}"));

        // The whole-array forms, which take the `to_cpu_vec` path in the device impl.
        assert_eq!(gpu.min(), cpu.min(), "min_all {ctx}");
        assert_eq!(gpu.max(), cpu.max(), "max_all {ctx}");
        assert_eq!(gpu.sum(), cpu.sum(), "sum_all {ctx}");
        assert!((gpu.mean() - cpu.mean()).abs() < 1e-12, "mean_all {ctx}");
        assert!((gpu.l2_norm() - cpu.l2_norm()).abs() < 1e-12, "l2_norm_all {ctx}");
    }
}

/// `prod` is its own case: a long contraction of values in [-4, 4] overflows and stops being exact.
///
/// Values are restricted to {-2, -1, 0, 1, 2} over eight elements, so every partial product is
/// representable and the comparison stays bitwise.
#[test]
fn reduce_prod_vs_cpu() {
    require_cuda!();
    let d = DeviceCuda::new(0);
    let dcpu = DeviceCpuSerial::default();
    let vals: Vec<f64> = (0..48).map(|k| [-2.0, -1.0, 0.0, 1.0, 2.0][(k * 7 + k / 5) % 5]).collect();

    let cpu: Tensor<f64, DeviceCpuSerial, _> = rt::asarray((vals, &dcpu)).into_shape([6, 8]);
    let gpu = cpu.to_device(&d);

    assert_eq!(host_vec1(&gpu.prod_axes(1)), host_vec1(&cpu.prod_axes(1)), "prod over axis 1");
    assert_eq!(host_vec1(&gpu.prod_axes(0)), host_vec1(&cpu.prod_axes(0)), "prod over axis 0");
    assert_eq!(gpu.prod(), cpu.prod(), "prod_all");
}

/// Three shapes `_axes` does not cover: several contracted axes, `keepdims`, and `axes = None`.
///
/// The last one is the interesting one structurally — it leaves the device through `sum_all` and
/// comes back through `outof_cpu_vec`, so a mistake in either transfer shows up only here.
#[test]
fn reduce_multi_axis_keepdims_and_whole() {
    require_cuda!();
    let d = DeviceCuda::new(0);
    let dcpu = DeviceCpuSerial::default();
    let cpu: Tensor<f64, DeviceCpuSerial, _> =
        rt::asarray((exact_data(24, 11), &dcpu)).into_shape([2, 3, 4]);
    let gpu = cpu.to_device(&d);

    // (1, 2) contracted: those axes are one contiguous 12-element block per output
    let got: Vec<f64> = host_vec1(&gpu.sum_axes([1isize, 2]));
    let want: Vec<f64> = host_vec1(&cpu.sum_axes([1isize, 2]));
    assert_eq!(got, want, "sum over (1, 2)");
    assert_eq!(got.len(), 2, "shape [2, 3, 4] reduced over (1, 2) leaves 2 elements");

    // (0, 2) contracted is NOT one block: the two runs of 4 sit 8 elements apart, because the
    // middle axis belongs to the output. The kernel walks one arithmetic progression, so this is
    // refused rather than mis-computed. Widening it needs a multi-run walk (or pack-then-reduce);
    // the refusal is asserted so it cannot regress silently into a wrong answer.
    match rt::sum_axes_f(&gpu, [0isize, 2]) {
        Ok(_) => panic!("a non-contiguous contraction should be refused, not computed"),
        Err(e) => {
            let msg = format!("{e}");
            assert!(msg.contains("not one contiguous block"), "got: {msg}");
        },
    }

    // keepdims is a host-side layout rewrite: the values must be untouched, the shape must not be
    let k = gpu.sum_with_args((-1isize, true));
    let k_cpu = cpu.sum_with_args((-1isize, true));
    assert_eq!(k.shape(), &[2, 3, 1]);
    assert_eq!(k.shape(), k_cpu.shape());
    assert_eq!(host_nd(&k), host_nd(&k_cpu), "keepdims values (rank-3, so host_nd)");

    // axes = None: 0-D output, built by a D2H + H2D round trip in the tensor tier
    let w = gpu.sum_with_args(());
    let w_cpu = cpu.sum_with_args(());
    assert_eq!(w.shape(), &[] as &[usize], "axes = None gives a 0-D tensor");
    assert_eq!(w.to_scalar(), w_cpu.to_scalar(), "axes = None values");
}

/// A view with a non-zero `layout.offset()`: the launcher folds the split layout's offset into the
/// base pointer, so a doubled or dropped offset shifts every result by a constant.
#[test]
fn reduce_offset_view() {
    require_cuda!();
    let d = DeviceCuda::new(0);
    let dcpu = DeviceCpuSerial::default();
    let cpu: Tensor<f64, DeviceCpuSerial, _> =
        rt::asarray((exact_data(100 * 40, 5), &dcpu)).into_shape([100, 40]);
    let gpu = cpu.to_device(&d);

    for axis in [0isize, 1] {
        let got: Vec<f64> = host_vec1(&gpu.i(3..).sum_axes(axis));
        let want: Vec<f64> = host_vec1(&cpu.i(3..).sum_axes(axis));
        assert_eq!(got, want, "offset view (97 x 40), axis {axis}");
    }
}

/// The four traits whose output is not the operand type: `all`, `any`, `count_nonzero`, and `sum`
/// over `bool` (which is `TensorSumBoolAPI`, not `OpSumAPI`).
#[test]
fn reduce_bool_family() {
    require_cuda!();
    let d = DeviceCuda::new(0);
    let dcpu = DeviceCpuSerial::default();
    let a: Tensor<f64, _, _> = rt::arange((0., 32., 1., &d)).into_shape([4, 8]);
    let b = rt::lt(&a, 20.0);
    let a_cpu: Tensor<f64, DeviceCpuSerial, _> = rt::arange((0., 32., 1., &dcpu)).into_shape([4, 8]);
    let b_cpu = rt::lt(&a_cpu, 20.0);

    // whole-array forms: bool in, bool out; bool in, usize out
    assert_eq!(b.all(), b_cpu.all(), "all");
    assert_eq!(b.any(), b_cpu.any(), "any");
    assert_eq!(b.sum(), b_cpu.sum(), "sum over bool (TensorSumBoolAPI)");

    // axes form: the `usize` output travels through `@TOUT@` and the store cast
    let got: Vec<usize> = host_vec1(&b.sum_axes(0));
    let want: Vec<usize> = host_vec1(&b_cpu.sum_axes(0));
    assert_eq!(got, want, "sum over bool, axis 0");
    assert_eq!(got.len(), 8);
    assert_eq!(got.iter().sum::<usize>(), b.sum() as usize, "the axis form must partition the total");

    // count_nonzero on a numeric tensor, where the fold's predicate is `x != 0`
    let z: Tensor<f64, _, _> = rt::full(([5usize], 0.0f64, &d));
    assert_eq!(z.count_nonzero(), 0);
    let m: Tensor<f64, _, _> = rt::asarray((vec![0.0f64, 1.0, 0.0, -2.0, 0.0], &d));
    assert_eq!(m.count_nonzero(), 2);
}

/// `min`/`max` are the only rows whose identity is an infinity, and the only rows where the CPU and
/// CUDA defaults disagree: `ExtReal::ext_min` propagates NaN (IEEE 754-2019 `minimum`), while CUDA's
/// `fmin` returns the non-NaN operand. These two cases are what the `RSTSR_MIN` guard exists for.
#[test]
fn reduce_min_max_nan_and_signed_zero() {
    require_cuda!();
    let d = DeviceCuda::new(0);
    let dcpu = DeviceCpuSerial::default();

    let with_nan: Tensor<f64, DeviceCpuSerial, _> =
        rt::asarray((vec![1.0f64, f64::NAN, -2.0, 3.0], &dcpu));
    let g = with_nan.to_device(&d);
    assert!(with_nan.min().is_nan(), "CPU reference: NaN propagates");
    assert!(g.min().is_nan(), "min must propagate NaN, like ExtReal::ext_min");
    assert!(g.max().is_nan(), "max must propagate NaN");

    // ±0. The device must be order-INDEPENDENT here; the CPU is not. `ExtReal::ext_min` calls
    // `T::min`, i.e. LLVM's `minnum`, which resolves an exact tie by operand position — so on this
    // machine the CPU answers +0.0 for min(-0.0, +0.0) and -0.0 for min(+0.0, -0.0). A commutative
    // tree cannot reproduce an order-dependent tie-break, and should not: IEEE 754-2019
    // `minimum`/`maximum` (which `ExtReal`'s own doc says it wants) fixes the sign regardless of
    // order, and that is what `fmin`/`fmax` give. The two are numerically equal either way, so the
    // assertion is on the sign being stable, plus plain `==` against the CPU.
    for vals in [vec![-0.0f64, 0.0], vec![0.0f64, -0.0]] {
        let c: Tensor<f64, DeviceCpuSerial, _> = rt::asarray((vals, &dcpu));
        let g = c.to_device(&d);
        assert_eq!(g.min().to_bits(), (-0.0f64).to_bits(), "device min keeps -0, either order");
        assert_eq!(g.max().to_bits(), 0.0f64.to_bits(), "device max keeps +0, either order");
        assert_eq!(g.min(), c.min(), "min agrees numerically with the CPU");
        assert_eq!(g.max(), c.max(), "max agrees numerically with the CPU");
    }
}

/// Empty input: the launcher returns without writing, so the caller must have filled the output with
/// the identity. `sum([]) == 0` reading uninitialised device memory that happens to look like zero is
/// exactly the bug this catches.
#[test]
fn reduce_empty_identities() {
    require_cuda!();
    let d = DeviceCuda::new(0);
    let e: Tensor<f64, _, _> = rt::zeros(([0usize, 3], &d));

    assert_eq!(e.sum(), 0.0, "sum of an empty tensor is the additive identity");
    assert_eq!(e.prod(), 1.0, "prod of an empty tensor is the multiplicative identity");
    assert_eq!(e.l2_norm(), 0.0);
    assert!(e.mean().is_nan(), "an empty mean is 0/0, as on the CPU");
    assert_eq!(e.count_nonzero(), 0);

    // min/max have no identity: the CPU raises, and so must we
    match rt::min_f(&e) {
        Ok(v) => panic!("min of an empty tensor should raise, got {v}"),
        Err(e) => assert!(format!("{e}").contains("zero-size"), "got: {e}"),
    }
    match rt::max_f(&e) {
        Ok(v) => panic!("max of an empty tensor should raise, got {v}"),
        Err(e) => assert!(format!("{e}").contains("zero-size"), "got: {e}"),
    }
}

/// The explicit-output-dtype path: `f32` operand accumulated in `f64`.
///
/// `0.1f32` is exact in `f64`, so summing 1000 of them in `f64` is exact while summing them in `f32`
/// is not — which is what makes this a test of `@TACC@` and not just of the kernel running.
#[test]
fn reduce_dtype_accumulator() {
    require_cuda!();
    let d = DeviceCuda::new(0);
    let dcpu = DeviceCpuSerial::default();

    let a: Tensor<f32, _, _> = rt::full(([1_000usize], 0.1f32, &d));
    let a_cpu: Tensor<f32, DeviceCpuSerial, _> = rt::full(([1_000usize], 0.1f32, &dcpu));

    // `axes = ()` gives a 0-D result, and `host_vec1` needs a 1-D tensor (`to_raw_f` asserts it),
    // so read the single element with `to_scalar` instead.
    let got: f64 = rt::sum_with_dtype::<f32, f64, _, _>(&a, ()).to_scalar();
    let want: f64 = rt::sum_with_dtype::<f32, f64, _, _>(&a_cpu, ()).to_scalar();
    let in_f32: f32 = rt::sum_with_dtype::<f32, f32, _, _>(&a_cpu, ()).to_scalar();

    assert_eq!(got, want, "f32 operand accumulated in f64");

    // `prod`/`mean` share the kernel and the helpers and differ only by the fold row, but neither
    // had a direct test. Exact values, so the comparison is bitwise.
    let small: Tensor<f32, _, _> = rt::full(([4usize], 2.0f32, &d));
    let small_cpu: Tensor<f32, DeviceCpuSerial, _> = rt::full(([4usize], 2.0f32, &dcpu));
    assert_eq!(
        rt::prod_with_dtype::<f32, f64, _, _>(&small, ()).to_scalar(),
        rt::prod_with_dtype::<f32, f64, _, _>(&small_cpu, ()).to_scalar(),
        "prod_with_dtype: 2^4"
    );
    assert_eq!(
        rt::mean_with_dtype::<f32, f64, _, _>(&small, ()).to_scalar(),
        rt::mean_with_dtype::<f32, f64, _, _>(&small_cpu, ()).to_scalar(),
        "mean_with_dtype"
    );
    assert_ne!(
        got as f32, in_f32,
        "the test data must actually distinguish an f64 accumulator from an f32 one"
    );
}

/// `var` and `std` are the two-slot folds: `(Σx, Σx²)` accumulated together, `Σx²/n − (Σx/n)²` at the
/// writer. Three things are tested separately because they fail for different reasons:
///
/// - the **arithmetic**, against the CPU's own one-pass formula;
/// - the **strided path**, where the contracted axis is a view's (the plan's step machinery);
/// - the **multi-warp stage**, a contraction long enough that one output spans several warps — the
///   only case that executes the shared-memory half of the tree, and the half a second accumulator
///   could plausibly break.
///
/// Tolerances are `assert_close`, not equality: the CPU sums sequentially and the kernel sums as a
/// tree, so a variance is the same quantity in two summation orders.
#[test]
fn var_std_vs_cpu() {
    require_cuda!();
    let d = DeviceCuda::new(0);
    let dcpu = DeviceCpuSerial::default();

    // Arithmetic: both axes, and the whole array. Integer-valued data with a mean near zero, so the
    // one-pass formula is not dominated by cancellation and the comparison is about the fold.
    let (m, k) = (6usize, 32usize);
    let a_cpu: Tensor<f64, DeviceCpuSerial, _> =
        rt::asarray((exact_data(m * k, 91), &dcpu)).into_shape([m, k]);
    let a = a_cpu.to_device(&d);
    assert_close(&host_nd(&rt::var_axes(&a, 1)), &host_nd(&rt::var_axes(&a_cpu, 1)), "var over the last axis");
    assert_close(&host_nd(&rt::var_axes(&a, 0)), &host_nd(&rt::var_axes(&a_cpu, 0)), "var over the first axis");
    // `var`/`var_all` reduce every axis and hand back the scalar itself, not a 0-D tensor.
    assert_close(&[rt::var(&a)], &[rt::var(&a_cpu)], "var over every axis");
    assert_close(&host_nd(&rt::std_axes(&a, 1)), &host_nd(&rt::std_axes(&a_cpu, 1)), "std over the last axis");

    // Strided: `a^T` has the `m` axis contiguous, so the contracted axis of `var(t, 0)` is a view.
    let t_cpu = a_cpu.swapaxes(0, 1);
    let t = t_cpu.to_device(&d);
    assert_eq!(t.layout().stride()[0], 1, "fixture: the m axis must be the contiguous one");
    assert_close(&host_nd(&rt::var_axes(&t, 0)), &host_nd(&rt::var_axes(&t_cpu, 0)), "var over a strided axis");
    assert_close(&host_nd(&rt::std_axes(&t, 0)), &host_nd(&rt::std_axes(&t_cpu, 0)), "std over a strided axis");

    // Multi-warp: the lane count is `max(free, need).next_power_of_two().min(cap)` with
    // `need = MIN_BLOCKS*BLOCK/n_out`, so it is *few outputs* rather than long contractions that push
    // lanes past a warp — here 4 outputs of 4096 give `lanes = 256`, which is what makes the
    // shared-memory half of the tree live. A single-warp fixture cannot reach that code.
    let (m2, k2) = (4usize, 4096usize);
    let big_cpu: Tensor<f64, DeviceCpuSerial, _> =
        rt::asarray((exact_data(m2 * k2, 92), &dcpu)).into_shape([m2, k2]);
    let big = big_cpu.to_device(&d);
    assert_close(&host_nd(&rt::var_axes(&big, 1)), &host_nd(&rt::var_axes(&big_cpu, 1)), "var over 4096 elements");
    assert_close(&host_nd(&rt::std_axes(&big, 1)), &host_nd(&rt::std_axes(&big_cpu, 1)), "std over 4096 elements");

    // A single `f32` instantiation: the template's `nan`/`sqrt` text and the accumulator width are
    // per-dtype, and an untested second instantiation is exactly the kind of thing that compiles and
    // is wrong.
    let f_cpu: Tensor<f32, DeviceCpuSerial, _> = rt::asarray((
        exact_data(64, 93).into_iter().map(|v| v as f32).collect::<Vec<f32>>(),
        &dcpu,
    ))
    .into_shape([8, 8]);
    let f = f_cpu.to_device(&d);
    let got: Vec<f64> = host_nd(&rt::var_axes(&f, 1)).into_iter().map(|v| v as f64).collect();
    let want: Vec<f64> = host_nd(&rt::var_axes(&f_cpu, 1)).into_iter().map(|v| v as f64).collect();
    assert_close(&got, &want, "f32 var");
}

/* #endregion */

/* #region outer + cross */

/// `outer` reuses the planned binary launcher with the operands widened onto `(n, m)`; the only new
/// thing is the zero stride on the axis each operand does not own. Strided and offset views are the
/// cases that would break if the widening dropped the operand's own stride or offset.
#[test]
fn outer_vs_cpu() {
    require_cuda!();
    let d = DeviceCuda::new(0);
    let dcpu = DeviceCpuSerial::default();
    let a: Tensor<f64, _, _> = rt::arange((0., 5., 1., &d));
    let b: Tensor<f64, _, _> = rt::arange((10., 13., 1., &d));
    let a_cpu: Tensor<f64, DeviceCpuSerial, _> = rt::arange((0., 5., 1., &dcpu));
    let b_cpu: Tensor<f64, DeviceCpuSerial, _> = rt::arange((10., 13., 1., &dcpu));

    assert_eq!(host_nd(&rt::outer(&a, &b)), host_nd(&rt::outer(&a_cpu, &b_cpu)), "dense");

    // offset view: a[1..]
    assert_eq!(
        host_nd(&rt::outer(a.i(1..), &b)),
        host_nd(&rt::outer(a_cpu.i(1..), &b_cpu)),
        "offset operand"
    );

    // negative-stride view: a[::-2]
    let rev = SliceI::new(None::<isize>, None::<isize>, Some(-2isize));
    assert_eq!(
        host_nd(&rt::outer(a.i(rev), &b)),
        host_nd(&rt::outer(a_cpu.i(rev), &b_cpu)),
        "negative-stride operand"
    );

    // into an existing output: the tier transmutes it into the write-only slot
    // `outer_from` wants a fixed-rank `Ix2` output; `into_shape` yields `IxD`, so narrow it
    let mut c: Tensor<f64, DeviceCuda, Ix2> = rt::zeros(([15usize], &d)).into_shape([5, 3]).into_dim::<Ix2>();
    let mut c_cpu: Tensor<f64, DeviceCpuSerial, Ix2> =
        rt::zeros(([15usize], &dcpu)).into_shape([5, 3]).into_dim::<Ix2>();
    rt::outer_from(&mut c, &a, &b);
    rt::outer_from(&mut c_cpu, &a_cpu, &b_cpu);
    assert_eq!(host_nd(&c), host_nd(&c_cpu), "outer_from");
}

/// `cross` has its own kernel (four reads, three writes per slot). The four cases pin the parts that
/// a one-value-per-operand template could not have handled: the cyclic component order, a broadcast
/// operand (stride 0 along the size-3 axis), a non-last axis, and a negative `axis`.
#[test]
fn cross_vs_cpu() {
    require_cuda!();
    let d = DeviceCuda::new(0);
    let dcpu = DeviceCpuSerial::default();

    let a: Tensor<f64, _, _> = rt::arange((0., 12., 1., &d)).into_shape([4, 3]);
    let b: Tensor<f64, _, _> = rt::arange((1., 13., 1., &d)).into_shape([4, 3]);
    let a_cpu: Tensor<f64, DeviceCpuSerial, _> = rt::arange((0., 12., 1., &dcpu)).into_shape([4, 3]);
    let b_cpu: Tensor<f64, DeviceCpuSerial, _> = rt::arange((1., 13., 1., &dcpu)).into_shape([4, 3]);

    // last axis, named and negative
    assert_eq!(host_nd(&rt::cross(&a, &b, 1)), host_nd(&rt::cross(&a_cpu, &b_cpu, 1)), "axis 1");
    assert_eq!(host_nd(&rt::cross(&a, &b, -1)), host_nd(&rt::cross(&a_cpu, &b_cpu, -1)), "axis -1");

    // a × a == 0
    let z = host_nd(&rt::cross(&a, &a, 1));
    assert!(z.iter().all(|v| *v == 0.0), "a × a must vanish, got {z:?}");

    // broadcast: a single vector against a stack — stride 0 along the cross axis for `b`
    let v: Tensor<f64, _, _> = rt::arange((1., 4., 1., &d));
    let v_cpu: Tensor<f64, DeviceCpuSerial, _> = rt::arange((1., 4., 1., &dcpu));
    assert_eq!(
        host_nd(&rt::cross(&a, &v, 1)),
        host_nd(&rt::cross(&a_cpu, &v_cpu, 1)),
        "broadcast operand"
    );

    // a middle axis of a rank-3 tensor, and a strided (non-contiguous) stack
    let a3: Tensor<f64, _, _> = rt::arange((0., 24., 1., &d)).into_shape([2, 3, 4]);
    let b3: Tensor<f64, _, _> = rt::arange((1., 25., 1., &d)).into_shape([2, 3, 4]);
    let a3_cpu: Tensor<f64, DeviceCpuSerial, _> = rt::arange((0., 24., 1., &dcpu)).into_shape([2, 3, 4]);
    let b3_cpu: Tensor<f64, DeviceCpuSerial, _> = rt::arange((1., 25., 1., &dcpu)).into_shape([2, 3, 4]);
    assert_eq!(host_nd(&rt::cross(&a3, &b3, 1)), host_nd(&rt::cross(&a3_cpu, &b3_cpu, 1)), "axis 1 of [2,3,4]");

    // every other row of the stack: the collapsed plan is then strided, not flat
    let step = SliceI::new(None::<isize>, None::<isize>, Some(2isize));
    assert_eq!(
        host_nd(&rt::cross(a.i(step), b.i(step), 1)),
        host_nd(&rt::cross(a_cpu.i(step), b_cpu.i(step), 1)),
        "strided stack"
    );
}

/* #endregion */

/* #region matmul */

/// One operand on both devices.
type F64Pair = (Tensor<f64, DeviceCuda, IxD>, Tensor<f64, DeviceCpuSerial, IxD>);

/// Integer-valued `[m, k]` and `[k, n]` on both devices, so every product and every partial sum is
/// exact and the comparison stays an `assert_eq!`.
fn matmul_pair(m: usize, k: usize, n: usize, seed: u64) -> (F64Pair, F64Pair) {
    let d = DeviceCuda::new(0);
    let dcpu = DeviceCpuSerial::default();
    let a_cpu: Tensor<f64, DeviceCpuSerial, _> =
        rt::asarray((exact_data(m * k, seed), &dcpu)).into_shape([m, k]);
    let b_cpu: Tensor<f64, DeviceCpuSerial, _> =
        rt::asarray((exact_data(k * n, seed ^ 0x5bf0_3635), &dcpu)).into_shape([k, n]);
    ((a_cpu.to_device(&d), a_cpu), (b_cpu.to_device(&d), b_cpu))
}

/// Plain `[m, k] @ [k, n]`, including shapes that are not multiples of the 64x64 tile in any axis —
/// those are what exercise the three clipping guards (A slab, B slab, epilogue).
#[test]
fn matmul_gemm22_vs_cpu() {
    require_cuda!();
    for &(m, k, n) in &[
        (32usize, 32usize, 32usize),
        (64, 64, 64),
        (17, 33, 5),
        (1, 8, 9),
        (130, 70, 129),
    ] {
        let ((a, a_cpu), (b, b_cpu)) = matmul_pair(m, k, n, m as u64 * 7 + n as u64);
        let got = host_nd(&rt::matmul(&a, &b));
        let want = host_nd(&rt::matmul(&a_cpu, &b_cpu));
        assert_eq!(got.len(), m * n, "({m} x {k}) @ ({k} x {n}) has {m}x{n} outputs");
        assert_eq!(got, want, "({m} x {k}) @ ({k} x {n})");
    }
}

/// The two non-default operand orientations, which select a different thread map each.
///
/// `a[m, k]` built from a `[k, m]` array has its `m` axis contiguous → the lanes follow `m`.
/// `b[k, n]` built from `[n, k]` has its `k` axis contiguous → the lanes follow `k`. Both are plain
/// views, so a transposed matmul costs no copy; that is the whole reason the lane map is a
/// compile-time choice instead of a fixed axis.
#[test]
fn matmul_transposed_operands() {
    require_cuda!();
    let d = DeviceCuda::new(0);
    let dcpu = DeviceCpuSerial::default();
    let (m, k, n) = (40usize, 24usize, 20usize);

    // a^T-style left operand: [m, k] whose m axis is contiguous. `swapaxes` yields a view, which is
    // the point — a transposed matmul must not require a copy on either side.
    let a_base = rt::asarray((exact_data(k * m, 3), &dcpu)).into_shape([k, m]);
    let at_cpu = a_base.swapaxes(0, 1);
    // b^T-style right operand: [k, n] whose k axis is contiguous
    let b_base = rt::asarray((exact_data(n * k, 4), &dcpu)).into_shape([n, k]);
    let bt_cpu = b_base.swapaxes(0, 1);
    let b0_cpu: Tensor<f64, DeviceCpuSerial, _> =
        rt::asarray((exact_data(k * n, 5), &dcpu)).into_shape([k, n]);
    let a0_cpu: Tensor<f64, DeviceCpuSerial, _> =
        rt::asarray((exact_data(m * k, 6), &dcpu)).into_shape([m, k]);

    let at = at_cpu.to_device(&d);
    let bt = bt_cpu.to_device(&d);
    let a0 = a0_cpu.to_device(&d);
    let b0 = b0_cpu.to_device(&d);

    assert_eq!(at.layout().stride()[0], 1, "fixture: the m axis must be the contiguous one");
    assert_eq!(bt.layout().stride()[0], 1, "fixture: the k axis must be the contiguous one");

    for (label, got, want) in [
        ("a(m-contig) @ b", &rt::matmul(&at, &b0), &rt::matmul(&at_cpu, &b0_cpu)),
        ("a @ b(k-contig)", &rt::matmul(&a0, &bt), &rt::matmul(&a0_cpu, &bt_cpu)),
        ("a(m-contig) @ b(k-contig)", &rt::matmul(&at, &bt), &rt::matmul(&at_cpu, &bt_cpu)),
    ] {
        assert_eq!(host_nd(got), host_nd(want), "{label}");
    }
}

/// Batched matmul: one `grid.z` slice per output matrix.
#[test]
fn matmul_batched_vs_cpu() {
    require_cuda!();
    let d = DeviceCuda::new(0);
    let dcpu = DeviceCpuSerial::default();
    let (p, m, k, n) = (5usize, 12usize, 16usize, 8usize);

    let a_cpu: Tensor<f64, DeviceCpuSerial, _> =
        rt::asarray((exact_data(p * m * k, 11), &dcpu)).into_shape([p, m, k]);
    let b_cpu: Tensor<f64, DeviceCpuSerial, _> =
        rt::asarray((exact_data(p * k * n, 12), &dcpu)).into_shape([p, k, n]);
    let a = a_cpu.to_device(&d);
    let b = b_cpu.to_device(&d);

    let got = host_nd(&rt::matmul(&a, &b));
    assert_eq!(got.len(), p * m * n);
    assert_eq!(got, host_nd(&rt::matmul(&a_cpu, &b_cpu)), "batched [p,m,k] @ [p,k,n]");

    // broadcast batch: the left operand has no batch axis, so its batch stride is 0
    let a2_cpu: Tensor<f64, DeviceCpuSerial, _> =
        rt::asarray((exact_data(m * k, 13), &dcpu)).into_shape([m, k]);
    let a2 = a2_cpu.to_device(&d);
    assert_eq!(
        host_nd(&rt::matmul(&a2, &b)),
        host_nd(&rt::matmul(&a2_cpu, &b_cpu)),
        "broadcast batch: [m,k] @ [p,k,n]"
    );
}

/// `matmul_from`: the `beta` path, which reads `c`. The opposite of `matmul_uninit`, whose buffer is
/// `MaybeUninit` and must never be read — hence the two entry points.
#[test]
fn matmul_from_alpha_beta() {
    require_cuda!();
    let d = DeviceCuda::new(0);
    let dcpu = DeviceCpuSerial::default();
    let (m, k, n) = (20usize, 12usize, 16usize);
    let ((a, a_cpu), (b, b_cpu)) = matmul_pair(m, k, n, 21);

    let mut c: Tensor<f64, _, _> = rt::full(([m * n], 1.0f64, &d)).into_shape([m, n]);
    let mut c_cpu: Tensor<f64, DeviceCpuSerial, _> =
        rt::full(([m * n], 1.0f64, &dcpu)).into_shape([m, n]);

    rt::matmul_from(&mut c, &a, &b, 2.0f64, 3.0f64);
    rt::matmul_from(&mut c_cpu, &a_cpu, &b_cpu, 2.0f64, 3.0f64);
    assert_eq!(host_nd(&c), host_nd(&c_cpu), "c = 2*(a@b) + 3*c");

    // beta = 0 must not read c: same answer from a NaN-filled buffer
    let mut z: Tensor<f64, _, _> = rt::full(([m * n], f64::NAN, &d)).into_shape([m, n]);
    let mut z_cpu: Tensor<f64, DeviceCpuSerial, _> =
        rt::full(([m * n], f64::NAN, &dcpu)).into_shape([m, n]);
    rt::matmul_from(&mut z, &a, &b, 1.0f64, 0.0f64);
    rt::matmul_from(&mut z_cpu, &a_cpu, &b_cpu, 1.0f64, 0.0f64);
    assert_eq!(host_nd(&z), host_nd(&z_cpu), "beta = 0 must ignore c's contents");
    assert!(host_nd(&z).iter().all(|v| v.is_finite()), "NaN leaked in through beta = 0");
}

/// Rules 3 and 4 (GEMV / GEVM). Both go through the same kernel with a length-1 tile axis, because
/// the layout module presents every rule as a 2-D matmul on its `_matmul` parts.
#[test]
fn matmul_gemv_gevm_vs_cpu() {
    require_cuda!();
    let d = DeviceCuda::new(0);
    let dcpu = DeviceCpuSerial::default();
    let (m, k, n) = (24usize, 16usize, 18usize);

    let a_cpu: Tensor<f64, DeviceCpuSerial, _> =
        rt::asarray((exact_data(m * k, 31), &dcpu)).into_shape([m, k]);
    let v_cpu: Tensor<f64, DeviceCpuSerial, _> = rt::asarray((exact_data(k, 32), &dcpu));
    let w_cpu: Tensor<f64, DeviceCpuSerial, _> = rt::asarray((exact_data(k, 33), &dcpu));
    let c_cpu: Tensor<f64, DeviceCpuSerial, _> =
        rt::asarray((exact_data(k * n, 34), &dcpu)).into_shape([k, n]);

    let a = a_cpu.to_device(&d);
    let v = v_cpu.to_device(&d);
    let w = w_cpu.to_device(&d);
    let c = c_cpu.to_device(&d);

    // rule 4: [m, k] @ [k] -> [m]
    assert_eq!(
        host_vec1(&rt::matmul(&a, &v)),
        host_vec1(&rt::matmul(&a_cpu, &v_cpu)),
        "gemv: [m,k] @ [k]"
    );
    // rule 3: [k] @ [k, n] -> [n]
    assert_eq!(
        host_vec1(&rt::matmul(&w, &c)),
        host_vec1(&rt::matmul(&w_cpu, &c_cpu)),
        "gevm: [k] @ [k,n]"
    );
}

/// The inner-dot rule is a reduction, not a GEMM, and is refused by name rather than run as a
/// `1 x 1` tile of 256 redundant threads.
#[test]
fn matmul_inner_dot_is_refused_by_name() {
    require_cuda!();
    let d = DeviceCuda::new(0);
    let v: Tensor<f64, _, _> = rt::arange((0., 16., 1., &d));
    match rt::matmul_f(&v, &v) {
        Ok(_) => panic!("inner dot should be refused, but it computed a result"),
        Err(e) => {
            let msg = format!("{e}");
            assert!(msg.contains("single-axis contraction"), "expected the reduction reason, got: {msg}");
        },
    }
}

/// `ColMajor`: the device reverses every layout and swaps the operands, per the identity in
/// `rstsr_common::layout::matmul`. If that reversal is wrong the numbers come out transposed, which
/// only a rectangular case can show.
#[test]
fn matmul_col_major_vs_cpu() {
    require_cuda!();
    let mut d = DeviceCuda::new(0);
    d.set_default_order(ColMajor);
    let mut dcpu = DeviceCpuSerial::default();
    dcpu.set_default_order(ColMajor);
    let (m, k, n) = (20usize, 12usize, 32usize);

    let a_cpu: Tensor<f64, DeviceCpuSerial, _> =
        rt::asarray((exact_data(m * k, 41), &dcpu)).into_shape([m, k]);
    let b_cpu: Tensor<f64, DeviceCpuSerial, _> =
        rt::asarray((exact_data(k * n, 42), &dcpu)).into_shape([k, n]);
    let a = a_cpu.to_device(&d);
    let b = b_cpu.to_device(&d);

    let got = host_nd(&rt::matmul(&a, &b));
    let want = host_nd(&rt::matmul(&a_cpu, &b_cpu));
    assert_eq!(got.len(), m * n, "a transposed answer would have {k} elements");
    assert_eq!(got, want, "col-major ({m} x {k}) @ ({k} x {n})");
}

/// Several threads using one device at once, which is one stream and therefore one cuBLAS handle.
///
/// `cublasHandle_t` is not thread-safe, but `cudarc` declares `unsafe impl Sync for CudaBlas`, and
/// the registry's one-stream-per-ordinal cache hands every `DeviceCuda::new(0)` the same handle. The
/// lock inside `CublasHandle` is what makes this legal, and this is the test that exercises it: the
/// library is expected to support several threads on one device, not merely to survive `cargo test`'s
/// default parallelism.
///
/// Each thread uses its **own** data on purpose. An earlier version gave every thread the same
/// matrices, which made the test blind to exactly what it should catch: if two threads' buffers or
/// results were confused for one another, both would still compute the expected answer.
#[test]
fn matmul_from_several_threads_on_one_device() {
    require_cuda!();
    const THREADS: u64 = 4;
    const ROUNDS: usize = 8;
    let (m, k, n) = (24usize, 16usize, 20usize);
    let dcpu = DeviceCpuSerial::default();

    // (a, b, reference) per thread, all distinct.
    let cases: Vec<(Vec<f64>, Vec<f64>, Vec<f64>)> = (0..THREADS)
        .map(|t| {
            let a_data = exact_data(m * k, 61 + 2 * t);
            let b_data = exact_data(k * n, 62 + 2 * t);
            let a_cpu: Tensor<f64, DeviceCpuSerial, _> =
                rt::asarray((a_data.clone(), &dcpu)).into_shape([m, k]);
            let b_cpu: Tensor<f64, DeviceCpuSerial, _> =
                rt::asarray((b_data.clone(), &dcpu)).into_shape([k, n]);
            let want = host_nd(&rt::matmul(&a_cpu, &b_cpu));
            (a_data, b_data, want)
        })
        .collect();

    std::thread::scope(|scope| {
        for (t, (a_data, b_data, want)) in cases.iter().enumerate() {
            scope.spawn(move || {
                let d = DeviceCuda::new(0);
                let a = rt::asarray((a_data.clone(), &d)).into_shape([m, k]);
                let b = rt::asarray((b_data.clone(), &d)).into_shape([k, n]);
                for round in 0..ROUNDS {
                    assert_eq!(
                        host_nd(&rt::matmul(&a, &b)),
                        *want,
                        "thread {t}, round {round}: a concurrent call on the shared stream returned \
                         another thread's product"
                    );
                }
            });
        }
    });
}

/* #endregion */

/* #region tensordot */

/// The tier-0 promise: a contraction that is a GEMM over **views**, so no copy is made.
///
/// Every case here is one the planner accepts; the refusals are pinned by
/// [`tensordot_refuses_a_contraction_that_needs_a_pack`]. What is being tested is not the arithmetic
/// (cuBLAS owns that) but the *scheduling*: the axis reorder to `[free…, ctr…]` / `[ctr…, free…]`,
/// the merge of each block into one `M`/`K`/`N` factor, and the handover to `matmul`.
#[test]
fn tensordot_views_only_vs_cpu() {
    require_cuda!();
    let d = DeviceCuda::new(0);
    let dcpu = DeviceCpuSerial::default();

    // (5,4) @ (4,6), contracting a's trailing with b's leading axis.
    let a_cpu: Tensor<f64, DeviceCpuSerial, _> =
        rt::asarray((exact_data(5 * 4, 71), &dcpu)).into_shape([5, 4]);
    let b_cpu: Tensor<f64, DeviceCpuSerial, _> =
        rt::asarray((exact_data(4 * 6, 72), &dcpu)).into_shape([4, 6]);
    let (a, b) = (a_cpu.to_device(&d), b_cpu.to_device(&d));
    assert_eq!(
        host_nd(&rt::tensordot(&a, &b, 1)),
        host_nd(&rt::tensordot(&a_cpu, &b_cpu, 1)),
        "2-D: contract a[i, k] with b[k, j]"
    );

    // A left operand whose `m` axis is contiguous. The canonical block is already `[m, k]` with
    // strides `[1, m]`, i.e. a 2-D layout cuBLAS takes as `OP_T` with `lda = m` — the case that
    // would silently transpose if the planner or the geometry got the roles wrong.
    let base_cpu: Tensor<f64, DeviceCpuSerial, _> =
        rt::asarray((exact_data(4 * 5, 73), &dcpu)).into_shape([4, 5]);
    let at_cpu = base_cpu.swapaxes(0, 1);
    let at = at_cpu.to_device(&d);
    assert_eq!(at.layout().stride()[0], 1, "fixture: the m axis must be the contiguous one");
    assert_eq!(
        host_nd(&rt::tensordot(&at, &b, 1)),
        host_nd(&rt::tensordot(&at_cpu, &b_cpu, 1)),
        "2-D with an m-contiguous left operand (OP_T)"
    );

    // (2,3,4) @ (3,4,5) contracting a's trailing *block* with b's leading block: the K factor is a
    // merge of two axes on each side, still a view.
    let a3_cpu: Tensor<f64, DeviceCpuSerial, _> =
        rt::asarray((exact_data(2 * 3 * 4, 74), &dcpu)).into_shape([2, 3, 4]);
    let b3_cpu: Tensor<f64, DeviceCpuSerial, _> =
        rt::asarray((exact_data(3 * 4 * 5, 75), &dcpu)).into_shape([3, 4, 5]);
    let (a3, b3) = (a3_cpu.to_device(&d), b3_cpu.to_device(&d));
    assert_eq!(
        host_nd(&rt::tensordot(&a3, &b3, ([1, 2], [0, 1]))),
        host_nd(&rt::tensordot(&a3_cpu, &b3_cpu, ([1, 2], [0, 1]))),
        "3-D: contract a's trailing block"
    );

    // The *leading*-axis orientation: contract a's axis 0 of (3,4,5) with (3,2). The free block of
    // `a` is (4,5) with the contracted axis trailing in memory, so the 2-D block comes back as
    // [20, 3] with strides [1, 20] — unit stride on `M`, i.e. an `OP_T` handover. This was the case
    // I expected tier 0 to refuse; the audit in `probe/geomprobe` showed it does not, so it is pinned
    // here rather than left to the doc comment.
    let l_cpu: Tensor<f64, DeviceCpuSerial, _> =
        rt::asarray((exact_data(3 * 4 * 5, 82), &dcpu)).into_shape([3, 4, 5]);
    let m_cpu: Tensor<f64, DeviceCpuSerial, _> =
        rt::asarray((exact_data(3 * 2, 83), &dcpu)).into_shape([3, 2]);
    let (l, m) = (l_cpu.to_device(&d), m_cpu.to_device(&d));
    assert_eq!(
        host_nd(&rt::tensordot(&l, &m, ([0], [0]))),
        host_nd(&rt::tensordot(&l_cpu, &m_cpu, ([0], [0]))),
        "3-D: contract a's leading axis (the OP_T handover)"
    );

    // The DF-MP2 shape: one contracted axis, two free axes on the left, one on the right. This is
    // `(i, a, P) x (P, b) -> (i, a, b)`, the pair-energy tile, and it must not need a pack.
    let c_cpu: Tensor<f64, DeviceCpuSerial, _> =
        rt::asarray((exact_data(3 * 4 * 5, 76), &dcpu)).into_shape([3, 4, 5]);
    let v_cpu: Tensor<f64, DeviceCpuSerial, _> =
        rt::asarray((exact_data(5 * 2, 77), &dcpu)).into_shape([5, 2]);
    let (c, v) = (c_cpu.to_device(&d), v_cpu.to_device(&d));
    assert_eq!(
        host_nd(&rt::tensordot(&c, &v, ([2], [0]))),
        host_nd(&rt::tensordot(&c_cpu, &v_cpu, ([2], [0]))),
        "DF-MP2 tile: (i, a, P) x (P, b)"
    );
}

/// `ColMajor`: the planner is given col-major canonical blocks, and the geometry reverses and swaps
/// them. A transposed answer is the failure mode, so the fixture is rectangular in every axis.
#[test]
fn tensordot_col_major_vs_cpu() {
    require_cuda!();
    let mut d = DeviceCuda::new(0);
    d.set_default_order(ColMajor);
    let mut dcpu = DeviceCpuSerial::default();
    dcpu.set_default_order(ColMajor);

    let a_cpu: Tensor<f64, DeviceCpuSerial, _> =
        rt::asarray((exact_data(4 * 5 * 3, 78), &dcpu)).into_shape([4, 5, 3]);
    let b_cpu: Tensor<f64, DeviceCpuSerial, _> =
        rt::asarray((exact_data(3 * 6, 79), &dcpu)).into_shape([3, 6]);
    let (a, b) = (a_cpu.to_device(&d), b_cpu.to_device(&d));

    let got = host_nd(&rt::tensordot(&a, &b, ([2], [0])));
    assert_eq!(got.len(), 4 * 5 * 6, "a transposed answer would have the wrong extent");
    assert_eq!(got, host_nd(&rt::tensordot(&a_cpu, &b_cpu, ([2], [0]))), "col-major tensordot");
}

/// The boundary, pinned rather than discovered later: a `K` that is not one stride declines.
///
/// Contracting `a`'s axes `[0, 2]` of `(2, 3, 4)` leaves the two contracted axes with a hole between
/// them in memory (strides `24` and `1`), so no single leading dimension spans `K`. That is the only
/// kind of case tier 0 refuses: an *orientation* is not a problem — a block with unit stride on `M`
/// instead of `N` is handed to cuBLAS as `OP_T`, which is why the leading-axis form in
/// [`tensordot_views_only_vs_cpu`] works.
///
/// The verdict is a property of shape and strides alone, so it reproduces without a GPU; the
/// companion audit in `probe/geomprobe` (bin `tensordot`) walks both sides of it.
#[test]
fn tensordot_refuses_a_contraction_that_needs_a_pack() {
    require_cuda!();
    let d = DeviceCuda::new(0);
    let a: Tensor<f64, _, _> = rt::asarray((exact_data(2 * 3 * 4, 80), &d)).into_shape([2, 3, 4]);
    let b: Tensor<f64, _, _> = rt::asarray((exact_data(2 * 4 * 5, 81), &d)).into_shape([2, 4, 5]);

    match rt::tensordot_f(&a, &b, ([0, 2], [0, 1])) {
        Ok(_) => panic!("a K with a hole should be refused, but it computed a result"),
        Err(e) => {
            let msg = format!("{e}");
            assert!(msg.contains("packing pass"), "expected the pack reason, got: {msg}");
        },
    }
}

/* #endregion */

/* #region arg reductions + scan + var_with_dtype */

/// `var_with_dtype` and `std_with_dtype` on CUDA.
///
/// The `correction` is handed to the device, which computes the factor from the reduced size and
/// folds it in; the public wrappers carry no element bound and no host-closure bound.
///
/// The first assertion is the `None` path (no factor at all); the second pins `M / (M - correction)`
/// against the CPU.
#[test]
fn var_std_with_dtype_vs_cpu() {
    require_cuda!();
    let d = DeviceCuda::new(0);
    let dcpu = DeviceCpuSerial::default();

    let mut data = exact_data(6 * 8, 121);
    for (i, v) in data.iter_mut().enumerate() {
        *v += (i % 7) as f64 * 0.25; // keep the variance away from zero
    }
    let a_cpu: Tensor<f64, DeviceCpuSerial, _> = rt::asarray((data, &dcpu)).into_shape([6, 8]);
    let a = a_cpu.to_device(&d);

    // (1) no correction: must not need the correction machinery at all
    // The turbofish pins `TOut`: a `Tensor<f64, _, Ix1>` annotation would not, because the second
    // parameter is the backend and `_` there leaves the whole type ambiguous.
    let got_none = rt::var_with_dtype::<f64, f64, _, _>(&a, 1);
    let want_none = rt::var_with_dtype::<f64, f64, _, _>(&a_cpu, 1);
    assert_close(&host_nd(&got_none), &host_nd(&want_none), "var_with_dtype, correction None");

    // (2) with a correction: `M / (M - 1)`, applied by the device
    let args = |ap: AxesIndex<isize>| VarArgs { axes: ap, keepdims: false, correction: Some(1.0) };
    let got = rt::var_with_dtype::<f64, f64, _, _>(&a, args(AxesIndex::Val(1)));
    let want = rt::var_with_dtype::<f64, f64, _, _>(&a_cpu, args(AxesIndex::Val(1)));
    assert_close(&host_nd(&got), &host_nd(&want), "var_with_dtype, correction Some(1.0)");

    let got_std = rt::std_with_dtype::<f64, f64, _, _>(&a, args(AxesIndex::Val(1)));
    let want_std = rt::std_with_dtype::<f64, f64, _, _>(&a_cpu, args(AxesIndex::Val(1)));
    assert_close(&host_nd(&got_std), &host_nd(&want_std), "std_with_dtype, correction Some(1.0)");

    // (3) an explicit narrower accumulator, which is the reason the API exists at all.
    let got_f32 = rt::var_with_dtype::<f64, f32, _, _>(&a, 1);
    let want_f32 = rt::var_with_dtype::<f64, f32, _, _>(&a_cpu, 1);
    let got64: Vec<f64> = host_nd(&got_f32).into_iter().map(|v| v as f64).collect();
    let want64: Vec<f64> = host_nd(&want_f32).into_iter().map(|v| v as f64).collect();
    assert_close(&got64, &want64, "var_with_dtype<f32>");
}

/// `argmax`/`argmin`/`nanargmax`/`nanargmin`, compared against the CPU's own answer.
///
/// The comparison is the test: the index of an extremum is a *convention* (which of two equal values
/// wins, what a NaN does), and the only authority is what the CPU device returns. Ties and NaNs are
/// therefore placed deliberately rather than left to the data.
#[test]
fn arg_reductions_vs_cpu() {
    require_cuda!();
    let d = DeviceCuda::new(0);
    let dcpu = DeviceCpuSerial::default();

    // (1) plain data through both axes, plus the full reduction
    let (m, k) = (6usize, 32usize);
    let a_cpu: Tensor<f64, DeviceCpuSerial, _> = rt::asarray((exact_data(m * k, 131), &dcpu)).into_shape([m, k]);
    let a = a_cpu.to_device(&d);
    for (lbl, got, want) in [
        ("argmax axis 1", rt::argmax_axes(&a, 1), rt::argmax_axes(&a_cpu, 1)),
        ("argmax axis 0", rt::argmax_axes(&a, 0), rt::argmax_axes(&a_cpu, 0)),
        ("argmin axis 1", rt::argmin_axes(&a, 1), rt::argmin_axes(&a_cpu, 1)),
        ("argmin axis 0", rt::argmin_axes(&a, 0), rt::argmin_axes(&a_cpu, 0)),
    ] {
        assert_eq!(host_nd(&got), host_nd(&want), "{lbl}");
    }
    assert_eq!(rt::argmax(&a), rt::argmax(&a_cpu), "argmax over every axis");

    // (2) ties. `[3, 3, 1, 3]` has its maximum at three positions; the CPU keeps the smallest index,
    // and the kernel's lexicographic combine has to agree even though the tree visits blocks in an
    // order the fold never sees.
    let tie_cpu: Tensor<f64, DeviceCpuSerial, _> = rt::asarray((
        vec![3.0, 3.0, 1.0, 3.0, 1.0, 3.0, 3.0, 1.0, 2.0, 2.0, 2.0, 2.0],
        &dcpu,
    ))
    .into_shape([3, 4]);
    let tie = tie_cpu.to_device(&d);
    assert_eq!(
        host_nd(&rt::argmax_axes(&tie, 1)),
        host_nd(&rt::argmax_axes(&tie_cpu, 1)),
        "argmax must take the first of several equal maxima"
    );
    assert_eq!(
        host_nd(&rt::argmin_axes(&tie, 0)),
        host_nd(&rt::argmin_axes(&tie_cpu, 0)),
        "argmin must take the first of several equal minima"
    );

    // (3) NaN. The two policies disagree *on purpose*: a leading NaN wins for `argmax` and is
    // skipped for `nanargmax` (CPU: `f_comp_std_max` vs `f_comp_nan_max`). Both are pinned, because
    // "do the sane thing" would silently differ from the CPU on one of them.
    let nan_cpu: Tensor<f64, DeviceCpuSerial, _> = rt::asarray((
        vec![f64::NAN, 5.0, 1.0, 9.0, 5.0, f64::NAN, 9.0, f64::NAN, 2.0, 2.0, f64::NAN, 2.0],
        &dcpu,
    ))
    .into_shape([3, 4]);
    let nan = nan_cpu.to_device(&d);
    assert_eq!(
        host_nd(&rt::argmax_axes(&nan, 1)),
        host_nd(&rt::argmax_axes(&nan_cpu, 1)),
        "argmax with NaNs (a leading NaN wins)"
    );
    assert_eq!(
        host_nd(&rt::nanargmax_axes(&nan, 1)),
        host_nd(&rt::nanargmax_axes(&nan_cpu, 1)),
        "nanargmax with NaNs (NaN is skipped)"
    );
    assert_eq!(
        host_nd(&rt::nanargmin_axes(&nan, 1)),
        host_nd(&rt::nanargmin_axes(&nan_cpu, 1)),
        "nanargmin with NaNs"
    );

    // (4) an all-NaN slice: the CPU raises, so the kernel writes an out-of-range index and the impl
    // turns it into the same error.
    let all_nan_cpu: Tensor<f64, DeviceCpuSerial, _> =
        rt::asarray((vec![f64::NAN, f64::NAN, 1.0, 2.0], &dcpu)).into_shape([2, 2]);
    let all_nan = all_nan_cpu.to_device(&d);
    assert!(rt::nanargmax_axes_f(&all_nan_cpu, 1).is_err(), "cpu: all-NaN must raise");
    match rt::nanargmax_axes_f(&all_nan, 1) {
        Ok(_) => panic!("all-NaN slice should raise"),
        Err(e) => assert!(format!("{e}").contains("All-NaN"), "unexpected error: {e}"),
    }

    // (5) a contraction long enough that one output spans several warps: `n_out = 4` pushes `lanes`
    // to 256, which is the only way the shared-memory stage runs.
    let (m2, k2) = (4usize, 4096usize);
    let big_cpu: Tensor<f64, DeviceCpuSerial, _> = rt::asarray((exact_data(m2 * k2, 132), &dcpu)).into_shape([m2, k2]);
    let big = big_cpu.to_device(&d);
    assert_eq!(
        host_nd(&rt::argmax_axes(&big, 1)),
        host_nd(&rt::argmax_axes(&big_cpu, 1)),
        "argmax across the multi-warp stage"
    );
}

/// The index is raveled over the **logical** C-order of the contracted axes, so a contraction whose
/// memory order is not its logical order is refused rather than answered wrongly.
///
/// The two halves are the point:
///
/// - a *single* axis is always in logical order, however strided it is — `argmax(t, 1)` on a
///   transposed view has one contracted axis with stride 2, and must simply work;
/// - a *multi-axis* contraction can be out of order. Over a transposed `[2, 3]` view (strides
///   `[1, 2]`), the flat enumeration runs axis 0 fastest while the logical index runs axis 1
///   fastest. The reduce plan merges it happily — it sorts by stride, and `1*2 == 2` is a clean
///   arithmetic progression — so the step is 1 where the logical step is 2. Values would be right
///   and every index wrong, which is why this is checked rather than assumed.
#[test]
fn arg_refuses_a_contraction_not_in_logical_order() {
    require_cuda!();
    let d = DeviceCuda::new(0);
    let dcpu = DeviceCpuSerial::default();
    let n = 6usize;
    let base_cpu: Tensor<f64, DeviceCpuSerial, _> = rt::asarray((exact_data(n, 133), &dcpu)).into_shape([3, 2]);
    let base = base_cpu.to_device(&d);
    let (t_cpu, t) = (base_cpu.swapaxes(0, 1), base.swapaxes(0, 1));
    assert_eq!(t.layout().stride()[0], 1, "fixture: axis 0 must be the fast axis");

    // One contracted axis: strided, still logical. This is the case the first version of this test
    // accidentally asserted (and so proved nothing about).
    assert_eq!(
        host_nd(&rt::argmax_axes(&t, 1)),
        host_nd(&rt::argmax_axes(&t_cpu, 1)),
        "a single contracted axis is in logical order whatever its stride"
    );

    // Both axes: the enumeration order is the transpose of the logical order.
    match rt::argmax_f(&t) {
        Ok(v) => panic!("a transposed multi-axis contraction should be refused, got {v}"),
        Err(e) => {
            let msg = format!("{e}");
            assert!(msg.contains("not contiguous in C-order"), "unexpected error: {msg}");
        },
    }
    // ... and the documented way out works.
    let contig = t.to_contig(RowMajor);
    assert_eq!(rt::argmax(&contig), rt::argmax(&t_cpu), "to_contig should make it answerable");
}

/// `cumulative_sum`/`cumulative_prod` against the CPU, over the shapes that exercise each part of the
/// kernel:
///
/// - a 1-D scan, which is the trivial line split;
/// - axis 1 and axis 0 of a 2-D array, i.e. the innermost and the outermost axis (different strides
///   on both sides);
/// - a **middle** axis of a 3-D array, where a line's base offset is neither a row nor a column —
///   the case that only works because the kernel computes offsets from the rest layout instead of
///   assuming a merged stride;
/// - `include_initial`, which grows the axis and writes the identity at position 0;
/// - a scan longer than one block (`600 > 256`), which is the only fixture that runs the chunk
///   `carry` more than once;
/// - a transposed input, to check the two rest layouts really are read independently.
#[test]
fn cumulative_sum_prod_vs_cpu() {
    require_cuda!();
    let d = DeviceCuda::new(0);
    let dcpu = DeviceCpuSerial::default();

    // 1-D
    let v_cpu: Tensor<f64, DeviceCpuSerial, _> = rt::asarray((exact_data(9, 141), &dcpu));
    let v = v_cpu.to_device(&d);
    assert_close(
        &host_nd(&rt::cumulative_sum(&v, 0)),
        &host_nd(&rt::cumulative_sum(&v_cpu, 0)),
        "cumsum 1-D",
    );
    assert_close(
        &host_nd(&rt::cumulative_sum(&v, (0, true))),
        &host_nd(&rt::cumulative_sum(&v_cpu, (0, true))),
        "cumsum 1-D, include_initial",
    );

    // 2-D, both axes
    let (m, n) = (5usize, 6usize);
    let a_cpu: Tensor<f64, DeviceCpuSerial, _> = rt::asarray((exact_data(m * n, 142), &dcpu)).into_shape([m, n]);
    let a = a_cpu.to_device(&d);
    for axis in [0isize, 1] {
        for init in [false, true] {
            assert_close(
                &host_nd(&rt::cumulative_sum(&a, (axis, init))),
                &host_nd(&rt::cumulative_sum(&a_cpu, (axis, init))),
                &format!("cumsum 2-D axis {axis} include_initial={init}"),
            );
        }
    }

    // 3-D, middle axis
    let b_cpu: Tensor<f64, DeviceCpuSerial, _> = rt::asarray((exact_data(2 * 3 * 4, 143), &dcpu)).into_shape([2, 3, 4]);
    let b = b_cpu.to_device(&d);
    for axis in [0isize, 1, 2] {
        assert_close(
            &host_nd(&rt::cumulative_sum(&b, axis)),
            &host_nd(&rt::cumulative_sum(&b_cpu, axis)),
            &format!("cumsum 3-D axis {axis}"),
        );
    }

    // longer than one block: the chunk carry must be right, or everything past element 256 drifts
    let (m2, n2) = (3usize, 600usize);
    let long_cpu: Tensor<f64, DeviceCpuSerial, _> = rt::asarray((exact_data(m2 * n2, 144), &dcpu)).into_shape([m2, n2]);
    // scale down so the cumulative sum stays in a range where an f64 comparison is meaningful
    let long_cpu = long_cpu / 8.0;
    let long = long_cpu.to_device(&d);
    assert_close(
        &host_nd(&rt::cumulative_sum(&long, 1)),
        &host_nd(&rt::cumulative_sum(&long_cpu, 1)),
        "cumsum 600 elements (two chunks)",
    );

    // transposed input: the two rest layouts have different strides, so a shared assumption would fail
    let t_cpu = a_cpu.swapaxes(0, 1);
    let t = t_cpu.to_device(&d);
    assert_close(
        &host_nd(&rt::cumulative_sum(&t, 0)),
        &host_nd(&rt::cumulative_sum(&t_cpu, 0)),
        "cumsum of a transposed view",
    );

    // cumprod on small integers, so every partial product is exact
    let p_cpu: Tensor<f64, DeviceCpuSerial, _> = rt::asarray((
        (0..24).map(|i| 1.0 + (i % 3) as f64).collect::<Vec<f64>>(),
        &dcpu,
    ))
    .into_shape([4, 6]);
    let p = p_cpu.to_device(&d);
    for init in [false, true] {
        assert_close(
            &host_nd(&rt::cumulative_prod(&p, (1, init))),
            &host_nd(&rt::cumulative_prod(&p_cpu, (1, init))),
            &format!("cumprod include_initial={init}"),
        );
    }
}

/* #endregion */

/* #region owned binary operands (consume traits) */

/// `a + b` on **owned** tensors — the `OpLConsume*`/`OpRConsume*` family.
///
/// These had no CUDA impls at all, and nothing caught it: `&a + &b` goes through the out-of-place
/// traits and worked, so every existing test passed while the most natural expression in the crate
/// failed to compile. The census missed it too, because the CPU device declares these traits through
/// `#[duplicate_item]`, which hides both sides from a scan for `impl … for DeviceCpuSerial`.
///
/// The two directions are separate impls because they are not symmetric: the left-consuming form
/// writes `a = a op b` (destination is the first operand) and the right-consuming form writes
/// `b = a op b` (destination is the second). `sub` and `div` are the ones that expose a swap.
#[test]
fn owned_binary_operands_vs_cpu() {
    require_cuda!();
    let d = DeviceCuda::new(0);
    let dcpu = DeviceCpuSerial::default();

    let (m, n) = (5usize, 6usize);
    let a_cpu: Tensor<f64, DeviceCpuSerial, _> = rt::asarray((exact_data(m * n, 161), &dcpu)).into_shape([m, n]);
    let b_cpu: Tensor<f64, DeviceCpuSerial, _> = rt::asarray((exact_data(m * n, 162), &dcpu)).into_shape([m, n]);
    let (a, b) = (a_cpu.to_device(&d), b_cpu.to_device(&d));

    // left-consuming, commutative
    assert_eq!(
        host_nd(&(a.clone() + b.clone())),
        host_nd(&(a_cpu.clone() + b_cpu.clone())),
        "owned a + b"
    );
    // left-consuming, non-commutative: a swapped operand order would show here and nowhere else
    assert_eq!(
        host_nd(&(a.clone() - b.clone())),
        host_nd(&(a_cpu.clone() - b_cpu.clone())),
        "owned a - b"
    );
    // right-consuming: the scalar is the left operand, the tensor the destination
    assert_eq!(host_nd(&(2.0 - a.clone())), host_nd(&(2.0 - a_cpu.clone())), "owned 2.0 - a");
    assert_eq!(host_nd(&(3.0 * b.clone())), host_nd(&(3.0 * b_cpu.clone())), "owned 3.0 * b");
    // and the borrowed forms must still work — they are a different trait, and the new impls must not
    // have shadowed them
    assert_eq!(host_nd(&(&a * &b)), host_nd(&(&a_cpu * &b_cpu)), "borrowed a * b");
}

/* #endregion */

/* #region index_select + index_put */

/// `index_select` and `index_put`: one axis addressed indirectly, everything else copied.
///
/// The fixtures are chosen so that each part of the addressing is load-bearing:
///
/// - axis 1 and axis 0 of a 2-D array (different strides, different position in the index order);
/// - a **middle** axis of a 3-D array, where the indirect axis is neither first nor last;
/// - a transposed input, indexed along both its fast and its slow axis;
/// - a repeated index for the select — legal, and the CPU does it too;
/// - an in-place put, which must leave every unselected element untouched.
///
/// The error path is the other half of the contract: duplicate indices on a put are refused rather
/// than raced, because the CPU's answer (last write wins) needs an order that threads do not have.
#[test]
fn index_select_and_put_vs_cpu() {
    require_cuda!();
    let d = DeviceCuda::new(0);
    let dcpu = DeviceCpuSerial::default();

    let (m, n) = (5usize, 6usize);
    let a_cpu: Tensor<f64, DeviceCpuSerial, _> = rt::asarray((exact_data(m * n, 151), &dcpu)).into_shape([m, n]);
    let a = a_cpu.to_device(&d);

    // (1) along the inner axis, with a repeat and an out-of-order pick
    let idx = vec![2isize, 0, 2, 5];
    assert_eq!(
        host_nd(&rt::index_select(&a, 1, idx.clone())),
        host_nd(&rt::index_select(&a_cpu, 1, idx.clone())),
        "index_select axis 1 (repeat + reorder)"
    );

    // (2) along the outer axis
    let idx0 = vec![4isize, 1];
    assert_eq!(
        host_nd(&rt::index_select(&a, 0, idx0.clone())),
        host_nd(&rt::index_select(&a_cpu, 0, idx0.clone())),
        "index_select axis 0"
    );

    // (3) middle axis of a 3-D array
    let b_cpu: Tensor<f64, DeviceCpuSerial, _> = rt::asarray((exact_data(2 * 3 * 4, 152), &dcpu)).into_shape([2, 3, 4]);
    let b = b_cpu.to_device(&d);
    assert_eq!(
        host_nd(&rt::index_select(&b, 1, vec![2isize, 0])),
        host_nd(&rt::index_select(&b_cpu, 1, vec![2isize, 0])),
        "index_select, middle axis of a 3-D array"
    );

    // (4) a transposed source, indexed both ways. `t = aᵀ` has shape [n, m] and strides [1, n], so
    // axis 0 is the fast one and axis 1 the slow one: the two calls below are the two possible
    // orientations of an indirect axis over the same data.
    let t_cpu = a_cpu.swapaxes(0, 1);
    let t = t_cpu.to_device(&d);
    assert_eq!(t.layout().stride(), &[1isize, n as isize], "fixture: t = aᵀ");
    assert_eq!(
        host_nd(&rt::index_select(&t, 0, vec![5isize, 0, 3])),
        host_nd(&rt::index_select(&t_cpu, 0, vec![5isize, 0, 3])),
        "index_select on a transposed view, indirect axis contiguous"
    );
    assert_eq!(
        host_nd(&rt::index_select(&t, 1, vec![4isize, 1])),
        host_nd(&rt::index_select(&t_cpu, 1, vec![4isize, 1])),
        "index_select on a transposed view, indirect axis strided"
    );

    // (5) put: in place, and every unselected row must survive
    let put_idx = vec![1isize, 4, 0];
    let v_cpu: Tensor<f64, DeviceCpuSerial, _> = rt::asarray((exact_data(m * 3, 153), &dcpu)).into_shape([m, 3]);
    let v = v_cpu.to_device(&d);
    let mut want = a_cpu.clone();
    rt::index_put(&mut want, 1, put_idx.clone(), &v_cpu);
    let mut got = a_cpu.to_device(&d);
    rt::index_put(&mut got, 1, put_idx.clone(), &v);
    assert_eq!(host_nd(&got), host_nd(&want), "index_put axis 1");

    // (6) put along axis 0 of the contiguous array ...
    let v0_cpu: Tensor<f64, DeviceCpuSerial, _> = rt::asarray((exact_data(2 * n, 154), &dcpu)).into_shape([2, n]);
    let v0 = v0_cpu.to_device(&d);
    let mut want0 = a_cpu.clone();
    rt::index_put(&mut want0, 0, vec![3isize, 0], &v0_cpu);
    let mut got0 = a_cpu.to_device(&d);
    rt::index_put(&mut got0, 0, vec![3isize, 0], &v0);
    assert_eq!(host_nd(&got0), host_nd(&want0), "index_put axis 0");

    // (6b) ... and one where the destination is itself a strided view, so every write offset comes
    // from the view's strides instead of from a contiguous row. The previous fixture claimed this
    // coverage and did not have it: `a` is C-contiguous, so the destination was contiguous too.
    let v1_cpu: Tensor<f64, DeviceCpuSerial, _> = rt::asarray((exact_data(6 * 2, 156), &dcpu)).into_shape([6, 2]);
    let v1 = v1_cpu.to_device(&d);
    // `to_owned` gathers the visible elements, so the reference is contiguous — the *values* are
    // what is being compared, and the device side is the one that must cope with the view's strides.
    let mut want_t = t_cpu.to_owned();
    rt::index_put(&mut want_t, 1, vec![4isize, 1], &v1_cpu);
    let mut got_t = t_cpu.to_device(&d);
    rt::index_put(&mut got_t, 1, vec![4isize, 1], &v1);
    assert_eq!(host_nd(&got_t), host_nd(&want_t), "index_put into a strided destination");

    // (7) duplicates on a put: refused, with the index named
    let mut dup = a_cpu.to_device(&d);
    let v2_cpu: Tensor<f64, DeviceCpuSerial, _> = rt::asarray((exact_data(m * 2, 155), &dcpu)).into_shape([m, 2]);
    let v2 = v2_cpu.to_device(&d);
    match rt::index_put_f(&mut dup, 1, vec![2isize, 2], &v2) {
        Ok(()) => panic!("repeated indices should be refused"),
        Err(e) => {
            let msg = format!("{e}");
            assert!(msg.contains("more than once"), "unexpected error: {msg}");
        },
    }
}

/* #endregion */

/* #region compute type (TF32 / 16F / 16BF) */

/// Relative 2-norm error of an `f32` result against an `f64` reference.
///
/// Both sides come from the **same `f32` inputs**, so this measures the GEMM's arithmetic alone —
/// not the rounding of the operands, and not a difference of fixture. Scale-free by construction,
/// which is what lets the range test below use magnitudes of 1e5 with the same threshold.
fn rel_l2(got: &[f32], want: &[f64]) -> f64 {
    let num = got.iter().zip(want).map(|(g, w)| (*g as f64 - w).powi(2)).sum::<f64>().sqrt();
    let den = want.iter().map(|w| w * w).sum::<f64>().sqrt();
    num / den
}

/// The `RSTSR_CUDA_COMPUTE_TYPE_32F` flag must actually reach cuBLAS, and `Default` must stay exact.
///
/// The decisive assertion is **not** "TF32 is accurate enough" — that is a chemistry question for a
/// real system — but "TF32 is not silently ignored". If the flag never reached `cublasGemmEx`, both
/// runs would be bit-identical, and the experiment would read as *no speedup for no reason*, which
/// is the one failure mode that cannot be distinguished from "the hardware has no TF32".
///
/// The reference is f64 computed from the **same f32 operands**, so the difference measured is the
/// GEMM's arithmetic alone, not the rounding of the inputs. The values are `k/7`, which needs the
/// full 24-bit mantissa: integers would be exactly representable in TF32's 10 bits and the test
/// would pass no matter what the flag did.
#[test]
fn compute_type_reaches_cublas() {
    require_cuda!();
    let dcpu = DeviceCpuSerial::default();
    let (m, k, n) = (64usize, 512usize, 64usize);

    let gen = |len: usize, phase: usize| -> Vec<f32> {
        (0..len).map(|i| (((i * 7 + phase) % 23) as f32 - 11.0) / 7.0).collect()
    };
    let (av, bv) = (gen(m * k, 0), gen(k * n, 3));

    // Reference: f64 matmul of the f32 operands, promoted. Exact to f64 rounding.
    let a64: Tensor<f64, DeviceCpuSerial, _> =
        rt::asarray((av.iter().map(|v| *v as f64).collect::<Vec<f64>>(), &dcpu)).into_shape([m, k]);
    let b64: Tensor<f64, DeviceCpuSerial, _> =
        rt::asarray((bv.iter().map(|v| *v as f64).collect::<Vec<f64>>(), &dcpu)).into_shape([k, n]);
    let want: Vec<f64> = host_nd(&rt::matmul(&a64, &b64));

    let run = |flag: ComputeType32F| -> Vec<f32> {
        let mut d = DeviceCuda::new(0);
        // Per-device override, not the global: a global write here would perturb every other test
        // running in parallel, and the failure would land somewhere else entirely.
        d.set_flags_override(Some(CudaRuntimeFlags { compute_type_32f: flag }));
        let a: Tensor<f32, _, _> = rt::asarray((av.clone(), &d)).into_shape([m, k]);
        let b: Tensor<f32, _, _> = rt::asarray((bv.clone(), &d)).into_shape([k, n]);
        host_nd(&rt::matmul(&a, &b))
    };

    let rel_err = |got: &[f32], want: &[f64]| rel_l2(got, want);

    let e_exact = rel_err(&run(ComputeType32F::Default), &want);
    let e_tf32 = rel_err(&run(ComputeType32F::FastTF32), &want);

    assert!(e_exact < 1e-5, "f32 gemm on the Default path should be near-exact vs f64: {e_exact:e}");
    // 10-bit mantissa, ~4.9e-4 unit roundoff; accumulated over k=512 with mixed signs.
    assert!(e_tf32 < 1e-1, "TF32 should still be accurate to ~1e-4..1e-3: {e_tf32:e}");
    assert!(
        e_tf32 > 10.0 * e_exact,
        "TF32 and Default agree to within rounding ({e_exact:e} vs {e_tf32:e}) — the flag is not \
         reaching cuBLAS, so the compute-type experiment would measure nothing"
    );

    // The batched form is a *separate* entry point (`cublasGemmStridedBatchedEx`), and its strides
    // have their own history: a units bug there once left slice 0 exact and every later slice
    // unwritten. So it gets its own assertion, and the slices carry different data so a stride error
    // cannot hide behind repetition.
    let p = 3usize;
    let ap: Vec<f32> = (0..p).flat_map(|s| gen(m * k, 5 * s)).collect();
    let ap64: Vec<f64> = ap.iter().map(|v| *v as f64).collect();
    let a64b: Tensor<f64, DeviceCpuSerial, _> = rt::asarray((ap64, &dcpu)).into_shape([p, m, k]);
    let want_b: Vec<f64> = host_nd(&rt::matmul(&a64b, &b64));

    let run_b = |flag: ComputeType32F| -> Vec<f32> {
        let mut d = DeviceCuda::new(0);
        d.set_flags_override(Some(CudaRuntimeFlags { compute_type_32f: flag }));
        let a: Tensor<f32, _, _> = rt::asarray((ap.clone(), &d)).into_shape([p, m, k]);
        let b: Tensor<f32, _, _> = rt::asarray((bv.clone(), &d)).into_shape([k, n]);
        host_nd(&rt::matmul(&a, &b))
    };
    let eb_exact = rel_err(&run_b(ComputeType32F::Default), &want_b);
    let eb_tf32 = rel_err(&run_b(ComputeType32F::FastTF32), &want_b);
    assert!(eb_exact < 1e-5, "batched f32 gemm on the Default path: {eb_exact:e}");
    assert!(
        eb_tf32 > 10.0 * eb_exact,
        "batched TF32 not reaching cuBLAS ({eb_exact:e} vs {eb_tf32:e}) — the stride path is a \
         different call and needs its own check"
    );
}

/// The `FAST_*` compute types are a *licence*, not a switch: cuBLAS may compute in fp32 anyway.
///
/// The manual says "**Allows** the library to use Tensor Cores with automatic down-conversion",
/// for all three fast types. There is no promise that it will, no status bit to say which way it
/// went, and an exact answer is therefore a legal outcome — this test exists because the first
/// version of it assumed otherwise and failed on an A100 with a relative error of `4.7e-8`, i.e. a
/// perfectly exact fp32 matmul.
///
/// So the assertions are the ones the contract supports:
///
/// - **TF32 must engage**, at the shape where it is known to (`64x512x64`, the shape
///   [`compute_type_reaches_cublas`] uses). Without this control, "16F did not engage" cannot be
///   distinguished from "this shape never reaches cuBLAS at all".
/// - **16F and 16BF may engage or not**, and the test reports which, per shape. What it refuses to
///   accept is the middle: an answer that is neither exact nor predictably broken. That middle is
///   the only genuinely alarming regime — it would mean a conversion this crate cannot characterise.
/// - **If 16F engages, the range claim holds.** fp16 has a 5-bit exponent: it saturates at 65504 and
///   has nothing to say below ~6e-8, so on inputs at ~1e5 the result is `inf` (overflow) or off by
///   an O(1) factor (saturation), never "three digits worse".
///
/// The fixture is deliberately **all-positive** at ~1e5: additions then do not cancel, the sums sit
/// at `k*1e10`, and a measured error is the mantissa alone. A mixed-sign fixture would instead test
/// error amplification by cancellation, which is a property of the matrix, not of the compute type.
///
/// Run this with `--nocapture` to get the engagement map; it prints one line per (shape, flag).
#[test]
fn compute_type_engagement_is_the_librarys_choice() {
    require_cuda!();
    let dcpu = DeviceCpuSerial::default();

    // (m, k, n): the shape TF32 is known to engage at, and the small one that first showed this
    // behaviour. Engagement is size-dependent — that is the point of testing both.
    let shapes = [(64usize, 512usize, 64usize), (48, 64, 48)];
    let gen = |len: usize, phase: usize| -> Vec<f32> {
        (0..len).map(|i| 1.0e5 + ((i * 7 + phase) % 29) as f32).collect()
    };

    for (m, k, n) in shapes {
        let (av, bv) = (gen(m * k, 0), gen(k * n, 3));

        // f64 reference from the *same* f32 inputs. Every entry exceeds fp16's 65504 ceiling, which
        // is the whole point; it is nowhere near fp32's 3.4e38 ceiling.
        let a64: Tensor<f64, DeviceCpuSerial, _> =
            rt::asarray((av.iter().map(|v| *v as f64).collect::<Vec<f64>>(), &dcpu)).into_shape([m, k]);
        let b64: Tensor<f64, DeviceCpuSerial, _> =
            rt::asarray((bv.iter().map(|v| *v as f64).collect::<Vec<f64>>(), &dcpu)).into_shape([k, n]);
        let want: Vec<f64> = host_nd(&rt::matmul(&a64, &b64));

        let run = |flag: ComputeType32F| -> Vec<f32> {
            let mut d = DeviceCuda::new(0);
            d.set_flags_override(Some(CudaRuntimeFlags { compute_type_32f: flag }));
            let a: Tensor<f32, _, _> = rt::asarray((av.clone(), &d)).into_shape([m, k]);
            let b: Tensor<f32, _, _> = rt::asarray((bv.clone(), &d)).into_shape([k, n]);
            host_nd(&rt::matmul(&a, &b))
        };

        let e_exact = rel_l2(&run(ComputeType32F::Default), &want);
        assert!(e_exact < 1e-5, "the typed path should be near-exact vs f64: {e_exact:e}");
        // "Engaged" is defined against the typed path rather than an absolute number: any conversion
        // to a narrower format has to show up as a departure from fp32 rounding.
        let engaged = |e: f64| e > 10.0 * e_exact;

        // Control: TF32 is the one mode that is expected to engage, because on Ampere it is the
        // ordinary f32 path rather than an extra conversion. If it stops engaging at this shape the
        // rest of this test says nothing.
        let e_tf32 = rel_l2(&run(ComputeType32F::FastTF32), &want);
        eprintln!("[{m}x{k}x{n}] fasttf32: e={e_tf32:e} engaged={}", engaged(e_tf32));
        if (m, k, n) == (64, 512, 64) {
            assert!(
                engaged(e_tf32),
                "TF32 did not engage even at {m}x{k}x{n} ({e_exact:e} vs {e_tf32:e}) — the control \
                 for this test, so the compute type is not reaching cuBLAS at a shape where it \
                 already did"
            );
        }
        assert!(e_tf32 < 1e-1, "TF32 engaged but reads the operands wrongly: {e_tf32:e}");

        // bf16 keeps fp32's 8-bit exponent, so engagement can only cost mantissa digits.
        let e_bf = rel_l2(&run(ComputeType32F::Fast16BF), &want);
        eprintln!("[{m}x{k}x{n}] fast16bf: e={e_bf:e} engaged={}", engaged(e_bf));
        assert!(
            e_bf < 1e-1,
            "16BF is off by more than a mantissa ({e_bf:e}) — bf16 has fp32's exponent range, so \
             this cannot be a range effect"
        );

        // fp16 does not, and that is the difference between the two 16-bit modes.
        let got16f = run(ComputeType32F::Fast16F);
        let finite = got16f.iter().all(|v| v.is_finite());
        let e_16f = rel_l2(&got16f, &want);
        eprintln!(
            "[{m}x{k}x{n}] fast16f: e={e_16f:e} finite={finite} engaged={}",
            engaged(e_16f)
        );
        if engaged(e_16f) || !finite {
            assert!(
                !finite || e_16f.is_nan() || e_16f > 1e-2,
                "16F engaged and returned a *nearly* right answer ({e_16f:e}). A 5-bit exponent at \
                 ~1e5 magnitude cannot do that: the result should be inf or wrong by an O(1) \
                 factor. Something is converting the operands that this crate does not model"
            );
        }
        // Not engaging is allowed and is not a failure: the manual says "Allows the library to use",
        // so keeping fp32 is legal. That is exactly why this flag cannot be sold as a rate.
    }
}

/// The `OP_T` orientation — a `swapaxes`d operand, which is what `tensordot` hands cuBLAS when the
/// free axis is the unit-stride one.
///
/// cuBLAS is passed `transb = true` here and steps along the other axis, which selects a different
/// kernel family internally. The compute type is a per-call argument, so "TF32 works on the
/// contiguous orientation" says nothing about this one, and this is the orientation a real
/// contraction is more likely to produce.
///
/// The window is the assertion, and each bound rules out a different bug: **above** 10x the fp32
/// rounding means the `Ex` path really ran (a layout cuBLAS could not express, or a refused compute
/// type, would fall back to the tiled kernel and land back on the exact answer); **below** 1e-1
/// means the leading dimensions were read correctly — a wrong `ldb` on a transposed operand gives an
/// O(1) error, not a 1e-4 one.
#[test]
fn transposed_operand_still_honours_the_compute_type() {
    require_cuda!();
    let dcpu = DeviceCpuSerial::default();
    let (m, k, n) = (64usize, 512usize, 64usize);

    let gen = |len: usize, phase: usize| -> Vec<f32> {
        (0..len).map(|i| (((i * 7 + phase) % 23) as f32 - 11.0) / 7.0).collect()
    };
    // `a_c` is contiguous k x m, so `a_c.swapaxes(0, 1)` is m x k with strides (1, m): unit stride
    // on the M axis, i.e. the `OP_T` side.
    let (acv, bv) = (gen(k * m, 0), gen(k * n, 3));

    let ac64: Tensor<f64, DeviceCpuSerial, _> =
        rt::asarray((acv.iter().map(|v| *v as f64).collect::<Vec<f64>>(), &dcpu)).into_shape([k, m]);
    let b64: Tensor<f64, DeviceCpuSerial, _> =
        rt::asarray((bv.iter().map(|v| *v as f64).collect::<Vec<f64>>(), &dcpu)).into_shape([k, n]);
    let at64 = ac64.swapaxes(0, 1);
    let want: Vec<f64> = host_nd(&rt::matmul(&at64, &b64));

    let run = |flag: ComputeType32F| -> Vec<f32> {
        let mut d = DeviceCuda::new(0);
        d.set_flags_override(Some(CudaRuntimeFlags { compute_type_32f: flag }));
        let ac: Tensor<f32, _, _> = rt::asarray((acv.clone(), &d)).into_shape([k, m]);
        let b: Tensor<f32, _, _> = rt::asarray((bv.clone(), &d)).into_shape([k, n]);
        let at = ac.swapaxes(0, 1);
        host_nd(&rt::matmul(&at, &b))
    };

    let e_exact = rel_l2(&run(ComputeType32F::Default), &want);
    let e_tf32 = rel_l2(&run(ComputeType32F::FastTF32), &want);

    assert!(e_exact < 1e-5, "the transposed orientation should be near-exact on the typed path: {e_exact:e}");
    assert!(
        e_tf32 < 1e-1,
        "TF32 on the transposed orientation is wrong, not merely inaccurate ({e_tf32:e}) — the \
         leading dimension of the transposed operand is being read along the wrong axis"
    );
    assert!(
        e_tf32 > 10.0 * e_exact,
        "the transposed orientation left the gemmEx path ({e_exact:e} vs {e_tf32:e}) — this layout \
         was not expressible to cuBLAS, or the compute type was refused, and it fell through to the \
         tiled kernel where the flag cannot apply"
    );
}

/// `Pedantic` is the other `Ex` entry, and it must not be a synonym for fast.
///
/// `32F_PEDANTIC` and `32F_FAST_TF32` are the two flags that differ *only* in intent — same entry
/// point, same dtype, adjacent constants — so a swapped mapping would show up as "quietly three
/// digits worse" rather than as a failure. The fixture is the one from
/// [`compute_type_reaches_cublas`]: `k/7` values with mixed signs, so TF32 has to be visibly worse
/// there and Pedantic must not be.
///
/// The bound is `100x` rather than `10x` on purpose: dropping the FMA contraction rounds each
/// multiply and each add separately, which can cost up to a factor of two or four — anything beyond
/// two orders of magnitude is a different arithmetic regime, not a different rounding.
///
/// A *refusal* is undetectable here, because the fallback is exact and therefore passes. That is
/// acceptable: the claim this test defends is "Pedantic never becomes silently fast", and a fallback
/// preserves it.
#[test]
fn pedantic_keeps_fp32_accuracy() {
    require_cuda!();
    let dcpu = DeviceCpuSerial::default();
    let (m, k, n) = (64usize, 512usize, 64usize);

    let gen = |len: usize, phase: usize| -> Vec<f32> {
        (0..len).map(|i| (((i * 7 + phase) % 23) as f32 - 11.0) / 7.0).collect()
    };
    let (av, bv) = (gen(m * k, 0), gen(k * n, 3));

    let a64: Tensor<f64, DeviceCpuSerial, _> =
        rt::asarray((av.iter().map(|v| *v as f64).collect::<Vec<f64>>(), &dcpu)).into_shape([m, k]);
    let b64: Tensor<f64, DeviceCpuSerial, _> =
        rt::asarray((bv.iter().map(|v| *v as f64).collect::<Vec<f64>>(), &dcpu)).into_shape([k, n]);
    let want: Vec<f64> = host_nd(&rt::matmul(&a64, &b64));

    let run = |flag: ComputeType32F| -> Vec<f32> {
        let mut d = DeviceCuda::new(0);
        d.set_flags_override(Some(CudaRuntimeFlags { compute_type_32f: flag }));
        let a: Tensor<f32, _, _> = rt::asarray((av.clone(), &d)).into_shape([m, k]);
        let b: Tensor<f32, _, _> = rt::asarray((bv.clone(), &d)).into_shape([k, n]);
        host_nd(&rt::matmul(&a, &b))
    };

    let e_exact = rel_l2(&run(ComputeType32F::Default), &want);
    let e_pedantic = rel_l2(&run(ComputeType32F::Pedantic), &want);

    assert!(e_exact < 1e-5, "the typed path should be near-exact vs f64: {e_exact:e}");
    assert!(e_pedantic < 1e-5, "Pedantic is IEEE fp32 and must stay there: {e_pedantic:e}");
    assert!(
        e_pedantic < 100.0 * e_exact,
        "Pedantic ({e_pedantic:e}) is a different arithmetic regime from the typed path \
         ({e_exact:e}) — `pedantic` is mapped to a fast compute type"
    );
}

/// `f64` must ignore every compute type, bit for bit.
///
/// The pure test asserts the decision (`entry_for::<f64>` is always `Typed`); this asserts the
/// consequence on hardware. The failure it guards against is the one that cannot raise: a `32F`
/// compute type against `CUDA_R_64F` operands makes cuBLAS read four bytes of an eight-byte value,
/// so the answer would be plausible-looking nonsense with a success status.
///
/// One device, one pair of buffers, the flag the only thing that changes — so a difference here
/// cannot come from a different handle, a different algorithm choice, or a different copy of the
/// data. Comparison is on the bit patterns, because `==` would call `+0.0` and `-0.0` equal.
#[test]
fn f64_is_bit_identical_under_every_compute_type() {
    require_cuda!();
    let (m, k, n) = (64usize, 128usize, 64usize);

    let gen = |len: usize, phase: usize| -> Vec<f64> {
        (0..len).map(|i| (((i * 7 + phase) % 23) as f64 - 11.0) / 7.0).collect()
    };
    let (av, bv) = (gen(m * k, 0), gen(k * n, 3));

    let mut d = DeviceCuda::new(0);
    let a: Tensor<f64, _, _> = rt::asarray((av, &d)).into_shape([m, k]);
    let b: Tensor<f64, _, _> = rt::asarray((bv, &d)).into_shape([k, n]);

    let run = |d: &mut DeviceCuda, flag: ComputeType32F| -> Vec<u64> {
        d.set_flags_override(Some(CudaRuntimeFlags { compute_type_32f: flag }));
        host_nd(&rt::matmul(&a, &b)).iter().map(|v| v.to_bits()).collect()
    };

    let base = run(&mut d, ComputeType32F::Default);
    for flag in [
        ComputeType32F::Pedantic,
        ComputeType32F::FastTF32,
        ComputeType32F::Fast16F,
        ComputeType32F::Fast16BF,
    ] {
        assert_eq!(
            run(&mut d, flag),
            base,
            "{flag:?} changed an f64 matmul. A 32F compute type against 64-bit operands is a \
             misread, not a loss of precision — the flag must be ignored for f64"
        );
    }
}

/* #endregion */
