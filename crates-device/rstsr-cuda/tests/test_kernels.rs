//! Kernel-round tests: strided/offset assigns, cast assigns, sequence
//! creation, elementwise ops, stream mixing — all vs CPU reference
//! (copy-and-compare parity shape).

use rstsr::prelude::*;
use rstsr_cuda::funcs;
use rstsr_cuda::prelude_dev::*;
use std::sync::Arc;

/// D2H of a tensor's visible elements by layout gather (house helper).
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

/// Layout-ordered host values of an f64 tensor (N-d safe: the D2H goes
/// through the trusted CPU device, so strided views gather correctly).
fn host_nd<'a, Tr, B, D>(t: &TensorAny<Tr, f64, B, D>) -> Vec<f64>
where
    Tr: DataAPI<Data = <B as DeviceRawAPI<f64>>::Raw> + DataIntoCowAPI<'a>,
    D: DimAPI,
    B: DeviceAPI<f64> + DeviceToCpuAPI<f64> + DeviceCreationAnyAPI<f64> + OpAssignArbitaryAPI<f64, D, D>,
{
    let contig = t.to_contig(RowMajor);
    DeviceToCpuAPI::<f64>::to_cpu_vec(contig.storage()).rstsr_unwrap()
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
    let b = rt::arange((0.1, 16.1, 1., &d)).into_shape([4, 4]);
    let b_cpu = rt::arange((0.1, 16.1, 1., &dcpu)).into_shape([4, 4]);
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

/// The two shapes the backend refuses, each by name. Both are reachable through the public API, and
/// both must fail rather than fall back to the host or silently pick a different kernel.
#[test]
fn vecdot_refuses_unplannable_shapes() {
    require_cuda!();
    let d = DeviceCuda::new(0);

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

    // Rank-3 contracted on the last axis: the output part is two axes, which one `out_step` cannot
    // describe. This is missing capability, not a layout defect, and it is reported as such.
    let a3: Tensor<f64, _, _> = rt::arange((0., 4. * 8. * 8., 1., &d)).into_shape([4, 8, 8]);
    let b3: Tensor<f64, _, _> =
        rt::arange((1., (4. * 8. * 8.) + 1., 1., &d)).into_shape([4, 8, 8]);
    match rt::vecdot_f(&a3, &b3, 2) {
        Ok(_) => panic!("multi-axis output should be refused, but it computed a result"),
        Err(e) => {
            let msg = format!("{e}");
            assert!(msg.contains("not a single axis"), "expected the axis reason, got: {msg}");
        },
    }
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

    // (0, 2) contracted: the remaining part is the middle axis, so `n_contract` is 8 on both sides
    let got: Vec<f64> = host_vec1(&gpu.sum_axes([0isize, 2]));
    let want: Vec<f64> = host_vec1(&cpu.sum_axes([0isize, 2]));
    assert_eq!(got, want, "sum over (0, 2)");
    assert_eq!(got.len(), 3, "shape [2, 3, 4] reduced over (0, 2) leaves 3 elements");

    // keepdims is a host-side layout rewrite: the values must be untouched, the shape must not be
    let k = gpu.sum_with_args((-1isize, true));
    let k_cpu = cpu.sum_with_args((-1isize, true));
    assert_eq!(k.shape(), &[2, 3, 1]);
    assert_eq!(k.shape(), k_cpu.shape());
    assert_eq!(host_vec1(&k), host_vec1(&k_cpu), "keepdims values");

    // axes = None: 0-D output, built by a D2H + H2D round trip in the tensor tier
    let w = gpu.sum_with_args(());
    let w_cpu = cpu.sum_with_args(());
    assert_eq!(w.shape(), &[] as &[usize], "axes = None gives a 0-D tensor");
    assert_eq!(host_vec1(&w), host_vec1(&w_cpu), "axes = None values");
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

    // -0.0 vs +0.0: IEEE minimum keeps the negative zero, and so must the device, in either order
    for vals in [vec![-0.0f64, 0.0], vec![0.0f64, -0.0]] {
        let c: Tensor<f64, DeviceCpuSerial, _> = rt::asarray((vals, &dcpu));
        let g = c.to_device(&d);
        assert_eq!(g.min().to_bits(), c.min().to_bits(), "min(-0.0, +0.0) sign");
        assert_eq!(g.max().to_bits(), c.max().to_bits(), "max(-0.0, +0.0) sign");
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

    let got = host_vec1(&rt::sum_with_dtype::<f32, f64, _, _>(&a, ()))[0];
    let want = host_vec1(&rt::sum_with_dtype::<f32, f64, _, _>(&a_cpu, ()))[0];
    let in_f32 = host_vec1(&rt::sum_with_dtype::<f32, f32, _, _>(&a_cpu, ()))[0];

    assert_eq!(got, want, "f32 operand accumulated in f64");
    assert_ne!(
        got as f32, in_f32,
        "the test data must actually distinguish an f64 accumulator from an f32 one"
    );
}

/* #endregion */
