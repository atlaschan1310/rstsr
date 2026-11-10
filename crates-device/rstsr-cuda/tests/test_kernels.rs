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
        if !funcs::is_available() {
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

/// Deterministic f64 in `[-4, 4)` with a full mantissa.
fn vecdot_probe_data(n: usize, seed: u64) -> Vec<f64> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u = ((s >> 11) as f64) / ((1u64 << 53) as f64); // [0, 1)
            (u - 0.5) * 8.0 // [-4, 4)
        })
        .collect()
}

/// `(n_contract, n_out)` with the contraction on **axis 0** — `(i, p)` orientation.
///
/// Contracted stride is `n_out`, output stride is 1, so `flag_contig_s` is false and
/// `flag_contig_m` is true on both devices: one accumulator per output, sequential.
fn vecdot_contig_m_case(n_contract: usize, n_out: usize, seed: u64) {
    let d = DeviceCuda::new(0);
    // Pin the host device: `DeviceCpu` resolves to `DeviceFaer` under `faer_as_default`, whose
    // impls are feature-gated (same reason as in `assign_cast_pairs`).
    let dcpu = DeviceCpuSerial::default();

    let a_cpu: Tensor<f64, DeviceCpuSerial, _> =
        rt::asarray((vecdot_probe_data(n_contract * n_out, seed), &dcpu)).into_shape([n_contract, n_out]);
    let b_cpu: Tensor<f64, DeviceCpuSerial, _> =
        rt::asarray((vecdot_probe_data(n_contract * n_out, seed ^ 0x9e37_79b9), &dcpu))
            .into_shape([n_contract, n_out]);

    let want: Vec<f64> = host_vec1(&rt::vecdot(&a_cpu, &b_cpu, 0));

    let a = a_cpu.to_device(&d);
    let b = b_cpu.to_device(&d);
    let got: Vec<f64> = host_vec1(&rt::vecdot(&a, &b, 0));

    assert_eq!(got.len(), n_out, "output length wrong for (n_contract={n_contract}, n_out={n_out})");
    assert_eq!(
        got, want,
        "contig_m parity failed for ({n_contract}, {n_out}). \
         ~1e-16 relative => the kernel contracts to FMA or uses a different tree. \
         ~1e-6 relative => something is computing in f32 (check the @ADD@/@MUL@ mapping). \
         O(1) differences => addressing, not arithmetic."
    );
}

/// `(n_p, n_i)` with the contraction on the **last axis** — `(p, i)` orientation.
///
/// This is the layout ordinary C-order code produces, and it takes the *other* CPU branch:
/// `flag_contig_s` runs `unrolled_binary_reduce` over the contiguous run — eight congruence-class
/// accumulators, combined as `(p0+p4) (p1+p5) (p2+p6) (p3+p7)`, then a tail.
fn vecdot_contig_s_case(n_p: usize, n_i: usize, seed: u64) {
    let d = DeviceCuda::new(0);
    let dcpu = DeviceCpuSerial::default();

    let a_cpu: Tensor<f64, DeviceCpuSerial, _> =
        rt::asarray((vecdot_probe_data(n_p * n_i, seed), &dcpu)).into_shape([n_p, n_i]);
    let b_cpu: Tensor<f64, DeviceCpuSerial, _> =
        rt::asarray((vecdot_probe_data(n_p * n_i, seed ^ 0x5bf0_3635), &dcpu)).into_shape([n_p, n_i]);

    let want: Vec<f64> = host_vec1(&rt::vecdot(&a_cpu, &b_cpu, 1));

    let a = a_cpu.to_device(&d);
    let b = b_cpu.to_device(&d);
    let got: Vec<f64> = host_vec1(&rt::vecdot(&a, &b, 1));

    assert_eq!(got.len(), n_p, "output length wrong for (n_p={n_p}, n_i={n_i})");
    assert_eq!(
        got, want,
        "contig_s parity failed for ({n_p}, {n_i}). \
         Eight accumulators combined as (p0+p4),(p1+p5),(p2+p6),(p3+p7) then a tail; \
         a single-accumulator kernel differs here in the last bits, and a block reduction \
         differs in a way that depends on the block size."
    );
}

#[test]
fn vecdot_contig_m_vs_cpu() {
    require_cuda!();
    // an exact multiple of the 256-thread launch block, a ragged tail, and a
    // single-element contraction (where the CPU's branches degenerate to one multiply-add)
    for (n_contract, n_out) in [(1024usize, 256usize), (4096, 17), (1, 1000), (37, 300)] {
        vecdot_contig_m_case(n_contract, n_out, (n_contract as u64) * 131 + n_out as u64);
    }
}

#[test]
fn vecdot_contig_m_offset_view() {
    require_cuda!();
    let d = DeviceCuda::new(0);
    let dcpu = DeviceCpuSerial::default();
    let (n_contract, n_out) = (2048usize, 33usize);

    let a_cpu: Tensor<f64, DeviceCpuSerial, _> =
        rt::asarray((vecdot_probe_data(n_contract * n_out, 5), &dcpu)).into_shape([n_contract, n_out]);
    let b_cpu: Tensor<f64, DeviceCpuSerial, _> =
        rt::asarray((vecdot_probe_data(n_contract * n_out, 6), &dcpu)).into_shape([n_contract, n_out]);

    // Drop the first three contracted entries: offset != 0, strides unchanged, so this is still
    // contig_m. The launcher folds `las.offset()`/`lbs.offset()` into the base pointer; a
    // double-count or a dropped offset shows up HERE and nowhere else, as wrong numbers rather
    // than as an error.
    let a_view = a_cpu.i(3..);
    let b_view = b_cpu.i(3..);
    let want: Vec<f64> = host_vec1(&rt::vecdot(&a_view, &b_view, 0));

    let a = a_cpu.to_device(&d);
    let b = b_cpu.to_device(&d);
    let got: Vec<f64> = host_vec1(&rt::vecdot(&a.i(3..), &b.i(3..), 0));

    assert_eq!(got.len(), n_out);
    assert_eq!(
        got, want,
        "offset-view parity failed: the base pointer is off by a multiple of the element offset \
         (check whether dim_split_axes preserves the parent offset, and whether the launcher \
         then adds it a second time)"
    );
}

/// Independent oracle: recompute the contraction in plain Rust and compare bit-for-bit.
///
/// `vecdot_contig_m_case` compares GPU against the CPU *device*. This one compares against a
/// hand-written fold, so a defect shared by both backends (a promotion table, a shared helper, a
/// wrong layout convention) cannot hide. It is also the test that caught the `__f*_rn` /
/// `__d*_rn` overload: it prints nothing now, but if it fails, compute the same sum in `f32` and
/// compare — a match there means the arithmetic is single-precision.
#[test]
fn vecdot_contig_m_independent_oracle() {
    require_cuda!();
    let d = DeviceCuda::new(0);
    let dcpu = DeviceCpuSerial::default();
    let (n_contract, n_out) = (1024usize, 4usize);

    let a_cpu: Tensor<f64, DeviceCpuSerial, _> =
        rt::asarray((vecdot_probe_data(n_contract * n_out, 1), &dcpu)).into_shape([n_contract, n_out]);
    let b_cpu: Tensor<f64, DeviceCpuSerial, _> =
        rt::asarray((vecdot_probe_data(n_contract * n_out, 2), &dcpu)).into_shape([n_contract, n_out]);

    let a = a_cpu.to_device(&d);
    let b = b_cpu.to_device(&d);
    let got: Vec<f64> = host_vec1(&rt::vecdot(&a, &b, 0));

    // host_nd returns C-order, so column j of an (n_contract, n_out) tensor is j + m * n_out.
    let av = host_nd(&a_cpu);
    let bv = host_nd(&b_cpu);

    for j in 0..n_out {
        let exact: f64 = (0..n_contract).fold(0.0, |acc, m| acc + av[j + m * n_out] * bv[j + m * n_out]);
        assert_eq!(got[j], exact, "oracle mismatch at output {j}: the fold is m-ascending, one term at a time");
    }
}

#[test]
fn vecdot_contig_s_vs_cpu() {
    require_cuda!();
    // n_i must exceed 4 * BLOCK = 1024 to pass the classifier's contig_s gate.
    //   4096: exact multiple of 8, no tail
    //   4093: exercises the <=7 tail loop
    //   1024: exactly at the gate
    //      1: n_p = 1, a single output, still contig_s
    for (n_p, n_i) in [(37usize, 4096usize), (37, 4093), (11, 1024), (1, 2048)] {
        vecdot_contig_s_case(n_p, n_i, (n_p as u64) * 977 + n_i as u64);
    }
}

/// The classifier's gate, at `4 * BLOCK - 1`.
///
/// A run of 1023 is contiguous but shorter than four blocks, so `ContigS` must not be chosen for a
/// block-per-output kernel. Either a plan is picked and it is still bit-exact, or the call is
/// refused by name — what must never happen is a silent switch to a different accumulation order.
#[test]
fn vecdot_contig_s_gate_boundary() {
    require_cuda!();
    let d = DeviceCuda::new(0);
    let dcpu = DeviceCpuSerial::default();
    let (n_p, n_i) = (8usize, 1023usize);

    let a_cpu: Tensor<f64, DeviceCpuSerial, _> =
        rt::asarray((vecdot_probe_data(n_p * n_i, 3), &dcpu)).into_shape([n_p, n_i]);
    let b_cpu: Tensor<f64, DeviceCpuSerial, _> =
        rt::asarray((vecdot_probe_data(n_p * n_i, 4), &dcpu)).into_shape([n_p, n_i]);

    let a = a_cpu.to_device(&d);
    let b = b_cpu.to_device(&d);

    match rt::vecdot_f(&a, &b, 1) {
        Ok(c_gpu) => {
            let want: Vec<f64> = host_vec1(&rt::vecdot(&a_cpu, &b_cpu, 1));
            assert_eq!(host_vec1(&c_gpu), want, "short-run parity failed at the contig_s gate boundary");
        },
        Err(e) => {
            let msg = format!("{e}");
            assert!(
                msg.contains("contig_s") || msg.contains("general"),
                "unexpected error at the gate boundary: {e}"
            );
        },
    }
}

/// `General` is still unimplemented, and it has to say so by name.
///
/// Packed-then-reduced needs a pass that allocates two temporaries, and there is no pooling
/// allocator yet — so the policy question ("when do we double the caller's memory") is open. A
/// loud refusal is the correct behaviour; a silent host fallback or a scalar-loop kernel would be
/// 10-30x off peak with nothing in the output to say so.
#[test]
fn vecdot_general_is_loud() {
    require_cuda!();
    let d = DeviceCuda::new(0);

    // step 2 on both axes: neither a contiguous contracted run nor a contiguous output axis
    let r = SliceI::new(None::<isize>, None::<isize>, Some(2isize));
    let a: Tensor<f64, _, _> = rt::arange((0., (512. * 512.) as f64, 1., &d)).into_shape([512, 512]);
    let b: Tensor<f64, _, _> = rt::arange((1., (512. * 512.) as f64 + 1., 1., &d)).into_shape([512, 512]);

    match rt::vecdot_f(&a.i((r, r)), &b.i((r, r)), 0) {
        Ok(_) => panic!("general plan should be UnImplemented, but it computed a result"),
        Err(e) => assert!(
            format!("{e}").contains("general"),
            "general plan should be reported by name, got: {e}"
        ),
    }
}
/* #endregion */



/* #endregion */
