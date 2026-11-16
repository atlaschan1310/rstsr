//! The tiled GEMM kernel, exercised on hardware.
//!
//! `cupy`-style delegation means cuBLAS is the default path, so the tiled kernel would otherwise
//! never run on a GPU again — and it is the path a build without `libcublas` takes. This file forces
//! it with `RSTSR_CUDA_MATMUL=tiled`, which is read once per process, and therefore sets the
//! variable in every test that runs a matmul.
//!
//! `tests/test_kernels.rs` covers the same shapes through whatever the default is, so between the
//! two files both paths have hardware evidence.

use rstsr::prelude::*;
use rstsr_cuda::funcs;
use rstsr_cuda::prelude_dev::*;

macro_rules! tiled_and_cuda {
    () => {
        // SAFETY: every test in this binary sets the same value before its first matmul, and the
        // flag is cached on first read, so there is no ordering under which they disagree.
        std::env::set_var("RSTSR_CUDA_MATMUL", "tiled");
        let available = std::panic::catch_unwind(funcs::is_available).unwrap_or(false);
        if !available {
            eprintln!("skipping: no CUDA device visible");
            return;
        }
    };
}

/// Integer-valued data: every product and partial sum is exact, so the comparison is `assert_eq!`.
fn exact_data(n: usize, seed: u64) -> Vec<f64> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 33) % 9) as f64 - 4.0
        })
        .collect()
}

/// Row-major host values of an f64 tensor of any rank.
fn host_nd<'a, Tr, B, D>(t: &TensorAny<Tr, f64, B, D>) -> Vec<f64>
where
    Tr: DataAPI<Data = <B as DeviceRawAPI<f64>>::Raw> + DataIntoCowAPI<'a>,
    D: DimAPI,
    B: DeviceAPI<f64> + DeviceToCpuAPI<f64> + DeviceCreationAnyAPI<f64> + OpAssignArbitaryAPI<f64, D, D>,
{
    let contig = t.to_contig(RowMajor);
    DeviceToCpuAPI::<f64>::to_cpu_vec(contig.storage()).rstsr_unwrap()
}

/// `[m, k] @ [k, n]`, including shapes that clip every tile edge.
#[test]
fn tiled_gemm22_vs_cpu() {
    tiled_and_cuda!();
    let d = DeviceCuda::new(0);
    let dcpu = DeviceCpuSerial::default();
    for &(m, k, n) in &[(32usize, 32usize, 32usize), (17, 33, 5), (130, 70, 129)] {
        let a_cpu: Tensor<f64, DeviceCpuSerial, _> =
            rt::asarray((exact_data(m * k, m as u64 + 1), &dcpu)).into_shape([m, k]);
        let b_cpu: Tensor<f64, DeviceCpuSerial, _> =
            rt::asarray((exact_data(k * n, n as u64 + 2), &dcpu)).into_shape([k, n]);
        let a = a_cpu.to_device(&d);
        let b = b_cpu.to_device(&d);
        assert_eq!(
            host_nd(&rt::matmul(&a, &b)),
            host_nd(&rt::matmul(&a_cpu, &b_cpu)),
            "({m} x {k}) @ ({k} x {n})"
        );
    }
}

/// Both non-default operand orientations, which select a different thread map each.
#[test]
fn tiled_transposed_operands() {
    tiled_and_cuda!();
    let d = DeviceCuda::new(0);
    let dcpu = DeviceCpuSerial::default();
    let (m, k, n) = (40usize, 24usize, 20usize);

    let a_base = rt::asarray((exact_data(k * m, 3), &dcpu)).into_shape([k, m]);
    let at_cpu = a_base.swapaxes(0, 1);
    let b_base = rt::asarray((exact_data(n * k, 4), &dcpu)).into_shape([n, k]);
    let bt_cpu = b_base.swapaxes(0, 1);
    let a0_cpu: Tensor<f64, DeviceCpuSerial, _> =
        rt::asarray((exact_data(m * k, 6), &dcpu)).into_shape([m, k]);
    let b0_cpu: Tensor<f64, DeviceCpuSerial, _> =
        rt::asarray((exact_data(k * n, 5), &dcpu)).into_shape([k, n]);

    let at = at_cpu.to_device(&d);
    let bt = bt_cpu.to_device(&d);
    let a0 = a0_cpu.to_device(&d);
    let b0 = b0_cpu.to_device(&d);

    assert_eq!(host_nd(&rt::matmul(&at, &b0)), host_nd(&rt::matmul(&at_cpu, &b0_cpu)), "a m-contig");
    assert_eq!(host_nd(&rt::matmul(&a0, &bt)), host_nd(&rt::matmul(&a0_cpu, &bt_cpu)), "b k-contig");
    assert_eq!(host_nd(&rt::matmul(&at, &bt)), host_nd(&rt::matmul(&at_cpu, &bt_cpu)), "both");
}

/// `grid.z` over the batch, plus a broadcast batch.
#[test]
fn tiled_batched_vs_cpu() {
    tiled_and_cuda!();
    let d = DeviceCuda::new(0);
    let dcpu = DeviceCpuSerial::default();
    let (p, m, k, n) = (5usize, 12usize, 16usize, 8usize);

    let a_cpu: Tensor<f64, DeviceCpuSerial, _> =
        rt::asarray((exact_data(p * m * k, 11), &dcpu)).into_shape([p, m, k]);
    let b_cpu: Tensor<f64, DeviceCpuSerial, _> =
        rt::asarray((exact_data(p * k * n, 12), &dcpu)).into_shape([p, k, n]);
    let a = a_cpu.to_device(&d);
    let b = b_cpu.to_device(&d);
    assert_eq!(host_nd(&rt::matmul(&a, &b)), host_nd(&rt::matmul(&a_cpu, &b_cpu)), "batched");

    let a2_cpu: Tensor<f64, DeviceCpuSerial, _> =
        rt::asarray((exact_data(m * k, 13), &dcpu)).into_shape([m, k]);
    let a2 = a2_cpu.to_device(&d);
    assert_eq!(
        host_nd(&rt::matmul(&a2, &b)),
        host_nd(&rt::matmul(&a2_cpu, &b_cpu)),
        "broadcast batch"
    );
}

/// `matmul_from`: the path that reads `c`, and the only one that hands the device a strided output.
#[test]
fn tiled_matmul_from_vs_cpu() {
    tiled_and_cuda!();
    let d = DeviceCuda::new(0);
    let dcpu = DeviceCpuSerial::default();
    let (m, k, n) = (20usize, 12usize, 16usize);

    let a_cpu: Tensor<f64, DeviceCpuSerial, _> =
        rt::asarray((exact_data(m * k, 21), &dcpu)).into_shape([m, k]);
    let b_cpu: Tensor<f64, DeviceCpuSerial, _> =
        rt::asarray((exact_data(k * n, 22), &dcpu)).into_shape([k, n]);
    let a = a_cpu.to_device(&d);
    let b = b_cpu.to_device(&d);

    let mut c: Tensor<f64, _, _> = rt::full(([m * n], 1.0f64, &d)).into_shape([m, n]);
    let mut c_cpu: Tensor<f64, DeviceCpuSerial, _> =
        rt::full(([m * n], 1.0f64, &dcpu)).into_shape([m, n]);
    rt::matmul_from(&mut c, &a, &b, 2.0f64, 3.0f64);
    rt::matmul_from(&mut c_cpu, &a_cpu, &b_cpu, 2.0f64, 3.0f64);
    assert_eq!(host_nd(&c), host_nd(&c_cpu), "c = 2*(a@b) + 3*c");
}

/// The two fixtures that used to be cuBLAS-only: `ColMajor`, and the length-1 axis of GEMV/GEVM.
///
/// This exists to make one comparison possible. Both paths share `resolve_geom` — the col-major
/// reversal, the operand swap, the batch steps, the degenerate-axis strides — and differ only in how
/// the geometry is consumed. So if these shapes pass here and fail through cuBLAS, the geometry is
/// not the difference. That is exactly what happened: `cuda matmul` failed on col-major, batched,
/// transposed and gevm while this file was green for the latter two, and a parallel-run corruption of
/// the shared cuBLAS handle was the only explanation left standing.
#[test]
fn tiled_col_major_and_gemv_gevm() {
    tiled_and_cuda!();

    // ColMajor: reverse every layout, swap the operands.
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
    assert_eq!(
        host_nd(&rt::matmul(&a, &b)),
        host_nd(&rt::matmul(&a_cpu, &b_cpu)),
        "col-major ({m} x {k}) @ ({k} x {n})"
    );

    // Rules 3 and 4: a length-1 tile axis, which is where the vector's own rank has to be folded
    // into a 2-D matmul part (`dim_insert`, i.e. a stride-1 axis of extent 1).
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
    assert_eq!(
        host_nd(&rt::matmul(&a, &v).to_contig(RowMajor)),
        host_nd(&rt::matmul(&a_cpu, &v_cpu).to_contig(RowMajor)),
        "gemv: [m,k] @ [k]"
    );
    assert_eq!(
        host_nd(&rt::matmul(&w, &c).to_contig(RowMajor)),
        host_nd(&rt::matmul(&w_cpu, &c_cpu).to_contig(RowMajor)),
        "gevm: [k] @ [k,n]"
    );
}
