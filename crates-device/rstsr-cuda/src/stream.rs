//! Stream wrapper for `DeviceCuda`.

use crate::error::{map_driver_error, map_driver_result};
use crate::prelude_dev::*;
use cudarc::cublas::CudaBlas;
use cudarc::driver::CudaContext;
use cudarc::driver::CudaStream;
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

/// A cuBLAS handle plus the lock that makes sharing it between threads legal.
///
/// # Why the lock is not optional
///
/// `cudarc` declares `unsafe impl Sync for CudaBlas`, so an `Arc<CudaBlas>` can be handed to any
/// number of threads and the compiler will not object — but a `cublasHandle_t` is **not**
/// thread-safe: two host threads entering `cublas*gemm` on one handle at the same time is undefined
/// behaviour, and it fails silently rather than with an error.
///
/// The sharing itself is deliberate, not accidental: [`crate::registry`] caches one
/// [`CudaStreamObj`] per ordinal, so every `DeviceCuda::new(0)` in a process runs on one stream. That
/// is a good default — a stream *is* thread-safe at the CUDA API level, so kernel launches and copies
/// from several threads order correctly — and it is why the tiled kernel never needed this while the
/// cuBLAS path did. The handle therefore needs its own serialization, and it is kept *inside* this
/// type so a caller cannot forget it: the guard is taken by
/// [`cublas_gemm::launch`][crate::kernels::cublas_gemm::launch], the only place that talks to cuBLAS.
///
/// Serializing the calls costs nothing measurable: `cublas*gemm` returns after enqueueing, and the
/// work it enqueued is on one stream, so it was going to be ordered anyway.
pub struct CublasHandle {
    blas: CudaBlas,
    /// Held for the duration of one cuBLAS API call. Not a device-side lock: it orders *host*
    /// entries into the handle, which is the only thing cuBLAS requires and the only thing it fails
    /// to provide itself.
    api: Mutex<()>,
}

impl CublasHandle {
    fn new(blas: CudaBlas) -> Self {
        Self { blas, api: Mutex::new(()) }
    }

    /// The raw handle. Callers must hold the guard from [`Self::lock`] around every use.
    pub(crate) fn blas(&self) -> &CudaBlas {
        &self.blas
    }

    /// Serialize one cuBLAS API call.
    pub(crate) fn lock(&self) -> MutexGuard<'_, ()> {
        self.api.lock().unwrap_or_else(|e| e.into_inner())
    }
}

impl std::fmt::Debug for CublasHandle {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("CublasHandle").finish_non_exhaustive()
    }
}

/// A CUDA stream of one ordinal. Identity is (handle, context); the device
/// value's stream participates in `same_device` strictly.
#[derive(Debug)]
pub struct CudaStreamObj {
    pub(crate) stream: Arc<CudaStream>,
    /// Cached ordinal of the owning context (for equality and display).
    pub(crate) ordinal: usize,
    /// cuBLAS handle bound to this stream, or a cached "not available".
    ///
    /// The handle must be per-stream: `cublasSetStream` is part of creating it, and a handle shared
    /// across streams would enqueue work on the wrong one. `None` is cached because probing means
    /// dlopening `libcublas`, which either works once or never — retrying per launch would turn a
    /// missing library into a per-call panic.
    cublas_slot: OnceLock<Option<Arc<CublasHandle>>>,
}

impl CudaStreamObj {
    /// Create a new stream on `ctx`.
    pub(crate) fn new(ctx: &Arc<CudaContext>) -> Result<Self> {
        let stream = ctx.new_stream().map_err(map_driver_error)?;
        Ok(Self { stream, ordinal: ctx.ordinal(), cublas_slot: OnceLock::new() })
    }

    /// The underlying stream handle (launches, copies, synchronization).
    pub fn inner(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    /// Wrap an existing cudarc stream of the given ordinal (test/interop
    /// helper; production code goes through `CudaOrdinalObj::new`).
    pub fn wrap(ordinal: usize, stream: Arc<CudaStream>) -> Self {
        Self { stream, ordinal, cublas_slot: OnceLock::new() }
    }

    /// The cuBLAS handle for this stream, or `None` when the library cannot be loaded.
    ///
    /// `cudarc` resolves cuBLAS symbols lazily through `libloading` and **panics** when a symbol is
    /// missing rather than returning an error, so a machine without `libcublas` would abort here.
    /// The probe is therefore wrapped in `catch_unwind`; the panic message still reaches stderr once
    /// per stream, which is the only signal the user gets that they are on the slow path.
    ///
    /// The returned handle is shared by every thread that uses this stream, and a `cublasHandle_t`
    /// is not thread-safe — take [`CublasHandle::lock`] before every call, or better, call
    /// [`cublas_gemm::launch`][crate::kernels::cublas_gemm::launch], which does it for you.
    pub(crate) fn cublas(&self) -> Option<&Arc<CublasHandle>> {
        self.cublas_slot
            .get_or_init(|| {
                let made = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    CudaBlas::new(self.stream.clone())
                }));
                made.ok().and_then(|r| r.ok()).map(|blas| Arc::new(CublasHandle::new(blas)))
            })
            .as_ref()
    }

    /// The ordinal of this stream's context.
    pub fn ordinal(&self) -> usize {
        self.ordinal
    }

    /// Block the host until all work submitted to this stream completes.
    pub fn synchronize(&self) -> Result<()> {
        map_driver_result(self.stream.synchronize())
    }
}

impl PartialEq for CudaStreamObj {
    fn eq(&self, other: &Self) -> bool {
        self.stream == other.stream
    }
}
impl Eq for CudaStreamObj {}
