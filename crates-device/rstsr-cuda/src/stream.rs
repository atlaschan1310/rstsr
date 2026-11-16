//! Stream wrapper for `DeviceCuda`.

use crate::error::{map_driver_error, map_driver_result};
use crate::prelude_dev::*;
use cudarc::cublas::CudaBlas;
use cudarc::driver::CudaContext;
use cudarc::driver::CudaStream;
use std::sync::{Arc, OnceLock};

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
    cublas_slot: OnceLock<Option<Arc<CudaBlas>>>,
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
    pub(crate) fn cublas(&self) -> Option<&Arc<CudaBlas>> {
        self.cublas_slot
            .get_or_init(|| {
                let made = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    CudaBlas::new(self.stream.clone())
                }));
                made.ok().and_then(|r| r.ok()).map(Arc::new)
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
