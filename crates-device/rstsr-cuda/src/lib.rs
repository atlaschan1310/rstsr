//! RSTSR CUDA device backend.
//!
//! - [`buffer::CudaBuffer`]: the raw storage handle (deep-copy `Clone`, logical length for
//!   `unique*` truncation, byte-erased device pointer).
//! - [`device::DeviceCuda`]: registry-canonical per-ordinal state, an execution-stream selector, a
//!   flags override, `default_order`.
//! - [`flags`]: `Option<CudaRuntimeFlags>` per device + env-seeded global default; compared as
//!   stored by `same_device`.
//! - [`stream::CudaStreamObj`]: user-visible, strictly compared.
//!
//! Execution model: every op is issued on the device value's stream and
//! returns immediately; host reads (`to_vec`, `to_scalar`, `Display`, ...)
//! synchronize explicitly. Buffer frees are stream-ordered on the buffer's
//! allocation stream.
//!
//! # Divergences from the CPU reference devices
//!
//! - `where` and `clip` are same-dtype only; the CPU devices promote scalar and mixed tensor
//!   operands through `DTypePromoteAPI`.
//! - The promoting binaries (`maximum`, `minimum`, `pow`, `atan2`, ...) are same-dtype only for the
//!   same reason. `ext_add`/`ext_sub`/`ext_mul`/`ext_div` do follow the CPU promotion table
//!   (`integer`/`float`, `real`/`complex` pairs).
//! - Transcendental unaries accept `f32`/`f64`/complex only; the CPU devices widen integer inputs
//!   to float.
//! - Float-to-integer casts use the C conversion rather than Rust's saturating `as` (`NaN -> 0` and
//!   out-of-range clamping differ); integer-to-float and width changes follow the CPU.
//! - Integer division and remainder by zero yield `0` (NumPy); the CPU reference devices panic on
//!   Rust's `/` and `%`.
//! - `fill`/`ones`/`zeros` take the kernel path for every supported dtype (complex included).

#![allow(clippy::needless_return)]

pub mod assignment;
pub mod buffer;
pub mod conversion;
pub mod cpu;
pub mod creation;
pub mod device;
pub mod dtype;
pub mod error;
pub mod flags;
pub mod format;
pub mod kernel;
pub mod linalg;
pub(crate) mod kernels;

pub(crate) mod kernels_ops;
pub mod operators;
pub mod reduce_ops;
pub mod registry;
pub mod stream;
pub mod transfer;

pub mod prelude_dev {
    pub use crate::buffer::CudaBuffer;
    pub use crate::device::DeviceCuda;
    pub use crate::flags::{default_flags, set_default_flags, ComputeType32F, CudaRuntimeFlags};
    pub use crate::registry::CudaOrdinalObj;
    pub use crate::stream::CudaStreamObj;
    pub use rstsr_core::prelude_dev::*;
}

/// Device-level free functions (`rt::cuda::*` candidates for the facade).
pub mod funcs {
    pub use crate::flags::{default_flags, set_default_flags};
    use crate::prelude_dev::*;
    use cudarc::driver::CudaContext;

    /// A device on ordinal `ordinal` (registry-canonical).
    pub fn device(ordinal: usize) -> DeviceCuda {
        DeviceCuda::new(ordinal)
    }

    /// Number of visible CUDA devices (0 if CUDA is unavailable).
    pub fn device_count() -> usize {
        CudaContext::device_count().unwrap_or(0).max(0) as usize
    }

    /// Whether any CUDA device is available.
    pub fn is_available() -> bool {
        device_count() > 0
    }

    /// `(free, total)` memory of `device` in bytes.
    pub fn memory_info(device: &DeviceCuda) -> Result<(usize, usize)> {
        device.memory_info()
    }

    /// Block until all work on `device`'s execution stream completes.
    pub fn synchronize(device: &DeviceCuda) -> Result<()> {
        device.synchronize()
    }
}
