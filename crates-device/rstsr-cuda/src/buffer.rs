//! Raw device buffer.

use crate::error::{map_driver_error, map_driver_result_alloc};
use crate::prelude_dev::*;
use core::marker::PhantomData;
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use std::sync::Arc;

/// Owning device buffer of `len` elements of `T`; `DeviceRawAPI::Raw` for
/// `DeviceCuda`.
///
/// - `Clone` is a **deep device-to-device copy** on the allocation stream (value semantics:
///   `Tensor::clone` routes through `Raw::clone`).
/// - `len` is the *logical* element count and may be shrunk by `truncate` (the `unique*` family
///   relies on this); the underlying block keeps its capacity until drop.
/// - The allocation stream (inside `CudaSlice`) owns free/reuse ordering; it is not the same as the
///   device value's execution stream.
/// - Layout-identical to `CudaBuffer<MaybeUninit<T>>` (all `T`-dependence is behind `PhantomData`),
///   so the tensor layer's `Raw<T> <-> Raw<MaybeUninit<T>>` transmutes and the hand-written
///   `assume_init_impl` are sound.
/// - Allocation is byte-based through `CudaSlice<u8>`; the POD (`Copy`) restriction is enforced
///   where bytes cross the host boundary, not here.
pub struct CudaBuffer<T> {
    pub(crate) slice: CudaSlice<u8>,
    pub(crate) len: usize,
    pub(crate) _m: PhantomData<*const T>,
}

impl<T> CudaBuffer<T> {
    /// Allocate an uninitialized buffer of `len` elements on `stream`
    /// (infallible: driver failure panics through the RSTSR error mapping).
    pub fn new(stream: &Arc<CudaStreamObj>, len: usize) -> Self {
        Self::new_f(stream, len).rstsr_unwrap()
    }

    /// Fallible allocation of an uninitialized buffer.
    pub fn new_f(stream: &Arc<CudaStreamObj>, len: usize) -> Result<Self> {
        let byte_len = len
            .checked_mul(core::mem::size_of::<T>())
            .ok_or_else(|| rstsr_error!(MemoryError, "CUDA buffer length overflow: {len} elements"))?;
        let slice = if byte_len == 0 {
            stream.inner().null().map_err(map_driver_error)?
        } else {
            let res = unsafe { stream.inner().alloc::<u8>(byte_len) };
            map_driver_result_alloc(res, byte_len)?
        };
        Ok(Self { slice, len, _m: PhantomData })
    }

    /// Logical element count.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the logical length is zero.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Shrink the logical length (capacity untouched). Used by `unique*`.
    pub fn truncate(&mut self, len: usize) {
        self.len = self.len.min(len);
    }

    /// Base device pointer (erased). Never dereference on the host; this is
    /// the address device kernels and `get_index_ptr` operate on.
    pub fn device_ptr(&self) -> *const T {
        let (ptr, _record) = DevicePtr::<u8>::device_ptr(self.as_bytes(), self.slice.stream());
        ptr as *const T
    }

    /// Base device pointer with the read event recorded on `exec` — the
    /// stream that will actually read the buffer. Kernel argument paths
    /// must use this form so the allocation stream's free waits for the
    /// consuming launch (the plain accessor records on the allocation
    /// stream, which does not cover cross-stream readers).
    pub fn device_ptr_on(&self, exec: &Arc<CudaStreamObj>) -> *const T {
        self.device_ptr_raw_on(exec.inner())
    }

    /// `device_ptr_on` over a raw cudarc stream (memcpy fast paths).
    pub(crate) fn device_ptr_raw_on(&self, stream: &Arc<cudarc::driver::CudaStream>) -> *const T {
        let (ptr, _record) = DevicePtr::<u8>::device_ptr(self.as_bytes(), stream);
        ptr as *const T
    }

    /// Mutable base device pointer (same caveats as [`Self::device_ptr`]).
    pub fn device_ptr_mut(&mut self) -> *mut T {
        let stream = self.as_bytes().stream().clone();
        let (ptr, _record) = DevicePtrMut::<u8>::device_ptr_mut(self.as_mut_bytes(), &stream);
        ptr as *mut T
    }

    /// The stream this buffer was allocated on (owns free/reuse ordering).
    pub fn alloc_stream(&self) -> Arc<CudaStreamObj> {
        let stream = self.slice.stream();
        Arc::new(CudaStreamObj::wrap(stream.context().ordinal(), stream.clone()))
    }

    /// Take ownership of the underlying allocation as a raw device pointer
    /// (the wrapper's tracking events are dropped; the caller must re-wrap
    /// or free — used by `assume_init_impl`).
    pub(crate) fn leak_ptr(self) -> cudarc::driver::sys::CUdeviceptr {
        self.slice.leak()
    }

    /// Borrow as a raw byte slice (for memcpy sources).
    pub(crate) fn as_bytes(&self) -> &CudaSlice<u8> {
        &self.slice
    }

    /// Borrow as a mutable raw byte slice (for memcpy destinations).
    pub(crate) fn as_mut_bytes(&mut self) -> &mut CudaSlice<u8> {
        &mut self.slice
    }
}

impl<T> Clone for CudaBuffer<T> {
    /// Deep D2D copy on the allocation stream (value semantics).
    /// Panics through the RSTSR error mapping on driver failure (same class
    /// as `Vec::clone` on OOM).
    fn clone(&self) -> Self {
        self.clone_f().rstsr_unwrap()
    }
}

impl<T> CudaBuffer<T> {
    /// Fallible deep copy.
    pub fn clone_f(&self) -> Result<Self> {
        if self.len == 0 {
            let slice = self.slice.stream().null().map_err(map_driver_error)?;
            return Ok(Self { slice, len: 0, _m: PhantomData });
        }
        let slice = self.slice.try_clone().map_err(map_driver_error)?;
        Ok(Self { slice, len: self.len, _m: PhantomData })
    }
}

impl<T> core::fmt::Debug for CudaBuffer<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("CudaBuffer").field("len", &self.len).field("ptr", &self.device_ptr()).finish()
    }
}

// # Safety
//
// The buffer is a plain device allocation; the data is only ever touched by
// device-side operations (kernels, memcpys) through the owning stream. The
// `PhantomData<*const T>` poison is deliberate (no host aliasing), and the
// manual impls are the same pattern cudarc uses for `CudaSlice`.
unsafe impl<T: Send> Send for CudaBuffer<T> {}
unsafe impl<T: Sync> Sync for CudaBuffer<T> {}
