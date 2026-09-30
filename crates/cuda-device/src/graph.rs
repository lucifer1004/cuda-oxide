/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Device-side control of CUDA Graph conditional nodes.
//!
//! A conditional node (CUDA 12.3+) runs or skips its body graph according to
//! a value that kernels earlier in the same graph set on the device, so a
//! graph can take a data-dependent branch without returning to the host:
//!
//! ```text
//! selector kernel --set_conditional(handle, value)--> IF node: body runs iff value != 0
//! ```
//!
//! The host creates the handle for the graph (`cuGraphConditionalHandleCreate`)
//! and passes it to the selector as an ordinary kernel argument. The CUDA
//! driver resolves [`set_conditional`] when it loads the module; no device
//! runtime archive is linked.

/// A CUDA Graph conditional handle (`cudaGraphConditionalHandle`), passed to
/// a kernel as a 64-bit value.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ConditionalHandle(u64);

impl ConditionalHandle {
    /// Wrap the raw value of a handle the host created for the executing
    /// graph.
    pub const fn from_raw(raw: u64) -> Self {
        Self(raw)
    }

    /// The raw handle value.
    pub const fn raw(self) -> u64 {
        self.0
    }
}

unsafe extern "C" {
    fn cudaGraphSetConditional(handle: u64, value: u32);
}

/// Set the value of the conditional node `handle` names: nonzero runs an IF
/// node's body (or repeats a WHILE node's), zero skips it. The last value set
/// before the node executes decides it.
///
/// # Safety
///
/// - `handle` must have been created by `cuGraphConditionalHandleCreate` for
///   the graph whose launch is executing this kernel, in the same CUDA
///   context; a handle of another graph or context is undefined behavior.
/// - The kernel must run as a node of that graph that precedes the
///   conditional node. Outside a graph launch the call is undefined behavior.
/// - Threads that set the same handle race; the node sees one of the values.
///   Set it from one thread, or from threads that all set the same value.
#[inline(always)]
pub unsafe fn set_conditional(handle: ConditionalHandle, value: u32) {
    // SAFETY: forwarded from the caller.
    unsafe { cudaGraphSetConditional(handle.raw(), value) }
}
