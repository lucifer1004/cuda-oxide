/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Device-side CUDA Graph runtime declarations.

unsafe extern "C" {
    /// Select the body of a CUDA Graph conditional node from device code.
    ///
    /// `handle` must be a conditional handle created for the graph containing
    /// the calling kernel. CUDA interprets zero as false and every other value
    /// as true. The symbol is supplied by `libcudadevrt.a` during final device
    /// link.
    ///
    /// # Safety
    ///
    /// The caller must supply a live conditional handle belonging to the
    /// executing graph.
    #[allow(non_snake_case)]
    pub fn cudaGraphSetConditional(handle: u64, value: u32);
}
