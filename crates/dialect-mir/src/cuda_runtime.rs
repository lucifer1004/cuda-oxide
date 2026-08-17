/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! CUDA device-runtime link symbols recognized by the Rust device compiler.

/// Device-runtime symbol supplied by `libcudadevrt.a` at the final link.
pub const GRAPH_SET_CONDITIONAL_SYMBOL: &str = "cudaGraphSetConditional";
