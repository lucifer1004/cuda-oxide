/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! The CUDA device runtime APIs the Rust device compiler admits.
//!
//! A device runtime API is an `extern "C"` function that no Rust crate
//! defines. The compiler recognizes exactly the functions listed here, each
//! with its exact C ABI and resolution route; any other extern stays an
//! ordinary external symbol that a link input must define.

/// A scalar in a device runtime API's C signature.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DeviceRuntimeScalar {
    /// `unsigned int` / `int`.
    Int32,
    /// A 64-bit integer or opaque handle.
    Int64,
}

impl DeviceRuntimeScalar {
    /// Bit width of the scalar.
    pub const fn bits(self) -> u32 {
        match self {
            Self::Int32 => 32,
            Self::Int64 => 64,
        }
    }
}

/// Where a device runtime API's definition comes from.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DeviceRuntimeResolution {
    /// The CUDA driver resolves the call when it loads the module, from PTX
    /// or from a cubin, so the call needs no output route or link input
    /// beyond the module's own code.
    Driver,
}

/// One admitted device runtime API.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DeviceRuntimeApi {
    /// C symbol name.
    pub symbol: &'static str,
    /// Parameter scalars, in order.
    pub params: &'static [DeviceRuntimeScalar],
    /// Return scalar, or `None` for `void`.
    pub returns: Option<DeviceRuntimeScalar>,
    /// Where its definition comes from.
    pub resolution: DeviceRuntimeResolution,
    /// First CUDA release, as `(major, minor)`, whose toolkit and driver
    /// provide it.
    pub since_cuda: (u32, u32),
}

/// `void cudaGraphSetConditional(cudaGraphConditionalHandle, unsigned int)`.
pub const GRAPH_SET_CONDITIONAL: DeviceRuntimeApi = DeviceRuntimeApi {
    symbol: "cudaGraphSetConditional",
    params: &[DeviceRuntimeScalar::Int64, DeviceRuntimeScalar::Int32],
    returns: None,
    resolution: DeviceRuntimeResolution::Driver,
    since_cuda: (12, 3),
};

/// Every admitted device runtime API.
pub const DEVICE_RUNTIME_APIS: &[DeviceRuntimeApi] = &[GRAPH_SET_CONDITIONAL];

/// The admitted device runtime API named `symbol`.
pub fn device_runtime_api(symbol: &str) -> Option<&'static DeviceRuntimeApi> {
    DEVICE_RUNTIME_APIS.iter().find(|api| api.symbol == symbol)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn apis_are_found_by_exact_symbol_only() {
        assert_eq!(
            device_runtime_api("cudaGraphSetConditional"),
            Some(&GRAPH_SET_CONDITIONAL)
        );
        assert_eq!(device_runtime_api("cudaGraphSetConditional_ptsz"), None);
        assert_eq!(device_runtime_api("cudaMalloc"), None);
        for (index, api) in DEVICE_RUNTIME_APIS.iter().enumerate() {
            assert!(
                DEVICE_RUNTIME_APIS[..index]
                    .iter()
                    .all(|other| other.symbol != api.symbol)
            );
        }
    }
}
