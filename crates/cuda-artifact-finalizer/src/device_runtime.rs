/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! The CUDA device runtime archive (`libcudadevrt.a`).
//!
//! Device-side CUDA runtime APIs (graph launch and node updates, device
//! memory management, ...) are defined in this archive and resolved at the
//! final device link. Without it nvJitLink reports success but drops every
//! kernel of a module that calls them; with it the calls link.
//!
//! The archive must come from the same toolkit as the linker, so it is taken
//! from the directory of the loaded nvJitLink library, where every CUDA
//! toolkit layout installs both. `CUDA_OXIDE_CUDADEVRT` names an archive
//! explicitly instead.

use crate::FinalizerError;
use crate::provenance::StableDigest;
use std::path::{Path, PathBuf};

/// Environment variable naming the device runtime archive explicitly.
pub const CUDA_DEVICE_RUNTIME_ENV: &str = "CUDA_OXIDE_CUDADEVRT";

const ARCHIVE_NAME: &str = "libcudadevrt.a";

/// The exact device runtime archive bytes a link uses. The digest is taken
/// over the same bytes the linker receives, so no file change between the two
/// can go unnoticed.
#[derive(Debug)]
pub struct DeviceRuntimeArchive {
    path: PathBuf,
    bytes: Vec<u8>,
    sha256: [u8; 32],
}

impl DeviceRuntimeArchive {
    /// Read the archive for the nvJitLink library loaded from `nvjitlink`.
    pub(crate) fn discover(nvjitlink: Option<&Path>) -> Result<Self, FinalizerError> {
        let explicit = std::env::var_os(CUDA_DEVICE_RUNTIME_ENV).map(PathBuf::from);
        let candidate = explicit.or_else(|| Some(nvjitlink?.parent()?.join(ARCHIVE_NAME)));
        let Some(path) = candidate else {
            return Err(FinalizerError::DeviceRuntimeNotFound {
                tried: format!(
                    "(the loaded nvJitLink library has no known file; set {CUDA_DEVICE_RUNTIME_ENV})"
                ),
            });
        };
        let bytes = std::fs::read(&path).map_err(|_| FinalizerError::DeviceRuntimeNotFound {
            tried: path.display().to_string(),
        })?;
        if bytes.is_empty() {
            return Err(FinalizerError::EmptyInput {
                name: path.display().to_string(),
            });
        }
        let sha256 = StableDigest::new()
            .field("cuda-device-runtime", &bytes)
            .finish();
        Ok(Self {
            path,
            bytes,
            sha256,
        })
    }

    /// Path the archive was read from.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Digest of the archive bytes the linker receives.
    pub fn sha256(&self) -> [u8; 32] {
        self.sha256
    }

    pub(crate) fn bytes(&self) -> &[u8] {
        &self.bytes
    }
}
