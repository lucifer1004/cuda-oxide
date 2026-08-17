/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};
use thiserror::Error;

/// `libcudadevrt.a` could not be located in the selected CUDA toolkit.
#[derive(Debug, Error)]
#[error(
    "Could not locate libcudadevrt.a. Set CUDA_OXIDE_CUDADEVRT, CUDA_TOOLKIT_PATH, or CUDA_HOME, or install the CUDA Toolkit. Tried:\n  {tried}"
)]
pub struct CudaDeviceRuntimeNotFound {
    /// Newline-separated paths checked in discovery order.
    pub tried: String,
}

/// Locate the CUDA device-runtime archive used to resolve device-side runtime APIs.
pub fn find_cuda_device_runtime() -> Result<PathBuf, CudaDeviceRuntimeNotFound> {
    find_cuda_device_runtime_with(|name| std::env::var(name).ok(), Path::is_file)
}

/// Hash the exact CUDA device-runtime archive selected by discovery.
pub fn cuda_device_runtime_digest() -> Result<[u8; 32], crate::FinalizerError> {
    let path = find_cuda_device_runtime()?;
    let bytes =
        std::fs::read(&path).map_err(|source| crate::FinalizerError::Io { path, source })?;
    Ok(Sha256::digest(bytes).into())
}

fn find_cuda_device_runtime_with(
    env: impl Fn(&str) -> Option<String>,
    exists: impl Fn(&Path) -> bool,
) -> Result<PathBuf, CudaDeviceRuntimeNotFound> {
    let mut candidates = Vec::new();
    if let Some(path) = env("CUDA_OXIDE_CUDADEVRT") {
        candidates.push(PathBuf::from(path));
    }
    for root in ["CUDA_TOOLKIT_PATH", "CUDA_HOME", "CUDA_PATH"]
        .into_iter()
        .filter_map(&env)
        .map(PathBuf::from)
        .chain([PathBuf::from("/usr/local/cuda"), PathBuf::from("/opt/cuda")])
    {
        candidates.push(root.join("lib/libcudadevrt.a"));
        candidates.push(root.join("lib64/libcudadevrt.a"));
        for target in ["x86_64-linux", "aarch64-linux", "sbsa-linux"] {
            candidates.push(root.join("targets").join(target).join("lib/libcudadevrt.a"));
        }
    }
    candidates.dedup();
    candidates
        .iter()
        .find(|candidate| exists(candidate))
        .cloned()
        .ok_or_else(|| CudaDeviceRuntimeNotFound {
            tried: candidates
                .iter()
                .map(|path| path.display().to_string())
                .collect::<Vec<_>>()
                .join("\n  "),
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_archive_override_wins() {
        let found = find_cuda_device_runtime_with(
            |name| match name {
                "CUDA_OXIDE_CUDADEVRT" => Some("/chosen/libcudadevrt.a".to_owned()),
                "CUDA_TOOLKIT_PATH" => Some("/toolkit".to_owned()),
                _ => None,
            },
            |path| path == Path::new("/chosen/libcudadevrt.a"),
        )
        .unwrap();
        assert_eq!(found, Path::new("/chosen/libcudadevrt.a"));
    }

    #[test]
    fn target_directory_toolkit_layout_is_supported() {
        let found = find_cuda_device_runtime_with(
            |name| (name == "CUDA_TOOLKIT_PATH").then(|| "/toolkit".to_owned()),
            |path| path == Path::new("/toolkit/targets/x86_64-linux/lib/libcudadevrt.a"),
        )
        .unwrap();
        assert_eq!(
            found,
            Path::new("/toolkit/targets/x86_64-linux/lib/libcudadevrt.a")
        );
    }
}
