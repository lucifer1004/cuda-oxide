/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! What a module's final device link must include beyond its own code.

use crate::options::FinalizationOptions;

/// First line of a `<module>.requires` sidecar. Each following line names one
/// requirement; a reader rejects a name it does not know, so an older reader
/// fails instead of linking without something the module needs.
pub const LINK_REQUIREMENTS_SIDECAR_HEADER: &str = "cuda-oxide link-requirements v1";

const DEVICE_RUNTIME: &str = "device-runtime";

/// The final-link requirements of one module.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct LinkRequirements {
    /// The module calls CUDA device runtime APIs, which resolve against the
    /// toolkit's device runtime archive.
    pub device_runtime: bool,
}

impl LinkRequirements {
    /// Text of a `<module>.requires` sidecar recording these requirements.
    pub fn sidecar_text(self) -> String {
        let mut text = format!("{LINK_REQUIREMENTS_SIDECAR_HEADER}\n");
        if self.device_runtime {
            text.push_str(DEVICE_RUNTIME);
            text.push('\n');
        }
        text
    }

    /// Requirements recorded by a `<module>.requires` sidecar, or `None` when
    /// it has another header or names a requirement this reader does not know.
    pub fn parse_sidecar(text: &str) -> Option<Self> {
        let mut lines = text.lines();
        if lines.next()? != LINK_REQUIREMENTS_SIDECAR_HEADER {
            return None;
        }
        let mut requirements = Self::default();
        for line in lines {
            match line {
                DEVICE_RUNTIME if !requirements.device_runtime => {
                    requirements.device_runtime = true;
                }
                _ => return None,
            }
        }
        Some(requirements)
    }

    /// Finalization options that satisfy these requirements.
    #[must_use]
    pub fn apply(self, options: FinalizationOptions) -> FinalizationOptions {
        options.with_device_runtime(self.device_runtime)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn requirements_round_trip_and_unknown_ones_fail_closed() {
        for requirements in [
            LinkRequirements::default(),
            LinkRequirements {
                device_runtime: true,
            },
        ] {
            assert_eq!(
                LinkRequirements::parse_sidecar(&requirements.sidecar_text()),
                Some(requirements)
            );
        }
        let header = LINK_REQUIREMENTS_SIDECAR_HEADER;
        for rejected in [
            String::new(),
            "device-runtime\n".to_string(),
            format!("{header}\ndevice-runtime\ndevice-runtime\n"),
            format!("{header}\na-future-requirement\n"),
            format!("{header}\n\n"),
        ] {
            assert_eq!(
                LinkRequirements::parse_sidecar(&rejected),
                None,
                "{rejected:?}"
            );
        }
    }
}
