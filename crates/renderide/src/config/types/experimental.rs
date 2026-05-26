//! Experimental renderer settings. Persisted as `[experimental]`.

use serde::{Deserialize, Serialize};

/// Feature flags for renderer behavior that is still experimental.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct ExperimentalSettings {
    /// Maximum number of local reflection probes that can contribute to reflections on a single mesh.
    pub max_local_reflection_probes: usize,
    /// Whether reflection probes may contribute SH2 indirect diffuse lighting.
    pub reflection_probe_sh2_enabled: bool,
    /// Whether local `shaders/target/*.wgsl` edits invalidate and reload material pipelines in development builds.
    pub material_shader_hot_reload_enabled: bool,
}

impl Default for ExperimentalSettings {
    fn default() -> Self {
        Self {
            max_local_reflection_probes: 2,
            reflection_probe_sh2_enabled: false,
            material_shader_hot_reload_enabled: false,
        }
    }
}
