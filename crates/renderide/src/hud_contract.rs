//! Debug-HUD contracts shared by graph execution, passes, backend state, and diagnostics.

use thiserror::Error;

/// Failure during ImGui draw-list submission or related setup for the overlay pass.
#[derive(Debug, Error)]
pub enum DebugHudEncodeError {
    /// The wgpu renderer for ImGui returned an error string.
    #[error("imgui-wgpu render: {0}")]
    ImguiWgpu(String),
}

/// HUD payload produced by one view during recording.
#[derive(Default)]
pub struct PerViewHudOutputs {
    /// Latest world-mesh draw stats for the view when the Stats tab is active.
    pub world_mesh_draw_stats: Option<crate::world_mesh::WorldMeshDrawStats>,
    /// Latest world-mesh draw-state rows for the view when the Draw state tab is active.
    pub world_mesh_draw_state_rows: Option<Vec<crate::world_mesh::WorldMeshDrawStateRow>>,
    /// Texture2D asset ids used by the view when the textures HUD is enabled.
    pub current_view_texture_2d_asset_ids: Vec<i32>,
}

/// Read-only HUD capture switches needed during per-view recording.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PerViewHudConfig {
    /// Whether the Stats tab wants world-mesh stats from the current view.
    pub capture_world_mesh_draw_stats: bool,
    /// Whether the Draw state tab wants resolved draw rows from the current view.
    pub capture_world_mesh_draw_state_rows: bool,
    /// Whether the textures HUD wants current-view Texture2D ids.
    pub capture_current_view_texture_2d_asset_ids: bool,
}
