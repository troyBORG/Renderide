//! Display-related renderer-config HUD controls.

use crate::config::RendererSettings;

use super::controls::drag_u32_slider_setting;

/// Foreground and background desktop FPS caps.
pub(super) fn display_section(ui: &imgui::Ui, g: &mut RendererSettings, dirty: &mut bool) {
    ui.text("Display");
    ui.indent();
    if drag_u32_slider_setting(
        ui,
        "Foreground FPS cap (0 = uncapped)",
        &mut g.display.focused_fps_cap,
        0,
        2000,
    ) {
        *dirty = true;
    }
    if drag_u32_slider_setting(
        ui,
        "Background FPS cap (0 = uncapped)",
        &mut g.display.unfocused_fps_cap,
        0,
        2000,
    ) {
        *dirty = true;
    }
    ui.unindent();
}
