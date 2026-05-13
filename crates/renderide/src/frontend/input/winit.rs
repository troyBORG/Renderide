//! Adapts winit window events into [`WindowInputAccumulator`](super::WindowInputAccumulator).
//!
//! Submodules:
//! - [`event_transition`] -- pure mapping from winit events to host-shaped transitions.
//! - [`key_map`] -- platform-key to host [`crate::shared::Key`] table consumed by `event_transition`.

mod event_transition;
mod key_map;

use std::path::{Path, PathBuf};

use winit::dpi::{LogicalSize, PhysicalPosition};
use winit::event::{
    ButtonSource, DeviceEvent, ElementState, Ime, KeyEvent, MouseButton, MouseScrollDelta,
    PointerKind, PointerSource, WindowEvent,
};
use winit::window::Window;

use self::event_transition::{
    HeldKeyTransition, KeyboardEventTransition, MouseButtonSlot, host_scroll_delta_from_wheel,
    imgui_scroll_delta_from_wheel, keyboard_event_transition, mouse_button_transition,
};
use super::accumulator::WindowInputAccumulator;

/// Applies a [`WindowEvent`] from winit to the accumulator.
///
/// [`WindowEvent::SurfaceResized`], [`WindowEvent::ScaleFactorChanged`], and pointer move use the same
/// **logical** pixel space as [`WindowInputAccumulator::window_position`].
pub fn apply_window_event(
    acc: &mut WindowInputAccumulator,
    window: &dyn Window,
    event: &WindowEvent,
) {
    match event {
        WindowEvent::SurfaceResized(size) => {
            profiling::scope!("frontend::window_event", "resize");
            let logical: LogicalSize<f64> = size.to_logical(window.scale_factor());
            acc.window_resolution = (logical.width.round() as u32, logical.height.round() as u32);
        }
        WindowEvent::ScaleFactorChanged { .. } => {
            profiling::scope!("frontend::window_event", "scale_factor");
            acc.sync_window_resolution_logical(window);
        }
        WindowEvent::PointerMoved { .. } => {
            profiling::scope!("frontend::window_event", "cursor_moved");
            let _ = apply_mouse_pointer_move(acc, window.scale_factor(), event);
        }
        WindowEvent::PointerEntered { .. } => {
            profiling::scope!("frontend::window_event", "cursor_entered");
            let _ = apply_mouse_pointer_presence(acc, event);
        }
        WindowEvent::PointerLeft { .. } => {
            profiling::scope!("frontend::window_event", "cursor_left");
            let _ = apply_mouse_pointer_presence(acc, event);
        }
        WindowEvent::Focused(focused) => {
            profiling::scope!("frontend::window_event", "focus");
            acc.window_focused = *focused;
            if !*focused {
                acc.clear_stuck_keyboard_on_focus_lost();
            }
        }
        WindowEvent::ModifiersChanged(modifiers) => {
            profiling::scope!("frontend::window_event", "modifiers");
            acc.set_keyboard_modifiers(modifiers.state());
        }
        WindowEvent::PointerButton {
            state,
            button: ButtonSource::Mouse(mouse_button),
            ..
        } => {
            profiling::scope!("frontend::window_event", "mouse_button");
            apply_mouse_button(acc, *state, *mouse_button);
        }
        WindowEvent::MouseWheel { delta, .. } => {
            profiling::scope!("frontend::window_event", "scroll");
            apply_mouse_wheel(acc, delta);
        }
        WindowEvent::KeyboardInput {
            event,
            is_synthetic,
            ..
        } => {
            profiling::scope!("frontend::window_event", "key");
            if *is_synthetic {
                return;
            }
            apply_keyboard_event(acc, event);
        }
        WindowEvent::Ime(ime) => {
            profiling::scope!("frontend::window_event", "ime");
            match ime {
                Ime::Commit(s) => acc.push_ime_commit(s.as_str()),
                Ime::Enabled
                | Ime::Disabled
                | Ime::Preedit(_, _)
                | Ime::DeleteSurrounding { .. } => {}
            }
        }
        WindowEvent::DragDropped { paths, position } => {
            profiling::scope!("frontend::window_event", "dropped_file");
            apply_drag_dropped_paths(acc, window.scale_factor(), paths, *position);
        }
        _ => {}
    }
}

fn apply_mouse_pointer_move(
    acc: &mut WindowInputAccumulator,
    scale_factor: f64,
    event: &WindowEvent,
) -> bool {
    let WindowEvent::PointerMoved {
        position, source, ..
    } = event
    else {
        return false;
    };
    if !matches!(source, PointerSource::Mouse) {
        return false;
    }
    acc.set_cursor_from_physical(*position, scale_factor);
    true
}

fn apply_mouse_pointer_presence(acc: &mut WindowInputAccumulator, event: &WindowEvent) -> bool {
    match event {
        WindowEvent::PointerEntered {
            kind: PointerKind::Mouse,
            ..
        } => {
            acc.mouse_active = true;
            true
        }
        WindowEvent::PointerLeft {
            kind: PointerKind::Mouse,
            ..
        } => {
            acc.mouse_active = false;
            true
        }
        _ => false,
    }
}

fn apply_drag_dropped_paths(
    acc: &mut WindowInputAccumulator,
    scale_factor: f64,
    paths: &[PathBuf],
    position: PhysicalPosition<f64>,
) {
    acc.set_cursor_from_physical(position, scale_factor);
    for path in paths {
        acc.push_dropped_file_path(path_to_string_lossy(path));
    }
}

/// Updates per-button held flags for a [`WindowEvent::PointerButton`].
fn apply_mouse_button(acc: &mut WindowInputAccumulator, state: ElementState, button: MouseButton) {
    let Some(transition) = mouse_button_transition(state, button) else {
        return;
    };
    match transition.slot {
        MouseButtonSlot::Left => acc.left_held = transition.pressed,
        MouseButtonSlot::Right => acc.right_held = transition.pressed,
        MouseButtonSlot::Middle => acc.middle_held = transition.pressed,
        MouseButtonSlot::Button4 => acc.button4_held = transition.pressed,
        MouseButtonSlot::Button5 => acc.button5_held = transition.pressed,
    }
}

/// Accumulates scroll delta for a [`WindowEvent::MouseWheel`] in host and HUD units.
fn apply_mouse_wheel(acc: &mut WindowInputAccumulator, delta: &MouseScrollDelta) {
    acc.push_scroll_delta(
        host_scroll_delta_from_wheel(delta),
        imgui_scroll_delta_from_wheel(delta),
    );
}

/// Updates held-key list and queued text-input strings for a non-synthetic [`KeyEvent`].
fn apply_keyboard_event(acc: &mut WindowInputAccumulator, event: &KeyEvent) {
    let transition = keyboard_event_transition(event);
    apply_keyboard_transition(acc, transition);
}

fn apply_keyboard_transition(
    acc: &mut WindowInputAccumulator,
    transition: KeyboardEventTransition,
) {
    match transition.held_key {
        Some(HeldKeyTransition::Press(key)) if !acc.held_keys.contains(&key) => {
            acc.held_keys.push(key);
        }
        Some(HeldKeyTransition::Release(key)) => {
            acc.held_keys.retain(|held| *held != key);
        }
        Some(HeldKeyTransition::Press(_)) | None => {}
    }
    if let Some(text) = transition.text {
        acc.push_key_text(text.as_str());
    }
}

fn path_to_string_lossy(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}

/// Applies relative pointer motion when the cursor is captured (locked / confined).
pub fn apply_device_event(acc: &mut WindowInputAccumulator, event: &DeviceEvent) {
    if let DeviceEvent::PointerMotion { delta } = event {
        profiling::scope!("frontend::device_event", "mouse_motion");
        acc.mouse_delta.x += delta.0 as f32;
        acc.mouse_delta.y -= delta.1 as f32;
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use glam::{IVec2, Vec2};
    use winit::dpi::PhysicalPosition;
    use winit::event::{
        DeviceId, FingerId, PointerSource, TabletToolData, TabletToolKind, WindowEvent,
    };

    use super::{apply_drag_dropped_paths, apply_mouse_pointer_move};
    use crate::frontend::input::WindowInputAccumulator;

    fn pointer_moved(source: PointerSource) -> WindowEvent {
        WindowEvent::PointerMoved {
            device_id: Some(DeviceId::from_raw(7)),
            position: PhysicalPosition::new(20.0, 10.0),
            primary: true,
            source,
        }
    }

    #[test]
    fn mouse_pointer_move_updates_cursor_position() {
        let mut acc = WindowInputAccumulator::default();
        let event = pointer_moved(PointerSource::Mouse);

        assert!(apply_mouse_pointer_move(&mut acc, 2.0, &event));
        assert_eq!(acc.window_position, Vec2::new(10.0, 5.0));
    }

    #[test]
    fn non_mouse_pointer_move_does_not_update_cursor_position() {
        for source in [
            PointerSource::Touch {
                finger_id: FingerId::from_raw(1),
                force: None,
            },
            PointerSource::TabletTool {
                kind: TabletToolKind::Pen,
                data: TabletToolData::default(),
            },
        ] {
            let mut acc = WindowInputAccumulator::default();
            acc.window_position = Vec2::new(3.0, 4.0);
            let event = pointer_moved(source);

            assert!(!apply_mouse_pointer_move(&mut acc, 2.0, &event));
            assert_eq!(acc.window_position, Vec2::new(3.0, 4.0));
        }
    }

    #[test]
    fn drag_dropped_records_all_paths_and_event_position() {
        let mut acc = WindowInputAccumulator::default();
        let paths = vec![PathBuf::from("first.txt"), PathBuf::from("second.txt")];

        apply_drag_dropped_paths(&mut acc, 2.0, &paths, PhysicalPosition::new(21.0, 43.0));

        let input = acc.take_input_state(false);
        let drag = input
            .window
            .and_then(|window| window.drag_and_drop_event)
            .expect("drag event");
        assert_eq!(
            drag.paths,
            vec![
                Some("first.txt".to_string()),
                Some("second.txt".to_string())
            ]
        );
        assert_eq!(drag.drop_point, IVec2::new(21, 43));
        assert_eq!(
            input.mouse.expect("mouse state").window_position,
            Vec2::new(10.5, 21.5)
        );
    }
}
