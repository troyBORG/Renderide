//! Per-frame resolved controller pose before IPC mapping.

use glam::{Quat, Vec3};

use crate::shared::Chirality;
use crate::xr::session::openxr_tracking_pose_to_host;

use super::pose::{
    bound_hand_pose_defaults, controller_pose_from_aim, openxr_grip_to_steamvr_raw,
    touch_pose_correction,
};
use super::profile::ActiveControllerProfile;

/// Resolved controller and optional bound-hand pose in tracking space.
#[derive(Clone, Copy)]
pub(super) struct ControllerFrame {
    /// Controller position in host-tracking space.
    pub(super) position: Vec3,
    /// Controller orientation in host-tracking space.
    pub(super) rotation: Quat,
    /// Whether the frame carries a calibrated bound-hand offset.
    pub(super) has_bound_hand: bool,
    /// Bound-hand position relative to the controller pose.
    pub(super) hand_position: Vec3,
    /// Bound-hand rotation relative to the controller pose.
    pub(super) hand_rotation: Quat,
}

/// Solves a host-facing controller pose whose configured bound-hand offset lands on `hand_anchor`.
fn controller_from_hand_anchor(
    hand_anchor_position: Vec3,
    hand_anchor_rotation: Quat,
    hand_position: Vec3,
    hand_rotation: Quat,
) -> (Vec3, Quat) {
    let rotation = (hand_anchor_rotation * hand_rotation.inverse()).normalize();
    let position = hand_anchor_position - rotation * hand_position;
    (position, rotation)
}

/// Per-profile pose resolution. `palm_ext` is preferred when available because it directly
/// describes the hand anchor; otherwise grip poses are preferred and aim poses are used only as a
/// fallback. The grip/aim fallback preserves the SteamVR raw-style controller pose path before any
/// host-specific Touch correction is applied.
pub(super) fn resolve_controller_frame(
    profile: ActiveControllerProfile,
    side: Chirality,
    grip_pose: Option<(Vec3, Quat)>,
    aim_pose: Option<(Vec3, Quat)>,
    palm_ext_pose: Option<(Vec3, Quat)>,
) -> Option<ControllerFrame> {
    let (has_bound_hand, hand_position_default, hand_rotation_default) =
        bound_hand_pose_defaults(profile, side);

    if let Some((palm_position, palm_rotation)) = palm_ext_pose {
        let (hand_anchor_position, hand_anchor_rotation) =
            openxr_tracking_pose_to_host(palm_position, palm_rotation);
        let (position, rotation) = controller_from_hand_anchor(
            hand_anchor_position,
            hand_anchor_rotation,
            hand_position_default,
            hand_rotation_default,
        );
        return Some(ControllerFrame {
            position,
            rotation,
            has_bound_hand,
            hand_position: hand_position_default,
            hand_rotation: hand_rotation_default,
        });
    }

    let (grip_position, grip_rotation) = grip_pose.or_else(|| {
        aim_pose.map(|(position, rotation)| controller_pose_from_aim(position, rotation))
    })?;
    let (position, rotation) =
        openxr_grip_to_steamvr_raw(profile, side, grip_position, grip_rotation);
    let (position, rotation) = match profile {
        ActiveControllerProfile::Touch => touch_pose_correction(side, position, rotation),
        _ => (position, rotation),
    };

    Some(ControllerFrame {
        position,
        rotation,
        has_bound_hand,
        hand_position: hand_position_default,
        hand_rotation: hand_rotation_default,
    })
}

#[cfg(test)]
mod tests {
    use glam::{Quat, Vec3};

    use crate::shared::Chirality;
    use crate::xr::session::openxr_tracking_pose_to_host;

    use super::super::pose::{
        bound_hand_pose_defaults, controller_pose_from_aim, openxr_grip_to_steamvr_raw,
        touch_pose_correction,
    };
    use super::super::profile::ActiveControllerProfile;
    use super::resolve_controller_frame;

    fn assert_vec3_near(actual: Vec3, expected: Vec3) {
        let delta = (actual - expected).length();
        assert!(
            delta < 1e-4,
            "vec3 mismatch: actual={actual:?} expected={expected:?} delta={delta}"
        );
    }

    fn assert_quat_near(actual: Quat, expected: Quat) {
        let dot = actual.normalize().dot(expected.normalize()).abs();
        assert!(
            (1.0 - dot) < 1e-4,
            "quat mismatch: actual={actual:?} expected={expected:?} dot={dot}"
        );
    }

    fn rotation_delta_angle(a: Quat, b: Quat) -> f32 {
        2.0 * a
            .normalize()
            .dot(b.normalize())
            .abs()
            .clamp(-1.0, 1.0)
            .acos()
    }

    #[test]
    fn index_grip_is_raw_pose_corrected_with_identity_bound_hand() {
        let grip_position = Vec3::new(0.2, 1.3, -0.4);
        let grip_rotation = (Quat::from_rotation_y(0.6) * Quat::from_rotation_x(-0.2)).normalize();
        let aim_position = Vec3::new(0.24, 1.34, -0.28);
        let aim_rotation = (Quat::from_rotation_y(0.75) * Quat::from_rotation_x(-0.1)).normalize();

        let frame = resolve_controller_frame(
            ActiveControllerProfile::Index,
            Chirality::Left,
            Some((grip_position, grip_rotation)),
            Some((aim_position, aim_rotation)),
            None,
        )
        .expect("frame");

        let (expected_position, expected_rotation) = openxr_grip_to_steamvr_raw(
            ActiveControllerProfile::Index,
            Chirality::Left,
            grip_position,
            grip_rotation,
        );
        assert_vec3_near(frame.position, expected_position);
        assert_quat_near(frame.rotation, expected_rotation);
        assert!(frame.has_bound_hand);
        assert_vec3_near(frame.hand_position, Vec3::ZERO);
        assert_quat_near(frame.hand_rotation, Quat::IDENTITY);
    }

    #[test]
    fn index_aim_fallback_is_raw_pose_corrected() {
        let aim_position = Vec3::new(0.24, 1.34, -0.28);
        let aim_rotation = (Quat::from_rotation_y(0.75) * Quat::from_rotation_x(-0.1)).normalize();
        let frame = resolve_controller_frame(
            ActiveControllerProfile::Index,
            Chirality::Left,
            None,
            Some((aim_position, aim_rotation)),
            None,
        )
        .expect("frame");
        let (aim_grip_position, aim_grip_rotation) =
            controller_pose_from_aim(aim_position, aim_rotation);
        let (expected_position, expected_rotation) = openxr_grip_to_steamvr_raw(
            ActiveControllerProfile::Index,
            Chirality::Left,
            aim_grip_position,
            aim_grip_rotation,
        );
        assert_vec3_near(frame.position, expected_position);
        assert_quat_near(frame.rotation, expected_rotation);
    }

    #[test]
    fn generic_uses_aim_when_grip_missing() {
        let aim_position = Vec3::new(0.1, 1.2, -0.3);
        let aim_rotation = Quat::from_rotation_x(0.3);
        let frame = resolve_controller_frame(
            ActiveControllerProfile::Generic,
            Chirality::Right,
            None,
            Some((aim_position, aim_rotation)),
            None,
        )
        .expect("frame");
        let (aim_grip_position, aim_grip_rotation) =
            controller_pose_from_aim(aim_position, aim_rotation);
        let (expected_controller_position, expected_controller_rotation) =
            openxr_tracking_pose_to_host(aim_grip_position, aim_grip_rotation);
        assert_vec3_near(frame.position, expected_controller_position);
        assert_quat_near(frame.rotation, expected_controller_rotation);
    }

    #[test]
    fn touch_uses_aim_when_grip_missing() {
        let aim_position = Vec3::new(-0.2, 1.1, -0.25);
        let aim_rotation = Quat::from_rotation_y(-0.4);
        let frame = resolve_controller_frame(
            ActiveControllerProfile::Touch,
            Chirality::Left,
            None,
            Some((aim_position, aim_rotation)),
            None,
        )
        .expect("frame");
        let (aim_grip_position, aim_grip_rotation) =
            controller_pose_from_aim(aim_position, aim_rotation);
        let (raw_position, raw_rotation) = openxr_grip_to_steamvr_raw(
            ActiveControllerProfile::Touch,
            Chirality::Left,
            aim_grip_position,
            aim_grip_rotation,
        );
        let (expected_position, expected_rotation) =
            touch_pose_correction(Chirality::Left, raw_position, raw_rotation);
        assert_vec3_near(frame.position, expected_position);
        assert_quat_near(frame.rotation, expected_rotation);
    }

    #[test]
    fn touch_prefers_grip_when_both_present() {
        let grip_position = Vec3::new(0.2, 1.3, -0.4);
        let grip_rotation = (Quat::from_rotation_y(0.6) * Quat::from_rotation_x(-0.2)).normalize();
        let aim_position = Vec3::new(0.5, 0.5, 0.5);
        let aim_rotation = Quat::IDENTITY;
        let frame = resolve_controller_frame(
            ActiveControllerProfile::Touch,
            Chirality::Left,
            Some((grip_position, grip_rotation)),
            Some((aim_position, aim_rotation)),
            None,
        )
        .expect("frame");
        let (raw_position, raw_rotation) = openxr_grip_to_steamvr_raw(
            ActiveControllerProfile::Touch,
            Chirality::Left,
            grip_position,
            grip_rotation,
        );
        let (expected_pos, expected_rot) =
            touch_pose_correction(Chirality::Left, raw_position, raw_rotation);
        assert_vec3_near(frame.position, expected_pos);
        assert_quat_near(frame.rotation, expected_rot);
    }

    #[test]
    fn palm_ext_pose_takes_priority_over_grip_and_aim() {
        let palm_position = Vec3::new(0.3, 1.2, -0.5);
        let palm_rotation = Quat::from_rotation_x(0.25).normalize();
        let grip_position = Vec3::new(0.9, 0.8, -0.1);
        let grip_rotation = Quat::from_rotation_y(0.8).normalize();
        let aim_position = Vec3::new(-0.9, 1.4, -0.4);
        let aim_rotation = Quat::from_rotation_z(0.5).normalize();

        let frame = resolve_controller_frame(
            ActiveControllerProfile::Generic,
            Chirality::Left,
            Some((grip_position, grip_rotation)),
            Some((aim_position, aim_rotation)),
            Some((palm_position, palm_rotation)),
        )
        .expect("frame");

        let (hand_position, hand_rotation) =
            openxr_tracking_pose_to_host(palm_position, palm_rotation);
        let composed_position = frame.position + frame.rotation * frame.hand_position;
        let composed_rotation = (frame.rotation * frame.hand_rotation).normalize();
        assert_vec3_near(composed_position, hand_position);
        assert_quat_near(composed_rotation, hand_rotation);

        let (fallback_position, fallback_rotation) = openxr_grip_to_steamvr_raw(
            ActiveControllerProfile::Generic,
            Chirality::Left,
            grip_position,
            grip_rotation,
        );
        assert!(
            (frame.position - fallback_position).length() > 0.01
                || rotation_delta_angle(frame.rotation, fallback_rotation) > 0.01,
            "palm_ext should override grip fallback"
        );
    }

    #[test]
    fn touch_palm_ext_solves_hand_anchor_without_touch_controller_correction() {
        let palm_position = Vec3::new(-0.25, 1.3, -0.45);
        let palm_rotation = (Quat::from_rotation_y(-0.3) * Quat::from_rotation_x(0.1)).normalize();
        let frame = resolve_controller_frame(
            ActiveControllerProfile::Touch,
            Chirality::Left,
            Some((Vec3::ZERO, Quat::IDENTITY)),
            None,
            Some((palm_position, palm_rotation)),
        )
        .expect("frame");

        let (hand_position, hand_rotation) =
            openxr_tracking_pose_to_host(palm_position, palm_rotation);
        assert_vec3_near(
            frame.position + frame.rotation * frame.hand_position,
            hand_position,
        );
        assert_quat_near(
            (frame.rotation * frame.hand_rotation).normalize(),
            hand_rotation,
        );

        let (_, touch_offset_position, touch_offset_rotation) =
            bound_hand_pose_defaults(ActiveControllerProfile::Touch, Chirality::Left);
        assert_vec3_near(frame.hand_position, touch_offset_position);
        assert_quat_near(frame.hand_rotation, touch_offset_rotation);
    }

    #[test]
    fn non_touch_profiles_skip_touch_correction() {
        let grip_position = Vec3::new(0.3, 1.2, -0.5);
        let grip_rotation = Quat::from_rotation_x(0.25).normalize();
        for profile in [
            ActiveControllerProfile::Index,
            ActiveControllerProfile::ViveFocus3,
            ActiveControllerProfile::Vive,
            ActiveControllerProfile::Generic,
        ] {
            let frame = resolve_controller_frame(
                profile,
                Chirality::Right,
                Some((grip_position, grip_rotation)),
                None,
                None,
            )
            .unwrap_or_else(|| panic!("frame for {profile:?}"));
            let (expected_position, expected_rotation) =
                openxr_grip_to_steamvr_raw(profile, Chirality::Right, grip_position, grip_rotation);
            assert_vec3_near(frame.position, expected_position);
            assert_quat_near(frame.rotation, expected_rotation);
        }
    }

    #[test]
    fn identity_offset_profiles_pass_grip_through() {
        let grip_position = Vec3::new(0.3, 1.2, -0.5);
        let grip_rotation = Quat::from_rotation_x(0.25).normalize();
        for profile in [
            ActiveControllerProfile::Vive,
            ActiveControllerProfile::WindowsMr,
            ActiveControllerProfile::HpReverbG2,
            ActiveControllerProfile::Pico4,
            ActiveControllerProfile::PicoNeo3,
            ActiveControllerProfile::ViveCosmos,
            ActiveControllerProfile::Generic,
            ActiveControllerProfile::Simple,
        ] {
            let frame = resolve_controller_frame(
                profile,
                Chirality::Right,
                Some((grip_position, grip_rotation)),
                None,
                None,
            )
            .unwrap_or_else(|| panic!("frame for {profile:?}"));
            let (expected_position, expected_rotation) =
                openxr_tracking_pose_to_host(grip_position, grip_rotation);
            assert_vec3_near(frame.position, expected_position);
            assert_quat_near(frame.rotation, expected_rotation);
        }
    }

    #[test]
    fn raw_corrected_profiles_shift_grip_pose() {
        let grip_position = Vec3::new(0.3, 1.2, -0.5);
        let grip_rotation = Quat::from_rotation_x(0.25).normalize();
        let (host_grip_position, host_grip_rotation) =
            openxr_tracking_pose_to_host(grip_position, grip_rotation);
        for profile in [
            ActiveControllerProfile::Index,
            ActiveControllerProfile::Touch,
            ActiveControllerProfile::ViveFocus3,
        ] {
            let frame = resolve_controller_frame(
                profile,
                Chirality::Right,
                Some((grip_position, grip_rotation)),
                None,
                None,
            )
            .unwrap_or_else(|| panic!("frame for {profile:?}"));
            assert!(
                (frame.position - host_grip_position).length() > 0.05,
                "{profile:?}: expected non-trivial raw-pose position correction",
            );
            assert!(
                rotation_delta_angle(frame.rotation, host_grip_rotation) > 0.2,
                "{profile:?}: expected non-trivial raw-pose rotation correction",
            );
        }
    }

    #[test]
    fn bound_hand_offsets_do_not_change_controller_pose() {
        let grip_position = Vec3::new(0.3, 1.2, -0.5);
        let grip_rotation = Quat::from_rotation_x(0.25).normalize();
        let frame = resolve_controller_frame(
            ActiveControllerProfile::Generic,
            Chirality::Left,
            Some((grip_position, grip_rotation)),
            None,
            None,
        )
        .expect("frame");
        let (expected_position, expected_rotation) =
            openxr_tracking_pose_to_host(grip_position, grip_rotation);

        assert_vec3_near(frame.position, expected_position);
        assert_quat_near(frame.rotation, expected_rotation);
        assert!(frame.hand_position.length() > 0.01);
        assert!(rotation_delta_angle(frame.hand_rotation, Quat::IDENTITY) > 0.2);
    }
}
