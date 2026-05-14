//! OpenXR hand-joint sampling and conversion into host [`HandState`] packets.
//!
//! OpenXR reports hand joints in the selected base space using OpenXR's right-handed tracking
//! basis. The host consumes the same wrist and wrist-relative segment convention that the
//! SteamVR skeleton path produced, so this module converts OpenXR joints into host tracking space,
//! makes finger segments relative to the compensated wrist, and applies the same side-specific
//! wrist/finger basis corrections used by that skeleton path.

use std::{mem::MaybeUninit, ptr};

use glam::{Mat3, Quat, Vec3};
use openxr as xr;

use crate::shared::{Chirality, HandState};
use crate::xr::session::openxr_tracking_pose_to_host;

/// Number of host finger segments carried by [`HandState`].
const SEGMENT_COUNT: usize = 24;

/// Stable IPC identifier for the left OpenXR hand.
const LEFT_HAND_ID: &str = "renderide_left_hand";

/// Stable IPC identifier for the right OpenXR hand.
const RIGHT_HAND_ID: &str = "renderide_right_hand";

/// OpenXR hand-joint motion range requested while locating joints.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum HandMotionRange {
    /// Use the runtime's default hand-joint motion range.
    RuntimeDefault,
    /// Ask the runtime for controller-conforming joints when the extension is available.
    ControllerConforming,
}

/// OpenXR joints in host flat segment order.
const SEGMENT_JOINTS: [xr::HandJoint; SEGMENT_COUNT] = [
    xr::HandJoint::THUMB_METACARPAL,
    xr::HandJoint::THUMB_PROXIMAL,
    xr::HandJoint::THUMB_DISTAL,
    xr::HandJoint::THUMB_TIP,
    xr::HandJoint::INDEX_METACARPAL,
    xr::HandJoint::INDEX_PROXIMAL,
    xr::HandJoint::INDEX_INTERMEDIATE,
    xr::HandJoint::INDEX_DISTAL,
    xr::HandJoint::INDEX_TIP,
    xr::HandJoint::MIDDLE_METACARPAL,
    xr::HandJoint::MIDDLE_PROXIMAL,
    xr::HandJoint::MIDDLE_INTERMEDIATE,
    xr::HandJoint::MIDDLE_DISTAL,
    xr::HandJoint::MIDDLE_TIP,
    xr::HandJoint::RING_METACARPAL,
    xr::HandJoint::RING_PROXIMAL,
    xr::HandJoint::RING_INTERMEDIATE,
    xr::HandJoint::RING_DISTAL,
    xr::HandJoint::RING_TIP,
    xr::HandJoint::LITTLE_METACARPAL,
    xr::HandJoint::LITTLE_PROXIMAL,
    xr::HandJoint::LITTLE_INTERMEDIATE,
    xr::HandJoint::LITTLE_DISTAL,
    xr::HandJoint::LITTLE_TIP,
];

/// Left and right OpenXR hand trackers plus the active motion-range policy.
pub(super) struct OpenxrHandTrackers {
    /// Left-hand tracker when the runtime created one successfully.
    left: Option<xr::HandTracker>,
    /// Right-hand tracker when the runtime created one successfully.
    right: Option<xr::HandTracker>,
    /// Motion range requested when locating hand joints.
    motion_range: HandMotionRange,
}

impl OpenxrHandTrackers {
    /// Creates optional left/right hand trackers for the OpenXR session.
    pub(super) fn new(session: &xr::Session<xr::Vulkan>) -> Option<Self> {
        if session.instance().exts().ext_hand_tracking.is_none() {
            logger::debug!("OpenXR hand tracking: XR_EXT_hand_tracking is not enabled");
            return None;
        }
        let motion_range = if session
            .instance()
            .exts()
            .ext_hand_joints_motion_range
            .is_some()
        {
            HandMotionRange::ControllerConforming
        } else {
            HandMotionRange::RuntimeDefault
        };
        let left = create_hand_tracker(session, Chirality::Left);
        let right = create_hand_tracker(session, Chirality::Right);
        if left.is_none() && right.is_none() {
            logger::debug!("OpenXR hand tracking: no hand trackers were created");
            return None;
        }
        logger::info!(
            "OpenXR hand tracking initialized: left={}, right={}, motion_range={motion_range:?}",
            left.is_some(),
            right.is_some()
        );
        Some(Self {
            left,
            right,
            motion_range,
        })
    }

    /// Samples all active hand trackers and returns host-facing hand states.
    pub(super) fn sample(
        &self,
        session: &xr::Session<xr::Vulkan>,
        stage: &xr::Space,
        predicted_time: xr::Time,
    ) -> Vec<HandState> {
        let mut hands = Vec::with_capacity(2);
        self.sample_one(
            session,
            stage,
            predicted_time,
            Chirality::Left,
            self.left.as_ref(),
            &mut hands,
        );
        self.sample_one(
            session,
            stage,
            predicted_time,
            Chirality::Right,
            self.right.as_ref(),
            &mut hands,
        );
        if !hands.is_empty() {
            log_openxr_hand_source_once();
        }
        hands
    }

    /// Samples one tracker and appends its hand state when the runtime provides a complete hand.
    fn sample_one(
        &self,
        session: &xr::Session<xr::Vulkan>,
        stage: &xr::Space,
        predicted_time: xr::Time,
        side: Chirality,
        tracker: Option<&xr::HandTracker>,
        hands: &mut Vec<HandState>,
    ) {
        let Some(tracker) = tracker else {
            return;
        };
        match locate_hand_joints(session, stage, tracker, predicted_time, self.motion_range) {
            Ok(Some(locations)) => {
                if let Some(hand) = hand_state_from_openxr_joints(side, &locations) {
                    hands.push(hand);
                }
            }
            Ok(None) => {}
            Err(error) => logger::trace!("OpenXR {side:?} hand joint locate failed: {error:?}"),
        }
    }
}

/// Creates one OpenXR hand tracker for a chirality.
fn create_hand_tracker(
    session: &xr::Session<xr::Vulkan>,
    side: Chirality,
) -> Option<xr::HandTracker> {
    let hand = match side {
        Chirality::Left => xr::Hand::LEFT,
        Chirality::Right => xr::Hand::RIGHT,
    };
    match session.create_hand_tracker(hand) {
        Ok(tracker) => Some(tracker),
        Err(error) => {
            logger::debug!("OpenXR {side:?} hand tracker creation failed: {error:?}");
            None
        }
    }
}

/// Locates hand joints, using controller-conforming motion range when requested and available.
fn locate_hand_joints(
    session: &xr::Session<xr::Vulkan>,
    stage: &xr::Space,
    tracker: &xr::HandTracker,
    predicted_time: xr::Time,
    motion_range: HandMotionRange,
) -> Result<Option<xr::HandJointLocations>, xr::sys::Result> {
    match motion_range {
        HandMotionRange::RuntimeDefault => stage.locate_hand_joints(tracker, predicted_time),
        HandMotionRange::ControllerConforming => {
            locate_controller_conforming_hand_joints(session, stage, tracker, predicted_time)
        }
    }
}

/// Locates hand joints with `XR_EXT_hand_joints_motion_range` set to controller-conforming.
fn locate_controller_conforming_hand_joints(
    session: &xr::Session<xr::Vulkan>,
    stage: &xr::Space,
    tracker: &xr::HandTracker,
    predicted_time: xr::Time,
) -> Result<Option<xr::HandJointLocations>, xr::sys::Result> {
    let Some(fp) = session.instance().exts().ext_hand_tracking.as_ref() else {
        return Err(xr::sys::Result::ERROR_EXTENSION_NOT_PRESENT);
    };
    let motion_range_info = xr::sys::HandJointsMotionRangeInfoEXT {
        ty: xr::sys::HandJointsMotionRangeInfoEXT::TYPE,
        next: ptr::null(),
        hand_joints_motion_range: xr::sys::HandJointsMotionRangeEXT::CONFORMING_TO_CONTROLLER,
    };
    let motion_range_next = ptr::from_ref(&motion_range_info).cast();
    let locate_info = xr::sys::HandJointsLocateInfoEXT {
        ty: xr::sys::HandJointsLocateInfoEXT::TYPE,
        next: motion_range_next,
        base_space: stage.as_raw(),
        time: predicted_time,
    };
    let mut locations = MaybeUninit::<xr::HandJointLocations>::uninit();
    let joint_locations = locations.as_mut_ptr().cast::<xr::HandJointLocation>();
    let mut location_info = xr::sys::HandJointLocationsEXT {
        ty: xr::sys::HandJointLocationsEXT::TYPE,
        next: ptr::null_mut(),
        is_active: false.into(),
        joint_count: xr::HAND_JOINT_COUNT as u32,
        joint_locations,
    };
    // SAFETY: `tracker` is created by this module from `session`, `stage` is the same session's
    // reference space supplied by the XR session handles, and all out-pointers reference live stack
    // storage for the duration of the OpenXR call.
    unsafe {
        cvt_xr((fp.locate_hand_joints)(
            tracker.as_raw(),
            ptr::from_ref(&locate_info),
            ptr::from_mut(&mut location_info),
        ))?;
        let is_active = bool::from(location_info.is_active);
        Ok(is_active.then(|| locations.assume_init()))
    }
}

/// Converts an OpenXR result code into the `openxr` crate's result shape.
fn cvt_xr(result: xr::sys::Result) -> Result<xr::sys::Result, xr::sys::Result> {
    if result.into_raw() >= 0 {
        Ok(result)
    } else {
        Err(result)
    }
}

/// Converts one complete OpenXR hand-joint sample into a host [`HandState`].
fn hand_state_from_openxr_joints(
    side: Chirality,
    locations: &xr::HandJointLocations,
) -> Option<HandState> {
    let (wrist_position, wrist_rotation) = host_joint_pose(locations, xr::HandJoint::WRIST)?;
    let wrist_rotation = (wrist_rotation * wrist_compensation(side)).normalize();
    let inverse_wrist_rotation = wrist_rotation.inverse();
    let finger_compensation = finger_compensation(side);
    let mut segment_positions = Vec::with_capacity(SEGMENT_COUNT);
    let mut segment_rotations = Vec::with_capacity(SEGMENT_COUNT);
    for joint in SEGMENT_JOINTS {
        let (segment_position, segment_rotation) = host_joint_pose(locations, joint)?;
        segment_positions.push(inverse_wrist_rotation * (segment_position - wrist_position));
        segment_rotations
            .push((inverse_wrist_rotation * segment_rotation * finger_compensation).normalize());
    }
    Some(HandState {
        unique_id: Some(hand_unique_id(side).to_string()),
        priority: 0,
        chirality: side,
        is_device_active: true,
        is_tracking: true,
        tracks_metacarpals: true,
        confidence: 1.0,
        wrist_position,
        wrist_rotation,
        segment_positions,
        segment_rotations,
    })
}

/// Returns the stable IPC identifier for a hand side.
fn hand_unique_id(side: Chirality) -> &'static str {
    match side {
        Chirality::Left => LEFT_HAND_ID,
        Chirality::Right => RIGHT_HAND_ID,
    }
}

/// Converts an OpenXR joint location into host tracking-space pose components.
fn host_joint_pose(
    locations: &xr::HandJointLocations,
    joint: xr::HandJoint,
) -> Option<(Vec3, Quat)> {
    let location = locations.get(joint_index(joint))?;
    if !has_valid_pose(location) {
        return None;
    }
    let pose = &location.pose;
    let position = Vec3::new(pose.position.x, pose.position.y, pose.position.z);
    let orientation = pose.orientation;
    let rotation = normalize_quat_or_identity(Quat::from_xyzw(
        orientation.x,
        orientation.y,
        orientation.z,
        orientation.w,
    ));
    Some(openxr_tracking_pose_to_host(position, rotation))
}

/// Returns true when the joint has valid position and orientation data.
fn has_valid_pose(location: &xr::HandJointLocation) -> bool {
    location
        .location_flags
        .contains(xr::SpaceLocationFlags::POSITION_VALID)
        && location
            .location_flags
            .contains(xr::SpaceLocationFlags::ORIENTATION_VALID)
}

/// Converts an OpenXR hand-joint enum into an array index.
fn joint_index(joint: xr::HandJoint) -> usize {
    joint.into_raw() as usize
}

/// Normalizes a quaternion, falling back to identity for invalid values.
fn normalize_quat_or_identity(rotation: Quat) -> Quat {
    let len_sq = rotation.length_squared();
    if len_sq.is_finite() && len_sq >= 1e-10 {
        rotation.normalize()
    } else {
        Quat::IDENTITY
    }
}

/// Unity-style `Quaternion.LookRotation(forward, upwards)`.
fn unity_look_rotation(forward: Vec3, upwards: Vec3) -> Quat {
    let forward = forward.normalize();
    let right = upwards.cross(forward).normalize();
    let up = forward.cross(right).normalize();
    Quat::from_mat3(&Mat3::from_cols(right, up, forward)).normalize()
}

/// Side-specific wrist-space compensation used by the SteamVR skeleton path.
fn wrist_compensation(side: Chirality) -> Quat {
    match side {
        Chirality::Left => unity_look_rotation(Vec3::Z, -Vec3::X),
        Chirality::Right => unity_look_rotation(Vec3::Z, Vec3::X),
    }
}

/// Side-specific finger-space compensation used by the SteamVR skeleton path.
fn finger_compensation(side: Chirality) -> Quat {
    match side {
        Chirality::Left => unity_look_rotation(-Vec3::X, -Vec3::Y),
        Chirality::Right => unity_look_rotation(Vec3::X, Vec3::Y),
    }
}

/// Logs the first frame that uses real OpenXR hand joints.
fn log_openxr_hand_source_once() {
    use std::sync::atomic::{AtomicBool, Ordering};

    static LOGGED: AtomicBool = AtomicBool::new(false);
    if !LOGGED.swap(true, Ordering::Relaxed) {
        logger::info!("OpenXR hand input: using tracked hand joints for IPC hand poses");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::BodyNode;

    /// Asserts that two vectors are nearly equal.
    fn assert_vec3_near(actual: Vec3, expected: Vec3) {
        let delta = (actual - expected).length();
        assert!(
            delta < 1e-5,
            "vec3 mismatch: actual={actual:?} expected={expected:?} delta={delta}"
        );
    }

    /// Asserts that two quaternions are nearly equal, accepting equivalent signs.
    fn assert_quat_near(actual: Quat, expected: Quat) {
        let dot = actual.normalize().dot(expected.normalize()).abs();
        assert!(
            (1.0 - dot) < 1e-5,
            "quat mismatch: actual={actual:?} expected={expected:?} dot={dot}"
        );
    }

    /// Builds a valid OpenXR hand-joint location.
    fn valid_location(position: Vec3, rotation: Quat) -> xr::HandJointLocation {
        xr::HandJointLocation {
            location_flags: xr::SpaceLocationFlags::POSITION_VALID
                | xr::SpaceLocationFlags::ORIENTATION_VALID
                | xr::SpaceLocationFlags::POSITION_TRACKED
                | xr::SpaceLocationFlags::ORIENTATION_TRACKED,
            pose: xr::Posef {
                orientation: xr::Quaternionf {
                    x: rotation.x,
                    y: rotation.y,
                    z: rotation.z,
                    w: rotation.w,
                },
                position: xr::Vector3f {
                    x: position.x,
                    y: position.y,
                    z: position.z,
                },
            },
            radius: 0.01,
        }
    }

    /// Builds a complete valid joint array with deterministic positions.
    fn valid_joint_locations() -> xr::HandJointLocations {
        let mut locations = [xr::HandJointLocation::default(); xr::HAND_JOINT_COUNT];
        for (index, location) in locations.iter_mut().enumerate() {
            *location = valid_location(
                Vec3::new(index as f32 * 0.01, 1.0 + index as f32 * 0.005, -0.2),
                Quat::IDENTITY,
            );
        }
        locations
    }

    #[test]
    fn segment_mapping_matches_host_body_node_order() {
        let body_nodes = [
            BodyNode::LeftThumbMetacarpal,
            BodyNode::LeftThumbProximal,
            BodyNode::LeftThumbDistal,
            BodyNode::LeftThumbTip,
            BodyNode::LeftIndexFingerMetacarpal,
            BodyNode::LeftIndexFingerProximal,
            BodyNode::LeftIndexFingerIntermediate,
            BodyNode::LeftIndexFingerDistal,
            BodyNode::LeftIndexFingerTip,
            BodyNode::LeftMiddleFingerMetacarpal,
            BodyNode::LeftMiddleFingerProximal,
            BodyNode::LeftMiddleFingerIntermediate,
            BodyNode::LeftMiddleFingerDistal,
            BodyNode::LeftMiddleFingerTip,
            BodyNode::LeftRingFingerMetacarpal,
            BodyNode::LeftRingFingerProximal,
            BodyNode::LeftRingFingerIntermediate,
            BodyNode::LeftRingFingerDistal,
            BodyNode::LeftRingFingerTip,
            BodyNode::LeftPinkyMetacarpal,
            BodyNode::LeftPinkyProximal,
            BodyNode::LeftPinkyIntermediate,
            BodyNode::LeftPinkyDistal,
            BodyNode::LeftPinkyTip,
        ];
        for (index, body_node) in body_nodes.iter().enumerate() {
            assert_eq!(
                *body_node as i32,
                BodyNode::LeftThumbMetacarpal as i32 + index as i32
            );
        }
        assert_eq!(
            body_nodes.len(),
            SEGMENT_JOINTS.len(),
            "host BodyNode segment count must match OpenXR joint segment count"
        );
    }

    #[test]
    fn segment_mapping_uses_openxr_finger_joint_order() {
        assert_eq!(
            SEGMENT_JOINTS,
            [
                xr::HandJoint::THUMB_METACARPAL,
                xr::HandJoint::THUMB_PROXIMAL,
                xr::HandJoint::THUMB_DISTAL,
                xr::HandJoint::THUMB_TIP,
                xr::HandJoint::INDEX_METACARPAL,
                xr::HandJoint::INDEX_PROXIMAL,
                xr::HandJoint::INDEX_INTERMEDIATE,
                xr::HandJoint::INDEX_DISTAL,
                xr::HandJoint::INDEX_TIP,
                xr::HandJoint::MIDDLE_METACARPAL,
                xr::HandJoint::MIDDLE_PROXIMAL,
                xr::HandJoint::MIDDLE_INTERMEDIATE,
                xr::HandJoint::MIDDLE_DISTAL,
                xr::HandJoint::MIDDLE_TIP,
                xr::HandJoint::RING_METACARPAL,
                xr::HandJoint::RING_PROXIMAL,
                xr::HandJoint::RING_INTERMEDIATE,
                xr::HandJoint::RING_DISTAL,
                xr::HandJoint::RING_TIP,
                xr::HandJoint::LITTLE_METACARPAL,
                xr::HandJoint::LITTLE_PROXIMAL,
                xr::HandJoint::LITTLE_INTERMEDIATE,
                xr::HandJoint::LITTLE_DISTAL,
                xr::HandJoint::LITTLE_TIP,
            ]
        );
    }

    #[test]
    fn compensation_quaternions_match_expected_axes() {
        assert_vec3_near(finger_compensation(Chirality::Right) * Vec3::Z, Vec3::X);
        assert_vec3_near(finger_compensation(Chirality::Right) * Vec3::Y, Vec3::Y);
        assert_vec3_near(finger_compensation(Chirality::Left) * Vec3::Z, -Vec3::X);
        assert_vec3_near(finger_compensation(Chirality::Left) * Vec3::Y, -Vec3::Y);
        assert_vec3_near(wrist_compensation(Chirality::Right) * Vec3::Z, Vec3::Z);
        assert_vec3_near(wrist_compensation(Chirality::Right) * Vec3::Y, Vec3::X);
        assert_vec3_near(wrist_compensation(Chirality::Left) * Vec3::Z, Vec3::Z);
        assert_vec3_near(wrist_compensation(Chirality::Left) * Vec3::Y, -Vec3::X);
    }

    #[test]
    fn openxr_joints_convert_to_tracked_metacarpal_hand_state() {
        let locations = valid_joint_locations();
        let hand = hand_state_from_openxr_joints(Chirality::Left, &locations)
            .expect("valid joints should produce a hand");

        assert_eq!(hand.unique_id.as_deref(), Some(LEFT_HAND_ID));
        assert_eq!(hand.chirality, Chirality::Left);
        assert!(hand.is_tracking);
        assert!(hand.tracks_metacarpals);
        assert_eq!(hand.segment_positions.len(), SEGMENT_COUNT);
        assert_eq!(hand.segment_rotations.len(), SEGMENT_COUNT);
    }

    #[test]
    fn openxr_joint_positions_are_wrist_relative_after_host_conversion() {
        let mut locations = valid_joint_locations();
        locations[joint_index(xr::HandJoint::WRIST)] =
            valid_location(Vec3::new(0.0, 1.0, -0.5), Quat::IDENTITY);
        locations[joint_index(xr::HandJoint::INDEX_PROXIMAL)] =
            valid_location(Vec3::new(0.1, 1.2, -0.8), Quat::IDENTITY);

        let hand = hand_state_from_openxr_joints(Chirality::Right, &locations)
            .expect("valid joints should produce a hand");
        let (wrist_position, wrist_rotation) =
            openxr_tracking_pose_to_host(Vec3::new(0.0, 1.0, -0.5), Quat::IDENTITY);
        let wrist_rotation = (wrist_rotation * wrist_compensation(Chirality::Right)).normalize();
        let (joint_position, _) =
            openxr_tracking_pose_to_host(Vec3::new(0.1, 1.2, -0.8), Quat::IDENTITY);
        let expected = wrist_rotation.inverse() * (joint_position - wrist_position);

        assert_vec3_near(hand.segment_positions[5], expected);
    }

    #[test]
    fn openxr_joint_rotations_are_wrist_relative_with_finger_compensation() {
        let mut locations = valid_joint_locations();
        let joint_rotation = Quat::from_rotation_y(0.4).normalize();
        locations[joint_index(xr::HandJoint::WRIST)] = valid_location(Vec3::ZERO, Quat::IDENTITY);
        locations[joint_index(xr::HandJoint::MIDDLE_DISTAL)] =
            valid_location(Vec3::new(0.0, 0.0, -0.1), joint_rotation);

        let hand = hand_state_from_openxr_joints(Chirality::Left, &locations)
            .expect("valid joints should produce a hand");
        let (_, host_joint_rotation) = openxr_tracking_pose_to_host(Vec3::ZERO, joint_rotation);
        let wrist_rotation = wrist_compensation(Chirality::Left);
        let expected =
            (wrist_rotation.inverse() * host_joint_rotation * finger_compensation(Chirality::Left))
                .normalize();

        assert_quat_near(hand.segment_rotations[12], expected);
    }

    #[test]
    fn invalid_required_joint_suppresses_hand_state() {
        let mut locations = valid_joint_locations();
        locations[joint_index(xr::HandJoint::INDEX_DISTAL)] = xr::HandJointLocation::default();

        let hand = hand_state_from_openxr_joints(Chirality::Right, &locations);

        assert!(hand.is_none());
    }
}
