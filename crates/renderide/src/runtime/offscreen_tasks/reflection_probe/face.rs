//! Per-face geometry for cubemap captures: face enum, world-space basis, view descriptor, and the
//! per-face host-camera frame plus the small clear/clip/filter helpers shared by the queue path
//! and the OnChanges multi-tick capture path.

use glam::{Mat4, Vec3};
use hashbrown::HashSet;

use crate::camera::{
    CameraClipPlanes, CameraPose, CameraProjectionKind, EyeView, HostCameraFrame, Viewport,
};
use crate::render_graph::{FrameViewClear, ViewPostProcessing};
use crate::scene::reflection_probe_skybox_only;
use crate::shared::{CameraClearMode, ReflectionProbeClear, ReflectionProbeState};
use crate::world_mesh::CameraTransformDrawFilter;

/// Number of faces in a cubemap (six).
pub(super) const CUBE_FACE_COUNT: usize = crate::gpu::CUBEMAP_ARRAY_LAYERS as usize;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum ProbeCubeFace {
    PosX,
    NegX,
    PosY,
    NegY,
    PosZ,
    NegZ,
}

impl ProbeCubeFace {
    pub(super) const ALL: [Self; CUBE_FACE_COUNT] = [
        Self::PosX,
        Self::NegX,
        Self::PosY,
        Self::NegY,
        Self::PosZ,
        Self::NegZ,
    ];
    pub(super) const ALL_MASK: u8 = 0b0011_1111;

    pub(super) const fn index(self) -> usize {
        match self {
            Self::PosX => 0,
            Self::NegX => 1,
            Self::PosY => 2,
            Self::NegY => 3,
            Self::PosZ => 4,
            Self::NegZ => 5,
        }
    }

    pub(super) const fn layer(self) -> u32 {
        self.index() as u32
    }

    pub(super) const fn view_id_face_index(self) -> u8 {
        self.index() as u8
    }

    pub(super) const fn bit(self) -> u8 {
        1 << self.index()
    }

    pub(super) const fn basis(self) -> ProbeFaceBasis {
        match self {
            Self::PosX => ProbeFaceBasis {
                forward: Vec3::X,
                right: Vec3::NEG_Z,
                up: Vec3::Y,
            },
            Self::NegX => ProbeFaceBasis {
                forward: Vec3::NEG_X,
                right: Vec3::Z,
                up: Vec3::Y,
            },
            Self::PosY => ProbeFaceBasis {
                forward: Vec3::Y,
                right: Vec3::X,
                up: Vec3::NEG_Z,
            },
            Self::NegY => ProbeFaceBasis {
                forward: Vec3::NEG_Y,
                right: Vec3::X,
                up: Vec3::Z,
            },
            Self::PosZ => ProbeFaceBasis {
                forward: Vec3::Z,
                right: Vec3::X,
                up: Vec3::Y,
            },
            Self::NegZ => ProbeFaceBasis {
                forward: Vec3::NEG_Z,
                right: Vec3::NEG_X,
                up: Vec3::Y,
            },
        }
    }

    #[cfg(test)]
    pub(super) fn direction_for_uv(self, u: f32, v: f32) -> Vec3 {
        let x = 2.0 * u - 1.0;
        let y = 1.0 - 2.0 * v;
        let basis = self.basis();
        (basis.forward + basis.right * x + basis.up * y).normalize()
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct ProbeFaceBasis {
    pub(super) forward: Vec3,
    pub(super) right: Vec3,
    pub(super) up: Vec3,
}

pub(super) fn face_view_desc(
    label: &'static str,
    face: u32,
    format: wgpu::TextureFormat,
    usage: wgpu::TextureUsages,
) -> wgpu::TextureViewDescriptor<'static> {
    wgpu::TextureViewDescriptor {
        label: Some(label),
        format: Some(format),
        dimension: Some(wgpu::TextureViewDimension::D2),
        usage: Some(usage),
        aspect: wgpu::TextureAspect::All,
        base_mip_level: 0,
        mip_level_count: Some(1),
        base_array_layer: face,
        array_layer_count: Some(1),
    }
}

pub(super) fn probe_face_world_matrix(position: Vec3, face: ProbeCubeFace) -> Mat4 {
    let basis = face.basis();
    Mat4::from_cols(
        basis.right.extend(0.0),
        basis.up.extend(0.0),
        basis.forward.extend(0.0),
        position.extend(1.0),
    )
}

pub(super) fn host_camera_frame_for_probe_face(
    base: &HostCameraFrame,
    state: ReflectionProbeState,
    viewport_px: (u32, u32),
    position: Vec3,
    face: ProbeCubeFace,
) -> HostCameraFrame {
    let clip = reflection_probe_clip(state);
    let world_matrix = probe_face_world_matrix(position, face);
    let pose = CameraPose::from_world_matrix(world_matrix);
    let viewport = Viewport::from_tuple(viewport_px);
    HostCameraFrame {
        frame_index: base.frame_index,
        clip,
        desktop_fov_degrees: 90.0,
        vr_active: false,
        output_device: base.output_device,
        projection_kind: CameraProjectionKind::Perspective,
        primary_ortho_task: None,
        stereo: None,
        head_output_transform: base.head_output_transform,
        explicit_view: Some(EyeView::from_pose_projection(
            pose,
            if viewport.is_empty() {
                Mat4::IDENTITY
            } else {
                EyeView::perspective_from_pose(pose, viewport, 90.0, clip).proj
            },
        )),
        eye_world_position: Some(position),
        suppress_occlusion_temporal: true,
    }
}

pub(super) fn reflection_probe_clip(state: ReflectionProbeState) -> CameraClipPlanes {
    let near = finite_positive_or(state.near_clip, CameraClipPlanes::default().near).max(0.01);
    let far_default = CameraClipPlanes::default().far;
    let far = finite_positive_or(state.far_clip, far_default).max(near + 0.01);
    CameraClipPlanes::new(near, far)
}

pub(super) fn finite_positive_or(value: f32, fallback: f32) -> f32 {
    if value.is_finite() && value > 0.0 {
        value
    } else {
        fallback
    }
}

pub(super) fn clear_from_reflection_probe_state(state: ReflectionProbeState) -> FrameViewClear {
    if state.clear_flags == ReflectionProbeClear::Color {
        FrameViewClear {
            mode: CameraClearMode::Color,
            color: state.background_color,
        }
    } else {
        FrameViewClear::skybox()
    }
}

pub(super) fn draw_filter_from_reflection_probe_state(
    state: &ReflectionProbeState,
) -> CameraTransformDrawFilter {
    if reflection_probe_skybox_only(state.flags) {
        CameraTransformDrawFilter {
            only: Some(HashSet::new()),
            exclude: HashSet::new(),
        }
    } else {
        CameraTransformDrawFilter {
            only: None,
            exclude: HashSet::new(),
        }
    }
}

pub(super) fn reflection_probe_bake_post_processing() -> ViewPostProcessing {
    ViewPostProcessing::disabled()
}
