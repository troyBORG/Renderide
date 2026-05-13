//! OpenXR frame wait/locate and host camera sync.

use std::sync::atomic::{AtomicU32, Ordering};

use crate::camera::{StereoViewMatrices, effective_head_output_clip_planes};
use crate::gpu::GpuQueueAccessGate;
use crate::xr::{XrHostCameraSync, XrWgpuHandles};

use super::types::OpenxrFrameTick;

static WAIT_FRAME_FAILURE_STREAK: AtomicU32 = AtomicU32::new(0);
static LOCATE_VIEWS_FAILURE_STREAK: AtomicU32 = AtomicU32::new(0);

/// Single `wait_frame` + `locate_views` for stereo uniforms; used for both mirror and HMD paths.
pub fn openxr_begin_frame_tick(
    handles: &mut XrWgpuHandles,
    runtime: &mut impl XrHostCameraSync,
    gpu_queue_access_gate: &GpuQueueAccessGate,
) -> Option<OpenxrFrameTick> {
    profiling::scope!("xr::begin_frame_tick");
    {
        profiling::scope!("xr::poll_events");
        match handles.xr_session.poll_events() {
            Ok(_) => {}
            Err(e) => logger::warn!("OpenXR poll_events failed: {e:?}"),
        }
    }
    if handles.xr_session.exit_requested() {
        // Exit is driven by the app loop reading `exit_requested()`; here we just skip starting a
        // new frame so we don't `xrBeginFrame` on a terminating session.
        return None;
    }
    let fs = {
        profiling::scope!("xr::wait_frame");
        match handles.xr_session.wait_frame(gpu_queue_access_gate) {
            Ok(Some(state)) => {
                WAIT_FRAME_FAILURE_STREAK.store(0, Ordering::Relaxed);
                state
            }
            Ok(None) => return None,
            Err(e) => {
                let streak = WAIT_FRAME_FAILURE_STREAK.fetch_add(1, Ordering::Relaxed) + 1;
                logger::warn!("OpenXR wait_frame failed: {e:?} consecutive_failures={streak}");
                runtime.note_openxr_wait_frame_failed();
                return None;
            }
        }
    };
    let views = if fs.should_render {
        profiling::scope!("xr::locate_views");
        match handles.xr_session.locate_views(fs.predicted_display_time) {
            Ok(v) => {
                LOCATE_VIEWS_FAILURE_STREAK.store(0, Ordering::Relaxed);
                v
            }
            Err(e) => {
                let streak = LOCATE_VIEWS_FAILURE_STREAK.fetch_add(1, Ordering::Relaxed) + 1;
                logger::warn!("OpenXR locate_views failed: {e:?} consecutive_failures={streak}");
                runtime.note_openxr_locate_views_failed();
                Vec::new()
            }
        }
    } else {
        Vec::new()
    };
    if views.len() >= 2 {
        if runtime.vr_active() {
            let (near, far) = effective_head_output_clip_planes(
                runtime.near_clip(),
                runtime.far_clip(),
                runtime.output_device(),
                runtime.scene_root_scale_for_clip(),
            );
            let center_pose = crate::xr::headset_center_pose_from_stereo_views(&views);
            let world_from_tracking = runtime.world_from_tracking(center_pose);
            runtime.set_head_output_transform(world_from_tracking);
            let left =
                crate::xr::eye_view_from_xr_view_aligned(&views[0], near, far, world_from_tracking);
            let right =
                crate::xr::eye_view_from_xr_view_aligned(&views[1], near, far, world_from_tracking);
            runtime.set_eye_world_position((left.world_position + right.world_position) * 0.5);
            let stereo = StereoViewMatrices::new(left, right);
            runtime.set_stereo(Some(&stereo));
            return Some(OpenxrFrameTick {
                predicted_display_time: fs.predicted_display_time,
                should_render: fs.should_render,
                views,
            });
        }
        // Desktop (`!vr_active`): keep [`HostCameraFrame::head_output_transform`] from
        // [`RendererRuntime::apply_frame_submit_data`](crate::runtime::RendererRuntime::apply_frame_submit_data)
        // (host `root_transform`). OpenXR still supplies views for IPC pose.
        return Some(OpenxrFrameTick {
            predicted_display_time: fs.predicted_display_time,
            should_render: fs.should_render,
            views,
        });
    }
    Some(OpenxrFrameTick {
        predicted_display_time: fs.predicted_display_time,
        should_render: fs.should_render,
        views,
    })
}
