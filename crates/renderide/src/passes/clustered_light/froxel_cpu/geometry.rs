use crate::gpu::GpuLight;
use crate::world_mesh::cluster::ClusterFrameParams;

use super::bounds::{BoundedLight, FroxelSphere, LIGHT_TYPE_DIRECTIONAL, light_froxel_bounds};
use super::types::FroxelLayout;

pub(super) fn assign_eye_lights(
    lights: &[GpuLight],
    params: ClusterFrameParams,
    layout: FroxelLayout,
    froxel_spheres: &[FroxelSphere],
    cluster_base: usize,
    emit: &mut impl FnMut(usize, u32),
) -> u32 {
    assign_eye_lights_slice(
        lights,
        0,
        params,
        layout,
        froxel_spheres,
        cluster_base,
        emit,
    )
}

pub(super) fn assign_eye_lights_slice(
    lights: &[GpuLight],
    light_index_base: usize,
    params: ClusterFrameParams,
    layout: FroxelLayout,
    froxel_spheres: &[FroxelSphere],
    cluster_base: usize,
    emit: &mut impl FnMut(usize, u32),
) -> u32 {
    let view = params.world_to_view;
    let view_scale = params.world_to_view_scale_max();
    let mut culled_lights = 0u32;
    for (local_light_idx, light) in lights.iter().enumerate() {
        let Some(light_idx) = light_index_base
            .checked_add(local_light_idx)
            .and_then(|idx| u32::try_from(idx).ok())
        else {
            culled_lights = culled_lights.saturating_add(1);
            continue;
        };
        if light.light_type == LIGHT_TYPE_DIRECTIONAL {
            assign_directional(light_idx, layout, cluster_base, emit);
            continue;
        }
        let Some(bounds) =
            light_froxel_bounds(light, view, params.proj, view_scale, layout, params)
        else {
            culled_lights = culled_lights.saturating_add(1);
            continue;
        };
        assign_bounded_light(
            light_idx,
            bounds,
            layout,
            froxel_spheres,
            cluster_base,
            emit,
        );
    }
    culled_lights
}

fn assign_directional(
    light_idx: u32,
    layout: FroxelLayout,
    cluster_base: usize,
    emit: &mut impl FnMut(usize, u32),
) {
    let Some(cluster_count) = layout.cluster_count() else {
        return;
    };
    for cluster_local in 0..cluster_count {
        emit(cluster_base + cluster_local, light_idx);
    }
}

/// Assigns a bounded local light to its touched froxel range.
fn assign_bounded_light(
    light_idx: u32,
    light: BoundedLight,
    layout: FroxelLayout,
    froxel_spheres: &[FroxelSphere],
    cluster_base: usize,
    emit: &mut impl FnMut(usize, u32),
) {
    for z in light.bounds.z0..=light.bounds.z1 {
        for y in light.bounds.y0..=light.bounds.y1 {
            for x in light.bounds.x0..=light.bounds.x1 {
                let local = x + layout.cluster_count_x * (y + layout.cluster_count_y * z);
                let local_usize = local as usize;
                if let Some(spot) = light.spot {
                    let Some(&froxel_sphere) = froxel_spheres.get(local_usize) else {
                        continue;
                    };
                    if !spot.intersects_froxel_sphere(froxel_sphere) {
                        continue;
                    }
                }
                emit(cluster_base + local_usize, light_idx);
            }
        }
    }
}
