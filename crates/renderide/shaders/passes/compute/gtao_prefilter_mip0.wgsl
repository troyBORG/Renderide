//! Compute pass: raw reverse-Z depth -> GTAO view-space depth mip 0.

#import renderide::frame::types as ft
#import renderide::post::gtao_params as gparams

#ifdef MULTIVIEW
@group(0) @binding(0) var raw_depth: texture_depth_2d_array;
#else
@group(0) @binding(0) var raw_depth: texture_depth_2d;
#endif
@group(0) @binding(1) var<uniform> frame: ft::FrameGlobals;
@group(0) @binding(2) var<uniform> gtao: gparams::GtaoParams;
#ifdef MULTIVIEW
@group(0) @binding(3) var dst_mip0: texture_storage_2d_array<r32float, write>;
#else
@group(0) @binding(3) var dst_mip0: texture_storage_2d<r32float, write>;
#endif

fn projection_flags_for_layer(layer: u32) -> u32 {
#ifdef MULTIVIEW
    if ((layer & 1u) != 0u) {
        return frame.frame_tail.z;
    }
#endif
    return frame.frame_tail.y;
}

fn view_is_orthographic(layer: u32) -> bool {
    let flags = projection_flags_for_layer(layer);
    return (flags & ft::FRAME_PROJECTION_FLAG_ORTHOGRAPHIC) != 0u;
}

fn linearize_depth(d: f32, layer: u32) -> f32 {
    let near = frame.near_clip;
    let far = frame.far_clip;
    if (view_is_orthographic(layer)) {
        return far - d * (far - near);
    }
    let denom = d * (far - near) + near;
    return (near * far) / max(denom, 1e-6);
}

@compute @workgroup_size(8, 8, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dim = textureDimensions(dst_mip0);
    if (gid.x >= dim.x || gid.y >= dim.y) {
        return;
    }

    let pix = vec2<i32>(i32(gid.x), i32(gid.y));
#ifdef MULTIVIEW
    let layer = gid.z;
    let raw = textureLoad(raw_depth, pix, i32(layer), 0);
    let view_z = select(0.0, linearize_depth(raw, layer), raw > 0.0);
    textureStore(dst_mip0, pix, i32(layer), vec4<f32>(view_z, 0.0, 0.0, 1.0));
#else
    let raw = textureLoad(raw_depth, pix, 0);
    let view_z = select(0.0, linearize_depth(raw, 0u), raw > 0.0);
    textureStore(dst_mip0, pix, vec4<f32>(view_z, 0.0, 0.0, 1.0));
#endif
}
