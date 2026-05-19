//! Helpers for sampling the renderer-produced scene-depth snapshot.

#define_import_path renderide::frame::scene_depth_sample

#import renderide::frame::globals as rg

fn scene_depth_max_xy() -> vec2<i32> {
    return vec2<i32>(
        i32(rg::frame.viewport_width) - 1,
        i32(rg::frame.viewport_height) - 1,
    );
}

fn scene_depth_xy_from_frag_pos(frag_pos: vec4<f32>) -> vec2<i32> {
    return clamp(vec2<i32>(frag_pos.xy), vec2<i32>(0, 0), scene_depth_max_xy());
}

fn scene_depth_xy_from_uv(uv: vec2<f32>) -> vec2<i32> {
    let clamped_uv = clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0));
    let size = vec2<f32>(f32(rg::frame.viewport_width), f32(rg::frame.viewport_height));
    return clamp(vec2<i32>(clamped_uv * size), vec2<i32>(0, 0), scene_depth_max_xy());
}

fn linear_depth_from_raw(raw_depth: f32, view_layer: u32) -> f32 {
    if (rg::view_is_orthographic(view_layer)) {
        return rg::frame.far_clip - raw_depth * (rg::frame.far_clip - rg::frame.near_clip);
    }
    let denom = max(
        raw_depth * (rg::frame.far_clip - rg::frame.near_clip) + rg::frame.near_clip,
        1e-6,
    );
    return (rg::frame.near_clip * rg::frame.far_clip) / denom;
}

fn raw_depth_at_xy(xy: vec2<i32>, view_layer: u32) -> f32 {
#ifdef MULTIVIEW
    return textureLoad(rg::scene_depth_array, xy, i32(rg::view_index_from_layer(view_layer)), 0);
#else
    return textureLoad(rg::scene_depth, xy, 0);
#endif
}

fn scene_linear_depth_at_xy(xy: vec2<i32>, view_layer: u32) -> f32 {
    return linear_depth_from_raw(raw_depth_at_xy(xy, view_layer), view_layer);
}

fn scene_linear_depth(frag_pos: vec4<f32>, view_layer: u32) -> f32 {
    return scene_linear_depth_at_xy(scene_depth_xy_from_frag_pos(frag_pos), view_layer);
}

fn scene_linear_depth_at_uv(uv: vec2<f32>, view_layer: u32) -> f32 {
    return scene_linear_depth_at_xy(scene_depth_xy_from_uv(uv), view_layer);
}

fn fragment_linear_depth(world_pos: vec3<f32>, view_layer: u32) -> f32 {
    let z_coeffs = rg::view_space_z_coeffs_for_view(view_layer);
    let view_z = dot(z_coeffs.xyz, world_pos) + z_coeffs.w;
    return -view_z;
}

fn depth_fade(frag_pos: vec4<f32>, world_pos: vec3<f32>, view_layer: u32, divisor: f32) -> f32 {
    let denom = max(abs(divisor), 1e-6);
    let diff = scene_linear_depth(frag_pos, view_layer) - fragment_linear_depth(world_pos, view_layer);
    return clamp(diff / denom, 0.0, 1.0);
}

fn depth_fade_at_uv(uv: vec2<f32>, world_pos: vec3<f32>, view_layer: u32, divisor: f32) -> f32 {
    let denom = max(abs(divisor), 1e-6);
    let diff = scene_linear_depth_at_uv(uv, view_layer) - fragment_linear_depth(world_pos, view_layer);
    return clamp(diff / denom, 0.0, 1.0);
}
