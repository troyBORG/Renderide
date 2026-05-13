//! Fullscreen ProceduralSkybox sky draw.
//!
//! Material struct matches `materials/proceduralskybox.wgsl` so the reflected
//! `@group(1) @binding(0)` layout (which is taken from the material-side shader) matches
//! this pass-side shader's bind requirement. Froox variant bits populate
//! `_RenderideVariantBits`; this shader decodes ProceduralSky's shader-specific keyword
//! bits locally. `UNITY_COLORSPACE_GAMMA` is reserved in the bit table but never consulted
//! because the renderer is linear-only. The sun-disk group has no `_` placeholder, so the
//! high-quality keyword is the zero-bit default.

#import renderide::frame::globals as rg
#import renderide::skybox::procedural as ps
#import renderide::skybox::common as skybox
#import renderide::material::variant_bits as vb

struct ProceduralSkyboxMaterial {
    _SkyTint: vec4<f32>,
    _GroundColor: vec4<f32>,
    _SunColor: vec4<f32>,
    _SunDirection: vec4<f32>,
    _Exposure: f32,
    _SunSize: f32,
    _AtmosphereThickness: f32,
    _RenderideVariantBits: u32,
}

const PROCSKY_KW_SUNDISK_HIGH_QUALITY: u32 = 1u << 0u;
const PROCSKY_KW_SUNDISK_NONE: u32 = 1u << 1u;
const PROCSKY_KW_SUNDISK_SIMPLE: u32 = 1u << 2u;
const PROCSKY_KW_UNITY_COLORSPACE_GAMMA: u32 = 1u << 3u;
const PROCSKY_GROUP_SUNDISK: u32 =
    PROCSKY_KW_SUNDISK_HIGH_QUALITY | PROCSKY_KW_SUNDISK_NONE | PROCSKY_KW_SUNDISK_SIMPLE;

@group(1) @binding(0) var<uniform> mat: ProceduralSkyboxMaterial;
@group(2) @binding(0) var<uniform> view: skybox::SkyboxView;

fn procsky_kw(mask: u32) -> bool {
    return vb::enabled(mat._RenderideVariantBits, mask);
}

fn kw_SUNDISK_NONE() -> bool {
    return procsky_kw(PROCSKY_KW_SUNDISK_NONE);
}

fn kw_SUNDISK_HIGH_QUALITY() -> bool {
    return (mat._RenderideVariantBits & PROCSKY_GROUP_SUNDISK) == 0u
        || procsky_kw(PROCSKY_KW_SUNDISK_HIGH_QUALITY);
}

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) ground_color: vec3<f32>,
    @location(1) sky_color: vec3<f32>,
    @location(2) sun_color: vec3<f32>,
    @location(3) fragment_ray: vec3<f32>,
    @location(4) sky_ground_factor: f32,
}

fn procedural_sky_params() -> ps::ProceduralSkyParams {
    return ps::ProceduralSkyParams(
        mat._SkyTint.rgb,
        mat._GroundColor.rgb,
        mat._SunColor.rgb,
        mat._SunDirection.xyz,
        mat._Exposure,
        mat._SunSize,
        mat._AtmosphereThickness,
        procedural_sun_disk_mode(),
    );
}

fn procedural_sun_disk_mode() -> f32 {
    if (kw_SUNDISK_NONE()) {
        return 0.0;
    }
    if (kw_SUNDISK_HIGH_QUALITY()) {
        return 2.0;
    }
    return 1.0;
}

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
#ifdef MULTIVIEW
    @builtin(view_index) view_idx: u32,
#endif
) -> VertexOutput {
    let clip = skybox::fullscreen_quad_clip_pos(vertex_index);
    var out: VertexOutput;
    out.clip_pos = clip;
#ifdef MULTIVIEW
    let view_layer = view_idx;
#else
    let view_layer = 0u;
#endif
    let ndc = vec2<f32>(clip.x, clip.y * view.ndc_y_sign_pad.x);
    let proj_params = select(rg::frame.proj_params_left, rg::frame.proj_params_right, view_layer != 0u);
    let view_ray = skybox::view_ray_from_ndc(
        ndc,
        proj_params,
        skybox::view_is_orthographic(view, view_layer),
    );
    let world_ray = skybox::world_ray_from_view_ray(view_ray, view, view_layer);
    let terms = ps::visible_vertex_terms(procedural_sky_params(), world_ray);
    out.ground_color = terms.ground_color;
    out.sky_color = terms.sky_color;
    out.sun_color = terms.sun_color;
    out.fragment_ray = terms.fragment_ray;
    out.sky_ground_factor = terms.sky_ground_factor;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let terms = ps::ProceduralSkyVisibleTerms(
        in.ground_color,
        in.sky_color,
        in.sun_color,
        in.fragment_ray,
        in.sky_ground_factor,
    );
    return rg::retain_globals_additive(vec4<f32>(
        ps::visible_fragment_color(procedural_sky_params(), terms),
        1.0,
    ));
}
