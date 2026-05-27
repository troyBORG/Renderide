use super::assignment::{LightCookieAssignment, LightCookieAtlasState};
use super::atlas::{LIGHT_COOKIE_ATLAS_EDGE, white_layer_bytes};
use super::blit::LIGHT_COOKIE_BLIT_2D_STEM;
use super::format::{
    LightCookieAtlasFormat, LightCookieSourceChannel, LightCookieSourceSampling,
    light_cookie_atlas_format_supported, light_cookie_wrap_bits, select_light_cookie_atlas_format,
    source_channel_for_host_format, source_sampling_for_limits,
};
use crate::assets::texture::HostTextureAssetKind;
use crate::gpu::{
    GpuLimits, LIGHT_COOKIE_KIND_DIRECTIONAL_2D, LIGHT_COOKIE_WRAP_MODE_CLAMP,
    LIGHT_COOKIE_WRAP_MODE_MIRROR_ONCE, LIGHT_COOKIE_WRAP_U_SHIFT, LIGHT_COOKIE_WRAP_V_SHIFT,
};
use crate::gpu_pools::SamplerState;
use crate::shared::{LightType, TextureFormat, TextureWrapMode};

use hashbrown::HashMap;

#[test]
fn blit_shader_stem_resolves_to_embedded_wgsl() {
    let wgsl = crate::embedded_shaders::embedded_target_wgsl(LIGHT_COOKIE_BLIT_2D_STEM);
    assert!(wgsl.is_some_and(|source| !source.trim().is_empty()));
}

#[test]
fn atlas_format_prefers_signed_filterable_r16_float() {
    let limits = limits_with_format(
        wgpu::TextureFormat::R16Float,
        wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::COPY_DST,
        wgpu::TextureFormatFeatureFlags::FILTERABLE,
    );

    assert!(light_cookie_atlas_format_supported(
        &limits,
        LightCookieAtlasFormat::R16Float
    ));
    assert_eq!(
        select_light_cookie_atlas_format(&limits),
        LightCookieAtlasFormat::R16Float
    );
}

#[test]
fn atlas_format_falls_back_when_r16_float_is_not_filterable() {
    let limits = limits_with_format_features([
        (
            wgpu::TextureFormat::R16Float,
            wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::COPY_DST,
            wgpu::TextureFormatFeatureFlags::empty(),
        ),
        (
            wgpu::TextureFormat::Rgba16Float,
            wgpu::TextureUsages::empty(),
            wgpu::TextureFormatFeatureFlags::empty(),
        ),
    ]);

    assert!(!light_cookie_atlas_format_supported(
        &limits,
        LightCookieAtlasFormat::R16Float
    ));
    assert_eq!(
        select_light_cookie_atlas_format(&limits),
        LightCookieAtlasFormat::R8Unorm
    );
}

#[test]
fn atlas_format_uses_rgba16_float_when_scalar_hdr_is_unavailable() {
    let limits = limits_with_format_features([
        (
            wgpu::TextureFormat::R16Float,
            wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::COPY_DST,
            wgpu::TextureFormatFeatureFlags::empty(),
        ),
        (
            wgpu::TextureFormat::Rgba16Float,
            wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::COPY_DST,
            wgpu::TextureFormatFeatureFlags::FILTERABLE,
        ),
    ]);

    assert!(!light_cookie_atlas_format_supported(
        &limits,
        LightCookieAtlasFormat::R16Float
    ));
    assert!(light_cookie_atlas_format_supported(
        &limits,
        LightCookieAtlasFormat::Rgba16Float
    ));
    assert_eq!(
        select_light_cookie_atlas_format(&limits),
        LightCookieAtlasFormat::Rgba16Float
    );
}

#[test]
fn source_channel_uses_alpha_for_alpha_capable_formats() {
    for format in [
        TextureFormat::ARGB32,
        TextureFormat::RGBA32,
        TextureFormat::BGRA32,
        TextureFormat::RGBAHalf,
        TextureFormat::ARGBHalf,
        TextureFormat::RGBAFloat,
        TextureFormat::ARGBFloat,
        TextureFormat::BC3,
        TextureFormat::BC7,
        TextureFormat::ETC2RGBA8,
        TextureFormat::ASTC4x4,
    ] {
        assert_eq!(
            source_channel_for_host_format(format),
            LightCookieSourceChannel::Alpha,
            "{format:?}"
        );
    }
}

#[test]
fn source_channel_uses_red_for_scalar_and_no_alpha_formats() {
    for format in [
        TextureFormat::Alpha8,
        TextureFormat::R8,
        TextureFormat::RHalf,
        TextureFormat::RFloat,
        TextureFormat::RGFloat,
        TextureFormat::RGB24,
        TextureFormat::BC1,
        TextureFormat::BC4,
        TextureFormat::ETC2RGB,
    ] {
        assert_eq!(
            source_channel_for_host_format(format),
            LightCookieSourceChannel::Red,
            "{format:?}"
        );
    }
}

#[test]
fn source_sampling_accepts_filtering_and_non_filtering_textures() {
    let filterable = limits_with_format(
        wgpu::TextureFormat::Rgba16Float,
        wgpu::TextureUsages::TEXTURE_BINDING,
        wgpu::TextureFormatFeatureFlags::FILTERABLE,
    );
    let non_filterable = limits_with_format(
        wgpu::TextureFormat::Rgba32Float,
        wgpu::TextureUsages::TEXTURE_BINDING,
        wgpu::TextureFormatFeatureFlags::empty(),
    );
    let not_sampled = limits_with_format(
        wgpu::TextureFormat::Rgba32Float,
        wgpu::TextureUsages::COPY_DST,
        wgpu::TextureFormatFeatureFlags::empty(),
    );

    assert_eq!(
        source_sampling_for_limits(&filterable, wgpu::TextureFormat::Rgba16Float),
        Some(LightCookieSourceSampling::Filtering)
    );
    assert_eq!(
        source_sampling_for_limits(&non_filterable, wgpu::TextureFormat::Rgba32Float),
        Some(LightCookieSourceSampling::NonFiltering)
    );
    assert_eq!(
        source_sampling_for_limits(&not_sampled, wgpu::TextureFormat::Rgba32Float),
        None
    );
}

#[test]
fn white_layer_bytes_match_atlas_formats() {
    let texels = (LIGHT_COOKIE_ATLAS_EDGE * LIGHT_COOKIE_ATLAS_EDGE) as usize;
    let r8 = white_layer_bytes(LightCookieAtlasFormat::R8Unorm);
    assert_eq!(r8.len(), texels);
    assert!(r8.iter().all(|&b| b == 255));

    let r16 = white_layer_bytes(LightCookieAtlasFormat::R16Float);
    assert_eq!(r16.len(), texels * 2);
    assert!(r16.chunks_exact(2).all(|bytes| bytes == [0x00, 0x3c]));

    let rgba16 = white_layer_bytes(LightCookieAtlasFormat::Rgba16Float);
    assert_eq!(rgba16.len(), texels * 8);
    assert!(rgba16.chunks_exact(2).all(|bytes| bytes == [0x00, 0x3c]));
}

#[test]
fn light_cookie_wrap_bits_pack_u_and_v_modes() {
    let sampler = SamplerState {
        wrap_u: TextureWrapMode::Clamp,
        wrap_v: TextureWrapMode::MirrorOnce,
        ..Default::default()
    };
    let bits = light_cookie_wrap_bits(&sampler);

    assert_eq!(
        bits,
        (LIGHT_COOKIE_WRAP_MODE_CLAMP << LIGHT_COOKIE_WRAP_U_SHIFT)
            | (LIGHT_COOKIE_WRAP_MODE_MIRROR_ONCE << LIGHT_COOKIE_WRAP_V_SHIFT)
    );
}

#[test]
fn assigns_directional_cookies_to_2d_atlas() {
    let mut state = LightCookieAtlasState::new();
    let binding = state.assign(
        LightCookieAssignment {
            light_type: LightType::Directional,
            packed_id: 42,
            asset_id: 7,
            kind: HostTextureAssetKind::Texture2D,
            wrap_bits: 0x0d,
        },
        8,
        8,
    );

    assert_eq!(binding.kind, LIGHT_COOKIE_KIND_DIRECTIONAL_2D);
    assert_eq!(binding.layer, 1);
    assert_eq!(binding.wrap_bits, 0x0d);
    let (two_d, point) = state.requests();
    assert_eq!(two_d.len(), 1);
    assert_eq!(two_d[0].asset_id, 7);
    assert!(point.is_empty());
}

fn limits_with_format(
    format: wgpu::TextureFormat,
    allowed_usages: wgpu::TextureUsages,
    flags: wgpu::TextureFormatFeatureFlags,
) -> GpuLimits {
    limits_with_format_features([(format, allowed_usages, flags)])
}

fn limits_with_format_features<const N: usize>(
    features: [(
        wgpu::TextureFormat,
        wgpu::TextureUsages,
        wgpu::TextureFormatFeatureFlags,
    ); N],
) -> GpuLimits {
    let mut format_features = HashMap::new();
    for (format, allowed_usages, flags) in features {
        format_features.insert(
            format,
            wgpu::TextureFormatFeatures {
                allowed_usages,
                flags,
            },
        );
    }
    GpuLimits::synthetic_for_tests(
        wgpu::Limits {
            max_texture_dimension_2d: 4096,
            max_texture_dimension_3d: 4096,
            max_texture_array_layers: 64,
            max_storage_buffer_binding_size: 256 * 1024,
            max_buffer_size: 256 * 1024,
            ..Default::default()
        },
        wgpu::Features::empty(),
        format_features,
    )
}
