//! Shared direct-light intensity and attenuation helpers.

#import renderide::frame::types as ft

#define_import_path renderide::lighting::birp

/// Quadratic coefficient used by Unity BiRP's normalized punctual-light attenuation LUT.
const BIRP_ATTENUATION_QUADRATIC: f32 = 25.0;

/// Temporary direct-light multiplier used to match BiRP-authored scene brightness.
const INTENSITY_BOOST: f32 = 3.1415927;

/// Unity BiRP-style direct light intensity with scene-brightness boost applied.
fn direct_light_intensity(intensity: f32) -> f32 {
    return intensity * INTENSITY_BOOST;
}

/// Squared edge fade for a precomputed attenuation curve input.
fn squared_edge_fade(t: f32) -> f32 {
    let fade = clamp(1.0 - t, 0.0, 1.0);
    return fade * fade;
}

/// Quartic window that smoothly fades normalized attenuation to zero at the boundary.
fn range_fade(t: f32) -> f32 {
    let t2 = t * t;
    return squared_edge_fade(t2 * t2);
}

/// Unity BiRP-style distance attenuation for punctual lights.
///
/// `1 / (1 + 25*t^2)` with `t = dist/range` approximates the Built-in RP attenuation LUT while
/// keeping the light's peak brightness independent of range. The quartic range window prevents
/// clustered lights from leaking past their declared range.
fn distance_attenuation(dist: f32, range: f32) -> f32 {
    if (range <= 0.0) {
        return 0.0;
    }
    let t = dist / range;
    let t2 = t * t;
    let lut = 1.0 / (1.0 + BIRP_ATTENUATION_QUADRATIC * t2);
    return lut * range_fade(t) * INTENSITY_BOOST;
}

/// Unity BiRP-style punctual attenuation with light intensity applied.
fn punctual_attenuation(intensity: f32, dist: f32, range: f32) -> f32 {
    return intensity * distance_attenuation(dist, range);
}

fn normalized_spot_direction(direction: vec3<f32>) -> vec3<f32> {
    let len_sq = dot(direction, direction);
    if (len_sq <= 1e-16) {
        return vec3<f32>(0.0, 0.0, 1.0);
    }
    return direction * inverseSqrt(len_sq);
}

/// Projected radial quadratic spotlight cone attenuation.
///
/// `l` is the normalized direction from the shaded surface toward the light. The packed light
/// stores the outer half-angle cosine plus the reciprocal squared tangent of that angle.
fn spot_angle_attenuation(light: ft::GpuLight, l: vec3<f32>) -> f32 {
    let rho = dot(-l, normalized_spot_direction(light.direction));
    if (rho <= light.spot_cos_half_angle) {
        return 0.0;
    }
    if (light.spot_angle_scale <= 0.0) {
        return select(0.0, 1.0, light.spot_cos_half_angle <= 0.0);
    }
    let rho2 = max(rho * rho, 1e-6);
    let tan2_theta = max(1.0 - rho2, 0.0) / rho2;
    let r2 = clamp(tan2_theta * light.spot_angle_scale, 0.0, 1.0);
    return squared_edge_fade(r2);
}
