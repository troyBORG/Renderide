//! Audits that the Projection360 / ProceduralSkybox shaders reconstruct Unity's
//! "first multi_compile keyword in a no-`_`-placeholder group is the implicit default"
//! semantics, plus the `clamp_intensity` divide-by-zero guard. Regression coverage for
//! the black-render bug introduced by commit `dc04effc` ("Decode Projection360 keywords
//! from `_RenderideVariantBits`").

use super::*;

#[test]
fn projection360_reconstructs_implicit_first_keyword_defaults() -> io::Result<()> {
    for (path_label, src) in [
        (
            "materials/projection360.wgsl",
            material_source("projection360.wgsl")?,
        ),
        (
            "passes/backend/skybox_projection360.wgsl",
            source_file(manifest_dir().join("shaders/passes/backend/skybox_projection360.wgsl"))?,
        ),
    ] {
        for group_const in [
            "P360_GROUP_VIEW",
            "P360_GROUP_OUTSIDE",
            "P360_GROUP_TINT_TEX",
            "P360_GROUP_TEXTURE_MODE",
        ] {
            assert!(
                src.contains(group_const),
                "{path_label} must declare {group_const} -- the alphabetically-first \
                 keyword in each no-`_` multi_compile group is Unity's implicit default \
                 and Froox does not ship that bit when the material picks nothing.",
            );
        }
        assert!(
            src.contains("fn proj360_group_default("),
            "{path_label} must declare proj360_group_default to reconstruct \
             Unity's implicit-first-keyword default for each group.",
        );
        for (helper, group, bit) in [
            ("kw_VIEW", "P360_GROUP_VIEW", "P360_KW_VIEW"),
            (
                "kw_OUTSIDE_CLIP",
                "P360_GROUP_OUTSIDE",
                "P360_KW_OUTSIDE_CLIP",
            ),
            (
                "kw_TINT_TEX_NONE",
                "P360_GROUP_TINT_TEX",
                "P360_KW_TINT_TEX_NONE",
            ),
            (
                "kw_EQUIRECTANGULAR",
                "P360_GROUP_TEXTURE_MODE",
                "P360_KW_EQUIRECTANGULAR",
            ),
        ] {
            let expected = format!("proj360_group_default({group}, {bit})");
            assert!(
                src.contains(&format!("fn {helper}()")) && src.contains(&expected),
                "{path_label} must route {helper}() through {expected} so a material in \
                 the default state (no group bit set) lights up the first keyword.",
            );
        }
    }
    Ok(())
}

#[test]
fn proceduralskybox_defaults_sun_disk_to_high_quality() -> io::Result<()> {
    for (path_label, src) in [
        (
            "materials/proceduralskybox.wgsl",
            material_source("proceduralskybox.wgsl")?,
        ),
        (
            "passes/backend/skybox_proceduralskybox.wgsl",
            source_file(
                manifest_dir().join("shaders/passes/backend/skybox_proceduralskybox.wgsl"),
            )?,
        ),
    ] {
        assert!(
            src.contains("PROCSKY_GROUP_SUNDISK"),
            "{path_label} must declare PROCSKY_GROUP_SUNDISK (no-`_` multi_compile group)",
        );
        assert!(
            src.contains("(mat._RenderideVariantBits & PROCSKY_GROUP_SUNDISK) == 0u")
                && src.contains("procsky_kw(PROCSKY_KW_SUNDISK_HIGH_QUALITY)"),
            "{path_label} must default kw_SUNDISK_HIGH_QUALITY() to true when no \
             _SUNDISK_* bit is set.",
        );
    }
    Ok(())
}

#[test]
fn proceduralskybox_visible_paths_interpolate_unity_vertex_terms() -> io::Result<()> {
    for (path_label, src) in [
        (
            "materials/proceduralskybox.wgsl",
            material_source("proceduralskybox.wgsl")?,
        ),
        (
            "passes/backend/skybox_proceduralskybox.wgsl",
            source_file(
                manifest_dir().join("shaders/passes/backend/skybox_proceduralskybox.wgsl"),
            )?,
        ),
    ] {
        assert!(
            src.contains("ps::visible_vertex_terms(")
                && src.contains("ps::visible_fragment_color("),
            "{path_label} must split ProceduralSkybox into Unity-style vertex sky terms \
             plus fragment horizon/sun mixing.",
        );
        assert!(
            !src.contains("ps::sample("),
            "{path_label} must not call ps::sample in the visible fragment path; that \
             recomputes the discontinuous sky/ground scattering branch per pixel and \
             can draw a bright horizon line.",
        );
    }
    Ok(())
}

#[test]
fn projection360_clamp_intensity_guards_against_zero_max() -> io::Result<()> {
    let src = source_file(manifest_dir().join("shaders/modules/skybox/projection360.wgsl"))?;
    assert!(
        src.contains("if (clamp_intensity && max_intensity > 0.0) {"),
        "clamp_intensity in modules/skybox/projection360.wgsl must guard the divide \
         against max_intensity <= 0.0 so a stray `_CLAMP_INTENSITY = on` with a missing \
         `_MaxIntensity` write doesn't zero every channel and turn the material black.",
    );
    Ok(())
}

/// `prepare_material_skybox` must thread the shader-asset's variant bits into the bind
/// group builder. Hard-coding `None` zeroes `_RenderideVariantBits` in the packed
/// uniform, so every keyword-driven branch in the projection360 / proceduralskybox
/// shaders takes the wrong path -- that was the actual regression `dc04effc` exposed.
#[test]
fn skybox_pass_threads_shader_variant_bits_into_bind_group() -> io::Result<()> {
    let src = source_file(manifest_dir().join("src/passes/world_mesh_forward/skybox.rs"))?;
    assert!(
        src.contains("registry.variant_bits_for_shader_asset(shader_asset_id)"),
        "skybox.rs must look up the shader-asset variant bits before building the \
         material bind group; without them, `_RenderideVariantBits` packs as 0 and the \
         shader takes the all-keywords-off path (Projection360 falls into the equirect \
         fallback and samples the placeholder _MainTex, producing a black sky).",
    );
    assert!(
        src.contains("EmbeddedMaterialBindShader {") && src.contains("shader_variant_bits,"),
        "skybox.rs must pass `shader_variant_bits` through `EmbeddedMaterialBindShader` \
         when calling `embedded_material_bind_group_with_cache_key`.",
    );
    Ok(())
}
