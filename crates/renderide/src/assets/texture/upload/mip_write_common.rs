//! Shared mip offset validation and [`wgpu::Queue::write_texture`] layout for full mip and subregion paths.

use crate::shared::SetTexture2DData;

use super::super::decode::decode_mip_to_rgba8;
use super::super::layout::{host_format_is_compressed, mip_byte_len};
use super::error::TextureUploadError;
use super::mip_chain_walk::{MipChainStop, resolve_mip_payload_slot};
use crate::gpu::GpuQueueAccessMode;

/// Format-side context shared by every mip in one texture upload (2D, cubemap, 3D).
///
/// Bundled so the per-mip decode functions don't take the same four handles on every call.
/// Fields are [`Copy`] so the context can be captured into a `rayon::spawn` closure by value.
#[derive(Copy, Clone)]
pub(super) struct MipUploadFormatCtx {
    /// Host asset id for logging and diagnostics.
    pub asset_id: i32,
    /// Host-side texel format from the upload descriptor.
    pub fmt_format: crate::shared::TextureFormat,
    /// GPU-facing texel format the material system expects.
    pub wgpu_format: wgpu::TextureFormat,
    /// Whether host bytes must be decoded to RGBA8 before upload.
    pub needs_rgba8_decode: bool,
}

/// CPU-side bytes for one mip plus the storage-orientation flag.
///
/// The renderer uploads host texture bytes as-is (Unity V=0 bottom). For host-uploaded 2D
/// textures, `storage_v_inverted` is `true` because the bytes are in Unity orientation. Cubemap
/// upload paths override the flag because their face layout is already native cube orientation.
/// For renderer-baked sources it is `false`.
#[derive(Debug)]
pub(super) struct MipUploadPixels {
    /// Bytes ready for [`wgpu::Queue::write_texture`].
    pub bytes: Vec<u8>,
    /// Whether the bytes need shader-side storage-orientation compensation.
    pub storage_v_inverted: bool,
}

impl MipUploadPixels {
    /// Builds an upload from host bytes (Unity V=0 bottom).
    pub fn host(bytes: Vec<u8>) -> Self {
        Self {
            bytes,
            storage_v_inverted: true,
        }
    }

    /// Backwards-compatible alias for [`Self::host`] retained for renderer-baked tail synthesis.
    pub fn normal(bytes: Vec<u8>) -> Self {
        Self::host(bytes)
    }

    /// Returns this upload with an explicit storage-orientation flag.
    pub fn with_storage_v_inverted(mut self, storage_v_inverted: bool) -> Self {
        self.storage_v_inverted = storage_v_inverted;
        self
    }
}

/// Texture family being converted for upload diagnostics.
#[derive(Copy, Clone, Debug)]
pub(super) enum MipUploadKind {
    /// Texture2D mip-chain upload.
    Texture2d,
    /// Cubemap face upload.
    Cubemap {
        /// Cubemap face index.
        face: u32,
    },
}

/// Per-mip label used to keep shared conversion diagnostics clear.
#[derive(Copy, Clone, Debug)]
pub(super) struct MipUploadLabel {
    /// Upload family.
    pub kind: MipUploadKind,
    /// Mip index inside the texture or cubemap face.
    pub mip_index: usize,
}

impl MipUploadLabel {
    /// Builds a label for a Texture2D mip.
    pub fn texture2d(mip_index: usize) -> Self {
        Self {
            kind: MipUploadKind::Texture2d,
            mip_index,
        }
    }

    /// Builds a label for one cubemap face mip.
    pub fn cubemap(face: u32, mip_index: usize) -> Self {
        Self {
            kind: MipUploadKind::Cubemap { face },
            mip_index,
        }
    }

    /// Asset-qualified diagnostic label.
    fn asset_mip(self, asset_id: i32) -> String {
        match self.kind {
            MipUploadKind::Texture2d => format!("texture {asset_id} mip {}", self.mip_index),
            MipUploadKind::Cubemap { face } => {
                format!("cubemap {asset_id} face {face} mip {}", self.mip_index)
            }
        }
    }
}

/// Whether a sampled Texture2D upload keeps host bytes in Unity orientation.
pub(crate) fn upload_uses_storage_v_inversion(
    _host_format: crate::shared::TextureFormat,
    _wgpu_format: wgpu::TextureFormat,
    _flip_y: bool,
) -> bool {
    true
}

/// Picks the descriptor offset bias that maximizes how many mips fit in the SHM payload.
pub(super) fn choose_mip_start_bias(
    format: crate::shared::TextureFormat,
    upload: &SetTexture2DData,
    payload_len: usize,
) -> Result<(usize, usize), TextureUploadError> {
    let offset_bias = upload.data.offset.max(0) as usize;
    let candidates = if offset_bias > 0 {
        [0usize, offset_bias]
    } else {
        [0usize, 0usize]
    };
    let mut best_bias = 0usize;
    let mut best_prefix = 0usize;
    for bias in candidates {
        let prefix = valid_mip_prefix_len(format, upload, payload_len, bias)?;
        if prefix > best_prefix {
            best_prefix = prefix;
            best_bias = bias;
        }
    }
    if best_prefix == 0 {
        return Err(TextureUploadError::from(format!(
            "mip region exceeds shared memory descriptor (payload_len={payload_len}, descriptor_offset={offset_bias})"
        )));
    }
    Ok((best_bias, best_prefix))
}

/// Counts how many descriptor mips fit inside `payload_len` after applying `bias`.
pub(super) fn valid_mip_prefix_len(
    format: crate::shared::TextureFormat,
    upload: &SetTexture2DData,
    payload_len: usize,
    bias: usize,
) -> Result<usize, TextureUploadError> {
    let mut count = 0usize;
    for (i, sz) in upload.mip_map_sizes.iter().enumerate() {
        if sz.x <= 0 || sz.y <= 0 {
            return Err("non-positive mip dimensions".into());
        }
        let w = sz.x as u32;
        let h = sz.y as u32;
        let host_len = mip_byte_len(format, w, h).ok_or_else(|| {
            TextureUploadError::from(format!("mip byte size unsupported for {format:?}"))
        })? as usize;
        let start_raw = upload.mip_starts[i];
        match resolve_mip_payload_slot(format, host_len, start_raw, bias, payload_len, || {
            format!("mip {i}")
        })? {
            Ok(()) => count += 1,
            Err(
                MipChainStop::NegativeStart | MipChainStop::BeforeBias | MipChainStop::OutOfPayload,
            ) => break,
        }
    }
    Ok(count)
}

/// Returns whether `gpu` is an RGBA8 texture format accepted by the direct upload path.
pub(super) fn is_rgba8_family(gpu: wgpu::TextureFormat) -> bool {
    matches!(
        gpu,
        wgpu::TextureFormat::Rgba8Unorm | wgpu::TextureFormat::Rgba8UnormSrgb
    )
}

/// Validates host mip bytes and produces a buffer for a full-mip texture upload.
///
/// The renderer no longer flips host data on ingestion. Sampled textures use Unity (V=0 bottom)
/// orientation throughout: host bytes are stored as-is, mesh UVs match, and shaders apply no V flip.
/// Cubemap face orientation is the only remaining storage-orientation concern; that is handled in
/// the cubemap pool and the projection360 sampling helpers, not here.
///
/// `_flip_y` is currently ignored: the host contract is to send Unity-oriented bytes regardless.
/// The parameter is retained for IPC compatibility and to keep call sites stable while the host
/// continues to set it.
pub(super) fn mip_src_to_upload_pixels(
    ctx: MipUploadFormatCtx,
    width: u32,
    height: u32,
    _flip_y: bool,
    mip_src: &[u8],
    label: MipUploadLabel,
) -> Result<MipUploadPixels, TextureUploadError> {
    profiling::scope!("asset::texture_convert_mip_pixels");
    let MipUploadFormatCtx {
        asset_id,
        fmt_format,
        wgpu_format,
        needs_rgba8_decode,
    } = ctx;
    let bytes_result: Result<Vec<u8>, TextureUploadError> = if is_rgba8_family(wgpu_format) {
        if needs_rgba8_decode || host_format_is_compressed(fmt_format) {
            decode_mip_to_rgba8(fmt_format, width, height, false, mip_src).ok_or_else(|| {
                TextureUploadError::from(format!(
                    "RGBA decode failed for {} ({:?})",
                    label.asset_mip(asset_id),
                    fmt_format
                ))
            })
        } else {
            Ok(mip_src.to_vec())
        }
    } else if needs_rgba8_decode {
        Err(TextureUploadError::from(format!(
            "host {fmt_format:?} must use RGBA decode but GPU format is {wgpu_format:?}"
        )))
    } else if host_format_is_compressed(fmt_format) {
        let expected_len = mip_byte_len(fmt_format, width, height).ok_or_else(|| {
            TextureUploadError::from(format!(
                "{}: mip byte size unknown for {:?}",
                label.asset_mip(asset_id),
                fmt_format
            ))
        })? as usize;
        if mip_src.len() != expected_len {
            return Err(TextureUploadError::from(format!(
                "{}: mip len {} != expected {} for {:?}",
                label.asset_mip(asset_id),
                mip_src.len(),
                expected_len,
                fmt_format
            )));
        }
        Ok(mip_src.to_vec())
    } else {
        Ok(mip_src.to_vec())
    };
    bytes_result.map(MipUploadPixels::host)
}

/// Descriptor for [`write_one_mip`]: one mip of a 2D texture via [`wgpu::Queue::write_texture`].
pub(super) struct Texture2dMipWrite<'a> {
    /// Queue used for the texel copy.
    pub queue: &'a wgpu::Queue,
    /// Shared GPU queue access gate for [`wgpu::Queue::write_texture`]; see
    /// [`crate::gpu::GpuQueueAccessGate`].
    pub gpu_queue_access_gate: &'a crate::gpu::GpuQueueAccessGate,
    /// Queue-gate acquisition policy for this write.
    pub queue_access_mode: GpuQueueAccessMode,
    /// Destination texture.
    pub texture: &'a wgpu::Texture,
    /// Mip level index.
    pub mip_level: u32,
    /// Logical width in texels.
    pub width: u32,
    /// Logical height in texels.
    pub height: u32,
    /// Texel format (must match texture creation).
    pub format: wgpu::TextureFormat,
    /// Tightly packed mip bytes.
    pub bytes: &'a [u8],
}

/// Writes one full 2D mip level.
pub(super) fn write_one_mip(write: &Texture2dMipWrite<'_>) -> Result<(), TextureUploadError> {
    profiling::scope!("asset::texture_write_mip");
    let Texture2dMipWrite {
        queue,
        gpu_queue_access_gate,
        queue_access_mode,
        texture,
        mip_level,
        width,
        height,
        format,
        bytes,
    } = *write;
    write_texture_region(TextureRegionWrite {
        queue,
        gpu_queue_access_gate,
        queue_access_mode,
        destination: wgpu::TexelCopyTextureInfo {
            texture,
            mip_level,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        width,
        height,
        depth_or_array_layers: 1,
        format,
        bytes,
        label: "mip",
    })
}

/// Writes one full 2D mip level while the caller already holds the queue gate.
pub(super) fn write_one_mip_with_gate(
    write: &Texture2dMipWrite<'_>,
    gate: &parking_lot::MutexGuard<'_, ()>,
) -> Result<(), TextureUploadError> {
    profiling::scope!("asset::texture_write_mip_locked");
    let Texture2dMipWrite {
        queue,
        gpu_queue_access_gate,
        queue_access_mode,
        texture,
        mip_level,
        width,
        height,
        format,
        bytes,
    } = *write;
    write_texture_region_with_gate(
        TextureRegionWrite {
            queue,
            gpu_queue_access_gate,
            queue_access_mode,
            destination: wgpu::TexelCopyTextureInfo {
                texture,
                mip_level,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            width,
            height,
            depth_or_array_layers: 1,
            format,
            bytes,
            label: "mip",
        },
        gate,
    )
}

/// Descriptor for [`write_texture3d_volume_mip`]: one full 3D subresource write via [`wgpu::Queue::write_texture`].
pub struct Texture3dVolumeMipWrite<'a> {
    /// Queue used for the texel copy.
    pub queue: &'a wgpu::Queue,
    /// Shared GPU queue access gate for [`wgpu::Queue::write_texture`]; see
    /// [`crate::gpu::GpuQueueAccessGate`].
    pub gpu_queue_access_gate: &'a crate::gpu::GpuQueueAccessGate,
    /// Queue-gate acquisition policy for this write.
    pub queue_access_mode: GpuQueueAccessMode,
    /// Destination texture.
    pub texture: &'a wgpu::Texture,
    /// Mip level index.
    pub mip_level: u32,
    /// Logical width in texels.
    pub width: u32,
    /// Logical height in texels.
    pub height: u32,
    /// Depth in texels (array layers for 3D).
    pub depth: u32,
    /// Texel format (must match texture creation).
    pub format: wgpu::TextureFormat,
    /// Tightly packed mip bytes for the full volume at `mip_level`.
    pub bytes: &'a [u8],
}

/// Writes one mip level of a 3D texture (full `width` x `height` x `depth` volume).
pub fn write_texture3d_volume_mip(
    write: &Texture3dVolumeMipWrite<'_>,
) -> Result<(), TextureUploadError> {
    profiling::scope!("asset::texture3d_write_volume_mip");
    let Texture3dVolumeMipWrite {
        queue,
        gpu_queue_access_gate,
        queue_access_mode,
        texture,
        mip_level,
        width,
        height,
        depth,
        format,
        bytes,
    } = *write;
    write_texture_region(TextureRegionWrite {
        queue,
        gpu_queue_access_gate,
        queue_access_mode,
        destination: wgpu::TexelCopyTextureInfo {
            texture,
            mip_level,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        width,
        height,
        depth_or_array_layers: depth,
        format,
        bytes,
        label: "3d mip",
    })
}

/// Writes one mip level of a 3D texture while the caller already holds the queue gate.
pub fn write_texture3d_volume_mip_with_gate(
    write: &Texture3dVolumeMipWrite<'_>,
    gate: &parking_lot::MutexGuard<'_, ()>,
) -> Result<(), TextureUploadError> {
    profiling::scope!("asset::texture3d_write_volume_mip_locked");
    let Texture3dVolumeMipWrite {
        queue,
        gpu_queue_access_gate,
        queue_access_mode,
        texture,
        mip_level,
        width,
        height,
        depth,
        format,
        bytes,
    } = *write;
    write_texture_region_with_gate(
        TextureRegionWrite {
            queue,
            gpu_queue_access_gate,
            queue_access_mode,
            destination: wgpu::TexelCopyTextureInfo {
                texture,
                mip_level,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            width,
            height,
            depth_or_array_layers: depth,
            format,
            bytes,
            label: "3d mip",
        },
        gate,
    )
}

/// Descriptor for [`write_cubemap_face_mip`]: one cubemap face x one mip (2D array layer).
pub struct CubemapFaceMipWrite<'a> {
    /// Queue used for the texel copy.
    pub queue: &'a wgpu::Queue,
    /// Shared GPU queue access gate for [`wgpu::Queue::write_texture`]; see
    /// [`crate::gpu::GpuQueueAccessGate`].
    pub gpu_queue_access_gate: &'a crate::gpu::GpuQueueAccessGate,
    /// Queue-gate acquisition policy for this write.
    pub queue_access_mode: GpuQueueAccessMode,
    /// Destination cubemap texture (`D2` array with six layers).
    pub texture: &'a wgpu::Texture,
    /// Mip level index.
    pub mip_level: u32,
    /// Array layer index `0..6` for the cube face.
    pub face_layer: u32,
    /// Face width in texels.
    pub width: u32,
    /// Face height in texels.
    pub height: u32,
    /// Texel format (must match texture creation).
    pub format: wgpu::TextureFormat,
    /// Tightly packed mip bytes for this face.
    pub bytes: &'a [u8],
}

/// Writes one face x one mip of a cubemap (`D2` texture with six array layers).
pub fn write_cubemap_face_mip(write: &CubemapFaceMipWrite<'_>) -> Result<(), TextureUploadError> {
    profiling::scope!("asset::cubemap_write_face_mip");
    let CubemapFaceMipWrite {
        queue,
        gpu_queue_access_gate,
        queue_access_mode,
        texture,
        mip_level,
        face_layer,
        width,
        height,
        format,
        bytes,
    } = *write;
    write_texture_region(TextureRegionWrite {
        queue,
        gpu_queue_access_gate,
        queue_access_mode,
        destination: wgpu::TexelCopyTextureInfo {
            texture,
            mip_level,
            origin: wgpu::Origin3d {
                x: 0,
                y: 0,
                z: face_layer,
            },
            aspect: wgpu::TextureAspect::All,
        },
        width,
        height,
        depth_or_array_layers: 1,
        format,
        bytes,
        label: "cubemap mip",
    })
}

/// Writes one face x one mip of a cubemap while the caller already holds the queue gate.
pub fn write_cubemap_face_mip_with_gate(
    write: &CubemapFaceMipWrite<'_>,
    gate: &parking_lot::MutexGuard<'_, ()>,
) -> Result<(), TextureUploadError> {
    profiling::scope!("asset::cubemap_write_face_mip_locked");
    let CubemapFaceMipWrite {
        queue,
        gpu_queue_access_gate,
        queue_access_mode,
        texture,
        mip_level,
        face_layer,
        width,
        height,
        format,
        bytes,
    } = *write;
    write_texture_region_with_gate(
        TextureRegionWrite {
            queue,
            gpu_queue_access_gate,
            queue_access_mode,
            destination: wgpu::TexelCopyTextureInfo {
                texture,
                mip_level,
                origin: wgpu::Origin3d {
                    x: 0,
                    y: 0,
                    z: face_layer,
                },
                aspect: wgpu::TextureAspect::All,
            },
            width,
            height,
            depth_or_array_layers: 1,
            format,
            bytes,
            label: "cubemap mip",
        },
        gate,
    )
}

/// Descriptor for a generic texture region write.
pub(super) struct TextureRegionWrite<'a> {
    /// Queue used for the texel copy.
    pub queue: &'a wgpu::Queue,
    /// Shared GPU queue access gate for [`wgpu::Queue::write_texture`].
    pub gpu_queue_access_gate: &'a crate::gpu::GpuQueueAccessGate,
    /// Queue-gate acquisition policy for this write.
    pub queue_access_mode: GpuQueueAccessMode,
    /// Destination texture subresource.
    pub destination: wgpu::TexelCopyTextureInfo<'a>,
    /// Logical width in texels.
    pub width: u32,
    /// Logical height in texels.
    pub height: u32,
    /// Number of array layers or 3D depth slices to write.
    pub depth_or_array_layers: u32,
    /// Texel format.
    pub format: wgpu::TextureFormat,
    /// Tightly packed bytes.
    pub bytes: &'a [u8],
    /// Diagnostic label used in length mismatch errors.
    pub label: &'static str,
}

/// Validated region write data ready for queue submission.
struct PreparedTextureRegionWrite<'a> {
    queue: &'a wgpu::Queue,
    gpu_queue_access_gate: &'a crate::gpu::GpuQueueAccessGate,
    queue_access_mode: GpuQueueAccessMode,
    destination: wgpu::TexelCopyTextureInfo<'a>,
    bytes: &'a [u8],
    layout: wgpu::TexelCopyBufferLayout,
    size: wgpu::Extent3d,
}

/// Physical copy extent required by wgpu for a logical mip size.
pub(super) fn copy_extent_for_mip(
    format: wgpu::TextureFormat,
    width: u32,
    height: u32,
    depth_or_array_layers: u32,
) -> wgpu::Extent3d {
    // For block-compressed formats wgpu requires the copy extent to be a multiple of the
    // block dimensions (the "physical" mip size). The layout already covers the padded block grid.
    let (bw, bh) = format.block_dimensions();
    let copy_width = if bw > 1 {
        width.div_ceil(bw) * bw
    } else {
        width
    };
    let copy_height = if bh > 1 {
        height.div_ceil(bh) * bh
    } else {
        height
    };
    wgpu::Extent3d {
        width: copy_width,
        height: copy_height,
        depth_or_array_layers,
    }
}

/// Writes a texture subresource after shared layout, extent, and length validation.
pub(super) fn write_texture_region(
    write: TextureRegionWrite<'_>,
) -> Result<(), TextureUploadError> {
    profiling::scope!("asset::texture_write_region");
    let prepared = prepare_texture_region_write(write)?;
    let gpu_queue_access_gate = prepared.gpu_queue_access_gate;
    let queue_access_mode = prepared.queue_access_mode;
    let Some(_gate) = gpu_queue_access_gate.lock_for(queue_access_mode) else {
        return Err(TextureUploadError::QueueAccessBusy);
    };
    submit_prepared_texture_region_write(prepared);
    Ok(())
}

/// Writes a texture subresource while the caller already holds the queue gate.
pub(super) fn write_texture_region_with_gate(
    write: TextureRegionWrite<'_>,
    _gate: &parking_lot::MutexGuard<'_, ()>,
) -> Result<(), TextureUploadError> {
    profiling::scope!("asset::texture_write_region_locked");
    let prepared = prepare_texture_region_write(write)?;
    submit_prepared_texture_region_write(prepared);
    Ok(())
}

fn prepare_texture_region_write(
    write: TextureRegionWrite<'_>,
) -> Result<PreparedTextureRegionWrite<'_>, TextureUploadError> {
    let size = copy_extent_for_mip(
        write.format,
        write.width,
        write.height,
        write.depth_or_array_layers,
    );
    let (layout, slice_len) = copy_layout_for_mip(write.format, write.width, write.height)?;
    let expected = slice_len
        .checked_mul(write.depth_or_array_layers as usize)
        .ok_or_else(|| {
            TextureUploadError::from(format!("{} expected bytes overflow", write.label))
        })?;
    if write.bytes.len() != expected {
        return Err(TextureUploadError::from(format!(
            "{} data len {} != expected {} ({}x{}x{} {:?})",
            write.label,
            write.bytes.len(),
            expected,
            write.width,
            write.height,
            write.depth_or_array_layers,
            write.format
        )));
    }

    Ok(PreparedTextureRegionWrite {
        queue: write.queue,
        gpu_queue_access_gate: write.gpu_queue_access_gate,
        queue_access_mode: write.queue_access_mode,
        destination: write.destination,
        bytes: write.bytes,
        layout,
        size,
    })
}

fn submit_prepared_texture_region_write(write: PreparedTextureRegionWrite<'_>) {
    write
        .queue
        .write_texture(write.destination, write.bytes, write.layout, write.size);
}

/// Builds a tight-copy layout and per-layer byte length for one mip.
pub(super) fn copy_layout_for_mip(
    format: wgpu::TextureFormat,
    width: u32,
    height: u32,
) -> Result<(wgpu::TexelCopyBufferLayout, usize), TextureUploadError> {
    let (bw, bh) = format.block_dimensions();
    let block_bytes = format
        .block_copy_size(None)
        .ok_or_else(|| TextureUploadError::from(format!("no block copy size for {format:?}")))?;
    if bw == 1 && bh == 1 {
        let bpp = block_bytes as usize;
        let bpr = bpp
            .checked_mul(width as usize)
            .ok_or_else(|| TextureUploadError::from("bytes_per_row overflow"))?;
        let expected = bpr
            .checked_mul(height as usize)
            .ok_or_else(|| TextureUploadError::from("expected bytes overflow"))?;
        #[expect(
            clippy::map_err_ignore,
            reason = "TryFromIntError adds no detail beyond the overflow label"
        )]
        let bpr_u32 =
            u32::try_from(bpr).map_err(|_| TextureUploadError::from("bpr u32 overflow"))?;
        return Ok((
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bpr_u32),
                rows_per_image: Some(height),
            },
            expected,
        ));
    }

    let blocks_x = width.div_ceil(bw);
    let blocks_y = height.div_ceil(bh);
    let row_bytes_u = blocks_x
        .checked_mul(block_bytes)
        .ok_or_else(|| TextureUploadError::from("row bytes overflow"))?;
    let expected_u = row_bytes_u
        .checked_mul(blocks_y)
        .ok_or_else(|| TextureUploadError::from("expected size overflow"))?;
    let expected = expected_u as usize;
    Ok((
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(row_bytes_u),
            rows_per_image: Some(blocks_y),
        },
        expected,
    ))
}

#[cfg(test)]
mod tests {
    use glam::IVec2;

    use super::{
        MipUploadFormatCtx, MipUploadLabel, choose_mip_start_bias, copy_extent_for_mip,
        copy_layout_for_mip, is_rgba8_family, mip_src_to_upload_pixels,
        upload_uses_storage_v_inversion, valid_mip_prefix_len,
    };
    use crate::shared::{SetTexture2DData, TextureFormat};

    #[test]
    fn relative_mip_starts_need_no_rebase() {
        let mut upload = SetTexture2DData::default();
        upload.data.length = 80;
        upload.mip_map_sizes = vec![IVec2::new(4, 4), IVec2::new(2, 2)];
        // `mip_starts` are linear texel indices into the chain; texel 16 begins the 2x2 mip (byte 64).
        upload.mip_starts = vec![0, 16];

        let (bias, prefix) = choose_mip_start_bias(TextureFormat::RGBA32, &upload, 80).unwrap();
        assert_eq!(bias, 0);
        assert_eq!(prefix, 2);
    }

    #[test]
    fn absolute_mip_starts_rebase_to_descriptor_offset() {
        let mut upload = SetTexture2DData::default();
        upload.data.offset = 128;
        upload.data.length = 80;
        upload.mip_map_sizes = vec![IVec2::new(4, 4), IVec2::new(2, 2)];
        // Absolute SHM indices: base mip at descriptor offset; second mip at texel 144 (= 128 + 16).
        upload.mip_starts = vec![128, 144];

        let (bias, prefix) = choose_mip_start_bias(TextureFormat::RGBA32, &upload, 80).unwrap();
        assert_eq!(bias, 128);
        assert_eq!(prefix, 2);
    }

    #[test]
    fn valid_prefix_len_stops_when_later_mip_exceeds_payload() {
        let mut upload = SetTexture2DData::default();
        upload.data.length = 68;
        upload.mip_map_sizes = vec![IVec2::new(4, 4), IVec2::new(2, 2)];
        upload.mip_starts = vec![0, 64];

        let prefix = valid_mip_prefix_len(TextureFormat::RGBA32, &upload, 68, 0).unwrap();
        assert_eq!(prefix, 1);
    }

    #[test]
    fn choose_mip_start_bias_rejects_uploads_with_no_fitting_mips() {
        let mut upload = SetTexture2DData::default();
        upload.data.length = 4;
        upload.mip_map_sizes = vec![IVec2::new(4, 4)];
        upload.mip_starts = vec![0];

        let err = choose_mip_start_bias(TextureFormat::RGBA32, &upload, 4)
            .expect_err("no mip fits payload");

        assert!(
            err.to_string()
                .contains("mip region exceeds shared memory descriptor"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn valid_prefix_len_errors_on_non_positive_dimensions() {
        let mut upload = SetTexture2DData::default();
        upload.data.length = 64;
        upload.mip_map_sizes = vec![IVec2::new(4, 0)];
        upload.mip_starts = vec![0];

        let err = valid_mip_prefix_len(TextureFormat::RGBA32, &upload, 64, 0)
            .expect_err("bad mip dimensions");

        assert!(err.to_string().contains("non-positive mip dimensions"));
    }

    #[test]
    fn valid_prefix_len_stops_at_negative_tail_start() {
        let mut upload = SetTexture2DData::default();
        upload.data.length = 64;
        upload.mip_map_sizes = vec![IVec2::new(4, 4), IVec2::new(2, 2)];
        upload.mip_starts = vec![0, -1];

        let prefix = valid_mip_prefix_len(TextureFormat::RGBA32, &upload, 64, 0).unwrap();
        assert_eq!(prefix, 1);
    }

    #[test]
    fn valid_prefix_len_stops_when_start_is_before_bias() {
        let mut upload = SetTexture2DData::default();
        upload.data.offset = 128;
        upload.data.length = 64;
        upload.mip_map_sizes = vec![IVec2::new(4, 4)];
        upload.mip_starts = vec![64];

        let prefix = valid_mip_prefix_len(TextureFormat::RGBA32, &upload, 64, 128).unwrap();

        assert_eq!(prefix, 0);
    }

    #[test]
    fn valid_prefix_len_uses_compressed_byte_offsets() {
        let mut upload = SetTexture2DData::default();
        upload.data.length = 8;
        upload.mip_map_sizes = vec![IVec2::new(4, 4)];
        upload.mip_starts = vec![0];
        assert_eq!(
            valid_mip_prefix_len(TextureFormat::BC1, &upload, 8, 0).unwrap(),
            1
        );

        upload.mip_starts = vec![16];
        assert_eq!(
            valid_mip_prefix_len(TextureFormat::BC1, &upload, 8, 0).unwrap(),
            0
        );
    }

    #[test]
    fn copy_extent_aligns_block_compressed_mips() {
        let extent = copy_extent_for_mip(wgpu::TextureFormat::Bc1RgbaUnorm, 7, 5, 1);

        assert_eq!(extent.width, 8);
        assert_eq!(extent.height, 8);
        assert_eq!(extent.depth_or_array_layers, 1);
    }

    #[test]
    fn copy_extent_keeps_uncompressed_mips_tight() {
        let extent = copy_extent_for_mip(wgpu::TextureFormat::Rgba8Unorm, 7, 5, 3);

        assert_eq!(extent.width, 7);
        assert_eq!(extent.height, 5);
        assert_eq!(extent.depth_or_array_layers, 3);
    }

    #[test]
    fn copy_layout_for_mip_reports_tight_uncompressed_layout() {
        let (layout, expected) =
            copy_layout_for_mip(wgpu::TextureFormat::Rgba8Unorm, 3, 2).expect("layout");

        assert_eq!(layout.bytes_per_row, Some(12));
        assert_eq!(layout.rows_per_image, Some(2));
        assert_eq!(expected, 24);
    }

    #[test]
    fn copy_layout_for_mip_reports_block_compressed_layout() {
        let (layout, expected) =
            copy_layout_for_mip(wgpu::TextureFormat::Bc1RgbaUnorm, 5, 5).expect("layout");

        assert_eq!(layout.bytes_per_row, Some(16));
        assert_eq!(layout.rows_per_image, Some(2));
        assert_eq!(expected, 32);
    }

    #[test]
    fn rgba8_family_and_storage_orientation_contract_are_stable() {
        assert!(is_rgba8_family(wgpu::TextureFormat::Rgba8Unorm));
        assert!(is_rgba8_family(wgpu::TextureFormat::Rgba8UnormSrgb));
        assert!(!is_rgba8_family(wgpu::TextureFormat::Bgra8Unorm));
        assert!(upload_uses_storage_v_inversion(
            TextureFormat::RGBA32,
            wgpu::TextureFormat::Rgba8Unorm,
            false,
        ));
    }

    #[test]
    fn mip_src_to_upload_pixels_direct_rgba8_keeps_bytes_and_marks_host_orientation() {
        let ctx = MipUploadFormatCtx {
            asset_id: 7,
            fmt_format: TextureFormat::RGBA32,
            wgpu_format: wgpu::TextureFormat::Rgba8Unorm,
            needs_rgba8_decode: false,
        };

        let pixels = mip_src_to_upload_pixels(
            ctx,
            1,
            1,
            false,
            &[1, 2, 3, 4],
            MipUploadLabel::texture2d(0),
        )
        .expect("direct rgba upload");

        assert_eq!(pixels.bytes, vec![1, 2, 3, 4]);
        assert!(pixels.storage_v_inverted);
    }
}
