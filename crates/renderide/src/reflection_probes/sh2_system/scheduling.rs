//! GPU-job scheduling for queued SH2 projection sources.

use std::sync::Arc;

use glam::Vec4;

use super::super::projection_pipeline::{
    ProjectionBinding, ProjectionPipeline, ProjectionPipelineKind, encode_projection_job,
};
use super::super::readback_jobs::SubmittedGpuSh2Job;
use super::ReflectionProbeSh2System;
use super::gpu_source::GpuSh2Source;
use super::source_keys::{Sh2ProjectParams, Sh2SourceKey, SkyParamMode};
use crate::backend::AssetTransferQueue;
use crate::gpu::GpuContext;
use crate::profiling;
use crate::skybox::params::storage_v_inverted_flag;
use crate::skybox::specular::solid_color_params;

/// Maximum pending GPU jobs kept alive at once.
const MAX_IN_FLIGHT_JOBS: usize = 6;

/// Outcome of trying to submit one queued source to the GPU.
pub(super) enum ScheduleSourceOutcome {
    /// Source was submitted and is now awaiting readback.
    Submitted(SubmittedGpuSh2Job),
    /// Source could not be scheduled this tick; requeue and retry later.
    RetryLater(GpuSh2Source),
}

impl ReflectionProbeSh2System {
    /// Schedules queued sources until the in-flight cap is reached.
    pub(super) fn schedule_queued_sources(
        &mut self,
        gpu: &mut GpuContext,
        assets: &AssetTransferQueue,
    ) {
        profiling::scope!("reflection_probe_sh2::schedule_queued_sources");
        let attempts = self.queue_order.len();
        for _ in 0..attempts {
            if self.readback_jobs.len() >= MAX_IN_FLIGHT_JOBS {
                break;
            }
            let Some(key) = self.queue_order.pop_front() else {
                break;
            };
            let Some(source) = self.queued_sources.remove(&key) else {
                continue;
            };
            if self.completed.contains_key(&key)
                || self.readback_jobs.contains_key(&key)
                || self.failed.contains(&key)
            {
                continue;
            }
            match self.schedule_source(gpu, assets, key.clone(), source) {
                Ok(ScheduleSourceOutcome::Submitted(job)) => {
                    self.readback_jobs.insert(key, job);
                }
                Ok(ScheduleSourceOutcome::RetryLater(source)) => {
                    self.queue_order.push_back(key.clone());
                    self.queued_sources.insert(key, source);
                }
                Err(e) => {
                    logger::warn!("reflection_probe_sh2: GPU SH2 schedule failed: {e}");
                    self.failed.insert(key);
                }
            }
        }
    }

    /// Encodes and submits one source projection.
    fn schedule_source(
        &mut self,
        gpu: &mut GpuContext,
        assets: &AssetTransferQueue,
        key: Sh2SourceKey,
        source: GpuSh2Source,
    ) -> Result<ScheduleSourceOutcome, String> {
        profiling::scope!("reflection_probe_sh2::schedule_source");
        let pipeline_kind = projection_pipeline_kind(&source);
        if !self
            .pipeline_cache
            .ensure_ready(gpu.device(), pipeline_kind)?
        {
            return Ok(ScheduleSourceOutcome::RetryLater(source));
        }
        let pipeline = self.pipeline_cache.get(pipeline_kind).ok_or_else(|| {
            format!(
                "projection pipeline {} missing after build",
                pipeline_kind.stem()
            )
        })?;
        match source {
            GpuSh2Source::Cubemap {
                asset_id,
                storage_v_inverted,
                clear_color,
            } => self
                .schedule_cubemap_source(
                    gpu,
                    assets,
                    key,
                    asset_id,
                    &cubemap_projection_params(clear_color, storage_v_inverted),
                    pipeline,
                )
                .map(ScheduleSourceOutcome::Submitted),
            GpuSh2Source::RuntimeCubemap {
                texture,
                view,
                clear_color,
            } => self
                .schedule_runtime_cubemap_source(gpu, key, texture, view, clear_color, pipeline)
                .map(ScheduleSourceOutcome::Submitted),
        }
    }

    fn schedule_cubemap_source(
        &self,
        gpu: &mut GpuContext,
        assets: &AssetTransferQueue,
        key: Sh2SourceKey,
        asset_id: i32,
        params: &Sh2ProjectParams,
        pipeline: &ProjectionPipeline,
    ) -> Result<SubmittedGpuSh2Job, String> {
        profiling::scope!("reflection_probe_sh2::schedule_cubemap");
        let tex = assets
            .cubemap_pool()
            .get(asset_id)
            .filter(|t| t.mip_levels_resident > 0)
            .ok_or_else(|| format!("cubemap {asset_id} not resident"))?;
        let sampler = sh2_cubemap_sampler(gpu.device(), "SH2 cubemap sampler");
        let view = tex.view.clone();
        let submit_done_tx = self.readback_jobs.submit_done_sender();
        encode_projection_job(
            gpu,
            key,
            pipeline,
            &[
                ProjectionBinding::TextureView(view.as_ref()),
                ProjectionBinding::Sampler(&sampler),
            ],
            params,
            &submit_done_tx,
            "reflection_probe_sh2::project_cubemap",
        )
    }

    fn schedule_runtime_cubemap_source(
        &self,
        gpu: &mut GpuContext,
        key: Sh2SourceKey,
        texture: Arc<wgpu::Texture>,
        view: Arc<wgpu::TextureView>,
        clear_color: Option<Vec4>,
        pipeline: &ProjectionPipeline,
    ) -> Result<SubmittedGpuSh2Job, String> {
        profiling::scope!("reflection_probe_sh2::schedule_runtime_cubemap");
        let sampler = sh2_cubemap_sampler(gpu.device(), "SH2 runtime cubemap sampler");
        let submit_done_tx = self.readback_jobs.submit_done_sender();
        let mut job = encode_projection_job(
            gpu,
            key,
            pipeline,
            &[
                ProjectionBinding::TextureView(view.as_ref()),
                ProjectionBinding::Sampler(&sampler),
            ],
            &projection_params(clear_color),
            &submit_done_tx,
            "reflection_probe_sh2::project_runtime_cubemap",
        )?;
        job.textures.push(texture);
        job.source_views.push(view);
        Ok(job)
    }
}

fn projection_pipeline_kind(source: &GpuSh2Source) -> ProjectionPipelineKind {
    match source {
        GpuSh2Source::Cubemap { .. } => ProjectionPipelineKind::Cubemap,
        GpuSh2Source::RuntimeCubemap { .. } => ProjectionPipelineKind::Cubemap,
    }
}

fn sh2_cubemap_sampler(device: &wgpu::Device, label: &'static str) -> wgpu::Sampler {
    device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some(label),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::MipmapFilterMode::Nearest,
        ..Default::default()
    })
}

fn projection_params(clear_color: Option<Vec4>) -> Sh2ProjectParams {
    match clear_color {
        Some(color) => solid_color_params(color.to_array()),
        None => Sh2ProjectParams::empty(SkyParamMode::Procedural),
    }
}

fn cubemap_projection_params(
    clear_color: Option<Vec4>,
    storage_v_inverted: bool,
) -> Sh2ProjectParams {
    let mut params = projection_params(clear_color);
    params.scalars[0] = storage_v_inverted_flag(storage_v_inverted);
    params
}
