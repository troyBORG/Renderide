use crate::gpu::GpuLight;
use crate::world_mesh::cluster::{CLUSTER_COUNT_Z, ClusterFrameParams};

use super::bounds::FroxelSphere;

/// Cluster-grid layout for one eye.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct FroxelLayout {
    /// Cluster count in screen X.
    pub cluster_count_x: u32,
    /// Cluster count in screen Y.
    pub cluster_count_y: u32,
    /// Cluster count in depth.
    pub cluster_count_z: u32,
    /// Viewport width in physical pixels.
    pub viewport_width: u32,
    /// Viewport height in physical pixels.
    pub viewport_height: u32,
}

impl FroxelLayout {
    /// Builds a layout from the frame's clustered camera params.
    pub(super) fn from_cluster_params(params: &ClusterFrameParams) -> Self {
        Self {
            cluster_count_x: params.cluster_count_x.max(1),
            cluster_count_y: params.cluster_count_y.max(1),
            cluster_count_z: CLUSTER_COUNT_Z.max(1),
            viewport_width: params.viewport_width.max(1),
            viewport_height: params.viewport_height.max(1),
        }
    }

    /// Number of froxels in this eye.
    pub(super) fn cluster_count(self) -> Option<usize> {
        let xy = self.cluster_count_x.checked_mul(self.cluster_count_y)?;
        xy.checked_mul(self.cluster_count_z).map(|v| v as usize)
    }
}

/// Per-frame CPU-produced cluster storage matching the existing WGSL buffers.
#[derive(Clone, Debug, Default)]
pub(in crate::passes::clustered_light) struct CpuClusterAssignments {
    /// Per-froxel `[offset, count]` rows addressing [`Self::indices`].
    pub ranges: Vec<[u32; 2]>,
    /// Compact light indices for every froxel membership.
    pub indices: Vec<u32>,
    /// Assignment diagnostics for profiling and tests.
    pub stats: CpuFroxelStats,
}

/// CPU froxel assignment diagnostics.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(in crate::passes::clustered_light) struct CpuFroxelStats {
    /// Number of light/froxel memberships emitted into compact storage.
    pub assigned_memberships: u64,
    /// Number of light/froxel memberships dropped because compact storage could not represent them.
    pub overflowed_memberships: u64,
    /// Number of lights rejected before assignment because their conservative bounds miss the view.
    pub culled_lights: u32,
}

pub(super) struct CpuFroxelCountChunk {
    pub(super) counts: Vec<u32>,
    pub(super) stats: CpuFroxelStats,
}

/// Local prefix-sum result for one cluster-count chunk.
pub(super) struct CpuFroxelPrefixChunk {
    /// Range rows with offsets relative to the start of this chunk.
    pub(super) ranges: Vec<[u32; 2]>,
    /// Sum of every count in this chunk.
    pub(super) total_count: u64,
}

pub(super) struct CpuFroxelParallelInputs<'a> {
    pub(super) lights: &'a [GpuLight],
    pub(super) eye_params: &'a [ClusterFrameParams],
    pub(super) layouts: &'a [FroxelLayout],
    pub(super) froxel_spheres_by_eye: &'a [Vec<FroxelSphere>],
    pub(super) expected_clusters: usize,
    pub(super) total_clusters: usize,
}
