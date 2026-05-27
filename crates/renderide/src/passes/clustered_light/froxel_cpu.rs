//! CPU light-centric froxel assignment for clustered forward lighting.
//!
//! This path mirrors the clustered-light storage contract (`cluster_light_counts` range rows plus
//! compact `cluster_light_indices`) so dense-light frames can use a light-centric alternative to
//! the O(froxels x lights) GPU scan.

/// Conservative froxel-space bounds for CPU light assignment.
mod bounds;
/// Per-eye froxel traversal and light emission helpers.
mod geometry;
/// Rayon-backed CPU froxel assignment implementation.
mod parallel;
/// CPU froxel assignment entry point and admission thresholds.
mod planner;
/// Prefix-sum and compact membership writing helpers.
mod prefix;
#[cfg(test)]
/// CPU froxel assignment unit tests.
mod tests;
/// Shared CPU froxel assignment data structures.
mod types;

pub(super) use planner::{AUTO_CPU_FROXEL_LIGHT_THRESHOLD, FroxelLightPlanner};
