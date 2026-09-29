//! Geometry, tunnel-path search, smoothing, and train-gauge calculations.

use crate::compute::ComputeEngine;
use crate::config::{NumericSettings, TunnelPathSettings};
use anyhow::{anyhow, Result};
use rayon::prelude::*;
use rerun::external::glam::Vec3;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// Axis-aligned 3D bounding box.
#[derive(Debug, Clone, Copy)]
pub struct Box3 {
    /// Minimum corner.
    pub min: Vec3,
    /// Maximum corner.
    pub max: Vec3,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
struct Box3Serde {
    min: [f32; 3],
    max: [f32; 3],
}

impl Serialize for Box3 {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        Box3Serde {
            min: [self.min.x, self.min.y, self.min.z],
            max: [self.max.x, self.max.y, self.max.z],
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Box3 {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = Box3Serde::deserialize(deserializer)?;
        Ok(Self::new(
            Vec3::new(value.min[0], value.min[1], value.min[2]),
            Vec3::new(value.max[0], value.max[1], value.max[2]),
        ))
    }
}

impl Box3 {
    /// Creates a normalized box from two arbitrary corners.
    pub fn new(a: Vec3, b: Vec3) -> Self {
        Self {
            min: a.min(b),
            max: a.max(b),
        }
    }

    /// Checks whether the point lies inside the inclusive box bounds.
    #[inline]
    pub fn contains(&self, point: Vec3) -> bool {
        point.x >= self.min.x
            && point.x <= self.max.x
            && point.y >= self.min.y
            && point.y <= self.max.y
            && point.z >= self.min.z
            && point.z <= self.max.z
    }

    /// Filters points into an existing output buffer.
    ///
    /// The output buffer is cleared and reused, reducing per-frame allocations.
    pub fn filter_points_into(&self, points: &[Vec3], output: &mut Vec<Vec3>) {
        output.clear();
        output.par_extend(
            points
                .par_iter()
                .copied()
                .filter(|&point| self.contains(point)),
        );
    }

    /// Returns the center of the box.
    #[inline]
    pub fn center(&self) -> Vec3 {
        (self.min + self.max) * 0.5
    }

    /// Returns half of the box dimensions along each axis.
    #[inline]
    pub fn half_size(&self) -> Vec3 {
        (self.max - self.min) * 0.5
    }
}

/// Reusable temporary storage for tunnel-path Y slicing.
#[derive(Default)]
pub struct TunnelPathCache {
    sorted_indices: Vec<usize>,
    slice_indices: Vec<usize>,
}

impl TunnelPathCache {
    fn prepare(&mut self, points: &[Vec3], y_min: f32, y_max: f32) {
        self.sorted_indices.clear();
        self.sorted_indices
            .extend(points.iter().enumerate().filter_map(|(index, point)| {
                (point.y >= y_min && point.y <= y_max).then_some(index)
            }));
        self.sorted_indices
            .sort_unstable_by(|&a, &b| points[a].y.total_cmp(&points[b].y));
    }
}

#[inline]
fn lower_bound_y(points: &[Vec3], indices: &[usize], target: f32) -> usize {
    let mut left = 0;
    let mut right = indices.len();
    while left < right {
        let mid = left + (right - left) / 2;
        if points[indices[mid]].y < target {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    left
}

#[inline]
fn upper_bound_y(points: &[Vec3], indices: &[usize], target: f32) -> usize {
    let mut left = 0;
    let mut right = indices.len();
    while left < right {
        let mid = left + (right - left) / 2;
        if points[indices[mid]].y <= target {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    left
}

/// Stateful tunnel-center searcher that reuses temporary allocations between frames.
pub struct TunnelPathSearcher {
    cache: TunnelPathCache,
    compute: ComputeEngine,
}

impl TunnelPathSearcher {
    /// Creates a tunnel-path searcher with the requested compute backend.
    pub fn new(settings: &crate::config::ComputeSettings) -> Self {
        Self {
            cache: TunnelPathCache::default(),
            compute: ComputeEngine::new(settings),
        }
    }

    /// Returns the name of the active center-search backend.
    #[allow(unused)]
    pub fn backend_name(&self) -> &'static str {
        self.compute.backend_name()
    }

    /// Finds the tunnel centerline through successive longitudinal slices.
    pub fn search(
        &mut self,
        points: &[Vec3],
        settings: &TunnelPathSettings,
        numeric: &NumericSettings,
    ) -> Vec<Vec3> {
        if points.is_empty()
            || settings.max_steps == 0
            || settings.slice_thickness <= 0.0
            || settings.step_length <= 0.0
            || settings.grid_resolution <= 0.0
            || settings.width <= 0.0
            || settings.height <= 0.0
        {
            return Vec::new();
        }

        let mut path = Vec::with_capacity(settings.max_steps);
        let start_center = settings.start_box.center();
        let half_search = settings.start_box.half_size();
        let tunnel_half_width = settings.width * 0.5;
        let tunnel_half_height = settings.height * 0.5;
        let tunnel_z_min = start_center.z - tunnel_half_height;
        let tunnel_z_max = start_center.z + tunnel_half_height;
        let mut current_y = start_center.y;
        let mut previous_center = None;

        let max_backtrack = settings.max_steps.saturating_sub(1) as f32 * settings.step_length;
        let global_y_min = current_y - max_backtrack - settings.slice_thickness * 0.5;
        let global_y_max = current_y + settings.slice_thickness * 0.5;
        self.cache.prepare(points, global_y_min, global_y_max);

        for _ in 0..settings.max_steps {
            let y_min = current_y - settings.slice_thickness * 0.5;
            let y_max = current_y + settings.slice_thickness * 0.5;

            let center_x = previous_center.map_or(start_center.x, |center: Vec3| center.x);
            let corridor_min_x = center_x - tunnel_half_width;
            let corridor_max_x = center_x + tunnel_half_width;

            let (mut x_min, mut x_max, mut z_min, mut z_max) = if let Some(previous) = previous_center {
                (
                    previous.x - half_search.x,
                    previous.x + half_search.x,
                    previous.z - half_search.z,
                    previous.z + half_search.z,
                )
            } else {
                (
                    settings.start_box.min.x,
                    settings.start_box.max.x,
                    settings.start_box.min.z,
                    settings.start_box.max.z,
                )
            };

            x_min = x_min.max(corridor_min_x);
            x_max = x_max.min(corridor_max_x);
            z_min = z_min.max(tunnel_z_min);
            z_max = z_max.min(tunnel_z_max);

            if x_min >= x_max || z_min >= z_max {
                break;
            }

            let begin = lower_bound_y(points, &self.cache.sorted_indices, y_min);
            let end = upper_bound_y(points, &self.cache.sorted_indices, y_max);
            if begin >= end {
                break;
            }

            self.cache.slice_indices.clear();
            self.cache.slice_indices.extend(
                self.cache.sorted_indices[begin..end]
                    .iter()
                    .copied()
                    .filter(|&index| {
                        let point = points[index];
                        if point.x < corridor_min_x || point.x > corridor_max_x {
                            return false;
                        }

                        if point.z < tunnel_z_min || point.z > tunnel_z_max {
                            return false;
                        }

                        if settings.restrict_points_to_search_window {
                            point.x >= x_min
                                && point.x <= x_max
                                && point.z >= z_min
                                && point.z <= z_max
                        } else {
                            true
                        }
                    }),
            );

            if self.cache.slice_indices.is_empty() {
                break;
            }

            let (x_steps, z_steps) = grid_size(x_min, x_max, z_min, z_max, settings.grid_resolution);
            let candidate_count = x_steps.saturating_mul(z_steps);
            if candidate_count == 0 {
                break;
            }

            let best_index = self.compute.best_candidate(
                &self.cache.slice_indices,
                points,
                x_min,
                z_min,
                current_y,
                settings.grid_resolution,
                x_steps,
                z_steps,
                settings.repulsion_factor,
                settings.spring_factor,
                numeric.cost_distance_offset,
            );

            let Some(best_index) = best_index else {
                break;
            };

            let best_x = x_min + (best_index / z_steps) as f32 * settings.grid_resolution;
            let best_z = z_min + (best_index % z_steps) as f32 * settings.grid_resolution;
            let best = Vec3::new(best_x, current_y, best_z);

            path.push(best);
            previous_center = Some(best);
            current_y -= settings.step_length;
        }

        path
    }
}

#[inline]
fn grid_size(x_min: f32, x_max: f32, z_min: f32, z_max: f32, resolution: f32) -> (usize, usize) {
    let x_steps = ((x_max - x_min).max(0.0) / resolution).floor() as usize + 1;
    let z_steps = ((z_max - z_min).max(0.0) / resolution).floor() as usize + 1;
    (x_steps, z_steps)
}

/// Smooths path position and height using independent adaptive neighborhoods.
///
/// This is the current two-pass implementation used by the project. The
/// positional pass produces `smoothed`; the height pass follows the existing
/// implementation's source values and then writes the resulting Z coordinate
/// back into `smoothed`.
pub fn smooth_path(points: &[Vec3], position_radius: usize, height_radius: usize) -> Vec<Vec3> {
    if points.is_empty() {
        return Vec::new();
    }

    let mut smoothed = Vec::with_capacity(points.len());
    for i in 0..points.len() {
        let radius = i.min(points.len() - 1 - i).min(position_radius);
        if radius == 0 {
            smoothed.push(points[i]);
            continue;
        }

        let mut weighted_sum = Vec3::ZERO;
        let mut total_weight = 0.0;
        for (j, cur_pt) in points.iter().enumerate().take(i + radius + 1).skip(i - radius) {
            let distance = (j as isize - i as isize).unsigned_abs() as f32;
            let weight = radius as f32 + 1.0 - distance;
            weighted_sum += cur_pt * weight;
            total_weight += weight;
        }
        smoothed.push(weighted_sum / total_weight);
    }

    let mut heights = Vec::with_capacity(smoothed.len());
    for i in 0..smoothed.len() {
        let radius = i.min(smoothed.len() - 1 - i).min(height_radius);
        if radius == 0 {
            heights.push(smoothed[i].z);
            continue;
        }

        let mut weighted_sum = 0.0;
        let mut total_weight = 0.0;
        for (j, cur_smoothed_pt) in points.iter().enumerate().take(i + radius + 1).skip(i - radius) {
            let distance = (j as isize - i as isize).unsigned_abs() as f32;
            let weight = radius as f32 + 1.0 - distance;
            weighted_sum += cur_smoothed_pt.z * weight;
            total_weight += weight;
        }
        heights.push(weighted_sum / total_weight);
    }

    for (point, height) in smoothed.iter_mut().zip(heights) {
        point.z = height;
    }

    smoothed
}

/// Calculates the train-center start point and longitudinal direction from both rails.
#[inline]
pub fn calculate_train_motion(
    left: &crate::calibration::CalibrationResult,
    right: &crate::calibration::CalibrationResult,
    train_height: f32,
) -> (Vec3, Vec3) {
    let center = (left.start_point + right.start_point) * 0.5;
    let start = center + Vec3::new(0.0, 0.0, train_height * 0.5);
    let direction = (left.direction + right.direction)
        .try_normalize()
        .unwrap_or(-Vec3::Y);
    (start, direction)
}

#[derive(Debug, Clone, Copy)]
struct TrackSegment {
    start: Vec3,
    end: Vec3,
    forward: Vec3,
    right: Vec3,
    up: Vec3,
    length: f32,
    cumulative_start: f32,
}

/// Result of projecting a point onto the closest valid track segment.
#[derive(Debug, Clone, Copy)]
pub struct TrackProjection {
    /// Index of the selected segment.
    pub segment_index: usize,
    /// Closest point on the selected segment.
    pub projection: Vec3,
    /// Squared Euclidean distance from the query point to the projection.
    pub distance_squared: f32,
    /// Distance from the beginning of the track to the projection.
    pub distance_along_track: f32,
}

/// Precomputed geometry used to test LiDAR points against the train gauge.
pub struct TrackGeometry {
    segments: Vec<TrackSegment>,
    total_length: f32,
    bounds_min: Vec3,
    bounds_max: Vec3,
    half_width: f32,
    half_height: f32,
    broad_phase_radius_squared: f32,
    numeric: NumericSettings,
}

impl TrackGeometry {
    /// Builds reusable segment geometry for a path.
    ///
    /// # Errors
    ///
    /// Returns an error if the path has fewer than two points or consists only
    /// of degenerate zero-length segments.
    pub fn new(
        path: &[Vec3],
        train_width: f32,
        train_height: f32,
        numeric: NumericSettings,
    ) -> Result<Self> {
        if path.len() < 2 {
            return Err(anyhow!("Для TrackGeometry нужно минимум 2 точки"));
        }

        let half_width = train_width * 0.5;
        let half_height = train_height * 0.5;
        let broad_phase_radius = train_width.max(train_height) * numeric.broad_phase_scale;
        let broad_phase_radius_squared = broad_phase_radius * broad_phase_radius;

        let mut segments = Vec::with_capacity(path.len() - 1);
        let mut total_length = 0.0;
        let mut bounds_min = Vec3::splat(f32::INFINITY);
        let mut bounds_max = Vec3::splat(f32::NEG_INFINITY);

        for i in 0..path.len() - 1 {
            let start = path[i];
            let end = path[i + 1];
            let delta = end - start;
            let length = delta.length();
            if length < numeric.minimum_segment_length {
                continue;
            }

            let forward = delta / length;
            let mut right = forward.cross(Vec3::Z);
            if right.length_squared() < numeric.basis_epsilon {
                right = forward.cross(Vec3::X);
            }
            let right = right.normalize_or_zero();
            let up = right.cross(forward).normalize_or_zero();

            bounds_min = bounds_min.min(start).min(end);
            bounds_max = bounds_max.max(start).max(end);

            segments.push(TrackSegment {
                start,
                end,
                forward,
                right,
                up,
                length,
                cumulative_start: total_length,
            });
            total_length += length;
        }

        if segments.is_empty() {
            return Err(anyhow!("Траектория содержит только нулевые сегменты"));
        }

        Ok(Self {
            segments,
            total_length,
            bounds_min: bounds_min - Vec3::splat(broad_phase_radius),
            bounds_max: bounds_max + Vec3::splat(broad_phase_radius),
            half_width,
            half_height,
            broad_phase_radius_squared,
            numeric,
        })
    }

    /// Projects a point onto the nearest non-degenerate track segment.
    #[inline]
    pub fn project(&self, point: Vec3) -> Option<TrackProjection> {
        if point.x < self.bounds_min.x
            || point.x > self.bounds_max.x
            || point.y < self.bounds_min.y
            || point.y > self.bounds_max.y
            || point.z < self.bounds_min.z
            || point.z > self.bounds_max.z
        {
            return None;
        }

        let mut best = None;
        for (index, segment) in self.segments.iter().enumerate() {
            let direction = segment.end - segment.start;
            let length_squared = direction.length_squared();
            if length_squared < self.numeric.minimum_segment_length * self.numeric.minimum_segment_length {
                continue;
            }

            let t = ((point - segment.start).dot(direction) / length_squared).clamp(0.0, 1.0);
            let projection = segment.start + direction * t;
            let distance_squared = point.distance_squared(projection);

            if best.is_none_or(|current: TrackProjection| distance_squared < current.distance_squared) {
                best = Some(TrackProjection {
                    segment_index: index,
                    projection,
                    distance_squared,
                    distance_along_track: segment.cumulative_start + segment.length * t,
                });
            }
        }
        best
    }
}

/// LiDAR point that lies inside the train gauge.
#[derive(Debug, Clone, Copy)]
pub struct CollisionPoint {
    /// Original LiDAR point.
    pub point: Vec3,
    /// Distance from the start of the track to the projected point.
    pub distance_along_track: f32,
    /// Euclidean offset from the track centerline.
    pub offset: f32,
}

/// Finds points that intersect the train gauge around a precomputed path.
pub fn find_points_in_train_gauge(points: &[Vec3], geometry: &TrackGeometry) -> Vec<CollisionPoint> {
    let last_segment_index = geometry.segments.len() - 1;

    points
        .par_iter()
        .copied()
        .filter_map(|point| {
            let projection = geometry.project(point)?;

            if projection.segment_index == 0 {
                let segment = geometry.segments[0];
                if projection.distance_along_track <= 0.0
                    && (point - segment.start).dot(segment.forward) < 0.0
                {
                    return None;
                }
            }

            if projection.segment_index == last_segment_index {
                let segment = geometry.segments[last_segment_index];
                if projection.distance_along_track >= geometry.total_length
                    && (point - segment.end).dot(segment.forward) > 0.0
                {
                    return None;
                }
            }

            if projection.distance_squared > geometry.broad_phase_radius_squared {
                return None;
            }

            let segment = geometry.segments[projection.segment_index];
            let local = point - projection.projection;
            let lateral = local.dot(segment.right);
            let vertical = local.dot(segment.up);
            if lateral.abs() > geometry.half_width || vertical.abs() > geometry.half_height {
                return None;
            }

            Some(CollisionPoint {
                point,
                distance_along_track: projection.distance_along_track,
                offset: projection.distance_squared.sqrt(),
            })
        })
        .collect()
}

/// Builds the four longitudinal lines describing the train envelope.
pub fn build_train_envelope(
    path: &[Vec3],
    train_width: f32,
    train_height: f32,
    numeric: NumericSettings,
) -> Vec<Vec<Vec3>> {
    if path.len() < 2 {
        return vec![Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    }

    let half_width = train_width * 0.5;
    let mut bottom_left = Vec::with_capacity(path.len());
    let mut bottom_right = Vec::with_capacity(path.len());
    let mut top_left = Vec::with_capacity(path.len());
    let mut top_right = Vec::with_capacity(path.len());

    for i in 0..path.len() {
        let forward = if i + 1 < path.len() {
            (path[i + 1] - path[i]).normalize_or_zero()
        } else {
            (path[i] - path[i - 1]).normalize_or_zero()
        };

        let mut right = forward.cross(Vec3::Z);
        if right.length_squared() < numeric.basis_epsilon {
            right = forward.cross(Vec3::X);
        }
        let right = right.normalize_or_zero();
        let up = right.cross(forward).normalize_or_zero();
        let center = path[i];
        let half_height = train_height * 0.5;

        bottom_left.push(center - right * half_width - up * half_height);
        bottom_right.push(center + right * half_width - up * half_height);
        top_left.push(center - right * half_width + up * half_height);
        top_right.push(center + right * half_width + up * half_height);
    }

    vec![bottom_left, bottom_right, top_left, top_right]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn box3_normalizes_corners() {
        let box3 = Box3::new(Vec3::new(2.0, 3.0, 4.0), Vec3::new(-1.0, -2.0, 0.0));
        assert_eq!(box3.min, Vec3::new(-1.0, -2.0, 0.0));
        assert_eq!(box3.max, Vec3::new(2.0, 3.0, 4.0));
        assert!(box3.contains(Vec3::ZERO));
        assert!(!box3.contains(Vec3::new(3.0, 0.0, 0.0)));
    }

    #[test]
    fn track_projection_returns_distance_along_track() {
        let path = [Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, -10.0, 0.0)];
        let geometry = TrackGeometry::new(&path, 2.0, 2.0, NumericSettings::default())
            .expect("valid path should build geometry");
        let projection = geometry
            .project(Vec3::new(0.5, -3.0, 0.0))
            .expect("point should be within broad phase");

        assert!((projection.distance_along_track - 3.0).abs() < 1e-5);
        assert!((projection.distance_squared - 0.25).abs() < 1e-5);
    }

    #[test]
    fn train_gauge_rejects_points_outside_height() {
        let path = [Vec3::ZERO, Vec3::new(0.0, -10.0, 0.0)];
        let geometry = TrackGeometry::new(&path, 2.0, 2.0, NumericSettings::default())
            .expect("valid path should build geometry");
        let points = [Vec3::new(0.2, -3.0, 0.5), Vec3::new(0.2, -3.0, 1.1)];
        let collisions = find_points_in_train_gauge(&points, &geometry);
        assert_eq!(collisions.len(), 1);
    }

    #[test]
    fn smoothing_preserves_empty_input() {
        assert!(smooth_path(&[], 20, 20).is_empty());
    }
}
