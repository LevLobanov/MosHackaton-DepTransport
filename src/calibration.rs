//! Rail calibration and the transform used to align the track coordinate frame.

use rerun::external::glam::{Mat4, Quat, Vec3};

/// Rigid transform that maps points from the LiDAR frame into the calibrated
/// track frame.
#[derive(Debug, Clone, Copy, Default)]
pub struct RailTransform {
    /// Homogeneous transform matrix.
    pub matrix: Mat4,
}

impl RailTransform {
    /// Transforms a point, including translation.
    #[inline]
    pub fn transform_point(&self, point: Vec3) -> Vec3 {
        self.matrix.transform_point3(point)
    }

    /// Transforms a direction vector without applying translation.
    #[inline]
    pub fn transform_vector(&self, vector: Vec3) -> Vec3 {
        self.matrix.transform_vector3(vector)
    }
}

/// Result of estimating one rail's longitudinal direction.
#[allow(unused)]
#[derive(Debug, Clone, Copy, Default)]
pub struct CalibrationResult {
    /// Point selected near the beginning of the rail ROI.
    pub start_point: Vec3,
    /// Point one unit along the estimated rail direction from `start_point`.
    pub end_point: Vec3,
    /// Normalized rail direction.
    pub direction: Vec3,
}

/// Calibrates both rails and builds the transform of the complete track frame.
///
/// # Errors
///
/// The function uses `Option` because insufficient or degenerate point data is
/// an expected runtime condition rather than an exceptional application error.
/// `None` means that one or both rails could not be estimated reliably.
pub fn calibrate_track(
    left_points: &[Vec3],
    right_points: &[Vec3],
    settings: &crate::config::CalibrationSettings,
) -> Option<(CalibrationResult, CalibrationResult, RailTransform)> {
    let left = calibrate_rail(left_points, settings)?;
    let right = calibrate_rail(right_points, settings)?;

    let center = (left.start_point + right.start_point) * 0.5;
    let track_direction = (left.direction + right.direction).try_normalize()?;
    let track_rotation = Quat::from_rotation_arc(track_direction, -Vec3::Y);

    let left_rotated = track_rotation * (left.start_point - center);
    let right_rotated = track_rotation * (right.start_point - center);
    let rail_delta = right_rotated - left_rotated;
    let roll = rail_delta.z.atan2(rail_delta.x.abs());
    let rotation = Quat::from_rotation_y(-roll) * track_rotation;

    let rotated_center = rotation * center;
    let translation = Vec3::new(-rotated_center.x, 0.0, -rotated_center.z);
    let matrix = Mat4::from_translation(translation) * Mat4::from_quat(rotation);

    Some((left, right, RailTransform { matrix }))
}

fn calibrate_rail(
    points: &[Vec3],
    settings: &crate::config::CalibrationSettings,
) -> Option<CalibrationResult> {
    let (start, direction) = estimate_rail_direction(
        points,
        settings.bins,
        settings.selector_rank,
        settings.height_tolerance,
        settings.minimum_y_span,
    )?;

    let end = start + direction;
    Some(CalibrationResult {
        start_point: start,
        end_point: end,
        direction,
    })
}

fn estimate_rail_direction(
    points: &[Vec3],
    bin_count: usize,
    rank: usize,
    height_tolerance: f32,
    minimum_y_span: f32,
) -> Option<(Vec3, Vec3)> {
    if points.len() < bin_count.max(1) || bin_count == 0 {
        return None;
    }

    let (min_y, max_y) = points.iter().fold(
        (f32::INFINITY, f32::NEG_INFINITY),
        |(min_y, max_y), point| (min_y.min(point.y), max_y.max(point.y)),
    );

    if max_y - min_y < minimum_y_span {
        return None;
    }

    let bin_size = (max_y - min_y) / bin_count as f32;
    let mut bins: Vec<Vec<usize>> = (0..bin_count).map(|_| Vec::new()).collect();

    for (index, point) in points.iter().enumerate() {
        let mut bin = ((point.y - min_y) / bin_size).floor() as usize;
        bin = bin.min(bin_count - 1);
        bins[bin].push(index);
    }

    let start_bin = (0..bin_count).rev().find(|&bin| !bins[bin].is_empty())?;
    let end_bin = (0..bin_count).find(|&bin| !bins[bin].is_empty())?;

    let start = select_top_band_center(points, &bins[start_bin], rank, height_tolerance)?;
    let end = select_top_band_center(points, &bins[end_bin], rank, height_tolerance)?;
    let direction = (end - start).try_normalize()?;

    Some((start, direction))
}

fn select_top_band_center(
    points: &[Vec3],
    indices: &[usize],
    rank: usize,
    height_tolerance: f32,
) -> Option<Vec3> {
    if indices.is_empty() {
        return None;
    }

    let target = rank.saturating_sub(1).min(indices.len() - 1);
    let mut heights: Vec<f32> = indices.iter().map(|&index| points[index].z).collect();
    heights.select_nth_unstable_by(target, |left, right| right.total_cmp(left));

    let reference = heights[target];
    let lower = reference - height_tolerance;
    let upper = reference + height_tolerance;

    let mut sum = Vec3::ZERO;
    let mut count = 0.0;
    for &index in indices {
        let point = points[index];
        if point.z >= lower && point.z <= upper {
            sum += point;
            count += 1.0;
        }
    }

    (count > 0.0).then_some(sum / count)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::CalibrationSettings;

    #[test]
    fn rail_transform_vector_does_not_translate() {
        let transform = RailTransform {
            matrix: Mat4::from_translation(Vec3::new(10.0, 20.0, 30.0)),
        };
        assert_eq!(transform.transform_vector(Vec3::X), Vec3::X);
        assert_eq!(transform.transform_point(Vec3::ZERO), Vec3::new(10.0, 20.0, 30.0));
    }

    #[test]
    fn calibrate_track_finds_two_parallel_rails() {
        let settings = CalibrationSettings {
            bins: 2,
            selector_rank: 1,
            height_tolerance: 0.02,
            minimum_y_span: 0.5,
            ..CalibrationSettings::default()
        };

        let left = vec![
            Vec3::new(0.5, 0.0, 1.0),
            Vec3::new(0.5, -1.0, 1.0),
            Vec3::new(0.5, -2.0, 1.0),
            Vec3::new(0.5, -3.0, 1.0),
        ];
        let right = left
            .iter()
            .map(|point| *point + Vec3::new(-1.0, 0.0, 0.0))
            .collect::<Vec<_>>();

        let result = calibrate_track(&left, &right, &settings);
        assert!(result.is_some());
    }
}
