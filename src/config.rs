//! Runtime configuration loaded from `settings.json`.
//!
//! The configuration is deliberately grouped by responsibility so that the
//! numerical parameters of the point-cloud pipeline are not scattered through
//! the implementation.

use crate::calculations::Box3;
use anyhow::{Context, Result};
use rerun::external::glam::Vec3;
use serde::{Deserialize, Serialize};
use std::{fs, path::Path};

/// Selects the execution backend for the tunnel-center cost function.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ComputeBackend {
    /// Prefer GPU when a suitable discrete GPU is available, otherwise use CPU.
    Auto,
    /// Force the Rayon CPU implementation.
    Cpu,
    /// Prefer GPU and fall back to CPU when initialization or execution fails.
    Gpu,
}

/// ROS 2 topics and transport-loop parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct RosSettings {
    /// Input `sensor_msgs/PointCloud2` topic.
    pub input_topic: String,
    /// JSON obstacle report topic.
    pub obstacle_topic: String,
    /// Boolean obstacle-presence topic.
    pub obstacle_detected_topic: String,
    /// Nearest-obstacle distance topic in metres.
    pub nearest_obstacle_distance_topic: String,
    /// Value published when no obstacle is present.
    pub no_obstacle_distance: f32,
    /// Capacity of the synchronisation channel between the ROS subscriber thread and the processing loop.
    pub channel_size: usize,
    /// `Node::spin_once` period in milliseconds.
    pub spin_period_ms: u64,
}

impl Default for RosSettings {
    fn default() -> Self {
        Self {
            input_topic: "/lidar_points".to_string(),
            obstacle_topic: "/lidar/obstacles".to_string(),
            obstacle_detected_topic: "/lidar/obstacle_detected".to_string(),
            nearest_obstacle_distance_topic: "/lidar/nearest_obstacle_distance".to_string(),
            no_obstacle_distance: -1.0,
            channel_size: 5,
            spin_period_ms: 10,
        }
    }
}

/// Parameters used to estimate the left and right rail directions.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct CalibrationSettings {
    /// ROI containing the rail on the left side of the track.
    pub left_rail_box: Box3,
    /// ROI containing the rail on the right side of the track.
    pub right_rail_box: Box3,
    /// Number of bins along the Y axis.
    pub bins: usize,
    /// One-based rank of the high-Z point used as the reference height inside a bin.
    pub selector_rank: usize,
    /// Allowed Z deviation around the reference height.
    pub height_tolerance: f32,
    /// Minimum Y span required to accept a rail estimate.
    pub minimum_y_span: f32,
}

impl Default for CalibrationSettings {
    fn default() -> Self {
        Self {
            right_rail_box: Box3::new(
                Vec3::new(-1.2, -17.0, -1.9),
                Vec3::new(-0.7, -2.0, -0.9),
            ),
            left_rail_box: Box3::new(
                Vec3::new(0.4, -17.0, -1.9),
                Vec3::new(0.9, -2.0, -0.9),
            ),
            bins: 10,
            selector_rank: 3,
            height_tolerance: 0.015,
            minimum_y_span: 0.05,
        }
    }
}

/// Parameters controlling the tunnel-center search.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct TunnelPathSettings {
    /// Initial search box and reference tunnel volume.
    pub start_box: Box3,
    /// Assumed tunnel width used to restrict candidate points.
    pub width: f32,
    /// Assumed tunnel height used to restrict candidate points.
    pub height: f32,
    /// Thickness of a longitudinal point-cloud slice.
    pub slice_thickness: f32,
    /// Step length between successive tunnel-center points.
    pub step_length: f32,
    /// Candidate grid spacing in X and Z.
    pub grid_resolution: f32,
    /// Repulsion term coefficient in the center cost function.
    pub repulsion_factor: f64,
    /// Spring term coefficient in the center cost function.
    pub spring_factor: f64,
    /// Maximum number of longitudinal search steps.
    pub max_steps: usize,
    /// Restrict source points to the local search window around the previous center.
    pub restrict_points_to_search_window: bool,
}

impl Default for TunnelPathSettings {
    fn default() -> Self {
        Self {
            start_box: Box3::new(
                Vec3::new(-2.5, -3.0, -0.2),
                Vec3::new(1.5, 0.0, 2.8),
            ),
            width: 6.0,
            height: 7.0,
            slice_thickness: 5.0,
            step_length: 1.0,
            grid_resolution: 0.1,
            repulsion_factor: 40.0,
            spring_factor: 1.0,
            max_steps: 250,
            restrict_points_to_search_window: true,
        }
    }
}

/// Train dimensions and trajectory smoothing parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct TrainSettings {
    /// Train width in metres.
    pub width: f32,
    /// Train height in metres.
    pub height: f32,
    /// Radius of the positional smoothing window.
    pub path_smoothing_radius: usize,
    /// Radius of the height smoothing window.
    pub height_smoothing_radius: usize,
    /// Number of points removed from the end of the path after smoothing.
    pub trim_end_points: usize,
}

impl Default for TrainSettings {
    fn default() -> Self {
        Self {
            width: 2.0,
            height: 2.8,
            path_smoothing_radius: 20,
            height_smoothing_radius: 20,
            trim_end_points: 5,
        }
    }
}

/// Point-cluster parameters used to turn in-gauge points into obstacles.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct DetectionSettings {
    /// Radius used by the spatial-hash clustering stage.
    pub cluster_radius: f32,
    /// Minimum number of points required to report a cluster as an obstacle.
    pub minimum_cluster_size: usize,
    /// Optional maximum distance along the track for obstacle reporting.
    pub maximum_alert_distance: Option<f32>,
}

impl Default for DetectionSettings {
    fn default() -> Self {
        Self {
            cluster_radius: 0.35,
            minimum_cluster_size: 3,
            maximum_alert_distance: None,
        }
    }
}

/// Numerical tolerances shared by geometry calculations.
#[derive(Debug, Clone, Serialize, Deserialize, Copy)]
#[serde(default)]
pub struct NumericSettings {
    /// Minimum accepted segment length.
    pub minimum_segment_length: f32,
    /// Multiplier used to build the broad-phase collision radius.
    pub broad_phase_scale: f32,
    /// Squared-length threshold for choosing a fallback basis vector.
    pub basis_epsilon: f32,
    /// Positive distance offset used by the tunnel-center cost function.
    pub cost_distance_offset: f32,
}

impl Default for NumericSettings {
    fn default() -> Self {
        Self {
            minimum_segment_length: 1e-6,
            broad_phase_scale: 1.5,
            basis_epsilon: 1e-5,
            cost_distance_offset: 0.01,
        }
    }
}

/// GPU/CPU execution thresholds.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ComputeSettings {
    /// Preferred compute backend.
    pub backend: ComputeBackend,
    /// Minimum number of grid candidates before attempting GPU execution.
    pub gpu_candidate_threshold: usize,
    /// Minimum number of points in a slice before attempting GPU execution.
    pub gpu_point_threshold: usize,
}

impl Default for ComputeSettings {
    fn default() -> Self {
        Self {
            backend: ComputeBackend::Auto,
            gpu_candidate_threshold: 512,
            gpu_point_threshold: 1_024,
        }
    }
}

/// Rerun visualization switches, colors and point/line sizes.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct VisualizationSettings {
    /// Enables creation of the Rerun connection.
    pub enabled: bool,
    /// Draw raw LiDAR points.
    pub raw_points: bool,
    /// Draw calibration and search ROI boxes.
    pub roi_boxes: bool,
    /// Draw rail direction vectors.
    pub calibration: bool,
    /// Draw train motion vector.
    pub train_motion: bool,
    /// Draw tunnel-center points.
    pub tunnel_center: bool,
    /// Draw tunnel-center polyline.
    pub tunnel_center_line: bool,
    /// Draw the smoothed train-center path.
    pub train_center_path: bool,
    /// Draw collision points.
    pub collisions: bool,
    /// Draw the train envelope.
    pub envelope: bool,
    /// Raw point radius.
    pub raw_point_radius: f32,
    /// Train path point radius.
    pub train_path_radius: f32,
    /// Collision point radius.
    pub collision_point_radius: f32,
    /// Line width.
    pub line_radius: f32,
    /// Length of displayed calibration/motion vectors.
    pub calibration_vector_length: f32,
    /// Raw point RGB color.
    pub raw_color: [u8; 3],
    /// Rail ROI RGB color.
    pub roi_color: [u8; 3],
    /// Center-box RGB color.
    pub center_box_color: [u8; 3],
    /// Calibration RGB color.
    pub calibration_color: [u8; 3],
    /// Motion-vector RGB color.
    pub motion_color: [u8; 3],
    /// Tunnel-center RGB color.
    pub tunnel_center_color: [u8; 3],
    /// Train-path RGB color.
    pub train_path_color: [u8; 3],
    /// Collision-point RGB color.
    pub collision_color: [u8; 3],
    /// Train-envelope RGB color.
    pub envelope_color: [u8; 3],
}

impl Default for VisualizationSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            raw_points: true,
            roi_boxes: false,
            calibration: true,
            train_motion: true,
            tunnel_center: true,
            tunnel_center_line: true,
            train_center_path: true,
            collisions: true,
            envelope: true,
            raw_point_radius: 0.02,
            train_path_radius: 0.05,
            collision_point_radius: 0.10,
            line_radius: 0.02,
            calibration_vector_length: 10.0,
            raw_color: [180, 180, 180],
            roi_color: [255, 0, 0],
            center_box_color: [255, 255, 0],
            calibration_color: [255, 0, 0],
            motion_color: [0, 0, 255],
            tunnel_center_color: [255, 100, 255],
            train_path_color: [0, 240, 240],
            collision_color: [255, 0, 0],
            envelope_color: [255, 100, 255],
        }
    }
}

/// Complete application configuration.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct Settings {
    /// ROS 2 configuration.
    pub ros: RosSettings,
    /// Rail-calibration configuration.
    pub calibration: CalibrationSettings,
    /// Tunnel-center search configuration.
    pub tunnel_path: TunnelPathSettings,
    /// Train geometry and smoothing configuration.
    pub train: TrainSettings,
    /// Obstacle-clustering configuration.
    pub detection: DetectionSettings,
    /// Shared numerical tolerances.
    pub numeric: NumericSettings,
    /// CPU/GPU execution configuration.
    pub compute: ComputeSettings,
    /// Rerun visualization configuration.
    pub visualize: VisualizationSettings,
}

/// Loads application settings from JSON.
///
/// When no explicit path is provided and `settings.json` does not exist, the
/// built-in defaults are returned. Explicit paths and malformed JSON are
/// treated as errors and include the path in the error context.
///
/// # Errors
///
/// Returns an error when an explicitly selected file cannot be read or when
/// the file contents are not valid for [`Settings`].
pub fn read_settings(settings_file: Option<&Path>) -> Result<Settings> {
    let path = settings_file.unwrap_or_else(|| Path::new("settings.json"));

    if settings_file.is_none() && !path.exists() {
        return Ok(Settings::default());
    }

    let json = fs::read_to_string(path)
        .with_context(|| format!("Не удалось прочитать настройки: {}", path.display()))?;
    serde_json::from_str::<Settings>(&json)
        .with_context(|| format!("Некорректный JSON конфигурации: {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn settings_default_round_trip_json() {
        let settings = Settings::default();
        let json = serde_json::to_string(&settings).expect("default settings should serialize");
        let restored: Settings = serde_json::from_str(&json).expect("serialized settings should deserialize");
        assert_eq!(restored.ros.channel_size, settings.ros.channel_size);
        assert_eq!(restored.tunnel_path.max_steps, settings.tunnel_path.max_steps);
        assert_eq!(restored.train.width, settings.train.width);
    }
}
