//! Rerun visualization support.
//!
//! Empty visualization samples are represented using Rerun clear tombstones,
//! so a previous frame's geometry does not remain visible when the current
//! frame has no corresponding points or lines.

use crate::calculations::Box3;
use crate::config::VisualizationSettings;
use rerun::{
    Arrows3D, Boxes3D, Clear, Color, EntityPath, FillMode, LineStrips3D, Points3D,
    RecordingStream, RecordingStreamBuilder, external::glam::Vec3,
};
use std::ops::Deref;

/// Rerun recording connection used by the application.
pub struct RerunConnector {
    recording: RecordingStream,
}

#[inline]
fn color(value: [u8; 3]) -> Color {
    Color::from_rgb(value[0], value[1], value[2])
}

impl RerunConnector {
    /// Creates a gRPC recording stream for the `Metro_lidar` recording.
    ///
    /// # Errors
    ///
    /// Returns an error when Rerun cannot start its gRPC service.
    pub fn new() -> Result<Self, anyhow::Error> {
        let recording = RecordingStreamBuilder::new("Metro_lidar").serve_grpc()?;
        Ok(Self { recording })
    }

    /// Sets the application frame number on the Rerun timeline.
    pub fn set_frame(&self, frame: i64) {
        self.set_time_sequence("frame", frame);
    }

    /// Logs points to an entity path.
    ///
    /// If `points` is empty, a flat clear is logged instead of silently doing
    /// nothing. This prevents old points from staying visible in subsequent
    /// frames.
    ///
    /// # Errors
    ///
    /// Returns an error when Rerun rejects the log operation.
    pub fn log_points(
        &self,
        points: &[Vec3],
        color: Color,
        path: impl Into<EntityPath>,
        radius: f32,
    ) -> Result<(), anyhow::Error> {
        let path = path.into();
        if points.is_empty() {
            self.log(path, &Clear::flat())?;
            return Ok(());
        }

        let xyz: Vec<(f32, f32, f32)> = points.iter().map(|p| (p.x, p.y, p.z)).collect();
        self.log(
            path,
            &Points3D::new(xyz)
                .with_radii([radius])
                .with_colors([color]),
        )?;
        Ok(())
    }

    /// Logs an axis-aligned box.
    ///
    /// # Errors
    ///
    /// Returns an error when Rerun rejects the log operation.
    pub fn log_box(
        &self,
        bounding_box: &Box3,
        color: Color,
        path: impl Into<EntityPath>,
    ) -> Result<(), anyhow::Error> {
        let center = bounding_box.center();
        let size = bounding_box.max - bounding_box.min;
        self.log(
            path,
            &Boxes3D::from_centers_and_sizes(
                [(center.x, center.y, center.z)],
                [(size.x, size.y, size.z)],
            )
            .with_fill_mode(FillMode::Solid)
            .with_colors([color]),
        )?;
        Ok(())
    }

    /// Logs a 3D arrow.
    ///
    /// # Errors
    ///
    /// Returns an error when Rerun rejects the log operation.
    pub fn log_arrow(
        &self,
        origin: Vec3,
        vector: Vec3,
        color: Color,
        path: impl Into<EntityPath>,
    ) -> Result<(), anyhow::Error> {
        self.log(
            path,
            &Arrows3D::from_vectors([vector])
                .with_origins([origin])
                .with_colors([color]),
        )?;
        Ok(())
    }

    /// Logs one or more polyline strips.
    ///
    /// If every line is empty, the entity is explicitly cleared so data from a
    /// previous frame is removed from the latest-at view.
    ///
    /// # Errors
    ///
    /// Returns an error when Rerun rejects the log operation.
    pub fn log_lines(
        &self,
        lines: Vec<Vec<Vec3>>,
        color: Color,
        path: impl Into<EntityPath>,
        radius: f32,
    ) -> Result<(), anyhow::Error> {
        let path = path.into();
        if lines.iter().all(Vec::is_empty) {
            self.log(path, &Clear::flat())?;
            return Ok(());
        }
        self.log(
            path,
            &LineStrips3D::new(lines)
                .with_colors([color])
                .with_radii([radius]),
        )?;
        Ok(())
    }

    /// Logs all enabled visualization layers for one processed frame.
    ///
    /// Individual layers are controlled by [`VisualizationSettings`]. Empty
    /// point and line layers are cleared automatically by [`Self::log_points`]
    /// and [`Self::log_lines`].
    ///
    /// # Errors
    ///
    /// Returns the first Rerun logging error encountered while writing a layer.
    pub fn log_all(
        &self,
        settings: &VisualizationSettings,
        raw_points: &[Vec3],
        left_box: &Box3,
        right_box: &Box3,
        center_box: &Box3,
        left_start: Vec3,
        left_direction: Vec3,
        right_start: Vec3,
        right_direction: Vec3,
        train_origin: Vec3,
        train_direction: Vec3,
        tunnel_path: &[Vec3],
        train_path: &[Vec3],
        collision_points: &[Vec3],
        envelope: Vec<Vec<Vec3>>,
    ) -> Result<(), anyhow::Error> {
        if settings.raw_points {
            self.log_points(
                raw_points,
                color(settings.raw_color),
                "world/raw_points",
                settings.raw_point_radius,
            )?;
        }
        if settings.roi_boxes {
            self.log_box(left_box, color(settings.roi_color), "world/left_rail_box")?;
            self.log_box(right_box, color(settings.roi_color), "world/right_rail_box")?;
            self.log_box(
                center_box,
                color(settings.center_box_color),
                "world/center_box",
            )?;
        }
        if settings.calibration {
            self.log_arrow(
                left_start,
                left_direction * settings.calibration_vector_length,
                color(settings.calibration_color),
                "world/left_rail_vector",
            )?;
            self.log_arrow(
                right_start,
                right_direction * settings.calibration_vector_length,
                color(settings.calibration_color),
                "world/right_rail_vector",
            )?;
        }
        if settings.train_motion {
            self.log_arrow(
                train_origin,
                train_direction * settings.calibration_vector_length,
                color(settings.motion_color),
                "world/train_motion",
            )?;
        }
        if settings.tunnel_center {
            self.log_points(
                tunnel_path,
                color(settings.tunnel_center_color),
                "world/tunnel_center_points",
                settings.raw_point_radius,
            )?;
        }
        if settings.tunnel_center_line {
            let lines = if tunnel_path.len() > 1 {
                vec![tunnel_path.to_vec()]
            } else {
                vec![Vec::new()]
            };
            self.log_lines(
                lines,
                color(settings.tunnel_center_color),
                "world/tunnel_center_line",
                settings.line_radius,
            )?;
        }
        if settings.train_center_path {
            self.log_points(
                train_path,
                color(settings.train_path_color),
                "world/train_center_path",
                settings.train_path_radius,
            )?;
        }
        if settings.collisions {
            self.log_points(
                collision_points,
                color(settings.collision_color),
                "world/collisions",
                settings.collision_point_radius,
            )?;
        }
        if settings.envelope {
            self.log_lines(
                envelope,
                color(settings.envelope_color),
                "world/train_envelope",
                settings.line_radius,
            )?;
        }
        Ok(())
    }
}

impl Deref for RerunConnector {
    type Target = RecordingStream;

    fn deref(&self) -> &Self::Target {
        &self.recording
    }
}
