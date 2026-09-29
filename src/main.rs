//! LiDAR-based obstacle detection pipeline for an autonomous metro train.
//!
//! The application receives ROS 2 point clouds, calibrates the rails, searches
//! for the tunnel centerline, builds the train trajectory, finds points inside
//! the train gauge, clusters them into obstacles, and publishes the result.

#![deny(missing_docs)]
#![allow(clippy::too_many_arguments)]

mod calculations;
mod calibration;
mod compute;
mod config;
mod obstacles;
mod ros2;
mod visualize;

use crate::calculations::{
    build_train_envelope, calculate_train_motion, find_points_in_train_gauge, smooth_path,
    TrackGeometry, TunnelPathSearcher,
};
use crate::calibration::{calibrate_track, CalibrationResult};
use crate::config::Settings;
use crate::obstacles::ObstacleReport;
use crate::ros2::RosConnector;
use crate::visualize::RerunConnector;
use anyhow::Result;
use rerun::external::glam::Vec3;
use std::path::Path;
use std::time::Duration;

fn calibrate_frame(
    raw_points: &[Vec3],
    settings: &Settings,
    left_points: &mut Vec<Vec3>,
    right_points: &mut Vec<Vec3>,
) -> Option<(
    CalibrationResult,
    CalibrationResult,
    crate::calibration::RailTransform,
)> {
    settings
        .calibration
        .left_rail_box
        .filter_points_into(raw_points, left_points);
    settings
        .calibration
        .right_rail_box
        .filter_points_into(raw_points, right_points);

    calibrate_track(left_points, right_points, &settings.calibration)
}

fn main() -> Result<()> {
    println!("Starting...");

    let settings_path = std::env::args().nth(1);
    let settings = crate::config::read_settings(settings_path.as_deref().map(Path::new))?;

    println!("  [Settings loaded]");

    let rerun = if settings.visualize.enabled {
        Some(RerunConnector::new()?)
    } else {
        None
    };

    let mut ros = RosConnector::new(
        &settings.ros.input_topic,
        &settings.ros.obstacle_topic,
        &settings.ros.obstacle_detected_topic,
        &settings.ros.nearest_obstacle_distance_topic,
        settings.ros.channel_size,
        settings.ros.no_obstacle_distance,
    )?;

    let mut frame_id = 0_i64;
    let mut raw_points = Vec::new();
    let mut left_points = Vec::new();
    let mut right_points = Vec::new();
    let mut path_searcher = TunnelPathSearcher::new(&settings.compute);

    println!("Waiting for data to calibrate...");

    loop {
        ros.spin_once(Duration::from_millis(settings.ros.spin_period_ms));
        if !ros.try_recv_cloud_into(&mut raw_points) {
            continue;
        }

        if calibrate_frame(&raw_points, &settings, &mut left_points, &mut right_points).is_some() {
            frame_id += 1;
            println!("  [Calibrated]");
            break;
        }

        println!("Something went wrong with calibration, continue...");
    }

    loop {
        ros.spin_once(Duration::from_millis(settings.ros.spin_period_ms));

        while ros.try_recv_cloud_into(&mut raw_points) {
            frame_id += 1;

            let tunnel_path = path_searcher.search(
                &raw_points,
                &settings.tunnel_path,
                &settings.numeric,
            );

            let Some((left_rail, right_rail, rail_transform)) = calibrate_frame(
                &raw_points,
                &settings,
                &mut left_points,
                &mut right_points,
            ) else {
                eprintln!("Frame {frame_id}: can't calibrate railing");
                continue;
            };

            let (train_start, train_direction) = calculate_train_motion(
                &left_rail,
                &right_rail,
                settings.train.height,
            );

            let mut train_path: Vec<Vec3> = smooth_path(
                &tunnel_path,
                settings.train.path_smoothing_radius,
                settings.train.height_smoothing_radius,
            );
            train_path.truncate(
                train_path
                    .len()
                    .saturating_sub(settings.train.trim_end_points),
            );

            if train_path.len() < 2 {
                let report = ObstacleReport::new(frame_id, Vec::new());
                ros.publish_obstacles(&report)?;
                println!("Frame {frame_id}: can't calculate a trajectory");
                continue;
            }

            let track_geometry = TrackGeometry::new(
                &train_path,
                settings.train.width,
                settings.train.height,
                settings.numeric,
            )?;

            let collision_points = find_points_in_train_gauge(&raw_points, &track_geometry);
            let obstacles = crate::obstacles::cluster_obstacles(&collision_points, &settings.detection);
            let report = ObstacleReport::new(frame_id, obstacles);
            ros.publish_obstacles(&report)?;

            if let Some(rerun) = rerun.as_ref() {
                rerun.set_frame(frame_id);

                let collision_points_for_viz: Vec<Vec3> = collision_points
                    .iter()
                    .map(|hit| hit.point)
                    .collect();

                let envelope = if settings.visualize.envelope {
                    build_train_envelope(
                        &train_path,
                        settings.train.width,
                        settings.train.height,
                        settings.numeric,
                    )
                } else {
                    Vec::new()
                };

                let train_origin = rail_transform.transform_point(train_start);
                let transformed_direction = rail_transform.transform_vector(train_direction);

                rerun.log_all(
                    &settings.visualize,
                    &raw_points,
                    &settings.calibration.left_rail_box,
                    &settings.calibration.right_rail_box,
                    &settings.tunnel_path.start_box,
                    left_rail.start_point,
                    left_rail.direction,
                    right_rail.start_point,
                    right_rail.direction,
                    train_origin,
                    transformed_direction,
                    &tunnel_path,
                    &train_path,
                    &collision_points_for_viz,
                    envelope,
                )?;
            }

            println!("  [Frame]");
        }
    }
}
