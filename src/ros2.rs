//! ROS 2 input/output integration.

use crate::obstacles::ObstacleReport;
use futures::stream::StreamExt;
use r2r::sensor_msgs::msg::PointCloud2;
use r2r::std_msgs::msg::{Bool, Float32, String as RosString};
use rerun::external::glam::Vec3;
use std::sync::mpsc;
use std::time::Duration;

/// ROS 2 connector used by the processing loop.
pub struct RosConnector {
    /// Underlying r2r node.
    pub node: r2r::Node,
    cloud_rx: mpsc::Receiver<PointCloud2>,
    obstacle_publisher: r2r::Publisher<RosString>,
    obstacle_detected_publisher: r2r::Publisher<Bool>,
    nearest_distance_publisher: r2r::Publisher<Float32>,
    no_obstacle_distance: f32,
}

impl RosConnector {
    /// Creates a node, subscribes to the point-cloud topic and creates the
    /// three obstacle result publishers.
    ///
    /// # Errors
    ///
    /// Returns an error if the ROS context, node, subscription, or any publisher
    /// cannot be created.
    pub fn new(
        input_topic: &str,
        obstacle_topic: &str,
        obstacle_detected_topic: &str,
        nearest_distance_topic: &str,
        channel_size: usize,
        no_obstacle_distance: f32,
    ) -> Result<Self, anyhow::Error> {
        let context = r2r::Context::create()?;
        let mut node = r2r::Node::create(context, "lidar_obstacle_node", "")?;
        let (sender, cloud_rx) = mpsc::sync_channel::<PointCloud2>(channel_size);
        let mut subscription =
            node.subscribe::<PointCloud2>(input_topic, r2r::QosProfile::default())?;

        let obstacle_publisher =
            node.create_publisher::<RosString>(obstacle_topic, r2r::QosProfile::default())?;
        let obstacle_detected_publisher = node
            .create_publisher::<Bool>(obstacle_detected_topic, r2r::QosProfile::default())?;
        let nearest_distance_publisher = node.create_publisher::<Float32>(
            nearest_distance_topic,
            r2r::QosProfile::default(),
        )?;

        std::thread::spawn(move || {
            futures::executor::block_on(async move {
                while let Some(message) = subscription.next().await {
                    if let Err(error) = sender.try_send(message) {
                        eprintln!("ROS point-cloud queue is full or disconnected: {error}");
                    }
                }
            });
        });

        Ok(Self {
            node,
            cloud_rx,
            obstacle_publisher,
            obstacle_detected_publisher,
            nearest_distance_publisher,
            no_obstacle_distance,
        })
    }

    /// Spins the ROS node for at most the requested duration.
    pub fn spin_once(&mut self, timeout: Duration) {
        self.node.spin_once(timeout);
    }

    /// Receives one cloud, parses it into an existing output buffer and returns
    /// whether a message was available.
    pub fn try_recv_cloud_into(&self, output: &mut Vec<Vec3>) -> bool {
        match self.cloud_rx.try_recv() {
            Ok(message) => {
                parse_point_cloud2_into(&message, output);
                true
            }
            Err(_) => false,
        }
    }

    /// Publishes a JSON obstacle report, obstacle-presence flag and nearest
    /// obstacle distance.
    ///
    /// # Errors
    ///
    /// Returns an error if report serialization or any ROS publication fails.
    pub fn publish_obstacles(&self, report: &ObstacleReport) -> Result<(), anyhow::Error> {
        let json = serde_json::to_string(report)?;
        self.obstacle_publisher.publish(&RosString { data: json })?;
        self.obstacle_detected_publisher.publish(&Bool {
            data: report.obstacle_detected,
        })?;
        self.nearest_distance_publisher.publish(&Float32 {
            data: report
                .nearest_distance_m
                .unwrap_or(self.no_obstacle_distance),
        })?;
        Ok(())
    }
}

/// Parses XYZ coordinates from a `sensor_msgs/PointCloud2` message into an
/// existing vector buffer.
///
/// The current project expects XYZ to occupy the first 12 bytes of each point
/// in little-endian `f32` representation. Invalid floating-point values are
/// skipped. A `point_step` smaller than 12 is treated as an empty cloud.
pub fn parse_point_cloud2_into(message: &PointCloud2, output: &mut Vec<Vec3>) {
    let point_step = message.point_step as usize;
    let total_points = (message.width * message.height) as usize;

    output.clear();
    if point_step < 12 {
        return;
    }
    if output.capacity() < total_points {
        output.reserve(total_points - output.capacity());
    }

    for chunk in message.data.chunks_exact(point_step) {
        let x = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        let y = f32::from_le_bytes([chunk[4], chunk[5], chunk[6], chunk[7]]);
        let z = f32::from_le_bytes([chunk[8], chunk[9], chunk[10], chunk[11]]);
        if x.is_finite() && y.is_finite() && z.is_finite() {
            output.push(Vec3::new(x, y, z));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn point_cloud_parser_reuses_output_and_skips_invalid_points() {
        let mut data = Vec::new();
        data.extend_from_slice(&1.0f32.to_le_bytes());
        data.extend_from_slice(&2.0f32.to_le_bytes());
        data.extend_from_slice(&3.0f32.to_le_bytes());
        data.extend_from_slice(&4.0f32.to_le_bytes());
        data.extend_from_slice(&5.0f32.to_le_bytes());
        data.extend_from_slice(&f32::NAN.to_le_bytes());
        data.extend_from_slice(&6.0f32.to_le_bytes());
        data.extend_from_slice(&7.0f32.to_le_bytes());
        data.extend_from_slice(&8.0f32.to_le_bytes());

        let message = PointCloud2 {
            height: 1,
            width: 3,
            fields: Vec::new(),
            is_bigendian: false,
            point_step: 12,
            row_step: 36,
            data,
            is_dense: false,
            ..Default::default()
        };
        let mut points = vec![Vec3::ONE];
        parse_point_cloud2_into(&message, &mut points);

        assert_eq!(points.len(), 2);
        assert_eq!(points[0], Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(points[1], Vec3::new(6.0, 7.0, 8.0));
    }
}
