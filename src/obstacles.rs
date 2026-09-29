//! Clustering of train-gauge points into physical obstacle candidates.

use crate::calculations::CollisionPoint;
use crate::config::DetectionSettings;
use serde::Serialize;
use std::borrow::Cow;
use std::collections::HashMap;

/// A clustered obstacle detected inside the train gauge.
#[derive(Debug, Clone, Serialize)]
pub struct Obstacle {
    /// Stable identifier within the current frame, ordered by distance after clustering.
    pub id: usize,
    /// Number of source LiDAR points in the cluster.
    pub point_count: usize,
    /// Distance from the beginning of the path to the nearest cluster point, in metres.
    pub distance_m: f32,
    /// Lateral/Euclidean offset of the nearest cluster point from the path.
    pub offset_m: f32,
    /// Arithmetic center of the cluster.
    pub center: [f32; 3],
    /// Minimum XYZ bounds of the cluster.
    pub min: [f32; 3],
    /// Maximum XYZ bounds of the cluster.
    pub max: [f32; 3],
}

/// Detection summary published for every processed frame.
#[derive(Debug, Clone, Serialize)]
pub struct ObstacleReport {
    /// Source processing frame number.
    pub frame: i64,
    /// Whether at least one obstacle survived filtering and clustering.
    pub obstacle_detected: bool,
    /// Distance to the nearest reported obstacle, or `None` when no obstacle exists.
    pub nearest_distance_m: Option<f32>,
    /// Number of reported obstacles.
    pub obstacle_count: usize,
    /// Full obstacle descriptions.
    pub obstacles: Vec<Obstacle>,
}

impl ObstacleReport {
    /// Creates a report and derives the convenience summary fields.
    pub fn new(frame: i64, obstacles: Vec<Obstacle>) -> Self {
        Self {
            frame,
            obstacle_detected: !obstacles.is_empty(),
            nearest_distance_m: obstacles.first().map(|obstacle| obstacle.distance_m),
            obstacle_count: obstacles.len(),
            obstacles,
        }
    }
}

#[derive(Clone, Copy, Hash, PartialEq, Eq)]
struct ClusterCell(i32, i32, i32);

struct DisjointSet {
    parent: Vec<usize>,
    rank: Vec<u8>,
}

impl DisjointSet {
    fn new(size: usize) -> Self {
        Self {
            parent: (0..size).collect(),
            rank: vec![0; size],
        }
    }

    fn find(&mut self, mut value: usize) -> usize {
        while self.parent[value] != value {
            self.parent[value] = self.parent[self.parent[value]];
            value = self.parent[value];
        }
        value
    }

    fn union(&mut self, left: usize, right: usize) {
        let left_root = self.find(left);
        let right_root = self.find(right);
        if left_root == right_root {
            return;
        }

        match self.rank[left_root].cmp(&self.rank[right_root]) {
            std::cmp::Ordering::Less => self.parent[left_root] = right_root,
            std::cmp::Ordering::Greater => self.parent[right_root] = left_root,
            std::cmp::Ordering::Equal => {
                self.parent[right_root] = left_root;
                self.rank[left_root] += 1;
            }
        }
    }
}

#[derive(Default)]
struct ClusterAggregate {
    count: usize,
    sum: [f32; 3],
    min: [f32; 3],
    max: [f32; 3],
    nearest_distance: f32,
    nearest_offset: f32,
}

impl ClusterAggregate {
    fn new(point: CollisionPoint) -> Self {
        Self {
            count: 1,
            sum: [point.point.x, point.point.y, point.point.z],
            min: [point.point.x, point.point.y, point.point.z],
            max: [point.point.x, point.point.y, point.point.z],
            nearest_distance: point.distance_along_track,
            nearest_offset: point.offset,
        }
    }

    fn add(&mut self, point: CollisionPoint) {
        self.count += 1;
        self.sum[0] += point.point.x;
        self.sum[1] += point.point.y;
        self.sum[2] += point.point.z;
        self.min[0] = self.min[0].min(point.point.x);
        self.min[1] = self.min[1].min(point.point.y);
        self.min[2] = self.min[2].min(point.point.z);
        self.max[0] = self.max[0].max(point.point.x);
        self.max[1] = self.max[1].max(point.point.y);
        self.max[2] = self.max[2].max(point.point.z);

        if point.distance_along_track < self.nearest_distance {
            self.nearest_distance = point.distance_along_track;
            self.nearest_offset = point.offset;
        }
    }

    fn finish(self, id: usize) -> Obstacle {
        let count = self.count as f32;
        Obstacle {
            id,
            point_count: self.count,
            distance_m: self.nearest_distance,
            offset_m: self.nearest_offset,
            center: [self.sum[0] / count, self.sum[1] / count, self.sum[2] / count],
            min: self.min,
            max: self.max,
        }
    }
}

#[inline]
fn cell_of(point: CollisionPoint, radius: f32) -> ClusterCell {
    ClusterCell(
        (point.point.x / radius).floor() as i32,
        (point.point.y / radius).floor() as i32,
        (point.point.z / radius).floor() as i32,
    )
}

/// Groups nearby in-gauge points into obstacle clusters.
///
/// A spatial hash avoids comparing every point with every other point. A
/// disjoint-set structure then merges points whose actual Euclidean distance
/// is within the configured cluster radius.
pub fn cluster_obstacles(points: &[CollisionPoint], settings: &DetectionSettings) -> Vec<Obstacle> {
    if points.is_empty() || settings.cluster_radius <= 0.0 {
        return Vec::new();
    }

    let filtered_points: Cow<'_, [CollisionPoint]> = match settings.maximum_alert_distance {
        Some(limit) => Cow::Owned(
            points
                .iter()
                .copied()
                .filter(|point| point.distance_along_track <= limit)
                .collect(),
        ),
        None => Cow::Borrowed(points),
    };

    if filtered_points.is_empty() {
        return Vec::new();
    }

    let points = filtered_points.as_ref();
    let radius_squared = settings.cluster_radius * settings.cluster_radius;
    let mut cells: HashMap<ClusterCell, Vec<usize>> = HashMap::new();
    for (index, &point) in points.iter().enumerate() {
        cells
            .entry(cell_of(point, settings.cluster_radius))
            .or_default()
            .push(index);
    }

    let mut sets = DisjointSet::new(points.len());

    for (index, &point) in points.iter().enumerate() {
        let cell = cell_of(point, settings.cluster_radius);
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    let neighbour = ClusterCell(cell.0 + dx, cell.1 + dy, cell.2 + dz);
                    let Some(indices) = cells.get(&neighbour) else {
                        continue;
                    };

                    for &other in indices {
                        if other <= index {
                            continue;
                        }
                        if point.point.distance_squared(points[other].point) <= radius_squared {
                            sets.union(index, other);
                        }
                    }
                }
            }
        }
    }

    let mut clusters: HashMap<usize, ClusterAggregate> = HashMap::new();
    for (index, &point) in points.iter().enumerate() {
        let root = sets.find(index);
        clusters
            .entry(root)
            .and_modify(|cluster| cluster.add(point))
            .or_insert_with(|| ClusterAggregate::new(point));
    }

    let mut obstacles = clusters
        .into_values()
        .filter(|cluster| cluster.count >= settings.minimum_cluster_size)
        .enumerate()
        .map(|(id, cluster)| cluster.finish(id + 1))
        .collect::<Vec<_>>();

    obstacles.sort_unstable_by(|left, right| left.distance_m.total_cmp(&right.distance_m));
    for (id, obstacle) in obstacles.iter_mut().enumerate() {
        obstacle.id = id + 1;
    }
    obstacles
}

#[cfg(test)]
mod tests {
    use super::*;
    use rerun::external::glam::Vec3;

    #[test]
    fn nearby_collision_points_form_one_obstacle() {
        let points = [
            CollisionPoint {
                point: Vec3::new(0.0, -10.0, 1.0),
                distance_along_track: 10.0,
                offset: 0.2,
            },
            CollisionPoint {
                point: Vec3::new(0.1, -10.0, 1.0),
                distance_along_track: 10.1,
                offset: 0.21,
            },
            CollisionPoint {
                point: Vec3::new(0.0, -10.1, 1.0),
                distance_along_track: 10.1,
                offset: 0.22,
            },
            CollisionPoint {
                point: Vec3::new(4.0, -10.0, 1.0),
                distance_along_track: 10.0,
                offset: 4.0,
            },
        ];

        let obstacles = cluster_obstacles(&points, &DetectionSettings::default());
        assert_eq!(obstacles.len(), 1);
        assert_eq!(obstacles[0].point_count, 3);
        assert!((obstacles[0].distance_m - 10.0).abs() < 1e-5);
    }
}
