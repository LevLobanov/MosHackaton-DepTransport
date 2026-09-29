//! CPU and GPU implementations of the tunnel-center candidate search.
//!
//! The CPU implementation uses Rayon to evaluate independent grid candidates
//! in parallel. The GPU implementation uses a small wgpu compute shader and
//! automatically falls back to the CPU when the GPU is unavailable or fails.

use crate::config::{ComputeBackend, ComputeSettings};
use anyhow::{anyhow, Context, Result};
use rayon::prelude::*;
use rerun::external::glam::Vec3;
use std::borrow::Cow;
use std::sync::mpsc;

const GPU_WORKGROUP_SIZE: u32 = 64;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuParameters {
    bounds: [f32; 4],
    factors: [f32; 4],
    dimensions: [u32; 4],
}

/// Executes tunnel-center candidate evaluations on CPU and, when available,
/// on a discrete GPU.
pub struct ComputeEngine {
    gpu: Option<GpuEvaluator>,
    settings: ComputeSettings,
}

impl ComputeEngine {
    /// Creates a compute engine according to the requested backend.
    ///
    /// GPU initialization is intentionally best-effort. When it fails, the
    /// engine remains usable with the Rayon CPU implementation.
    pub fn new(settings: &ComputeSettings) -> Self {
        let gpu = match settings.backend {
            ComputeBackend::Cpu => None,
            ComputeBackend::Auto | ComputeBackend::Gpu => match GpuEvaluator::new() {
                Ok(gpu) => {
                    println!("GPU compute: {}", gpu.adapter_name());
                    Some(gpu)
                }
                Err(error) => {
                    eprintln!("GPU compute недоступен, используется Rayon CPU: {error:#}");
                    None
                }
            },
        };

        Self {
            gpu,
            settings: settings.clone(),
        }
    }

    /// Returns a human-readable description of the active execution backend.
    #[allow(unused)]
    pub fn backend_name(&self) -> &'static str {
        if self.gpu.is_some() {
            "wgpu + rayon fallback"
        } else {
            "rayon cpu"
        }
    }

    /// Finds the cheapest candidate from a regular X/Z search grid.
    ///
    /// Returns the flattened candidate index, or `None` when the grid is empty
    /// or there are no source points.
    pub fn best_candidate(
        &mut self,
        point_indices: &[usize],
        points: &[Vec3],
        x_min: f32,
        z_min: f32,
        current_y: f32,
        grid_resolution: f32,
        x_steps: usize,
        z_steps: usize,
        repulsion_factor: f64,
        spring_factor: f64,
        distance_offset: f32,
    ) -> Option<usize> {
        let candidate_count = x_steps.checked_mul(z_steps)?;
        if candidate_count == 0 || point_indices.is_empty() {
            return None;
        }

        let use_gpu = self.gpu.is_some()
            && candidate_count >= self.settings.gpu_candidate_threshold
            && point_indices.len() >= self.settings.gpu_point_threshold;

        if use_gpu {
            if let Some(gpu) = self.gpu.as_mut() {
                match gpu.find_best_candidate(
                    point_indices,
                    points,
                    x_min,
                    z_min,
                    current_y,
                    grid_resolution,
                    x_steps,
                    z_steps,
                    repulsion_factor as f32,
                    spring_factor as f32,
                    distance_offset,
                ) {
                    Ok(index) => return Some(index),
                    Err(error) => {
                        eprintln!("GPU compute error, переключаюсь на Rayon CPU: {error:#}");
                    }
                }
            }
            self.gpu = None;
        }

        cpu_best_candidate(
            point_indices,
            points,
            x_min,
            z_min,
            current_y,
            grid_resolution,
            x_steps,
            z_steps,
            repulsion_factor,
            spring_factor,
            distance_offset,
        )
    }
}

/// Evaluates every grid candidate in parallel on the CPU.
fn cpu_best_candidate(
    point_indices: &[usize],
    points: &[Vec3],
    x_min: f32,
    z_min: f32,
    current_y: f32,
    grid_resolution: f32,
    x_steps: usize,
    z_steps: usize,
    repulsion_factor: f64,
    spring_factor: f64,
    distance_offset: f32,
) -> Option<usize> {
    let candidate_count = x_steps.checked_mul(z_steps)?;

    (0..candidate_count)
        .into_par_iter()
        .map(|candidate_index| {
            let x_index = candidate_index / z_steps;
            let z_index = candidate_index % z_steps;
            let x = x_min + x_index as f32 * grid_resolution;
            let z = z_min + z_index as f32 * grid_resolution;

            let mut cost = 0.0_f64;
            for &point_index in point_indices {
                let point = points[point_index];
                let dx = x as f64 - point.x as f64;
                let dy = current_y as f64 - point.y as f64;
                let dz = z as f64 - point.z as f64;
                let distance_squared = dx * dx + dy * dy + dz * dz;
                let denominator = distance_squared + distance_offset as f64;

                cost += (repulsion_factor / denominator).exp();
                cost += (denominator * spring_factor).exp();
            }

            (candidate_index, cost)
        })
        .reduce_with(|left, right| {
            if left.1.total_cmp(&right.1).is_le() {
                left
            } else {
                right
            }
        })
        .map(|(index, _)| index)
}

struct GpuEvaluator {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    params_buffer: wgpu::Buffer,
    point_buffer: wgpu::Buffer,
    cost_buffer: wgpu::Buffer,
    readback_buffer: wgpu::Buffer,
    point_capacity: usize,
    candidate_capacity: usize,
    packed_points: Vec<[f32; 4]>,
    adapter_name: String,
}

impl GpuEvaluator {
    fn new() -> Result<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            flags: wgpu::InstanceFlags::default(),
            backend_options: wgpu::BackendOptions::default(),
            memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
            display: None,
        });

        let adapter = futures::executor::block_on(instance.request_adapter(
            &wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            },
        ))
        .context("не найден GPU adapter")?;

        let info = adapter.get_info();
        if info.device_type != wgpu::DeviceType::DiscreteGpu {
            return Err(anyhow!(
                "найден adapter '{}', но это не discrete GPU ({:?})",
                info.name,
                info.device_type
            ));
        }

        let (device, queue) = futures::executor::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("metro-lidar-compute"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::MemoryUsage,
                trace: wgpu::Trace::default(),
                experimental_features: wgpu::ExperimentalFeatures::default(),
            },
        ))
        .context("не удалось создать GPU device")?;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("tunnel-center-cost"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(TUNNEL_COST_SHADER)),
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("tunnel-center-cost-pipeline"),
            layout: None,
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tunnel-center-params"),
            size: std::mem::size_of::<GpuParameters>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let point_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tunnel-center-points"),
            size: 16,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let cost_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tunnel-center-costs"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tunnel-center-readback"),
            size: 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Ok(Self {
            device,
            queue,
            pipeline,
            params_buffer,
            point_buffer,
            cost_buffer,
            readback_buffer,
            point_capacity: 1,
            candidate_capacity: 1,
            packed_points: Vec::new(),
            adapter_name: info.name,
        })
    }

    fn adapter_name(&self) -> &str {
        &self.adapter_name
    }

    fn ensure_buffers(&mut self, point_count: usize, candidate_count: usize) {
        if point_count > self.point_capacity {
            self.point_capacity = point_count.next_power_of_two();
            self.point_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("tunnel-center-points"),
                size: (self.point_capacity * std::mem::size_of::<[f32; 4]>()) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        }

        if candidate_count > self.candidate_capacity {
            self.candidate_capacity = candidate_count.next_power_of_two();
            self.cost_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("tunnel-center-costs"),
                size: (self.candidate_capacity * std::mem::size_of::<f32>()) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            self.readback_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("tunnel-center-readback"),
                size: (self.candidate_capacity * std::mem::size_of::<f32>()) as u64,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        }
    }

    fn find_best_candidate(
        &mut self,
        point_indices: &[usize],
        points: &[Vec3],
        x_min: f32,
        z_min: f32,
        current_y: f32,
        grid_resolution: f32,
        x_steps: usize,
        z_steps: usize,
        repulsion_factor: f32,
        spring_factor: f32,
        distance_offset: f32,
    ) -> Result<usize> {
        let candidate_count = x_steps
            .checked_mul(z_steps)
            .context("переполнение числа GPU-кандидатов")?;
        self.packed_points.clear();
        self.packed_points.extend(point_indices.iter().map(|&index| {
            let point = points[index];
            [point.x, point.y, point.z, 0.0]
        }));
        self.ensure_buffers(self.packed_points.len(), candidate_count);

        self.queue.write_buffer(
            &self.point_buffer,
            0,
            bytemuck::cast_slice(&self.packed_points),
        );

        let params = GpuParameters {
            bounds: [x_min, z_min, current_y, grid_resolution],
            factors: [repulsion_factor, spring_factor, distance_offset, 0.0],
            dimensions: [x_steps as u32, z_steps as u32, self.packed_points.len() as u32, 0],
        };
        self.queue
            .write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&params));

        let bind_group_layout = self.pipeline.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tunnel-center-cost-bind-group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.point_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.cost_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.params_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("tunnel-center-cost-encoder"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("tunnel-center-cost-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups = (candidate_count as u32).div_ceil(GPU_WORKGROUP_SIZE);
            pass.dispatch_workgroups(groups, 1, 1);
        }

        encoder.copy_buffer_to_buffer(
            &self.cost_buffer,
            0,
            &self.readback_buffer,
            0,
            (candidate_count * std::mem::size_of::<f32>()) as u64,
        );

        self.queue.submit(Some(encoder.finish()));
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .context("GPU poll завершился ошибкой")?;

        let readback_bytes = (candidate_count * std::mem::size_of::<f32>()) as u64;
        let slice = self.readback_buffer.slice(..readback_bytes);
        let (sender, receiver) = mpsc::sync_channel(1);
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .context("GPU readback poll завершился ошибкой")?;
        receiver
            .recv()
            .context("не удалось дождаться GPU readback")?
            .context("не удалось отобразить GPU readback buffer")?;

        let data = slice.get_mapped_range();
        let values: &[f32] = bytemuck::cast_slice(&data);
        let mut best_index = 0;
        let mut best_cost = f32::INFINITY;
        for (index, &cost) in values.iter().enumerate() {
            if cost < best_cost {
                best_cost = cost;
                best_index = index;
            }
        }
        drop(data);
        self.readback_buffer.unmap();

        Ok(best_index)
    }
}

const TUNNEL_COST_SHADER: &str = r#"
struct Params {
    bounds: vec4<f32>,
    factors: vec4<f32>,
    dimensions: vec4<u32>,
};

@group(0) @binding(0)
var<storage, read> points: array<vec4<f32>>;

@group(0) @binding(1)
var<storage, read_write> costs: array<f32>;

@group(0) @binding(2)
var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let candidate_index = global_id.x;
    let x_steps = params.dimensions.x;
    let z_steps = params.dimensions.y;
    let point_count = params.dimensions.z;
    let candidate_count = x_steps * z_steps;

    if (candidate_index >= candidate_count) {
        return;
    }

    let x_index = candidate_index / z_steps;
    let z_index = candidate_index % z_steps;
    let x = params.bounds.x + f32(x_index) * params.bounds.w;
    let z = params.bounds.y + f32(z_index) * params.bounds.w;
    let y = params.bounds.z;

    var cost = 0.0;
    for (var index = 0u; index < point_count; index = index + 1u) {
        let point = points[index].xyz;
        let delta = vec3<f32>(x, y, z) - point;
        let distance_squared = dot(delta, delta);
        let denominator = distance_squared + params.factors.z;
        cost = cost + exp(params.factors.x / denominator);
        cost = cost + exp(denominator * params.factors.y);
    }

    costs[candidate_index] = cost;
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_search_returns_lowest_candidate_for_simple_input() {
        let points = [Vec3::new(0.0, 0.0, 0.0)];
        let indices = [0usize];
        let index = cpu_best_candidate(
            &indices,
            &points,
            0.0,
            0.0,
            0.0,
            1.0,
            2,
            2,
            0.0,
            1.0,
            0.01,
        )
        .expect("candidate should exist");

        assert_eq!(index, 0);
    }
}
