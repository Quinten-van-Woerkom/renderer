/**
 * Implementation of a general-use model type to store models as loaded from
 * Wavefront .obj files.
 */
use crate::errors::*;
use crate::texture::*;

use std::ops::Range;
use std::vec::Vec;

// /**
//  * The graphical world is divided into regions that each store their own
//  * contained set of models, lights, etc. and that can be rendered separately.
//  * This could be used later to efficiently perform frustum culling etc.
//  */
// pub struct GraphicsRegion<'a> {
//     pub device: &'a wgpu::Device,
//     pub models: Vec<Model>,
//     pub lights: Vec<wgpu::Buffer>,
// }

// impl<'a> GraphicsRegion<'a> {
//     pub fn new(device: &'a wgpu::Device) -> Self {
//         Self {
//             device: device,
//             models: Vec::new(),
//             lights: Vec::new(),
//         }
//     }

//     pub fn add_model(&mut self, model: Model) {
//         self.models.push(model);
//     }

//     pub fn add_light(&mut self, light: Light) {
//         let light_buffer = self.device.create_buffer_with_data(
//             bytemuck::cast_slice(&[light]),
//             wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_DST,
//         );

//         self.lights.push(light_buffer);
//     }
// }

/**
 * A 3D model is stored as a combination of meshes and textures.
 * Currently, no animation support (e.g. joints) is present.
 */
pub struct Model {
    pub meshes: Vec<Mesh>,
    pub materials: Vec<Material>,
    pub instances: wgpu::Buffer,
    pub instance_count: u32,
}

impl Model {
    pub fn load<P: AsRef<std::path::Path>>(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        path: P
    ) -> Result<(Self, Vec<wgpu::CommandBuffer>)> {
        let (models, loaded_materials) = tobj::load_obj(path.as_ref(), true)?;

        let mut command_buffers = Vec::new();
        let mut materials = Vec::new();
        let materials_folder = path.as_ref().parent().expect("Invalid file path for model");

        for material in loaded_materials {
            let texture_path = materials_folder.join(material.diffuse_texture);
            let (texture, command_buffer) = Texture::load_image(device, texture_path)?;

            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout,
                bindings: &[
                    wgpu::Binding {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&texture.view),
                    },
                    wgpu::Binding {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&texture.sampler),
                    },
                ],
                label: None,
            });

            materials.push(Material { diffuse_texture: texture, bind_group: bind_group });
            command_buffers.push(command_buffer);
        }

        let mut meshes = Vec::new();
        for model in models {
            let mut vertices = Vec::new();
            for i in 0..model.mesh.positions.len() / 3 {
                vertices.push(ModelVertex {
                    position: [
                        model.mesh.positions[i * 3],
                        model.mesh.positions[i * 3 + 1],
                        model.mesh.positions[i * 3 + 2],
                    ],
                    texture_coordinates: [
                        model.mesh.texcoords[i * 2],
                        model.mesh.texcoords[i * 2  + 1],
                    ],
                    normal: [
                        model.mesh.normals[i * 3],
                        model.mesh.normals[i * 3 + 1],
                        model.mesh.normals[i * 3 + 2],
                    ],
                });
            }

            let vertex_buffer = device.create_buffer_with_data(
                bytemuck::cast_slice(&vertices),
                wgpu::BufferUsage::VERTEX,
            );
            let index_buffer = device.create_buffer_with_data(
                bytemuck::cast_slice(&model.mesh.indices),
                wgpu::BufferUsage::INDEX,
            );

            meshes.push(Mesh {
                vertex_buffer,
                index_buffer,
                size: model.mesh.indices.len() as u32,
                material: model.mesh.material_id.unwrap_or(0),
            });
        }

        let instances = device.create_buffer_with_data(
            bytemuck::cast_slice(&[Instance::default()]),
            wgpu::BufferUsage::STORAGE_READ,
        );

        Ok((Self { meshes, materials, instances, instance_count: 1 }, command_buffers))
    }

    /**
     * Makes a grid of instances for testing purposes.
     */
    pub fn make_testing_grid(&mut self, device: &wgpu::Device) {
        const NUM_INSTANCES_PER_ROW: u32 = 10;
        const SPACE_BETWEEN: f32 = 3.0;
        let instances = (0..NUM_INSTANCES_PER_ROW).flat_map(|z| {
            (0..NUM_INSTANCES_PER_ROW).map(move |x| {
                let x = SPACE_BETWEEN * (x as f32 - NUM_INSTANCES_PER_ROW as f32 / 2.0);
                let z = SPACE_BETWEEN * (z as f32 - NUM_INSTANCES_PER_ROW as f32 / 2.0);

                let position = nalgebra::Vector3::new(x, 0.0, z);
                
                let rotation = if position == nalgebra::Vector3::zeros() {
                    nalgebra::Rotation3::from_axis_angle(&nalgebra::Vector3::z_axis(), 0.0)
                } else {
                    nalgebra::Rotation3::from_axis_angle(&nalgebra::Unit::new_normalize(position), std::f32::consts::PI/4.0)
                };

                Instance::new(position, rotation)
            })
        }).collect::<Vec<_>>();

        self.instances = device.create_buffer_with_data(
            bytemuck::cast_slice(&instances),
            wgpu::BufferUsage::STORAGE_READ
        );

        self.instance_count = instances.len() as u32;
    }
}

pub struct Mesh {
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub size: u32,
    pub material: usize,
}

pub struct Material {
    pub diffuse_texture: Texture,
    pub bind_group: wgpu::BindGroup,
}

/**
 * Drawing a model consists of drawing all its meshes, with the corresponding
 * materials. This is done by the render pass, such that implementing it as a
 * trait is the cleanest approach for an extended interface.
 */
pub trait DrawModel<'a, 'b> where 'b: 'a {    
    fn draw_mesh(&mut self, mesh: &'b Mesh, material: &'b Material,
        uniforms: &'b wgpu::BindGroup, instances: Range<u32>);
        
    fn draw_model(&mut self, model: &'b Model, uniforms: &'b wgpu::BindGroup);
}

impl<'a, 'b> DrawModel<'a, 'b> for wgpu::RenderPass<'a> where 'b: 'a {
    fn draw_mesh(&mut self, mesh: &'b Mesh, material: &'b Material,
        uniforms: &'b wgpu::BindGroup, instances: Range<u32>)
    {
        self.set_vertex_buffer(0, &mesh.vertex_buffer, 0, 0);
        self.set_index_buffer(&mesh.index_buffer, 0, 0);
        self.set_bind_group(0, &uniforms, &[]);
        self.set_bind_group(1, &material.bind_group, &[]);
        self.draw_indexed(0..mesh.size, 0, instances);
    }

    fn draw_model(&mut self, model: &'b Model, uniforms: &'b wgpu::BindGroup)
    {
        for mesh in &model.meshes {
            self.draw_mesh(mesh, &model.materials[mesh.material],
                uniforms, 0..model.instance_count as u32);
        }
    }
}

/**
 * Often, the same model is reused repeatedly, albeit in different positions
 * and rotations. This is achieved using instancing, where the model is stored
 * once and only instance-specific data is repeated.
 */
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct Instance {
    model: nalgebra::Matrix4<f32>,
}

impl Instance {
    fn new(position: nalgebra::Vector3<f32>, rotation: nalgebra::Rotation3<f32>) -> Instance {
        Instance {
            model: nalgebra::Isometry3::from_parts(position.into(), rotation.into()).to_homogeneous()
        }
    }
}

impl Default for Instance {
    fn default() -> Self {
        let position = nalgebra::Vector3::zeros();
        let rotation = nalgebra::Rotation3::from_axis_angle(
            &nalgebra::Vector3::x_axis(), 0.0);
        Self::new(position.into(), rotation.into())
    }
}

unsafe impl bytemuck::Pod for Instance {}
unsafe impl bytemuck::Zeroable for Instance {}

/**
 * Any type that is to be used as vertex must have a buffer descriptor to be
 * able to be used in the rendering pipeline. Might be worth looking into
 * creating a macro for this.
 */
pub trait Vertex: bytemuck::Pod + bytemuck::Zeroable {
    fn description<'a>() -> wgpu::VertexBufferDescriptor<'a>;
}

/**
 * Models as loaded from .obj files contain vertices storing only position,
 * texture coordinates, and vertex normals.
 */
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct ModelVertex {
    position: [f32; 3],
    normal: [f32; 3],
    texture_coordinates: [f32; 2],
}

unsafe impl bytemuck::Pod for ModelVertex {}
unsafe impl bytemuck::Zeroable for ModelVertex {}

impl Vertex for ModelVertex {
    fn description<'a>() -> wgpu::VertexBufferDescriptor<'a> {
        wgpu::VertexBufferDescriptor {
            stride: std::mem::size_of::<ModelVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::InputStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttributeDescriptor {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float3,
                },
                wgpu::VertexAttributeDescriptor {
                    offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float3,
                },
                wgpu::VertexAttributeDescriptor {
                    offset: std::mem::size_of::<[f32; 6]>() as wgpu::BufferAddress,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float2,
                },
            ]
        }
    }
}


// /**
//  * Lights are stored in terms of their position and colour only, for now.
//  */
// #[repr(C)]
// #[derive(Copy, Clone, Debug)]
// pub struct Light {
//     position: nalgebra::Vector3<f32>,
//     _padding: u32,
//     color: nalgebra::Vector3<f32>,
// }

// impl Light {
//     fn new(position: nalgebra::Vector3<f32>, color: nalgebra::Vector3<f32>) -> Self {
//         Self {
//             position,
//             color,
//             _padding: 0,
//         }
//     }
// }

// unsafe impl bytemuck::Zeroable for Light {}
// unsafe impl bytemuck::Pod for Light {}