mod camera;
mod errors;
mod keyboard;
mod model;
mod texture;

use crate::errors::*;
use crate::model::*;

use futures::executor::block_on;

use winit::{
    dpi::PhysicalSize,
    event::*,
    event_loop::{ EventLoop, ControlFlow },
    window::{Window, WindowBuilder},
};

use wgpu::*;


/**
 * Returns a logical graphical device that fits the given power preference
 * and is compatible with the given surface. If no such device is possible,
 * like when no physical device is compatible with the given surface, None
 * is retured.
 */
async fn request_graphical_device(
    power_preference: wgpu::PowerPreference,
    compatible_surface: Option<&Surface>
) -> Option<(Device, Queue)> {
    let adapter = Adapter::request(
        &wgpu::RequestAdapterOptions {
            power_preference: power_preference,
            compatible_surface: compatible_surface,
        },
        wgpu::BackendBit::PRIMARY,
    ).await?;

    let (device, queue) = adapter.request_device( &wgpu::DeviceDescriptor {
        extensions: wgpu::Extensions {
            anisotropic_filtering: false,
        },
        ..Default::default()
    }).await;

    Some((device, queue))
}


/**
 * A handle to the screen and all related graphical components.
 * Effectively a wrapper over the surface, swap chain, and screen dimensions.
 */
struct Screen {
    surface: Surface,
    swap_chain: SwapChain,
    swap_chain_descriptor: SwapChainDescriptor,
}

impl Screen {
    /**
     * Constructs a screen that utilises the given graphical device to draw to
     * a surface of given dimensions.
     * The surface must be pre-created as the choice of graphical device
     * depends on it.
     */
    fn new(device: &Device, surface: Surface, dimensions: PhysicalSize<u32>) -> Self {
        let swap_chain_descriptor = SwapChainDescriptor {
            usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT,
            format: wgpu::TextureFormat::Bgra8UnormSrgb,
            width: dimensions.width,
            height: dimensions.height,
            present_mode: wgpu::PresentMode::Fifo,
        };

        let swap_chain = device.create_swap_chain(&surface, &swap_chain_descriptor);

        Self {
            surface,
            swap_chain,
            swap_chain_descriptor,
        }
    }

    /**
     * Reconfigures the screen to work with the given screen and dimensions by
     * recreating the swap chain that is drawn to. This is necessary if the
     * window size is changed, for example.
     */
    fn reconfigure(&mut self, device: &Device, dimensions: PhysicalSize<u32>) {
        self.swap_chain_descriptor.width = dimensions.width;
        self.swap_chain_descriptor.height = dimensions.height;
        self.swap_chain = device.create_swap_chain(
            &self.surface,
            &self.swap_chain_descriptor
        );
    }

    /**
     * Returns the next available framebuffer in the swap chain to draw to.
     */
    fn next_framebuffer(&mut self) -> SwapChainOutput {
        self.swap_chain.get_next_texture().expect("Time-out obtaining next framebuffer")
    }

    /**
     * Returns the physical dimensions of the screen, i.e. the actual number of
     * pixels shown.
     */
    fn physical_dimensions(&self) -> PhysicalSize<u32> {
        PhysicalSize {
            width: self.swap_chain_descriptor.width,
            height: self.swap_chain_descriptor.height,
        }
    }

    /**
     * Current screen aspect ratio.
     */
    fn aspect_ratio(&self) -> f32 {
        self.swap_chain_descriptor.width as f32 / self.swap_chain_descriptor.height as f32
    }

    /**
     * The format associated with this screen, describing the output type to be
     * used when writing to it.
     */
    fn format(&self) -> wgpu::TextureFormat {
        self.swap_chain_descriptor.format
    }
}


// /**
//  * Representation of the computational graph of the rendering process.
//  */
// struct RenderProgram {
//     pipeline: wgpu::RenderPipeline,
//     bind_group_layouts: Vec<wgpu::BindGroupLayout>,
//     uniform_buffer: wgpu::Buffer,
//     depth_buffer: texture::Texture,
// }

// impl RenderProgram {
//     fn new(vertex_shader: &str, fragment_shader: &str, ...)
// }


/**
 * Consumes and renders a mesh.
 * TODO: split up class responsibilities, it is humongous right now.
 */
struct RenderingEngine {
    device: Device,
    queue: Queue,
    screen: Screen,

    pipeline: wgpu::RenderPipeline,
    input_layout: wgpu::BindGroupLayout,
    uniform_buffer: wgpu::Buffer,
    depth_buffer: texture::Texture,

    model: Model,
}

impl RenderingEngine {
    async fn new(window: &Window) -> Result<Self> {
        let dimensions = window.inner_size();
        let surface = Surface::create(window);

        let (device, queue) = request_graphical_device(
            wgpu::PowerPreference::HighPerformance,
            Some(&surface),
        ).await.unwrap();

        let screen = Screen::new(&device, surface, dimensions);

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: std::mem::size_of::<Uniforms>() as wgpu::BufferAddress,
            usage: wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_DST,
        });

        let input_layout = device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                bindings: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStage::VERTEX,
                        ty: wgpu::BindingType::UniformBuffer {
                            dynamic: false,
                        },
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStage::VERTEX,
                        ty: wgpu::BindingType::StorageBuffer {
                            dynamic: false,
                            readonly: true,
                        }
                    }
                ],
                label: None,
            }
        );

        let diffuse_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            bindings: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStage::FRAGMENT,
                    ty: wgpu::BindingType::SampledTexture {
                        multisampled: false,
                        dimension: wgpu::TextureViewDimension::D2,
                        component_type: wgpu::TextureComponentType::Uint,
                    },
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStage::FRAGMENT,
                    ty: wgpu::BindingType::Sampler {
                        comparison: false
                    },
                },
            ],
            label: None
        });

        // TODO: this should probably be a Device method
        let depth_buffer = texture::Texture::depth_buffer(&device, dimensions);

        let pipeline = {
            let mut shader_compiler = shaderc::Compiler::new().unwrap();

            let vertex_shader = {
                let spirv = shader_compiler.compile_into_spirv(include_str!("vertex.glsl"),
                shaderc::ShaderKind::Vertex, "vertex.glsl", "main", None).unwrap();
                let data = wgpu::read_spirv(std::io::Cursor::new(spirv.as_binary_u8())).unwrap();
                device.create_shader_module(&data)
            };

            let fragment_shader = {
                let spirv = shader_compiler.compile_into_spirv(include_str!("fragment.glsl"),
                shaderc::ShaderKind::Fragment, "fragment.glsl", "main", None).unwrap();
                let data = wgpu::read_spirv(std::io::Cursor::new(spirv.as_binary_u8())).unwrap();
                device.create_shader_module(&data)
            };

            let pipeline_layout = device.create_pipeline_layout(
                &wgpu::PipelineLayoutDescriptor {
                    bind_group_layouts: &[&input_layout, &diffuse_layout]
            });

            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                layout: &pipeline_layout,
                primitive_topology: wgpu::PrimitiveTopology::TriangleList,
                vertex_stage: wgpu::ProgrammableStageDescriptor {
                    module: &vertex_shader,
                    entry_point: "main",
                },
                fragment_stage: Some(wgpu::ProgrammableStageDescriptor {
                    module: &fragment_shader,
                    entry_point: "main",
                }),
                rasterization_state: Some(wgpu::RasterizationStateDescriptor {
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: wgpu::CullMode::Back,
                    depth_bias: 0,
                    depth_bias_slope_scale: 0.0,
                    depth_bias_clamp: 0.0,
                }),
                color_states: &[
                    wgpu::ColorStateDescriptor {
                        format: screen.format(),
                        color_blend: wgpu::BlendDescriptor::REPLACE,
                        alpha_blend: wgpu::BlendDescriptor::REPLACE,
                        write_mask: wgpu::ColorWrite::ALL,
                    },
                ],
                depth_stencil_state: Some(wgpu::DepthStencilStateDescriptor {
                    format: wgpu::TextureFormat::Depth32Float,
                    depth_write_enabled: true,
                    depth_compare: wgpu::CompareFunction::Less,
                    stencil_front: wgpu::StencilStateFaceDescriptor::IGNORE,
                    stencil_back: wgpu::StencilStateFaceDescriptor::IGNORE,
                    stencil_read_mask: 0,
                    stencil_write_mask: 0,
                }),
                vertex_state: wgpu::VertexStateDescriptor {
                    index_format: wgpu::IndexFormat::Uint32,
                    vertex_buffers: &[ModelVertex::description()],
                },
                sample_count: 1,
                sample_mask: !0,
                alpha_to_coverage_enabled: false,
            })
        };

        // TODO: This should probably be a Device method
        let (mut model, command_load_model) = Model::load(
            &device,
            &diffuse_layout,
            "resources/cube.obj"
        )?;
        model.make_testing_grid(&device);
        queue.submit(&command_load_model);

        Ok(Self {
            device,
            queue,
            screen,

            pipeline,
            input_layout,
            uniform_buffer,
            depth_buffer,

            model,
        })
    }

    fn reconfigure_window(&mut self, dimensions: PhysicalSize<u32>) {
        self.screen.reconfigure(&self.device, dimensions);
        self.depth_buffer = texture::Texture::depth_buffer(
            &self.device,
            self.screen.physical_dimensions()
        );
    }

    fn render(&mut self, camera: &camera::Camera) {
        let uniforms = Uniforms{ view_projection: camera.projection_matrix() };
        let frame = self.screen.next_framebuffer();

        let mut encoder = self.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor{ label: None }
        );

        {
            let staging_buffer = self.device.create_buffer_with_data(
                bytemuck::cast_slice(&[uniforms]),
                wgpu::BufferUsage::COPY_SRC,
            );

            encoder.copy_buffer_to_buffer(
                &staging_buffer, 0, &self.uniform_buffer, 0,
                std::mem::size_of::<Uniforms>() as wgpu::BufferAddress);
            
            let uniform_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &self.input_layout,
                bindings: &[
                    wgpu::Binding {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer {
                            buffer: &self.uniform_buffer,
                            range: 0..std::mem::size_of_val(&uniforms) as wgpu::BufferAddress,
                        }
                    },
                    wgpu::Binding {
                        binding: 1,
                        resource: wgpu::BindingResource::Buffer {
                            buffer: &self.model.instances,
                            range: 0..(self.model.instance_count as usize
                                * std::mem::size_of::<cgmath::Matrix4<f32>>()) as wgpu::BufferAddress
                        }
                    }
                ],
                label: None
            });

            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: &[
                    wgpu::RenderPassColorAttachmentDescriptor {
                        attachment: &frame.view,
                        resolve_target: None,
                        load_op: wgpu::LoadOp::Clear,
                        store_op: wgpu::StoreOp::Store,
                        clear_color: wgpu::Color { r: 0.1, g: 0.2, b: 0.3, a: 1.0 },
                    }
                ],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachmentDescriptor {
                    attachment: &self.depth_buffer.view,
                    depth_load_op: wgpu::LoadOp::Clear,
                    depth_store_op: wgpu::StoreOp::Store,
                    clear_depth: 1.0,
                    stencil_load_op: wgpu::LoadOp::Clear,
                    stencil_store_op: wgpu::StoreOp::Store,
                    clear_stencil: 0,
                }),
            });

            render_pass.set_pipeline(&self.pipeline);
            render_pass.draw_model(&self.model, &uniform_bind_group);
        }
        self.queue.submit(&[ encoder.finish() ]);
    }
}


#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct Uniforms {
    view_projection: cgmath::Matrix4<f32>,
}

unsafe impl bytemuck::Pod for Uniforms {}
unsafe impl bytemuck::Zeroable for Uniforms {}


fn main() {
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new().with_title("Object renderer").build(&event_loop).expect("Unable to open window");
    let mut renderer = block_on(RenderingEngine::new(&window)).expect("Unable to construct rendering engine");

    let mut camera = camera::Camera::from_frustum(
        (0.0, 1.0, 2.0).into(),
        (0.0, -1.0, -2.0).into(),
        cgmath::Vector3::unit_y(),
        renderer.screen.aspect_ratio(),
        cgmath::Deg(45.0).into(),
        0.1,
        100.0,
    );

    let mut keyboard = keyboard::Keyboard::new();

    let mut time = std::time::Instant::now();
    let mut average = 0.0;
    let mut frame = 0;

    event_loop.run(move |event, _, control_flow| {
        match event {
            Event::WindowEvent { ref event, window_id }
            if window_id == window.id() => {
                match event {
                    WindowEvent::CloseRequested => {
                        *control_flow = ControlFlow::Exit
                    },
                    WindowEvent::KeyboardInput { input, .. } => {
                        keyboard.process_input(input);
                    },
                    WindowEvent::Resized(dimensions) => {
                        renderer.reconfigure_window(*dimensions);
                    },
                    WindowEvent::ScaleFactorChanged{ new_inner_size, ..}  => {
                        renderer.reconfigure_window(**new_inner_size);
                    },
                    _ => ()
                }
            },
            Event::MainEventsCleared => {
                if keyboard.is_pressed(VirtualKeyCode::Escape) {
                    println!("Average fps: {:.1} Hz", 1.0/average);
                    *control_flow = ControlFlow::Exit;
                    return;
                }
                
                let time_step = time.elapsed();
                time = std::time::Instant::now();
                average = (frame as f64 * average + time_step.as_secs_f64()) / (frame + 1) as f64;
                frame += 1;

                camera.update(&keyboard, time_step);
                renderer.render(&camera);
            }
            _ => ()
        }
    });
}