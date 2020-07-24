use image::GenericImageView;

use crate::errors::*;

pub struct Texture {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub sampler: wgpu::Sampler,
}

impl Texture {
    #[allow(dead_code)]
    pub fn from_bytes(device: &wgpu::Device, bytes: &[u8]) -> Result<(Self, wgpu::CommandBuffer)> {
        let image = image::load_from_memory(bytes)?;
        Self::from_image(device, &image)
    }

    pub fn from_image(device: &wgpu::Device, image: &image::DynamicImage) -> Result<(Self, wgpu::CommandBuffer)> {
        let dimensions = image.dimensions();
        
        let size = wgpu::Extent3d {
            width: dimensions.0,
            height: dimensions.1,
            depth: 1,
        };

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            size: size,
            array_layer_count: 1,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsage::SAMPLED | wgpu::TextureUsage::COPY_DST,
            label: None
        });

        let mut encoder = device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor{ label: None }
        );

        encoder.copy_buffer_to_texture(
            wgpu::BufferCopyView {
                buffer: &device.create_buffer_with_data(
                    &image.to_rgba(),
                    wgpu::BufferUsage::COPY_SRC,
                ),
                offset: 0,
                bytes_per_row: 4 * dimensions.0,
                rows_per_image: dimensions.1,
            },
            wgpu::TextureCopyView {
                texture: &texture,
                mip_level: 0,
                array_layer: 0,
                origin: wgpu::Origin3d::ZERO,
            },
            size,
        );
        
        let command_buffer = encoder.finish();

        let view = texture.create_default_view();

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            lod_min_clamp: -100.,
            lod_max_clamp: 100.,
            compare: wgpu::CompareFunction::Always,
        });

        Ok((Self { texture, view, sampler }, command_buffer))
    }

    pub fn load_image<P: AsRef<std::path::Path>>(device: &wgpu::Device, path: P)
        -> Result<(Self, wgpu::CommandBuffer)> {
        Self::from_image(device, &image::open(path)?)
    }

    pub fn depth_buffer(
        device: &wgpu::Device,
        dimensions: winit::dpi::PhysicalSize<u32>,
    ) -> Self {
        let dimensions = wgpu::Extent3d {
            width: dimensions.width,
            height: dimensions.height,
            depth: 1,
        };

        let texture_descriptor = wgpu::TextureDescriptor {
            label: None,
            size: dimensions,
            array_layer_count: 1,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT
                | wgpu::TextureUsage::SAMPLED
                | wgpu::TextureUsage::COPY_SRC,
        };

        let texture = device.create_texture(&texture_descriptor);
        let view = texture.create_default_view();
        
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            lod_min_clamp: -100.0,
            lod_max_clamp: 100.0,
            compare: wgpu::CompareFunction::LessEqual,
        });

        Self{ texture, view, sampler }
    }
}