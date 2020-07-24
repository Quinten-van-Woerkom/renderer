/**
 * Implementation of the camera object, that represents the view into the
 * rendered world.
 */
use crate::keyboard::*;

use std::time::Duration;


pub struct Camera {
    position: nalgebra::Point3<f32>,
    direction: nalgebra::Unit<nalgebra::Vector3<f32>>,
    up: nalgebra::Unit<nalgebra::Vector3<f32>>,
    aspect_ratio: f32,
    field_of_view: f32,
    near: f32,
    far: f32,
}

impl Camera {
    pub fn from_frustum(
        position: nalgebra::Point3<f32>,
        direction: nalgebra::Unit<nalgebra::Vector3<f32>>,
        up: nalgebra::Unit<nalgebra::Vector3<f32>>,
        aspect_ratio: f32,
        field_of_view: f32,
        near: f32,
        far: f32,
    ) -> Self {
        Self {
            position: position,
            direction: direction,
            up: up,
            aspect_ratio: aspect_ratio,
            field_of_view: field_of_view,
            near: near,
            far: far,
        }
    }

    pub fn projection_matrix(&self) -> nalgebra::Matrix4<f32> {
        // This can probably be done in a cleaner way
        let target = self.position + self.direction.into_inner();
        let view = nalgebra::Isometry3::look_at_rh(&self.position, &target, &self.up);

        // While Vulkan requires a perspective projection cube with z
        // coordinates from 0 to 1, nalgebra uses the -1 to 1 range, following
        // OpenGL convention.
        // TODO: make this transformation implicit in the perspective creation.
        let adapt_projection: nalgebra::Matrix4<f32> = nalgebra::Matrix4::new(
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 0.5, 0.0,
            0.0, 0.0, 0.5, 1.0,
        );

        let projection = nalgebra::Perspective3::new(
            self.aspect_ratio, self.field_of_view,
            self.near, self.far);
        adapt_projection * projection.as_matrix() * view.to_homogeneous()
    }

    pub fn translate(&mut self, translation: nalgebra::Vector3<f32>) {
        self.position += translation;
    }

    pub fn rotate(&mut self, rotation: nalgebra::Rotation3<f32>)
    {
        self.direction = rotation * self.direction;
    }

    pub fn update(&mut self, keyboard: &Keyboard, time_step: Duration) {
        let rotation = time_step.as_secs_f32();
        let left = nalgebra::Unit::new_normalize(self.up.cross(&self.direction));

        let step = if keyboard.is_pressed(VirtualKeyCode::LAlt) {
            5.0 * time_step.as_secs_f32()
        } else {
            time_step.as_secs_f32()
        };

        if keyboard.is_pressed(VirtualKeyCode::W) {
            self.translate(step * self.direction.into_inner());
        }

        if keyboard.is_pressed(VirtualKeyCode::S) {
            self.translate(-step * self.direction.into_inner());
        }

        if keyboard.is_pressed(VirtualKeyCode::A) {
            self.translate(step * left.into_inner());
        }

        if keyboard.is_pressed(VirtualKeyCode::D) {
            self.translate(-step * left.into_inner());
        }

        if keyboard.is_pressed(VirtualKeyCode::Up) {
            self.rotate(nalgebra::Rotation3::from_axis_angle(&left, -rotation));
        }

        if keyboard.is_pressed(VirtualKeyCode::Down) {
            self.rotate(nalgebra::Rotation3::from_axis_angle(&left, rotation));
        }

        if keyboard.is_pressed(VirtualKeyCode::Left) {
            self.rotate(nalgebra::Rotation3::from_axis_angle(&self.up, rotation));
        }

        if keyboard.is_pressed(VirtualKeyCode::Right) {
            self.rotate(nalgebra::Rotation3::from_axis_angle(&self.up, -rotation));
        }

        if keyboard.is_pressed(VirtualKeyCode::LShift) {
            self.translate(-step * self.up.into_inner());
        }

        if keyboard.is_pressed(VirtualKeyCode::Space) {
            self.translate(step * self.up.into_inner());
        }
    }
}