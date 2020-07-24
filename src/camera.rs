/**
 * Implementation of the camera object, that represents the view into the
 * rendered world.
 */
use crate::keyboard::*;

use std::time::Duration;

/**
 * TODO: make this transformation implicit in the perspective creation.
 */
#[cfg_attr(rustfmt, rustfmt_skip)]
pub const OPENGL_TO_WGPU_MATRIX: cgmath::Matrix4<f32> = cgmath::Matrix4::new(
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.5, 0.0,
    0.0, 0.0, 0.5, 1.0,
);


pub struct Camera {
    position: cgmath::Point3<f32>,
    direction: cgmath::Vector3<f32>,
    up: cgmath::Vector3<f32>,
    aspect_ratio: f32,
    field_of_view: cgmath::Rad<f32>,
    near: f32,
    far: f32,
}

impl Camera {
    pub fn from_frustum(
        position: cgmath::Point3<f32>,
        direction: cgmath::Vector3<f32>,
        up: cgmath::Vector3<f32>,
        aspect_ratio: f32,
        field_of_view: cgmath::Rad<f32>,
        near: f32,
        far: f32,
    ) -> Self {
        use cgmath::InnerSpace;
        Self {
            position: position,
            direction: direction.normalize(),
            up: up,
            aspect_ratio: aspect_ratio,
            field_of_view: field_of_view,
            near: near,
            far: far,
        }
    }

    pub fn projection_matrix(&self) -> cgmath::Matrix4<f32> {
        let target = self.position + self.direction;
        let view = cgmath::Matrix4::look_at(self.position, target, self.up);
        let projection = cgmath::perspective(self.field_of_view,
            self.aspect_ratio, self.near, self.far);
        OPENGL_TO_WGPU_MATRIX * projection * view
    }

    pub fn translate(&mut self, translation: cgmath::Vector3<f32>) {
        self.position += translation;
    }

    pub fn rotate<R>(&mut self, rotation: R)
        where R: cgmath::Rotation<cgmath::Point3<f32>>
    {
        self.direction = rotation.rotate_vector(self.direction);
    }

    pub fn update(&mut self, keyboard: &Keyboard, time_step: Duration) {
        use cgmath::InnerSpace;
        use cgmath::Rotation3;

        let rotation = time_step.as_secs_f32();
        let left = self.up.cross(self.direction).normalize();

        let step = if keyboard.is_pressed(VirtualKeyCode::LAlt) {
            5.0 * time_step.as_secs_f32()
        } else {
            time_step.as_secs_f32()
        };

        if keyboard.is_pressed(VirtualKeyCode::W) {
            self.translate(step * self.direction);
        }

        if keyboard.is_pressed(VirtualKeyCode::S) {
            self.translate(-step * self.direction);
        }

        if keyboard.is_pressed(VirtualKeyCode::A) {
            self.translate(step * left);
        }

        if keyboard.is_pressed(VirtualKeyCode::D) {
            self.translate(-step * left);
        }

        if keyboard.is_pressed(VirtualKeyCode::Up) {
            self.rotate(cgmath::Quaternion::from_axis_angle(left, cgmath::Rad(-rotation)));
        }

        if keyboard.is_pressed(VirtualKeyCode::Down) {
            self.rotate(cgmath::Quaternion::from_axis_angle(left, cgmath::Rad(rotation)));
        }

        if keyboard.is_pressed(VirtualKeyCode::Left) {
            self.rotate(cgmath::Quaternion::from_axis_angle(self.up, cgmath::Rad(rotation)));
        }

        if keyboard.is_pressed(VirtualKeyCode::Right) {
            self.rotate(cgmath::Quaternion::from_axis_angle(self.up, cgmath::Rad(-rotation)));
        }

        if keyboard.is_pressed(VirtualKeyCode::LShift) {
            self.translate(-step * self.up);
        }

        if keyboard.is_pressed(VirtualKeyCode::Space) {
            self.translate(step * self.up);
        }
    }
}