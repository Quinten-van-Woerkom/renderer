#version 450

layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_normal; // Unused
layout(location = 2) in vec2 a_texture_coordinates;

layout(location = 0) out vec2 v_texture_coordinates;

layout(set = 0, binding = 0)
uniform Uniforms {
    mat4 u_view_projection;
};

layout(set = 0, binding = 1)
buffer Instances {
    mat4 s_models[];
};

void main() {
    v_texture_coordinates = a_texture_coordinates;
    gl_Position = u_view_projection * s_models[gl_InstanceIndex] * vec4(a_position, 1.0);
}