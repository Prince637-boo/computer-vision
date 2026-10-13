#version 120

attribute vec4 p3d_Vertex;
attribute vec3 p3d_Normal;
uniform mat4 p3d_ModelViewProjectionMatrix;
uniform mat4 p3d_ModelViewMatrix;
uniform mat3 p3d_NormalMatrix;
uniform vec4 p3d_ColorScale;

varying vec3 view_normal;
varying vec3 view_position;
varying vec4 view_color_scale;

void main() {
    vec4 eye_position = p3d_ModelViewMatrix * p3d_Vertex;
    view_position = eye_position.xyz;
    view_normal = normalize(p3d_NormalMatrix * p3d_Normal);
    view_color_scale = p3d_ColorScale;
    gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
}
