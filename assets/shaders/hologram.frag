#version 120

uniform float time;

varying vec3 view_normal;
varying vec3 view_position;
varying vec4 view_color_scale;

void main() {
    vec3 normal = normalize(view_normal);
    vec3 view_direction = normalize(-view_position);
    float rim = pow(1.0 - max(dot(normal, view_direction), 0.0), 1.65);
    float scanline = 0.93 + 0.07 * sin(view_position.y * 34.0 + time * 2.6);
    float flicker = 0.98 + 0.02 * sin(time * 19.0 + view_position.x * 7.0);

    vec3 cyan = vec3(0.02, 0.82, 1.0);
    vec3 emission = cyan * (0.42 + rim * 2.6) * scanline * flicker;
    emission *= view_color_scale.rgb;
    float alpha = clamp(0.13 + rim * 0.58, 0.13, 0.72) * view_color_scale.a;

    gl_FragColor = vec4(emission, alpha);
}
