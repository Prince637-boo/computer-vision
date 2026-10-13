#version 120

varying vec3 view_normal;
varying vec3 view_position;
varying vec4 view_color_scale;

void main() {
    vec3 normal = normalize(view_normal);
    vec3 view_direction = normalize(-view_position);
    float rim = pow(1.0 - max(dot(normal, view_direction), 0.0), 2.4);

    vec3 cyan = vec3(0.02, 0.82, 1.0);
    vec3 emission = cyan * (0.42 + rim * 2.1) * view_color_scale.rgb;
    float alpha = clamp(0.13 + rim * 0.48, 0.13, 0.58) * view_color_scale.a;

    gl_FragColor = vec4(emission, alpha);
}
