#version 450

/*
   |\
   |   \
   |      \
   _________ \ 
   |screen |    \
   |       |       \
   |_______|__________\
*/
vec2 verts[3] = {vec2(-1.0, -3.0), vec2(-1.0, 1.0), vec2(3.0, 1.0)}; 

void main() {
    gl_Position = vec4(verts[gl_VertexIndex], 0.0, 1.0);
}