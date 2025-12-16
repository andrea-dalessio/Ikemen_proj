#if __VERSION__ >= 450
layout(location = 0) in vec2 vert;
layout(location = 1) in vec2 vertTexCoord;
layout(push_constant, std430) uniform u {
layout(offset = 16) vec2 resolution;
};
layout(location = 0) out vec2 fragTexCoord;
#else
#if __VERSION__ >= 130
#define COMPAT_VARYING out
#define COMPAT_ATTRIBUTE in
#define COMPAT_TEXTURE texture
#else
#define COMPAT_VARYING varying
#define COMPAT_ATTRIBUTE attribute
#define COMPAT_TEXTURE texture2D
#endif
//vertex position
COMPAT_ATTRIBUTE vec2 vert;

//pass through to fragTexCoord
COMPAT_ATTRIBUTE vec2 vertTexCoord;

//window res
uniform vec2 resolution;

//pass to frag
COMPAT_VARYING vec2 fragTexCoord;
#endif

void main() {
   // convert the rectangle from pixels to 0.0 to 1.0
   vec2 zeroToOne = vert / resolution;

   // convert from 0->1 to 0->2
   vec2 zeroToTwo = zeroToOne * 2.0;

   // convert from 0->2 to -1->+1 (clipspace)
   vec2 clipSpace = zeroToTwo - 1.0;

   fragTexCoord = vertTexCoord;

   gl_Position = vec4(clipSpace * vec2(1, -1), 0, 1);
}