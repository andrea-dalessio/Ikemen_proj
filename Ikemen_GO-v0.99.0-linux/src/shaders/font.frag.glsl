#if __VERSION__ >= 450
#define COMPAT_TEXTURE texture
#define COMPAT_FRAGCOLOR FragColor
layout(location = 0) out vec4 FragColor;
layout(location = 0) in vec2 fragTexCoord;

layout(push_constant, std430) uniform u {
	vec4 textColor;
};
layout(binding = 0) uniform sampler2D tex;
#else
#if __VERSION__ >= 130
#define COMPAT_VARYING in
#define COMPAT_ATTRIBUTE in
#define COMPAT_TEXTURE texture
#define COMPAT_FRAGCOLOR FragColor
out vec4 FragColor;
#else
#define COMPAT_VARYING varying
#define COMPAT_ATTRIBUTE attribute
#define COMPAT_TEXTURE texture2D
#define COMPAT_FRAGCOLOR gl_FragColor
#endif
COMPAT_VARYING vec2 fragTexCoord;
uniform vec4 textColor;
uniform sampler2D tex;
#endif


void main()
{
    vec4 sampled = vec4(1.0, 1.0, 1.0, COMPAT_TEXTURE(tex, fragTexCoord).r);
    COMPAT_FRAGCOLOR = min(textColor, vec4(1.0, 1.0, 1.0, 1.0)) * sampled;
}