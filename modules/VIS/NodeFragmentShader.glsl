R"(
#version 330 core

// Interpolated values from the vertex shaders
in vec3 fragmentColor;

// Ouput data
out vec4 color;


void main()
{
	// calculate normal from texture coordinates    
	vec3 N;    
	N.xy = gl_PointCoord * 2.0 - vec2(1);        
	float r = dot(N.xy, N.xy);    
	if (r > 1.0) discard;    // kill pixels outside circle    

	color = vec4(fragmentColor, 1); 
}
)"
